#include "stdafx.h"
#include "MTLOverlays.h"

#include "MTLFormats.h"
#include "MTLPipelineCompiler.h"
#include "MTLRenderTargets.h"
#include "MTLResourceManager.h"
#include "mtlutils/barriers.h"
#include "mtlutils/image_helpers.h"
#include "mtlutils/shared.h"

#include "Emu/RSX/Overlays/overlays.h"
#include "Emu/RSX/Program/RSXOverlay.h"
#include "Utilities/stereo_config.h"
#include "Emu/Cell/timers.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

namespace mtl
{
	namespace
	{
		[[nodiscard]] u32 mip_dimension(u32 value, u32 level)
		{
			return std::max(value >> level, 1u);
		}

		[[nodiscard]] NSString* native_string(std::string_view value)
		{
			return [NSString stringWithUTF8String:std::string(value).c_str()];
		}

		[[noreturn]] void throw_native_error(NSError* error, std::string_view operation)
		{
			const char* description = error ? error.localizedDescription.UTF8String : nullptr;
			fmt::throw_exception("Metal %s failed: %s", operation,
				description ? description : "unknown error");
		}

		[[nodiscard]] id<MTLLibrary> compile_library(id<MTL4Compiler> compiler,
			std::string_view source, std::string_view label)
		{
			MTLCompileOptions* options = [MTLCompileOptions new];
			options.languageVersion = MTLLanguageVersion4_0;
			options.mathMode = MTLMathModeFast;
			MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
			descriptor.name = native_string(label);
			descriptor.source = native_string(source);
			descriptor.options = options;
			NSError* error = nil;
			id<MTLLibrary> library = [compiler newLibraryWithDescriptor:descriptor error:&error];
			if (!library) throw_native_error(error, fmt::format("%s library compilation", label));
			return library;
		}

		[[nodiscard]] MTLPrimitiveTopologyClass topology_class(primitive_topology topology)
		{
			switch (topology)
			{
			case primitive_topology::point: return MTLPrimitiveTopologyClassPoint;
			case primitive_topology::line:
			case primitive_topology::line_strip: return MTLPrimitiveTopologyClassLine;
			case primitive_topology::triangle:
			case primitive_topology::triangle_strip: return MTLPrimitiveTopologyClassTriangle;
			}
			fmt::throw_exception("Invalid Metal overlay primitive topology");
		}

		[[nodiscard]] MTLPrimitiveType primitive_type(primitive_topology topology)
		{
			switch (topology)
			{
			case primitive_topology::point: return MTLPrimitiveTypePoint;
			case primitive_topology::line: return MTLPrimitiveTypeLine;
			case primitive_topology::line_strip: return MTLPrimitiveTypeLineStrip;
			case primitive_topology::triangle: return MTLPrimitiveTypeTriangle;
			case primitive_topology::triangle_strip: return MTLPrimitiveTypeTriangleStrip;
			}
			fmt::throw_exception("Invalid Metal overlay primitive topology");
		}

		[[nodiscard]] MTLColorWriteMask native_color_mask(u8 mask)
		{
			MTLColorWriteMask result = MTLColorWriteMaskNone;
			if (mask & color_write_red) result |= MTLColorWriteMaskRed;
			if (mask & color_write_green) result |= MTLColorWriteMaskGreen;
			if (mask & color_write_blue) result |= MTLColorWriteMaskBlue;
			if (mask & color_write_alpha) result |= MTLColorWriteMaskAlpha;
			return result;
		}

		[[nodiscard]] subresource_range target_range(const overlay_render_target& target)
		{
			return {
				.first_mip = target.mip_level,
				.mip_count = 1,
				.first_slice = target.array_slice,
				.slice_count = 1,
				.color = bool(target.write_aspects & texture_aspect_color),
				.depth = bool(target.write_aspects & texture_aspect_depth),
				.stencil = bool(target.write_aspects & texture_aspect_stencil),
			};
		}

		[[nodiscard]] image_state operation_state(const image& resource, u64 stages, u64 access)
		{
			const image_state previous = resource.state();
			return {
				.queue = previous.initialized ? previous.queue : queue_kind::graphics,
				.stages = stages,
				.access = access,
				.submission = get_submission_id(),
				.initialized = true,
			};
		}

		void end_active_encoder(command_buffer& command)
		{
			if (command.active_encoder() != encoder_kind::none) command.end_encoding();
		}

		[[nodiscard]] id<MTLBuffer> make_shared_buffer(id<MTLDevice> device,
			std::span<const std::byte> bytes, std::string_view label)
		{
			if (bytes.empty()) return nil;
			const NSUInteger length = std::max<NSUInteger>(bytes.size(), 256);
			id<MTLBuffer> result = [device newBufferWithLength:length
				options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
			if (!result) fmt::throw_exception("Metal overlay buffer allocation failed");
			result.label = native_string(label);
			std::memcpy(result.contents, bytes.data(), bytes.size());
			return result;
		}

		[[nodiscard]] id<MTL4ArgumentTable> make_argument_table(id<MTLDevice> device,
			u32 buffers, u32 textures, u32 samplers, std::string_view label)
		{
			MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
			descriptor.maxBufferBindCount = buffers;
			descriptor.maxTextureBindCount = textures;
			descriptor.maxSamplerStateBindCount = samplers;
			descriptor.initializeBindings = YES;
			descriptor.label = native_string(label);
			NSError* error = nil;
			id<MTL4ArgumentTable> result = [device newArgumentTableWithDescriptor:descriptor error:&error];
			if (!result) throw_native_error(error, fmt::format("%s argument-table creation", label));
			return result;
		}

		void retire_texture(std::unique_ptr<image_view>& view, std::unique_ptr<viewable_image>& texture,
			std::string_view label)
		{
			auto& manager = get_resource_manager();
			if (manager)
			{
				manager.retire(view, {.resource_class = managed_resource_class::texture_view,
					.label = fmt::format("%s view", label)});
				manager.retire(texture, {.resource_class = managed_resource_class::texture,
					.label = std::string(label)});
			}
			else
			{
				view.reset();
				texture.reset();
			}
		}

		struct alignas(16) ui_constants
		{
			std::array<f32, 4> ui_scale{};
			std::array<f32, 4> albedo{};
			std::array<f32, 4> viewport{};
			std::array<f32, 4> clip_bounds{};
			u32 vertex_config = 0;
			u32 fragment_config = 0;
			f32 timestamp = 0.f;
			f32 blur_intensity = 0.f;
			std::array<f32, 4> sdf_params{};
			std::array<f32, 4> sdf_origin{};
			std::array<f32, 4> sdf_border_color{};
		};
		static_assert(sizeof(ui_constants) == 128);

		[[nodiscard]] std::string ui_vertex_source()
		{
			return R"MSL(
#include <metal_stdlib>
using namespace metal;
#ifndef RPCS3_MTL_UI_TYPES
#define RPCS3_MTL_UI_TYPES 1
struct ui_constants
{
	float4 ui_scale;
	float4 albedo;
	float4 viewport;
	float4 clip_bounds;
	uint vertex_config;
	uint fragment_config;
	float timestamp;
	float blur_intensity;
	float4 sdf_params;
	float4 sdf_origin;
	float4 sdf_border_color;
};
struct ui_vertex_output
{
	float4 position [[position]];
	float2 texture_coordinates;
	float4 color;
	float4 clip_rectangle;
};
#endif
float4 rsx_ui_clip_to_normalized(float4 coordinate, constant ui_constants& constants, bool flip)
{
	float4 result = (coordinate * constants.ui_scale.zwzw) / constants.ui_scale.xyxy;
	if (flip) result.yw = 1.0 - result.yw;
	return result;
}
vertex ui_vertex_output rsx_ui_vertex(uint vertex_id [[vertex_id]],
	device const float4* vertices [[buffer(0)]], constant ui_constants& constants [[buffer(1)]])
{
	const float4 input = vertices[vertex_id];
	const bool disable_snap = (constants.vertex_config & 1u) != 0;
	const bool flip = (constants.vertex_config & 2u) != 0;
	float4 clip = rsx_ui_clip_to_normalized(constants.clip_bounds, constants, flip);
	clip = clip * constants.viewport.xyxy + constants.viewport.zwzw;
	if (clip.x > clip.z) clip.xz = clip.zx;
	if (clip.y > clip.w) clip.yw = clip.wy;
	float2 position = rsx_ui_clip_to_normalized(input, constants, flip).xy;
	if (!disable_snap)
		position = floor(fma(position, constants.viewport.xy, float2(0.5))) / constants.viewport.xy;
	ui_vertex_output output;
	output.position = float4(position * 2.0 - 1.0, 0.5, 1.0);
	output.texture_coordinates = input.zw;
	output.color = constants.albedo;
	output.clip_rectangle = clip;
	return output;
}
)MSL";
		}

		[[nodiscard]] std::string ui_fragment_source()
		{
			return R"MSL(
#include <metal_stdlib>
using namespace metal;
#ifndef RPCS3_MTL_UI_TYPES
#define RPCS3_MTL_UI_TYPES 1
struct ui_constants
{
	float4 ui_scale;
	float4 albedo;
	float4 viewport;
	float4 clip_bounds;
	uint vertex_config;
	uint fragment_config;
	float timestamp;
	float blur_intensity;
	float4 sdf_params;
	float4 sdf_origin;
	float4 sdf_border_color;
};
struct ui_vertex_output
{
	float4 position [[position]];
	float2 texture_coordinates;
	float4 color;
	float4 clip_rectangle;
};
#endif
float4 rsx_ui_sdf_blend(float distance, float border_width, float4 inner_color,
	float4 border_color, float4 outer_color)
{
	const float width = abs(dfdx(distance)) + abs(dfdy(distance));
	const float inner = 1.0 - smoothstep(-border_width - width, -border_width + width, distance);
	const float outer = 1.0 - smoothstep(-width, width, distance);
	return mix(mix(outer_color, border_color, outer), inner_color, inner);
}
float rsx_ui_sdf(uint function, float2 fragment_position, constant ui_constants& constants)
{
	const float2 point = floor(fragment_position) - constants.sdf_origin.xy;
	const float2 half_size = constants.sdf_params.xy;
	const float radius = constants.sdf_params.z;
	if (function == 1u)
		return (length(point / half_size) - 1.0) * length(half_size);
	if (function == 2u)
	{
		const float2 value = abs(point) - half_size;
		return length(max(value, 0.0)) + min(max(value.x, value.y), 0.0);
	}
	if (function == 3u)
	{
		const float2 value = abs(point) - (half_size - radius);
		return length(max(value, 0.0)) + min(max(value.x, value.y), 0.0) - radius;
	}
	return -1.0;
}
float4 rsx_ui_blur(texture2d<float> texture, sampler texture_sampler,
	float2 coordinate, float2 offset)
{
	constexpr float2 positions[9] = {
		float2(-1,-1), float2(0,-1), float2(1,-1), float2(-1,0), float2(0,0),
		float2(1,0), float2(-1,1), float2(0,1), float2(1,1)};
	constexpr float weights[9] = {1,2,1,2,4,2,1,2,1};
	float4 result = 0.0;
	for (uint index = 0; index < 9; ++index)
		result += texture.sample(texture_sampler, coordinate + positions[index] * offset) * weights[index];
	return result / 16.0;
}
float4 rsx_ui_sample_image(texture2d<float> texture, sampler texture_sampler,
	float2 coordinate, float strength)
{
	const float4 original = texture.sample(texture_sampler, coordinate);
	if (strength == 0.0) return original;
	const float2 resolution_offset = 1.0 / float2(texture.get_width(), texture.get_height());
	const float2 offset = max(resolution_offset, 1.0 / float2(640.0, 360.0));
	float4 blurred = rsx_ui_blur(texture, texture_sampler, coordinate - float2(resolution_offset.x, 0), offset);
	blurred += rsx_ui_blur(texture, texture_sampler, coordinate + float2(resolution_offset.x, 0), offset);
	blurred += rsx_ui_blur(texture, texture_sampler, coordinate + float2(0, resolution_offset.y), offset);
	return mix(original, blurred / 3.0, strength);
}
fragment float4 rsx_ui_fragment(ui_vertex_output input [[stage_in]],
	constant ui_constants& constants [[buffer(0)]], texture2d<float> texture_2d [[texture(0)]],
	texture2d_array<float> texture_array [[texture(1)]], sampler texture_sampler [[sampler(0)]])
{
	const uint configuration = constants.fragment_config;
	if ((configuration & 1u) != 0 &&
		(input.position.x < input.clip_rectangle.x || input.position.x > input.clip_rectangle.z ||
		 input.position.y < input.clip_rectangle.y || input.position.y > input.clip_rectangle.w))
		discard_fragment();
	float4 color = input.color;
	if ((configuration & 2u) != 0) color.a *= (sin(constants.timestamp) + 1.0) * 0.5;
	const uint sdf_function = (configuration >> 4u) & 3u;
	if (sdf_function != 0)
	{
		const float distance = rsx_ui_sdf(sdf_function, input.position.xy, constants);
		color = rsx_ui_sdf_blend(distance, constants.sdf_params.w, color,
			constants.sdf_border_color, float4(0.0));
	}
	const uint sampling_mode = (configuration >> 2u) & 3u;
	if (sampling_mode == 1u)
		return texture_2d.sample(texture_sampler, input.texture_coordinates).rrrr * color;
	if (sampling_mode == 2u)
	{
		const float layer_coordinate = input.texture_coordinates.y;
		return texture_array.sample(texture_sampler,
			float2(input.texture_coordinates.x, fract(layer_coordinate)), uint(trunc(layer_coordinate))).rrrr * color;
	}
	if (sampling_mode == 3u)
		return rsx_ui_sample_image(texture_2d, texture_sampler,
			input.texture_coordinates, constants.blur_intensity).bgra * color;
	return color;
}
)MSL";
		}

		[[nodiscard]] std::string quad_vertex_source()
		{
			return R"MSL(
#include <metal_stdlib>
using namespace metal;
#ifndef RPCS3_OVERLAY_QUAD_OUTPUT
#define RPCS3_OVERLAY_QUAD_OUTPUT 1
struct overlay_quad_output { float4 position [[position]]; float2 coordinates; };
#endif
vertex overlay_quad_output rsx_overlay_quad_vertex(uint vertex_id [[vertex_id]])
{
	constexpr float2 positions[4] = {float2(-1,-1), float2(1,-1), float2(-1,1), float2(1,1)};
	constexpr float2 coordinates[4] = {float2(0,0), float2(1,0), float2(0,1), float2(1,1)};
	return {float4(positions[vertex_id & 3u], 0.0, 1.0), coordinates[vertex_id & 3u]};
}
)MSL";
		}

		[[nodiscard]] std::string color_clear_fragment_source()
		{
			return R"MSL(
#include <metal_stdlib>
using namespace metal;
struct clear_constants { float4 color; };
#ifndef RPCS3_OVERLAY_QUAD_OUTPUT
#define RPCS3_OVERLAY_QUAD_OUTPUT 1
struct overlay_quad_output { float4 position [[position]]; float2 coordinates; };
#endif
fragment float4 rsx_color_clear_fragment(overlay_quad_output input [[stage_in]],
	constant clear_constants& constants [[buffer(0)]])
{
	(void)input;
	return constants.color;
}
)MSL";
		}

		[[nodiscard]] std::string stencil_clear_fragment_source()
		{
			return R"MSL(
#include <metal_stdlib>
using namespace metal;
#ifndef RPCS3_OVERLAY_QUAD_OUTPUT
#define RPCS3_OVERLAY_QUAD_OUTPUT 1
struct overlay_quad_output { float4 position [[position]]; float2 coordinates; };
#endif
fragment void rsx_stencil_clear_fragment(overlay_quad_output input [[stage_in]]) { (void)input; }
)MSL";
		}

		[[nodiscard]] std::string depth_clear_fragment_source()
		{
			return R"MSL(
#include <metal_stdlib>
using namespace metal;
#ifndef RPCS3_OVERLAY_QUAD_OUTPUT
#define RPCS3_OVERLAY_QUAD_OUTPUT 1
struct overlay_quad_output { float4 position [[position]]; float2 coordinates; };
#endif
struct depth_clear_constants { float depth; };
fragment float rsx_depth_clear_fragment(overlay_quad_output input [[stage_in]],
	constant depth_clear_constants& constants [[buffer(0)]]) [[depth(any)]]
{
	(void)input;
	return constants.depth;
}
)MSL";
		}

		[[nodiscard]] std::string calibration_fragment_source()
		{
			return R"MSL(
#include <metal_stdlib>
using namespace metal;
#ifndef RPCS3_OVERLAY_QUAD_OUTPUT
#define RPCS3_OVERLAY_QUAD_OUTPUT 1
struct overlay_quad_output { float4 position [[position]]; float2 coordinates; };
#endif
struct calibration_constants
{
	float gamma;
	int limited_range;
	int stereo_display_mode;
	int stereo_image_count;
	float4 left_matrix[3];
	float4 right_matrix[3];
};
float4 rsx_calibration_anaglyph(float4 left, float4 right,
	constant calibration_constants& constants)
{
	float3 color = constants.left_matrix[0].xyz * left.r +
		constants.left_matrix[1].xyz * left.g + constants.left_matrix[2].xyz * left.b;
	color += constants.right_matrix[0].xyz * right.r +
		constants.right_matrix[1].xyz * right.g + constants.right_matrix[2].xyz * right.b;
	return float4(clamp(color, 0.0, 1.0), 1.0);
}
bool rsx_calibration_anaglyph_mode(int mode) { return mode >= 4 && mode <= 10; }
float4 rsx_calibration_source(float2 coordinate, float2 fragment_position,
	constant calibration_constants& constants, texture2d<float> first,
	texture2d<float> second, sampler source_sampler)
{
	constexpr float2 left_single = float2(1.0, 0.4898);
	constexpr float2 right_single = float2(0.0, 0.510204);
	if (constants.stereo_display_mode == 0)
		return first.sample(source_sampler, coordinate);
	if (constants.stereo_image_count == 1)
	{
		if (rsx_calibration_anaglyph_mode(constants.stereo_display_mode))
			return rsx_calibration_anaglyph(first.sample(source_sampler, coordinate * left_single),
				first.sample(source_sampler, coordinate * left_single + right_single), constants);
		if (constants.stereo_display_mode == 1)
			return coordinate.x < 0.5 ? first.sample(source_sampler, coordinate * float2(2.0, 0.4898)) :
				first.sample(source_sampler, coordinate * float2(2.0, 0.4898) + float2(-1.0, 0.510204));
		if (constants.stereo_display_mode == 2)
			return coordinate.y < 0.5 ? first.sample(source_sampler, coordinate * float2(1.0, 0.9796)) :
				first.sample(source_sampler, coordinate * float2(1.0, 0.9796) + float2(0.0, 0.020408));
		if (constants.stereo_display_mode == 3)
			return (uint(fragment_position.y) & 1u) ? first.sample(source_sampler, coordinate * left_single) :
				first.sample(source_sampler, coordinate * left_single + right_single);
		return first.sample(source_sampler, coordinate);
	}
	if (constants.stereo_image_count == 2)
	{
		if (rsx_calibration_anaglyph_mode(constants.stereo_display_mode))
			return rsx_calibration_anaglyph(first.sample(source_sampler, coordinate),
				second.sample(source_sampler, coordinate), constants);
		if (constants.stereo_display_mode == 1)
			return coordinate.x < 0.5 ? first.sample(source_sampler, coordinate * float2(2.0, 1.0)) :
				second.sample(source_sampler, coordinate * float2(2.0, 1.0) + float2(-1.0, 0.0));
		if (constants.stereo_display_mode == 2)
			return coordinate.y < 0.5 ? first.sample(source_sampler, coordinate * float2(1.0, 2.0)) :
				second.sample(source_sampler, coordinate * float2(1.0, 2.0) + float2(0.0, -1.0));
		if (constants.stereo_display_mode == 3)
			return (uint(fragment_position.y) & 1u) ? first.sample(source_sampler, coordinate) :
				second.sample(source_sampler, coordinate);
		return first.sample(source_sampler, coordinate);
	}
	return float4(1.0, 0.0, 0.0, 1.0);
}
fragment float4 rsx_calibration_fragment(overlay_quad_output input [[stage_in]],
	constant calibration_constants& constants [[buffer(0)]],
	texture2d<float> first [[texture(0)]], texture2d<float> second [[texture(1)]],
	sampler source_sampler [[sampler(0)]])
{
	float4 color = rsx_calibration_source(input.coordinates, input.position.xy,
		constants, first, second, source_sampler);
	color.rgb = pow(color.rgb, constants.gamma);
	if (constants.limited_range != 0) color = (color * 220.0 + 16.0) / 255.0;
	color.a = 1.0;
	return color;
}
)MSL";
		}
	}

	void overlay_render_target::validate() const
	{
		if (!resource || !*resource || !write_aspects ||
			(write_aspects & ~resource->aspects()) || mip_level >= resource->mipmaps() ||
			array_slice >= resource->layers() ||
			(write_aspects & texture_aspect_color && write_aspects != texture_aspect_color))
		{
			fmt::throw_exception("Invalid Metal overlay render target");
		}
		if ((write_aspects & texture_aspect_color) && !(resource->info().usage & texture_usage_render_target))
			fmt::throw_exception("Metal overlay color target lacks render-target usage");
		if ((write_aspects & (texture_aspect_depth | texture_aspect_stencil)) &&
			!(resource->info().usage & texture_usage_depth_stencil))
			fmt::throw_exception("Metal overlay depth/stencil target lacks attachment usage");
	}

	overlay_render_target::operator bool() const
	{
		return resource && static_cast<bool>(*resource);
	}

	u32 overlay_render_target::width() const
	{
		validate();
		return mip_dimension(resource->width(), mip_level);
	}

	u32 overlay_render_target::height() const
	{
		validate();
		return mip_dimension(resource->height(), mip_level);
	}

	u32 overlay_render_target::samples() const
	{
		validate();
		return resource->samples();
	}

	u64 overlay_pipeline_key::hash() const
	{
		u64 result = 0x9e3779b97f4a7c15ull;
		auto mix = [&](u64 value)
		{
			result ^= value + 0x9e3779b97f4a7c15ull + (result << 6) + (result >> 2);
		};
		mix(color_format);
		mix(depth_stencil_format);
		mix(sample_count);
		mix(static_cast<u8>(topology));
		mix(color_write_mask);
		mix(blending);
		mix(depth_write);
		mix(stencil_write);
		return result;
	}

	usz overlay_pipeline_key_hash::operator()(const overlay_pipeline_key& key) const noexcept
	{
		return static_cast<usz>(key.hash());
	}

	struct overlay_pass::impl
	{
		render_device* device = nullptr;
		memory_allocator* allocator = nullptr;
		MTLPipelineCompiler* compiler = nullptr;
		std::string label;
		std::string vertex_source;
		std::string fragment_source;
		std::string vertex_function;
		std::string fragment_function;
		std::vector<std::byte> vertex_bytes;
		std::vector<std::byte> constant_bytes;
		std::unordered_map<overlay_pipeline_key, id<MTLRenderPipelineState>,
			overlay_pipeline_key_hash> pipelines;
		std::unique_ptr<sampler> texture_sampler;
		overlay_pass_statistics stats;
		sampler_filter filter = sampler_filter::linear;
		primitive_topology topology = primitive_topology::triangle_strip;
		u32 source_texture_count = 0;
		u32 vertex_stride = 0;
		u8 color_mask = color_write_all;
		bool blending = false;
		bool depth_write = false;
		bool stencil_write = false;
		mutable std::mutex mutex;

		[[nodiscard]] id<MTLRenderPipelineState> pipeline(const overlay_pipeline_key& key)
		{
			std::lock_guard lock(mutex);
			if (const auto found = pipelines.find(key); found != pipelines.end())
			{
				++stats.pipeline_cache_hits;
				return found->second;
			}
			if (vertex_source.empty() || fragment_source.empty() ||
				vertex_function.empty() || fragment_function.empty())
			{
				fmt::throw_exception("Metal overlay pass '%s' has no shader program", label);
			}
			const std::string combined = vertex_source + "\n" + fragment_source;
			id<MTLLibrary> library = compile_library(compiler->native_compiler(), combined, label);
			MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new];
			vertex.library = library;
			vertex.name = native_string(vertex_function);
			MTL4LibraryFunctionDescriptor* fragment = [MTL4LibraryFunctionDescriptor new];
			fragment.library = library;
			fragment.name = native_string(fragment_function);
			MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
			descriptor.label = native_string(label);
			descriptor.vertexFunctionDescriptor = vertex;
			descriptor.fragmentFunctionDescriptor = fragment;
			descriptor.inputPrimitiveTopology = topology_class(key.topology);
			descriptor.rasterSampleCount = key.sample_count;
			if (key.color_format)
			{
				auto* attachment = descriptor.colorAttachments[0];
				attachment.pixelFormat = static_cast<MTLPixelFormat>(key.color_format);
				attachment.writeMask = native_color_mask(key.color_write_mask);
				attachment.blendingState = key.blending ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
				if (key.blending)
				{
					attachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
					attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
					attachment.rgbBlendOperation = MTLBlendOperationAdd;
					attachment.sourceAlphaBlendFactor = MTLBlendFactorZero;
					attachment.destinationAlphaBlendFactor = MTLBlendFactorOne;
					attachment.alphaBlendOperation = MTLBlendOperationAdd;
				}
			}
			NSError* error = nil;
			id<MTLRenderPipelineState> result = [compiler->native_compiler()
				newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
			if (!result) throw_native_error(error, fmt::format("%s pipeline compilation", label));
			pipelines.emplace(key, result);
			++stats.pipeline_builds;
			return result;
		}
	};

	overlay_pass::overlay_pass(std::string label)
		: m_impl(std::make_unique<impl>())
	{
		m_impl->label = std::move(label);
	}

	overlay_pass::~overlay_pass()
	{
		destroy();
	}

	void overlay_pass::set_shader_sources(std::string vertex_source, std::string fragment_source,
		std::string vertex_function, std::string fragment_function)
	{
		if (m_impl->device) fmt::throw_exception("Cannot replace an initialized Metal overlay shader");
		m_impl->vertex_source = std::move(vertex_source);
		m_impl->fragment_source = std::move(fragment_source);
		m_impl->vertex_function = std::move(vertex_function);
		m_impl->fragment_function = std::move(fragment_function);
	}

	void overlay_pass::set_sampler_filter(sampler_filter filter) { m_impl->filter = filter; }
	void overlay_pass::set_source_texture_count(u32 count) { m_impl->source_texture_count = count; }
	void overlay_pass::set_primitive_topology(primitive_topology topology) { m_impl->topology = topology; }
	void overlay_pass::set_color_write_mask(u8 mask) { m_impl->color_mask = mask & color_write_all; }
	void overlay_pass::set_blending(bool enabled) { m_impl->blending = enabled; }
	void overlay_pass::set_depth_write(bool enabled) { m_impl->depth_write = enabled; }
	void overlay_pass::set_stencil_write(bool enabled) { m_impl->stencil_write = enabled; }

	void overlay_pass::upload_vertex_bytes(std::span<const std::byte> bytes, u32 stride)
	{
		if (bytes.empty() || !stride || bytes.size() % stride)
			fmt::throw_exception("Invalid Metal overlay vertex payload");
		m_impl->vertex_bytes.assign(bytes.begin(), bytes.end());
		m_impl->vertex_stride = stride;
	}

	void overlay_pass::upload_constant_bytes(std::span<const std::byte> bytes)
	{
		if (bytes.empty()) fmt::throw_exception("Invalid Metal overlay constant payload");
		m_impl->constant_bytes.assign(bytes.begin(), bytes.end());
	}

	void overlay_pass::record_texture_upload(u64 bytes)
	{
		std::lock_guard lock(m_impl->mutex);
		++m_impl->stats.texture_uploads;
		m_impl->stats.uploaded_bytes += bytes;
	}

	void overlay_pass::initialize(render_device& device, memory_allocator& allocator,
		MTLPipelineCompiler& compiler)
	{
		if (!device || !compiler || &compiler.owner() != &device || &allocator.device() != &device)
			fmt::throw_exception("Metal overlay pass requires matching device resources");
		if (m_impl->device && (m_impl->device != &device || m_impl->allocator != &allocator ||
			m_impl->compiler != &compiler))
		{
			fmt::throw_exception("Metal overlay pass cannot change devices while initialized");
		}
		m_impl->device = &device;
		m_impl->allocator = &allocator;
		m_impl->compiler = &compiler;
		if (m_impl->source_texture_count && !m_impl->texture_sampler)
		{
			sampler_description description;
			description.address_s = sampler_address_mode::clamp_to_edge;
			description.address_t = sampler_address_mode::clamp_to_edge;
			description.address_r = sampler_address_mode::clamp_to_edge;
			description.min_filter = m_impl->filter;
			description.mag_filter = m_impl->filter;
			description.mip_filter = sampler_mip_filter::nearest;
			description.border = border_color::opaque_black();
			m_impl->texture_sampler = std::make_unique<sampler>(device, description,
				fmt::format("%s sampler", m_impl->label));
		}
	}

	void overlay_pass::destroy()
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->pipelines.clear();
		m_impl->texture_sampler.reset();
		m_impl->vertex_bytes.clear();
		m_impl->constant_bytes.clear();
		m_impl->device = nullptr;
		m_impl->allocator = nullptr;
		m_impl->compiler = nullptr;
		m_impl->stats = {};
	}

	void overlay_pass::reclaim(u64 completed_submission)
	{
		if (!completed_submission) fmt::throw_exception("Invalid Metal overlay completion value");
		get_resource_manager().complete(completed_submission);
	}

	void overlay_pass::trim(memory_pressure pressure)
	{
		if (pressure == memory_pressure::normal) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->pipelines.clear();
		if (pressure == memory_pressure::critical)
		{
			m_impl->vertex_bytes.clear();
			m_impl->constant_bytes.clear();
		}
	}

	overlay_pass::operator bool() const
	{
		return m_impl && m_impl->device && m_impl->allocator && m_impl->compiler;
	}

	overlay_pass_statistics overlay_pass::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->stats;
	}

	void overlay_pass::draw(command_buffer& command, const areau& viewport,
		const overlay_render_target& target, std::span<image_view* const> sources,
		u32 vertex_count, u32 first_vertex, u32 instance_count, u32 base_instance,
		const areau* scissor, u32 stencil_reference, u32 stencil_write_mask)
	{
		target.validate();
		if (!*this || !command.is_recording() || !vertex_count || !instance_count ||
			viewport.width() == 0 || viewport.height() == 0 ||
			viewport.x2 > target.width() || viewport.y2 > target.height() ||
			sources.size() > m_impl->source_texture_count)
		{
			fmt::throw_exception("Invalid Metal overlay draw");
		}

		for (image_view* source : sources)
		{
			if (!source) continue;
			image* resource = source->image();
			if (!resource || !*resource || !(resource->info().usage & texture_usage_shader_read))
				fmt::throw_exception("Invalid Metal overlay source texture");
			transition_image(command, *resource,
				operation_state(*resource, stage_fragment, access_shader_read), source->range(), false);
		}
		const u64 target_access = target.write_aspects & texture_aspect_color
			? access_color_write : access_depth_stencil_write;
		transition_image(command, *target.resource,
			operation_state(*target.resource, stage_fragment | stage_tile, target_access),
			target_range(target), false);
		end_active_encoder(command);

		const overlay_pipeline_key key{
			.color_format = target.write_aspects & texture_aspect_color ? target.resource->format() : 0,
			.depth_stencil_format = target.write_aspects & (texture_aspect_depth | texture_aspect_stencil)
				? target.resource->format() : 0,
			.sample_count = target.samples(),
			.topology = m_impl->topology,
			.color_write_mask = m_impl->color_mask,
			.blending = m_impl->blending,
			.depth_write = m_impl->depth_write,
			.stencil_write = m_impl->stencil_write,
		};
		id<MTLRenderPipelineState> pipeline = m_impl->pipeline(key);
		id<MTLDevice> device = m_impl->device->native_handle();
		id<MTLBuffer> vertex_buffer = make_shared_buffer(device, m_impl->vertex_bytes,
			fmt::format("%s vertices", m_impl->label));
		id<MTLBuffer> constant_buffer = make_shared_buffer(device, m_impl->constant_bytes,
			fmt::format("%s constants", m_impl->label));
		id<MTL4ArgumentTable> vertex_table = make_argument_table(device, 2, 0, 0,
			fmt::format("%s vertex arguments", m_impl->label));
		id<MTL4ArgumentTable> fragment_table = make_argument_table(device, 1,
			m_impl->source_texture_count, m_impl->source_texture_count ? 1 : 0,
			fmt::format("%s fragment arguments", m_impl->label));
		if (vertex_buffer) [vertex_table setAddress:vertex_buffer.gpuAddress atIndex:0];
		if (constant_buffer)
		{
			[vertex_table setAddress:constant_buffer.gpuAddress atIndex:1];
			[fragment_table setAddress:constant_buffer.gpuAddress atIndex:0];
		}
		for (u32 index = 0; index < sources.size(); ++index)
		{
			if (sources[index])
				[fragment_table setTexture:sources[index]->native_handle().gpuResourceID atIndex:index];
		}
		if (m_impl->source_texture_count)
			[fragment_table setSamplerState:m_impl->texture_sampler->native_handle().gpuResourceID atIndex:0];

		command.retain_native_object((__bridge void*)target.resource->native_handle(), true);
		command.retain_native_object((__bridge void*)pipeline, false);
		command.retain_native_object((__bridge void*)vertex_table, false);
		command.retain_native_object((__bridge void*)fragment_table, false);
		if (vertex_buffer) command.retain_native_object((__bridge void*)vertex_buffer, true);
		if (constant_buffer) command.retain_native_object((__bridge void*)constant_buffer, true);
		if (m_impl->texture_sampler)
			command.retain_native_object((__bridge void*)m_impl->texture_sampler->native_handle(), false);
		for (image_view* source : sources)
		{
			if (!source) continue;
			command.retain_native_object((__bridge void*)source->native_handle(), true);
			command.retain_native_object((__bridge void*)source->image()->native_handle(), true);
		}

		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		pass.renderTargetWidth = target.width();
		pass.renderTargetHeight = target.height();
		pass.defaultRasterSampleCount = target.samples();
		const MTLLoadAction load = target.preserve_contents ? MTLLoadActionLoad : MTLLoadActionDontCare;
		if (target.write_aspects & texture_aspect_color)
		{
			auto* attachment = pass.colorAttachments[0];
			attachment.texture = target.resource->native_handle();
			attachment.level = target.mip_level;
			attachment.slice = target.array_slice;
			attachment.loadAction = load;
			attachment.storeAction = MTLStoreActionStore;
		}
		if (target.write_aspects & texture_aspect_depth)
		{
			pass.depthAttachment.texture = target.resource->native_handle();
			pass.depthAttachment.level = target.mip_level;
			pass.depthAttachment.slice = target.array_slice;
			pass.depthAttachment.loadAction = load;
			pass.depthAttachment.storeAction = MTLStoreActionStore;
		}
		if (target.write_aspects & texture_aspect_stencil)
		{
			pass.stencilAttachment.texture = target.resource->native_handle();
			pass.stencilAttachment.level = target.mip_level;
			pass.stencilAttachment.slice = target.array_slice;
			pass.stencilAttachment.loadAction = load;
			pass.stencilAttachment.storeAction = MTLStoreActionStore;
			if ((target.resource->aspects() & texture_aspect_depth) &&
				!(target.write_aspects & texture_aspect_depth))
			{
				pass.depthAttachment.texture = target.resource->native_handle();
				pass.depthAttachment.level = target.mip_level;
				pass.depthAttachment.slice = target.array_slice;
				pass.depthAttachment.loadAction = MTLLoadActionLoad;
				pass.depthAttachment.storeAction = MTLStoreActionStore;
			}
		}

		id<MTLDepthStencilState> depth_stencil_state = nil;
		if (m_impl->depth_write || m_impl->stencil_write)
		{
			MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
			descriptor.depthCompareFunction = MTLCompareFunctionAlways;
			descriptor.depthWriteEnabled = m_impl->depth_write;
			if (m_impl->stencil_write)
			{
				MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
				stencil.stencilCompareFunction = MTLCompareFunctionAlways;
				stencil.stencilFailureOperation = MTLStencilOperationReplace;
				stencil.depthFailureOperation = MTLStencilOperationReplace;
				stencil.depthStencilPassOperation = MTLStencilOperationReplace;
				stencil.readMask = 0xff;
				stencil.writeMask = stencil_write_mask;
				descriptor.frontFaceStencil = stencil;
				descriptor.backFaceStencil = stencil;
			}
			depth_stencil_state = [device newDepthStencilStateWithDescriptor:descriptor];
			if (!depth_stencil_state) fmt::throw_exception("Metal overlay depth/stencil state creation failed");
			command.retain_native_object((__bridge void*)depth_stencil_state, false);
		}

		id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)
			command.begin_render_encoding((__bridge void*)pass);
		[encoder setRenderPipelineState:pipeline];
		[encoder setArgumentTable:vertex_table atStages:MTLRenderStageVertex];
		[encoder setArgumentTable:fragment_table atStages:MTLRenderStageFragment];
		if (depth_stencil_state) [encoder setDepthStencilState:depth_stencil_state];
		if (m_impl->stencil_write) [encoder setStencilReferenceValue:stencil_reference];
		[encoder setViewport:MTLViewport{static_cast<f64>(viewport.x1), static_cast<f64>(viewport.y1),
			static_cast<f64>(viewport.width()), static_cast<f64>(viewport.height()), 0.0, 1.0}];
		const areau& clipping = scissor ? *scissor : viewport;
		if (!clipping.width() || !clipping.height() || clipping.x2 > target.width() ||
			clipping.y2 > target.height())
		{
			fmt::throw_exception("Invalid Metal overlay scissor rectangle");
		}
		[encoder setScissorRect:MTLScissorRect{clipping.x1, clipping.y1,
			clipping.width(), clipping.height()}];
		[encoder drawPrimitives:primitive_type(m_impl->topology) vertexStart:first_vertex
			vertexCount:vertex_count instanceCount:instance_count baseInstance:base_instance];
		barrier_plan visibility;
		visibility.scope = barrier_scope::between_encoders;
		visibility.after_stages = stage_fragment | stage_tile;
		visibility.before_stages = stage_all_gpu;
		visibility.flush_caches = true;
		visibility.end_encoder = true;
		visibility.producer_barrier = true;
		encode_barrier(command.active_native_encoder(), visibility);
		command.end_encoding();
		command.set_flag(command_reload_dynamic_state);

		std::lock_guard lock(m_impl->mutex);
		++m_impl->stats.draw_calls;
		m_impl->stats.vertices += static_cast<u64>(vertex_count) * instance_count;
	}

	struct ui_overlay_renderer::ui_impl
	{
		struct texture_entry
		{
			u32 owner_uid = umax;
			std::unique_ptr<viewable_image> texture;
			std::unique_ptr<image_view> view;
		};

		memory_allocator* allocator = nullptr;
		std::unordered_map<u64, texture_entry> resources;
		std::unordered_map<u64, texture_entry> fonts;
		std::unordered_map<u64, texture_entry> temporary;
	};

	ui_overlay_renderer::ui_overlay_renderer()
		: overlay_pass("RPCS3 UI overlay")
		, m_ui(std::make_unique<ui_impl>())
	{
		set_shader_sources(ui_vertex_source(), ui_fragment_source(),
			"rsx_ui_vertex", "rsx_ui_fragment");
		set_source_texture_count(2);
		set_sampler_filter(sampler_filter::linear);
		set_primitive_topology(primitive_topology::triangle_strip);
		set_color_write_mask(color_write_all);
		set_blending(true);
	}

	ui_overlay_renderer::~ui_overlay_renderer()
	{
		destroy();
	}

	void ui_overlay_renderer::initialize(render_device& device, memory_allocator& allocator,
		MTLPipelineCompiler& compiler)
	{
		overlay_pass::initialize(device, allocator, compiler);
		m_ui->allocator = &allocator;
	}

	image_view* ui_overlay_renderer::upload_simple_texture(command_buffer& command,
		data_heap& upload_heap, u64 key, u32 width, u32 height, u32 layers,
		bool font, bool temporary, const void* pixels, u32 owner_uid)
	{
		if (!*this || !m_ui->allocator || !key || !width || !height || !layers ||
			!command.is_recording() || upload_heap.size() == 0)
		{
			fmt::throw_exception("Invalid Metal UI texture upload");
		}
		const u32 bytes_per_pixel = font ? 1 : 4;
		const u64 source_row_bytes = static_cast<u64>(width) * bytes_per_pixel;
		const u64 row_bytes = (source_row_bytes + 255) & ~255ull;
		const u64 layer_bytes = row_bytes * height;
		if (layer_bytes > std::numeric_limits<u64>::max() / layers)
			fmt::throw_exception("Metal UI texture upload size overflows");
		const u64 upload_bytes = layer_bytes * layers;
		data_heap_slice slice = upload_heap.allocate(upload_bytes, 512);
		auto* destination = static_cast<u8*>(upload_heap.map(slice));
		std::memset(destination, 0, upload_bytes);
		if (pixels)
		{
			const auto* source = static_cast<const u8*>(pixels);
			for (u32 layer = 0; layer < layers; ++layer)
			{
				for (u32 row = 0; row < height; ++row)
				{
					std::memcpy(destination + layer * layer_bytes + row * row_bytes,
						source + (static_cast<u64>(layer) * height + row) * source_row_bytes,
						source_row_bytes);
				}
			}
		}
		upload_heap.mark_modified(slice);
		upload_heap.unmap();
		upload_heap.flush(command);

		const MTLPixelFormat format = font ? MTLPixelFormatR8Unorm : MTLPixelFormatBGRA8Unorm;
		image_create_info info;
		info.type = layers > 1 ? texture_type::texture_2d_array : texture_type::texture_2d;
		info.formats = get_view_compatibility(format);
		info.width = width;
		info.height = height;
		info.array_layers = layers;
		info.usage = texture_usage_shader_read | texture_usage_copy_source |
			texture_usage_copy_destination | texture_usage_pixel_format_view;
		info.aspects = texture_aspect_color;
		info.storage = storage_mode::private_;
		info.pool = allocation_pool::texture_cache;
		info.label = font ? "RPCS3 UI font" : "RPCS3 UI image";
		auto texture = std::make_unique<viewable_image>(*m_ui->allocator, info);
		const buffer_image_copy_region region{
			.buffer_offset = slice.offset,
			.bytes_per_row = row_bytes,
			.bytes_per_image = layer_bytes,
			.subresource = {.aspects = texture_aspect_color},
			.extent = {width, height, 1},
			.layer_count = layers,
		};
		upload_image(command, upload_heap.target_buffer(), *texture, std::span{&region, 1});
		auto view = std::make_unique<image_view>(*texture, format, info.type,
			default_component_map, subresource_range{0, 1, 0, layers, true, false, false});
		image_view* result = view.get();
		ui_impl::texture_entry entry{owner_uid, std::move(texture), std::move(view)};
		auto& cache = font ? m_ui->fonts : (temporary ? m_ui->temporary : m_ui->resources);
		if (auto found = cache.find(key); found != cache.end())
		{
			retire_texture(found->second.view, found->second.texture, "RPCS3 replaced UI texture");
			cache.erase(found);
		}
		cache.emplace(key, std::move(entry));
		record_texture_upload(upload_bytes);
		return result;
	}

	void ui_overlay_renderer::initialize_resources(command_buffer& command, data_heap& upload_heap)
	{
		rsx::overlays::resource_config configuration;
		configuration.load_files();
		u64 key = 1;
		for (const auto& resource : configuration.texture_raw_data)
		{
			static_cast<void>(upload_simple_texture(command, upload_heap, key++, resource->w,
				resource->h, 1, false, false, resource->get_data(), umax));
		}
		configuration.free_resources();
	}

	void ui_overlay_renderer::destroy()
	{
		if (m_ui)
		{
			m_ui->temporary.clear();
			m_ui->fonts.clear();
			m_ui->resources.clear();
			m_ui->allocator = nullptr;
		}
		overlay_pass::destroy();
	}

	void ui_overlay_renderer::remove_temporary_resources(u32 owner_uid)
	{
		for (auto iterator = m_ui->temporary.begin(); iterator != m_ui->temporary.end();)
		{
			if (iterator->second.owner_uid != owner_uid)
			{
				++iterator;
				continue;
			}
			retire_texture(iterator->second.view, iterator->second.texture,
				"RPCS3 temporary UI texture");
			iterator = m_ui->temporary.erase(iterator);
		}
	}

	image_view* ui_overlay_renderer::find_font(const rsx::overlays::font* font,
		command_buffer& command, data_heap& upload_heap)
	{
		if (!font) fmt::throw_exception("Metal UI font reference is null");
		const auto dimensions = font->get_glyph_data_dimensions();
		const u64 key = reinterpret_cast<u64>(font);
		if (auto found = m_ui->fonts.find(key); found != m_ui->fonts.end())
		{
			if (found->second.texture->width() == dimensions.width &&
				found->second.texture->height() == dimensions.height &&
				found->second.texture->layers() == dimensions.depth)
			{
				return found->second.view.get();
			}
			retire_texture(found->second.view, found->second.texture, "RPCS3 resized UI font");
			m_ui->fonts.erase(found);
		}
		const std::vector<u8>& bytes = font->get_glyph_data();
		return upload_simple_texture(command, upload_heap, key, dimensions.width,
			dimensions.height, dimensions.depth, true, false, bytes.data(), umax);
	}

	image_view* ui_overlay_renderer::find_temporary_image(
		const rsx::overlays::image_info_base* description, command_buffer& command,
		data_heap& upload_heap, u32 owner_uid)
	{
		if (!description || description->w <= 0 || description->h <= 0)
			fmt::throw_exception("Invalid Metal temporary UI image description");
		const u64 key = reinterpret_cast<u64>(description);
		const bool dirty = std::exchange(description->dirty, false);
		if (auto found = m_ui->temporary.find(key); found != m_ui->temporary.end() && !dirty &&
			found->second.texture->width() == static_cast<u32>(description->w) &&
			found->second.texture->height() == static_cast<u32>(description->h))
		{
			return found->second.view.get();
		}
		return upload_simple_texture(command, upload_heap, key,
			static_cast<u32>(description->w), static_cast<u32>(description->h), 1,
			false, true, description->get_data(), owner_uid);
	}

	void ui_overlay_renderer::run(command_buffer& command, const areau& viewport,
		const overlay_render_target& target, data_heap& upload_heap,
		rsx::overlays::overlay& ui)
	{
		if (!viewport.width() || !viewport.height()) return;
		ui.set_render_viewport(
			static_cast<u16>(std::min<u32>(viewport.width(), std::numeric_limits<u16>::max())),
			static_cast<u16>(std::min<u32>(viewport.height(), std::numeric_limits<u16>::max())));
		if (ui.status_flags & rsx::overlays::status_bits::invalidate_image_cache)
		{
			remove_temporary_resources(ui.uid);
			ui.status_flags.clear(rsx::overlays::status_bits::invalidate_image_cache);
		}

		auto compiled = ui.get_compiled();
		for (const auto& draw_command : compiled.draw_commands)
		{
			if (draw_command.verts.empty()) continue;
			std::array<image_view*, 2> sources{};
			rsx::overlays::texture_sampling_mode texture_mode =
				rsx::overlays::texture_sampling_mode::texture2D;
			image_view* source = nullptr;
			switch (draw_command.config.texture_ref)
			{
			case rsx::overlays::image_resource_id::game_icon:
			case rsx::overlays::image_resource_id::backbuffer:
			case rsx::overlays::image_resource_id::none:
				texture_mode = rsx::overlays::texture_sampling_mode::none;
				break;
			case rsx::overlays::image_resource_id::font_file:
				source = find_font(draw_command.config.font_ref, command, upload_heap);
				texture_mode = source->image()->layers() == 1
					? rsx::overlays::texture_sampling_mode::font2D
					: rsx::overlays::texture_sampling_mode::font3D;
				break;
			case rsx::overlays::image_resource_id::raw_image:
				source = find_temporary_image(static_cast<const rsx::overlays::image_info_base*>(
					draw_command.config.external_data_ref), command, upload_heap, ui.uid);
				break;
			default:
				if (const auto found = m_ui->resources.find(draw_command.config.texture_ref);
					found != m_ui->resources.end())
				{
					source = found->second.view.get();
				}
				else
				{
					fmt::throw_exception("Metal UI references missing image resource %u",
						draw_command.config.texture_ref);
				}
				break;
			}
			if (source) sources[source->image()->layers() > 1 ? 1 : 0] = source;

			ui_constants constants;
			constants.ui_scale = {static_cast<f32>(ui.get_virtual_width()),
				static_cast<f32>(ui.get_virtual_height()), 1.f, 1.f};
			std::copy_n(draw_command.config.color.rgba, 4, constants.albedo.begin());
			constants.viewport = {static_cast<f32>(viewport.width()), static_cast<f32>(viewport.height()),
				static_cast<f32>(viewport.x1), static_cast<f32>(viewport.y1)};
			constants.clip_bounds = {draw_command.config.clip_rect.x1, draw_command.config.clip_rect.y1,
				draw_command.config.clip_rect.x2, draw_command.config.clip_rect.y2};
			constants.vertex_config = rsx::overlays::vertex_options{}
				.disable_vertex_snap(draw_command.config.disable_vertex_snap).get();
			constants.fragment_config = rsx::overlays::fragment_options{}
				.texture_mode(texture_mode)
				.clip_fragments(draw_command.config.clip_region)
				.pulse_glow(draw_command.config.pulse_glow)
				.set_sdf(draw_command.config.sdf_config.func).get();
			constants.timestamp = draw_command.config.get_sinus_value();
			constants.blur_intensity = static_cast<f32>(draw_command.config.blur_strength) * 0.01f;
			auto sdf = draw_command.config.sdf_config;
			sdf.transform(static_cast<areaf>(viewport),
				{constants.ui_scale[0], constants.ui_scale[1]});
			constants.sdf_params = {sdf.hx, sdf.hy, sdf.br, sdf.bw};
			constants.sdf_origin = {sdf.cx, sdf.cy, 0.f, 0.f};
			std::copy_n(sdf.border_color.rgba, 4, constants.sdf_border_color.begin());
			upload_constants(constants);

			std::vector<vertex> expanded_vertices;
			std::span<const vertex> vertices = draw_command.verts;
			primitive_topology topology = primitive_topology::triangle_strip;
			switch (draw_command.config.primitives)
			{
			case rsx::overlays::primitive_type::quad_list:
			case rsx::overlays::primitive_type::triangle_strip:
				topology = primitive_topology::triangle_strip;
				break;
			case rsx::overlays::primitive_type::line_list:
				topology = primitive_topology::line;
				break;
			case rsx::overlays::primitive_type::line_strip:
				topology = primitive_topology::line_strip;
				break;
			case rsx::overlays::primitive_type::triangle_fan:
				topology = primitive_topology::triangle;
				if (vertices.size() >= 3)
				{
					expanded_vertices.reserve((vertices.size() - 2) * 3);
					for (usz index = 1; index + 1 < vertices.size(); ++index)
					{
						expanded_vertices.push_back(vertices[0]);
						expanded_vertices.push_back(vertices[index]);
						expanded_vertices.push_back(vertices[index + 1]);
					}
					vertices = expanded_vertices;
				}
				break;
			}
			if (vertices.empty()) continue;
			set_primitive_topology(topology);
			upload_vertex_data(vertices);
			overlay_render_target preserved_target = target;
			preserved_target.preserve_contents = true;
			if (draw_command.config.primitives == rsx::overlays::primitive_type::quad_list)
			{
				if (vertices.size() % 4)
					fmt::throw_exception("Metal UI quad list has an incomplete quad");
				for (u32 first = 0; first < vertices.size(); first += 4)
					draw(command, viewport, preserved_target, sources, 4, first);
			}
			else
			{
				draw(command, viewport, preserved_target, sources,
					static_cast<u32>(vertices.size()));
			}
		}
		ui.update(get_system_time());
	}

	attachment_clear_pass::attachment_clear_pass()
		: overlay_pass("RPCS3 color attachment clear")
	{
		set_shader_sources(quad_vertex_source(), color_clear_fragment_source(),
			"rsx_overlay_quad_vertex", "rsx_color_clear_fragment");
		set_source_texture_count(0);
		set_primitive_topology(primitive_topology::triangle_strip);
		set_blending(false);
	}

	void attachment_clear_pass::run(command_buffer& command,
		const overlay_render_target& input_target, const areau& rectangle,
		u32 clear_mask, color4f color)
	{
		overlay_render_target target = input_target;
		target.write_aspects = texture_aspect_color;
		target.preserve_contents = true;
		u8 write_mask = color_write_none;
		if (clear_mask & 0x10) write_mask |= color_write_red;
		if (clear_mask & 0x20) write_mask |= color_write_green;
		if (clear_mask & 0x40) write_mask |= color_write_blue;
		if (clear_mask & 0x80) write_mask |= color_write_alpha;
		if (!write_mask) return;
		set_color_write_mask(write_mask);
		struct alignas(16) clear_constants { std::array<f32, 4> color; } constants;
		std::copy_n(color.rgba, 4, constants.color.begin());
		upload_constants(constants);
		draw(command, {0, 0, target.width(), target.height()}, target, {}, 4,
			0, 1, 0, &rectangle);
	}

	stencil_clear_pass::stencil_clear_pass()
		: overlay_pass("RPCS3 stencil attachment clear")
	{
		set_shader_sources(quad_vertex_source(), stencil_clear_fragment_source(),
			"rsx_overlay_quad_vertex", "rsx_stencil_clear_fragment");
		set_source_texture_count(0);
		set_primitive_topology(primitive_topology::triangle_strip);
		set_color_write_mask(color_write_none);
		set_stencil_write(true);
	}

	void stencil_clear_pass::run(command_buffer& command, render_target& target,
		const areau& rectangle, u32 stencil_value, u32 stencil_write_mask)
	{
		if (!(target.aspects() & texture_aspect_stencil) || !stencil_write_mask) return;
		overlay_render_target attachment{&target, 0, 0, texture_aspect_stencil, true};
		draw(command, {0, 0, target.width(), target.height()}, attachment, {}, 4,
			0, 1, 0, &rectangle, stencil_value & 0xff, stencil_write_mask & 0xff);
		if ((stencil_write_mask & 0xff) == 0xff)
			target.stencil_init_flags = (stencil_value & 0xff) | 0x100;
		else if (target.stencil_init_flags & 0xff00)
			target.stencil_init_flags = ((target.stencil_init_flags & ~stencil_write_mask) |
				(stencil_value & stencil_write_mask)) | 0x100;
	}

	depth_clear_pass::depth_clear_pass()
		: overlay_pass("RPCS3 depth attachment clear")
	{
		set_shader_sources(quad_vertex_source(), depth_clear_fragment_source(),
			"rsx_overlay_quad_vertex", "rsx_depth_clear_fragment");
		set_source_texture_count(0);
		set_primitive_topology(primitive_topology::triangle_strip);
		set_color_write_mask(color_write_none);
		set_depth_write(true);
	}

	void depth_clear_pass::run(command_buffer& command, render_target& target,
		const areau& rectangle, f32 depth_value)
	{
		if (!(target.aspects() & texture_aspect_depth) || !std::isfinite(depth_value))
			fmt::throw_exception("Invalid Metal depth clear request");
		struct alignas(16) depth_constants
		{
			f32 depth;
			std::array<f32, 3> padding{};
		} constants{depth_value};
		upload_constants(constants);
		overlay_render_target attachment{&target, 0, 0, texture_aspect_depth, true};
		draw(command, {0, 0, target.width(), target.height()}, attachment, {}, 4,
			0, 1, 0, &rectangle);
	}

	video_out_calibration_pass::video_out_calibration_pass()
		: overlay_pass("RPCS3 video output calibration")
	{
		set_shader_sources(quad_vertex_source(), calibration_fragment_source(),
			"rsx_overlay_quad_vertex", "rsx_calibration_fragment");
		set_source_texture_count(2);
		set_sampler_filter(sampler_filter::linear);
		set_primitive_topology(primitive_topology::triangle_strip);
		set_color_write_mask(color_write_all);
		set_blending(false);
	}

	void video_out_calibration_pass::run(command_buffer& command, const areau& viewport,
		const overlay_render_target& target, std::span<viewable_image* const> source_images,
		f32 gamma, bool limited_rgb, bool stereo_enabled)
	{
		if (source_images.empty() || source_images.size() > 2)
			fmt::throw_exception("Metal video calibration requires one or two source images");
		static stereo_config stereo_configuration(true);
		stereo_configuration.update_from_config(stereo_enabled);
		const auto& matrices = stereo_configuration.matrices();
		m_config = {};
		m_config.gamma = gamma;
		m_config.limited_range = limited_rgb ? 1 : 0;
		m_config.stereo_display_mode = static_cast<u8>(stereo_configuration.stereo_mode());
		m_config.stereo_image_count = static_cast<s32>(source_images.size());
		for (u32 row = 0; row < 3; ++row)
		{
			std::memcpy(m_config.left_anaglyph_matrix[row].rgba, matrices.left[row].rgb,
				sizeof(matrices.left[row].rgb));
			std::memcpy(m_config.right_anaglyph_matrix[row].rgba, matrices.right[row].rgb,
				sizeof(matrices.right[row].rgb));
		}
		upload_constants(m_config);
		std::array<image_view*, 2> views{};
		for (u32 index = 0; index < source_images.size(); ++index)
		{
			viewable_image* image = source_images[index];
			if (!image || !*image || image->samples() != 1)
				fmt::throw_exception("Invalid Metal video calibration source");
			views[index] = image->get_view(image->format(), image->type(), default_component_map,
				{0, 1, 0, 1, true, false, false});
		}
		draw(command, viewport, target, views, 4);
	}

	struct overlay_pass_manager::impl
	{
		render_device* device = nullptr;
		memory_allocator* allocator = nullptr;
		MTLPipelineCompiler* compiler = nullptr;
		std::unique_ptr<ui_overlay_renderer> ui_pass;
		std::unique_ptr<attachment_clear_pass> color_clear_pass;
		std::unique_ptr<stencil_clear_pass> stencil_clear_pass;
		std::unique_ptr<depth_clear_pass> depth_pass;
		std::unique_ptr<video_out_calibration_pass> calibration_pass;
	};

	overlay_pass_manager::overlay_pass_manager()
		: m_impl(std::make_unique<impl>())
	{
	}

	overlay_pass_manager::~overlay_pass_manager()
	{
		destroy();
	}

	void overlay_pass_manager::initialize(render_device& device, memory_allocator& allocator,
		MTLPipelineCompiler& compiler)
	{
		if (m_impl->device && (m_impl->device != &device || m_impl->allocator != &allocator ||
			m_impl->compiler != &compiler))
		{
			fmt::throw_exception("Metal overlay manager cannot change devices while initialized");
		}
		m_impl->device = &device;
		m_impl->allocator = &allocator;
		m_impl->compiler = &compiler;
	}

	void overlay_pass_manager::destroy()
	{
		if (!m_impl) return;
		m_impl->calibration_pass.reset();
		m_impl->depth_pass.reset();
		m_impl->stencil_clear_pass.reset();
		m_impl->color_clear_pass.reset();
		m_impl->ui_pass.reset();
		m_impl->device = nullptr;
		m_impl->allocator = nullptr;
		m_impl->compiler = nullptr;
	}

	void overlay_pass_manager::reclaim(u64 completed_submission)
	{
		if (!*this || !completed_submission)
			fmt::throw_exception("Invalid Metal overlay manager completion value");
		get_resource_manager().complete(completed_submission);
	}

	void overlay_pass_manager::trim(memory_pressure pressure)
	{
		if (m_impl->ui_pass) m_impl->ui_pass->trim(pressure);
		if (m_impl->color_clear_pass) m_impl->color_clear_pass->trim(pressure);
		if (m_impl->stencil_clear_pass) m_impl->stencil_clear_pass->trim(pressure);
		if (m_impl->depth_pass) m_impl->depth_pass->trim(pressure);
		if (m_impl->calibration_pass) m_impl->calibration_pass->trim(pressure);
	}

	ui_overlay_renderer& overlay_pass_manager::ui()
	{
		if (!*this) fmt::throw_exception("Metal overlay manager is not initialized");
		if (!m_impl->ui_pass)
		{
			m_impl->ui_pass = std::make_unique<ui_overlay_renderer>();
			m_impl->ui_pass->initialize(*m_impl->device, *m_impl->allocator, *m_impl->compiler);
		}
		return *m_impl->ui_pass;
	}

	attachment_clear_pass& overlay_pass_manager::color_clear()
	{
		if (!*this) fmt::throw_exception("Metal overlay manager is not initialized");
		if (!m_impl->color_clear_pass)
		{
			m_impl->color_clear_pass = std::make_unique<attachment_clear_pass>();
			m_impl->color_clear_pass->initialize(*m_impl->device, *m_impl->allocator, *m_impl->compiler);
		}
		return *m_impl->color_clear_pass;
	}

	stencil_clear_pass& overlay_pass_manager::stencil_clear()
	{
		if (!*this) fmt::throw_exception("Metal overlay manager is not initialized");
		if (!m_impl->stencil_clear_pass)
		{
			m_impl->stencil_clear_pass = std::make_unique<stencil_clear_pass>();
			m_impl->stencil_clear_pass->initialize(*m_impl->device, *m_impl->allocator, *m_impl->compiler);
		}
		return *m_impl->stencil_clear_pass;
	}

	depth_clear_pass& overlay_pass_manager::depth_clear()
	{
		if (!*this) fmt::throw_exception("Metal overlay manager is not initialized");
		if (!m_impl->depth_pass)
		{
			m_impl->depth_pass = std::make_unique<depth_clear_pass>();
			m_impl->depth_pass->initialize(*m_impl->device, *m_impl->allocator, *m_impl->compiler);
		}
		return *m_impl->depth_pass;
	}

	video_out_calibration_pass& overlay_pass_manager::video_calibration()
	{
		if (!*this) fmt::throw_exception("Metal overlay manager is not initialized");
		if (!m_impl->calibration_pass)
		{
			m_impl->calibration_pass = std::make_unique<video_out_calibration_pass>();
			m_impl->calibration_pass->initialize(*m_impl->device, *m_impl->allocator, *m_impl->compiler);
		}
		return *m_impl->calibration_pass;
	}

	overlay_pass_manager::operator bool() const
	{
		return m_impl && m_impl->device && m_impl->allocator && m_impl->compiler;
	}
}
