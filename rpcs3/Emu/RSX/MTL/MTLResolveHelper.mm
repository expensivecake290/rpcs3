#include "stdafx.h"
#include "MTLResolveHelper.h"

#include "MTLFormats.h"
#include "MTLPipelineCompiler.h"
#include "mtlutils/barriers.h"
#include "mtlutils/image_helpers.h"
#include "mtlutils/shared.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace mtl
{
	namespace
	{
		struct alignas(16) resolve_parameters
		{
			std::array<u32, 2> multisampled_origin{};
			std::array<u32, 2> expanded_origin{};
			std::array<u32, 2> output_extent{};
			std::array<u32, 2> sample_grid{};
			u32 copy_stencil = 1;
			std::array<u32, 3> padding{};
		};

		struct render_pipeline_key
		{
			resolve_direction direction = resolve_direction::multisample_to_expanded;
			resolve_execution_path path = resolve_execution_path::color_render;
			u64 format = 0;
			u32 samples = 1;

			[[nodiscard]] bool operator==(const render_pipeline_key&) const = default;
		};

		struct render_pipeline_key_hash
		{
			[[nodiscard]] usz operator()(const render_pipeline_key& key) const noexcept
			{
				u64 hash = 0x9e3779b97f4a7c15ull;
				auto mix = [&](u64 value)
				{
					hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
				};
				mix(static_cast<u8>(key.direction));
				mix(static_cast<u8>(key.path));
				mix(key.format);
				mix(key.samples);
				return static_cast<usz>(hash);
			}
		};

		[[nodiscard]] u8 effective_aspects(const resolve_request& request)
		{
			if (request.aspects != texture_aspect_none)
				return request.aspects;
			return request.multisampled && request.expanded
				? request.multisampled->aspects() & request.expanded->aspects()
				: texture_aspect_none;
		}

		[[nodiscard]] bool supports_float_color_access(u64 format)
		{
			switch (static_cast<MTLPixelFormat>(format))
			{
			case MTLPixelFormatR8Unorm:
			case MTLPixelFormatR8Unorm_sRGB:
			case MTLPixelFormatR8Snorm:
			case MTLPixelFormatRG8Unorm:
			case MTLPixelFormatRG8Unorm_sRGB:
			case MTLPixelFormatRG8Snorm:
			case MTLPixelFormatRGBA8Unorm:
			case MTLPixelFormatRGBA8Unorm_sRGB:
			case MTLPixelFormatRGBA8Snorm:
			case MTLPixelFormatBGRA8Unorm:
			case MTLPixelFormatBGRA8Unorm_sRGB:
			case MTLPixelFormatR16Unorm:
			case MTLPixelFormatR16Snorm:
			case MTLPixelFormatR16Float:
			case MTLPixelFormatRG16Unorm:
			case MTLPixelFormatRG16Snorm:
			case MTLPixelFormatRG16Float:
			case MTLPixelFormatRGBA16Unorm:
			case MTLPixelFormatRGBA16Snorm:
			case MTLPixelFormatRGBA16Float:
			case MTLPixelFormatR32Float:
			case MTLPixelFormatRGBA32Float:
				return true;
			default:
				return false;
			}
		}

		[[nodiscard]] u32 mip_dimension(u32 value, u32 level)
		{
			return std::max(value >> level, 1u);
		}

		[[nodiscard]] bool range_fits(u32 origin, u32 extent, u32 limit)
		{
			return extent && origin <= limit && extent <= limit - origin;
		}

		[[nodiscard]] bool full_region(const image& resource, const resolve_subresource& region)
		{
			return region.origin_x == 0 && region.origin_y == 0 &&
				region.width == mip_dimension(resource.width(), region.mip_level) &&
				region.height == mip_dimension(resource.height(), region.mip_level);
		}

		[[nodiscard]] subresource_range make_range(const resolve_subresource& region, u8 aspects)
		{
			return {
				.first_mip = region.mip_level,
				.mip_count = 1,
				.first_slice = region.array_slice,
				.slice_count = region.layer_count,
				.color = bool(aspects & texture_aspect_color),
				.depth = bool(aspects & texture_aspect_depth),
				.stencil = bool(aspects & texture_aspect_stencil),
			};
		}

		[[nodiscard]] image_state make_state(const image& resource, u64 stages, u64 access)
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
			if (command.active_encoder() != encoder_kind::none)
				command.end_encoding();
		}

		void publish_encoder_writes(command_buffer& command, u64 producer_stages)
		{
			barrier_plan visibility;
			visibility.scope = barrier_scope::between_encoders;
			visibility.after_stages = producer_stages;
			visibility.before_stages = stage_all_gpu;
			visibility.flush_caches = true;
			visibility.end_encoder = true;
			visibility.producer_barrier = true;
			encode_barrier(command.active_native_encoder(), visibility);
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

		[[nodiscard]] id<MTLComputePipelineState> compile_color_compute_pipeline(id<MTL4Compiler> compiler)
		{
			static constexpr std::string_view source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct resolve_parameters
{
	uint2 multisampled_origin;
	uint2 expanded_origin;
	uint2 output_extent;
	uint2 sample_grid;
	uint copy_stencil;
	uint3 padding;
};
kernel void rsx_color_resolve(constant resolve_parameters& parameters [[buffer(0)]],
	texture2d_ms<float, access::read> multisampled [[texture(0)]],
	texture2d<float, access::write> expanded [[texture(1)]],
	uint2 position [[thread_position_in_grid]])
{
	if (any(position >= parameters.output_extent)) return;
	uint2 multisampled_position = parameters.multisampled_origin + position / parameters.sample_grid;
	uint2 sample_position = position % parameters.sample_grid;
	uint sample = sample_position.x + sample_position.y * parameters.sample_grid.x;
	expanded.write(multisampled.read(multisampled_position, sample),
		parameters.expanded_origin + position);
}
)MSL";
			id<MTLLibrary> library = compile_library(compiler, source, "RPCS3 color resolve");
			MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
			function.library = library;
			function.name = @"rsx_color_resolve";
			MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
			descriptor.label = @"RPCS3 color resolve";
			descriptor.computeFunctionDescriptor = function;
			descriptor.maxTotalThreadsPerThreadgroup = 256;
			NSError* error = nil;
			id<MTLComputePipelineState> pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor
				compilerTaskOptions:nil error:&error];
			if (!pipeline) throw_native_error(error, "color resolve pipeline compilation");
			return pipeline;
		}

		[[nodiscard]] std::string vertex_source()
		{
			return R"MSL(
#include <metal_stdlib>
using namespace metal;
struct vertex_output
{
	float4 position [[position]];
	uint stencil_value [[flat]];
};
#define RPCS3_RESOLVE_VERTEX_OUTPUT_DEFINED 1
vertex vertex_output rsx_resolve_vertex(uint vertex_id [[vertex_id]], uint instance_id [[instance_id]])
{
	float2 position = vertex_id == 0 ? float2(-1.0, -1.0) :
		(vertex_id == 1 ? float2(3.0, -1.0) : float2(-1.0, 3.0));
	return {float4(position, 0.0, 1.0), instance_id};
}
)MSL";
		}

		[[nodiscard]] std::string parameter_declaration()
		{
			return R"MSL(
struct resolve_parameters
{
	uint2 multisampled_origin;
	uint2 expanded_origin;
	uint2 output_extent;
	uint2 sample_grid;
	uint copy_stencil;
	uint3 padding;
};
)MSL";
		}

		[[nodiscard]] std::string color_fragment_source(resolve_direction direction)
		{
			std::string result = "#include <metal_stdlib>\nusing namespace metal;\n" + parameter_declaration() + R"MSL(
#ifndef RPCS3_RESOLVE_VERTEX_OUTPUT_DEFINED
struct vertex_output { float4 position [[position]]; uint stencil_value [[flat]]; };
#endif
)MSL";
			if (direction == resolve_direction::multisample_to_expanded)
			{
				result += R"MSL(
fragment float4 rsx_resolve_fragment(vertex_output input [[stage_in]],
	constant resolve_parameters& parameters [[buffer(0)]],
	texture2d_ms<float, access::read> source [[texture(0)]])
{
	uint2 relative = uint2(input.position.xy) - parameters.expanded_origin;
	uint2 source_position = parameters.multisampled_origin + relative / parameters.sample_grid;
	uint2 sample_position = relative % parameters.sample_grid;
	uint sample = sample_position.x + sample_position.y * parameters.sample_grid.x;
	return source.read(source_position, sample);
}
)MSL";
			}
			else
			{
				result += R"MSL(
fragment float4 rsx_resolve_fragment(vertex_output input [[stage_in]], uint sample [[sample_id]],
	constant resolve_parameters& parameters [[buffer(0)]],
	texture2d<float, access::read> source [[texture(0)]])
{
	uint2 relative = uint2(input.position.xy) - parameters.multisampled_origin;
	uint2 sample_position = uint2(sample % parameters.sample_grid.x, sample / parameters.sample_grid.x);
	return source.read(parameters.expanded_origin + relative * parameters.sample_grid + sample_position);
}
)MSL";
			}
			return result;
		}

		[[nodiscard]] std::string depth_fragment_source(resolve_direction direction)
		{
			std::string result = "#include <metal_stdlib>\nusing namespace metal;\n" + parameter_declaration() + R"MSL(
#ifndef RPCS3_RESOLVE_VERTEX_OUTPUT_DEFINED
struct vertex_output { float4 position [[position]]; uint stencil_value [[flat]]; };
#endif
struct depth_output { float depth [[depth(any)]]; };
)MSL";
			if (direction == resolve_direction::multisample_to_expanded)
			{
				result += R"MSL(
fragment depth_output rsx_resolve_fragment(vertex_output input [[stage_in]],
	constant resolve_parameters& parameters [[buffer(0)]],
	depth2d_ms<float, access::read> source [[texture(0)]])
{
	uint2 relative = uint2(input.position.xy) - parameters.expanded_origin;
	uint2 source_position = parameters.multisampled_origin + relative / parameters.sample_grid;
	uint2 sample_position = relative % parameters.sample_grid;
	uint sample = sample_position.x + sample_position.y * parameters.sample_grid.x;
	return {source.read(source_position, sample)};
}
)MSL";
			}
			else
			{
				result += R"MSL(
fragment depth_output rsx_resolve_fragment(vertex_output input [[stage_in]], uint sample [[sample_id]],
	constant resolve_parameters& parameters [[buffer(0)]],
	depth2d<float, access::read> source [[texture(0)]])
{
	uint2 relative = uint2(input.position.xy) - parameters.multisampled_origin;
	uint2 sample_position = uint2(sample % parameters.sample_grid.x, sample / parameters.sample_grid.x);
	return {source.read(parameters.expanded_origin + relative * parameters.sample_grid + sample_position)};
}
)MSL";
			}
			return result;
		}

		[[nodiscard]] std::string stencil_fragment_source(resolve_direction direction)
		{
			std::string result = "#include <metal_stdlib>\nusing namespace metal;\n" + parameter_declaration() + R"MSL(
#ifndef RPCS3_RESOLVE_VERTEX_OUTPUT_DEFINED
struct vertex_output { float4 position [[position]]; uint stencil_value [[flat]]; };
#endif
)MSL";
			if (direction == resolve_direction::multisample_to_expanded)
			{
				result += R"MSL(
fragment void rsx_resolve_fragment(vertex_output input [[stage_in]],
	constant resolve_parameters& parameters [[buffer(0)]],
	texture2d_ms<uint, access::read> source [[texture(0)]])
{
	if (!parameters.copy_stencil) return;
	uint2 relative = uint2(input.position.xy) - parameters.expanded_origin;
	uint2 source_position = parameters.multisampled_origin + relative / parameters.sample_grid;
	uint2 sample_position = relative % parameters.sample_grid;
	uint sample = sample_position.x + sample_position.y * parameters.sample_grid.x;
	if (source.read(source_position, sample).x != input.stencil_value) discard_fragment();
}
)MSL";
			}
			else
			{
				result += R"MSL(
fragment void rsx_resolve_fragment(vertex_output input [[stage_in]], uint sample [[sample_id]],
	constant resolve_parameters& parameters [[buffer(0)]],
	texture2d<uint, access::read> source [[texture(0)]])
{
	if (!parameters.copy_stencil) return;
	uint2 relative = uint2(input.position.xy) - parameters.multisampled_origin;
	uint2 sample_position = uint2(sample % parameters.sample_grid.x, sample / parameters.sample_grid.x);
	uint value = source.read(parameters.expanded_origin + relative * parameters.sample_grid + sample_position).x;
	if (value != input.stencil_value) discard_fragment();
}
)MSL";
			}
			return result;
		}

		[[nodiscard]] id<MTLRenderPipelineState> compile_render_pipeline(id<MTL4Compiler> compiler,
			const render_pipeline_key& key)
		{
			const std::string vertex = vertex_source();
			std::string fragment;
			switch (key.path)
			{
			case resolve_execution_path::color_render:
				fragment = color_fragment_source(key.direction);
				break;
			case resolve_execution_path::depth_render:
				fragment = depth_fragment_source(key.direction);
				break;
			case resolve_execution_path::stencil_render:
				fragment = stencil_fragment_source(key.direction);
				break;
			case resolve_execution_path::color_compute:
			case resolve_execution_path::depth_stencil_render:
				fmt::throw_exception("Invalid Metal render resolve pipeline path");
			}
			id<MTLLibrary> library = compile_library(compiler, vertex + fragment, "RPCS3 planar sample transfer");
			MTL4LibraryFunctionDescriptor* vertex_function = [MTL4LibraryFunctionDescriptor new];
			vertex_function.library = library;
			vertex_function.name = @"rsx_resolve_vertex";
			MTL4LibraryFunctionDescriptor* fragment_function = [MTL4LibraryFunctionDescriptor new];
			fragment_function.library = library;
			fragment_function.name = @"rsx_resolve_fragment";
			MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
			descriptor.label = @"RPCS3 planar sample transfer";
			descriptor.vertexFunctionDescriptor = vertex_function;
			descriptor.fragmentFunctionDescriptor = fragment_function;
			descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
			descriptor.rasterSampleCount = key.samples;
			if (key.path == resolve_execution_path::color_render)
				descriptor.colorAttachments[0].pixelFormat = static_cast<MTLPixelFormat>(key.format);
			NSError* error = nil;
			id<MTLRenderPipelineState> pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor
				compilerTaskOptions:nil error:&error];
			if (!pipeline) throw_native_error(error, "planar sample pipeline compilation");
			return pipeline;
		}

		[[nodiscard]] id<MTLDepthStencilState> create_depth_state(id<MTLDevice> device)
		{
			MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
			descriptor.depthCompareFunction = MTLCompareFunctionAlways;
			descriptor.depthWriteEnabled = YES;
			id<MTLDepthStencilState> state = [device newDepthStencilStateWithDescriptor:descriptor];
			if (!state) fmt::throw_exception("Metal depth resolve state creation failed");
			return state;
		}

		[[nodiscard]] id<MTLDepthStencilState> create_stencil_state(id<MTLDevice> device)
		{
			MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
			stencil.stencilCompareFunction = MTLCompareFunctionAlways;
			stencil.stencilFailureOperation = MTLStencilOperationKeep;
			stencil.depthFailureOperation = MTLStencilOperationKeep;
			stencil.depthStencilPassOperation = MTLStencilOperationReplace;
			stencil.readMask = 0xff;
			stencil.writeMask = 0xff;
			MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
			descriptor.frontFaceStencil = stencil;
			descriptor.backFaceStencil = stencil;
			id<MTLDepthStencilState> state = [device newDepthStencilStateWithDescriptor:descriptor];
			if (!state) fmt::throw_exception("Metal stencil resolve state creation failed");
			return state;
		}

		[[nodiscard]] id<MTL4ArgumentTable> create_argument_table(id<MTLDevice> device,
			u32 buffers, u32 textures, std::string_view label)
		{
			MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
			descriptor.maxBufferBindCount = buffers;
			descriptor.maxTextureBindCount = textures;
			descriptor.initializeBindings = YES;
			descriptor.label = native_string(label);
			NSError* error = nil;
			id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:descriptor error:&error];
			if (!table) throw_native_error(error, fmt::format("%s argument-table creation", label));
			return table;
		}

		[[nodiscard]] id<MTLBuffer> create_parameter_buffer(id<MTLDevice> device,
			const resolve_parameters& parameters, std::string_view label)
		{
			id<MTLBuffer> result = [device newBufferWithLength:256
				options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
			if (!result) fmt::throw_exception("Metal resolve parameter-buffer allocation failed");
			result.label = native_string(label);
			std::memcpy(result.contents, &parameters, sizeof(parameters));
			return result;
		}

		[[nodiscard]] u64 stencil_view_format(u64 format)
		{
			switch (static_cast<MTLPixelFormat>(format))
			{
			case MTLPixelFormatDepth32Float_Stencil8:
			case MTLPixelFormatX32_Stencil8:
				return MTLPixelFormatX32_Stencil8;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
			case MTLPixelFormatDepth24Unorm_Stencil8:
			case MTLPixelFormatX24_Stencil8:
				return MTLPixelFormatX24_Stencil8;
#pragma clang diagnostic pop
			default:
				fmt::throw_exception("Metal stencil resolve requires a packed depth/stencil format");
			}
		}

		[[nodiscard]] std::unique_ptr<image_view> make_layer_view(image& resource,
			const resolve_subresource& region, u32 layer, u8 aspects)
		{
			const bool multisampled = resource.samples() > 1;
			u64 format = resource.format();
			if (aspects == texture_aspect_stencil)
				format = stencil_view_format(format);
			return std::make_unique<image_view>(resource, format,
				multisampled ? texture_type::texture_2d_multisample : texture_type::texture_2d,
				default_component_map,
				subresource_range{
					.first_mip = region.mip_level,
					.mip_count = 1,
					.first_slice = region.array_slice + layer,
					.slice_count = 1,
					.color = bool(aspects & texture_aspect_color),
					.depth = bool(aspects & texture_aspect_depth),
					.stencil = bool(aspects & texture_aspect_stencil),
				});
		}

		[[nodiscard]] resolve_parameters make_parameters(const resolve_request& request)
		{
			const auto grid = request.sample_grid();
			resolve_parameters result;
			result.multisampled_origin = {request.multisampled_region.origin_x,
				request.multisampled_region.origin_y};
			result.expanded_origin = {request.expanded_region.origin_x,
				request.expanded_region.origin_y};
			result.output_extent = request.direction == resolve_direction::multisample_to_expanded
				? std::array<u32, 2>{request.expanded_region.width, request.expanded_region.height}
				: std::array<u32, 2>{request.multisampled_region.width, request.multisampled_region.height};
			result.sample_grid = {grid.x, grid.y};
			result.copy_stencil = request.stencil_contents_initialized;
			return result;
		}

		void retain_transfer_objects(command_buffer& command, image& source, image& destination,
			id<MTLBuffer> parameters, id<MTL4ArgumentTable> table, id pipeline,
			const image_view& source_view, const image_view* destination_view = nullptr)
		{
			command.retain_native_object((__bridge void*)source.native_handle(), true);
			command.retain_native_object((__bridge void*)destination.native_handle(), true);
			command.retain_native_object((__bridge void*)source_view.native_handle(), true);
			if (destination_view)
				command.retain_native_object((__bridge void*)destination_view->native_handle(), true);
			command.retain_native_object((__bridge void*)parameters, true);
			command.retain_native_object((__bridge void*)table, false);
			command.retain_native_object((__bridge void*)pipeline, false);
		}

		void configure_render_pass_attachment(MTL4RenderPassDescriptor* pass, image& destination,
			const resolve_subresource& region, u32 layer, resolve_execution_path path)
		{
			pass.renderTargetWidth = mip_dimension(destination.width(), region.mip_level);
			pass.renderTargetHeight = mip_dimension(destination.height(), region.mip_level);
			pass.defaultRasterSampleCount = destination.samples();
			const MTLLoadAction primary_load = full_region(destination, region)
				? MTLLoadActionDontCare : MTLLoadActionLoad;
			if (path == resolve_execution_path::color_render)
			{
				auto* attachment = pass.colorAttachments[0];
				attachment.texture = destination.native_handle();
				attachment.level = region.mip_level;
				attachment.slice = region.array_slice + layer;
				attachment.loadAction = primary_load;
				attachment.storeAction = MTLStoreActionStore;
				return;
			}
			if (path == resolve_execution_path::depth_render)
			{
				pass.depthAttachment.texture = destination.native_handle();
				pass.depthAttachment.level = region.mip_level;
				pass.depthAttachment.slice = region.array_slice + layer;
				pass.depthAttachment.loadAction = primary_load;
				pass.depthAttachment.storeAction = MTLStoreActionStore;
				if (destination.aspects() & texture_aspect_stencil)
				{
					pass.stencilAttachment.texture = destination.native_handle();
					pass.stencilAttachment.level = region.mip_level;
					pass.stencilAttachment.slice = region.array_slice + layer;
					pass.stencilAttachment.loadAction = MTLLoadActionLoad;
					pass.stencilAttachment.storeAction = MTLStoreActionStore;
				}
				return;
			}
			pass.stencilAttachment.texture = destination.native_handle();
			pass.stencilAttachment.level = region.mip_level;
			pass.stencilAttachment.slice = region.array_slice + layer;
			pass.stencilAttachment.loadAction = primary_load;
			pass.stencilAttachment.storeAction = MTLStoreActionStore;
			if (destination.aspects() & texture_aspect_depth)
			{
				pass.depthAttachment.texture = destination.native_handle();
				pass.depthAttachment.level = region.mip_level;
				pass.depthAttachment.slice = region.array_slice + layer;
				pass.depthAttachment.loadAction = MTLLoadActionLoad;
				pass.depthAttachment.storeAction = MTLStoreActionStore;
			}
		}
	}

	u32 resolve_sample_grid::count() const
	{
		return static_cast<u32>(x) * y;
	}

	resolve_sample_grid::operator bool() const
	{
		return x == 2 && (y == 1 || y == 2);
	}

	resolve_sample_grid resolve_sample_grid::from_sample_count(u32 samples)
	{
		switch (samples)
		{
		case 2: return {2, 1};
		case 4: return {2, 2};
		default: fmt::throw_exception("Metal planar sample transfer requires two or four samples");
		}
	}

	resolve_subresource::operator bool() const
	{
		return width != 0 && height != 0 && layer_count != 0;
	}

	resolve_sample_grid resolve_request::sample_grid() const
	{
		if (!multisampled) fmt::throw_exception("Metal resolve request has no multisample image");
		return resolve_sample_grid::from_sample_count(multisampled->samples());
	}

	resolve_execution_path resolve_request::execution_path() const
	{
		const u8 requested = effective_aspects(*this);
		if (requested == texture_aspect_color)
		{
			return direction == resolve_direction::multisample_to_expanded && expanded &&
				(expanded->info().usage & texture_usage_shader_write)
				? resolve_execution_path::color_compute : resolve_execution_path::color_render;
		}
		if (requested == texture_aspect_depth) return resolve_execution_path::depth_render;
		if (requested == texture_aspect_stencil) return resolve_execution_path::stencil_render;
		if (requested == (texture_aspect_depth | texture_aspect_stencil))
			return resolve_execution_path::depth_stencil_render;
		fmt::throw_exception("Metal resolve request has an invalid aspect combination");
	}

	void resolve_request::validate() const
	{
		if (!multisampled || !expanded || !*multisampled || !*expanded || multisampled == expanded ||
			!multisampled_region || !expanded_region || multisampled->samples() == 1 || expanded->samples() != 1 ||
			multisampled->format() != expanded->format())
		{
			fmt::throw_exception("Invalid Metal planar sample transfer images");
		}
		const auto grid = sample_grid();
		const u8 requested = effective_aspects(*this);
		if (!requested || (requested & ~(multisampled->aspects() & expanded->aspects())) ||
			(requested & texture_aspect_color && requested != texture_aspect_color))
		{
			fmt::throw_exception("Metal planar sample transfer requests incompatible aspects");
		}
		if (multisampled_region.mip_level >= multisampled->mipmaps() ||
			expanded_region.mip_level >= expanded->mipmaps() ||
			multisampled_region.array_slice > multisampled->layers() ||
			multisampled_region.layer_count > multisampled->layers() - multisampled_region.array_slice ||
			expanded_region.array_slice > expanded->layers() ||
			expanded_region.layer_count > expanded->layers() - expanded_region.array_slice ||
			multisampled_region.layer_count != expanded_region.layer_count)
		{
			fmt::throw_exception("Metal planar sample transfer subresources are out of range");
		}
		const u32 multisampled_width = mip_dimension(multisampled->width(), multisampled_region.mip_level);
		const u32 multisampled_height = mip_dimension(multisampled->height(), multisampled_region.mip_level);
		const u32 expanded_width = mip_dimension(expanded->width(), expanded_region.mip_level);
		const u32 expanded_height = mip_dimension(expanded->height(), expanded_region.mip_level);
		if (!range_fits(multisampled_region.origin_x, multisampled_region.width, multisampled_width) ||
			!range_fits(multisampled_region.origin_y, multisampled_region.height, multisampled_height) ||
			!range_fits(expanded_region.origin_x, expanded_region.width, expanded_width) ||
			!range_fits(expanded_region.origin_y, expanded_region.height, expanded_height) ||
			expanded_region.width != static_cast<u64>(multisampled_region.width) * grid.x ||
			expanded_region.height != static_cast<u64>(multisampled_region.height) * grid.y)
		{
			fmt::throw_exception("Metal planar sample transfer regions do not match the sample grid");
		}
		const image& source = direction == resolve_direction::multisample_to_expanded ? *multisampled : *expanded;
		const image& destination = direction == resolve_direction::multisample_to_expanded ? *expanded : *multisampled;
		if (!(source.info().usage & texture_usage_shader_read))
			fmt::throw_exception("Metal planar sample source was not created for shader reads");
		if (requested == texture_aspect_color)
		{
			if (!supports_float_color_access(multisampled->format()))
				fmt::throw_exception("Metal planar color transfer requires a normalized or floating-point format");
			const u32 needed = execution_path() == resolve_execution_path::color_compute
				? texture_usage_shader_write : texture_usage_render_target;
			if (!(destination.info().usage & needed))
				fmt::throw_exception("Metal planar color destination lacks required write usage");
		}
		else if (!(destination.info().usage & texture_usage_depth_stencil))
		{
			fmt::throw_exception("Metal planar depth/stencil destination lacks attachment usage");
		}
	}

	struct color_resolve_helper::impl
	{
		render_device* device = nullptr;
		MTLPipelineCompiler* compiler = nullptr;
		id<MTLComputePipelineState> compute_pipeline = nil;
		std::unordered_map<render_pipeline_key, id<MTLRenderPipelineState>, render_pipeline_key_hash> render_pipelines;
		resolve_helper_statistics stats;
		mutable std::mutex mutex;

		[[nodiscard]] id<MTLComputePipelineState> get_compute_pipeline()
		{
			std::lock_guard lock(mutex);
			if (compute_pipeline)
			{
				++stats.pipeline_cache_hits;
				return compute_pipeline;
			}
			compute_pipeline = compile_color_compute_pipeline(compiler->native_compiler());
			++stats.native_pipelines;
			return compute_pipeline;
		}

		[[nodiscard]] id<MTLRenderPipelineState> get_render_pipeline(const render_pipeline_key& key)
		{
			std::lock_guard lock(mutex);
			if (const auto found = render_pipelines.find(key); found != render_pipelines.end())
			{
				++stats.pipeline_cache_hits;
				return found->second;
			}
			id<MTLRenderPipelineState> pipeline = compile_render_pipeline(compiler->native_compiler(), key);
			render_pipelines.emplace(key, pipeline);
			++stats.native_pipelines;
			return pipeline;
		}
	};

	color_resolve_helper::color_resolve_helper()
		: m_impl(std::make_unique<impl>())
	{
	}

	color_resolve_helper::~color_resolve_helper()
	{
		destroy();
	}

	void color_resolve_helper::initialize(render_device& device, MTLPipelineCompiler& compiler)
	{
		if (!device || !compiler || &compiler.owner() != &device)
			fmt::throw_exception("Metal color resolve helper requires its device pipeline compiler");
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->device && (m_impl->device != &device || m_impl->compiler != &compiler))
			fmt::throw_exception("Metal color resolve helper cannot change devices while initialized");
		m_impl->device = &device;
		m_impl->compiler = &compiler;
	}

	void color_resolve_helper::run(command_buffer& command, const resolve_request& input)
	{
		resolve_request request = input;
		if (!request.aspects) request.aspects = effective_aspects(request);
		request.validate();
		if (request.aspects != texture_aspect_color)
			fmt::throw_exception("Metal color resolve helper received non-color aspects");
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->device || !m_impl->compiler || &command.allocator().owner() != m_impl->device)
				fmt::throw_exception("Metal color resolve helper is not initialized for this command buffer");
		}

		image& source = request.direction == resolve_direction::multisample_to_expanded
			? *request.multisampled : *request.expanded;
		image& destination = request.direction == resolve_direction::multisample_to_expanded
			? *request.expanded : *request.multisampled;
		const resolve_subresource& source_region = request.direction == resolve_direction::multisample_to_expanded
			? request.multisampled_region : request.expanded_region;
		const resolve_subresource& destination_region = request.direction == resolve_direction::multisample_to_expanded
			? request.expanded_region : request.multisampled_region;
		const resolve_execution_path path = request.execution_path();
		const u64 stages = path == resolve_execution_path::color_compute
			? stage_dispatch : stage_fragment | stage_tile;
		transition_image(command, source, make_state(source, stages, access_shader_read),
			make_range(source_region, texture_aspect_color), false);
		transition_image(command, destination, make_state(destination, stages,
			path == resolve_execution_path::color_compute ? access_shader_write : access_color_write),
			make_range(destination_region, texture_aspect_color), false);
		end_active_encoder(command);

		const resolve_parameters parameters = make_parameters(request);
		id<MTLDevice> device = m_impl->device->native_handle();
		if (path == resolve_execution_path::color_compute)
		{
			id<MTLComputePipelineState> pipeline = m_impl->get_compute_pipeline();
			id<MTL4ComputeCommandEncoder> encoder = (__bridge id<MTL4ComputeCommandEncoder>)command.begin_compute_encoding();
			[encoder setComputePipelineState:pipeline];
			for (u32 layer = 0; layer < request.multisampled_region.layer_count; ++layer)
			{
				auto source_view = make_layer_view(*request.multisampled,
					request.multisampled_region, layer, texture_aspect_color);
				auto destination_view = make_layer_view(*request.expanded,
					request.expanded_region, layer, texture_aspect_color);
				id<MTLBuffer> parameter_buffer = create_parameter_buffer(device, parameters,
					"RPCS3 color resolve parameters");
				id<MTL4ArgumentTable> table = create_argument_table(device, 1, 2,
					"RPCS3 color resolve arguments");
				[table setAddress:parameter_buffer.gpuAddress atIndex:0];
				[table setTexture:source_view->native_handle().gpuResourceID atIndex:0];
				[table setTexture:destination_view->native_handle().gpuResourceID atIndex:1];
				retain_transfer_objects(command, source, destination, parameter_buffer, table,
					pipeline, *source_view, destination_view.get());
				[encoder setArgumentTable:table];
				const NSUInteger group_x = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 16);
				const NSUInteger group_y = std::max<NSUInteger>(1,
					std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup / group_x, 16));
				[encoder dispatchThreads:MTLSizeMake(request.expanded_region.width,
					request.expanded_region.height, 1)
					threadsPerThreadgroup:MTLSizeMake(group_x, group_y, 1)];
			}
			command.set_flag(command_has_dma_transfer);
		}
		else
		{
			const render_pipeline_key key{request.direction, resolve_execution_path::color_render,
				destination.format(), destination.samples()};
			id<MTLRenderPipelineState> pipeline = m_impl->get_render_pipeline(key);
			for (u32 layer = 0; layer < request.multisampled_region.layer_count; ++layer)
			{
				auto source_view = make_layer_view(source, source_region, layer, texture_aspect_color);
				id<MTLBuffer> parameter_buffer = create_parameter_buffer(device, parameters,
					"RPCS3 color transfer parameters");
				id<MTL4ArgumentTable> table = create_argument_table(device, 1, 1,
					"RPCS3 color transfer arguments");
				[table setAddress:parameter_buffer.gpuAddress atIndex:0];
				[table setTexture:source_view->native_handle().gpuResourceID atIndex:0];
				retain_transfer_objects(command, source, destination, parameter_buffer, table,
					pipeline, *source_view);
				MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
				configure_render_pass_attachment(pass, destination, destination_region, layer,
					resolve_execution_path::color_render);
				id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)
					command.begin_render_encoding((__bridge void*)pass);
				[encoder setRenderPipelineState:pipeline];
				[encoder setArgumentTable:table atStages:MTLRenderStageFragment];
				[encoder setViewport:MTLViewport{static_cast<f64>(destination_region.origin_x),
					static_cast<f64>(destination_region.origin_y), static_cast<f64>(destination_region.width),
					static_cast<f64>(destination_region.height), 0.0, 1.0}];
				[encoder setScissorRect:MTLScissorRect{destination_region.origin_x,
					destination_region.origin_y, destination_region.width, destination_region.height}];
				[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
				publish_encoder_writes(command, stage_fragment | stage_tile);
				command.end_encoding();
			}
			command.set_flag(command_reload_dynamic_state);
		}

		std::lock_guard lock(m_impl->mutex);
		if (request.direction == resolve_direction::multisample_to_expanded) ++m_impl->stats.color_resolves;
		else ++m_impl->stats.color_unresolves;
		m_impl->stats.transient_bindings += request.multisampled_region.layer_count;
	}

	void color_resolve_helper::destroy()
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->render_pipelines.clear();
		m_impl->compute_pipeline = nil;
		m_impl->device = nullptr;
		m_impl->compiler = nullptr;
		m_impl->stats = {};
	}

	color_resolve_helper::operator bool() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->device && m_impl->compiler;
	}

	resolve_helper_statistics color_resolve_helper::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->stats;
	}

	struct depth_stencil_resolve_helper::impl
	{
		render_device* device = nullptr;
		MTLPipelineCompiler* compiler = nullptr;
		id<MTLDepthStencilState> depth_state = nil;
		id<MTLDepthStencilState> stencil_state = nil;
		std::unordered_map<render_pipeline_key, id<MTLRenderPipelineState>, render_pipeline_key_hash> pipelines;
		resolve_helper_statistics stats;
		mutable std::mutex mutex;

		[[nodiscard]] id<MTLRenderPipelineState> get_pipeline(const render_pipeline_key& key)
		{
			std::lock_guard lock(mutex);
			if (const auto found = pipelines.find(key); found != pipelines.end())
			{
				++stats.pipeline_cache_hits;
				return found->second;
			}
			id<MTLRenderPipelineState> pipeline = compile_render_pipeline(compiler->native_compiler(), key);
			pipelines.emplace(key, pipeline);
			++stats.native_pipelines;
			return pipeline;
		}
	};

	depth_stencil_resolve_helper::depth_stencil_resolve_helper()
		: m_impl(std::make_unique<impl>())
	{
	}

	depth_stencil_resolve_helper::~depth_stencil_resolve_helper()
	{
		destroy();
	}

	void depth_stencil_resolve_helper::initialize(render_device& device, MTLPipelineCompiler& compiler)
	{
		if (!device || !compiler || &compiler.owner() != &device)
			fmt::throw_exception("Metal depth/stencil resolve helper requires its device pipeline compiler");
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->device && (m_impl->device != &device || m_impl->compiler != &compiler))
			fmt::throw_exception("Metal depth/stencil resolve helper cannot change devices while initialized");
		m_impl->device = &device;
		m_impl->compiler = &compiler;
		if (!m_impl->depth_state)
		{
			m_impl->depth_state = create_depth_state(device.native_handle());
			m_impl->stencil_state = create_stencil_state(device.native_handle());
		}
	}

	void depth_stencil_resolve_helper::run(command_buffer& command, const resolve_request& input)
	{
		resolve_request request = input;
		if (!request.aspects) request.aspects = effective_aspects(request);
		request.validate();
		if (request.aspects & texture_aspect_color)
			fmt::throw_exception("Metal depth/stencil resolve helper received color aspects");
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->device || !m_impl->compiler || &command.allocator().owner() != m_impl->device)
				fmt::throw_exception("Metal depth/stencil resolve helper is not initialized for this command buffer");
		}

		image& source = request.direction == resolve_direction::multisample_to_expanded
			? *request.multisampled : *request.expanded;
		image& destination = request.direction == resolve_direction::multisample_to_expanded
			? *request.expanded : *request.multisampled;
		const resolve_subresource& source_region = request.direction == resolve_direction::multisample_to_expanded
			? request.multisampled_region : request.expanded_region;
		const resolve_subresource& destination_region = request.direction == resolve_direction::multisample_to_expanded
			? request.expanded_region : request.multisampled_region;
		transition_image(command, source, make_state(source, stage_fragment, access_shader_read),
			make_range(source_region, request.aspects), false);
		transition_image(command, destination,
			make_state(destination, stage_fragment | stage_tile, access_depth_stencil_write),
			make_range(destination_region, request.aspects), false);
		end_active_encoder(command);

		const resolve_parameters parameters = make_parameters(request);
		id<MTLDevice> device = m_impl->device->native_handle();
		auto run_aspect = [&](u8 aspect, resolve_execution_path path)
		{
			const render_pipeline_key key{request.direction, path, destination.format(), destination.samples()};
			id<MTLRenderPipelineState> pipeline = m_impl->get_pipeline(key);
			for (u32 layer = 0; layer < request.multisampled_region.layer_count; ++layer)
			{
				auto source_view = make_layer_view(source, source_region, layer, aspect);
				id<MTLBuffer> parameter_buffer = create_parameter_buffer(device, parameters,
					path == resolve_execution_path::depth_render
						? "RPCS3 depth transfer parameters" : "RPCS3 stencil transfer parameters");
				id<MTL4ArgumentTable> table = create_argument_table(device, 1, 1,
					path == resolve_execution_path::depth_render
						? "RPCS3 depth transfer arguments" : "RPCS3 stencil transfer arguments");
				[table setAddress:parameter_buffer.gpuAddress atIndex:0];
				[table setTexture:source_view->native_handle().gpuResourceID atIndex:0];
				retain_transfer_objects(command, source, destination, parameter_buffer, table,
					pipeline, *source_view);
				MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
				configure_render_pass_attachment(pass, destination, destination_region, layer, path);
				id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)
					command.begin_render_encoding((__bridge void*)pass);
				[encoder setRenderPipelineState:pipeline];
				[encoder setDepthStencilState:path == resolve_execution_path::depth_render
					? m_impl->depth_state : m_impl->stencil_state];
				[encoder setArgumentTable:table atStages:MTLRenderStageFragment];
				[encoder setViewport:MTLViewport{static_cast<f64>(destination_region.origin_x),
					static_cast<f64>(destination_region.origin_y), static_cast<f64>(destination_region.width),
					static_cast<f64>(destination_region.height), 0.0, 1.0}];
				[encoder setScissorRect:MTLScissorRect{destination_region.origin_x,
					destination_region.origin_y, destination_region.width, destination_region.height}];
				if (path == resolve_execution_path::depth_render)
				{
					[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
				}
				else if (!request.stencil_contents_initialized)
				{
					[encoder setStencilReferenceValue:request.stencil_initial_value];
					[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3
						instanceCount:1 baseInstance:request.stencil_initial_value];
				}
				else
				{
					for (u32 value = 0; value < 256; ++value)
					{
						[encoder setStencilReferenceValue:value];
						[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3
							instanceCount:1 baseInstance:value];
					}
				}
				publish_encoder_writes(command, stage_fragment | stage_tile);
				command.end_encoding();
			}
		};

		if (request.aspects & texture_aspect_depth)
			run_aspect(texture_aspect_depth, resolve_execution_path::depth_render);
		if (request.aspects & texture_aspect_stencil)
			run_aspect(texture_aspect_stencil, resolve_execution_path::stencil_render);
		command.set_flag(command_reload_dynamic_state);

		std::lock_guard lock(m_impl->mutex);
		const u64 layers = request.multisampled_region.layer_count;
		if (request.direction == resolve_direction::multisample_to_expanded)
		{
			if (request.aspects & texture_aspect_depth) ++m_impl->stats.depth_resolves;
			if (request.aspects & texture_aspect_stencil) ++m_impl->stats.stencil_resolves;
		}
		else
		{
			if (request.aspects & texture_aspect_depth) ++m_impl->stats.depth_unresolves;
			if (request.aspects & texture_aspect_stencil) ++m_impl->stats.stencil_unresolves;
		}
		if (request.aspects == (texture_aspect_depth | texture_aspect_stencil))
			++m_impl->stats.combined_depth_stencil_passes;
		m_impl->stats.transient_bindings += layers *
			((request.aspects & texture_aspect_depth ? 1 : 0) +
			 (request.aspects & texture_aspect_stencil ? 1 : 0));
	}

	void depth_stencil_resolve_helper::destroy()
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->pipelines.clear();
		m_impl->depth_state = nil;
		m_impl->stencil_state = nil;
		m_impl->device = nullptr;
		m_impl->compiler = nullptr;
		m_impl->stats = {};
	}

	depth_stencil_resolve_helper::operator bool() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->device && m_impl->compiler && m_impl->depth_state && m_impl->stencil_state;
	}

	resolve_helper_statistics depth_stencil_resolve_helper::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->stats;
	}

	struct resolve_helper::impl
	{
		color_resolve_helper color;
		depth_stencil_resolve_helper depth_stencil;
	};

	resolve_helper::resolve_helper()
		: m_impl(std::make_unique<impl>())
	{
	}

	resolve_helper::~resolve_helper()
	{
		destroy();
	}

	void resolve_helper::initialize(render_device& device, MTLPipelineCompiler& compiler)
	{
		m_impl->color.initialize(device, compiler);
		m_impl->depth_stencil.initialize(device, compiler);
	}

	void resolve_helper::resolve(command_buffer& command, image& destination, image& source, u8 aspects)
	{
		const auto grid = resolve_sample_grid::from_sample_count(source.samples());
		resolve_request request;
		request.multisampled = &source;
		request.expanded = &destination;
		request.direction = resolve_direction::multisample_to_expanded;
		request.multisampled_region = {0, 0, 0, 0, source.width(), source.height(), source.layers()};
		request.expanded_region = {0, 0, 0, 0, source.width() * grid.x,
			source.height() * grid.y, destination.layers()};
		request.aspects = aspects;
		run(command, request);
	}

	void resolve_helper::unresolve(command_buffer& command, image& destination, image& source, u8 aspects)
	{
		const auto grid = resolve_sample_grid::from_sample_count(destination.samples());
		resolve_request request;
		request.multisampled = &destination;
		request.expanded = &source;
		request.direction = resolve_direction::expanded_to_multisample;
		request.multisampled_region = {0, 0, 0, 0, destination.width(), destination.height(), destination.layers()};
		request.expanded_region = {0, 0, 0, 0, destination.width() * grid.x,
			destination.height() * grid.y, source.layers()};
		request.aspects = aspects;
		run(command, request);
	}

	void resolve_helper::run(command_buffer& command, const resolve_request& request)
	{
		const u8 aspects = effective_aspects(request);
		if (aspects == texture_aspect_color) m_impl->color.run(command, request);
		else m_impl->depth_stencil.run(command, request);
	}

	void resolve_helper::destroy()
	{
		if (!m_impl) return;
		m_impl->color.destroy();
		m_impl->depth_stencil.destroy();
	}

	resolve_helper::operator bool() const
	{
		return static_cast<bool>(m_impl->color) && static_cast<bool>(m_impl->depth_stencil);
	}

	resolve_helper_statistics resolve_helper::statistics() const
	{
		const auto color = m_impl->color.statistics();
		const auto depth = m_impl->depth_stencil.statistics();
		resolve_helper_statistics result;
#define MTL_SUM_RESOLVE_STAT(name) result.name = color.name + depth.name
		MTL_SUM_RESOLVE_STAT(color_resolves);
		MTL_SUM_RESOLVE_STAT(color_unresolves);
		MTL_SUM_RESOLVE_STAT(depth_resolves);
		MTL_SUM_RESOLVE_STAT(depth_unresolves);
		MTL_SUM_RESOLVE_STAT(stencil_resolves);
		MTL_SUM_RESOLVE_STAT(stencil_unresolves);
		MTL_SUM_RESOLVE_STAT(combined_depth_stencil_passes);
		MTL_SUM_RESOLVE_STAT(native_pipelines);
		MTL_SUM_RESOLVE_STAT(pipeline_cache_hits);
		MTL_SUM_RESOLVE_STAT(transient_bindings);
#undef MTL_SUM_RESOLVE_STAT
		return result;
	}

	resolve_helper& get_resolve_helper()
	{
		static resolve_helper helper;
		return helper;
	}

	void initialize_resolve_helpers(render_device& device, MTLPipelineCompiler& compiler)
	{
		get_resolve_helper().initialize(device, compiler);
	}

	void resolve_image(command_buffer& command, image& destination, image& source, u8 aspects)
	{
		get_resolve_helper().resolve(command, destination, source, aspects);
	}

	void unresolve_image(command_buffer& command, image& destination, image& source, u8 aspects)
	{
		get_resolve_helper().unresolve(command, destination, source, aspects);
	}

	void clear_resolve_helpers()
	{
		get_resolve_helper().destroy();
	}
}
