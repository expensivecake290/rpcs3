#include "stdafx.h"
#include "image_helpers.h"

#include "shared.h"
#include "../../color_utils.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace mtl
{
	const component_mapping default_component_map{};

	namespace
	{
		struct format_layout
		{
			u32 bytes = 0;
			u32 block_width = 1;
			u32 block_height = 1;
			bool depth = false;
			bool stencil = false;
		};

		format_layout get_format_layout(MTLPixelFormat format)
		{
			switch (format)
			{
			case MTLPixelFormatA8Unorm:
			case MTLPixelFormatR8Unorm:
			case MTLPixelFormatR8Unorm_sRGB:
			case MTLPixelFormatR8Snorm:
			case MTLPixelFormatR8Uint:
			case MTLPixelFormatR8Sint:
			case MTLPixelFormatStencil8:
				return {1, 1, 1, false, format == MTLPixelFormatStencil8};

			case MTLPixelFormatR16Unorm:
			case MTLPixelFormatR16Snorm:
			case MTLPixelFormatR16Uint:
			case MTLPixelFormatR16Sint:
			case MTLPixelFormatR16Float:
			case MTLPixelFormatRG8Unorm:
			case MTLPixelFormatRG8Unorm_sRGB:
			case MTLPixelFormatRG8Snorm:
			case MTLPixelFormatRG8Uint:
			case MTLPixelFormatRG8Sint:
			case MTLPixelFormatB5G6R5Unorm:
			case MTLPixelFormatA1BGR5Unorm:
			case MTLPixelFormatABGR4Unorm:
			case MTLPixelFormatBGR5A1Unorm:
				return {2};
			case MTLPixelFormatDepth16Unorm:
				return {2, 1, 1, true, false};

			case MTLPixelFormatR32Uint:
			case MTLPixelFormatR32Sint:
			case MTLPixelFormatR32Float:
			case MTLPixelFormatRG16Unorm:
			case MTLPixelFormatRG16Snorm:
			case MTLPixelFormatRG16Uint:
			case MTLPixelFormatRG16Sint:
			case MTLPixelFormatRG16Float:
			case MTLPixelFormatRGBA8Unorm:
			case MTLPixelFormatRGBA8Unorm_sRGB:
			case MTLPixelFormatRGBA8Snorm:
			case MTLPixelFormatRGBA8Uint:
			case MTLPixelFormatRGBA8Sint:
			case MTLPixelFormatBGRA8Unorm:
			case MTLPixelFormatBGRA8Unorm_sRGB:
			case MTLPixelFormatRGB10A2Unorm:
			case MTLPixelFormatRGB10A2Uint:
			case MTLPixelFormatRG11B10Float:
			case MTLPixelFormatRGB9E5Float:
			case MTLPixelFormatBGR10A2Unorm:
			case MTLPixelFormatBGR10_XR:
			case MTLPixelFormatBGR10_XR_sRGB:
			case MTLPixelFormatGBGR422:
			case MTLPixelFormatBGRG422:
				return {4};
			case MTLPixelFormatDepth32Float:
				return {4, 1, 1, true, false};

			case MTLPixelFormatRG32Uint:
			case MTLPixelFormatRG32Sint:
			case MTLPixelFormatRG32Float:
			case MTLPixelFormatRGBA16Unorm:
			case MTLPixelFormatRGBA16Snorm:
			case MTLPixelFormatRGBA16Uint:
			case MTLPixelFormatRGBA16Sint:
			case MTLPixelFormatRGBA16Float:
			case MTLPixelFormatBGRA10_XR:
			case MTLPixelFormatBGRA10_XR_sRGB:
				return {8};
			case MTLPixelFormatDepth32Float_Stencil8:
				return {8, 1, 1, true, true};
			case MTLPixelFormatX32_Stencil8:
				return {8, 1, 1, false, true};
			case static_cast<MTLPixelFormat>(255):
				return {4, 1, 1, true, true};
			case static_cast<MTLPixelFormat>(262):
				return {4, 1, 1, false, true};

			case MTLPixelFormatRGBA32Uint:
			case MTLPixelFormatRGBA32Sint:
			case MTLPixelFormatRGBA32Float:
				return {16};

			case MTLPixelFormatBC1_RGBA:
			case MTLPixelFormatBC1_RGBA_sRGB:
			case MTLPixelFormatBC4_RUnorm:
			case MTLPixelFormatBC4_RSnorm:
				return {8, 4, 4};
			case MTLPixelFormatBC2_RGBA:
			case MTLPixelFormatBC2_RGBA_sRGB:
			case MTLPixelFormatBC3_RGBA:
			case MTLPixelFormatBC3_RGBA_sRGB:
			case MTLPixelFormatBC5_RGUnorm:
			case MTLPixelFormatBC5_RGSnorm:
			case MTLPixelFormatBC6H_RGBFloat:
			case MTLPixelFormatBC6H_RGBUfloat:
			case MTLPixelFormatBC7_RGBAUnorm:
			case MTLPixelFormatBC7_RGBAUnorm_sRGB:
				return {16, 4, 4};

			default:
				fmt::throw_exception("Unsupported Metal pixel format %llu in image transfer", static_cast<u64>(format));
			}
		}

		u32 mip_dimension(u32 base, u32 level)
		{
			return std::max(1u, base >> std::min(level, 31u));
		}

		bool range_fits(u32 first, u32 count, u32 total)
		{
			return count != 0 && first < total && count <= total - first;
		}

		bool volume_type(texture_type type)
		{
			return type == texture_type::texture_3d;
		}

		void validate_aspects(const image& resource, u8 requested)
		{
			if (requested == texture_aspect_none || (requested & ~resource.aspects()) != 0)
			{
				fmt::throw_exception("Metal image operation requests unavailable texture aspects");
			}
		}

		void validate_texture_region(const image& resource, const texture_subresource& subresource,
			const texture_origin& origin, const texture_extent& extent, u32 layer_count)
		{
			validate_aspects(resource, subresource.aspects);
			if (subresource.mip_level >= resource.mipmaps() || extent.width == 0 || extent.height == 0 || extent.depth == 0)
			{
				fmt::throw_exception("Invalid Metal texture subresource or empty transfer extent");
			}

			const u32 width = mip_dimension(resource.width(), subresource.mip_level);
			const u32 height = mip_dimension(resource.height(), subresource.mip_level);
			const u32 depth = mip_dimension(resource.depth(), subresource.mip_level);
			if (!range_fits(origin.x, extent.width, width) || !range_fits(origin.y, extent.height, height) ||
				!range_fits(origin.z, extent.depth, depth))
			{
				fmt::throw_exception("Metal texture transfer region exceeds its mip dimensions");
			}
			const format_layout layout = get_format_layout(static_cast<MTLPixelFormat>(resource.format()));
			if ((origin.x % layout.block_width) != 0 || (origin.y % layout.block_height) != 0 ||
				((extent.width % layout.block_width) != 0 && origin.x + extent.width != width) ||
				((extent.height % layout.block_height) != 0 && origin.y + extent.height != height))
			{
				fmt::throw_exception("Compressed Metal transfer regions must align to blocks or mip edges");
			}

			if (volume_type(resource.type()))
			{
				if (subresource.array_slice != 0 || layer_count != 1)
				{
					fmt::throw_exception("Metal 3D texture transfers do not use array slices");
				}
			}
			else if (origin.z != 0 || extent.depth != 1 || !range_fits(subresource.array_slice, layer_count, resource.layers()))
			{
				fmt::throw_exception("Invalid Metal array/cube texture transfer slices");
			}
		}

		MTLOrigin native_origin(const texture_origin& value)
		{
			return MTLOriginMake(value.x, value.y, value.z);
		}

		MTLSize native_size(const texture_extent& value)
		{
			return MTLSizeMake(value.width, value.height, value.depth);
		}

		id<MTL4ComputeCommandEncoder> get_compute_encoder(command_buffer& command)
		{
			if (!command.is_recording())
			{
				fmt::throw_exception("Metal image operations require an active command recording");
			}
			if (command.active_encoder() == encoder_kind::render)
			{
				command.end_encoding();
			}
			if (command.active_encoder() == encoder_kind::none)
			{
				return (__bridge id<MTL4ComputeCommandEncoder>)command.begin_compute_encoding();
			}
			return (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder();
		}

		MTLBlitOption transfer_option(MTLPixelFormat format, u8 aspects)
		{
			const format_layout layout = get_format_layout(format);
			if (!(layout.depth && layout.stencil))
			{
				return MTLBlitOptionNone;
			}
			if (aspects == texture_aspect_depth)
			{
				return MTLBlitOptionDepthFromDepthStencil;
			}
			if (aspects == texture_aspect_stencil)
			{
				return MTLBlitOptionStencilFromDepthStencil;
			}
			fmt::throw_exception("Combined Metal depth/stencil buffer transfers must select exactly one plane");
		}

		struct resolved_buffer_region
		{
			u64 bytes_per_row = 0;
			u64 bytes_per_image = 0;
			u64 layer_stride = 0;
			u64 required_end = 0;
		};

		resolved_buffer_region resolve_buffer_region(const buffer_image_copy_region& region,
			MTLPixelFormat format, u64 buffer_size)
		{
			format_layout layout = get_format_layout(format);
			if (layout.depth && layout.stencil)
			{
				if (region.subresource.aspects == texture_aspect_depth)
				{
					layout = {4, 1, 1, true, false};
				}
				else if (region.subresource.aspects == texture_aspect_stencil)
				{
					layout = {1, 1, 1, false, true};
				}
			}
			auto checked_multiply = [](u64 left, u64 right)
			{
				if (right && left > std::numeric_limits<u64>::max() / right)
				{
					fmt::throw_exception("Metal buffer/image transfer size overflows");
				}
				return left * right;
			};
			auto checked_add = [](u64 left, u64 right)
			{
				if (left > std::numeric_limits<u64>::max() - right)
				{
					fmt::throw_exception("Metal buffer/image transfer size overflows");
				}
				return left + right;
			};
			const u64 blocks_wide = (region.extent.width + layout.block_width - 1) / layout.block_width;
			const u64 blocks_high = (region.extent.height + layout.block_height - 1) / layout.block_height;
			const u64 tight_row = checked_multiply(blocks_wide, layout.bytes);
			const u64 row_pitch = region.bytes_per_row ? region.bytes_per_row : tight_row;
			const u64 tight_image = checked_multiply(row_pitch, blocks_high);
			const u64 image_pitch = region.bytes_per_image ? region.bytes_per_image : tight_image;
			if (row_pitch < tight_row || row_pitch % layout.bytes != 0 || image_pitch < tight_image ||
				image_pitch % layout.bytes != 0 || region.buffer_offset % layout.bytes != 0)
			{
				fmt::throw_exception("Invalid Metal buffer/image row or image pitch");
			}
			if ((region.origin.x % layout.block_width) != 0 || (region.origin.y % layout.block_height) != 0)
			{
				fmt::throw_exception("Compressed Metal transfer origins must be block aligned");
			}

			const u64 layer_stride = checked_multiply(image_pitch, region.extent.depth);
			u64 required_end = region.buffer_offset;
			required_end = checked_add(required_end, checked_multiply(layer_stride, region.layer_count - 1));
			required_end = checked_add(required_end, checked_multiply(image_pitch, region.extent.depth - 1));
			required_end = checked_add(required_end, checked_multiply(row_pitch, blocks_high - 1));
			required_end = checked_add(required_end, tight_row);
			if (required_end > buffer_size)
			{
				fmt::throw_exception("Metal buffer/image transfer exceeds the buffer range");
			}
			return {row_pitch, region.extent.depth > 1 ? image_pitch : 0, layer_stride, required_end};
		}

		subresource_range make_range(const texture_subresource& resource, u32 layers)
		{
			return
			{
				.first_mip = resource.mip_level,
				.mip_count = 1,
				.first_slice = resource.array_slice,
				.slice_count = layers,
				.color = !!(resource.aspects & texture_aspect_color),
				.depth = !!(resource.aspects & texture_aspect_depth),
				.stencil = !!(resource.aspects & texture_aspect_stencil),
			};
		}

		image_state operation_state(const image& resource, u64 stages, u64 access)
		{
			const image_state previous = resource.state();
			return
			{
				.queue = previous.initialized ? previous.queue : queue_kind::graphics,
				.stages = stages,
				.access = access,
				.submission = get_submission_id(),
				.initialized = true,
			};
		}

		struct alignas(16) scale_parameters
		{
			float source_box[4];
			float destination_box[4];
			float source_size[2];
			float source_depth = 1.f;
			u32 source_level = 0;
			u32 channels[4];
			u32 flags = 0;
			u32 tail_padding[3]{};
		};

		struct image_operation_cache
		{
			id<MTLDevice> device;
			id<MTL4Compiler> compiler;
			id<MTLLibrary> library;
			id<MTLSamplerState> nearest_sampler;
			id<MTLSamplerState> linear_sampler;
			id<MTLDepthStencilState> depth_write_state;
			NSMutableDictionary<NSNumber*, id<MTLRenderPipelineState>>* pipelines;
			std::mutex mutex;
		};

		std::mutex g_operation_caches_mutex;
		std::unordered_map<void*, std::unique_ptr<image_operation_cache>> g_operation_caches;

		[[noreturn]] void throw_native_error(NSError* native_error, const char* operation)
		{
			fmt::throw_exception("%s failed: %s", operation,
				native_error.localizedDescription.UTF8String ?: "Metal returned no diagnostic");
		}

		std::unique_ptr<image_operation_cache> create_operation_cache(render_device& render)
		{
			auto result = std::make_unique<image_operation_cache>();
			result->device = render.native_handle();
			result->compiler = render.compiler();
			result->pipelines = [NSMutableDictionary dictionary];

			static constexpr const char* shader_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct scale_parameters
{
	float4 source_box;
	float4 destination_box;
	float2 source_size;
	float source_depth;
	uint source_level;
	uint4 channels;
	uint flags;
	uint3 tail_padding;
};

vertex float4 image_helper_vertex(uint vertex_id [[vertex_id]])
{
	const float2 positions[3] = { float2(-1.0, 1.0), float2(3.0, 1.0), float2(-1.0, -3.0) };
	return float4(positions[vertex_id], 0.0, 1.0);
}

float3 srgb_decode(float3 value)
{
	return select(value / 12.92, pow((value + 0.055) / 1.055, float3(2.4)), value > 0.04045);
}

float3 srgb_encode(float3 value)
{
	value = max(value, 0.0);
	return select(value * 12.92, 1.055 * pow(value, float3(1.0 / 2.4)) - 0.055, value > 0.0031308);
}

float select_component(float4 value, uint selector)
{
	switch (selector)
	{
	case 0: return 0.0;
	case 1: return 1.0;
	case 2: return value.r;
	case 3: return value.g;
	case 4: return value.b;
	default: return value.a;
	}
}

float4 transform_color(float4 value, constant scale_parameters& parameters)
{
	if (parameters.flags & 1) value.rgb = srgb_decode(value.rgb);
	value = float4(select_component(value, parameters.channels.x),
		select_component(value, parameters.channels.y),
		select_component(value, parameters.channels.z),
		select_component(value, parameters.channels.w));
	if (parameters.flags & 8) value.rgb = value.a > 0.0 ? value.rgb / value.a : float3(0.0);
	if (parameters.flags & 4) value.rgb *= value.a;
	if (parameters.flags & 2) value.rgb = srgb_encode(value.rgb);
	return value;
}

float2 image_helper_coordinates(float4 position, constant scale_parameters& parameters)
{
	const float2 ratio = (position.xy - parameters.destination_box.xy) / parameters.destination_box.zw;
	return mix(parameters.source_box.xy, parameters.source_box.zw, ratio) / parameters.source_size;
}

float image_helper_coordinate_1d(float4 position, constant scale_parameters& parameters)
{
	const float ratio = (position.x - parameters.destination_box.x) / parameters.destination_box.z;
	return mix(parameters.source_box.x, parameters.source_box.z, ratio) / parameters.source_size.x;
}

fragment float4 image_helper_color_color(float4 position [[position]],
	texture2d<float> source [[texture(0)]], sampler source_sampler [[sampler(0)]],
	constant scale_parameters& parameters [[buffer(0)]])
{
	return transform_color(source.sample(source_sampler, image_helper_coordinates(position, parameters)), parameters);
}

fragment float4 image_helper_depth_color(float4 position [[position]],
	depth2d<float> source [[texture(0)]], sampler source_sampler [[sampler(0)]],
	constant scale_parameters& parameters [[buffer(0)]])
{
	const float depth = source.sample(source_sampler, image_helper_coordinates(position, parameters));
	return transform_color(float4(depth, depth, depth, 1.0), parameters);
}

fragment float4 image_helper_color_color_1d(float4 position [[position]],
	texture1d<float> source [[texture(0)]], sampler source_sampler [[sampler(0)]],
	constant scale_parameters& parameters [[buffer(0)]])
{
	return transform_color(source.sample(source_sampler, image_helper_coordinate_1d(position, parameters)), parameters);
}

fragment float4 image_helper_color_color_3d(float4 position [[position]],
	texture3d<float> source [[texture(0)]], sampler source_sampler [[sampler(0)]],
	constant scale_parameters& parameters [[buffer(0)]])
{
	const float2 coordinates = image_helper_coordinates(position, parameters);
	const float3 volume_coordinates(coordinates, parameters.source_depth);
	return transform_color(source.sample(source_sampler, volume_coordinates, level(parameters.source_level)), parameters);
}

struct depth_output
{
	float depth [[depth(any)]];
};

fragment depth_output image_helper_color_depth(float4 position [[position]],
	texture2d<float> source [[texture(0)]], sampler source_sampler [[sampler(0)]],
	constant scale_parameters& parameters [[buffer(0)]])
{
	return { transform_color(source.sample(source_sampler, image_helper_coordinates(position, parameters)), parameters).r };
}

fragment depth_output image_helper_depth_depth(float4 position [[position]],
	depth2d<float> source [[texture(0)]], sampler source_sampler [[sampler(0)]],
	constant scale_parameters& parameters [[buffer(0)]])
{
	const float depth = source.sample(source_sampler, image_helper_coordinates(position, parameters));
	return { transform_color(float4(depth, depth, depth, 1.0), parameters).r };
}
)MSL";

			MTLCompileOptions* options = [MTLCompileOptions new];
			options.languageVersion = MTLLanguageVersion4_0;
			MTL4LibraryDescriptor* library_descriptor = [MTL4LibraryDescriptor new];
			library_descriptor.name = @"RPCS3 Metal image operations";
			library_descriptor.source = [NSString stringWithUTF8String:shader_source];
			library_descriptor.options = options;
			NSError* native_error = nil;
			result->library = [result->compiler newLibraryWithDescriptor:library_descriptor error:&native_error];
			if (!result->library)
			{
				throw_native_error(native_error, "Metal image-helper library compilation");
			}

			for (const auto [filter, destination] :
				{std::pair{MTLSamplerMinMagFilterNearest, &result->nearest_sampler},
				 std::pair{MTLSamplerMinMagFilterLinear, &result->linear_sampler}})
			{
				MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
				descriptor.minFilter = filter;
				descriptor.magFilter = filter;
				descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped;
				descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
				descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
				descriptor.rAddressMode = MTLSamplerAddressModeClampToEdge;
				descriptor.normalizedCoordinates = YES;
				*destination = [result->device newSamplerStateWithDescriptor:descriptor];
				if (!*destination)
				{
					fmt::throw_exception("Metal failed to create an image-helper sampler");
				}
			}

			MTLDepthStencilDescriptor* depth_descriptor = [MTLDepthStencilDescriptor new];
			depth_descriptor.depthCompareFunction = MTLCompareFunctionAlways;
			depth_descriptor.depthWriteEnabled = YES;
			result->depth_write_state = [result->device newDepthStencilStateWithDescriptor:depth_descriptor];
			if (!result->depth_write_state)
			{
				fmt::throw_exception("Metal failed to create image-helper depth state");
			}
			return result;
		}

		image_operation_cache& get_operation_cache(render_device& render)
		{
			const auto key = (__bridge void*)render.native_handle();
			std::lock_guard lock(g_operation_caches_mutex);
			auto& cache = g_operation_caches[key];
			if (!cache)
			{
				cache = create_operation_cache(render);
			}
			return *cache;
		}

		id<MTLRenderPipelineState> get_scale_pipeline(image_operation_cache& cache, bool source_depth,
			bool destination_depth, texture_type source_type, MTLPixelFormat destination_format)
		{
			const u32 index = source_type == texture_type::texture_3d ? 5u :
				(source_type == texture_type::texture_1d || source_type == texture_type::texture_1d_array ? 4u :
				 (source_depth ? 1u : 0u) | (destination_depth ? 2u : 0u));
			NSNumber* key = @(static_cast<u64>(destination_format) << 3 | index);
			std::lock_guard lock(cache.mutex);
			if (id<MTLRenderPipelineState> existing = cache.pipelines[key])
			{
				return existing;
			}

			static constexpr std::array<const char*, 6> fragment_names =
			{
				"image_helper_color_color",
				"image_helper_depth_color",
				"image_helper_color_depth",
				"image_helper_depth_depth",
				"image_helper_color_color_1d",
				"image_helper_color_color_3d",
			};
			MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new];
			vertex.library = cache.library;
			vertex.name = @"image_helper_vertex";
			MTL4LibraryFunctionDescriptor* fragment = [MTL4LibraryFunctionDescriptor new];
			fragment.library = cache.library;
			fragment.name = [NSString stringWithUTF8String:fragment_names[index]];
			MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
			descriptor.label = [NSString stringWithFormat:@"RPCS3 image scale pipeline %u", index];
			descriptor.vertexFunctionDescriptor = vertex;
			descriptor.fragmentFunctionDescriptor = fragment;
			descriptor.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
			descriptor.rasterSampleCount = 1;
			if (!destination_depth)
			{
				descriptor.colorAttachments[0].pixelFormat = destination_format;
			}
			NSError* native_error = nil;
			id<MTLRenderPipelineState> pipeline = [cache.compiler newRenderPipelineStateWithDescriptor:descriptor
				compilerTaskOptions:nil error:&native_error];
			if (!pipeline)
			{
				throw_native_error(native_error, "Metal image-helper pipeline compilation");
			}
			cache.pipelines[key] = pipeline;
			return pipeline;
		}

		u32 absolute_difference(s32 first, s32 second)
		{
			const s64 difference = static_cast<s64>(second) - first;
			const u64 magnitude = difference < 0 ? static_cast<u64>(-difference) : static_cast<u64>(difference);
			if (magnitude == 0 || magnitude > std::numeric_limits<u32>::max())
			{
				fmt::throw_exception("Invalid Metal scale box extent");
			}
			return static_cast<u32>(magnitude);
		}

		void validate_axis(s32 first, s32 second, u32 limit)
		{
			if (first < 0 || second < 0 || first == second || static_cast<u64>(first) > limit || static_cast<u64>(second) > limit)
			{
				fmt::throw_exception("Metal scale box exceeds its mip dimensions");
			}
		}

		bool is_2d_family(texture_type type)
		{
			return type == texture_type::texture_2d || type == texture_type::texture_2d_array ||
				type == texture_type::texture_cube || type == texture_type::texture_cube_array;
		}

		bool is_1d_family(texture_type type)
		{
			return type == texture_type::texture_1d || type == texture_type::texture_1d_array;
		}

		void validate_scale(const image& source, const image& destination, const image_scale_region& region,
			const image_conversion& conversion)
		{
			if (!source || !destination)
			{
				fmt::throw_exception("Invalid Metal scale resources");
			}
			const bool one_dimensional = is_1d_family(source.type()) && is_1d_family(destination.type());
			const bool two_dimensional = is_2d_family(source.type()) && is_2d_family(destination.type());
			const bool three_dimensional = source.type() == texture_type::texture_3d && destination.type() == texture_type::texture_3d;
			if (source.samples() != 1 || destination.samples() != 1 ||
				(!one_dimensional && !two_dimensional && !three_dimensional) || region.layer_count == 0 ||
				region.source.mip_level >= source.mipmaps() || region.destination.mip_level >= destination.mipmaps() ||
				!(source.info().usage & texture_usage_shader_read))
			{
				fmt::throw_exception("Invalid Metal scale resources");
			}

			validate_aspects(source, region.source.aspects);
			validate_aspects(destination, region.destination.aspects);
			const bool source_depth = region.source.aspects == texture_aspect_depth;
			const bool destination_depth = region.destination.aspects == texture_aspect_depth;
			if ((!source_depth && region.source.aspects != texture_aspect_color) ||
				(!destination_depth && region.destination.aspects != texture_aspect_color) ||
				(destination_depth ? !(destination.info().usage & texture_usage_depth_stencil)
					: !(destination.info().usage & texture_usage_render_target)))
			{
				fmt::throw_exception("Metal scaling supports one color or depth aspect at a time");
			}
			if ((one_dimensional || two_dimensional) &&
				(!range_fits(region.source.array_slice, region.layer_count, source.layers()) ||
				 !range_fits(region.destination.array_slice, region.layer_count, destination.layers())))
			{
				fmt::throw_exception("Metal scale operation exceeds array/cube slices");
			}
			if (three_dimensional && (region.layer_count != 1 || region.source.array_slice != 0 ||
				region.destination.array_slice != 0 || source_depth || destination_depth))
			{
				fmt::throw_exception("Metal 3D scaling requires one color volume without array slices");
			}
			if (one_dimensional && (source_depth || destination_depth))
			{
				fmt::throw_exception("Metal 1D scaling only supports color images");
			}

			const u32 source_width = mip_dimension(source.width(), region.source.mip_level);
			const u32 source_height = mip_dimension(source.height(), region.source.mip_level);
			const u32 destination_width = mip_dimension(destination.width(), region.destination.mip_level);
			const u32 destination_height = mip_dimension(destination.height(), region.destination.mip_level);
			validate_axis(region.source_box.x0, region.source_box.x1, source_width);
			validate_axis(region.source_box.y0, region.source_box.y1, source_height);
			validate_axis(region.destination_box.x0, region.destination_box.x1, destination_width);
			validate_axis(region.destination_box.y0, region.destination_box.y1, destination_height);
			if (one_dimensional && (region.source_box.y0 != 0 || region.source_box.y1 != 1 ||
				region.destination_box.y0 != 0 || region.destination_box.y1 != 1))
			{
				fmt::throw_exception("1D Metal scale boxes require the canonical height interval [0, 1]");
			}
			if (three_dimensional)
			{
				validate_axis(region.source_box.z0, region.source_box.z1,
					mip_dimension(source.depth(), region.source.mip_level));
				validate_axis(region.destination_box.z0, region.destination_box.z1,
					mip_dimension(destination.depth(), region.destination.mip_level));
			}
			else if (region.source_box.z0 != 0 || region.source_box.z1 != 1 ||
				region.destination_box.z0 != 0 || region.destination_box.z1 != 1)
			{
				fmt::throw_exception("2D Metal scale boxes require the canonical depth interval [0, 1]");
			}
			if (conversion.premultiply_alpha && conversion.unpremultiply_alpha)
			{
				fmt::throw_exception("Metal image conversion cannot multiply and divide alpha simultaneously");
			}

			switch (conversion.kind)
			{
			case image_conversion_kind::none:
				if (source_depth != destination_depth)
					fmt::throw_exception("Metal color/depth scaling requires an explicit conversion kind");
				break;
			case image_conversion_kind::channel_remap:
			case image_conversion_kind::color_to_color:
				if (source_depth || destination_depth)
					fmt::throw_exception("Metal color conversion requires color source and destination images");
				break;
			case image_conversion_kind::depth_to_color:
				if (!source_depth || destination_depth)
					fmt::throw_exception("Invalid Metal depth-to-color conversion resources");
				break;
			case image_conversion_kind::color_to_depth:
				if (source_depth || !destination_depth)
					fmt::throw_exception("Invalid Metal color-to-depth conversion resources");
				break;
			}
		}

		scale_parameters make_scale_parameters(const image& source, const image_scale_region& region,
			const image_conversion& conversion)
		{
			const bool reverse_destination_x = region.destination_box.x1 < region.destination_box.x0;
			const bool reverse_destination_y = region.destination_box.y1 < region.destination_box.y0;
			scale_parameters result{};
			result.source_box[0] = static_cast<float>(reverse_destination_x ? region.source_box.x1 : region.source_box.x0);
			result.source_box[1] = static_cast<float>(reverse_destination_y ? region.source_box.y1 : region.source_box.y0);
			result.source_box[2] = static_cast<float>(reverse_destination_x ? region.source_box.x0 : region.source_box.x1);
			result.source_box[3] = static_cast<float>(reverse_destination_y ? region.source_box.y0 : region.source_box.y1);
			result.destination_box[0] = static_cast<float>(std::min(region.destination_box.x0, region.destination_box.x1));
			result.destination_box[1] = static_cast<float>(std::min(region.destination_box.y0, region.destination_box.y1));
			result.destination_box[2] = static_cast<float>(absolute_difference(region.destination_box.x0, region.destination_box.x1));
			result.destination_box[3] = static_cast<float>(absolute_difference(region.destination_box.y0, region.destination_box.y1));
			result.source_size[0] = static_cast<float>(mip_dimension(source.width(), region.source.mip_level));
			result.source_size[1] = static_cast<float>(mip_dimension(source.height(), region.source.mip_level));
			result.channels[0] = static_cast<u32>(conversion.channels.red);
			result.channels[1] = static_cast<u32>(conversion.channels.green);
			result.channels[2] = static_cast<u32>(conversion.channels.blue);
			result.channels[3] = static_cast<u32>(conversion.channels.alpha);
			result.flags = (conversion.decode_srgb ? 1u : 0u) | (conversion.encode_srgb ? 2u : 0u) |
				(conversion.premultiply_alpha ? 4u : 0u) | (conversion.unpremultiply_alpha ? 8u : 0u);
			return result;
		}
	}

	u8 get_aspect_flags(u64 pixel_format)
	{
		const format_layout layout = get_format_layout(static_cast<MTLPixelFormat>(pixel_format));
		if (layout.depth && layout.stencil) return texture_aspect_depth | texture_aspect_stencil;
		if (layout.depth) return texture_aspect_depth;
		if (layout.stencil) return texture_aspect_stencil;
		return texture_aspect_color;
	}

	component_mapping apply_swizzle_remap(const std::array<component_swizzle, 4>& base_remap,
		const rsx::texture_channel_remap_t& remap_vector)
	{
		const auto final_mapping = remap_vector.remap(base_remap, component_swizzle::zero, component_swizzle::one);
		return {final_mapping[1], final_mapping[2], final_mapping[3], final_mapping[0]};
	}

	void transition_image(command_buffer& command, image& resource, const image_state& next,
		const subresource_range& range, bool preserve_encoder)
	{
		const hazard value = resource.transition_hazard(next, range, preserve_encoder);
		const barrier_plan plan = classify_hazard(value, command.active_encoder());
		if (plan.scope == barrier_scope::between_queues || plan.scope == barrier_scope::gpu_to_cpu ||
			plan.scope == barrier_scope::cpu_to_gpu)
		{
			fmt::throw_exception("Metal image transition requires queue-event or CPU synchronization outside a command encoder");
		}
		if (plan)
		{
			if (command.active_encoder() == encoder_kind::none)
			{
				static_cast<void>(command.begin_compute_encoding());
			}
			encode_barrier(command.active_native_encoder(), plan);
			if (plan.end_encoder && command.active_encoder() != encoder_kind::none)
			{
				command.end_encoding();
			}
		}
		resource.set_state(next);
	}

	void transition_image(command_buffer& command, image& resource, const image_state& next, bool preserve_encoder)
	{
		transition_image(command, resource, next,
			{0, resource.mipmaps(), 0, resource.layers(),
			 !!(resource.aspects() & texture_aspect_color),
			 !!(resource.aspects() & texture_aspect_depth),
			 !!(resource.aspects() & texture_aspect_stencil)}, preserve_encoder);
	}

	void copy_image(command_buffer& command, image& source, image& destination,
		std::span<const image_copy_region> regions)
	{
		if (!source || !destination || regions.empty() || source.format() != destination.format() ||
			source.samples() != destination.samples() || !(source.info().usage & texture_usage_copy_source) ||
			!(destination.info().usage & texture_usage_copy_destination))
		{
			fmt::throw_exception("Invalid Metal image copy resources");
		}

		for (const image_copy_region& region : regions)
		{
			validate_texture_region(source, region.source, region.source_origin, region.extent, region.layer_count);
			validate_texture_region(destination, region.destination, region.destination_origin, region.extent, region.layer_count);
			if (region.source.aspects != region.destination.aspects)
			{
				fmt::throw_exception("Metal image copies require matching source and destination aspects");
			}
			transition_image(command, source, operation_state(source, stage_blit, access_blit_read),
				make_range(region.source, region.layer_count), true);
			transition_image(command, destination, operation_state(destination, stage_blit, access_blit_write),
				make_range(region.destination, region.layer_count), true);
		}

		id<MTL4ComputeCommandEncoder> encoder = get_compute_encoder(command);
		id<MTLTexture> source_texture = source.native_handle();
		id<MTLTexture> destination_texture = destination.native_handle();
		command.retain_native_object((__bridge void*)source_texture, true);
		command.retain_native_object((__bridge void*)destination_texture, true);

		for (const image_copy_region& region : regions)
		{
			for (u32 layer = 0; layer < region.layer_count; ++layer)
			{
				if (source_texture == destination_texture)
				{
					MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
					descriptor.textureType = volume_type(source.type()) ? MTLTextureType3D :
						(source.samples() > 1 ? MTLTextureType2DMultisample : MTLTextureType2D);
					descriptor.pixelFormat = source_texture.pixelFormat;
					descriptor.width = region.extent.width;
					descriptor.height = region.extent.height;
					descriptor.depth = region.extent.depth;
					descriptor.sampleCount = source.samples();
					descriptor.storageMode = MTLStorageModePrivate;
					id<MTLTexture> scratch = [command.allocator().owner().native_handle() newTextureWithDescriptor:descriptor];
					if (!scratch)
					{
						fmt::throw_exception("Metal failed to allocate an overlap-safe image copy texture");
					}
					scratch.label = @"RPCS3 overlapping image copy source";
					command.retain_native_object((__bridge void*)scratch, true);
					[encoder copyFromTexture:source_texture
						sourceSlice:region.source.array_slice + layer
						sourceLevel:region.source.mip_level
						sourceOrigin:native_origin(region.source_origin)
						sourceSize:native_size(region.extent)
						toTexture:scratch
						destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
					[encoder copyFromTexture:scratch sourceSlice:0 sourceLevel:0
						sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:native_size(region.extent)
						toTexture:destination_texture
						destinationSlice:region.destination.array_slice + layer
						destinationLevel:region.destination.mip_level
						destinationOrigin:native_origin(region.destination_origin)];
				}
				else
				{
					[encoder copyFromTexture:source_texture
						sourceSlice:region.source.array_slice + layer
						sourceLevel:region.source.mip_level
						sourceOrigin:native_origin(region.source_origin)
						sourceSize:native_size(region.extent)
						toTexture:destination_texture
						destinationSlice:region.destination.array_slice + layer
						destinationLevel:region.destination.mip_level
						destinationOrigin:native_origin(region.destination_origin)];
				}
			}
		}
		command.set_flag(command_has_blit_transfer);
	}

	void upload_image(command_buffer& command, const buffer& source, image& destination,
		std::span<const buffer_image_copy_region> regions)
	{
		if (!source || !destination || regions.empty() || !(source.usage() & buffer_usage_copy_source) ||
			!(destination.info().usage & texture_usage_copy_destination))
		{
			fmt::throw_exception("Invalid Metal image upload resources");
		}

		const MTLPixelFormat format = static_cast<MTLPixelFormat>(destination.format());
		for (const buffer_image_copy_region& region : regions)
		{
			validate_texture_region(destination, region.subresource, region.origin, region.extent, region.layer_count);
			static_cast<void>(resolve_buffer_region(region, format, source.size()));
			transition_image(command, destination, operation_state(destination, stage_blit, access_blit_write),
				make_range(region.subresource, region.layer_count), true);
		}

		id<MTL4ComputeCommandEncoder> encoder = get_compute_encoder(command);
		id<MTLBuffer> source_buffer = source.native_handle();
		id<MTLTexture> destination_texture = destination.native_handle();
		command.retain_native_object((__bridge void*)source_buffer, true);
		command.retain_native_object((__bridge void*)destination_texture, true);
		for (const buffer_image_copy_region& region : regions)
		{
			const resolved_buffer_region resolved = resolve_buffer_region(region, format, source.size());
			const MTLBlitOption option = transfer_option(format, region.subresource.aspects);
			for (u32 layer = 0; layer < region.layer_count; ++layer)
			{
				[encoder copyFromBuffer:source_buffer
					sourceOffset:region.buffer_offset + resolved.layer_stride * layer
					sourceBytesPerRow:resolved.bytes_per_row
					sourceBytesPerImage:resolved.bytes_per_image
					sourceSize:native_size(region.extent)
					toTexture:destination_texture
					destinationSlice:region.subresource.array_slice + layer
					destinationLevel:region.subresource.mip_level
					destinationOrigin:native_origin(region.origin)
					options:option];
			}
		}
		command.set_flag(command_has_blit_transfer);
	}

	void download_image(command_buffer& command, image& source, buffer& destination,
		std::span<const buffer_image_copy_region> regions)
	{
		if (!source || !destination || regions.empty() || !(source.info().usage & texture_usage_copy_source) ||
			!(destination.usage() & buffer_usage_copy_destination))
		{
			fmt::throw_exception("Invalid Metal image download resources");
		}

		const MTLPixelFormat format = static_cast<MTLPixelFormat>(source.format());
		for (const buffer_image_copy_region& region : regions)
		{
			validate_texture_region(source, region.subresource, region.origin, region.extent, region.layer_count);
			static_cast<void>(resolve_buffer_region(region, format, destination.size()));
			transition_image(command, source, operation_state(source, stage_blit, access_blit_read),
				make_range(region.subresource, region.layer_count), true);
		}

		id<MTL4ComputeCommandEncoder> encoder = get_compute_encoder(command);
		id<MTLTexture> source_texture = source.native_handle();
		id<MTLBuffer> destination_buffer = destination.native_handle();
		command.retain_native_object((__bridge void*)source_texture, true);
		command.retain_native_object((__bridge void*)destination_buffer, true);
		for (const buffer_image_copy_region& region : regions)
		{
			const resolved_buffer_region resolved = resolve_buffer_region(region, format, destination.size());
			const MTLBlitOption option = transfer_option(format, region.subresource.aspects);
			for (u32 layer = 0; layer < region.layer_count; ++layer)
			{
				[encoder copyFromTexture:source_texture
					sourceSlice:region.subresource.array_slice + layer
					sourceLevel:region.subresource.mip_level
					sourceOrigin:native_origin(region.origin)
					sourceSize:native_size(region.extent)
					toBuffer:destination_buffer
					destinationOffset:region.buffer_offset + resolved.layer_stride * layer
					destinationBytesPerRow:resolved.bytes_per_row
					destinationBytesPerImage:resolved.bytes_per_image
					options:option];
			}
		}
		command.set_flag(command_has_blit_transfer);
	}

	void scale_image(command_buffer& command, image& source, image& destination,
		const image_scale_region& region, image_filter filter, const image_conversion& conversion)
	{
		validate_scale(source, destination, region, conversion);
		if (!command.is_recording())
		{
			fmt::throw_exception("Metal image scaling requires an active command recording");
		}

		const bool source_depth = region.source.aspects == texture_aspect_depth;
		const bool destination_depth = region.destination.aspects == texture_aspect_depth;
		const bool one_dimensional = is_1d_family(source.type());
		const bool three_dimensional = source.type() == texture_type::texture_3d;
		transition_image(command, source, operation_state(source, stage_fragment, access_shader_read),
			make_range(region.source, region.layer_count), false);
		transition_image(command, destination,
			operation_state(destination, stage_fragment, destination_depth ? access_depth_stencil_write : access_color_write),
			make_range(region.destination, region.layer_count), false);

		if (command.active_encoder() != encoder_kind::none)
		{
			command.end_encoding();
		}

		render_device& render = command.allocator().owner();
		image_operation_cache& cache = get_operation_cache(render);
		id<MTLRenderPipelineState> pipeline = get_scale_pipeline(cache, source_depth, destination_depth, source.type(),
			static_cast<MTLPixelFormat>(destination.format()));
		id<MTLSamplerState> sampler = filter == image_filter::linear ? cache.linear_sampler : cache.nearest_sampler;
		id<MTLTexture> source_texture = source.native_handle();
		id<MTLTexture> destination_texture = destination.native_handle();
		command.retain_native_object((__bridge void*)pipeline, true);
		command.retain_native_object((__bridge void*)source_texture, true);
		command.retain_native_object((__bridge void*)destination_texture, true);

		const u32 source_width = mip_dimension(source.width(), region.source.mip_level);
		const u32 source_height = mip_dimension(source.height(), region.source.mip_level);
		const u32 source_depth_size = mip_dimension(source.depth(), region.source.mip_level);
		NSMutableArray<id<MTLTexture>>* scratch_textures = [NSMutableArray array];
		if (source_texture == destination_texture)
		{
			id<MTL4ComputeCommandEncoder> copy_encoder = get_compute_encoder(command);
			const u32 scratch_count = three_dimensional ? 1 : region.layer_count;
			for (u32 layer = 0; layer < scratch_count; ++layer)
			{
				MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
				descriptor.textureType = three_dimensional ? MTLTextureType3D :
					(one_dimensional ? MTLTextureType1D : MTLTextureType2D);
				descriptor.pixelFormat = source_texture.pixelFormat;
				descriptor.width = source_width;
				descriptor.height = source_height;
				descriptor.depth = source_depth_size;
				descriptor.storageMode = MTLStorageModePrivate;
				descriptor.usage = MTLTextureUsageShaderRead;
				id<MTLTexture> scratch = [cache.device newTextureWithDescriptor:descriptor];
				if (!scratch)
				{
					fmt::throw_exception("Metal failed to allocate an overlap-safe image scaling texture");
				}
				scratch.label = @"RPCS3 overlapping image scale source";
				[copy_encoder copyFromTexture:source_texture
					sourceSlice:region.source.array_slice + layer
					sourceLevel:region.source.mip_level
					sourceOrigin:MTLOriginMake(0, 0, 0)
					sourceSize:MTLSizeMake(source_width, source_height, source_depth_size)
					toTexture:scratch
					destinationSlice:0
					destinationLevel:0
					destinationOrigin:MTLOriginMake(0, 0, 0)];
				[scratch_textures addObject:scratch];
				command.retain_native_object((__bridge void*)scratch, true);
			}
			command.end_encoding();
		}

		constexpr u64 parameter_stride = 256;
		static_assert(sizeof(scale_parameters) <= parameter_stride);
		const u32 destination_plane_count = three_dimensional
			? absolute_difference(region.destination_box.z0, region.destination_box.z1)
			: region.layer_count;
		id<MTLBuffer> parameter_buffer = [cache.device newBufferWithLength:parameter_stride * destination_plane_count
			options:MTLResourceStorageModeShared];
		if (!parameter_buffer)
		{
			fmt::throw_exception("Metal failed to allocate image scaling parameters");
		}
		parameter_buffer.label = @"RPCS3 image scale parameters";
		const scale_parameters base_parameters = make_scale_parameters(source, region, conversion);
		const bool reverse_destination_z = region.destination_box.z1 < region.destination_box.z0;
		const float source_z_start = static_cast<float>(reverse_destination_z ? region.source_box.z1 : region.source_box.z0);
		const float source_z_end = static_cast<float>(reverse_destination_z ? region.source_box.z0 : region.source_box.z1);
		for (u32 layer = 0; layer < destination_plane_count; ++layer)
		{
			scale_parameters parameters = base_parameters;
			if (three_dimensional)
			{
				const float ratio = (static_cast<float>(layer) + 0.5f) / destination_plane_count;
				parameters.source_depth = std::lerp(source_z_start, source_z_end, ratio) / source_depth_size;
				parameters.source_level = source_texture == destination_texture ? 0 : region.source.mip_level;
			}
			std::memcpy(static_cast<u8*>(parameter_buffer.contents) + parameter_stride * layer,
				&parameters, sizeof(parameters));
		}

		MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
		table_descriptor.maxBufferBindCount = 1;
		table_descriptor.maxTextureBindCount = 1;
		table_descriptor.maxSamplerStateBindCount = 1;
		table_descriptor.initializeBindings = YES;
		table_descriptor.label = @"RPCS3 image scale arguments";
		NSError* native_error = nil;
		id<MTL4ArgumentTable> table = [cache.device newArgumentTableWithDescriptor:table_descriptor error:&native_error];
		if (!table)
		{
			throw_native_error(native_error, "Metal image-helper argument-table creation");
		}
		[table setSamplerState:sampler.gpuResourceID atIndex:0];
		command.retain_native_object((__bridge void*)parameter_buffer, true);
		command.retain_native_object((__bridge void*)table, false);

		const u32 destination_width = mip_dimension(destination.width(), region.destination.mip_level);
		const u32 destination_height = mip_dimension(destination.height(), region.destination.mip_level);
		const u32 destination_x = static_cast<u32>(std::min(region.destination_box.x0, region.destination_box.x1));
		const u32 destination_y = static_cast<u32>(std::min(region.destination_box.y0, region.destination_box.y1));
		const u32 destination_box_width = absolute_difference(region.destination_box.x0, region.destination_box.x1);
		const u32 destination_box_height = absolute_difference(region.destination_box.y0, region.destination_box.y1);
		const u32 destination_depth_size = mip_dimension(destination.depth(), region.destination.mip_level);
		const u32 destination_z = static_cast<u32>(std::min(region.destination_box.z0, region.destination_box.z1));
		const bool full_destination = destination_x == 0 && destination_y == 0 &&
			destination_box_width == destination_width && destination_box_height == destination_height &&
			(!three_dimensional || (destination_z == 0 && destination_plane_count == destination_depth_size));
		if (destination.is_memoryless() && !full_destination)
		{
			fmt::throw_exception("A memoryless Metal scale destination must be fully overwritten");
		}

		for (u32 layer = 0; layer < destination_plane_count; ++layer)
		{
			id<MTLTexture> source_view = nil;
			if (source_texture == destination_texture)
			{
				source_view = scratch_textures[three_dimensional ? 0 : layer];
			}
			else if (three_dimensional)
			{
				source_view = source_texture;
			}
			else
			{
				source_view = [source_texture newTextureViewWithPixelFormat:source_texture.pixelFormat
					textureType:(one_dimensional ? MTLTextureType1D : MTLTextureType2D)
					levels:NSMakeRange(region.source.mip_level, 1)
					slices:NSMakeRange(region.source.array_slice + layer, 1)];
				if (!source_view)
				{
					fmt::throw_exception("Metal rejected the image scale source view");
				}
				command.retain_native_object((__bridge void*)source_view, true);
			}

			[table setAddress:parameter_buffer.gpuAddress + parameter_stride * layer atIndex:0];
			[table setTexture:source_view.gpuResourceID atIndex:0];

			MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
			pass.renderTargetWidth = destination_width;
			pass.renderTargetHeight = destination_height;
			if (destination_depth)
			{
				pass.depthAttachment.texture = destination_texture;
				pass.depthAttachment.level = region.destination.mip_level;
				pass.depthAttachment.slice = region.destination.array_slice + layer;
				pass.depthAttachment.loadAction = full_destination ? MTLLoadActionDontCare : MTLLoadActionLoad;
				pass.depthAttachment.storeAction = MTLStoreActionStore;
				if (destination.aspects() & texture_aspect_stencil)
				{
					pass.stencilAttachment.texture = destination_texture;
					pass.stencilAttachment.level = region.destination.mip_level;
					pass.stencilAttachment.slice = region.destination.array_slice + layer;
					pass.stencilAttachment.loadAction = MTLLoadActionLoad;
					pass.stencilAttachment.storeAction = MTLStoreActionStore;
				}
			}
			else
			{
				MTLRenderPassColorAttachmentDescriptor* attachment = pass.colorAttachments[0];
				attachment.texture = destination_texture;
				attachment.level = region.destination.mip_level;
				if (three_dimensional)
				{
					attachment.depthPlane = destination_z + layer;
				}
				else
				{
					attachment.slice = region.destination.array_slice + layer;
				}
				attachment.loadAction = full_destination ? MTLLoadActionDontCare : MTLLoadActionLoad;
				attachment.storeAction = MTLStoreActionStore;
			}

			id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)command.begin_render_encoding((__bridge void*)pass);
			[encoder setRenderPipelineState:pipeline];
			if (destination_depth)
			{
				[encoder setDepthStencilState:cache.depth_write_state];
			}
			[encoder setArgumentTable:table atStages:MTLRenderStageFragment];
			[encoder setViewport:MTLViewport{static_cast<double>(destination_x), static_cast<double>(destination_y),
				static_cast<double>(destination_box_width), static_cast<double>(destination_box_height), 0.0, 1.0}];
			[encoder setScissorRect:MTLScissorRect{destination_x, destination_y, destination_box_width, destination_box_height}];
			[encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
			command.end_encoding();
		}
		command.set_flag(command_reload_dynamic_state);
	}

	void convert_image(command_buffer& command, image& source, image& destination,
		const image_scale_region& region, const image_conversion& conversion)
	{
		if (conversion.kind == image_conversion_kind::none)
		{
			fmt::throw_exception("Metal image conversion requires a conversion operation");
		}
		scale_image(command, source, destination, region, image_filter::nearest, conversion);
	}

	void generate_mipmaps(command_buffer& command, image& resource)
	{
		if (!resource || resource.mipmaps() < 2 || resource.samples() != 1 ||
			!(resource.aspects() & texture_aspect_color) || !(resource.info().usage & texture_usage_shader_read) ||
			!(resource.info().usage & texture_usage_copy_destination))
		{
			fmt::throw_exception("Invalid Metal mipmap generation image");
		}

		const image_state next = operation_state(resource, stage_blit, access_blit_read | access_blit_write);
		transition_image(command, resource, next,
			{0, resource.mipmaps(), 0, resource.layers(), true, false, false}, true);
		id<MTL4ComputeCommandEncoder> encoder = get_compute_encoder(command);
		id<MTLTexture> texture = resource.native_handle();
		command.retain_native_object((__bridge void*)texture, true);
		[encoder generateMipmapsForTexture:texture];
		command.set_flag(command_has_blit_transfer);
	}
}
