#include "stdafx.h"
#include "MTLFormats.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>

namespace mtl
{
	namespace
	{
		constexpr u32 color_capabilities = format_capability_sampled | format_capability_filterable |
			format_capability_render_target | format_capability_blendable | format_capability_shader_write |
			format_capability_pixel_view;
		constexpr u32 integer_capabilities = format_capability_sampled | format_capability_render_target |
			format_capability_shader_write | format_capability_pixel_view;
		constexpr u32 depth_capabilities = format_capability_sampled | format_capability_depth |
			format_capability_render_target | format_capability_pixel_view;

		MTLPixelFormat linear_variant(MTLPixelFormat format)
		{
			switch (format)
			{
			case MTLPixelFormatR8Unorm_sRGB:
			case MTLPixelFormatR8Snorm: return MTLPixelFormatR8Unorm;
			case MTLPixelFormatRG8Unorm_sRGB:
			case MTLPixelFormatRG8Snorm: return MTLPixelFormatRG8Unorm;
			case MTLPixelFormatRGBA8Unorm_sRGB:
			case MTLPixelFormatRGBA8Snorm: return MTLPixelFormatRGBA8Unorm;
			case MTLPixelFormatBGRA8Unorm_sRGB: return MTLPixelFormatBGRA8Unorm;
			case MTLPixelFormatR16Snorm: return MTLPixelFormatR16Unorm;
			case MTLPixelFormatRG16Snorm: return MTLPixelFormatRG16Unorm;
			case MTLPixelFormatRGBA16Snorm: return MTLPixelFormatRGBA16Unorm;
			case MTLPixelFormatBC1_RGBA_sRGB: return MTLPixelFormatBC1_RGBA;
			case MTLPixelFormatBC2_RGBA_sRGB: return MTLPixelFormatBC2_RGBA;
			case MTLPixelFormatBC3_RGBA_sRGB: return MTLPixelFormatBC3_RGBA;
			default: return format;
			}
		}

		native_format_description make_format(MTLPixelFormat format, native_format_class format_class,
			u16 bytes, u8 block_width, u8 block_height, u8 texel_width,
			u8 element_size, u8 elements, u8 aspects, u32 capabilities)
		{
			native_format_description result;
			result.pixel_format = format;
			result.linear_format = linear_variant(format);
			result.srgb_format = get_srgb_format(result.linear_format);
			result.snorm_format = get_snorm_format(result.linear_format);
			result.compatibility_class = format_class;
			result.capabilities = capabilities;
			result.bytes_per_block = bytes;
			result.source_bytes_per_block = bytes;
			result.block_width = block_width;
			result.block_height = block_height;
			result.source_block_width = block_width;
			result.source_block_height = block_height;
			result.texel_width = texel_width;
			result.element_size = element_size;
			result.elements_per_texel = elements;
			result.aspects = aspects;
			result.native_storage = true;
			return result;
		}

		component_mapping opaque_mapping()
		{
			return {component_swizzle::red, component_swizzle::green,
				component_swizzle::blue, component_swizzle::one};
		}

		component_mapping zero_alpha_mapping()
		{
			return {component_swizzle::red, component_swizzle::green,
				component_swizzle::blue, component_swizzle::zero};
		}

		u32 base_texture_format(u32 format)
		{
			return format & ~(CELL_GCM_TEXTURE_LN | CELL_GCM_TEXTURE_UN);
		}

		void select_requested_variant(native_format_description& description,
			bool request_srgb, bool request_snorm)
		{
			if (request_srgb && request_snorm)
			{
				fmt::throw_exception("A Metal texture cannot request sRGB and signed-normalized decoding together");
			}
			if (request_srgb)
			{
				if (description.srgb_format)
				{
					description.pixel_format = description.srgb_format;
					description.capabilities |= format_capability_srgb;
				}
				else
				{
					description.conversion_flags |= format_conversion_srgb_decode;
				}
			}
			if (request_snorm)
			{
				if (description.snorm_format)
				{
					if (description.linear_format == MTLPixelFormatBGRA8Unorm &&
						description.snorm_format == MTLPixelFormatRGBA8Snorm)
					{
						description.conversion_flags |= format_conversion_channel_reorder;
					}
					description.pixel_format = description.snorm_format;
					description.capabilities |= format_capability_snorm;
				}
				else
				{
					description.conversion_flags |= format_conversion_signed_normalize;
				}
			}
		}

		native_format_description with_source(native_format_description description, u32 source,
			u32 conversions = format_conversion_none, component_mapping components = {})
		{
			description.source_format = source;
			description.conversion_flags |= conversions;
			description.components = components;
			constexpr u32 expanded_storage = format_conversion_expand_legacy | format_conversion_decompress |
				format_conversion_packed_yuv | format_conversion_depth16_float | format_conversion_depth24;
			description.native_storage = (description.conversion_flags & expanded_storage) == 0;
			return description;
		}

		native_format_description source_layout(native_format_description description,
			u16 bytes, u8 block_width = 1, u8 block_height = 1)
		{
			description.source_bytes_per_block = bytes;
			description.source_block_width = block_width;
			description.source_block_height = block_height;
			return description;
		}
	}

	u64 get_srgb_format(u64 linear_format)
	{
		switch (static_cast<MTLPixelFormat>(linear_format))
		{
		case MTLPixelFormatR8Unorm: return MTLPixelFormatR8Unorm_sRGB;
		case MTLPixelFormatRG8Unorm: return MTLPixelFormatRG8Unorm_sRGB;
		case MTLPixelFormatRGBA8Unorm: return MTLPixelFormatRGBA8Unorm_sRGB;
		case MTLPixelFormatBGRA8Unorm: return MTLPixelFormatBGRA8Unorm_sRGB;
		case MTLPixelFormatBC1_RGBA: return MTLPixelFormatBC1_RGBA_sRGB;
		case MTLPixelFormatBC2_RGBA: return MTLPixelFormatBC2_RGBA_sRGB;
		case MTLPixelFormatBC3_RGBA: return MTLPixelFormatBC3_RGBA_sRGB;
		default: return 0;
		}
	}

	u64 get_snorm_format(u64 unorm_format)
	{
		switch (static_cast<MTLPixelFormat>(unorm_format))
		{
		case MTLPixelFormatR8Unorm: return MTLPixelFormatR8Snorm;
		case MTLPixelFormatRG8Unorm: return MTLPixelFormatRG8Snorm;
		case MTLPixelFormatRGBA8Unorm: return MTLPixelFormatRGBA8Snorm;
		case MTLPixelFormatBGRA8Unorm: return MTLPixelFormatRGBA8Snorm;
		case MTLPixelFormatR16Unorm: return MTLPixelFormatR16Snorm;
		case MTLPixelFormatRG16Unorm: return MTLPixelFormatRG16Snorm;
		case MTLPixelFormatRGBA16Unorm: return MTLPixelFormatRGBA16Snorm;
		default: return 0;
		}
	}

	native_format_description describe_native_format(u64 pixel_format)
	{
		const auto format = static_cast<MTLPixelFormat>(pixel_format);
		switch (format)
		{
		case MTLPixelFormatR8Unorm:
		case MTLPixelFormatR8Unorm_sRGB:
		case MTLPixelFormatR8Snorm:
			return make_format(format, native_format_class::r8, 1, 1, 1, 1, 1, 1,
				texture_aspect_color, color_capabilities |
				(format == MTLPixelFormatR8Unorm_sRGB ? format_capability_srgb : 0) |
				(format == MTLPixelFormatR8Snorm ? format_capability_snorm : 0));
		case MTLPixelFormatR8Uint:
		case MTLPixelFormatR8Sint:
			return make_format(format, native_format_class::r8, 1, 1, 1, 1, 1, 1,
				texture_aspect_color, integer_capabilities);
		case MTLPixelFormatRG8Unorm:
		case MTLPixelFormatRG8Unorm_sRGB:
		case MTLPixelFormatRG8Snorm:
			return make_format(format, native_format_class::rg8, 2, 1, 1, 2, 2, 1,
				texture_aspect_color, color_capabilities |
				(format == MTLPixelFormatRG8Unorm_sRGB ? format_capability_srgb : 0) |
				(format == MTLPixelFormatRG8Snorm ? format_capability_snorm : 0));
		case MTLPixelFormatRG8Uint:
		case MTLPixelFormatRG8Sint:
			return make_format(format, native_format_class::rg8, 2, 1, 1, 2, 2, 1,
				texture_aspect_color, integer_capabilities);
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
		case MTLPixelFormatRGBA8Snorm:
			return make_format(format, native_format_class::rgba8, 4, 1, 1, 4, 4, 1,
				texture_aspect_color, color_capabilities |
				(format == MTLPixelFormatRGBA8Unorm_sRGB ? format_capability_srgb : 0) |
				(format == MTLPixelFormatRGBA8Snorm ? format_capability_snorm : 0));
		case MTLPixelFormatRGBA8Uint:
		case MTLPixelFormatRGBA8Sint:
			return make_format(format, native_format_class::rgba8, 4, 1, 1, 4, 4, 1,
				texture_aspect_color, integer_capabilities);
		case MTLPixelFormatBGRA8Unorm:
		case MTLPixelFormatBGRA8Unorm_sRGB:
			return make_format(format, native_format_class::bgra8, 4, 1, 1, 4, 4, 1,
				texture_aspect_color, color_capabilities |
				(format == MTLPixelFormatBGRA8Unorm_sRGB ? format_capability_srgb : 0));

		case MTLPixelFormatB5G6R5Unorm:
		case MTLPixelFormatA1BGR5Unorm:
		case MTLPixelFormatABGR4Unorm:
		case MTLPixelFormatBGR5A1Unorm:
			return make_format(format, native_format_class::packed16, 2, 1, 1, 2, 2, 1,
				texture_aspect_color, format_capability_sampled | format_capability_filterable);

		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatR16Snorm:
		case MTLPixelFormatR16Float:
			return make_format(format, native_format_class::r16, 2, 1, 1, 2, 2, 1,
				texture_aspect_color, color_capabilities |
				(format == MTLPixelFormatR16Snorm ? format_capability_snorm : 0));
		case MTLPixelFormatR16Uint:
		case MTLPixelFormatR16Sint:
			return make_format(format, native_format_class::r16, 2, 1, 1, 2, 2, 1,
				texture_aspect_color, integer_capabilities);
		case MTLPixelFormatRG16Unorm:
		case MTLPixelFormatRG16Snorm:
		case MTLPixelFormatRG16Float:
			return make_format(format, native_format_class::rg16, 4, 1, 1, 4, 2, 2,
				texture_aspect_color, color_capabilities |
				(format == MTLPixelFormatRG16Snorm ? format_capability_snorm : 0));
		case MTLPixelFormatRG16Uint:
		case MTLPixelFormatRG16Sint:
			return make_format(format, native_format_class::rg16, 4, 1, 1, 4, 2, 2,
				texture_aspect_color, integer_capabilities);
		case MTLPixelFormatRGBA16Unorm:
		case MTLPixelFormatRGBA16Snorm:
		case MTLPixelFormatRGBA16Float:
			return make_format(format, native_format_class::rgba16, 8, 1, 1, 8, 2, 4,
				texture_aspect_color, color_capabilities |
				(format == MTLPixelFormatRGBA16Snorm ? format_capability_snorm : 0));
		case MTLPixelFormatRGBA16Uint:
		case MTLPixelFormatRGBA16Sint:
			return make_format(format, native_format_class::rgba16, 8, 1, 1, 8, 2, 4,
				texture_aspect_color, integer_capabilities);

		case MTLPixelFormatR32Float:
			return make_format(format, native_format_class::r32, 4, 1, 1, 4, 4, 1,
				texture_aspect_color, color_capabilities);
		case MTLPixelFormatR32Uint:
		case MTLPixelFormatR32Sint:
			return make_format(format, native_format_class::r32, 4, 1, 1, 4, 4, 1,
				texture_aspect_color, integer_capabilities);
		case MTLPixelFormatRGBA32Float:
			return make_format(format, native_format_class::rgba32, 16, 1, 1, 16, 4, 4,
				texture_aspect_color, color_capabilities);
		case MTLPixelFormatRGBA32Uint:
		case MTLPixelFormatRGBA32Sint:
			return make_format(format, native_format_class::rgba32, 16, 1, 1, 16, 4, 4,
				texture_aspect_color, integer_capabilities);

		case MTLPixelFormatBC1_RGBA:
		case MTLPixelFormatBC1_RGBA_sRGB:
			return make_format(format, native_format_class::bc1, 8, 4, 4, 4, 4, 1,
				texture_aspect_color, format_capability_sampled | format_capability_filterable |
				format_capability_compressed | (format == MTLPixelFormatBC1_RGBA_sRGB ? format_capability_srgb : 0));
		case MTLPixelFormatBC2_RGBA:
		case MTLPixelFormatBC2_RGBA_sRGB:
			return make_format(format, native_format_class::bc2, 16, 4, 4, 4, 4, 1,
				texture_aspect_color, format_capability_sampled | format_capability_filterable |
				format_capability_compressed | (format == MTLPixelFormatBC2_RGBA_sRGB ? format_capability_srgb : 0));
		case MTLPixelFormatBC3_RGBA:
		case MTLPixelFormatBC3_RGBA_sRGB:
			return make_format(format, native_format_class::bc3, 16, 4, 4, 4, 4, 1,
				texture_aspect_color, format_capability_sampled | format_capability_filterable |
				format_capability_compressed | (format == MTLPixelFormatBC3_RGBA_sRGB ? format_capability_srgb : 0));

		case MTLPixelFormatDepth16Unorm:
			return make_format(format, native_format_class::depth16, 2, 1, 1, 2, 2, 1,
				texture_aspect_depth, depth_capabilities);
		case MTLPixelFormatDepth32Float:
			return make_format(format, native_format_class::depth32, 4, 1, 1, 4, 4, 1,
				texture_aspect_depth, depth_capabilities);
		case MTLPixelFormatDepth32Float_Stencil8:
			return make_format(format, native_format_class::depth32_stencil8, 8, 1, 1, 4, 4, 1,
				texture_aspect_depth | texture_aspect_stencil,
				depth_capabilities | format_capability_stencil);
		case MTLPixelFormatStencil8:
			return make_format(format, native_format_class::depth32_stencil8, 1, 1, 1, 1, 1, 1,
				texture_aspect_stencil, format_capability_stencil | format_capability_render_target);
		default:
			fmt::throw_exception("Unsupported native Metal pixel format %llu", pixel_format);
		}
	}

	native_format_description get_depth_surface_format(
		const device_info& device, rsx::surface_depth_format2 format)
	{
		switch (format)
		{
		case rsx::surface_depth_format2::z16_uint:
			return with_source(describe_native_format(MTLPixelFormatDepth16Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_16);
		case rsx::surface_depth_format2::z16_float:
			return source_layout(with_source(describe_native_format(MTLPixelFormatDepth32Float), static_cast<u32>(format),
				format_conversion_byte_swap_16 | format_conversion_depth16_float), 2);
		case rsx::surface_depth_format2::z24s8_uint:
			if (device.formats.depth24_unorm_stencil8)
			{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
				auto result = make_format(MTLPixelFormatDepth24Unorm_Stencil8,
					native_format_class::depth32_stencil8, 4, 1, 1, 4, 4, 1,
					texture_aspect_depth | texture_aspect_stencil,
					depth_capabilities | format_capability_stencil);
#pragma clang diagnostic pop
				return with_source(result, static_cast<u32>(format), format_conversion_byte_swap_32);
			}
			if (device.formats.depth32_float_stencil8)
			{
				return source_layout(with_source(describe_native_format(MTLPixelFormatDepth32Float_Stencil8),
					static_cast<u32>(format), format_conversion_byte_swap_32 | format_conversion_depth24), 4);
			}
			fmt::throw_exception("Metal device has no depth/stencil format for RSX Z24S8");
		case rsx::surface_depth_format2::z24s8_float:
			if (!device.formats.depth32_float_stencil8)
			{
				fmt::throw_exception("Metal device has no depth/stencil format for floating RSX Z24S8");
			}
			return source_layout(with_source(describe_native_format(MTLPixelFormatDepth32Float_Stencil8),
				static_cast<u32>(format), format_conversion_byte_swap_32 | format_conversion_depth24), 4);
		}
		fmt::throw_exception("Invalid RSX depth surface format 0x%x", static_cast<u32>(format));
	}

	native_format_description get_color_surface_format(
		const device_info& device, rsx::surface_color_format format)
	{
		switch (format)
		{
		case rsx::surface_color_format::r5g6b5:
			return source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_16 | format_conversion_expand_legacy | format_conversion_channel_reorder), 2);
		case rsx::surface_color_format::x1r5g5b5_o1r5g5b5:
			return source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_16 | format_conversion_expand_legacy | format_conversion_force_alpha_one,
				opaque_mapping()), 2);
		case rsx::surface_color_format::x1r5g5b5_z1r5g5b5:
			return source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_16 | format_conversion_expand_legacy, zero_alpha_mapping()), 2);
		case rsx::surface_color_format::a8r8g8b8:
			return with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_32);
		case rsx::surface_color_format::a8b8g8r8:
			return with_source(describe_native_format(MTLPixelFormatRGBA8Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_32);
		case rsx::surface_color_format::x8b8g8r8_o8b8g8r8:
			return with_source(describe_native_format(MTLPixelFormatRGBA8Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_32 | format_conversion_force_alpha_one, opaque_mapping());
		case rsx::surface_color_format::x8b8g8r8_z8b8g8r8:
			return with_source(describe_native_format(MTLPixelFormatRGBA8Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_32, zero_alpha_mapping());
		case rsx::surface_color_format::x8r8g8b8_o8r8g8b8:
			return with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_32 | format_conversion_force_alpha_one, opaque_mapping());
		case rsx::surface_color_format::x8r8g8b8_z8r8g8b8:
			return with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_32, zero_alpha_mapping());
		case rsx::surface_color_format::w16z16y16x16:
			if (!device.formats.rgba16_float) fmt::throw_exception("Metal device does not support RGBA16Float surfaces");
			return with_source(describe_native_format(MTLPixelFormatRGBA16Float), static_cast<u32>(format),
				format_conversion_byte_swap_16);
		case rsx::surface_color_format::w32z32y32x32:
			if (!device.formats.rgba32_float) fmt::throw_exception("Metal device does not support RGBA32Float surfaces");
			return with_source(describe_native_format(MTLPixelFormatRGBA32Float), static_cast<u32>(format),
				format_conversion_byte_swap_32);
		case rsx::surface_color_format::b8:
			return with_source(describe_native_format(MTLPixelFormatR8Unorm), static_cast<u32>(format),
				format_conversion_none,
				{component_swizzle::red, component_swizzle::red, component_swizzle::red, component_swizzle::one});
		case rsx::surface_color_format::g8b8:
			return with_source(describe_native_format(MTLPixelFormatRG8Unorm), static_cast<u32>(format),
				format_conversion_byte_swap_16,
				{component_swizzle::red, component_swizzle::green, component_swizzle::red, component_swizzle::green});
		case rsx::surface_color_format::x32:
			return with_source(describe_native_format(MTLPixelFormatR32Float), static_cast<u32>(format),
				format_conversion_byte_swap_32,
				{component_swizzle::red, component_swizzle::red, component_swizzle::red, component_swizzle::red});
		}
		fmt::throw_exception("Invalid RSX color surface format 0x%x", static_cast<u32>(format));
	}

	native_format_description get_sampler_format(const device_info& device, u32 format,
		bool request_srgb, bool request_snorm)
	{
		const u32 source = base_texture_format(format);
		native_format_description result;
		switch (source)
		{
		case CELL_GCM_TEXTURE_R5G6B5:
			result = source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
				format_conversion_byte_swap_16 | format_conversion_expand_legacy |
				format_conversion_channel_reorder), 2); break;
		case CELL_GCM_TEXTURE_R6G5B5:
			result = source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
				format_conversion_byte_swap_16 | format_conversion_expand_legacy | format_conversion_channel_reorder), 2); break;
		case CELL_GCM_TEXTURE_R5G5B5A1:
			result = source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
				format_conversion_byte_swap_16 | format_conversion_expand_legacy |
				format_conversion_channel_reorder), 2); break;
		case CELL_GCM_TEXTURE_D1R5G5B5:
			result = source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
				format_conversion_byte_swap_16 | format_conversion_expand_legacy |
				format_conversion_force_alpha_one, opaque_mapping()), 2); break;
		case CELL_GCM_TEXTURE_A1R5G5B5:
			result = source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
				format_conversion_byte_swap_16 | format_conversion_expand_legacy |
				format_conversion_channel_reorder), 2); break;
		case CELL_GCM_TEXTURE_A4R4G4B4:
			result = source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
				format_conversion_byte_swap_16 | format_conversion_expand_legacy |
				format_conversion_channel_reorder), 2); break;
		case CELL_GCM_TEXTURE_B8:
			result = with_source(describe_native_format(MTLPixelFormatR8Unorm), source, format_conversion_none,
				{component_swizzle::red, component_swizzle::red, component_swizzle::red, component_swizzle::one}); break;
		case CELL_GCM_TEXTURE_A8R8G8B8:
			result = with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
				format_conversion_byte_swap_32); break;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT1:
			result = device.formats.bc_texture_compression ?
				with_source(describe_native_format(MTLPixelFormatBC1_RGBA), source) :
				source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
					format_conversion_decompress), 8, 4, 4); break;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT23:
			result = device.formats.bc_texture_compression ?
				with_source(describe_native_format(MTLPixelFormatBC2_RGBA), source) :
				source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
					format_conversion_decompress), 16, 4, 4); break;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT45:
			result = device.formats.bc_texture_compression ?
				with_source(describe_native_format(MTLPixelFormatBC3_RGBA), source) :
				source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
					format_conversion_decompress), 16, 4, 4); break;
		case CELL_GCM_TEXTURE_G8B8:
			result = with_source(describe_native_format(MTLPixelFormatRG8Unorm), source, format_conversion_byte_swap_16,
				{component_swizzle::red, component_swizzle::green, component_swizzle::red, component_swizzle::green}); break;
		case CELL_GCM_TEXTURE_DEPTH24_D8:
			result = get_depth_surface_format(device, rsx::surface_depth_format2::z24s8_uint); result.source_format = source; break;
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
			result = get_depth_surface_format(device, rsx::surface_depth_format2::z24s8_float); result.source_format = source; break;
		case CELL_GCM_TEXTURE_DEPTH16:
			result = get_depth_surface_format(device, rsx::surface_depth_format2::z16_uint); result.source_format = source; break;
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			result = get_depth_surface_format(device, rsx::surface_depth_format2::z16_float); result.source_format = source; break;
		case CELL_GCM_TEXTURE_X16:
			result = with_source(describe_native_format(MTLPixelFormatR16Unorm), source, format_conversion_byte_swap_16,
				{component_swizzle::one, component_swizzle::red, component_swizzle::one, component_swizzle::red}); break;
		case CELL_GCM_TEXTURE_Y16_X16:
			result = with_source(describe_native_format(MTLPixelFormatRG16Unorm), source, format_conversion_byte_swap_16,
				{component_swizzle::red, component_swizzle::green, component_swizzle::red, component_swizzle::green}); break;
		case CELL_GCM_TEXTURE_Y16_X16_FLOAT:
			result = with_source(describe_native_format(MTLPixelFormatRG16Float), source, format_conversion_byte_swap_16,
				{component_swizzle::red, component_swizzle::green, component_swizzle::red, component_swizzle::green}); break;
		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT:
			result = with_source(describe_native_format(MTLPixelFormatRGBA16Float), source,
				format_conversion_byte_swap_16 | format_conversion_channel_reorder); break;
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT:
			result = with_source(describe_native_format(MTLPixelFormatRGBA32Float), source,
				format_conversion_byte_swap_32 | format_conversion_channel_reorder); break;
		case CELL_GCM_TEXTURE_X32_FLOAT:
			result = with_source(describe_native_format(MTLPixelFormatR32Float), source, format_conversion_byte_swap_32,
				{component_swizzle::red, component_swizzle::red, component_swizzle::red, component_swizzle::red}); break;
		case CELL_GCM_TEXTURE_D8R8G8B8:
			result = with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
				format_conversion_byte_swap_32 | format_conversion_force_alpha_one, opaque_mapping()); break;
		case CELL_GCM_TEXTURE_COMPRESSED_HILO8:
			result = with_source(describe_native_format(MTLPixelFormatRG8Unorm), source, format_conversion_none,
				{component_swizzle::red, component_swizzle::green, component_swizzle::red, component_swizzle::green}); break;
		case CELL_GCM_TEXTURE_COMPRESSED_HILO_S8:
			result = with_source(describe_native_format(MTLPixelFormatRG8Snorm), source, format_conversion_none,
				{component_swizzle::red, component_swizzle::green, component_swizzle::red, component_swizzle::green}); break;
		case CELL_GCM_TEXTURE_COMPRESSED_B8R8_G8R8:
		case CELL_GCM_TEXTURE_COMPRESSED_R8B8_R8G8:
			result = source_layout(with_source(describe_native_format(MTLPixelFormatBGRA8Unorm), source,
				format_conversion_packed_yuv | format_conversion_channel_reorder), 4, 2, 1); break;
		default:
			fmt::throw_exception("Invalid RSX sampler format 0x%x", source);
		}
		select_requested_variant(result, request_srgb, request_snorm);
		return result;
	}

	format_compatibility get_view_compatibility(u64 pixel_format)
	{
		format_compatibility result;
		result.base_format = pixel_format;
		const auto format = static_cast<MTLPixelFormat>(pixel_format);
		switch (format)
		{
		case MTLPixelFormatR8Unorm:
		case MTLPixelFormatR8Unorm_sRGB:
		case MTLPixelFormatR8Snorm:
			result.view_formats = {MTLPixelFormatR8Unorm, MTLPixelFormatR8Unorm_sRGB, MTLPixelFormatR8Snorm}; break;
		case MTLPixelFormatRG8Unorm:
		case MTLPixelFormatRG8Unorm_sRGB:
		case MTLPixelFormatRG8Snorm:
			result.view_formats = {MTLPixelFormatRG8Unorm, MTLPixelFormatRG8Unorm_sRGB, MTLPixelFormatRG8Snorm}; break;
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
		case MTLPixelFormatRGBA8Snorm:
			result.view_formats = {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB, MTLPixelFormatRGBA8Snorm}; break;
		case MTLPixelFormatBGRA8Unorm:
		case MTLPixelFormatBGRA8Unorm_sRGB:
			result.view_formats = {MTLPixelFormatBGRA8Unorm, MTLPixelFormatBGRA8Unorm_sRGB}; break;
		case MTLPixelFormatBC1_RGBA:
		case MTLPixelFormatBC1_RGBA_sRGB:
			result.view_formats = {MTLPixelFormatBC1_RGBA, MTLPixelFormatBC1_RGBA_sRGB}; break;
		case MTLPixelFormatBC2_RGBA:
		case MTLPixelFormatBC2_RGBA_sRGB:
			result.view_formats = {MTLPixelFormatBC2_RGBA, MTLPixelFormatBC2_RGBA_sRGB}; break;
		case MTLPixelFormatBC3_RGBA:
		case MTLPixelFormatBC3_RGBA_sRGB:
			result.view_formats = {MTLPixelFormatBC3_RGBA, MTLPixelFormatBC3_RGBA_sRGB}; break;
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatR16Snorm:
		case MTLPixelFormatR16Uint:
		case MTLPixelFormatR16Sint:
		case MTLPixelFormatR16Float:
			result.view_formats = {MTLPixelFormatR16Unorm, MTLPixelFormatR16Snorm,
				MTLPixelFormatR16Uint, MTLPixelFormatR16Sint, MTLPixelFormatR16Float}; break;
		case MTLPixelFormatRG16Unorm:
		case MTLPixelFormatRG16Snorm:
		case MTLPixelFormatRG16Uint:
		case MTLPixelFormatRG16Sint:
		case MTLPixelFormatRG16Float:
			result.view_formats = {MTLPixelFormatRG16Unorm, MTLPixelFormatRG16Snorm,
				MTLPixelFormatRG16Uint, MTLPixelFormatRG16Sint, MTLPixelFormatRG16Float}; break;
		case MTLPixelFormatRGBA16Unorm:
		case MTLPixelFormatRGBA16Snorm:
		case MTLPixelFormatRGBA16Uint:
		case MTLPixelFormatRGBA16Sint:
		case MTLPixelFormatRGBA16Float:
			result.view_formats = {MTLPixelFormatRGBA16Unorm, MTLPixelFormatRGBA16Snorm,
				MTLPixelFormatRGBA16Uint, MTLPixelFormatRGBA16Sint, MTLPixelFormatRGBA16Float}; break;
		case MTLPixelFormatR32Uint:
		case MTLPixelFormatR32Sint:
		case MTLPixelFormatR32Float:
			result.view_formats = {MTLPixelFormatR32Uint, MTLPixelFormatR32Sint, MTLPixelFormatR32Float}; break;
		case MTLPixelFormatRGBA32Uint:
		case MTLPixelFormatRGBA32Sint:
		case MTLPixelFormatRGBA32Float:
			result.view_formats = {MTLPixelFormatRGBA32Uint, MTLPixelFormatRGBA32Sint, MTLPixelFormatRGBA32Float}; break;
		case MTLPixelFormatDepth32Float_Stencil8:
			result.view_formats = {MTLPixelFormatDepth32Float_Stencil8, MTLPixelFormatX32_Stencil8}; break;
		case MTLPixelFormatX32_Stencil8:
			result.view_formats = {MTLPixelFormatX32_Stencil8}; break;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		case MTLPixelFormatDepth24Unorm_Stencil8:
			result.view_formats = {MTLPixelFormatDepth24Unorm_Stencil8, MTLPixelFormatX24_Stencil8}; break;
		case MTLPixelFormatX24_Stencil8:
			result.view_formats = {MTLPixelFormatX24_Stencil8}; break;
#pragma clang diagnostic pop
		default:
			break;
		}
		return result;
	}

	bool formats_are_bitcast_compatible(const native_format_description& first,
		const native_format_description& second, bool preserve_depth_encoding)
	{
		if (!first || !second) return false;
		if (first.pixel_format == second.pixel_format &&
			(!preserve_depth_encoding || first.source_format == second.source_format ||
				!first.source_format || !second.source_format)) return true;
		const bool first_depth = (first.aspects & texture_aspect_depth) != 0;
		const bool second_depth = (second.aspects & texture_aspect_depth) != 0;
		if (first_depth != second_depth) return false;
		if (preserve_depth_encoding && first_depth &&
			((first.conversion_flags ^ second.conversion_flags) &
				(format_conversion_depth16_float | format_conversion_depth24))) return false;
		if (first.bytes_per_block != second.bytes_per_block || first.block_width != second.block_width ||
			first.block_height != second.block_height) return false;
		if ((first.capabilities & format_capability_compressed) ||
			(second.capabilities & format_capability_compressed))
		{
			return first.compatibility_class == second.compatibility_class;
		}
		constexpr u32 transform_mask = format_conversion_byte_swap_16 | format_conversion_byte_swap_32 |
			format_conversion_expand_legacy | format_conversion_decompress | format_conversion_packed_yuv |
			format_conversion_depth16_float | format_conversion_depth24;
		return (first.conversion_flags & transform_mask) == (second.conversion_flags & transform_mask);
	}

	bool formats_are_bitcast_compatible(const image& first, const image& second)
	{
		return formats_are_bitcast_compatible(describe_native_format(first.format()),
			describe_native_format(second.format()), true);
	}

	minification_filter get_min_filter(rsx::texture_minify_filter filter)
	{
		switch (filter)
		{
		case rsx::texture_minify_filter::nearest: return {sampler_filter::nearest, sampler_mip_filter::none, false};
		case rsx::texture_minify_filter::linear: return {sampler_filter::linear, sampler_mip_filter::none, false};
		case rsx::texture_minify_filter::nearest_nearest: return {sampler_filter::nearest, sampler_mip_filter::nearest, true};
		case rsx::texture_minify_filter::linear_nearest: return {sampler_filter::linear, sampler_mip_filter::nearest, true};
		case rsx::texture_minify_filter::nearest_linear: return {sampler_filter::nearest, sampler_mip_filter::linear, true};
		case rsx::texture_minify_filter::linear_linear: return {sampler_filter::linear, sampler_mip_filter::linear, true};
		case rsx::texture_minify_filter::convolution_min: return {sampler_filter::linear, sampler_mip_filter::none, false};
		}
		fmt::throw_exception("Invalid RSX minification filter 0x%x", static_cast<u32>(filter));
	}

	sampler_filter get_mag_filter(rsx::texture_magnify_filter filter)
	{
		switch (filter)
		{
		case rsx::texture_magnify_filter::nearest: return sampler_filter::nearest;
		case rsx::texture_magnify_filter::linear:
		case rsx::texture_magnify_filter::convolution_mag: return sampler_filter::linear;
		}
		fmt::throw_exception("Invalid RSX magnification filter 0x%x", static_cast<u32>(filter));
	}

	border_color get_border_color(u32 color, u64 native_format, u8 aspects, bool integer)
	{
		switch (color)
		{
		case 0x00000000: return border_color::transparent_black();
		case 0xff000000: return border_color::opaque_black();
		case 0xffffffff: return border_color::opaque_white();
		default:
			return border_color::custom(
				{static_cast<f32>((color >> 16) & 0xff) / 255.f,
				 static_cast<f32>((color >> 8) & 0xff) / 255.f,
				 static_cast<f32>(color & 0xff) / 255.f,
				 static_cast<f32>(color >> 24) / 255.f}, native_format, aspects, integer);
		}
	}

	sampler_address_mode get_wrap_mode(rsx::texture_wrap_mode mode)
	{
		switch (mode)
		{
		case rsx::texture_wrap_mode::wrap: return sampler_address_mode::wrap;
		case rsx::texture_wrap_mode::mirror: return sampler_address_mode::mirror;
		case rsx::texture_wrap_mode::clamp_to_edge: return sampler_address_mode::clamp_to_edge;
		case rsx::texture_wrap_mode::border: return sampler_address_mode::border;
		case rsx::texture_wrap_mode::clamp: return sampler_address_mode::clamp;
		case rsx::texture_wrap_mode::mirror_once_clamp_to_edge: return sampler_address_mode::mirror_once_clamp_to_edge;
		case rsx::texture_wrap_mode::mirror_once_border: return sampler_address_mode::mirror_once_border;
		case rsx::texture_wrap_mode::mirror_once_clamp: return sampler_address_mode::mirror_once_clamp;
		}
		fmt::throw_exception("Invalid RSX texture wrap mode 0x%x", static_cast<u32>(mode));
	}

	f32 get_max_anisotropy(rsx::texture_max_anisotropy anisotropy)
	{
		switch (anisotropy)
		{
		case rsx::texture_max_anisotropy::x1: return 1.f;
		case rsx::texture_max_anisotropy::x2: return 2.f;
		case rsx::texture_max_anisotropy::x4: return 4.f;
		case rsx::texture_max_anisotropy::x6: return 6.f;
		case rsx::texture_max_anisotropy::x8: return 8.f;
		case rsx::texture_max_anisotropy::x10: return 10.f;
		case rsx::texture_max_anisotropy::x12: return 12.f;
		case rsx::texture_max_anisotropy::x16: return 16.f;
		}
		fmt::throw_exception("Invalid RSX anisotropy 0x%x", static_cast<u32>(anisotropy));
	}

	std::array<component_swizzle, 4> get_component_mapping(u32 format)
	{
		switch (base_texture_format(format))
		{
		case CELL_GCM_TEXTURE_A1R5G5B5:
		case CELL_GCM_TEXTURE_R5G5B5A1:
		case CELL_GCM_TEXTURE_R6G5B5:
		case CELL_GCM_TEXTURE_R5G6B5:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT1:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT23:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT45:
		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT:
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT:
		case CELL_GCM_TEXTURE_A8R8G8B8:
		case CELL_GCM_TEXTURE_COMPRESSED_B8R8_G8R8:
		case CELL_GCM_TEXTURE_COMPRESSED_R8B8_R8G8:
			return {component_swizzle::alpha, component_swizzle::red,
				component_swizzle::green, component_swizzle::blue};
		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
		case CELL_GCM_TEXTURE_X32_FLOAT:
			return {component_swizzle::red, component_swizzle::red,
				component_swizzle::red, component_swizzle::red};
		case CELL_GCM_TEXTURE_A4R4G4B4:
			return {component_swizzle::red, component_swizzle::green,
				component_swizzle::blue, component_swizzle::alpha};
		case CELL_GCM_TEXTURE_G8B8:
			return {component_swizzle::green, component_swizzle::red,
				component_swizzle::green, component_swizzle::red};
		case CELL_GCM_TEXTURE_B8:
			return {component_swizzle::one, component_swizzle::red,
				component_swizzle::red, component_swizzle::red};
		case CELL_GCM_TEXTURE_X16:
			return {component_swizzle::red, component_swizzle::one,
				component_swizzle::red, component_swizzle::one};
		case CELL_GCM_TEXTURE_Y16_X16:
			return {component_swizzle::green, component_swizzle::red,
				component_swizzle::green, component_swizzle::red};
		case CELL_GCM_TEXTURE_Y16_X16_FLOAT:
		case CELL_GCM_TEXTURE_COMPRESSED_HILO8:
		case CELL_GCM_TEXTURE_COMPRESSED_HILO_S8:
			return {component_swizzle::red, component_swizzle::green,
				component_swizzle::red, component_swizzle::green};
		case CELL_GCM_TEXTURE_D8R8G8B8:
		case CELL_GCM_TEXTURE_D1R5G5B5:
			return {component_swizzle::one, component_swizzle::red,
				component_swizzle::green, component_swizzle::blue};
		}
		fmt::throw_exception("Invalid RSX component mapping format 0x%x", format);
	}

	primitive_mapping get_primitive_mapping(rsx::primitive_type primitive)
	{
		switch (primitive)
		{
		case rsx::primitive_type::points: return {primitive_topology::point, false};
		case rsx::primitive_type::lines: return {primitive_topology::line, false};
		case rsx::primitive_type::line_loop: return {primitive_topology::line_strip, true};
		case rsx::primitive_type::line_strip: return {primitive_topology::line_strip, false};
		case rsx::primitive_type::triangles: return {primitive_topology::triangle, false};
		case rsx::primitive_type::triangle_strip:
		case rsx::primitive_type::quad_strip: return {primitive_topology::triangle_strip, false};
		case rsx::primitive_type::triangle_fan:
		case rsx::primitive_type::quads:
		case rsx::primitive_type::polygon: return {primitive_topology::triangle, true};
		}
		fmt::throw_exception("Invalid RSX primitive topology 0x%x", static_cast<u32>(primitive));
	}

	index_element_type get_index_type(rsx::index_array_type type)
	{
		switch (type)
		{
		case rsx::index_array_type::u16: return index_element_type::u16;
		case rsx::index_array_type::u32: return index_element_type::u32;
		}
		fmt::throw_exception("Invalid RSX index type 0x%x", static_cast<u32>(type));
	}
}
