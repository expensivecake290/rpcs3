#pragma once

#include <array>

#include "Emu/RSX/gcm_enums.h"
#include "mtlutils/device.h"
#include "mtlutils/graphics_pipeline_state.hpp"
#include "mtlutils/image.h"
#include "mtlutils/sampler.h"

namespace mtl
{
	enum native_format_capability : u32
	{
		format_capability_none = 0,
		format_capability_sampled = 1 << 0,
		format_capability_filterable = 1 << 1,
		format_capability_render_target = 1 << 2,
		format_capability_blendable = 1 << 3,
		format_capability_shader_write = 1 << 4,
		format_capability_depth = 1 << 5,
		format_capability_stencil = 1 << 6,
		format_capability_compressed = 1 << 7,
		format_capability_srgb = 1 << 8,
		format_capability_snorm = 1 << 9,
		format_capability_pixel_view = 1 << 10,
	};

	enum format_conversion_flag : u32
	{
		format_conversion_none = 0,
		format_conversion_byte_swap_16 = 1 << 0,
		format_conversion_byte_swap_32 = 1 << 1,
		format_conversion_expand_legacy = 1 << 2,
		format_conversion_decompress = 1 << 3,
		format_conversion_packed_yuv = 1 << 4,
		format_conversion_depth16_float = 1 << 5,
		format_conversion_depth24 = 1 << 6,
		format_conversion_force_alpha_one = 1 << 7,
		format_conversion_channel_reorder = 1 << 8,
		format_conversion_srgb_decode = 1 << 9,
		format_conversion_signed_normalize = 1 << 10,
	};

	enum class native_format_class : u16
	{
		invalid,
		r8,
		rg8,
		rgba8,
		bgra8,
		packed16,
		r16,
		rg16,
		rgba16,
		r32,
		rgba32,
		bc1,
		bc2,
		bc3,
		depth16,
		depth32,
		depth32_stencil8,
	};

	struct native_format_description
	{
		u64 pixel_format = 0;
		u64 linear_format = 0;
		u64 srgb_format = 0;
		u64 snorm_format = 0;
		native_format_class compatibility_class = native_format_class::invalid;
		component_mapping components;
		u32 capabilities = format_capability_none;
		u32 conversion_flags = format_conversion_none;
		u32 source_format = 0;
		u16 bytes_per_block = 0;
		u16 source_bytes_per_block = 0;
		u8 block_width = 1;
		u8 block_height = 1;
		u8 source_block_width = 1;
		u8 source_block_height = 1;
		u8 texel_width = 0;
		u8 element_size = 0;
		u8 elements_per_texel = 0;
		u8 aspects = texture_aspect_none;
		bool native_storage = false;

		[[nodiscard]] explicit operator bool() const
		{
			return pixel_format != 0 && bytes_per_block != 0 && aspects != texture_aspect_none;
		}

		[[nodiscard]] bool requires_conversion() const
		{
			return conversion_flags != format_conversion_none;
		}
	};

	struct minification_filter
	{
		sampler_filter filter = sampler_filter::nearest;
		sampler_mip_filter mip_filter = sampler_mip_filter::none;
		bool sample_mipmaps = false;
	};

	struct primitive_mapping
	{
		primitive_topology topology = primitive_topology::triangle;
		bool requires_index_emulation = false;
	};

	enum class index_element_type : u8
	{
		u16,
		u32,
	};

	[[nodiscard]] native_format_description get_depth_surface_format(
		const device_info& device, rsx::surface_depth_format2 format);
	[[nodiscard]] native_format_description get_color_surface_format(
		const device_info& device, rsx::surface_color_format format);
	[[nodiscard]] native_format_description get_sampler_format(
		const device_info& device, u32 format, bool request_srgb = false, bool request_snorm = false);
	[[nodiscard]] native_format_description describe_native_format(u64 pixel_format);

	[[nodiscard]] u64 get_srgb_format(u64 linear_format);
	[[nodiscard]] u64 get_snorm_format(u64 unorm_format);
	[[nodiscard]] format_compatibility get_view_compatibility(u64 pixel_format);
	[[nodiscard]] bool formats_are_bitcast_compatible(
		const native_format_description& first, const native_format_description& second,
		bool preserve_depth_encoding = true);
	[[nodiscard]] bool formats_are_bitcast_compatible(const image& first, const image& second);

	[[nodiscard]] minification_filter get_min_filter(rsx::texture_minify_filter filter);
	[[nodiscard]] sampler_filter get_mag_filter(rsx::texture_magnify_filter filter);
	[[nodiscard]] border_color get_border_color(u32 color, u64 native_format = 0,
		u8 aspects = texture_aspect_color, bool integer = false);
	[[nodiscard]] sampler_address_mode get_wrap_mode(rsx::texture_wrap_mode mode);
	[[nodiscard]] f32 get_max_anisotropy(rsx::texture_max_anisotropy anisotropy);
	[[nodiscard]] std::array<component_swizzle, 4> get_component_mapping(u32 format);
	[[nodiscard]] primitive_mapping get_primitive_mapping(rsx::primitive_type primitive);
	[[nodiscard]] index_element_type get_index_type(rsx::index_array_type type);
}
