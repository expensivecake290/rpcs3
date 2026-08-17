#pragma once

#include <array>
#include <span>

#include "buffer_object.h"
#include "commands.h"
#include "image.h"

namespace rsx
{
	struct texture_channel_remap_t;
}

namespace mtl
{
	struct texture_origin
	{
		u32 x = 0;
		u32 y = 0;
		u32 z = 0;

		[[nodiscard]] bool operator==(const texture_origin&) const = default;
	};

	struct texture_extent
	{
		u32 width = 1;
		u32 height = 1;
		u32 depth = 1;

		[[nodiscard]] bool operator==(const texture_extent&) const = default;
	};

	struct texture_subresource
	{
		u32 mip_level = 0;
		u32 array_slice = 0;
		u8 aspects = texture_aspect_color;

		[[nodiscard]] bool operator==(const texture_subresource&) const = default;
	};

	struct image_copy_region
	{
		texture_subresource source;
		texture_subresource destination;
		texture_origin source_origin;
		texture_origin destination_origin;
		texture_extent extent;
		u32 layer_count = 1;
	};

	struct buffer_image_copy_region
	{
		u64 buffer_offset = 0;
		u64 bytes_per_row = 0;
		u64 bytes_per_image = 0;
		texture_subresource subresource;
		texture_origin origin;
		texture_extent extent;
		u32 layer_count = 1;
	};

	struct signed_texture_box
	{
		s32 x0 = 0;
		s32 y0 = 0;
		s32 z0 = 0;
		s32 x1 = 1;
		s32 y1 = 1;
		s32 z1 = 1;

		[[nodiscard]] bool operator==(const signed_texture_box&) const = default;
	};

	struct image_scale_region
	{
		texture_subresource source;
		texture_subresource destination;
		signed_texture_box source_box;
		signed_texture_box destination_box;
		u32 layer_count = 1;
	};

	enum class image_filter : u8
	{
		nearest,
		linear,
	};

	enum class image_conversion_kind : u8
	{
		none,
		channel_remap,
		color_to_color,
		depth_to_color,
		color_to_depth,
	};

	struct image_conversion
	{
		image_conversion_kind kind = image_conversion_kind::none;
		component_mapping channels;
		bool decode_srgb = false;
		bool encode_srgb = false;
		bool premultiply_alpha = false;
		bool unpremultiply_alpha = false;
	};

	extern const component_mapping default_component_map;

	[[nodiscard]] u8 get_aspect_flags(u64 pixel_format);
	[[nodiscard]] component_mapping apply_swizzle_remap(
		const std::array<component_swizzle, 4>& base_remap,
		const rsx::texture_channel_remap_t& remap_vector);

	void transition_image(command_buffer& command, image& resource, const image_state& next,
		const subresource_range& range, bool preserve_encoder = false);
	void transition_image(command_buffer& command, image& resource, const image_state& next,
		bool preserve_encoder = false);

	void copy_image(command_buffer& command, image& source, image& destination,
		std::span<const image_copy_region> regions);
	void upload_image(command_buffer& command, const buffer& source, image& destination,
		std::span<const buffer_image_copy_region> regions);
	void download_image(command_buffer& command, image& source, buffer& destination,
		std::span<const buffer_image_copy_region> regions);
	void scale_image(command_buffer& command, image& source, image& destination,
		const image_scale_region& region, image_filter filter,
		const image_conversion& conversion = {});
	void convert_image(command_buffer& command, image& source, image& destination,
		const image_scale_region& region, const image_conversion& conversion);
	void generate_mipmaps(command_buffer& command, image& resource);
}
