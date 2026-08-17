#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "barriers.h"
#include "memory.h"
#include "unique_resource.h"

namespace mtl
{
	enum class texture_type : u8
	{
		texture_1d,
		texture_1d_array,
		texture_2d,
		texture_2d_array,
		texture_2d_multisample,
		texture_2d_multisample_array,
		texture_3d,
		texture_cube,
		texture_cube_array,
	};

	enum texture_usage : u32
	{
		texture_usage_none = 0,
		texture_usage_shader_read = 1 << 0,
		texture_usage_shader_write = 1 << 1,
		texture_usage_render_target = 1 << 2,
		texture_usage_depth_stencil = 1 << 3,
		texture_usage_copy_source = 1 << 4,
		texture_usage_copy_destination = 1 << 5,
		texture_usage_pixel_format_view = 1 << 6,
	};

	enum texture_aspect : u8
	{
		texture_aspect_none = 0,
		texture_aspect_color = 1 << 0,
		texture_aspect_depth = 1 << 1,
		texture_aspect_stencil = 1 << 2,
	};

	enum class component_swizzle : u8
	{
		zero,
		one,
		red,
		green,
		blue,
		alpha,
	};

	struct component_mapping
	{
		component_swizzle red = component_swizzle::red;
		component_swizzle green = component_swizzle::green;
		component_swizzle blue = component_swizzle::blue;
		component_swizzle alpha = component_swizzle::alpha;

		[[nodiscard]] bool operator==(const component_mapping&) const = default;
		[[nodiscard]] u32 encode() const;
	};

	struct image_create_info
	{
		texture_type type = texture_type::texture_2d;
		format_compatibility formats;
		u32 width = 1;
		u32 height = 1;
		u32 depth = 1;
		u32 mip_levels = 1;
		u32 array_layers = 1;
		u32 sample_count = 1;
		u32 usage = texture_usage_shader_read;
		u8 aspects = texture_aspect_color;
		storage_mode storage = storage_mode::private_;
		hazard_tracking hazards = hazard_tracking::tracked;
		allocation_pool pool = allocation_pool::texture_cache;
		std::string label;
		bool allow_failure = false;
		bool recover_on_failure = true;
		bool use_placement_heap = true;
		bool shareable = false;
	};

	struct image_state
	{
		queue_kind queue = queue_kind::graphics;
		u64 stages = stage_none;
		u64 access = access_none;
		u64 submission = 0;
		bool initialized = false;
	};

	class image : public unique_resource
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	protected:
		image();

	public:
		image(memory_allocator& allocator, const image_create_info& info);
		virtual ~image();
		image(const image&) = delete;
		image& operator=(const image&) = delete;
		image(image&&) = delete;
		image& operator=(image&&) = delete;

		void create(memory_allocator& allocator, const image_create_info& info);
		void wrap(texture_handle texture, const image_create_info& info);
		void destroy();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] texture_handle native_handle() const;
		[[nodiscard]] const image_create_info& info() const;
		[[nodiscard]] u32 width() const;
		[[nodiscard]] u32 height() const;
		[[nodiscard]] u32 depth() const;
		[[nodiscard]] u32 mipmaps() const;
		[[nodiscard]] u32 layers() const;
		[[nodiscard]] u32 samples() const;
		[[nodiscard]] u64 format() const;
		[[nodiscard]] texture_type type() const;
		[[nodiscard]] u8 aspects() const;
		[[nodiscard]] storage_mode storage() const;
		[[nodiscard]] bool is_memoryless() const;
		[[nodiscard]] bool is_shareable() const;
		[[nodiscard]] const memory_allocation& allocation() const;

		void set_debug_name(std::string_view name);
		[[nodiscard]] const std::string& debug_name() const;
		[[nodiscard]] image_state state() const;
		void set_state(const image_state& state);
		[[nodiscard]] hazard transition_hazard(const image_state& next, const subresource_range& range, bool preserve_encoder) const;
	};

	class image_view : public unique_resource
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		image_view();
		image_view(const image& resource, u64 format, texture_type type, component_mapping mapping, subresource_range range);
		~image_view();
		image_view(const image_view&) = delete;
		image_view& operator=(const image_view&) = delete;
		image_view(image_view&&) = delete;
		image_view& operator=(image_view&&) = delete;

		void create(const image& resource, u64 format, texture_type type, component_mapping mapping, subresource_range range);
		void destroy();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] texture_handle native_handle() const;
		[[nodiscard]] mtl::image* image() const;
		[[nodiscard]] u64 image_uid() const;
		[[nodiscard]] u64 format() const;
		[[nodiscard]] texture_type type() const;
		[[nodiscard]] component_mapping mapping() const;
		[[nodiscard]] u32 encoded_component_map() const;
		[[nodiscard]] subresource_range range() const;
	};

	class viewable_image : public image
	{
		std::unordered_map<u64, std::unique_ptr<image_view>> m_views;

	public:
		using image::image;

		void create(memory_allocator& allocator, const image_create_info& info);
		void wrap(texture_handle texture, const image_create_info& info);
		void destroy();

		[[nodiscard]] image_view* get_view(
			u64 format,
			texture_type type,
			component_mapping mapping,
			subresource_range range);
		void clear_views();
	};
}
