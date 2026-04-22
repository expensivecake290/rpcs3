#pragma once

#include "Emu/RSX/Core/RSXFrameBuffer.h"
#include "MTLRenderTargets.h"
#include "util/types.hpp"

#include <memory>

namespace rsx::metal
{
	class command_queue;
	class device;
	class texture;

	struct render_target_cache_stats
	{
		u32 color_target_count = 0;
		u32 color_target_create_count = 0;
		u32 color_target_reuse_count = 0;
		u32 color_target_clear_count = 0;
		u32 depth_stencil_target_count = 0;
		u32 depth_stencil_target_create_count = 0;
		u32 depth_stencil_target_reuse_count = 0;
		u32 depth_stencil_target_clear_count = 0;
		u32 draw_framebuffer_prepare_count = 0;
		u32 draw_color_target_bind_count = 0;
		u32 draw_depth_stencil_target_bind_count = 0;
	};

	using draw_target_binding = draw_render_pass_attachments;

	class render_target_cache
	{
	public:
		explicit render_target_cache(device& metal_device);
		~render_target_cache();

		render_target_cache(const render_target_cache&) = delete;
		render_target_cache& operator=(const render_target_cache&) = delete;

		texture& get_color_target(const rsx::framebuffer_layout& layout, u32 surface_index);
		texture& get_depth_stencil_target(const rsx::framebuffer_layout& layout);
		u32 get_color_target_metal_pixel_format(const rsx::framebuffer_layout& layout, u32 surface_index) const;
		draw_target_binding prepare_draw_targets(const rsx::framebuffer_layout& layout);
		void clear_color_target(command_queue& queue, const rsx::framebuffer_layout& layout, u32 surface_index, clear_color color);
		void clear_depth_stencil_target(command_queue& queue, const rsx::framebuffer_layout& layout, clear_depth_stencil clear);
		render_target_cache_stats stats() const;
		void report() const;

	private:
		struct render_target_cache_impl;
		std::unique_ptr<render_target_cache_impl> m_impl;
	};
}
