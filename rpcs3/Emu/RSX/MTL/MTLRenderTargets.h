#pragma once

#include "MTLResourceState.h"
#include "util/types.hpp"

#include <memory>

namespace rsx::metal
{
	class command_frame;
	class texture;

	struct clear_color
	{
		f32 red = 0.f;
		f32 green = 0.f;
		f32 blue = 0.f;
		f32 alpha = 1.f;
	};

	class drawable_render_target
	{
	public:
		drawable_render_target(command_frame& frame, texture& color_texture, u32 width, u32 height, clear_color color);
		~drawable_render_target();

		drawable_render_target(const drawable_render_target&) = delete;
		drawable_render_target& operator=(const drawable_render_target&) = delete;

		void* render_pass_descriptor_handle() const;
		const resource_barrier& color_barrier() const;
		u32 width() const;
		u32 height() const;

	private:
		struct drawable_render_target_impl;
		std::unique_ptr<drawable_render_target_impl> m_impl;
	};
}
