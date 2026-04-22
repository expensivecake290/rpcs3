#pragma once

#include "Emu/RSX/rsx_utils.h"
#include "MTLResourceState.h"
#include "util/types.hpp"

#include <array>
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

	struct clear_depth_stencil
	{
		b8 depth = false;
		b8 stencil = false;
		f64 depth_value = 1.;
		u32 stencil_value = 0;
	};

	void encode_clear_color_target(command_frame& frame, texture& color_texture, clear_color color);
	void encode_clear_depth_stencil_target(command_frame& frame, texture& depth_stencil_texture, clear_depth_stencil clear);

	struct draw_render_pass_attachments
	{
		std::array<texture*, rsx::limits::color_buffers_count> color_textures{};
		texture* depth_stencil_texture = nullptr;
		b8 stencil_attachment = false;
		u32 color_target_count = 0;
		u32 width = 0;
		u32 height = 0;
	};

	class draw_render_pass_descriptor
	{
	public:
		explicit draw_render_pass_descriptor(const draw_render_pass_attachments& attachments);
		~draw_render_pass_descriptor();

		draw_render_pass_descriptor(const draw_render_pass_descriptor&) = delete;
		draw_render_pass_descriptor& operator=(const draw_render_pass_descriptor&) = delete;

		void* handle() const;
		u32 color_target_count() const;
		b8 has_depth_stencil_target() const;
		b8 has_stencil_attachment() const;
		u32 width() const;
		u32 height() const;

	private:
		struct draw_render_pass_descriptor_impl;
		std::unique_ptr<draw_render_pass_descriptor_impl> m_impl;
	};

	class draw_render_encoder_scope
	{
	public:
		draw_render_encoder_scope(command_frame& frame, const draw_render_pass_attachments& attachments);
		~draw_render_encoder_scope();

		draw_render_encoder_scope(const draw_render_encoder_scope&) = delete;
		draw_render_encoder_scope& operator=(const draw_render_encoder_scope&) = delete;

		void* encoder_handle() const;
		void end_encoding();
		const resource_barrier& color_barrier(u32 index) const;
		const resource_barrier& depth_stencil_barrier() const;
		u32 color_target_count() const;
		b8 has_depth_stencil_target() const;
		b8 has_stencil_attachment() const;
		u32 width() const;
		u32 height() const;

	private:
		struct draw_render_encoder_scope_impl;
		std::unique_ptr<draw_render_encoder_scope_impl> m_impl;
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

	class depth_stencil_render_target
	{
	public:
		depth_stencil_render_target(command_frame& frame, texture& depth_stencil_texture, u32 width, u32 height, clear_depth_stencil clear);
		~depth_stencil_render_target();

		depth_stencil_render_target(const depth_stencil_render_target&) = delete;
		depth_stencil_render_target& operator=(const depth_stencil_render_target&) = delete;

		void* render_pass_descriptor_handle() const;
		const resource_barrier& barrier() const;
		u32 width() const;
		u32 height() const;

	private:
		struct depth_stencil_render_target_impl;
		std::unique_ptr<depth_stencil_render_target_impl> m_impl;
	};
}
