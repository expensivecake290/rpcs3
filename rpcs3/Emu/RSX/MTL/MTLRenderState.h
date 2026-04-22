#pragma once

#include "Emu/RSX/gcm_enums.h"
#include "util/types.hpp"

#include <memory>

namespace rsx::metal
{
	class command_frame;
	class device;

	struct render_viewport
	{
		f64 x = 0.;
		f64 y = 0.;
		f64 width = 0.;
		f64 height = 0.;
		f64 z_near = 0.;
		f64 z_far = 1.;
	};

	struct render_scissor
	{
		u32 x = 0;
		u32 y = 0;
		u32 width = 0;
		u32 height = 0;
	};

	struct dynamic_render_state_desc
	{
		render_viewport viewport{};
		render_scissor scissor{};
		u32 render_width = 0;
		u32 render_height = 0;
		rsx::primitive_type primitive = rsx::primitive_type::triangles;
		rsx::front_face front_face = rsx::front_face::ccw;
		rsx::cull_face cull_face = rsx::cull_face::back;
		rsx::polygon_mode front_polygon_mode = rsx::polygon_mode::fill;
		rsx::polygon_mode back_polygon_mode = rsx::polygon_mode::fill;
		rsx::comparison_function depth_func = rsx::comparison_function::always;
		rsx::comparison_function stencil_func = rsx::comparison_function::always;
		rsx::comparison_function back_stencil_func = rsx::comparison_function::always;
		rsx::stencil_op stencil_fail = rsx::stencil_op::keep;
		rsx::stencil_op stencil_zfail = rsx::stencil_op::keep;
		rsx::stencil_op stencil_zpass = rsx::stencil_op::keep;
		rsx::stencil_op back_stencil_fail = rsx::stencil_op::keep;
		rsx::stencil_op back_stencil_zfail = rsx::stencil_op::keep;
		rsx::stencil_op back_stencil_zpass = rsx::stencil_op::keep;
		u32 stencil_ref = 0;
		u32 back_stencil_ref = 0;
		u32 stencil_read_mask = 0xff;
		u32 back_stencil_read_mask = 0xff;
		u32 stencil_write_mask = 0xff;
		u32 back_stencil_write_mask = 0xff;
		f32 depth_bounds_min = 0.f;
		f32 depth_bounds_max = 1.f;
		f32 blend_color_red = 0.f;
		f32 blend_color_green = 0.f;
		f32 blend_color_blue = 0.f;
		f32 blend_color_alpha = 0.f;
		f32 polygon_offset_scale = 0.f;
		f32 polygon_offset_bias = 0.f;
		f32 line_width = 1.f;
		b8 has_depth_stencil_target = false;
		b8 has_stencil_attachment = false;
		b8 cull_face_enabled = false;
		b8 depth_clip_enabled = true;
		b8 depth_clamp_enabled = false;
		b8 depth_bounds_test_enabled = false;
		b8 depth_test_enabled = false;
		b8 depth_write_enabled = false;
		b8 stencil_test_enabled = false;
		b8 two_sided_stencil_test_enabled = false;
		b8 polygon_offset_fill_enabled = false;
		b8 polygon_offset_line_enabled = false;
		b8 polygon_offset_point_enabled = false;
		b8 blend_enabled = false;
		b8 logic_op_enabled = false;
		b8 alpha_test_enabled = false;
		b8 dither_enabled = false;
		b8 line_smooth_enabled = false;
		b8 poly_smooth_enabled = false;
		b8 polygon_stipple_enabled = false;
		b8 color_write_all = true;
	};

	class render_state_cache
	{
	public:
		explicit render_state_cache(device& dev);
		~render_state_cache();

		render_state_cache(const render_state_cache&) = delete;
		render_state_cache& operator=(const render_state_cache&) = delete;

		void bind_dynamic_render_state(command_frame& frame, void* render_encoder_handle, const dynamic_render_state_desc& desc);
		u32 retained_depth_stencil_state_count() const;

	private:
		void* get_depth_stencil_state_handle(const dynamic_render_state_desc& desc);

		struct render_state_cache_impl;
		std::unique_ptr<render_state_cache_impl> m_impl;
	};
}
