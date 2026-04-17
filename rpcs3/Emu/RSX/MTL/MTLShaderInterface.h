#pragma once

#include "MTLArgumentTable.h"
#include "MTLShaderRecompiler.h"

#include "util/types.hpp"

#include <array>
#include <string>
#include <string_view>

namespace rsx::metal
{
	constexpr u32 shader_binding_none = ~0u;
	constexpr u32 shader_vertex_input_count = 16;
	constexpr u32 shader_vertex_output_count = 16;
	constexpr u32 shader_fragment_input_count = 15;
	constexpr u32 shader_fragment_color_output_count = 4;

	struct shader_named_slot
	{
		u32 index = 0;
		std::string_view name;
	};

	struct shader_vertex_input_slot
	{
		u32 attribute_index = 0;
		u32 buffer_index = shader_binding_none;
		std::string_view name;
	};

	struct shader_interface_layout
	{
		shader_stage stage = shader_stage::vertex;
		argument_table_desc argument_table;
		u32 render_stage_mask = 0;
		u32 constants_buffer_index = shader_binding_none;
		u32 vertex_buffer_base_index = shader_binding_none;
		u32 vertex_buffer_count = 0;
		u32 texture_base_index = shader_binding_none;
		u32 texture_count = 0;
		u32 sampler_base_index = shader_binding_none;
		u32 sampler_count = 0;
		b8 uses_stage_inputs = false;
		b8 uses_mesh_grid = false;
	};

	shader_interface_layout make_vertex_shader_interface_layout();
	shader_interface_layout make_fragment_shader_interface_layout(u32 texture_count, u32 sampler_count);
	shader_interface_layout make_mesh_shader_interface_layout();
	void validate_shader_interface_layout(const shader_interface_layout& layout);
	void validate_shader_vertex_fetch_layout(const shader_interface_layout& layout);
	void validate_shader_stage_io_layout(const shader_interface_layout& layout);
	const std::array<shader_vertex_input_slot, shader_vertex_input_count>& vertex_input_slots();
	const std::array<shader_named_slot, shader_vertex_output_count>& vertex_output_slots();
	const std::array<shader_named_slot, shader_fragment_input_count>& fragment_input_slots();
	const std::array<shader_named_slot, shader_fragment_color_output_count>& fragment_color_output_slots();
	std::string describe_shader_interface_layout(const shader_interface_layout& layout);
	std::string describe_shader_stage_io_layout(const shader_interface_layout& layout);
	std::string describe_pipeline_entry_requirements(u32 requirement_mask);
}
