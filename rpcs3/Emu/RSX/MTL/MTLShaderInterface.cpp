#include "stdafx.h"
#include "MTLShaderInterface.h"

#include <array>
#include <string_view>

namespace
{
	constexpr u32 max_argument_table_buffers = 31;
	constexpr u32 max_argument_table_textures = 128;
	constexpr u32 max_argument_table_samplers = 16;

	constexpr u32 pipeline_requirement(rsx::metal::pipeline_entry_requirement requirement)
	{
		return static_cast<u32>(requirement);
	}

	u32 slot_index(const rsx::metal::shader_named_slot& slot)
	{
		return slot.index;
	}

	u32 slot_index(const rsx::metal::shader_vertex_input_slot& slot)
	{
		return slot.attribute_index;
	}

	template <typename Slot, std::size_t Count>
	std::string append_slots(std::string text, std::string_view label, const std::array<Slot, Count>& slots)
	{
		rsx_log.trace("append_slots(label=%s, count=%zu)", label, Count);

		if (!text.empty())
		{
			text += "; ";
		}

		text += label;
		text += "=";

		for (std::size_t index = 0; index < slots.size(); index++)
		{
			if (index)
			{
				text += ", ";
			}

			text += fmt::format("%u:%s", slot_index(slots[index]), std::string(slots[index].name));
		}

		return text;
	}

	void append_requirement(std::string& text, std::string_view name)
	{
		rsx_log.trace("append_requirement(name=%s)", name);

		if (!text.empty())
		{
			text += ", ";
		}

		text += name;
	}

	void validate_binding_range(const char* name, u32 base, u32 count, u32 max_count)
	{
		rsx_log.trace("validate_binding_range(name=%s, base=%u, count=%u, max_count=%u)",
			name, base, count, max_count);

		if (!count)
		{
			return;
		}

		if (base == rsx::metal::shader_binding_none || base >= max_count || count > max_count - base)
		{
			fmt::throw_exception("Metal shader interface binding range '%s' is invalid: base=%u, count=%u, max=%u",
				name, base, count, max_count);
		}
	}

	void validate_optional_buffer_binding(const char* name, u32 index, u32 max_count)
	{
		rsx_log.trace("validate_optional_buffer_binding(name=%s, index=%u, max_count=%u)",
			name, index, max_count);

		if (index == rsx::metal::shader_binding_none)
		{
			return;
		}

		if (index >= max_count)
		{
			fmt::throw_exception("Metal shader interface buffer binding '%s' is invalid: index=%u, max=%u",
				name, index, max_count);
		}
	}
}

namespace rsx::metal
{
	const std::array<shader_vertex_input_slot, shader_vertex_input_count>& vertex_input_slots()
	{
		rsx_log.trace("rsx::metal::vertex_input_slots()");

		static constexpr std::array<shader_vertex_input_slot, shader_vertex_input_count> s_slots =
		{{
			{ 0, 1, "in_pos" },
			{ 1, 2, "in_weight" },
			{ 2, 3, "in_normal" },
			{ 3, 4, "in_diff_color" },
			{ 4, 5, "in_spec_color" },
			{ 5, 6, "in_fog" },
			{ 6, 7, "in_point_size" },
			{ 7, 8, "in_7" },
			{ 8, 9, "in_tc0" },
			{ 9, 10, "in_tc1" },
			{ 10, 11, "in_tc2" },
			{ 11, 12, "in_tc3" },
			{ 12, 13, "in_tc4" },
			{ 13, 14, "in_tc5" },
			{ 14, 15, "in_tc6" },
			{ 15, 16, "in_tc7" },
		}};

		return s_slots;
	}

	const std::array<shader_named_slot, shader_vertex_output_count>& vertex_output_slots()
	{
		rsx_log.trace("rsx::metal::vertex_output_slots()");

		static constexpr std::array<shader_named_slot, shader_vertex_output_count> s_slots =
		{{
			{ 0, "dst_reg0" },
			{ 1, "dst_reg1" },
			{ 2, "dst_reg2" },
			{ 3, "dst_reg3" },
			{ 4, "dst_reg4" },
			{ 5, "dst_reg5" },
			{ 6, "dst_reg6" },
			{ 7, "dst_reg7" },
			{ 8, "dst_reg8" },
			{ 9, "dst_reg9" },
			{ 10, "dst_reg10" },
			{ 11, "dst_reg11" },
			{ 12, "dst_reg12" },
			{ 13, "dst_reg13" },
			{ 14, "dst_reg14" },
			{ 15, "dst_reg15" },
		}};

		return s_slots;
	}

	const std::array<shader_named_slot, shader_fragment_input_count>& fragment_input_slots()
	{
		rsx_log.trace("rsx::metal::fragment_input_slots()");

		static constexpr std::array<shader_named_slot, shader_fragment_input_count> s_slots =
		{{
			{ 0, "wpos" },
			{ 1, "diff_color" },
			{ 2, "spec_color" },
			{ 3, "fogc" },
			{ 4, "tc0" },
			{ 5, "tc1" },
			{ 6, "tc2" },
			{ 7, "tc3" },
			{ 8, "tc4" },
			{ 9, "tc5" },
			{ 10, "tc6" },
			{ 11, "tc7" },
			{ 12, "tc8" },
			{ 13, "tc9" },
			{ 14, "ssa" },
		}};

		return s_slots;
	}

	const std::array<shader_named_slot, shader_fragment_color_output_count>& fragment_color_output_slots()
	{
		rsx_log.trace("rsx::metal::fragment_color_output_slots()");

		static constexpr std::array<shader_named_slot, shader_fragment_color_output_count> s_slots =
		{{
			{ 0, "color0" },
			{ 1, "color1" },
			{ 2, "color2" },
			{ 3, "color3" },
		}};

		return s_slots;
	}

	shader_interface_layout make_vertex_shader_interface_layout()
	{
		rsx_log.trace("rsx::metal::make_vertex_shader_interface_layout()");

		shader_interface_layout layout =
		{
			.stage = shader_stage::vertex,
			.argument_table =
			{
				.label = "RPCS3 Metal vertex shader argument table",
				.max_buffers = 4,
				.max_textures = 0,
				.max_samplers = 0,
				.initialize_bindings = true,
				.support_attribute_strides = false,
			},
			.render_stage_mask = static_cast<u32>(argument_table_render_stage::vertex),
			.constants_buffer_index = 0,
			.vertex_layout_buffer_index = 1,
			.persistent_vertex_buffer_index = 2,
			.volatile_vertex_buffer_index = 3,
		};

		validate_shader_interface_layout(layout);
		return layout;
	}

	shader_interface_layout make_fragment_shader_interface_layout(u32 texture_count, u32 sampler_count)
	{
		rsx_log.trace("rsx::metal::make_fragment_shader_interface_layout(texture_count=%u, sampler_count=%u)",
			texture_count, sampler_count);

		if (texture_count != sampler_count)
		{
			fmt::throw_exception("Metal fragment shader interface requires matched texture and sampler counts");
		}

		shader_interface_layout layout =
		{
			.stage = shader_stage::fragment,
			.argument_table =
			{
				.label = "RPCS3 Metal fragment shader argument table",
				.max_buffers = 1,
				.max_textures = texture_count,
				.max_samplers = sampler_count,
				.initialize_bindings = true,
				.support_attribute_strides = false,
			},
			.render_stage_mask = static_cast<u32>(argument_table_render_stage::fragment),
			.constants_buffer_index = 0,
			.texture_base_index = texture_count ? 0 : shader_binding_none,
			.texture_count = texture_count,
			.sampler_base_index = sampler_count ? 0 : shader_binding_none,
			.sampler_count = sampler_count,
			.uses_stage_inputs = true,
		};

		validate_shader_interface_layout(layout);
		return layout;
	}

	shader_interface_layout make_mesh_shader_interface_layout()
	{
		rsx_log.trace("rsx::metal::make_mesh_shader_interface_layout()");

		shader_interface_layout layout =
		{
			.stage = shader_stage::mesh,
			.argument_table =
			{
				.label = "RPCS3 Metal mesh shader argument table",
				.max_buffers = 4,
				.max_textures = 0,
				.max_samplers = 0,
				.initialize_bindings = true,
				.support_attribute_strides = false,
			},
			.render_stage_mask =
				static_cast<u32>(argument_table_render_stage::object) |
				static_cast<u32>(argument_table_render_stage::mesh),
			.constants_buffer_index = 0,
			.vertex_layout_buffer_index = 1,
			.persistent_vertex_buffer_index = 2,
			.volatile_vertex_buffer_index = 3,
			.uses_mesh_grid = true,
		};

		validate_shader_interface_layout(layout);
		return layout;
	}

	void validate_shader_vertex_fetch_layout(const shader_interface_layout& layout)
	{
		rsx_log.trace("rsx::metal::validate_shader_vertex_fetch_layout(stage=%u, vertex_layout=%u, persistent=%u, volatile=%u, vertex_base=%u, vertex_count=%u)",
			static_cast<u32>(layout.stage),
			layout.vertex_layout_buffer_index,
			layout.persistent_vertex_buffer_index,
			layout.volatile_vertex_buffer_index,
			layout.vertex_buffer_base_index,
			layout.vertex_buffer_count);

		if (layout.stage != shader_stage::vertex && layout.stage != shader_stage::mesh)
		{
			if (layout.vertex_buffer_count ||
				layout.vertex_layout_buffer_index != shader_binding_none ||
				layout.persistent_vertex_buffer_index != shader_binding_none ||
				layout.volatile_vertex_buffer_index != shader_binding_none)
			{
				fmt::throw_exception("Metal shader interface stage %u cannot define vertex fetch inputs", static_cast<u32>(layout.stage));
			}

			return;
		}

		if (layout.vertex_layout_buffer_index == shader_binding_none ||
			layout.persistent_vertex_buffer_index == shader_binding_none ||
			layout.volatile_vertex_buffer_index == shader_binding_none)
		{
			fmt::throw_exception("Metal shader interface requires vertex layout, persistent stream, and volatile stream buffer slots");
		}

		if (layout.vertex_layout_buffer_index == layout.persistent_vertex_buffer_index ||
			layout.vertex_layout_buffer_index == layout.volatile_vertex_buffer_index ||
			layout.persistent_vertex_buffer_index == layout.volatile_vertex_buffer_index)
		{
			fmt::throw_exception("Metal shader interface vertex fetch buffer slots must be distinct");
		}

		validate_optional_buffer_binding("vertex layout", layout.vertex_layout_buffer_index, layout.argument_table.max_buffers);
		validate_optional_buffer_binding("persistent vertex stream", layout.persistent_vertex_buffer_index, layout.argument_table.max_buffers);
		validate_optional_buffer_binding("volatile vertex stream", layout.volatile_vertex_buffer_index, layout.argument_table.max_buffers);

		if (layout.vertex_buffer_count)
		{
			if (layout.vertex_buffer_count != shader_vertex_input_count)
			{
				fmt::throw_exception("Metal shader interface requires %u RSX vertex input slots when per-attribute fetch is enabled, found %u",
					shader_vertex_input_count, layout.vertex_buffer_count);
			}

			for (const shader_vertex_input_slot& slot : vertex_input_slots())
			{
				const u32 expected_buffer_index = layout.vertex_buffer_base_index + slot.attribute_index;

				if (slot.buffer_index != expected_buffer_index)
				{
					fmt::throw_exception("Metal shader vertex input slot %u has invalid buffer index: slot=%u, expected=%u",
						slot.attribute_index, slot.buffer_index, expected_buffer_index);
				}

				if (slot.buffer_index >= layout.argument_table.max_buffers)
				{
					fmt::throw_exception("Metal shader vertex input slot %u exceeds argument table buffers: buffer=%u, buffers=%u",
						slot.attribute_index, slot.buffer_index, layout.argument_table.max_buffers);
				}
			}
		}
	}

	void validate_shader_stage_io_layout(const shader_interface_layout& layout)
	{
		rsx_log.trace("rsx::metal::validate_shader_stage_io_layout(stage=%u, uses_stage_inputs=%u, uses_mesh_grid=%u)",
			static_cast<u32>(layout.stage), static_cast<u32>(layout.uses_stage_inputs), static_cast<u32>(layout.uses_mesh_grid));

		if (vertex_output_slots().size() != shader_vertex_output_count)
		{
			fmt::throw_exception("Metal shader interface vertex output slot table is invalid");
		}

		switch (layout.stage)
		{
		case shader_stage::vertex:
			return;
		case shader_stage::fragment:
			if (!layout.uses_stage_inputs)
			{
				fmt::throw_exception("Metal fragment shader interface requires stage input layout tracking");
			}

			if (fragment_input_slots().size() != shader_fragment_input_count ||
				fragment_color_output_slots().size() != shader_fragment_color_output_count)
			{
				fmt::throw_exception("Metal fragment shader interface slot tables are invalid");
			}

			return;
		case shader_stage::mesh:
			if (!layout.uses_mesh_grid)
			{
				fmt::throw_exception("Metal mesh shader interface requires mesh grid layout tracking");
			}

			return;
		}

		fmt::throw_exception("Unknown Metal shader stage %u", static_cast<u32>(layout.stage));
	}

	void validate_shader_interface_layout(const shader_interface_layout& layout)
	{
		rsx_log.trace("rsx::metal::validate_shader_interface_layout(stage=%u, buffers=%u, textures=%u, samplers=%u)",
			static_cast<u32>(layout.stage),
			layout.argument_table.max_buffers,
			layout.argument_table.max_textures,
			layout.argument_table.max_samplers);

		if (!layout.render_stage_mask)
		{
			fmt::throw_exception("Metal shader interface requires a render stage mask");
		}

		if (layout.argument_table.max_buffers > max_argument_table_buffers ||
			layout.argument_table.max_textures > max_argument_table_textures ||
			layout.argument_table.max_samplers > max_argument_table_samplers)
		{
			fmt::throw_exception("Metal shader interface exceeds MTL4ArgumentTable limits: buffers=%u, textures=%u, samplers=%u",
				layout.argument_table.max_buffers,
				layout.argument_table.max_textures,
				layout.argument_table.max_samplers);
		}

		if (layout.constants_buffer_index != shader_binding_none &&
			layout.constants_buffer_index >= layout.argument_table.max_buffers)
		{
			fmt::throw_exception("Metal shader interface constants buffer index is out of range: index=%u, buffers=%u",
				layout.constants_buffer_index, layout.argument_table.max_buffers);
		}

		validate_optional_buffer_binding("vertex layout", layout.vertex_layout_buffer_index, layout.argument_table.max_buffers);
		validate_optional_buffer_binding("persistent vertex stream", layout.persistent_vertex_buffer_index, layout.argument_table.max_buffers);
		validate_optional_buffer_binding("volatile vertex stream", layout.volatile_vertex_buffer_index, layout.argument_table.max_buffers);
		validate_binding_range("vertex buffers", layout.vertex_buffer_base_index, layout.vertex_buffer_count, layout.argument_table.max_buffers);
		validate_binding_range("textures", layout.texture_base_index, layout.texture_count, layout.argument_table.max_textures);
		validate_binding_range("samplers", layout.sampler_base_index, layout.sampler_count, layout.argument_table.max_samplers);

		if (layout.vertex_buffer_count && !layout.argument_table.support_attribute_strides)
		{
			fmt::throw_exception("Metal shader interface vertex buffers require attribute stride support");
		}

		validate_shader_vertex_fetch_layout(layout);
		validate_shader_stage_io_layout(layout);
	}

	std::string describe_shader_interface_layout(const shader_interface_layout& layout)
	{
		rsx_log.trace("rsx::metal::describe_shader_interface_layout(stage=%u)", static_cast<u32>(layout.stage));

		return fmt::format("buffers=%u, textures=%u, samplers=%u, stages=0x%x, constants=%u, vertex_layout=%u, persistent_vertex=%u, volatile_vertex=%u, vertex_base=%u, vertex_count=%u",
			layout.argument_table.max_buffers,
			layout.argument_table.max_textures,
			layout.argument_table.max_samplers,
			layout.render_stage_mask,
			layout.constants_buffer_index,
			layout.vertex_layout_buffer_index,
			layout.persistent_vertex_buffer_index,
			layout.volatile_vertex_buffer_index,
			layout.vertex_buffer_base_index,
			layout.vertex_buffer_count);
	}

	std::string describe_shader_stage_io_layout(const shader_interface_layout& layout)
	{
		rsx_log.trace("rsx::metal::describe_shader_stage_io_layout(stage=%u)", static_cast<u32>(layout.stage));

		switch (layout.stage)
		{
		case shader_stage::vertex:
		{
			std::string text = append_slots({}, "vertex_inputs", vertex_input_slots());
			return append_slots(text, "vertex_outputs", vertex_output_slots());
		}
		case shader_stage::fragment:
		{
			std::string text = append_slots({}, "fragment_inputs", fragment_input_slots());
			return append_slots(text, "fragment_color_outputs", fragment_color_output_slots());
		}
		case shader_stage::mesh:
		{
			std::string text = append_slots({}, "mesh_vertex_inputs", vertex_input_slots());
			text = append_slots(text, "mesh_outputs", vertex_output_slots());
			return fmt::format("%s; mesh_grid=%s", text, layout.uses_mesh_grid ? "tracked" : "missing");
		}
		}

		fmt::throw_exception("Unknown Metal shader stage %u", static_cast<u32>(layout.stage));
	}

	u32 known_pipeline_entry_requirement_mask()
	{
		rsx_log.trace("rsx::metal::known_pipeline_entry_requirement_mask()");

		return
			pipeline_requirement(pipeline_entry_requirement::argument_table_shader_binding) |
			pipeline_requirement(pipeline_entry_requirement::vertex_input_fetch) |
			pipeline_requirement(pipeline_entry_requirement::viewport_depth_transform) |
			pipeline_requirement(pipeline_entry_requirement::stage_input_layout) |
			pipeline_requirement(pipeline_entry_requirement::mrt_output_mapping) |
			pipeline_requirement(pipeline_entry_requirement::depth_export_mapping) |
			pipeline_requirement(pipeline_entry_requirement::mesh_object_mapping) |
			pipeline_requirement(pipeline_entry_requirement::mesh_grid_mapping);
	}

	u32 pipeline_entry_requirement_mask_for_stage(shader_stage stage)
	{
		rsx_log.trace("rsx::metal::pipeline_entry_requirement_mask_for_stage(stage=%u)", static_cast<u32>(stage));

		constexpr u32 argument_table_mask = pipeline_requirement(pipeline_entry_requirement::argument_table_shader_binding);

		switch (stage)
		{
		case shader_stage::vertex:
			return argument_table_mask |
				pipeline_requirement(pipeline_entry_requirement::vertex_input_fetch) |
				pipeline_requirement(pipeline_entry_requirement::viewport_depth_transform);
		case shader_stage::fragment:
			return argument_table_mask |
				pipeline_requirement(pipeline_entry_requirement::stage_input_layout) |
				pipeline_requirement(pipeline_entry_requirement::mrt_output_mapping) |
				pipeline_requirement(pipeline_entry_requirement::depth_export_mapping);
		case shader_stage::mesh:
			return argument_table_mask |
				pipeline_requirement(pipeline_entry_requirement::mesh_object_mapping) |
				pipeline_requirement(pipeline_entry_requirement::mesh_grid_mapping);
		}

		fmt::throw_exception("Unknown Metal shader stage %u", static_cast<u32>(stage));
	}

	void validate_pipeline_entry_requirement_mask(u32 requirement_mask)
	{
		rsx_log.trace("rsx::metal::validate_pipeline_entry_requirement_mask(requirement_mask=0x%x)", requirement_mask);

		const u32 unknown_mask = requirement_mask & ~known_pipeline_entry_requirement_mask();
		if (unknown_mask)
		{
			fmt::throw_exception("Metal pipeline entry requirement mask contains unknown bits: requirement_mask=0x%x, unknown=0x%x",
				requirement_mask,
				unknown_mask);
		}
	}

	void validate_pipeline_entry_requirement_mask_for_stage(shader_stage stage, u32 requirement_mask)
	{
		rsx_log.trace("rsx::metal::validate_pipeline_entry_requirement_mask_for_stage(stage=%u, requirement_mask=0x%x)",
			static_cast<u32>(stage), requirement_mask);

		validate_pipeline_entry_requirement_mask(requirement_mask);

		const u32 invalid_mask = requirement_mask & ~pipeline_entry_requirement_mask_for_stage(stage);
		if (invalid_mask)
		{
			fmt::throw_exception("Metal pipeline entry requirement mask contains invalid stage bits: stage=%u, requirement_mask=0x%x, invalid=0x%x",
				static_cast<u32>(stage),
				requirement_mask,
				invalid_mask);
		}
	}

	std::string describe_pipeline_entry_requirements(u32 requirement_mask)
	{
		rsx_log.trace("rsx::metal::describe_pipeline_entry_requirements(requirement_mask=0x%x)", requirement_mask);

		validate_pipeline_entry_requirement_mask(requirement_mask);

		if (!requirement_mask)
		{
			return "none";
		}

		std::string text;

		if (requirement_mask & pipeline_requirement(pipeline_entry_requirement::argument_table_shader_binding))
		{
			append_requirement(text, "argument-table shader binding");
		}

		if (requirement_mask & pipeline_requirement(pipeline_entry_requirement::vertex_input_fetch))
		{
			append_requirement(text, "vertex input fetch");
		}

		if (requirement_mask & pipeline_requirement(pipeline_entry_requirement::viewport_depth_transform))
		{
			append_requirement(text, "viewport/depth transform");
		}

		if (requirement_mask & pipeline_requirement(pipeline_entry_requirement::stage_input_layout))
		{
			append_requirement(text, "stage input layout");
		}

		if (requirement_mask & pipeline_requirement(pipeline_entry_requirement::mrt_output_mapping))
		{
			append_requirement(text, "MRT output mapping");
		}

		if (requirement_mask & pipeline_requirement(pipeline_entry_requirement::depth_export_mapping))
		{
			append_requirement(text, "depth export mapping");
		}

		if (requirement_mask & pipeline_requirement(pipeline_entry_requirement::mesh_object_mapping))
		{
			append_requirement(text, "mesh/object shader mapping");
		}

		if (requirement_mask & pipeline_requirement(pipeline_entry_requirement::mesh_grid_mapping))
		{
			append_requirement(text, "mesh grid layout");
		}

		return text;
	}
}
