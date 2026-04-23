#include "stdafx.h"
#include "MTLShaderEntrypoint.h"

#include "MTLPipelineState.h"
#include "MTLMeshPipelinePlan.h"
#include "MTLShaderEntrypointBuilder.h"
#include "MTLShaderInterface.h"

#include "Emu/RSX/Program/RSXFragmentProgram.h"

namespace
{
	constexpr u32 pipeline_requirement(rsx::metal::pipeline_entry_requirement requirement)
	{
		return static_cast<u32>(requirement);
	}

	void apply_pipeline_entry_build_result(
		rsx::metal::translated_shader& shader,
		rsx::metal::pipeline_entry_build_result result)
	{
		rsx_log.trace("apply_pipeline_entry_build_result(stage=%u, id=%u, source_hash=0x%llx, requirement_mask=0x%x, available=%u)",
			static_cast<u32>(shader.stage), shader.id, result.source_hash, result.requirement_mask, static_cast<u32>(result.available));

		rsx::metal::validate_pipeline_entry_build_result(shader.stage, result);

		if (result.available)
		{
			if (!result.source_hash || result.entry_point.empty() || result.source.empty() || result.requirement_mask)
			{
				fmt::throw_exception("Metal pipeline entry build result is marked available but is incomplete");
			}
		}
		else if (!result.requirement_mask || result.error.empty())
		{
			fmt::throw_exception("Metal pipeline entry build result is gated but lacks requirements or an error");
		}

		shader.pipeline_source_hash = result.source_hash;
		shader.pipeline_entry_point = std::move(result.entry_point);
		shader.pipeline_source = std::move(result.source);
		shader.pipeline_cache_path = std::move(result.cache_path);
		shader.pipeline_entry_error = std::move(result.error);
		shader.pipeline_requirement_mask = result.requirement_mask;
		shader.pipeline_entry_available = result.available;
	}

	void validate_helper_shader(const rsx::metal::translated_shader& shader, rsx::metal::shader_stage expected_stage)
	{
		rsx_log.trace("validate_helper_shader(stage=%u, expected_stage=%u, id=%u)",
			static_cast<u32>(shader.stage), static_cast<u32>(expected_stage), shader.id);

		if (shader.stage != expected_stage)
		{
			fmt::throw_exception("Metal shader pipeline entry stage mismatch: stage=%u, expected=%u",
				static_cast<u32>(shader.stage), static_cast<u32>(expected_stage));
		}

		if (!shader.source_hash || shader.source.empty() || shader.entry_point.empty())
		{
			fmt::throw_exception("Metal shader pipeline entry requires a translated helper shader");
		}
	}

	void validate_vertex_pipeline_entry_layout(const rsx::metal::shader_interface_layout& layout)
	{
		rsx_log.trace("validate_vertex_pipeline_entry_layout(stage=%u, buffers=%u, context=%u, constants=%u, vertex_layout=%u, persistent=%u, volatile=%u, vertex_count=%u)",
			static_cast<u32>(layout.stage),
			layout.argument_table.max_buffers,
			layout.context_buffer_index,
			layout.constants_buffer_index,
			layout.vertex_layout_buffer_index,
			layout.persistent_vertex_buffer_index,
			layout.volatile_vertex_buffer_index,
			layout.vertex_buffer_count);

		if (layout.stage != rsx::metal::shader_stage::vertex)
		{
			fmt::throw_exception("Metal vertex pipeline entry layout requires a vertex-stage interface");
		}

		if (layout.context_buffer_index == rsx::metal::shader_binding_none ||
			layout.constants_buffer_index == rsx::metal::shader_binding_none)
		{
			fmt::throw_exception("Metal vertex pipeline entry layout requires context and constants buffer bindings");
		}

		if (layout.vertex_buffer_count)
		{
			fmt::throw_exception("Metal vertex pipeline entry layout requires packed vertex fetch bindings");
		}
	}

	void validate_fragment_pipeline_entry_layout(const rsx::metal::shader_interface_layout& layout)
	{
		rsx_log.trace("validate_fragment_pipeline_entry_layout(stage=%u, buffers=%u, context=%u, constants=%u, uses_stage_inputs=%u)",
			static_cast<u32>(layout.stage),
			layout.argument_table.max_buffers,
			layout.context_buffer_index,
			layout.constants_buffer_index,
			static_cast<u32>(layout.uses_stage_inputs));

		if (layout.stage != rsx::metal::shader_stage::fragment)
		{
			fmt::throw_exception("Metal fragment pipeline entry layout requires a fragment-stage interface");
		}

		if (layout.context_buffer_index == rsx::metal::shader_binding_none)
		{
			fmt::throw_exception("Metal fragment pipeline entry layout requires a context buffer binding");
		}

		if (layout.constants_buffer_index == rsx::metal::shader_binding_none)
		{
			fmt::throw_exception("Metal fragment pipeline entry layout requires a constants buffer binding");
		}

		if (!layout.uses_stage_inputs)
		{
			fmt::throw_exception("Metal fragment pipeline entry layout requires fragment stage input tracking");
		}
	}

	void validate_mesh_pipeline_entry_layout(const rsx::metal::shader_interface_layout& layout)
	{
		rsx_log.trace("validate_mesh_pipeline_entry_layout(stage=%u, buffers=%u, context=%u, constants=%u, mesh_grid=%u, vertex_count=%u)",
			static_cast<u32>(layout.stage),
			layout.argument_table.max_buffers,
			layout.context_buffer_index,
			layout.constants_buffer_index,
			static_cast<u32>(layout.uses_mesh_grid),
			layout.vertex_buffer_count);

		if (layout.stage != rsx::metal::shader_stage::mesh)
		{
			fmt::throw_exception("Metal mesh pipeline entry layout requires a mesh-stage interface");
		}

		if (layout.context_buffer_index == rsx::metal::shader_binding_none ||
			layout.constants_buffer_index == rsx::metal::shader_binding_none)
		{
			fmt::throw_exception("Metal mesh pipeline entry layout requires context and constants buffer bindings");
		}

		if (!layout.uses_mesh_grid)
		{
			fmt::throw_exception("Metal mesh pipeline entry layout requires mesh grid tracking");
		}

		if (layout.vertex_buffer_count)
		{
			fmt::throw_exception("Metal mesh pipeline entry layout requires packed vertex fetch bindings");
		}
	}

	const char* shader_stage_name(rsx::metal::shader_stage stage)
	{
		rsx_log.trace("shader_stage_name(stage=%u)", static_cast<u32>(stage));

		switch (stage)
		{
		case rsx::metal::shader_stage::vertex:
			return "vertex";
		case rsx::metal::shader_stage::fragment:
			return "fragment";
		case rsx::metal::shader_stage::mesh:
			return "mesh";
		}

		fmt::throw_exception("Unknown Metal shader stage %u", static_cast<u32>(stage));
	}
}

namespace rsx::metal
{
	void mark_vertex_pipeline_entry_status(translated_shader& shader)
	{
		rsx_log.notice("rsx::metal::mark_vertex_pipeline_entry_status(id=%u, source_hash=0x%llx)", shader.id, shader.source_hash);

		validate_helper_shader(shader, shader_stage::vertex);

		const u32 requirement_mask =
			pipeline_requirement(pipeline_entry_requirement::argument_table_shader_binding) |
			pipeline_requirement(pipeline_entry_requirement::vertex_input_fetch) |
			pipeline_requirement(pipeline_entry_requirement::viewport_depth_transform);

		const shader_interface_layout layout = make_vertex_shader_interface_layout();
		validate_vertex_pipeline_entry_layout(layout);
		rsx_log.notice("Metal vertex shader interface: %s", describe_shader_interface_layout(layout));
		rsx_log.notice("Metal vertex shader stage I/O: %s", describe_shader_stage_io_layout(layout));

		apply_pipeline_entry_build_result(shader, build_pipeline_entry_source(
			shader,
			layout,
			requirement_mask,
			"Metal vertex pipeline entry generation is gated until verified MSL argument-table shader binding syntax, GPU vertex fetch wrapper source, and viewport/depth output mapping are implemented"));
	}

	void mark_fragment_pipeline_entry_status(translated_shader& shader, const RSXFragmentProgram& program)
	{
		rsx_log.notice("rsx::metal::mark_fragment_pipeline_entry_status(id=%u, source_hash=0x%llx, mrt_count=%u, ctrl=0x%x)",
			shader.id, shader.source_hash, program.mrt_buffers_count, program.ctrl);

		validate_helper_shader(shader, shader_stage::fragment);

		std::string reason = "Metal fragment pipeline entry generation is gated until verified MSL argument-table shader binding syntax and fragment stage input/output wrapper generation are implemented";
		u32 requirement_mask =
			pipeline_requirement(pipeline_entry_requirement::argument_table_shader_binding) |
			pipeline_requirement(pipeline_entry_requirement::stage_input_layout);

		const shader_interface_layout layout = make_fragment_shader_interface_layout(0, 0);
		validate_fragment_pipeline_entry_layout(layout);
		rsx_log.notice("Metal fragment shader interface: %s", describe_shader_interface_layout(layout));
		rsx_log.notice("Metal fragment shader stage I/O: %s", describe_shader_stage_io_layout(layout));

		if (program.mrt_buffers_count > 1)
		{
			reason += "; MRT output mapping is not enabled";
			requirement_mask |= pipeline_requirement(pipeline_entry_requirement::mrt_output_mapping);
		}

		if (program.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT)
		{
			reason += "; depth export mapping is not enabled";
			requirement_mask |= pipeline_requirement(pipeline_entry_requirement::depth_export_mapping);
		}

		apply_pipeline_entry_build_result(shader, build_pipeline_entry_source(
			shader,
			layout,
			requirement_mask,
			std::move(reason)));
	}

	void mark_mesh_pipeline_entry_status(translated_shader& shader)
	{
		rsx_log.notice("rsx::metal::mark_mesh_pipeline_entry_status(id=%u, source_hash=0x%llx)", shader.id, shader.source_hash);

		validate_helper_shader(shader, shader_stage::mesh);

		const mesh_pipeline_plan plan = make_mesh_pipeline_plan();
		validate_mesh_pipeline_entry_layout(plan.interface_layout);
		rsx_log.notice("Metal mesh pipeline plan: %s", describe_mesh_pipeline_plan(plan));

		apply_pipeline_entry_build_result(shader, build_pipeline_entry_source(
			shader,
			plan.interface_layout,
			plan.requirement_mask,
			plan.gated_reason));
	}

	void report_shader_pipeline_entry_status(const translated_shader& shader)
	{
		rsx_log.notice("rsx::metal::report_shader_pipeline_entry_status(stage=%u, id=%u, source_hash=0x%llx, pipeline_source_hash=0x%llx, requirement_mask=0x%x)",
			static_cast<u32>(shader.stage), shader.id, shader.source_hash, shader.pipeline_source_hash, shader.pipeline_requirement_mask);

		const char* stage_name = shader_stage_name(shader.stage);
		const render_pipeline_shader pipeline_shader = make_render_pipeline_shader(shader);

		if (pipeline_shader.entry_available)
		{
			rsx_log.notice("Metal %s shader pipeline entry ready: entry=%s, source=%s",
				stage_name, pipeline_shader.entry_point.c_str(), shader.pipeline_cache_path.c_str());
			return;
		}

		if (pipeline_shader.entry_error.empty())
		{
			rsx_log.error("Metal %s shader pipeline entry is unavailable without a recorded reason", stage_name);
			return;
		}

		rsx_log.warning("Metal %s shader pipeline entry gated: requirements=0x%x (%s), %s",
			stage_name,
			pipeline_shader.requirement_mask,
			describe_pipeline_entry_requirements(pipeline_shader.requirement_mask),
			pipeline_shader.entry_error.c_str());
	}
}
