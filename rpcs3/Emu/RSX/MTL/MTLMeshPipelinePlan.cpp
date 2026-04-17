#include "stdafx.h"
#include "MTLMeshPipelinePlan.h"

namespace
{
	constexpr u32 pipeline_requirement(rsx::metal::pipeline_entry_requirement requirement)
	{
		return static_cast<u32>(requirement);
	}
}

namespace rsx::metal
{
	mesh_pipeline_plan make_mesh_pipeline_plan()
	{
		rsx_log.notice("rsx::metal::make_mesh_pipeline_plan()");

		mesh_pipeline_plan plan =
		{
			.interface_layout = make_mesh_shader_interface_layout(),
			.requirement_mask =
				pipeline_requirement(pipeline_entry_requirement::argument_table_shader_binding) |
				pipeline_requirement(pipeline_entry_requirement::mesh_object_mapping) |
				pipeline_requirement(pipeline_entry_requirement::mesh_grid_mapping),
			.gated_reason = "Metal mesh pipeline entry generation is gated until RSX-to-MSL mesh/object shader mapping, mesh grid dispatch layout, and argument-table shader binding are implemented",
		};

		validate_mesh_pipeline_plan(plan);
		return plan;
	}

	void validate_mesh_pipeline_plan(const mesh_pipeline_plan& plan)
	{
		rsx_log.trace("rsx::metal::validate_mesh_pipeline_plan(stage=%u, requirement_mask=0x%x)",
			static_cast<u32>(plan.interface_layout.stage), plan.requirement_mask);

		if (plan.interface_layout.stage != shader_stage::mesh)
		{
			fmt::throw_exception("Metal mesh pipeline plan requires a mesh shader interface");
		}

		validate_shader_interface_layout(plan.interface_layout);

		if (!plan.interface_layout.uses_mesh_grid)
		{
			fmt::throw_exception("Metal mesh pipeline plan requires mesh grid layout tracking");
		}

		if (!plan.requirement_mask)
		{
			fmt::throw_exception("Metal mesh pipeline plan cannot be marked available without verified MSL mesh entry generation");
		}

		if (plan.gated_reason.empty())
		{
			fmt::throw_exception("Metal mesh pipeline plan requires a gated diagnostic reason");
		}

		constexpr u32 argument_table_mask = pipeline_requirement(pipeline_entry_requirement::argument_table_shader_binding);
		constexpr u32 mesh_object_mask = pipeline_requirement(pipeline_entry_requirement::mesh_object_mapping);
		constexpr u32 mesh_grid_mask = pipeline_requirement(pipeline_entry_requirement::mesh_grid_mapping);

		if (plan.requires_argument_table_binding != !!(plan.requirement_mask & argument_table_mask))
		{
			fmt::throw_exception("Metal mesh pipeline plan argument-table requirement mismatch");
		}

		if (plan.requires_mesh_object_mapping != !!(plan.requirement_mask & mesh_object_mask))
		{
			fmt::throw_exception("Metal mesh pipeline plan object mapping requirement mismatch");
		}

		if (plan.requires_mesh_grid_mapping != !!(plan.requirement_mask & mesh_grid_mask))
		{
			fmt::throw_exception("Metal mesh pipeline plan grid mapping requirement mismatch");
		}
	}

	std::string describe_mesh_pipeline_plan(const mesh_pipeline_plan& plan)
	{
		rsx_log.trace("rsx::metal::describe_mesh_pipeline_plan(requirement_mask=0x%x)", plan.requirement_mask);

		validate_mesh_pipeline_plan(plan);

		return fmt::format("requirements=0x%x (%s); interface={%s}; stage_io={%s}; %s",
			plan.requirement_mask,
			describe_pipeline_entry_requirements(plan.requirement_mask),
			describe_shader_interface_layout(plan.interface_layout),
			describe_shader_stage_io_layout(plan.interface_layout),
			plan.gated_reason);
	}
}
