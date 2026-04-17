#include "stdafx.h"
#include "MTLShaderEntrypointBuilder.h"

#include <utility>

namespace
{
	void validate_pipeline_entry_request(
		const rsx::metal::translated_shader& shader,
		const rsx::metal::shader_interface_layout& layout)
	{
		rsx_log.trace("validate_pipeline_entry_request(shader_stage=%u, layout_stage=%u, id=%u)",
			static_cast<u32>(shader.stage), static_cast<u32>(layout.stage), shader.id);

		if (shader.stage != layout.stage)
		{
			fmt::throw_exception("Metal pipeline entry request stage mismatch: shader=%u, layout=%u",
				static_cast<u32>(shader.stage), static_cast<u32>(layout.stage));
		}

		if (!shader.source_hash || shader.entry_point.empty() || shader.source.empty())
		{
			fmt::throw_exception("Metal pipeline entry request requires a translated helper shader");
		}

		rsx::metal::validate_shader_interface_layout(layout);
	}
}

namespace rsx::metal
{
	pipeline_entry_build_result build_pipeline_entry_source(
		const translated_shader& shader,
		const shader_interface_layout& layout,
		u32 requirement_mask,
		std::string reason)
	{
		rsx_log.notice("rsx::metal::build_pipeline_entry_source(stage=%u, id=%u, source_hash=0x%llx, requirement_mask=0x%x)",
			static_cast<u32>(shader.stage), shader.id, shader.source_hash, requirement_mask);

		validate_pipeline_entry_request(shader, layout);

		if (!requirement_mask)
		{
			fmt::throw_exception("Metal pipeline entry source generation requires verified MSL argument-table shader binding syntax");
		}

		if (reason.empty())
		{
			fmt::throw_exception("Metal gated pipeline entry requires a diagnostic reason");
		}

		reason += "; missing requirements: ";
		reason += describe_pipeline_entry_requirements(requirement_mask);

		return
		{
			.error = std::move(reason),
			.requirement_mask = requirement_mask,
			.available = false,
		};
	}
}
