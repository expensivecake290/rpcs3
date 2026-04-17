#include "stdafx.h"
#include "MTLShaderEntrypoint.h"

#include "Emu/RSX/Program/RSXFragmentProgram.h"

namespace
{
	void clear_pipeline_entry(rsx::metal::translated_shader& shader, std::string reason)
	{
		rsx_log.trace("clear_pipeline_entry(stage=%u, id=%u)", static_cast<u32>(shader.stage), shader.id);

		shader.pipeline_source_hash = 0;
		shader.pipeline_entry_point.clear();
		shader.pipeline_source.clear();
		shader.pipeline_cache_path.clear();
		shader.pipeline_entry_error = std::move(reason);
		shader.pipeline_entry_available = false;
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
}

namespace rsx::metal
{
	void mark_vertex_pipeline_entry_status(translated_shader& shader)
	{
		rsx_log.notice("rsx::metal::mark_vertex_pipeline_entry_status(id=%u, source_hash=0x%llx)", shader.id, shader.source_hash);

		validate_helper_shader(shader, shader_stage::vertex);
		clear_pipeline_entry(shader,
			"Metal vertex pipeline entry generation is gated until Phase 3 argument tables, vertex input fetch, and viewport/depth transform state are implemented");
	}

	void mark_fragment_pipeline_entry_status(translated_shader& shader, const RSXFragmentProgram& program)
	{
		rsx_log.notice("rsx::metal::mark_fragment_pipeline_entry_status(id=%u, source_hash=0x%llx, mrt_count=%u, ctrl=0x%x)",
			shader.id, shader.source_hash, program.mrt_buffers_count, program.ctrl);

		validate_helper_shader(shader, shader_stage::fragment);

		std::string reason = "Metal fragment pipeline entry generation is gated until Phase 3 argument tables and stage input layout are implemented";

		if (program.mrt_buffers_count > 1)
		{
			reason += "; MRT output mapping is not enabled";
		}

		if (program.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT)
		{
			reason += "; depth export mapping is not enabled";
		}

		clear_pipeline_entry(shader, std::move(reason));
	}

	void mark_mesh_pipeline_entry_status(translated_shader& shader)
	{
		rsx_log.notice("rsx::metal::mark_mesh_pipeline_entry_status(id=%u, source_hash=0x%llx)", shader.id, shader.source_hash);

		validate_helper_shader(shader, shader_stage::mesh);
		clear_pipeline_entry(shader,
			"Metal mesh pipeline entry generation is gated until RSX-to-MSL mesh/object shader mapping and Phase 3 argument tables are implemented");
	}
}
