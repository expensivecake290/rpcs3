#include "stdafx.h"
#include "MTLShaderEntrypointBuilder.h"

#include "util/fnv_hash.hpp"

#include <string_view>
#include <utility>

namespace
{
	u64 pipeline_source_text_hash(std::string_view source)
	{
		rsx_log.trace("pipeline_source_text_hash(size=0x%x)", source.size());

		usz hash = rpcs3::fnv_seed;
		for (const char c : source)
		{
			hash = rpcs3::hash64(hash, static_cast<u8>(c));
		}

		return static_cast<u64>(hash);
	}

	const char* pipeline_entry_stage_keyword(rsx::metal::shader_stage stage)
	{
		rsx_log.trace("pipeline_entry_stage_keyword(stage=%u)", static_cast<u32>(stage));

		switch (stage)
		{
		case rsx::metal::shader_stage::vertex:
			return "vertex";
		case rsx::metal::shader_stage::fragment:
			return "fragment";
		case rsx::metal::shader_stage::mesh:
			fmt::throw_exception("Metal mesh pipeline entry source validation requires confirmed MSL mesh function syntax");
		}

		fmt::throw_exception("Unknown Metal shader stage %u", static_cast<u32>(stage));
	}

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
	void validate_pipeline_entry_source(
		shader_stage stage,
		u64 source_hash,
		const std::string& entry_point,
		const std::string& source)
	{
		rsx_log.trace("rsx::metal::validate_pipeline_entry_source(stage=%u, source_hash=0x%llx, entry_point=%s, size=0x%x)",
			static_cast<u32>(stage), source_hash, entry_point.c_str(), static_cast<u32>(source.size()));

		if (!source_hash)
		{
			fmt::throw_exception("Metal pipeline entry source requires a non-zero source hash");
		}

		if (entry_point.empty())
		{
			fmt::throw_exception("Metal pipeline entry source requires an entry point");
		}

		if (source.empty())
		{
			fmt::throw_exception("Metal pipeline entry source requires MSL source");
		}

		if (pipeline_source_text_hash(source) != source_hash)
		{
			fmt::throw_exception("Metal pipeline entry source hash mismatch for entry '%s'", entry_point);
		}

		if (source.find("#include <metal_stdlib>") == std::string::npos)
		{
			fmt::throw_exception("Metal pipeline entry source '%s' is missing the Metal standard library include", entry_point);
		}

		const char* stage_keyword = pipeline_entry_stage_keyword(stage);
		if (source.find(stage_keyword) == std::string::npos ||
			source.find(entry_point + "(") == std::string::npos)
		{
			fmt::throw_exception("Metal pipeline entry source '%s' does not expose a %s entry function",
				entry_point,
				stage_keyword);
		}
	}

	void validate_pipeline_entry_build_result(shader_stage stage, const pipeline_entry_build_result& result)
	{
		rsx_log.trace("rsx::metal::validate_pipeline_entry_build_result(stage=%u, source_hash=0x%llx, requirement_mask=0x%x, available=%u)",
			static_cast<u32>(stage),
			result.source_hash,
			result.requirement_mask,
			static_cast<u32>(result.available));

		validate_pipeline_entry_requirement_mask(result.requirement_mask);

		if (result.available)
		{
			if (result.requirement_mask || !result.error.empty() || result.cache_path.empty())
			{
				fmt::throw_exception("Metal available pipeline entry result contains gated metadata");
			}

			validate_pipeline_entry_source(stage, result.source_hash, result.entry_point, result.source);
			return;
		}

		if (!result.requirement_mask || result.error.empty())
		{
			fmt::throw_exception("Metal gated pipeline entry result requires a requirement mask and diagnostic reason");
		}

		if (result.source_hash || !result.entry_point.empty() || !result.source.empty() || !result.cache_path.empty())
		{
			fmt::throw_exception("Metal gated pipeline entry result must not contain executable source metadata");
		}
	}

	pipeline_entry_build_result build_pipeline_entry_source(
		const translated_shader& shader,
		const shader_interface_layout& layout,
		u32 requirement_mask,
		std::string reason)
	{
		rsx_log.notice("rsx::metal::build_pipeline_entry_source(stage=%u, id=%u, source_hash=0x%llx, requirement_mask=0x%x)",
			static_cast<u32>(shader.stage), shader.id, shader.source_hash, requirement_mask);

		validate_pipeline_entry_request(shader, layout);
		validate_pipeline_entry_requirement_mask(requirement_mask);

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
