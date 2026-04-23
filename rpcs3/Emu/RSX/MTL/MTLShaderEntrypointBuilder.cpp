#include "stdafx.h"
#include "MTLShaderEntrypointBuilder.h"

#include "util/fnv_hash.hpp"

#include <string_view>
#include <utility>

namespace
{
	struct pipeline_entry_source_contract
	{
		const char* stage_keyword = nullptr;
		const char* helper_context_type = nullptr;
		const char* helper_entry_prefix = nullptr;
	};

	b8 reject_pipeline_entry_source(std::string* error, std::string reason)
	{
		rsx_log.trace("reject_pipeline_entry_source(reason=%s)", reason);

		if (error)
		{
			*error = std::move(reason);
		}

		return false;
	}

	b8 is_msl_identifier_char(char c)
	{
		rsx_log.trace("is_msl_identifier_char(c=0x%x)", static_cast<u8>(c));

		return (c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') ||
			c == '_';
	}

	b8 is_token_boundary(std::string_view text, usz index)
	{
		rsx_log.trace("is_token_boundary(size=0x%x, index=0x%x)", static_cast<u32>(text.size()), static_cast<u32>(index));

		return index >= text.size() || !is_msl_identifier_char(text[index]);
	}

	b8 has_msl_stage_token(std::string_view text, std::string_view token)
	{
		rsx_log.trace("has_msl_stage_token(size=0x%x, token=%s)", static_cast<u32>(text.size()), std::string(token));

		if (token.empty())
		{
			return false;
		}

		usz offset = 0;
		while ((offset = text.find(token, offset)) != std::string_view::npos)
		{
			const b8 left_boundary = !offset || is_token_boundary(text, offset - 1);
			const b8 right_boundary = is_token_boundary(text, offset + token.size());
			if (left_boundary && right_boundary)
			{
				return true;
			}

			offset++;
		}

		return false;
	}

	b8 has_msl_stage_function_declaration(std::string_view source, std::string_view stage_keyword, const std::string& entry_point)
	{
		rsx_log.trace("has_msl_stage_function_declaration(stage=%s, entry_point=%s, size=0x%x)",
			std::string(stage_keyword), entry_point, static_cast<u32>(source.size()));

		const std::string entry_declaration = entry_point + "(";
		usz offset = 0;
		while ((offset = source.find(entry_declaration, offset)) != std::string_view::npos)
		{
			const b8 left_boundary = !offset || is_token_boundary(source, offset - 1);
			const b8 right_boundary = is_token_boundary(source, offset + entry_point.size());
			if (!left_boundary || !right_boundary)
			{
				offset++;
				continue;
			}

			usz declaration_start = 0;
			for (const char delimiter : { ';', '{', '}' })
			{
				const usz position = source.rfind(delimiter, offset);
				if (position != std::string_view::npos && position + 1 > declaration_start)
				{
					declaration_start = position + 1;
				}
			}

			const std::string_view declaration_prefix = source.substr(declaration_start, offset - declaration_start);
			if (has_msl_stage_token(declaration_prefix, stage_keyword))
			{
				return true;
			}

			offset++;
		}

		return false;
	}

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

	const pipeline_entry_source_contract* try_pipeline_entry_source_contract(rsx::metal::shader_stage stage)
	{
		rsx_log.trace("try_pipeline_entry_source_contract(stage=%u)", static_cast<u32>(stage));

		static constexpr pipeline_entry_source_contract s_vertex_contract =
		{
			.stage_keyword = "vertex",
			.helper_context_type = "rpcs3_mtl_vertex_context",
			.helper_entry_prefix = "rpcs3_mtl_vp_",
		};

		static constexpr pipeline_entry_source_contract s_fragment_contract =
		{
			.stage_keyword = "fragment",
			.helper_context_type = "rpcs3_mtl_fragment_context",
			.helper_entry_prefix = "rpcs3_mtl_fp_",
		};

		switch (stage)
		{
		case rsx::metal::shader_stage::vertex:
			return &s_vertex_contract;
		case rsx::metal::shader_stage::fragment:
			return &s_fragment_contract;
		case rsx::metal::shader_stage::mesh:
			return nullptr;
		}

		return nullptr;
	}

	std::string describe_pipeline_entry_source_contract(rsx::metal::shader_stage stage)
	{
		rsx_log.trace("describe_pipeline_entry_source_contract(stage=%u)", static_cast<u32>(stage));

		if (const pipeline_entry_source_contract* contract = try_pipeline_entry_source_contract(stage))
		{
			return fmt::format("stage_keyword=%s, helper_context=%s, helper_entry_prefix=%s",
				contract->stage_keyword,
				contract->helper_context_type,
				contract->helper_entry_prefix);
		}

		if (stage == rsx::metal::shader_stage::mesh)
		{
			return "mesh wrapper contract unavailable until confirmed MSL mesh/object shader syntax is implemented";
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
	b8 is_valid_pipeline_entry_source(
		shader_stage stage,
		u64 source_hash,
		const std::string& entry_point,
		const std::string& source,
		std::string* error)
	{
		rsx_log.trace("rsx::metal::is_valid_pipeline_entry_source(stage=%u, source_hash=0x%llx, entry_point=%s, size=0x%x)",
			static_cast<u32>(stage), source_hash, entry_point.c_str(), static_cast<u32>(source.size()));

		if (!source_hash)
		{
			return reject_pipeline_entry_source(error, "Metal pipeline entry source requires a non-zero source hash");
		}

		if (entry_point.empty())
		{
			return reject_pipeline_entry_source(error, "Metal pipeline entry source requires an entry point");
		}

		if (source.empty())
		{
			return reject_pipeline_entry_source(error, "Metal pipeline entry source requires MSL source");
		}

		if (pipeline_source_text_hash(source) != source_hash)
		{
			return reject_pipeline_entry_source(error, fmt::format("Metal pipeline entry source hash mismatch for entry '%s'", entry_point));
		}

		if (source.find("#include <metal_stdlib>") == std::string::npos)
		{
			return reject_pipeline_entry_source(error, fmt::format("Metal pipeline entry source '%s' is missing the Metal standard library include", entry_point));
		}

		const pipeline_entry_source_contract* contract = try_pipeline_entry_source_contract(stage);
		if (!contract)
		{
			return reject_pipeline_entry_source(error, "Metal pipeline entry source validation requires a confirmed vertex or fragment wrapper contract");
		}

		if (!has_msl_stage_function_declaration(source, contract->stage_keyword, entry_point))
		{
			return reject_pipeline_entry_source(error, fmt::format("Metal pipeline entry source '%s' does not expose a %s entry function",
				entry_point,
				contract->stage_keyword));
		}

		if (source.find(contract->helper_context_type) == std::string::npos)
		{
			return reject_pipeline_entry_source(error, fmt::format("Metal pipeline entry source '%s' does not reference helper context '%s'",
				entry_point,
				contract->helper_context_type));
		}

		if (source.find(contract->helper_entry_prefix) == std::string::npos)
		{
			return reject_pipeline_entry_source(error, fmt::format("Metal pipeline entry source '%s' does not reference translated helper prefix '%s'",
				entry_point,
				contract->helper_entry_prefix));
		}

		if (error)
		{
			error->clear();
		}

		return true;
	}

	void validate_pipeline_entry_source(
		shader_stage stage,
		u64 source_hash,
		const std::string& entry_point,
		const std::string& source)
	{
		rsx_log.trace("rsx::metal::validate_pipeline_entry_source(stage=%u, source_hash=0x%llx, entry_point=%s, size=0x%x)",
			static_cast<u32>(stage), source_hash, entry_point.c_str(), static_cast<u32>(source.size()));

		std::string error;
		if (!is_valid_pipeline_entry_source(stage, source_hash, entry_point, source, &error))
		{
			fmt::throw_exception("%s", error);
		}
	}

	void validate_pipeline_entry_build_result(shader_stage stage, const pipeline_entry_build_result& result)
	{
		rsx_log.trace("rsx::metal::validate_pipeline_entry_build_result(stage=%u, source_hash=0x%llx, requirement_mask=0x%x, available=%u)",
			static_cast<u32>(stage),
			result.source_hash,
			result.requirement_mask,
			static_cast<u32>(result.available));

		validate_pipeline_entry_requirement_mask_for_stage(stage, result.requirement_mask);

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
		validate_pipeline_entry_requirement_mask_for_stage(shader.stage, requirement_mask);

		if (!requirement_mask)
		{
			fmt::throw_exception("Metal pipeline entry source generation requires a verified executable MSL wrapper implementation");
		}

		if (reason.empty())
		{
			fmt::throw_exception("Metal gated pipeline entry requires a diagnostic reason");
		}

		reason += "; wrapper contract: ";
		reason += describe_pipeline_entry_source_contract(shader.stage);
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
