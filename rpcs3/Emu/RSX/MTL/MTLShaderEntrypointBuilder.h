#pragma once

#include "MTLShaderInterface.h"
#include "MTLShaderRecompiler.h"

#include "util/types.hpp"

#include <string>

namespace rsx::metal
{
	struct pipeline_entry_build_result
	{
		u64 source_hash = 0;
		std::string entry_point;
		std::string source;
		std::string cache_path;
		std::string error;
		u32 requirement_mask = 0;
		b8 available = false;
	};

	void validate_pipeline_entry_source(
		shader_stage stage,
		u64 source_hash,
		const std::string& entry_point,
		const std::string& source);
	b8 is_valid_pipeline_entry_source(
		shader_stage stage,
		u64 source_hash,
		const std::string& entry_point,
		const std::string& source,
		std::string* error = nullptr);
	void validate_pipeline_entry_build_result(shader_stage stage, const pipeline_entry_build_result& result);
	pipeline_entry_build_result build_pipeline_entry_source(
		const translated_shader& shader,
		const shader_interface_layout& layout,
		u32 requirement_mask,
		std::string reason);
}
