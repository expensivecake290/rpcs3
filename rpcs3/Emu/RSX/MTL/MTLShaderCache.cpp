#include "stdafx.h"
#include "MTLShaderCache.h"

#include "MTLShaderEntrypointBuilder.h"
#include "MTLShaderInterface.h"

#include "Emu/cache_utils.hpp"
#include "Emu/system_config.h"
#include "Utilities/File.h"
#include "util/fnv_hash.hpp"

#include <charconv>
#include <limits>
#include <string_view>

namespace
{
	void increment_cache_counter(u32& counter, const char* counter_name)
	{
		if (counter == std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal shader cache %s counter overflow", counter_name);
		}

		counter++;
	}

	u32 checked_cache_total(u32 lhs, u32 rhs, const char* counter_name)
	{
		if (lhs > std::numeric_limits<u32>::max() - rhs)
		{
			fmt::throw_exception("Metal shader cache %s total overflow", counter_name);
		}

		return lhs + rhs;
	}

	void report_cache_split_mismatch(const char* label, u32 total, u32 available, u32 gated)
	{
		const u32 split_total = checked_cache_total(available, gated, label);
		if (total != split_total)
		{
			rsx_log.warning("Metal shader cache %s split mismatch: total=%u, available=%u, gated=%u",
				label,
				total,
				available,
				gated);
		}
	}

	std::string make_manifest_text(std::string_view version)
	{
		rsx_log.trace("make_manifest_text(version=%s)", version);

		if (version.empty())
		{
			fmt::throw_exception("Metal shader cache manifest requires a non-empty version");
		}

		return fmt::format(
			"RPCS3 Metal shader cache\n"
			"backend=metal\n"
			"version=%s\n",
			version);
	}

	u32 count_files_with_extension(const std::string& directory, std::string_view extension)
	{
		rsx_log.trace("count_files_with_extension(directory=%s, extension=%s)", directory, extension);

		fs::dir dir(directory);
		if (!dir)
		{
			return 0;
		}

		u32 count = 0;

		for (const fs::dir_entry& entry : dir)
		{
			if (entry.is_directory)
			{
				continue;
			}

			if (extension.empty() || std::string_view(entry.name).ends_with(extension))
			{
				increment_cache_counter(count, "raw shader file");
			}
		}

		return count;
	}

	std::string metadata_value(std::string_view value)
	{
		rsx_log.trace("metadata_value(size=0x%x)", value.size());

		std::string result;
		result.reserve(value.size());

		for (const char c : value)
		{
			result += (c == '\n' || c == '\r') ? ' ' : c;
		}

		return result;
	}

	std::string_view take_metadata_line(std::string_view& text)
	{
		rsx_log.trace("take_metadata_line(size=0x%x)", text.size());

		const usz line_end = text.find('\n');
		std::string_view line = line_end == umax ? text : text.substr(0, line_end);
		text = line_end == umax ? std::string_view{} : text.substr(line_end + 1);

		if (!line.empty() && line.back() == '\r')
		{
			line.remove_suffix(1);
		}

		return line;
	}

	std::string get_metadata_field(std::string_view text, std::string_view key)
	{
		rsx_log.trace("get_metadata_field(key=%s)", key);

		std::string result;
		b8 found = false;
		const std::string prefix = std::string(key) + "=";

		while (!text.empty())
		{
			const std::string_view line = take_metadata_line(text);
			if (!line.starts_with(prefix))
			{
				continue;
			}

			if (found)
			{
				fmt::throw_exception("Duplicate Metal pipeline entry metadata field '%s'", std::string(key));
			}

			result = std::string(line.substr(prefix.size()));
			found = true;
		}

		if (!found)
		{
			fmt::throw_exception("Missing Metal pipeline entry metadata field '%s'", std::string(key));
		}

		return result;
	}

	u64 parse_metadata_u64(const std::string& value, const char* field)
	{
		rsx_log.trace("parse_metadata_u64(field=%s, value=%s)", field, value);

		std::string_view digits = value;
		int base = 10;

		if (digits.starts_with("0x") || digits.starts_with("0X"))
		{
			digits.remove_prefix(2);
			base = 16;
		}

		u64 result = 0;
		const auto parse_result = std::from_chars(digits.data(), digits.data() + digits.size(), result, base);
		if (parse_result.ec != std::errc{} || parse_result.ptr != digits.data() + digits.size())
		{
			fmt::throw_exception("Invalid Metal pipeline entry metadata integer '%s=%s'", field, value);
		}

		return result;
	}

	u32 parse_metadata_u32(const std::string& value, const char* field)
	{
		rsx_log.trace("parse_metadata_u32(field=%s, value=%s)", field, value);

		const u64 result = parse_metadata_u64(value, field);
		if (result > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal pipeline entry metadata field '%s' is too large: %s", field, value);
		}

		return static_cast<u32>(result);
	}

	b8 parse_metadata_b8(const std::string& value, const char* field)
	{
		rsx_log.trace("parse_metadata_b8(field=%s, value=%s)", field, value);

		if (value == "0")
		{
			return false;
		}

		if (value == "1")
		{
			return true;
		}

		fmt::throw_exception("Metal pipeline entry metadata field '%s' must be 0 or 1", field);
	}

	std::string serialize_metadata_u32_list(const std::vector<u32>& values)
	{
		rsx_log.trace("serialize_metadata_u32_list(count=%u)", static_cast<u32>(values.size()));

		if (values.empty())
		{
			return {};
		}

		std::string result;
		for (usz index = 0; index < values.size(); index++)
		{
			if (index)
			{
				result += ",";
			}

			result += fmt::format("0x%x", values[index]);
		}

		return result;
	}

	std::vector<u32> parse_metadata_u32_list(const std::string& value, const char* field)
	{
		rsx_log.trace("parse_metadata_u32_list(field=%s, value=%s)", field, value);

		std::vector<u32> result;
		if (value.empty())
		{
			return result;
		}

		usz begin = 0;
		while (begin <= value.size())
		{
			const usz end = value.find(',', begin);
			const std::string token = value.substr(begin, end == umax ? std::string::npos : end - begin);
			if (token.empty())
			{
				fmt::throw_exception("Invalid Metal pipeline entry metadata list '%s=%s'", field, value);
			}

			result.push_back(parse_metadata_u32(token, field));
			if (end == umax)
			{
				break;
			}

			begin = end + 1;
		}

		return result;
	}

	b8 try_get_metadata_field(std::string_view text, std::string_view key, std::string& result)
	{
		rsx_log.trace("try_get_metadata_field(key=%s)", key);

		result.clear();

		b8 found = false;
		const std::string prefix = std::string(key) + "=";

		while (!text.empty())
		{
			const std::string_view line = take_metadata_line(text);
			if (!line.starts_with(prefix))
			{
				continue;
			}

			if (found)
			{
				return false;
			}

			result = std::string(line.substr(prefix.size()));
			found = true;
		}

		return found;
	}

	b8 try_parse_metadata_u64(const std::string& value, u64& result)
	{
		rsx_log.trace("try_parse_metadata_u64(value=%s)", value);

		std::string_view digits = value;
		int base = 10;

		if (digits.starts_with("0x") || digits.starts_with("0X"))
		{
			digits.remove_prefix(2);
			base = 16;
		}

		if (digits.empty())
		{
			return false;
		}

		result = 0;
		const auto parse_result = std::from_chars(digits.data(), digits.data() + digits.size(), result, base);
		return parse_result.ec == std::errc{} && parse_result.ptr == digits.data() + digits.size();
	}

	b8 try_parse_metadata_u32(const std::string& value, u32& result)
	{
		rsx_log.trace("try_parse_metadata_u32(value=%s)", value);

		u64 parsed_value = 0;
		if (!try_parse_metadata_u64(value, parsed_value) || parsed_value > std::numeric_limits<u32>::max())
		{
			return false;
		}

		result = static_cast<u32>(parsed_value);
		return true;
	}

	b8 try_parse_metadata_b8(const std::string& value, b8& result)
	{
		rsx_log.trace("try_parse_metadata_b8(value=%s)", value);

		if (value == "0")
		{
			result = false;
			return true;
		}

		if (value == "1")
		{
			result = true;
			return true;
		}

		return false;
	}

	b8 try_parse_metadata_u32_list(const std::string& value, std::vector<u32>& result)
	{
		rsx_log.trace("try_parse_metadata_u32_list(value=%s)", value);

		result.clear();
		if (value.empty())
		{
			return true;
		}

		usz begin = 0;
		while (begin <= value.size())
		{
			const usz end = value.find(',', begin);
			const std::string token = value.substr(begin, end == umax ? std::string::npos : end - begin);
			u32 parsed_value = 0;
			if (token.empty() || !try_parse_metadata_u32(token, parsed_value))
			{
				result.clear();
				return false;
			}

			result.push_back(parsed_value);
			if (end == umax)
			{
				break;
			}

			begin = end + 1;
		}

		return true;
	}

	b8 is_pipeline_entry_stage(std::string_view stage)
	{
		rsx_log.trace("is_pipeline_entry_stage(stage=%s)", stage);
		return stage == "vp" || stage == "fp" || stage == "mesh";
	}

	b8 pipeline_entry_shader_stage(std::string_view stage, rsx::metal::shader_stage& shader_stage)
	{
		rsx_log.trace("pipeline_entry_shader_stage(stage=%s)", stage);

		if (stage == "vp")
		{
			shader_stage = rsx::metal::shader_stage::vertex;
			return true;
		}

		if (stage == "fp")
		{
			shader_stage = rsx::metal::shader_stage::fragment;
			return true;
		}

		if (stage == "mesh")
		{
			shader_stage = rsx::metal::shader_stage::mesh;
			return true;
		}

		return false;
	}

	b8 is_pipeline_entry_requirement_mask_for_stage(std::string_view stage, u32 requirement_mask)
	{
		rsx_log.trace("is_pipeline_entry_requirement_mask_for_stage(stage=%s, requirement_mask=0x%x)",
			stage, requirement_mask);

		rsx::metal::shader_stage shader_stage = rsx::metal::shader_stage::vertex;
		if (!pipeline_entry_shader_stage(stage, shader_stage))
		{
			return false;
		}

		if (requirement_mask & ~rsx::metal::known_pipeline_entry_requirement_mask())
		{
			return false;
		}

		return !(requirement_mask & ~rsx::metal::pipeline_entry_requirement_mask_for_stage(shader_stage));
	}

	void validate_metadata_pipeline_entry_requirement_mask_for_stage(
		std::string_view stage,
		u32 requirement_mask,
		const char* metadata_kind)
	{
		rsx_log.trace("validate_metadata_pipeline_entry_requirement_mask_for_stage(stage=%s, requirement_mask=0x%x, metadata_kind=%s)",
			stage,
			requirement_mask,
			metadata_kind ? metadata_kind : "<null>");

		rsx::metal::shader_stage shader_stage = rsx::metal::shader_stage::vertex;
		if (!pipeline_entry_shader_stage(stage, shader_stage))
		{
			fmt::throw_exception("Metal %s metadata requires a valid shader stage", metadata_kind ? metadata_kind : "shader");
		}

		rsx::metal::validate_pipeline_entry_requirement_mask_for_stage(shader_stage, requirement_mask);
	}

	b8 is_shader_source_stage(std::string_view stage)
	{
		rsx_log.trace("is_shader_source_stage(stage=%s)", stage);
		return stage == "vp" || stage == "fp" || stage == "mesh";
	}

	b8 is_shader_library_stage(std::string_view stage)
	{
		rsx_log.trace("is_shader_library_stage(stage=%s)", stage);
		return stage == "vp" || stage == "fp" || stage == "mesh";
	}

	b8 is_shader_completion_stage(std::string_view stage)
	{
		rsx_log.trace("is_shader_completion_stage(stage=%s)", stage);
		return stage == "vp" || stage == "fp" || stage == "mesh";
	}

	b8 is_shader_translation_failure_stage(std::string_view stage)
	{
		rsx_log.trace("is_shader_translation_failure_stage(stage=%s)", stage);
		return stage == "vp" || stage == "fp" || stage == "mesh";
	}

	b8 is_pipeline_state_type(std::string_view pipeline_type)
	{
		rsx_log.trace("is_pipeline_state_type(pipeline_type=%s)", pipeline_type);
		return pipeline_type == "render" || pipeline_type == "mesh";
	}

	u32 pipeline_state_shader_dependency_count(
		std::string_view pipeline_type,
		u64 vertex_source_hash,
		u64 fragment_source_hash,
		u64 object_source_hash,
		u64 mesh_source_hash,
		b8 rasterization_enabled)
	{
		rsx_log.trace("pipeline_state_shader_dependency_count(pipeline_type=%s, vertex=0x%llx, fragment=0x%llx, object=0x%llx, mesh=0x%llx, rasterization_enabled=%u)",
			pipeline_type,
			vertex_source_hash,
			fragment_source_hash,
			object_source_hash,
			mesh_source_hash,
			static_cast<u32>(rasterization_enabled));

		if (pipeline_type == "render")
		{
			if (!vertex_source_hash || object_source_hash || mesh_source_hash ||
				(rasterization_enabled ? !fragment_source_hash : !!fragment_source_hash))
			{
				return 0;
			}

			return rasterization_enabled ? 2 : 1;
		}

		if (pipeline_type == "mesh")
		{
			if (vertex_source_hash || !mesh_source_hash ||
				(rasterization_enabled ? !fragment_source_hash : !!fragment_source_hash))
			{
				return 0;
			}

			return 1 + (object_source_hash ? 1 : 0) + (rasterization_enabled ? 1 : 0);
		}

		return 0;
	}

	u64 source_text_hash(std::string_view source)
	{
		rsx_log.trace("source_text_hash(size=0x%x)", source.size());

		usz hash = rpcs3::fnv_seed;
		for (const char c : source)
		{
			hash = rpcs3::hash64(hash, static_cast<u8>(c));
		}

		return static_cast<u64>(hash);
	}

	b8 get_file_text_hash(const std::string& path, u64& hash)
	{
		rsx_log.trace("get_file_text_hash(path=%s)", path);

		fs::file file{path, fs::read};
		if (!file)
		{
			return false;
		}

		const std::string text = file.to_string();
		if (text.empty())
		{
			return false;
		}

		hash = source_text_hash(text);
		return !!hash;
	}

	b8 get_file_content_hash(const std::string& path, u64& hash)
	{
		rsx_log.trace("get_file_content_hash(path=%s)", path);

		fs::file file{path, fs::read};
		if (!file)
		{
			return false;
		}

		const std::string bytes = file.to_string();
		if (bytes.empty())
		{
			return false;
		}

		hash = source_text_hash(bytes);
		return !!hash;
	}

	b8 validate_available_pipeline_entry_source(
		std::string_view stage,
		u64 source_hash,
		const std::string& entry_point,
		const std::string& source_path)
	{
		rsx_log.trace("validate_available_pipeline_entry_source(stage=%s, source_hash=0x%llx, entry_point=%s, source_path=%s)",
			stage, source_hash, entry_point.c_str(), source_path.c_str());

		rsx::metal::shader_stage shader_stage = rsx::metal::shader_stage::vertex;
		if (!pipeline_entry_shader_stage(stage, shader_stage))
		{
			return false;
		}

		if (shader_stage == rsx::metal::shader_stage::mesh)
		{
			return false;
		}

		fs::file file{source_path, fs::read};
		if (!file)
		{
			return false;
		}

		const std::string source = file.to_string();
		if (source.empty())
		{
			return false;
		}

		std::string error;
		if (!rsx::metal::is_valid_pipeline_entry_source(shader_stage, source_hash, entry_point, source, &error))
		{
			rsx_log.warning("Metal cached pipeline entry source '%s' is invalid: %s", source_path, error);
			return false;
		}

		return true;
	}

	std::string shader_library_metadata_mismatch(
		const rsx::metal::shader_library_metadata& metadata,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& library_path,
		const std::string& pipeline_entry_error,
		u32 pipeline_requirement_mask,
		b8 pipeline_entry_available)
	{
		rsx_log.trace("shader_library_metadata_mismatch(shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, library_path=%s, pipeline_requirement_mask=0x%x, pipeline_entry_available=%u)",
			shader_id,
			source_hash,
			source_text_hash,
			entry_point.c_str(),
			library_path.c_str(),
			pipeline_requirement_mask,
			static_cast<u32>(pipeline_entry_available));

		if (metadata.shader_id != shader_id)
		{
			return fmt::format("shader id %u does not match requested shader id %u",
				metadata.shader_id,
				shader_id);
		}

		if (metadata.source_hash != source_hash)
		{
			return fmt::format("source hash 0x%llx does not match requested source hash 0x%llx",
				metadata.source_hash,
				source_hash);
		}

		if (metadata.source_text_hash != source_text_hash)
		{
			return fmt::format("source text hash 0x%llx does not match requested source text hash 0x%llx",
				metadata.source_text_hash,
				source_text_hash);
		}

		if (metadata.entry_point != entry_point)
		{
			return fmt::format("entry point '%s' does not match requested entry point '%s'",
				metadata.entry_point,
				entry_point);
		}

		if (metadata.library_path != library_path)
		{
			return fmt::format("library path '%s' does not match requested library path '%s'",
				metadata.library_path,
				library_path);
		}

		if (metadata.pipeline_requirement_mask != pipeline_requirement_mask)
		{
			return fmt::format("pipeline requirement mask 0x%x does not match requested mask 0x%x",
				metadata.pipeline_requirement_mask,
				pipeline_requirement_mask);
		}

		if (metadata.pipeline_entry_error != pipeline_entry_error)
		{
			return fmt::format("pipeline entry error '%s' does not match requested error '%s'",
				metadata.pipeline_entry_error,
				pipeline_entry_error);
		}

		if (metadata.pipeline_entry_available != pipeline_entry_available)
		{
			return fmt::format("pipeline entry availability %u does not match requested availability %u",
				static_cast<u32>(metadata.pipeline_entry_available),
				static_cast<u32>(pipeline_entry_available));
		}

		return {};
	}

	std::string shader_completion_metadata_mismatch(
		const rsx::metal::shader_completion_metadata& metadata,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& source_path,
		const std::vector<u32>& fragment_constant_offsets,
		u64 pipeline_source_hash,
		const std::string& pipeline_entry_point,
		const std::string& pipeline_source_path,
		const std::string& pipeline_entry_error,
		u32 pipeline_requirement_mask,
		b8 pipeline_entry_available)
	{
		rsx_log.trace("shader_completion_metadata_mismatch(shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, fragment_constant_count=%u, pipeline_source_hash=0x%llx, pipeline_requirement_mask=0x%x, pipeline_entry_available=%u)",
			shader_id,
			source_hash,
			source_text_hash,
			static_cast<u32>(fragment_constant_offsets.size()),
			pipeline_source_hash,
			pipeline_requirement_mask,
			static_cast<u32>(pipeline_entry_available));

		if (metadata.shader_id != shader_id)
		{
			return fmt::format("shader id %u does not match requested shader id %u",
				metadata.shader_id,
				shader_id);
		}

		if (metadata.source_hash != source_hash)
		{
			return fmt::format("source hash 0x%llx does not match requested source hash 0x%llx",
				metadata.source_hash,
				source_hash);
		}

		if (metadata.source_text_hash != source_text_hash)
		{
			return fmt::format("source text hash 0x%llx does not match requested source text hash 0x%llx",
				metadata.source_text_hash,
				source_text_hash);
		}

		if (metadata.entry_point != entry_point)
		{
			return fmt::format("helper entry point '%s' does not match requested helper entry point '%s'",
				metadata.entry_point,
				entry_point);
		}

		if (metadata.source_path != source_path)
		{
			return fmt::format("helper source path '%s' does not match requested helper source path '%s'",
				metadata.source_path,
				source_path);
		}

		if (metadata.fragment_constant_offsets != fragment_constant_offsets)
		{
			return "fragment constant offsets do not match requested shader metadata";
		}

		if (metadata.pipeline_source_hash != pipeline_source_hash)
		{
			return fmt::format("pipeline source hash 0x%llx does not match requested pipeline source hash 0x%llx",
				metadata.pipeline_source_hash,
				pipeline_source_hash);
		}

		if (metadata.pipeline_entry_point != pipeline_entry_point)
		{
			return fmt::format("pipeline entry point '%s' does not match requested pipeline entry point '%s'",
				metadata.pipeline_entry_point,
				pipeline_entry_point);
		}

		if (metadata.pipeline_source_path != pipeline_source_path)
		{
			return fmt::format("pipeline source path '%s' does not match requested pipeline source path '%s'",
				metadata.pipeline_source_path,
				pipeline_source_path);
		}

		if (metadata.pipeline_entry_error != pipeline_entry_error)
		{
			return fmt::format("pipeline entry error '%s' does not match requested error '%s'",
				metadata.pipeline_entry_error,
				pipeline_entry_error);
		}

		if (metadata.pipeline_requirement_mask != pipeline_requirement_mask)
		{
			return fmt::format("pipeline requirement mask 0x%x does not match requested mask 0x%x",
				metadata.pipeline_requirement_mask,
				pipeline_requirement_mask);
		}

		if (metadata.pipeline_entry_available != pipeline_entry_available)
		{
			return fmt::format("pipeline entry availability %u does not match requested availability %u",
				static_cast<u32>(metadata.pipeline_entry_available),
				static_cast<u32>(pipeline_entry_available));
		}

		return {};
	}

	std::string pipeline_entry_metadata_mismatch(
		const rsx::metal::pipeline_entry_metadata& metadata,
		u32 shader_id,
		u64 source_hash,
		u64 pipeline_source_hash,
		const std::string& entry_point,
		const std::string& source_path,
		const std::string& entry_error,
		u32 requirement_mask,
		b8 entry_available)
	{
		rsx_log.trace("pipeline_entry_metadata_mismatch(shader_id=%u, source_hash=0x%llx, pipeline_source_hash=0x%llx, requirement_mask=0x%x, entry_available=%u)",
			shader_id,
			source_hash,
			pipeline_source_hash,
			requirement_mask,
			static_cast<u32>(entry_available));

		if (metadata.shader_id != shader_id)
		{
			return fmt::format("shader id %u does not match requested shader id %u",
				metadata.shader_id,
				shader_id);
		}

		if (metadata.source_hash != source_hash)
		{
			return fmt::format("source hash 0x%llx does not match requested source hash 0x%llx",
				metadata.source_hash,
				source_hash);
		}

		if (metadata.pipeline_source_hash != pipeline_source_hash)
		{
			return fmt::format("pipeline source hash 0x%llx does not match requested pipeline source hash 0x%llx",
				metadata.pipeline_source_hash,
				pipeline_source_hash);
		}

		if (metadata.entry_point != entry_point)
		{
			return fmt::format("entry point '%s' does not match requested entry point '%s'",
				metadata.entry_point,
				entry_point);
		}

		if (metadata.source_path != source_path)
		{
			return fmt::format("source path '%s' does not match requested source path '%s'",
				metadata.source_path,
				source_path);
		}

		if (metadata.entry_error != entry_error)
		{
			return fmt::format("entry error '%s' does not match requested error '%s'",
				metadata.entry_error,
				entry_error);
		}

		if (metadata.requirement_mask != requirement_mask)
		{
			return fmt::format("requirement mask 0x%x does not match requested mask 0x%x",
				metadata.requirement_mask,
				requirement_mask);
		}

		const std::string expected_requirement_description = rsx::metal::describe_pipeline_entry_requirements(requirement_mask);
		if (metadata.requirement_description != expected_requirement_description)
		{
			return fmt::format("requirement description '%s' does not match expected description '%s'",
				metadata.requirement_description,
				expected_requirement_description);
		}

		if (metadata.entry_available != entry_available)
		{
			return fmt::format("entry availability %u does not match requested availability %u",
				static_cast<u32>(metadata.entry_available),
				static_cast<u32>(entry_available));
		}

		return {};
	}

	std::string shader_source_metadata_mismatch(
		const rsx::metal::shader_source_metadata& metadata,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& source_path)
	{
		rsx_log.trace("shader_source_metadata_mismatch(shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, source_path=%s)",
			shader_id,
			source_hash,
			source_text_hash,
			entry_point.c_str(),
			source_path.c_str());

		if (metadata.shader_id != shader_id)
		{
			return fmt::format("shader id %u does not match requested shader id %u",
				metadata.shader_id,
				shader_id);
		}

		if (metadata.source_hash != source_hash)
		{
			return fmt::format("source hash 0x%llx does not match requested source hash 0x%llx",
				metadata.source_hash,
				source_hash);
		}

		if (metadata.source_text_hash != source_text_hash)
		{
			return fmt::format("source text hash 0x%llx does not match requested source text hash 0x%llx",
				metadata.source_text_hash,
				source_text_hash);
		}

		if (metadata.entry_point != entry_point)
		{
			return fmt::format("entry point '%s' does not match requested entry point '%s'",
				metadata.entry_point,
				entry_point);
		}

		if (metadata.source_path != source_path)
		{
			return fmt::format("source path '%s' does not match requested source path '%s'",
				metadata.source_path,
				source_path);
		}

		return {};
	}

	b8 is_valid_shader_source_metadata(const std::string& path, const std::string& cache_version)
	{
		rsx_log.trace("is_valid_shader_source_metadata(path=%s, cache_version=%s)", path, cache_version);

		fs::file file{path, fs::read};
		if (!file)
		{
			return false;
		}

		std::string text = file.to_string();
		std::string_view view = text;
		if (take_metadata_line(view) != "RPCS3 Metal shader source")
		{
			return false;
		}

		std::string field;
		if (!try_get_metadata_field(view, "backend", field) || field != "metal")
		{
			return false;
		}

		if (!try_get_metadata_field(view, "cache_version", field) || field != cache_version)
		{
			return false;
		}

		u32 metadata_u32 = 0;
		if (!try_get_metadata_field(view, "record_version", field) || !try_parse_metadata_u32(field, metadata_u32) || metadata_u32 != 1)
		{
			return false;
		}

		if (!try_get_metadata_field(view, "stage", field) || !is_shader_source_stage(field))
		{
			return false;
		}

		if (!try_get_metadata_field(view, "shader_id", field) || !try_parse_metadata_u32(field, metadata_u32))
		{
			return false;
		}

		u64 metadata_u64 = 0;
		if (!try_get_metadata_field(view, "source_hash", field) || !try_parse_metadata_u64(field, metadata_u64) || !metadata_u64)
		{
			return false;
		}

		u64 metadata_source_text_hash = 0;
		if (!try_get_metadata_field(view, "source_text_hash", field) || !try_parse_metadata_u64(field, metadata_source_text_hash) || !metadata_source_text_hash)
		{
			return false;
		}

		std::string entry_point;
		if (!try_get_metadata_field(view, "entry_point", entry_point) || entry_point.empty())
		{
			return false;
		}

		std::string source_path;
		if (!try_get_metadata_field(view, "source_path", source_path) || source_path.empty())
		{
			return false;
		}

		u64 actual_source_text_hash = 0;
		return get_file_text_hash(source_path, actual_source_text_hash) && actual_source_text_hash == metadata_source_text_hash;
	}

	b8 is_valid_pipeline_entry_metadata(const std::string& path, const std::string& cache_version)
	{
		rsx_log.trace("is_valid_pipeline_entry_metadata(path=%s, cache_version=%s)", path, cache_version);

		fs::file file{path, fs::read};
		if (!file)
		{
			return false;
		}

		std::string text = file.to_string();
		std::string_view view = text;
		if (take_metadata_line(view) != "RPCS3 Metal pipeline entry")
		{
			return false;
		}

		std::string field;
		if (!try_get_metadata_field(view, "backend", field) || field != "metal")
		{
			return false;
		}

		if (!try_get_metadata_field(view, "cache_version", field) || field != cache_version)
		{
			return false;
		}

		u32 metadata_u32 = 0;
		if (!try_get_metadata_field(view, "record_version", field) || !try_parse_metadata_u32(field, metadata_u32) || (metadata_u32 != 2 && metadata_u32 != 3))
		{
			return false;
		}
		const u32 record_version = metadata_u32;

		if (!try_get_metadata_field(view, "stage", field) || !is_pipeline_entry_stage(field))
		{
			return false;
		}
		const std::string stage = field;

		if (!try_get_metadata_field(view, "shader_id", field) || !try_parse_metadata_u32(field, metadata_u32))
		{
			return false;
		}

		u64 source_hash = 0;
		if (!try_get_metadata_field(view, "source_hash", field) || !try_parse_metadata_u64(field, source_hash) || !source_hash)
		{
			return false;
		}

		u64 pipeline_source_hash = 0;
		if (!try_get_metadata_field(view, "pipeline_source_hash", field) || !try_parse_metadata_u64(field, pipeline_source_hash))
		{
			return false;
		}

		b8 entry_available = false;
		if (!try_get_metadata_field(view, "entry_available", field) || !try_parse_metadata_b8(field, entry_available))
		{
			return false;
		}

		u32 requirement_mask = 0;
		if (!try_get_metadata_field(view, "requirement_mask", field) || !try_parse_metadata_u32(field, requirement_mask))
		{
			return false;
		}

		if (!is_pipeline_entry_requirement_mask_for_stage(stage, requirement_mask))
		{
			return false;
		}

		if (record_version >= 3)
		{
			std::string requirement_description;
			if (!try_get_metadata_field(view, "requirement_description", requirement_description) ||
				requirement_description != rsx::metal::describe_pipeline_entry_requirements(requirement_mask))
			{
				return false;
			}
		}

		std::string entry_point;
		if (!try_get_metadata_field(view, "entry_point", entry_point))
		{
			return false;
		}

		std::string source_path;
		if (!try_get_metadata_field(view, "source_path", source_path))
		{
			return false;
		}

		std::string entry_error;
		if (!try_get_metadata_field(view, "entry_error", entry_error))
		{
			return false;
		}

		if (entry_available)
		{
			if (!pipeline_source_hash || requirement_mask || entry_point.empty() || source_path.empty() || !entry_error.empty())
			{
				return false;
			}

			u64 pipeline_source_text_hash = 0;
			return get_file_text_hash(source_path, pipeline_source_text_hash) &&
				pipeline_source_text_hash == pipeline_source_hash &&
				validate_available_pipeline_entry_source(stage, pipeline_source_hash, entry_point, source_path);
		}

		return !pipeline_source_hash &&
			entry_point.empty() &&
			source_path.empty() &&
			requirement_mask &&
			!entry_error.empty();
	}

	b8 is_valid_shader_library_metadata(const std::string& path, const std::string& cache_version)
	{
		rsx_log.trace("is_valid_shader_library_metadata(path=%s, cache_version=%s)", path, cache_version);

		fs::file file{path, fs::read};
		if (!file)
		{
			return false;
		}

		std::string text = file.to_string();
		std::string_view view = text;
		if (take_metadata_line(view) != "RPCS3 Metal shader library")
		{
			return false;
		}

		std::string field;
		if (!try_get_metadata_field(view, "backend", field) || field != "metal")
		{
			return false;
		}

		if (!try_get_metadata_field(view, "cache_version", field) || field != cache_version)
		{
			return false;
		}

		u32 metadata_u32 = 0;
		if (!try_get_metadata_field(view, "record_version", field) || !try_parse_metadata_u32(field, metadata_u32) || metadata_u32 != 4)
		{
			return false;
		}

		if (!try_get_metadata_field(view, "stage", field) || !is_shader_library_stage(field))
		{
			return false;
		}
		const std::string stage = field;

		if (!try_get_metadata_field(view, "shader_id", field) || !try_parse_metadata_u32(field, metadata_u32))
		{
			return false;
		}

		u64 metadata_u64 = 0;
		if (!try_get_metadata_field(view, "source_hash", field) || !try_parse_metadata_u64(field, metadata_u64) || !metadata_u64)
		{
			return false;
		}

		if (!try_get_metadata_field(view, "source_text_hash", field) || !try_parse_metadata_u64(field, metadata_u64) || !metadata_u64)
		{
			return false;
		}

		std::string entry_point;
		if (!try_get_metadata_field(view, "entry_point", entry_point) || entry_point.empty())
		{
			return false;
		}

		std::string library_path;
		if (!try_get_metadata_field(view, "library_path", library_path) || library_path.empty())
		{
			return false;
		}

		u64 library_size = 0;
		if (!try_get_metadata_field(view, "library_size", field) || !try_parse_metadata_u64(field, library_size) || !library_size)
		{
			return false;
		}

		u64 library_hash = 0;
		if (!try_get_metadata_field(view, "library_hash", field) || !try_parse_metadata_u64(field, library_hash) || !library_hash)
		{
			return false;
		}

		b8 entry_available = false;
		if (!try_get_metadata_field(view, "pipeline_entry_available", field) || !try_parse_metadata_b8(field, entry_available))
		{
			return false;
		}

		u32 requirement_mask = 0;
		if (!try_get_metadata_field(view, "pipeline_requirement_mask", field) || !try_parse_metadata_u32(field, requirement_mask))
		{
			return false;
		}

		if (!is_pipeline_entry_requirement_mask_for_stage(stage, requirement_mask))
		{
			return false;
		}

		std::string requirement_description;
		if (!try_get_metadata_field(view, "pipeline_requirement_description", requirement_description) ||
			requirement_description != rsx::metal::describe_pipeline_entry_requirements(requirement_mask))
		{
			return false;
		}

		std::string pipeline_entry_error;
		if (!try_get_metadata_field(view, "pipeline_entry_error", pipeline_entry_error))
		{
			return false;
		}

		if (entry_available ? (!!requirement_mask || !pipeline_entry_error.empty()) : (!requirement_mask || pipeline_entry_error.empty()))
		{
			return false;
		}

		fs::stat_t library_stat{};
		u64 actual_library_hash = 0;
		return fs::get_stat(library_path, library_stat) &&
			!library_stat.is_directory &&
			library_stat.size == library_size &&
			get_file_content_hash(library_path, actual_library_hash) &&
			actual_library_hash == library_hash;
	}

	b8 is_valid_shader_completion_metadata(const std::string& path, const std::string& cache_version)
	{
		rsx_log.trace("is_valid_shader_completion_metadata(path=%s, cache_version=%s)", path, cache_version);

		fs::file file{path, fs::read};
		if (!file)
		{
			return false;
		}

		std::string text = file.to_string();
		std::string_view view = text;
		if (take_metadata_line(view) != "RPCS3 Metal shader completion")
		{
			return false;
		}

		std::string field;
		if (!try_get_metadata_field(view, "backend", field) || field != "metal")
		{
			return false;
		}

		if (!try_get_metadata_field(view, "cache_version", field) || field != cache_version)
		{
			return false;
		}

		u32 metadata_u32 = 0;
		if (!try_get_metadata_field(view, "record_version", field) || !try_parse_metadata_u32(field, metadata_u32) || metadata_u32 != 2)
		{
			return false;
		}

		if (!try_get_metadata_field(view, "stage", field) || !is_shader_completion_stage(field))
		{
			return false;
		}
		const std::string stage = field;

		if (!try_get_metadata_field(view, "shader_id", field) || !try_parse_metadata_u32(field, metadata_u32))
		{
			return false;
		}

		u64 source_hash = 0;
		if (!try_get_metadata_field(view, "source_hash", field) || !try_parse_metadata_u64(field, source_hash) || !source_hash)
		{
			return false;
		}

		u64 source_text_hash = 0;
		if (!try_get_metadata_field(view, "source_text_hash", field) || !try_parse_metadata_u64(field, source_text_hash) || !source_text_hash)
		{
			return false;
		}

		std::string entry_point;
		if (!try_get_metadata_field(view, "entry_point", entry_point) || entry_point.empty())
		{
			return false;
		}

		std::string source_path;
		if (!try_get_metadata_field(view, "source_path", source_path) || source_path.empty())
		{
			return false;
		}

		std::vector<u32> fragment_constant_offsets;
		if (!try_get_metadata_field(view, "fragment_constant_offsets", field) || !try_parse_metadata_u32_list(field, fragment_constant_offsets))
		{
			return false;
		}

		u64 actual_source_text_hash = 0;
		if (!get_file_text_hash(source_path, actual_source_text_hash) || actual_source_text_hash != source_text_hash)
		{
			return false;
		}

		u64 pipeline_source_hash = 0;
		if (!try_get_metadata_field(view, "pipeline_source_hash", field) || !try_parse_metadata_u64(field, pipeline_source_hash))
		{
			return false;
		}

		b8 pipeline_entry_available = false;
		if (!try_get_metadata_field(view, "pipeline_entry_available", field) || !try_parse_metadata_b8(field, pipeline_entry_available))
		{
			return false;
		}

		u32 pipeline_requirement_mask = 0;
		if (!try_get_metadata_field(view, "pipeline_requirement_mask", field) || !try_parse_metadata_u32(field, pipeline_requirement_mask))
		{
			return false;
		}

		if (!is_pipeline_entry_requirement_mask_for_stage(stage, pipeline_requirement_mask))
		{
			return false;
		}

		if (stage != "fragment" && !fragment_constant_offsets.empty())
		{
			return false;
		}

		std::string requirement_description;
		if (!try_get_metadata_field(view, "pipeline_requirement_description", requirement_description) ||
			requirement_description != rsx::metal::describe_pipeline_entry_requirements(pipeline_requirement_mask))
		{
			return false;
		}

		std::string pipeline_entry_point;
		if (!try_get_metadata_field(view, "pipeline_entry_point", pipeline_entry_point))
		{
			return false;
		}

		std::string pipeline_source_path;
		if (!try_get_metadata_field(view, "pipeline_source_path", pipeline_source_path))
		{
			return false;
		}

		std::string pipeline_entry_error;
		if (!try_get_metadata_field(view, "pipeline_entry_error", pipeline_entry_error))
		{
			return false;
		}

		if (pipeline_entry_available)
		{
			u64 actual_pipeline_source_hash = 0;
			return pipeline_source_hash &&
				!pipeline_requirement_mask &&
				!pipeline_entry_point.empty() &&
				!pipeline_source_path.empty() &&
				pipeline_entry_error.empty() &&
				get_file_text_hash(pipeline_source_path, actual_pipeline_source_hash) &&
				actual_pipeline_source_hash == pipeline_source_hash &&
				validate_available_pipeline_entry_source(stage, pipeline_source_hash, pipeline_entry_point, pipeline_source_path);
		}

		return !pipeline_source_hash &&
			pipeline_entry_point.empty() &&
			pipeline_source_path.empty() &&
			pipeline_requirement_mask &&
			!pipeline_entry_error.empty();
	}

	b8 is_valid_shader_translation_failure_metadata(const std::string& path, const std::string& cache_version)
	{
		rsx_log.trace("is_valid_shader_translation_failure_metadata(path=%s, cache_version=%s)", path, cache_version);

		fs::file file{path, fs::read};
		if (!file)
		{
			return false;
		}

		std::string text = file.to_string();
		std::string_view view = text;
		if (take_metadata_line(view) != "RPCS3 Metal shader translation failure")
		{
			return false;
		}

		std::string field;
		if (!try_get_metadata_field(view, "backend", field) || field != "metal")
		{
			return false;
		}

		if (!try_get_metadata_field(view, "cache_version", field) || field != cache_version)
		{
			return false;
		}

		u32 metadata_u32 = 0;
		if (!try_get_metadata_field(view, "record_version", field) || !try_parse_metadata_u32(field, metadata_u32) || metadata_u32 != 1)
		{
			return false;
		}

		if (!try_get_metadata_field(view, "stage", field) || !is_shader_translation_failure_stage(field))
		{
			return false;
		}

		if (!try_get_metadata_field(view, "shader_id", field) || !try_parse_metadata_u32(field, metadata_u32))
		{
			return false;
		}

		u64 metadata_u64 = 0;
		if (!try_get_metadata_field(view, "source_hash", field) || !try_parse_metadata_u64(field, metadata_u64) || !metadata_u64)
		{
			return false;
		}

		std::string failure_reason;
		return try_get_metadata_field(view, "failure_reason", failure_reason) && !failure_reason.empty();
	}

	b8 get_file_size(const std::string& path, u64& size)
	{
		rsx_log.trace("get_file_size(path=%s)", path);

		fs::stat_t stat{};
		if (!fs::get_stat(path, stat) || stat.is_directory)
		{
			return false;
		}

		size = stat.size;
		return true;
	}

	b8 is_valid_pipeline_archive_metadata(
		const std::string& path,
		const std::string& cache_version,
		const std::string& expected_script_path,
		const std::string& expected_archive_path)
	{
		rsx_log.trace("is_valid_pipeline_archive_metadata(path=%s, cache_version=%s, expected_script_path=%s, expected_archive_path=%s)",
			path,
			cache_version,
			expected_script_path,
			expected_archive_path);

		fs::file file{path, fs::read};
		if (!file)
		{
			return false;
		}

		std::string text = file.to_string();
		std::string_view view = text;
		if (take_metadata_line(view) != "RPCS3 Metal pipeline archive")
		{
			return false;
		}

		std::string field;
		if (!try_get_metadata_field(view, "backend", field) || field != "metal")
		{
			return false;
		}

		if (!try_get_metadata_field(view, "cache_version", field) || field != cache_version)
		{
			return false;
		}

		u32 metadata_u32 = 0;
		if (!try_get_metadata_field(view, "record_version", field) || !try_parse_metadata_u32(field, metadata_u32) || metadata_u32 != 2)
		{
			return false;
		}

		std::string script_path;
		if (!try_get_metadata_field(view, "script_path", script_path) || script_path.empty() || script_path != expected_script_path)
		{
			return false;
		}

		std::string archive_path;
		if (!try_get_metadata_field(view, "archive_path", archive_path) || archive_path.empty() || archive_path != expected_archive_path)
		{
			return false;
		}

		u64 script_size = 0;
		if (!try_get_metadata_field(view, "script_size", field) || !try_parse_metadata_u64(field, script_size) || !script_size)
		{
			return false;
		}

		u64 archive_size = 0;
		if (!try_get_metadata_field(view, "archive_size", field) || !try_parse_metadata_u64(field, archive_size) || !archive_size)
		{
			return false;
		}

		u64 script_hash = 0;
		if (!try_get_metadata_field(view, "script_hash", field) || !try_parse_metadata_u64(field, script_hash) || !script_hash)
		{
			return false;
		}

		u64 archive_hash = 0;
		if (!try_get_metadata_field(view, "archive_hash", field) || !try_parse_metadata_u64(field, archive_hash) || !archive_hash)
		{
			return false;
		}

		if (!try_get_metadata_field(view, "flushed_pipeline_count", field) || !try_parse_metadata_u32(field, metadata_u32) || !metadata_u32)
		{
			return false;
		}

		u64 actual_script_size = 0;
		u64 actual_archive_size = 0;
		u64 actual_script_hash = 0;
		u64 actual_archive_hash = 0;
		return get_file_size(script_path, actual_script_size) &&
			get_file_size(archive_path, actual_archive_size) &&
			get_file_content_hash(script_path, actual_script_hash) &&
			get_file_content_hash(archive_path, actual_archive_hash) &&
			actual_script_size == script_size &&
			actual_archive_size == archive_size &&
			actual_script_hash == script_hash &&
			actual_archive_hash == archive_hash;
	}

	b8 is_valid_pipeline_state_metadata(const std::string& path, const std::string& cache_version)
	{
		rsx_log.trace("is_valid_pipeline_state_metadata(path=%s, cache_version=%s)", path, cache_version);

		fs::file file{path, fs::read};
		if (!file)
		{
			return false;
		}

		std::string text = file.to_string();
		std::string_view view = text;
		if (take_metadata_line(view) != "RPCS3 Metal pipeline state")
		{
			return false;
		}

		std::string field;
		if (!try_get_metadata_field(view, "backend", field) || field != "metal")
		{
			return false;
		}

		if (!try_get_metadata_field(view, "cache_version", field) || field != cache_version)
		{
			return false;
		}

		u32 metadata_u32 = 0;
		if (!try_get_metadata_field(view, "record_version", field) || !try_parse_metadata_u32(field, metadata_u32) || metadata_u32 != 2)
		{
			return false;
		}

		std::string pipeline_type;
		if (!try_get_metadata_field(view, "pipeline_type", pipeline_type) || !is_pipeline_state_type(pipeline_type))
		{
			return false;
		}

		u64 pipeline_hash = 0;
		if (!try_get_metadata_field(view, "pipeline_hash", field) || !try_parse_metadata_u64(field, pipeline_hash) || !pipeline_hash)
		{
			return false;
		}

		u64 vertex_source_hash = 0;
		if (!try_get_metadata_field(view, "vertex_source_hash", field) || !try_parse_metadata_u64(field, vertex_source_hash))
		{
			return false;
		}

		u64 fragment_source_hash = 0;
		if (!try_get_metadata_field(view, "fragment_source_hash", field) || !try_parse_metadata_u64(field, fragment_source_hash))
		{
			return false;
		}

		u64 object_source_hash = 0;
		if (!try_get_metadata_field(view, "object_source_hash", field) || !try_parse_metadata_u64(field, object_source_hash))
		{
			return false;
		}

		u64 mesh_source_hash = 0;
		if (!try_get_metadata_field(view, "mesh_source_hash", field) || !try_parse_metadata_u64(field, mesh_source_hash))
		{
			return false;
		}

		u64 linked_library_hash = 0;
		if (!try_get_metadata_field(view, "linked_library_hash", field) || !try_parse_metadata_u64(field, linked_library_hash))
		{
			return false;
		}

		u32 linked_library_count = 0;
		if (!try_get_metadata_field(view, "linked_library_count", field) || !try_parse_metadata_u32(field, linked_library_count))
		{
			return false;
		}

		u32 shader_dependency_count = 0;
		if (!try_get_metadata_field(view, "shader_dependency_count", field) || !try_parse_metadata_u32(field, shader_dependency_count) || !shader_dependency_count)
		{
			return false;
		}

		u32 color_pixel_format = 0;
		if (!try_get_metadata_field(view, "color_pixel_format", field) || !try_parse_metadata_u32(field, color_pixel_format) || !color_pixel_format)
		{
			return false;
		}

		u32 raster_sample_count = 0;
		if (!try_get_metadata_field(view, "raster_sample_count", field) || !try_parse_metadata_u32(field, raster_sample_count) || !raster_sample_count)
		{
			return false;
		}

		b8 rasterization_enabled = false;
		if (!try_get_metadata_field(view, "rasterization_enabled", field) || !try_parse_metadata_b8(field, rasterization_enabled))
		{
			return false;
		}

		if ((linked_library_count && !linked_library_hash) || (!linked_library_count && linked_library_hash))
		{
			return false;
		}

		const u32 expected_shader_dependency_count = pipeline_state_shader_dependency_count(
			pipeline_type,
			vertex_source_hash,
			fragment_source_hash,
			object_source_hash,
			mesh_source_hash,
			rasterization_enabled);

		return expected_shader_dependency_count && shader_dependency_count == expected_shader_dependency_count;
	}
}

namespace rsx::metal
{
	persistent_shader_cache::persistent_shader_cache(std::string version)
		: m_version(std::move(version))
	{
		rsx_log.notice("rsx::metal::persistent_shader_cache::persistent_shader_cache(version=%s)", m_version);

		if (m_version.empty())
		{
			fmt::throw_exception("Metal shader cache requires a non-empty cache version");
		}
	}

	persistent_shader_cache::~persistent_shader_cache()
	{
		rsx_log.notice("rsx::metal::persistent_shader_cache::~persistent_shader_cache()");
	}

	void persistent_shader_cache::initialize()
	{
		rsx_log.notice("rsx::metal::persistent_shader_cache::initialize(version=%s)", m_version);

		if (g_cfg.video.disable_on_disk_shader_cache)
		{
			fmt::throw_exception("Metal backend requires the on-disk shader cache to be enabled");
		}

		std::string cache_path = rpcs3::cache::get_ppu_cache();
		if (cache_path.empty())
		{
			fmt::throw_exception("Metal backend requires an initialized PPU cache path");
		}

		m_root_path = std::move(cache_path) + "shaders_cache/metal/" + m_version + "/";
		m_raw_shader_path = m_root_path + "raw/";
		m_msl_path = m_root_path + "msl/";
		m_library_path = m_root_path + "libraries/";
		m_completion_path = m_root_path + "completed/";
		m_pipeline_path = m_root_path + "pipelines/";
		m_archive_path = m_root_path + "archives/";
		m_pipeline_script_path = m_root_path + "pipeline_scripts/";
		m_manifest_path = m_root_path + "manifest.txt";
		m_pipeline_script_file_path = m_pipeline_script_path + "pipelines.mtl4script";
		m_pipeline_archive_file_path = m_archive_path + "pipelines.mtl4archive";

		create_directory(m_root_path);
		create_directory(m_raw_shader_path);
		create_directory(m_msl_path);
		create_directory(m_library_path);
		create_directory(m_completion_path);
		create_directory(m_pipeline_path);
		create_directory(m_archive_path);
		create_directory(m_pipeline_script_path);
		validate_manifest();
		refresh_stats();
	}

	void persistent_shader_cache::report() const
	{
		rsx_log.notice("rsx::metal::persistent_shader_cache::report()");
		rsx_log.notice("Metal shader cache root: %s", m_root_path);
		rsx_log.notice("Metal shader cache entries: shaders=%u, completions=%u, translation_failures=%u, libraries=%u, pipelines=%u, archives=%u",
			m_stats.shader_entries,
			m_stats.completion_entries,
			m_stats.translation_failure_entries,
			m_stats.library_entries,
			m_stats.pipeline_entries,
			m_stats.archive_entries);
		rsx_log.notice("Metal shader completion cache: available=%u, gated=%u",
			m_stats.completion_available_entries,
			m_stats.completion_gated_entries);
		rsx_log.notice("Metal pipeline entry cache: available=%u, gated=%u, mesh=%u",
			m_stats.pipeline_entry_available_entries,
			m_stats.pipeline_entry_gated_entries,
			m_stats.mesh_pipeline_entry_entries);
		rsx_log.notice("Metal shader library cache: available=%u, gated=%u",
			m_stats.library_available_entries,
			m_stats.library_gated_entries);
		rsx_log.notice("Metal pipeline state cache: pipelines=%u, mesh=%u",
			m_stats.pipeline_state_entries,
			m_stats.mesh_pipeline_state_entries);

		report_cache_split_mismatch("completion metadata", m_stats.completion_entries, m_stats.completion_available_entries, m_stats.completion_gated_entries);
		report_cache_split_mismatch("pipeline entry metadata", m_stats.pipeline_entries, m_stats.pipeline_entry_available_entries, m_stats.pipeline_entry_gated_entries);
		report_cache_split_mismatch("shader library metadata", m_stats.library_entries, m_stats.library_available_entries, m_stats.library_gated_entries);
		if (m_stats.mesh_pipeline_entry_entries > m_stats.pipeline_entries)
		{
			rsx_log.warning("Metal shader cache mesh pipeline entry count exceeds total pipeline entries: mesh=%u, total=%u",
				m_stats.mesh_pipeline_entry_entries,
				m_stats.pipeline_entries);
		}
		if (m_stats.mesh_pipeline_state_entries > m_stats.pipeline_state_entries)
		{
			rsx_log.warning("Metal shader cache mesh pipeline state count exceeds total pipeline states: mesh=%u, total=%u",
				m_stats.mesh_pipeline_state_entries,
				m_stats.pipeline_state_entries);
		}
	}

	void persistent_shader_cache::store_shader_source_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& source_path)
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::store_shader_source_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, source_path=%s)",
			stage ? stage : "<null>", shader_id, source_hash, source_text_hash, entry_point.c_str(), source_path.c_str());

		if (!is_shader_source_stage(stage ? stage : ""))
		{
			fmt::throw_exception("Metal shader source metadata requires a valid stage");
		}

		if (!source_hash)
		{
			fmt::throw_exception("Metal shader source metadata requires a non-zero source hash");
		}

		if (!source_text_hash)
		{
			fmt::throw_exception("Metal shader source metadata requires a non-zero source text hash");
		}

		if (entry_point.empty())
		{
			fmt::throw_exception("Metal shader source metadata requires an entry point");
		}

		if (source_path.empty())
		{
			fmt::throw_exception("Metal shader source metadata requires a source path");
		}

		u64 actual_source_text_hash = 0;
		if (!get_file_text_hash(source_path, actual_source_text_hash) || actual_source_text_hash != source_text_hash)
		{
			fmt::throw_exception("Metal shader source metadata hash mismatch for '%s'", source_path);
		}

		const std::string path = shader_source_metadata_path(stage, source_hash);
		const std::string metadata = fmt::format(
			"RPCS3 Metal shader source\n"
			"backend=metal\n"
			"cache_version=%s\n"
			"record_version=1\n"
			"stage=%s\n"
			"shader_id=%u\n"
			"source_hash=0x%llx\n"
			"source_text_hash=0x%llx\n"
			"entry_point=%s\n"
			"source_path=%s\n",
			m_version,
			stage,
			shader_id,
			source_hash,
			source_text_hash,
			metadata_value(entry_point),
			metadata_value(source_path));

		if (!fs::write_file(path, fs::rewrite, metadata))
		{
			fmt::throw_exception("Metal shader source metadata write failed for '%s' (%s)", path, fs::g_tls_error);
		}

		shader_source_metadata loaded_metadata;
		if (!find_shader_source_metadata(stage, shader_id, source_hash, source_text_hash, entry_point, source_path, loaded_metadata))
		{
			fmt::throw_exception("Metal shader source metadata lookup failed after writing '%s'", path);
		}

		refresh_stats();
	}

	shader_source_metadata persistent_shader_cache::load_shader_source_metadata(const std::string& path) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::load_shader_source_metadata(path=%s)", path);

		fs::file file{path, fs::read};
		if (!file)
		{
			fmt::throw_exception("Metal shader source metadata '%s' is not readable", path);
		}

		std::string text = file.to_string();
		std::string_view view = text;
		const std::string_view header = take_metadata_line(view);
		if (header != "RPCS3 Metal shader source")
		{
			fmt::throw_exception("Metal shader source metadata '%s' has an invalid header", path);
		}

		const std::string backend = get_metadata_field(view, "backend");
		if (backend != "metal")
		{
			fmt::throw_exception("Metal shader source metadata '%s' has incompatible backend '%s'", path, backend);
		}

		const std::string cache_version = get_metadata_field(view, "cache_version");
		if (cache_version != m_version)
		{
			fmt::throw_exception("Metal shader source metadata '%s' has incompatible cache version '%s'", path, cache_version);
		}

		const u32 record_version = parse_metadata_u32(get_metadata_field(view, "record_version"), "record_version");
		if (record_version != 1)
		{
			fmt::throw_exception("Metal shader source metadata '%s' has unsupported record version %u", path, record_version);
		}

		shader_source_metadata metadata =
		{
			.stage = get_metadata_field(view, "stage"),
			.shader_id = parse_metadata_u32(get_metadata_field(view, "shader_id"), "shader_id"),
			.source_hash = parse_metadata_u64(get_metadata_field(view, "source_hash"), "source_hash"),
			.source_text_hash = parse_metadata_u64(get_metadata_field(view, "source_text_hash"), "source_text_hash"),
			.entry_point = get_metadata_field(view, "entry_point"),
			.source_path = get_metadata_field(view, "source_path"),
		};

		if (!is_shader_source_stage(metadata.stage))
		{
			fmt::throw_exception("Metal shader source metadata '%s' has invalid stage '%s'", path, metadata.stage);
		}

		if (!metadata.source_hash || !metadata.source_text_hash)
		{
			fmt::throw_exception("Metal shader source metadata '%s' has a zero hash", path);
		}

		if (metadata.entry_point.empty())
		{
			fmt::throw_exception("Metal shader source metadata '%s' has an empty entry point", path);
		}

		u64 actual_source_text_hash = 0;
		if (metadata.source_path.empty() ||
			!get_file_text_hash(metadata.source_path, actual_source_text_hash) ||
			actual_source_text_hash != metadata.source_text_hash)
		{
			fmt::throw_exception("Metal shader source metadata '%s' points to an invalid source '%s'",
				path, metadata.source_path);
		}

		return metadata;
	}

	b8 persistent_shader_cache::find_shader_source_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& source_path,
		shader_source_metadata& metadata) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::find_shader_source_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, source_path=%s)",
			stage ? stage : "<null>", shader_id, source_hash, source_text_hash, entry_point.c_str(), source_path.c_str());

		const std::string path = shader_source_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		shader_source_metadata loaded_metadata = load_shader_source_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			fmt::throw_exception("Metal shader source metadata '%s' has stage '%s' but lookup requested '%s'",
				path, loaded_metadata.stage, stage);
		}

		if (const std::string mismatch = shader_source_metadata_mismatch(
			loaded_metadata,
			shader_id,
			source_hash,
			source_text_hash,
			entry_point,
			source_path);
			!mismatch.empty())
		{
			rsx_log.warning("Metal shader source metadata mismatch for '%s': %s", path, mismatch);
			return false;
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	b8 persistent_shader_cache::try_find_shader_source_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& source_path,
		shader_source_metadata& metadata,
		std::string& error) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::try_find_shader_source_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, source_path=%s)",
			stage ? stage : "<null>",
			shader_id,
			source_hash,
			source_text_hash,
			entry_point.c_str(),
			source_path.c_str());

		error.clear();

		const std::string path = shader_source_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		if (!is_valid_shader_source_metadata(path, m_version))
		{
			error = fmt::format("invalid or stale shader source metadata '%s'", path);
			return false;
		}

		shader_source_metadata loaded_metadata = load_shader_source_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			error = fmt::format("metadata stage '%s' does not match requested stage '%s'", loaded_metadata.stage, stage);
			return false;
		}

		if (const std::string mismatch = shader_source_metadata_mismatch(
			loaded_metadata,
			shader_id,
			source_hash,
			source_text_hash,
			entry_point,
			source_path);
			!mismatch.empty())
		{
			error = fmt::format("metadata fields do not match requested shader source '%s': %s", path, mismatch);
			return false;
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	void persistent_shader_cache::store_pipeline_entry_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 pipeline_source_hash,
		const std::string& entry_point,
		const std::string& source_path,
		const std::string& entry_error,
		u32 requirement_mask,
		b8 entry_available)
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::store_pipeline_entry_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, pipeline_source_hash=0x%llx, entry_point=%s, source_path=%s, entry_error=%s, requirement_mask=0x%x, entry_available=%u)",
			stage ? stage : "<null>", shader_id, source_hash, pipeline_source_hash, entry_point.c_str(), source_path.c_str(), entry_error.c_str(), requirement_mask, static_cast<u32>(entry_available));

		if (!is_pipeline_entry_stage(stage ? stage : ""))
		{
			fmt::throw_exception("Metal pipeline entry metadata requires a valid stage");
		}

		validate_metadata_pipeline_entry_requirement_mask_for_stage(stage ? stage : "", requirement_mask, "pipeline entry");

		if (!source_hash)
		{
			fmt::throw_exception("Metal pipeline entry metadata requires a non-zero source hash");
		}

		if (entry_available)
		{
			if (!pipeline_source_hash || requirement_mask || entry_point.empty() || source_path.empty() || !entry_error.empty())
			{
				fmt::throw_exception("Metal pipeline entry metadata marks an entry available but the executable source record is incomplete");
			}

			u64 actual_pipeline_source_hash = 0;
			if (!get_file_text_hash(source_path, actual_pipeline_source_hash) || actual_pipeline_source_hash != pipeline_source_hash)
			{
				fmt::throw_exception("Metal pipeline entry metadata pipeline source hash mismatch for '%s'", source_path);
			}

			if (!validate_available_pipeline_entry_source(stage, pipeline_source_hash, entry_point, source_path))
			{
				fmt::throw_exception("Metal pipeline entry metadata references an invalid pipeline entry source '%s'", source_path);
			}
		}
		else if (pipeline_source_hash || !entry_point.empty() || !source_path.empty() || !requirement_mask || entry_error.empty())
		{
			fmt::throw_exception("Metal pipeline entry metadata marks an entry gated but the requirement record is incomplete");
		}

		const std::string path = pipeline_entry_metadata_path(stage, source_hash);
		const std::string requirement_description = describe_pipeline_entry_requirements(requirement_mask);
		const std::string metadata = fmt::format(
			"RPCS3 Metal pipeline entry\n"
			"backend=metal\n"
			"cache_version=%s\n"
			"record_version=3\n"
			"stage=%s\n"
			"shader_id=%u\n"
			"source_hash=0x%llx\n"
			"pipeline_source_hash=0x%llx\n"
			"entry_available=%u\n"
			"requirement_mask=0x%x\n"
			"requirement_description=%s\n"
			"entry_point=%s\n"
			"source_path=%s\n"
			"entry_error=%s\n",
			m_version,
			stage,
			shader_id,
			source_hash,
			pipeline_source_hash,
			static_cast<u32>(entry_available),
			requirement_mask,
			metadata_value(requirement_description),
			metadata_value(entry_point),
			metadata_value(source_path),
			metadata_value(entry_error));

		if (!fs::write_file(path, fs::rewrite, metadata))
		{
			fmt::throw_exception("Metal pipeline entry metadata write failed for '%s' (%s)", path, fs::g_tls_error);
		}

		pipeline_entry_metadata loaded_metadata;
		if (!find_pipeline_entry_metadata(stage, source_hash, loaded_metadata))
		{
			fmt::throw_exception("Metal pipeline entry metadata write could not be found after writing '%s'", path);
		}

		if (const std::string mismatch = pipeline_entry_metadata_mismatch(
			loaded_metadata,
			shader_id,
			source_hash,
			pipeline_source_hash,
			entry_point,
			source_path,
			entry_error,
			requirement_mask,
			entry_available);
			!mismatch.empty())
		{
			fmt::throw_exception("Metal pipeline entry metadata changed after writing '%s': %s", path, mismatch);
		}

		refresh_stats();
	}

	pipeline_entry_metadata persistent_shader_cache::load_pipeline_entry_metadata(const std::string& path) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::load_pipeline_entry_metadata(path=%s)", path);

		fs::file file{path, fs::read};
		if (!file)
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' is not readable", path);
		}

		std::string text = file.to_string();
		std::string_view view = text;
		const std::string_view header = take_metadata_line(view);
		if (header != "RPCS3 Metal pipeline entry")
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' has an invalid header", path);
		}

		const std::string backend = get_metadata_field(view, "backend");
		if (backend != "metal")
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' has incompatible backend '%s'", path, backend);
		}

		const std::string cache_version = get_metadata_field(view, "cache_version");
		if (cache_version != m_version)
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' has incompatible cache version '%s'", path, cache_version);
		}

		const u32 record_version = parse_metadata_u32(get_metadata_field(view, "record_version"), "record_version");
		if (record_version != 2 && record_version != 3)
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' has unsupported record version %u", path, record_version);
		}

		std::string requirement_description;
		if (record_version >= 3)
		{
			requirement_description = get_metadata_field(view, "requirement_description");
		}

		pipeline_entry_metadata metadata =
		{
			.stage = get_metadata_field(view, "stage"),
			.shader_id = parse_metadata_u32(get_metadata_field(view, "shader_id"), "shader_id"),
			.source_hash = parse_metadata_u64(get_metadata_field(view, "source_hash"), "source_hash"),
			.pipeline_source_hash = parse_metadata_u64(get_metadata_field(view, "pipeline_source_hash"), "pipeline_source_hash"),
			.entry_point = get_metadata_field(view, "entry_point"),
			.source_path = get_metadata_field(view, "source_path"),
			.entry_error = get_metadata_field(view, "entry_error"),
			.requirement_mask = parse_metadata_u32(get_metadata_field(view, "requirement_mask"), "requirement_mask"),
			.requirement_description = std::move(requirement_description),
			.entry_available = parse_metadata_b8(get_metadata_field(view, "entry_available"), "entry_available"),
		};

		if (metadata.stage != "vp" && metadata.stage != "fp" && metadata.stage != "mesh")
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' has invalid stage '%s'", path, metadata.stage);
		}

		if (!metadata.source_hash)
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' has a zero source hash", path);
		}

		validate_metadata_pipeline_entry_requirement_mask_for_stage(metadata.stage, metadata.requirement_mask, "pipeline entry");

		if (metadata.entry_available)
		{
			if (!metadata.pipeline_source_hash || metadata.requirement_mask || metadata.entry_point.empty() || metadata.source_path.empty() || !metadata.entry_error.empty())
			{
				fmt::throw_exception("Metal pipeline entry metadata '%s' marks an entry available without executable source data", path);
			}

			u64 pipeline_source_text_hash = 0;
			if (!get_file_text_hash(metadata.source_path, pipeline_source_text_hash) || pipeline_source_text_hash != metadata.pipeline_source_hash)
			{
				fmt::throw_exception("Metal pipeline entry metadata '%s' references stale pipeline source '%s'", path, metadata.source_path);
			}

			if (!validate_available_pipeline_entry_source(metadata.stage, metadata.pipeline_source_hash, metadata.entry_point, metadata.source_path))
			{
				fmt::throw_exception("Metal pipeline entry metadata '%s' references an invalid pipeline entry source '%s'", path, metadata.source_path);
			}
		}
		else if (metadata.pipeline_source_hash || !metadata.entry_point.empty() || !metadata.source_path.empty() || !metadata.requirement_mask || metadata.entry_error.empty())
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' marks an entry gated without requirements or a reason", path);
		}

		const std::string expected_requirement_description = describe_pipeline_entry_requirements(metadata.requirement_mask);
		if (metadata.requirement_description.empty())
		{
			metadata.requirement_description = expected_requirement_description;
		}
		else if (metadata.requirement_description != expected_requirement_description)
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' has requirement description '%s' but expected '%s'",
				path, metadata.requirement_description, expected_requirement_description);
		}

		return metadata;
	}

	b8 persistent_shader_cache::find_pipeline_entry_metadata(const char* stage, u64 source_hash, pipeline_entry_metadata& metadata) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::find_pipeline_entry_metadata(stage=%s, source_hash=0x%llx)",
			stage ? stage : "<null>", source_hash);

		const std::string path = pipeline_entry_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		pipeline_entry_metadata loaded_metadata = load_pipeline_entry_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' has stage '%s' but lookup requested '%s'",
				path, loaded_metadata.stage, stage);
		}

		if (loaded_metadata.source_hash != source_hash)
		{
			fmt::throw_exception("Metal pipeline entry metadata '%s' has source hash 0x%llx but lookup requested 0x%llx",
				path, loaded_metadata.source_hash, source_hash);
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	b8 persistent_shader_cache::try_find_pipeline_entry_metadata(
		const char* stage,
		u64 source_hash,
		pipeline_entry_metadata& metadata,
		std::string& error) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::try_find_pipeline_entry_metadata(stage=%s, source_hash=0x%llx)",
			stage ? stage : "<null>", source_hash);

		error.clear();

		const std::string path = pipeline_entry_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		if (!is_valid_pipeline_entry_metadata(path, m_version))
		{
			error = fmt::format("invalid or stale pipeline entry metadata '%s'", path);
			return false;
		}

		pipeline_entry_metadata loaded_metadata = load_pipeline_entry_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			error = fmt::format("metadata stage '%s' does not match requested stage '%s'", loaded_metadata.stage, stage);
			return false;
		}

		if (loaded_metadata.source_hash != source_hash)
		{
			error = fmt::format("metadata source hash 0x%llx does not match requested source hash 0x%llx",
				loaded_metadata.source_hash,
				source_hash);
			return false;
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	void persistent_shader_cache::store_shader_library_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& library_path,
		const std::string& pipeline_entry_error,
		u32 pipeline_requirement_mask,
		b8 pipeline_entry_available)
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::store_shader_library_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, library_path=%s, pipeline_requirement_mask=0x%x, pipeline_entry_available=%u)",
			stage ? stage : "<null>",
			shader_id,
			source_hash,
			source_text_hash,
			entry_point.c_str(),
			library_path.c_str(),
			pipeline_requirement_mask,
			static_cast<u32>(pipeline_entry_available));

		if (!source_text_hash)
		{
			fmt::throw_exception("Metal shader library metadata requires a non-zero source text hash");
		}

		if (!is_shader_library_stage(stage ? stage : ""))
		{
			fmt::throw_exception("Metal shader library metadata requires a valid stage");
		}

		if (!source_hash)
		{
			fmt::throw_exception("Metal shader library metadata requires a non-zero source hash");
		}

		validate_metadata_pipeline_entry_requirement_mask_for_stage(stage ? stage : "", pipeline_requirement_mask, "shader library");

		if (entry_point.empty())
		{
			fmt::throw_exception("Metal shader library metadata requires an entry point");
		}

		if (library_path.empty())
		{
			fmt::throw_exception("Metal shader library metadata requires a library path");
		}

		if (pipeline_entry_available && (pipeline_requirement_mask || !pipeline_entry_error.empty()))
		{
			fmt::throw_exception("Metal shader library metadata cannot mark a pipeline entry available with gated state");
		}

		if (!pipeline_entry_available && (!pipeline_requirement_mask || pipeline_entry_error.empty()))
		{
			fmt::throw_exception("Metal shader library metadata requires gated pipeline-entry requirements and a diagnostic reason");
		}

		fs::stat_t library_stat{};
		if (!fs::get_stat(library_path, library_stat) || library_stat.is_directory || !library_stat.size)
		{
			fmt::throw_exception("Metal shader library metadata requires a non-empty library '%s'", library_path);
		}

		u64 library_hash = 0;
		if (!get_file_content_hash(library_path, library_hash))
		{
			fmt::throw_exception("Metal shader library metadata could not hash library '%s'", library_path);
		}

		const std::string path = shader_library_metadata_path(stage, source_hash);
		const std::string pipeline_requirement_description = describe_pipeline_entry_requirements(pipeline_requirement_mask);
		const std::string metadata = fmt::format(
			"RPCS3 Metal shader library\n"
			"backend=metal\n"
			"cache_version=%s\n"
			"record_version=4\n"
			"stage=%s\n"
			"shader_id=%u\n"
			"source_hash=0x%llx\n"
			"source_text_hash=0x%llx\n"
			"entry_point=%s\n"
			"library_path=%s\n"
			"library_size=0x%llx\n"
			"library_hash=0x%llx\n"
			"pipeline_entry_available=%u\n"
			"pipeline_requirement_mask=0x%x\n"
			"pipeline_requirement_description=%s\n"
			"pipeline_entry_error=%s\n",
			m_version,
			stage,
			shader_id,
			source_hash,
			source_text_hash,
			metadata_value(entry_point),
			metadata_value(library_path),
			library_stat.size,
			library_hash,
			static_cast<u32>(pipeline_entry_available),
			pipeline_requirement_mask,
			metadata_value(pipeline_requirement_description),
			metadata_value(pipeline_entry_error));

		if (!fs::write_file(path, fs::rewrite, metadata))
		{
			fmt::throw_exception("Metal shader library metadata write failed for '%s' (%s)", path, fs::g_tls_error);
		}

		shader_library_metadata loaded_metadata;
		if (!find_shader_library_metadata(
			stage,
			shader_id,
			source_hash,
			source_text_hash,
			entry_point,
			library_path,
			pipeline_entry_error,
			pipeline_requirement_mask,
			pipeline_entry_available,
			loaded_metadata))
		{
			fmt::throw_exception("Metal shader library metadata lookup failed after writing '%s'", path);
		}

		refresh_stats();
	}

	shader_library_metadata persistent_shader_cache::load_shader_library_metadata(const std::string& path) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::load_shader_library_metadata(path=%s)", path);

		fs::file file{path, fs::read};
		if (!file)
		{
			fmt::throw_exception("Metal shader library metadata '%s' is not readable", path);
		}

		std::string text = file.to_string();
		std::string_view view = text;
		const std::string_view header = take_metadata_line(view);
		if (header != "RPCS3 Metal shader library")
		{
			fmt::throw_exception("Metal shader library metadata '%s' has an invalid header", path);
		}

		const std::string backend = get_metadata_field(view, "backend");
		if (backend != "metal")
		{
			fmt::throw_exception("Metal shader library metadata '%s' has incompatible backend '%s'", path, backend);
		}

		const std::string cache_version = get_metadata_field(view, "cache_version");
		if (cache_version != m_version)
		{
			fmt::throw_exception("Metal shader library metadata '%s' has incompatible cache version '%s'", path, cache_version);
		}

		const u32 record_version = parse_metadata_u32(get_metadata_field(view, "record_version"), "record_version");
		if (record_version != 4)
		{
			fmt::throw_exception("Metal shader library metadata '%s' has unsupported record version %u", path, record_version);
		}

		const b8 pipeline_entry_available = parse_metadata_b8(get_metadata_field(view, "pipeline_entry_available"), "pipeline_entry_available");
		const u32 pipeline_requirement_mask = parse_metadata_u32(get_metadata_field(view, "pipeline_requirement_mask"), "pipeline_requirement_mask");
		std::string pipeline_requirement_description = get_metadata_field(view, "pipeline_requirement_description");
		std::string pipeline_entry_error = get_metadata_field(view, "pipeline_entry_error");

		shader_library_metadata metadata =
		{
			.stage = get_metadata_field(view, "stage"),
			.shader_id = parse_metadata_u32(get_metadata_field(view, "shader_id"), "shader_id"),
			.source_hash = parse_metadata_u64(get_metadata_field(view, "source_hash"), "source_hash"),
			.source_text_hash = parse_metadata_u64(get_metadata_field(view, "source_text_hash"), "source_text_hash"),
			.entry_point = get_metadata_field(view, "entry_point"),
			.library_path = get_metadata_field(view, "library_path"),
			.library_size = parse_metadata_u64(get_metadata_field(view, "library_size"), "library_size"),
			.library_hash = parse_metadata_u64(get_metadata_field(view, "library_hash"), "library_hash"),
			.pipeline_requirement_mask = pipeline_requirement_mask,
			.pipeline_requirement_description = std::move(pipeline_requirement_description),
			.pipeline_entry_error = std::move(pipeline_entry_error),
			.pipeline_entry_available = pipeline_entry_available,
		};

		if (!is_shader_library_stage(metadata.stage))
		{
			fmt::throw_exception("Metal shader library metadata '%s' has invalid stage '%s'", path, metadata.stage);
		}

		if (!metadata.source_hash || !metadata.source_text_hash)
		{
			fmt::throw_exception("Metal shader library metadata '%s' has a zero hash", path);
		}

		if (metadata.entry_point.empty())
		{
			fmt::throw_exception("Metal shader library metadata '%s' has an empty entry point", path);
		}

		validate_metadata_pipeline_entry_requirement_mask_for_stage(metadata.stage, metadata.pipeline_requirement_mask, "shader library");

		fs::stat_t library_stat{};
		if (metadata.library_path.empty() ||
			!metadata.library_size ||
			!fs::get_stat(metadata.library_path, library_stat) ||
			library_stat.is_directory ||
			library_stat.size != metadata.library_size)
		{
			fmt::throw_exception("Metal shader library metadata '%s' points to a missing library '%s'",
				path, metadata.library_path);
		}

		u64 actual_library_hash = 0;
		if (!metadata.library_hash ||
			!get_file_content_hash(metadata.library_path, actual_library_hash) ||
			actual_library_hash != metadata.library_hash)
		{
			fmt::throw_exception("Metal shader library metadata '%s' has a stale library hash for '%s'",
				path, metadata.library_path);
		}

		const std::string expected_requirement_description = describe_pipeline_entry_requirements(metadata.pipeline_requirement_mask);
		if (metadata.pipeline_requirement_description.empty())
		{
			metadata.pipeline_requirement_description = expected_requirement_description;
		}
		else if (metadata.pipeline_requirement_description != expected_requirement_description)
		{
			fmt::throw_exception("Metal shader library metadata '%s' has pipeline requirement description '%s' but expected '%s'",
				path, metadata.pipeline_requirement_description, expected_requirement_description);
		}

		if (metadata.pipeline_entry_available && (metadata.pipeline_requirement_mask || !metadata.pipeline_entry_error.empty()))
		{
			fmt::throw_exception("Metal shader library metadata '%s' marks a pipeline entry available with gated state", path);
		}

		if (!metadata.pipeline_entry_available && (!metadata.pipeline_requirement_mask || metadata.pipeline_entry_error.empty()))
		{
			fmt::throw_exception("Metal shader library metadata '%s' marks a pipeline entry gated without requirements or a reason", path);
		}

		return metadata;
	}

	b8 persistent_shader_cache::find_shader_library_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& library_path,
		const std::string& pipeline_entry_error,
		u32 pipeline_requirement_mask,
		b8 pipeline_entry_available,
		shader_library_metadata& metadata) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::find_shader_library_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, library_path=%s, pipeline_requirement_mask=0x%x, pipeline_entry_available=%u)",
			stage ? stage : "<null>",
			shader_id,
			source_hash,
			source_text_hash,
			entry_point.c_str(),
			library_path.c_str(),
			pipeline_requirement_mask,
			static_cast<u32>(pipeline_entry_available));

		const std::string path = shader_library_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		shader_library_metadata loaded_metadata = load_shader_library_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			fmt::throw_exception("Metal shader library metadata '%s' has stage '%s' but lookup requested '%s'",
				path, loaded_metadata.stage, stage);
		}

		if (const std::string mismatch = shader_library_metadata_mismatch(
			loaded_metadata,
			shader_id,
			source_hash,
			source_text_hash,
			entry_point,
			library_path,
			pipeline_entry_error,
			pipeline_requirement_mask,
			pipeline_entry_available);
			!mismatch.empty())
		{
			rsx_log.warning("Metal shader library metadata mismatch for '%s': %s", path, mismatch);
			return false;
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	b8 persistent_shader_cache::try_find_shader_library_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& library_path,
		const std::string& pipeline_entry_error,
		u32 pipeline_requirement_mask,
		b8 pipeline_entry_available,
		shader_library_metadata& metadata,
		std::string& error) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::try_find_shader_library_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, library_path=%s, pipeline_requirement_mask=0x%x, pipeline_entry_available=%u)",
			stage ? stage : "<null>",
			shader_id,
			source_hash,
			source_text_hash,
			entry_point.c_str(),
			library_path.c_str(),
			pipeline_requirement_mask,
			static_cast<u32>(pipeline_entry_available));

		error.clear();

		const std::string path = shader_library_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		if (!is_valid_shader_library_metadata(path, m_version))
		{
			error = fmt::format("invalid or stale shader library metadata '%s'", path);
			return false;
		}

		shader_library_metadata loaded_metadata = load_shader_library_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			error = fmt::format("metadata stage '%s' does not match requested stage '%s'", loaded_metadata.stage, stage);
			return false;
		}

		if (const std::string mismatch = shader_library_metadata_mismatch(
			loaded_metadata,
			shader_id,
			source_hash,
			source_text_hash,
			entry_point,
			library_path,
			pipeline_entry_error,
			pipeline_requirement_mask,
			pipeline_entry_available);
			!mismatch.empty())
		{
			error = fmt::format("metadata fields do not match requested shader library '%s': %s", path, mismatch);
			return false;
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	void persistent_shader_cache::store_shader_completion_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& source_path,
		const std::vector<u32>& fragment_constant_offsets,
		u64 pipeline_source_hash,
		const std::string& pipeline_entry_point,
		const std::string& pipeline_source_path,
		const std::string& pipeline_entry_error,
		u32 pipeline_requirement_mask,
		b8 pipeline_entry_available)
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::store_shader_completion_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, fragment_constant_count=%u, pipeline_source_hash=0x%llx, pipeline_requirement_mask=0x%x, pipeline_entry_available=%u)",
			stage ? stage : "<null>",
			shader_id,
			source_hash,
			source_text_hash,
			static_cast<u32>(fragment_constant_offsets.size()),
			pipeline_source_hash,
			pipeline_requirement_mask,
			static_cast<u32>(pipeline_entry_available));

		if (!source_text_hash)
		{
			fmt::throw_exception("Metal shader completion metadata requires a non-zero source text hash");
		}

		if (!is_shader_completion_stage(stage ? stage : ""))
		{
			fmt::throw_exception("Metal shader completion metadata requires a valid stage");
		}

		if (!source_hash)
		{
			fmt::throw_exception("Metal shader completion metadata requires a non-zero source hash");
		}

		validate_metadata_pipeline_entry_requirement_mask_for_stage(stage ? stage : "", pipeline_requirement_mask, "shader completion");

		if (entry_point.empty())
		{
			fmt::throw_exception("Metal shader completion metadata requires a helper entry point");
		}

		if (source_path.empty())
		{
			fmt::throw_exception("Metal shader completion metadata requires a helper source path");
		}

		if ((std::strcmp(stage ? stage : "", "fragment") != 0) && !fragment_constant_offsets.empty())
		{
			fmt::throw_exception("Metal non-fragment shader completion metadata cannot retain fragment constant offsets");
		}

		u64 actual_source_text_hash = 0;
		if (!get_file_text_hash(source_path, actual_source_text_hash) || actual_source_text_hash != source_text_hash)
		{
			fmt::throw_exception("Metal shader completion metadata source hash mismatch for '%s'", source_path);
		}

		if (pipeline_entry_available)
		{
			if (!pipeline_source_hash || pipeline_entry_point.empty() || pipeline_source_path.empty() || !pipeline_entry_error.empty() || pipeline_requirement_mask)
			{
				fmt::throw_exception("Metal shader completion metadata marks a pipeline entry available but the executable source record is incomplete");
			}

			u64 actual_pipeline_source_hash = 0;
			if (!get_file_text_hash(pipeline_source_path, actual_pipeline_source_hash) || actual_pipeline_source_hash != pipeline_source_hash)
			{
				fmt::throw_exception("Metal shader completion metadata pipeline source hash mismatch for '%s'", pipeline_source_path);
			}

			if (!validate_available_pipeline_entry_source(stage ? stage : "", pipeline_source_hash, pipeline_entry_point, pipeline_source_path))
			{
				fmt::throw_exception("Metal shader completion metadata references an invalid pipeline entry source '%s'", pipeline_source_path);
			}
		}
		else if (pipeline_source_hash || !pipeline_entry_point.empty() || !pipeline_source_path.empty() || !pipeline_requirement_mask || pipeline_entry_error.empty())
		{
			fmt::throw_exception("Metal shader completion metadata marks a pipeline entry gated but the requirement record is incomplete");
		}

		const std::string path = shader_completion_metadata_path(stage, source_hash);
		const std::string pipeline_requirement_description = describe_pipeline_entry_requirements(pipeline_requirement_mask);
		const std::string fragment_constant_offset_text = serialize_metadata_u32_list(fragment_constant_offsets);
		const std::string metadata = fmt::format(
			"RPCS3 Metal shader completion\n"
			"backend=metal\n"
			"cache_version=%s\n"
			"record_version=2\n"
			"stage=%s\n"
			"shader_id=%u\n"
			"source_hash=0x%llx\n"
			"source_text_hash=0x%llx\n"
			"entry_point=%s\n"
			"source_path=%s\n"
			"fragment_constant_offsets=%s\n"
			"pipeline_source_hash=0x%llx\n"
			"pipeline_entry_available=%u\n"
			"pipeline_requirement_mask=0x%x\n"
			"pipeline_requirement_description=%s\n"
			"pipeline_entry_point=%s\n"
			"pipeline_source_path=%s\n"
			"pipeline_entry_error=%s\n",
			m_version,
			stage,
			shader_id,
			source_hash,
			source_text_hash,
			metadata_value(entry_point),
			metadata_value(source_path),
			metadata_value(fragment_constant_offset_text),
			pipeline_source_hash,
			static_cast<u32>(pipeline_entry_available),
			pipeline_requirement_mask,
			metadata_value(pipeline_requirement_description),
			metadata_value(pipeline_entry_point),
			metadata_value(pipeline_source_path),
			metadata_value(pipeline_entry_error));

		if (!fs::write_file(path, fs::rewrite, metadata))
		{
			fmt::throw_exception("Metal shader completion metadata write failed for '%s' (%s)", path, fs::g_tls_error);
		}

		shader_completion_metadata loaded_metadata;
		if (!find_shader_completion_metadata(
			stage,
			shader_id,
			source_hash,
			source_text_hash,
			entry_point,
			source_path,
			fragment_constant_offsets,
			pipeline_source_hash,
			pipeline_entry_point,
			pipeline_source_path,
			pipeline_entry_error,
			pipeline_requirement_mask,
			pipeline_entry_available,
			loaded_metadata))
		{
			fmt::throw_exception("Metal shader completion metadata lookup failed after writing '%s'", path);
		}

		refresh_stats();
	}

	shader_completion_metadata persistent_shader_cache::load_shader_completion_metadata(const std::string& path) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::load_shader_completion_metadata(path=%s)", path);

		fs::file file{path, fs::read};
		if (!file)
		{
			fmt::throw_exception("Metal shader completion metadata '%s' is not readable", path);
		}

		std::string text = file.to_string();
		std::string_view view = text;
		const std::string_view header = take_metadata_line(view);
		if (header != "RPCS3 Metal shader completion")
		{
			fmt::throw_exception("Metal shader completion metadata '%s' has an invalid header", path);
		}

		const std::string backend = get_metadata_field(view, "backend");
		if (backend != "metal")
		{
			fmt::throw_exception("Metal shader completion metadata '%s' has incompatible backend '%s'", path, backend);
		}

		const std::string cache_version = get_metadata_field(view, "cache_version");
		if (cache_version != m_version)
		{
			fmt::throw_exception("Metal shader completion metadata '%s' has incompatible cache version '%s'", path, cache_version);
		}

		const u32 record_version = parse_metadata_u32(get_metadata_field(view, "record_version"), "record_version");
		if (record_version != 2)
		{
			fmt::throw_exception("Metal shader completion metadata '%s' has unsupported record version %u", path, record_version);
		}

		shader_completion_metadata metadata =
		{
			.stage = get_metadata_field(view, "stage"),
			.shader_id = parse_metadata_u32(get_metadata_field(view, "shader_id"), "shader_id"),
			.source_hash = parse_metadata_u64(get_metadata_field(view, "source_hash"), "source_hash"),
			.source_text_hash = parse_metadata_u64(get_metadata_field(view, "source_text_hash"), "source_text_hash"),
			.entry_point = get_metadata_field(view, "entry_point"),
			.source_path = get_metadata_field(view, "source_path"),
			.fragment_constant_offsets = parse_metadata_u32_list(get_metadata_field(view, "fragment_constant_offsets"), "fragment_constant_offsets"),
			.pipeline_source_hash = parse_metadata_u64(get_metadata_field(view, "pipeline_source_hash"), "pipeline_source_hash"),
			.pipeline_entry_point = get_metadata_field(view, "pipeline_entry_point"),
			.pipeline_source_path = get_metadata_field(view, "pipeline_source_path"),
			.pipeline_entry_error = get_metadata_field(view, "pipeline_entry_error"),
			.pipeline_requirement_mask = parse_metadata_u32(get_metadata_field(view, "pipeline_requirement_mask"), "pipeline_requirement_mask"),
			.pipeline_requirement_description = get_metadata_field(view, "pipeline_requirement_description"),
			.pipeline_entry_available = parse_metadata_b8(get_metadata_field(view, "pipeline_entry_available"), "pipeline_entry_available"),
		};

		if (!is_shader_completion_stage(metadata.stage))
		{
			fmt::throw_exception("Metal shader completion metadata '%s' has invalid stage '%s'", path, metadata.stage);
		}

		if (!metadata.source_hash || !metadata.source_text_hash)
		{
			fmt::throw_exception("Metal shader completion metadata '%s' has a zero shader hash", path);
		}

		if (metadata.stage != "fragment" && !metadata.fragment_constant_offsets.empty())
		{
			fmt::throw_exception("Metal shader completion metadata '%s' retains fragment constant offsets for non-fragment stage '%s'",
				path,
				metadata.stage);
		}

		if (metadata.entry_point.empty())
		{
			fmt::throw_exception("Metal shader completion metadata '%s' has an empty helper entry point", path);
		}

		validate_metadata_pipeline_entry_requirement_mask_for_stage(metadata.stage, metadata.pipeline_requirement_mask, "shader completion");

		u64 actual_source_text_hash = 0;
		if (metadata.source_path.empty() ||
			!get_file_text_hash(metadata.source_path, actual_source_text_hash) ||
			actual_source_text_hash != metadata.source_text_hash)
		{
			fmt::throw_exception("Metal shader completion metadata '%s' points to an invalid helper source '%s'",
				path, metadata.source_path);
		}

		const std::string expected_requirement_description = describe_pipeline_entry_requirements(metadata.pipeline_requirement_mask);
		if (metadata.pipeline_requirement_description != expected_requirement_description)
		{
			fmt::throw_exception("Metal shader completion metadata '%s' has requirement description '%s' but expected '%s'",
				path, metadata.pipeline_requirement_description, expected_requirement_description);
		}

		if (metadata.pipeline_entry_available)
		{
			if (!metadata.pipeline_source_hash || metadata.pipeline_requirement_mask || metadata.pipeline_entry_point.empty() || metadata.pipeline_source_path.empty() || !metadata.pipeline_entry_error.empty())
			{
				fmt::throw_exception("Metal shader completion metadata '%s' marks a pipeline entry available without executable source data", path);
			}

			u64 actual_pipeline_source_hash = 0;
			if (!get_file_text_hash(metadata.pipeline_source_path, actual_pipeline_source_hash) ||
				actual_pipeline_source_hash != metadata.pipeline_source_hash)
			{
				fmt::throw_exception("Metal shader completion metadata '%s' points to an invalid pipeline source '%s'",
					path, metadata.pipeline_source_path);
			}

			if (!validate_available_pipeline_entry_source(metadata.stage, metadata.pipeline_source_hash, metadata.pipeline_entry_point, metadata.pipeline_source_path))
			{
				fmt::throw_exception("Metal shader completion metadata '%s' references an invalid pipeline entry source '%s'",
					path, metadata.pipeline_source_path);
			}
		}
		else if (metadata.pipeline_source_hash || !metadata.pipeline_entry_point.empty() || !metadata.pipeline_source_path.empty() || !metadata.pipeline_requirement_mask || metadata.pipeline_entry_error.empty())
		{
			fmt::throw_exception("Metal shader completion metadata '%s' marks a pipeline entry gated without requirements or a reason", path);
		}

		return metadata;
	}

	b8 persistent_shader_cache::find_shader_completion_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& source_path,
		const std::vector<u32>& fragment_constant_offsets,
		u64 pipeline_source_hash,
		const std::string& pipeline_entry_point,
		const std::string& pipeline_source_path,
		const std::string& pipeline_entry_error,
		u32 pipeline_requirement_mask,
		b8 pipeline_entry_available,
		shader_completion_metadata& metadata) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::find_shader_completion_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, fragment_constant_count=%u, pipeline_source_hash=0x%llx, pipeline_requirement_mask=0x%x, pipeline_entry_available=%u)",
			stage ? stage : "<null>",
			shader_id,
			source_hash,
			source_text_hash,
			static_cast<u32>(fragment_constant_offsets.size()),
			pipeline_source_hash,
			pipeline_requirement_mask,
			static_cast<u32>(pipeline_entry_available));

		const std::string path = shader_completion_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		shader_completion_metadata loaded_metadata = load_shader_completion_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			fmt::throw_exception("Metal shader completion metadata '%s' has stage '%s' but lookup requested '%s'",
				path, loaded_metadata.stage, stage);
		}

		if (const std::string mismatch = shader_completion_metadata_mismatch(
			loaded_metadata,
			shader_id,
			source_hash,
			source_text_hash,
			entry_point,
			source_path,
			fragment_constant_offsets,
			pipeline_source_hash,
			pipeline_entry_point,
			pipeline_source_path,
			pipeline_entry_error,
			pipeline_requirement_mask,
			pipeline_entry_available);
			!mismatch.empty())
		{
			rsx_log.warning("Metal shader completion metadata mismatch for '%s': %s", path, mismatch);
			return false;
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	b8 persistent_shader_cache::try_find_shader_completion_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& source_path,
		const std::vector<u32>& fragment_constant_offsets,
		u64 pipeline_source_hash,
		const std::string& pipeline_entry_point,
		const std::string& pipeline_source_path,
		const std::string& pipeline_entry_error,
		u32 pipeline_requirement_mask,
		b8 pipeline_entry_available,
		shader_completion_metadata& metadata,
		std::string& error) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::try_find_shader_completion_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, fragment_constant_count=%u, pipeline_source_hash=0x%llx, pipeline_requirement_mask=0x%x, pipeline_entry_available=%u)",
			stage ? stage : "<null>",
			shader_id,
			source_hash,
			source_text_hash,
			static_cast<u32>(fragment_constant_offsets.size()),
			pipeline_source_hash,
			pipeline_requirement_mask,
			static_cast<u32>(pipeline_entry_available));

		error.clear();

		const std::string path = shader_completion_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		if (!is_valid_shader_completion_metadata(path, m_version))
		{
			error = fmt::format("invalid or stale shader completion metadata '%s'", path);
			return false;
		}

		shader_completion_metadata loaded_metadata = load_shader_completion_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			error = fmt::format("metadata stage '%s' does not match requested stage '%s'", loaded_metadata.stage, stage);
			return false;
		}

		if (const std::string mismatch = shader_completion_metadata_mismatch(
			loaded_metadata,
			shader_id,
			source_hash,
			source_text_hash,
			entry_point,
			source_path,
			fragment_constant_offsets,
			pipeline_source_hash,
			pipeline_entry_point,
			pipeline_source_path,
			pipeline_entry_error,
			pipeline_requirement_mask,
			pipeline_entry_available);
			!mismatch.empty())
		{
			error = fmt::format("metadata fields do not match requested shader completion '%s': %s", path, mismatch);
			return false;
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	b8 persistent_shader_cache::try_load_shader_completion_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		shader_completion_metadata& metadata,
		std::string& error) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::try_load_shader_completion_metadata(stage=%s, shader_id=%u, source_hash=0x%llx)",
			stage ? stage : "<null>",
			shader_id,
			source_hash);

		error.clear();

		const std::string path = shader_completion_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		if (!is_valid_shader_completion_metadata(path, m_version))
		{
			error = fmt::format("invalid or stale shader completion metadata '%s'", path);
			return false;
		}

		shader_completion_metadata loaded_metadata = load_shader_completion_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			error = fmt::format("metadata stage '%s' does not match requested stage '%s'", loaded_metadata.stage, stage);
			return false;
		}

		if (loaded_metadata.source_hash != source_hash)
		{
			error = fmt::format("metadata source hash 0x%llx does not match requested source hash 0x%llx",
				loaded_metadata.source_hash,
				source_hash);
			return false;
		}

		if (loaded_metadata.shader_id != shader_id)
		{
			error = fmt::format("metadata shader id %u does not match requested shader id %u",
				loaded_metadata.shader_id,
				shader_id);
			return false;
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	void persistent_shader_cache::store_shader_translation_failure_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		const std::string& failure_reason)
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::store_shader_translation_failure_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, failure_reason=%s)",
			stage ? stage : "<null>", shader_id, source_hash, failure_reason.c_str());

		if (!is_shader_translation_failure_stage(stage ? stage : ""))
		{
			fmt::throw_exception("Metal shader translation failure metadata requires a valid stage");
		}

		if (!source_hash)
		{
			fmt::throw_exception("Metal shader translation failure metadata requires a non-zero source hash");
		}

		if (failure_reason.empty())
		{
			fmt::throw_exception("Metal shader translation failure metadata requires a failure reason");
		}

		const std::string path = shader_translation_failure_metadata_path(stage, source_hash);
		const std::string recorded_failure_reason = metadata_value(failure_reason);
		const std::string metadata = fmt::format(
			"RPCS3 Metal shader translation failure\n"
			"backend=metal\n"
			"cache_version=%s\n"
			"record_version=1\n"
			"stage=%s\n"
			"shader_id=%u\n"
			"source_hash=0x%llx\n"
			"failure_reason=%s\n",
			m_version,
			stage,
			shader_id,
			source_hash,
			recorded_failure_reason);

		if (!fs::write_file(path, fs::rewrite, metadata))
		{
			fmt::throw_exception("Metal shader translation failure metadata write failed for '%s' (%s)", path, fs::g_tls_error);
		}

		shader_translation_failure_metadata loaded_metadata;
		if (!find_shader_translation_failure_metadata(stage, shader_id, source_hash, loaded_metadata))
		{
			fmt::throw_exception("Metal shader translation failure metadata lookup failed after writing '%s'", path);
		}

		if (loaded_metadata.shader_id != shader_id ||
			loaded_metadata.failure_reason != recorded_failure_reason)
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' changed after writing: shader_id=%u expected=%u, reason='%s' expected='%s'",
				path,
				loaded_metadata.shader_id,
				shader_id,
				loaded_metadata.failure_reason,
				recorded_failure_reason);
		}

		refresh_stats();
	}

	shader_translation_failure_metadata persistent_shader_cache::load_shader_translation_failure_metadata(const std::string& path) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::load_shader_translation_failure_metadata(path=%s)", path);

		fs::file file{path, fs::read};
		if (!file)
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' is not readable", path);
		}

		std::string text = file.to_string();
		std::string_view view = text;
		const std::string_view header = take_metadata_line(view);
		if (header != "RPCS3 Metal shader translation failure")
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' has an invalid header", path);
		}

		const std::string backend = get_metadata_field(view, "backend");
		if (backend != "metal")
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' has incompatible backend '%s'", path, backend);
		}

		const std::string cache_version = get_metadata_field(view, "cache_version");
		if (cache_version != m_version)
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' has incompatible cache version '%s'", path, cache_version);
		}

		const u32 record_version = parse_metadata_u32(get_metadata_field(view, "record_version"), "record_version");
		if (record_version != 1)
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' has unsupported record version %u", path, record_version);
		}

		shader_translation_failure_metadata metadata =
		{
			.stage = get_metadata_field(view, "stage"),
			.shader_id = parse_metadata_u32(get_metadata_field(view, "shader_id"), "shader_id"),
			.source_hash = parse_metadata_u64(get_metadata_field(view, "source_hash"), "source_hash"),
			.failure_reason = get_metadata_field(view, "failure_reason"),
		};

		if (!is_shader_translation_failure_stage(metadata.stage))
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' has invalid stage '%s'", path, metadata.stage);
		}

		if (!metadata.source_hash)
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' has a zero source hash", path);
		}

		if (metadata.failure_reason.empty())
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' has an empty failure reason", path);
		}

		return metadata;
	}

	b8 persistent_shader_cache::find_shader_translation_failure_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		shader_translation_failure_metadata& metadata) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::find_shader_translation_failure_metadata(stage=%s, shader_id=%u, source_hash=0x%llx)",
			stage ? stage : "<null>", shader_id, source_hash);

		const std::string path = shader_translation_failure_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		shader_translation_failure_metadata loaded_metadata = load_shader_translation_failure_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' has stage '%s' but lookup requested '%s'",
				path, loaded_metadata.stage, stage);
		}

		if (loaded_metadata.source_hash != source_hash)
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' has source hash 0x%llx but lookup requested 0x%llx",
				path, loaded_metadata.source_hash, source_hash);
		}

		if (loaded_metadata.shader_id != shader_id)
		{
			fmt::throw_exception("Metal shader translation failure metadata '%s' has shader id %u but lookup requested %u",
				path, loaded_metadata.shader_id, shader_id);
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	b8 persistent_shader_cache::try_find_shader_translation_failure_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		shader_translation_failure_metadata& metadata,
		std::string& error) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::try_find_shader_translation_failure_metadata(stage=%s, shader_id=%u, source_hash=0x%llx)",
			stage ? stage : "<null>", shader_id, source_hash);

		error.clear();

		const std::string path = shader_translation_failure_metadata_path(stage, source_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		if (!is_valid_shader_translation_failure_metadata(path, m_version))
		{
			error = fmt::format("invalid or stale shader translation failure metadata '%s'", path);
			return false;
		}

		shader_translation_failure_metadata loaded_metadata = load_shader_translation_failure_metadata(path);
		if (loaded_metadata.stage != stage)
		{
			error = fmt::format("metadata stage '%s' does not match requested stage '%s'", loaded_metadata.stage, stage);
			return false;
		}

		if (loaded_metadata.source_hash != source_hash)
		{
			error = fmt::format("metadata source hash 0x%llx does not match requested source hash 0x%llx",
				loaded_metadata.source_hash,
				source_hash);
			return false;
		}

		if (loaded_metadata.shader_id != shader_id)
		{
			error = fmt::format("metadata shader id %u does not match requested shader id %u",
				loaded_metadata.shader_id,
				shader_id);
			return false;
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	void persistent_shader_cache::store_pipeline_archive_metadata(
		const std::string& script_path,
		const std::string& archive_path,
		u32 flushed_pipeline_count)
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::store_pipeline_archive_metadata(script_path=%s, archive_path=%s, flushed_pipeline_count=%u)",
			script_path, archive_path, flushed_pipeline_count);

		if (!flushed_pipeline_count)
		{
			fmt::throw_exception("Metal pipeline archive metadata requires at least one flushed pipeline");
		}

		if (script_path != m_pipeline_script_file_path)
		{
			fmt::throw_exception("Metal pipeline archive metadata script path '%s' does not match expected path '%s'",
				script_path,
				m_pipeline_script_file_path);
		}

		if (archive_path != m_pipeline_archive_file_path)
		{
			fmt::throw_exception("Metal pipeline archive metadata archive path '%s' does not match expected path '%s'",
				archive_path,
				m_pipeline_archive_file_path);
		}

		u64 script_size = 0;
		if (!get_file_size(script_path, script_size) || !script_size)
		{
			fmt::throw_exception("Metal pipeline archive metadata requires a non-empty script '%s'", script_path);
		}

		u64 archive_size = 0;
		if (!get_file_size(archive_path, archive_size) || !archive_size)
		{
			fmt::throw_exception("Metal pipeline archive metadata requires a non-empty archive '%s'", archive_path);
		}

		u64 script_hash = 0;
		if (!get_file_content_hash(script_path, script_hash))
		{
			fmt::throw_exception("Metal pipeline archive metadata could not hash script '%s'", script_path);
		}

		u64 archive_hash = 0;
		if (!get_file_content_hash(archive_path, archive_hash))
		{
			fmt::throw_exception("Metal pipeline archive metadata could not hash archive '%s'", archive_path);
		}

		const std::string path = pipeline_archive_metadata_path();
		const std::string metadata = fmt::format(
			"RPCS3 Metal pipeline archive\n"
			"backend=metal\n"
			"cache_version=%s\n"
			"record_version=2\n"
			"script_path=%s\n"
			"archive_path=%s\n"
			"script_size=0x%llx\n"
			"archive_size=0x%llx\n"
			"script_hash=0x%llx\n"
			"archive_hash=0x%llx\n"
			"flushed_pipeline_count=%u\n",
			m_version,
			metadata_value(script_path),
			metadata_value(archive_path),
			script_size,
			archive_size,
			script_hash,
			archive_hash,
			flushed_pipeline_count);

		if (!fs::write_file(path, fs::rewrite, metadata))
		{
			fmt::throw_exception("Metal pipeline archive metadata write failed for '%s' (%s)", path, fs::g_tls_error);
		}

		pipeline_archive_metadata loaded_metadata;
		if (!find_pipeline_archive_metadata(loaded_metadata))
		{
			fmt::throw_exception("Metal pipeline archive metadata lookup failed after writing '%s'", path);
		}

		if (loaded_metadata.flushed_pipeline_count != flushed_pipeline_count)
		{
			fmt::throw_exception("Metal pipeline archive metadata has flushed pipeline count %u after writing '%s' but expected %u",
				loaded_metadata.flushed_pipeline_count,
				path,
				flushed_pipeline_count);
		}

		if (loaded_metadata.script_size != script_size ||
			loaded_metadata.archive_size != archive_size ||
			loaded_metadata.script_hash != script_hash ||
			loaded_metadata.archive_hash != archive_hash)
		{
			fmt::throw_exception("Metal pipeline archive metadata changed after writing '%s'", path);
		}

		refresh_stats();
	}

	pipeline_archive_metadata persistent_shader_cache::load_pipeline_archive_metadata(const std::string& path) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::load_pipeline_archive_metadata(path=%s)", path);

		fs::file file{path, fs::read};
		if (!file)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' is not readable", path);
		}

		std::string text = file.to_string();
		std::string_view view = text;
		const std::string_view header = take_metadata_line(view);
		if (header != "RPCS3 Metal pipeline archive")
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has an invalid header", path);
		}

		const std::string backend = get_metadata_field(view, "backend");
		if (backend != "metal")
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has incompatible backend '%s'", path, backend);
		}

		const std::string cache_version = get_metadata_field(view, "cache_version");
		if (cache_version != m_version)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has incompatible cache version '%s'", path, cache_version);
		}

		const u32 record_version = parse_metadata_u32(get_metadata_field(view, "record_version"), "record_version");
		if (record_version != 2)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has unsupported record version %u", path, record_version);
		}

		pipeline_archive_metadata metadata =
		{
			.script_path = get_metadata_field(view, "script_path"),
			.archive_path = get_metadata_field(view, "archive_path"),
			.script_size = parse_metadata_u64(get_metadata_field(view, "script_size"), "script_size"),
			.archive_size = parse_metadata_u64(get_metadata_field(view, "archive_size"), "archive_size"),
			.script_hash = parse_metadata_u64(get_metadata_field(view, "script_hash"), "script_hash"),
			.archive_hash = parse_metadata_u64(get_metadata_field(view, "archive_hash"), "archive_hash"),
			.flushed_pipeline_count = parse_metadata_u32(get_metadata_field(view, "flushed_pipeline_count"), "flushed_pipeline_count"),
		};

		if (metadata.script_path != m_pipeline_script_file_path)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has script path '%s' but expected '%s'",
				path,
				metadata.script_path,
				m_pipeline_script_file_path);
		}

		if (metadata.archive_path != m_pipeline_archive_file_path)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has archive path '%s' but expected '%s'",
				path,
				metadata.archive_path,
				m_pipeline_archive_file_path);
		}

		u64 actual_script_size = 0;
		if (!metadata.script_size || !get_file_size(metadata.script_path, actual_script_size) || actual_script_size != metadata.script_size)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has an invalid script path '%s'", path, metadata.script_path);
		}

		u64 actual_archive_size = 0;
		if (!metadata.archive_size || !get_file_size(metadata.archive_path, actual_archive_size) || actual_archive_size != metadata.archive_size)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has an invalid archive path '%s'", path, metadata.archive_path);
		}

		u64 actual_script_hash = 0;
		if (!metadata.script_hash || !get_file_content_hash(metadata.script_path, actual_script_hash) || actual_script_hash != metadata.script_hash)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has a stale script hash for '%s'", path, metadata.script_path);
		}

		u64 actual_archive_hash = 0;
		if (!metadata.archive_hash || !get_file_content_hash(metadata.archive_path, actual_archive_hash) || actual_archive_hash != metadata.archive_hash)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has a stale archive hash for '%s'", path, metadata.archive_path);
		}

		if (!metadata.flushed_pipeline_count)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has a zero pipeline count", path);
		}

		return metadata;
	}

	b8 persistent_shader_cache::find_pipeline_archive_metadata(pipeline_archive_metadata& metadata) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::find_pipeline_archive_metadata()");

		const std::string path = pipeline_archive_metadata_path();
		if (!fs::is_file(path))
		{
			return false;
		}

		metadata = load_pipeline_archive_metadata(path);
		return true;
	}

	b8 persistent_shader_cache::try_find_pipeline_archive_metadata(pipeline_archive_metadata& metadata, std::string& error) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::try_find_pipeline_archive_metadata()");

		error.clear();

		const std::string path = pipeline_archive_metadata_path();
		if (!fs::is_file(path))
		{
			return false;
		}

		if (!is_valid_pipeline_archive_metadata(path, m_version, m_pipeline_script_file_path, m_pipeline_archive_file_path))
		{
			error = fmt::format("invalid or stale pipeline archive metadata '%s'", path);
			return false;
		}

		metadata = load_pipeline_archive_metadata(path);
		return true;
	}

	void persistent_shader_cache::store_pipeline_state_metadata(
		const char* pipeline_type,
		u64 pipeline_hash,
		u64 vertex_source_hash,
		u64 fragment_source_hash,
		u64 object_source_hash,
		u64 mesh_source_hash,
		u64 linked_library_hash,
		u32 linked_library_count,
		u32 color_pixel_format,
		u32 raster_sample_count,
		b8 rasterization_enabled)
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::store_pipeline_state_metadata(pipeline_type=%s, pipeline_hash=0x%llx, vertex=0x%llx, fragment=0x%llx, object=0x%llx, mesh=0x%llx, linked_library_hash=0x%llx, linked_library_count=%u)",
			pipeline_type ? pipeline_type : "<null>",
			pipeline_hash,
			vertex_source_hash,
			fragment_source_hash,
			object_source_hash,
			mesh_source_hash,
			linked_library_hash,
			linked_library_count);

		if (!pipeline_type || !is_pipeline_state_type(pipeline_type))
		{
			fmt::throw_exception("Metal pipeline state metadata requires a valid pipeline type");
		}

		if (!pipeline_hash)
		{
			fmt::throw_exception("Metal pipeline state metadata requires a non-zero pipeline hash");
		}

		if (!color_pixel_format || !raster_sample_count)
		{
			fmt::throw_exception("Metal pipeline state metadata requires color format and sample count");
		}

		if ((linked_library_count && !linked_library_hash) || (!linked_library_count && linked_library_hash))
		{
			fmt::throw_exception("Metal pipeline state metadata has inconsistent linked library dependency state");
		}

		const u32 shader_dependency_count = pipeline_state_shader_dependency_count(
			pipeline_type,
			vertex_source_hash,
			fragment_source_hash,
			object_source_hash,
			mesh_source_hash,
			rasterization_enabled);

		if (!shader_dependency_count)
		{
			fmt::throw_exception("Metal %s pipeline state metadata has invalid shader dependencies", pipeline_type);
		}

		const std::string path = pipeline_state_metadata_path(pipeline_type, pipeline_hash);
		const std::string metadata = fmt::format(
			"RPCS3 Metal pipeline state\n"
			"backend=metal\n"
			"cache_version=%s\n"
			"record_version=2\n"
			"pipeline_type=%s\n"
			"pipeline_hash=0x%llx\n"
			"vertex_source_hash=0x%llx\n"
			"fragment_source_hash=0x%llx\n"
			"object_source_hash=0x%llx\n"
			"mesh_source_hash=0x%llx\n"
			"linked_library_hash=0x%llx\n"
			"linked_library_count=%u\n"
			"shader_dependency_count=%u\n"
			"color_pixel_format=%u\n"
			"raster_sample_count=%u\n"
			"rasterization_enabled=%u\n",
			m_version,
			metadata_value(pipeline_type),
			pipeline_hash,
			vertex_source_hash,
			fragment_source_hash,
			object_source_hash,
			mesh_source_hash,
			linked_library_hash,
			linked_library_count,
			shader_dependency_count,
			color_pixel_format,
			raster_sample_count,
			static_cast<u32>(rasterization_enabled));

		if (!fs::write_file(path, fs::rewrite, metadata))
		{
			fmt::throw_exception("Metal pipeline state metadata write failed for '%s' (%s)", path, fs::g_tls_error);
		}

		pipeline_state_metadata loaded_metadata;
		if (!find_pipeline_state_metadata(pipeline_type, pipeline_hash, loaded_metadata))
		{
			fmt::throw_exception("Metal pipeline state metadata lookup failed after writing '%s'", path);
		}

		if (loaded_metadata.vertex_source_hash != vertex_source_hash ||
			loaded_metadata.fragment_source_hash != fragment_source_hash ||
			loaded_metadata.object_source_hash != object_source_hash ||
			loaded_metadata.mesh_source_hash != mesh_source_hash ||
			loaded_metadata.linked_library_hash != linked_library_hash ||
			loaded_metadata.linked_library_count != linked_library_count ||
			loaded_metadata.shader_dependency_count != shader_dependency_count ||
			loaded_metadata.color_pixel_format != color_pixel_format ||
			loaded_metadata.raster_sample_count != raster_sample_count ||
			loaded_metadata.rasterization_enabled != rasterization_enabled)
		{
			fmt::throw_exception("Metal pipeline state metadata changed after writing '%s'", path);
		}

		refresh_stats();
	}

	pipeline_state_metadata persistent_shader_cache::load_pipeline_state_metadata(const std::string& path) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::load_pipeline_state_metadata(path=%s)", path);

		fs::file file{path, fs::read};
		if (!file)
		{
			fmt::throw_exception("Metal pipeline state metadata '%s' is not readable", path);
		}

		std::string text = file.to_string();
		std::string_view view = text;
		const std::string_view header = take_metadata_line(view);
		if (header != "RPCS3 Metal pipeline state")
		{
			fmt::throw_exception("Metal pipeline state metadata '%s' has an invalid header", path);
		}

		const std::string backend = get_metadata_field(view, "backend");
		if (backend != "metal")
		{
			fmt::throw_exception("Metal pipeline state metadata '%s' has incompatible backend '%s'", path, backend);
		}

		const std::string cache_version = get_metadata_field(view, "cache_version");
		if (cache_version != m_version)
		{
			fmt::throw_exception("Metal pipeline state metadata '%s' has incompatible cache version '%s'", path, cache_version);
		}

		const u32 record_version = parse_metadata_u32(get_metadata_field(view, "record_version"), "record_version");
		if (record_version != 2)
		{
			fmt::throw_exception("Metal pipeline state metadata '%s' has unsupported record version %u", path, record_version);
		}

		pipeline_state_metadata metadata =
		{
			.pipeline_type = get_metadata_field(view, "pipeline_type"),
			.pipeline_hash = parse_metadata_u64(get_metadata_field(view, "pipeline_hash"), "pipeline_hash"),
			.vertex_source_hash = parse_metadata_u64(get_metadata_field(view, "vertex_source_hash"), "vertex_source_hash"),
			.fragment_source_hash = parse_metadata_u64(get_metadata_field(view, "fragment_source_hash"), "fragment_source_hash"),
			.object_source_hash = parse_metadata_u64(get_metadata_field(view, "object_source_hash"), "object_source_hash"),
			.mesh_source_hash = parse_metadata_u64(get_metadata_field(view, "mesh_source_hash"), "mesh_source_hash"),
			.linked_library_hash = parse_metadata_u64(get_metadata_field(view, "linked_library_hash"), "linked_library_hash"),
			.linked_library_count = parse_metadata_u32(get_metadata_field(view, "linked_library_count"), "linked_library_count"),
			.shader_dependency_count = parse_metadata_u32(get_metadata_field(view, "shader_dependency_count"), "shader_dependency_count"),
			.color_pixel_format = parse_metadata_u32(get_metadata_field(view, "color_pixel_format"), "color_pixel_format"),
			.raster_sample_count = parse_metadata_u32(get_metadata_field(view, "raster_sample_count"), "raster_sample_count"),
			.rasterization_enabled = parse_metadata_b8(get_metadata_field(view, "rasterization_enabled"), "rasterization_enabled"),
		};

		if (!is_pipeline_state_type(metadata.pipeline_type))
		{
			fmt::throw_exception("Metal pipeline state metadata '%s' has invalid pipeline type '%s'", path, metadata.pipeline_type);
		}

		if (!metadata.pipeline_hash || !metadata.color_pixel_format || !metadata.raster_sample_count)
		{
			fmt::throw_exception("Metal pipeline state metadata '%s' has missing pipeline state data", path);
		}

		if ((metadata.linked_library_count && !metadata.linked_library_hash) || (!metadata.linked_library_count && metadata.linked_library_hash))
		{
			fmt::throw_exception("Metal pipeline state metadata '%s' has inconsistent linked library dependency state", path);
		}

		const u32 expected_shader_dependency_count = pipeline_state_shader_dependency_count(
			metadata.pipeline_type,
			metadata.vertex_source_hash,
			metadata.fragment_source_hash,
			metadata.object_source_hash,
			metadata.mesh_source_hash,
			metadata.rasterization_enabled);

		if (!expected_shader_dependency_count || metadata.shader_dependency_count != expected_shader_dependency_count)
		{
			fmt::throw_exception("Metal %s pipeline state metadata '%s' has invalid shader dependency count: count=%u, expected=%u",
				metadata.pipeline_type,
				path,
				metadata.shader_dependency_count,
				expected_shader_dependency_count);
		}

		return metadata;
	}

	b8 persistent_shader_cache::find_pipeline_state_metadata(
		const char* pipeline_type,
		u64 pipeline_hash,
		pipeline_state_metadata& metadata) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::find_pipeline_state_metadata(pipeline_type=%s, pipeline_hash=0x%llx)",
			pipeline_type ? pipeline_type : "<null>", pipeline_hash);

		const std::string path = pipeline_state_metadata_path(pipeline_type, pipeline_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		pipeline_state_metadata loaded_metadata = load_pipeline_state_metadata(path);
		if (loaded_metadata.pipeline_type != pipeline_type)
		{
			fmt::throw_exception("Metal pipeline state metadata '%s' has type '%s' but lookup requested '%s'",
				path, loaded_metadata.pipeline_type, pipeline_type);
		}

		if (loaded_metadata.pipeline_hash != pipeline_hash)
		{
			fmt::throw_exception("Metal pipeline state metadata '%s' has pipeline hash 0x%llx but lookup requested 0x%llx",
				path, loaded_metadata.pipeline_hash, pipeline_hash);
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	b8 persistent_shader_cache::try_find_pipeline_state_metadata(
		const char* pipeline_type,
		u64 pipeline_hash,
		pipeline_state_metadata& metadata,
		std::string& error) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::try_find_pipeline_state_metadata(pipeline_type=%s, pipeline_hash=0x%llx)",
			pipeline_type ? pipeline_type : "<null>", pipeline_hash);

		error.clear();

		const std::string path = pipeline_state_metadata_path(pipeline_type, pipeline_hash);
		if (!fs::is_file(path))
		{
			return false;
		}

		if (!is_valid_pipeline_state_metadata(path, m_version))
		{
			error = fmt::format("invalid or stale pipeline state metadata '%s'", path);
			return false;
		}

		pipeline_state_metadata loaded_metadata = load_pipeline_state_metadata(path);
		if (loaded_metadata.pipeline_type != pipeline_type)
		{
			error = fmt::format("metadata pipeline type '%s' does not match requested type '%s'",
				loaded_metadata.pipeline_type,
				pipeline_type ? pipeline_type : "<null>");
			return false;
		}

		if (loaded_metadata.pipeline_hash != pipeline_hash)
		{
			error = fmt::format("metadata pipeline hash 0x%llx does not match requested pipeline hash 0x%llx",
				loaded_metadata.pipeline_hash,
				pipeline_hash);
			return false;
		}

		metadata = std::move(loaded_metadata);
		return true;
	}

	const std::string& persistent_shader_cache::root_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::root_path()");
		return m_root_path;
	}

	const std::string& persistent_shader_cache::raw_shader_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::raw_shader_path()");
		return m_raw_shader_path;
	}

	const std::string& persistent_shader_cache::msl_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::msl_path()");
		return m_msl_path;
	}

	const std::string& persistent_shader_cache::library_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::library_path()");
		return m_library_path;
	}

	const std::string& persistent_shader_cache::completion_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::completion_path()");
		return m_completion_path;
	}

	const std::string& persistent_shader_cache::pipeline_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::pipeline_path()");
		return m_pipeline_path;
	}

	const std::string& persistent_shader_cache::archive_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::archive_path()");
		return m_archive_path;
	}

	const std::string& persistent_shader_cache::pipeline_script_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::pipeline_script_path()");
		return m_pipeline_script_path;
	}

	const std::string& persistent_shader_cache::pipeline_script_file_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::pipeline_script_file_path()");
		return m_pipeline_script_file_path;
	}

	const std::string& persistent_shader_cache::pipeline_archive_file_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::pipeline_archive_file_path()");
		return m_pipeline_archive_file_path;
	}

	const shader_cache_stats& persistent_shader_cache::stats() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::stats()");
		return m_stats;
	}

	void persistent_shader_cache::create_directory(const std::string& path) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::create_directory(path=%s)", path);

		if (!fs::create_path(path) && !fs::is_dir(path))
		{
			fmt::throw_exception("Metal shader cache failed to create directory '%s' (%s)", path, fs::g_tls_error);
		}
	}

	void persistent_shader_cache::validate_manifest() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::validate_manifest(path=%s)", m_manifest_path);

		const std::string expected_manifest = make_manifest_text(m_version);

		if (fs::file manifest{m_manifest_path, fs::read})
		{
			if (manifest.to_string() != expected_manifest)
			{
				fmt::throw_exception("Metal shader cache metadata is incompatible at '%s'", m_manifest_path);
			}

			return;
		}

		if (!fs::write_file(m_manifest_path, fs::rewrite, expected_manifest))
		{
			fmt::throw_exception("Metal shader cache failed to write metadata '%s' (%s)", m_manifest_path, fs::g_tls_error);
		}
	}

	std::string persistent_shader_cache::shader_source_metadata_path(const char* stage, u64 source_hash) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::shader_source_metadata_path(stage=%s, source_hash=0x%llx)",
			stage ? stage : "<null>", source_hash);

		if (!stage || !stage[0])
		{
			fmt::throw_exception("Metal shader source metadata requires a shader stage");
		}

		if (!source_hash)
		{
			fmt::throw_exception("Metal shader source metadata requires a non-zero source hash");
		}

		return m_msl_path + fmt::format("%llX.%s.msl.txt", source_hash, stage);
	}

	std::string persistent_shader_cache::pipeline_entry_metadata_path(const char* stage, u64 source_hash) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::pipeline_entry_metadata_path(stage=%s, source_hash=0x%llx)",
			stage ? stage : "<null>", source_hash);

		if (!stage || !stage[0])
		{
			fmt::throw_exception("Metal pipeline entry metadata requires a shader stage");
		}

		if (!source_hash)
		{
			fmt::throw_exception("Metal pipeline entry metadata requires a non-zero source hash");
		}

		return m_pipeline_path + fmt::format("%llX.%s.entry.txt", source_hash, stage);
	}

	std::string persistent_shader_cache::shader_library_metadata_path(const char* stage, u64 source_hash) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::shader_library_metadata_path(stage=%s, source_hash=0x%llx)",
			stage ? stage : "<null>", source_hash);

		if (!stage || !stage[0])
		{
			fmt::throw_exception("Metal shader library metadata requires a shader stage");
		}

		if (!source_hash)
		{
			fmt::throw_exception("Metal shader library metadata requires a non-zero source hash");
		}

		return m_library_path + fmt::format("%llX.%s.metallib.txt", source_hash, stage);
	}

	std::string persistent_shader_cache::shader_completion_metadata_path(const char* stage, u64 source_hash) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::shader_completion_metadata_path(stage=%s, source_hash=0x%llx)",
			stage ? stage : "<null>", source_hash);

		if (!stage || !stage[0])
		{
			fmt::throw_exception("Metal shader completion metadata requires a shader stage");
		}

		if (!source_hash)
		{
			fmt::throw_exception("Metal shader completion metadata requires a non-zero source hash");
		}

		return m_completion_path + fmt::format("%llX.%s.complete.txt", source_hash, stage);
	}

	std::string persistent_shader_cache::shader_translation_failure_metadata_path(const char* stage, u64 source_hash) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::shader_translation_failure_metadata_path(stage=%s, source_hash=0x%llx)",
			stage ? stage : "<null>", source_hash);

		if (!stage || !stage[0])
		{
			fmt::throw_exception("Metal shader translation failure metadata requires a shader stage");
		}

		if (!source_hash)
		{
			fmt::throw_exception("Metal shader translation failure metadata requires a non-zero source hash");
		}

		return m_raw_shader_path + fmt::format("%llX.%s.failure.txt", source_hash, stage);
	}

	std::string persistent_shader_cache::pipeline_archive_metadata_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::pipeline_archive_metadata_path()");
		return m_pipeline_archive_file_path + ".txt";
	}

	std::string persistent_shader_cache::pipeline_state_metadata_path(const char* pipeline_type, u64 pipeline_hash) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::pipeline_state_metadata_path(pipeline_type=%s, pipeline_hash=0x%llx)",
			pipeline_type ? pipeline_type : "<null>", pipeline_hash);

		if (!pipeline_type || !is_pipeline_state_type(pipeline_type))
		{
			fmt::throw_exception("Metal pipeline state metadata requires a valid pipeline type");
		}

		if (!pipeline_hash)
		{
			fmt::throw_exception("Metal pipeline state metadata requires a non-zero pipeline hash");
		}

		return m_pipeline_path + fmt::format("%llX.%s.pipeline.txt", pipeline_hash, pipeline_type);
	}

	u32 persistent_shader_cache::count_shader_source_metadata() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::count_shader_source_metadata(path=%s)", m_msl_path);

		fs::dir dir(m_msl_path);
		if (!dir)
		{
			return 0;
		}

		u32 count = 0;

		for (const fs::dir_entry& entry : dir)
		{
			if (entry.is_directory || !std::string_view(entry.name).ends_with(".msl.txt"))
			{
				continue;
			}

			const std::string path = m_msl_path + entry.name;

			if (is_valid_shader_source_metadata(path, m_version))
			{
				increment_cache_counter(count, "shader source metadata");
			}
			else
			{
				rsx_log.warning("Ignoring invalid Metal shader source metadata '%s'", path);
			}
		}

		return count;
	}

	void persistent_shader_cache::count_shader_completion_metadata(
		u32& total_entries,
		u32& available_entries,
		u32& gated_entries) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::count_shader_completion_metadata(path=%s)", m_completion_path);

		total_entries = 0;
		available_entries = 0;
		gated_entries = 0;

		fs::dir dir(m_completion_path);
		if (!dir)
		{
			return;
		}

		for (const fs::dir_entry& entry : dir)
		{
			if (entry.is_directory || !std::string_view(entry.name).ends_with(".complete.txt"))
			{
				continue;
			}

			const std::string path = m_completion_path + entry.name;

			if (is_valid_shader_completion_metadata(path, m_version))
			{
				const shader_completion_metadata metadata = load_shader_completion_metadata(path);
				increment_cache_counter(total_entries, "shader completion metadata");

				if (metadata.pipeline_entry_available)
				{
					increment_cache_counter(available_entries, "available shader completion metadata");
				}
				else
				{
					increment_cache_counter(gated_entries, "gated shader completion metadata");
				}
			}
			else
			{
				rsx_log.warning("Ignoring invalid Metal shader completion metadata '%s'", path);
			}
		}
	}

	void persistent_shader_cache::count_pipeline_entry_metadata(
		u32& total_entries,
		u32& available_entries,
		u32& gated_entries,
		u32& mesh_entries) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::count_pipeline_entry_metadata(path=%s)", m_pipeline_path);

		total_entries = 0;
		available_entries = 0;
		gated_entries = 0;
		mesh_entries = 0;

		fs::dir dir(m_pipeline_path);
		if (!dir)
		{
			return;
		}

		for (const fs::dir_entry& entry : dir)
		{
			if (entry.is_directory || !std::string_view(entry.name).ends_with(".entry.txt"))
			{
				continue;
			}

			const std::string path = m_pipeline_path + entry.name;

			if (is_valid_pipeline_entry_metadata(path, m_version))
			{
				const pipeline_entry_metadata metadata = load_pipeline_entry_metadata(path);
				increment_cache_counter(total_entries, "pipeline entry metadata");

				if (metadata.entry_available)
				{
					increment_cache_counter(available_entries, "available pipeline entry metadata");
				}
				else
				{
					increment_cache_counter(gated_entries, "gated pipeline entry metadata");
				}

				if (metadata.stage == "mesh")
				{
					increment_cache_counter(mesh_entries, "mesh pipeline entry metadata");
				}
			}
			else
			{
				rsx_log.warning("Ignoring invalid Metal pipeline entry metadata '%s'", path);
			}
		}
	}

	u32 persistent_shader_cache::count_shader_translation_failure_metadata() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::count_shader_translation_failure_metadata(path=%s)", m_raw_shader_path);

		fs::dir dir(m_raw_shader_path);
		if (!dir)
		{
			return 0;
		}

		u32 count = 0;

		for (const fs::dir_entry& entry : dir)
		{
			if (entry.is_directory || !std::string_view(entry.name).ends_with(".failure.txt"))
			{
				continue;
			}

			const std::string path = m_raw_shader_path + entry.name;

			if (is_valid_shader_translation_failure_metadata(path, m_version))
			{
				increment_cache_counter(count, "shader translation failure metadata");
			}
			else
			{
				rsx_log.warning("Ignoring invalid Metal shader translation failure metadata '%s'", path);
			}
		}

		return count;
	}

	void persistent_shader_cache::count_shader_library_metadata(
		u32& total_entries,
		u32& available_entries,
		u32& gated_entries) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::count_shader_library_metadata(path=%s)", m_library_path);

		total_entries = 0;
		available_entries = 0;
		gated_entries = 0;

		fs::dir dir(m_library_path);
		if (!dir)
		{
			return;
		}

		for (const fs::dir_entry& entry : dir)
		{
			if (entry.is_directory || !std::string_view(entry.name).ends_with(".metallib.txt"))
			{
				continue;
			}

			const std::string path = m_library_path + entry.name;

			if (is_valid_shader_library_metadata(path, m_version))
			{
				const shader_library_metadata metadata = load_shader_library_metadata(path);
				increment_cache_counter(total_entries, "shader library metadata");

				if (metadata.pipeline_entry_available)
				{
					increment_cache_counter(available_entries, "available shader library metadata");
				}
				else
				{
					increment_cache_counter(gated_entries, "gated shader library metadata");
				}
			}
			else
			{
				rsx_log.warning("Ignoring invalid Metal shader library metadata '%s'", path);
			}
		}
	}

	u32 persistent_shader_cache::count_pipeline_archive_metadata() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::count_pipeline_archive_metadata()");

		const std::string path = pipeline_archive_metadata_path();
		if (!fs::is_file(path))
		{
			return 0;
		}

		if (is_valid_pipeline_archive_metadata(path, m_version, m_pipeline_script_file_path, m_pipeline_archive_file_path))
		{
			return 1;
		}

		rsx_log.warning("Ignoring invalid Metal pipeline archive metadata '%s'", path);
		return 0;
	}

	void persistent_shader_cache::count_pipeline_state_metadata(
		u32& total_entries,
		u32& mesh_entries) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::count_pipeline_state_metadata(path=%s)", m_pipeline_path);

		total_entries = 0;
		mesh_entries = 0;

		fs::dir dir(m_pipeline_path);
		if (!dir)
		{
			return;
		}

		for (const fs::dir_entry& entry : dir)
		{
			if (entry.is_directory || !std::string_view(entry.name).ends_with(".pipeline.txt"))
			{
				continue;
			}

			const std::string path = m_pipeline_path + entry.name;

			if (is_valid_pipeline_state_metadata(path, m_version))
			{
				const pipeline_state_metadata metadata = load_pipeline_state_metadata(path);
				increment_cache_counter(total_entries, "pipeline state metadata");

				if (metadata.pipeline_type == "mesh")
				{
					increment_cache_counter(mesh_entries, "mesh pipeline state metadata");
				}
			}
			else
			{
				rsx_log.warning("Ignoring invalid Metal pipeline state metadata '%s'", path);
			}
		}
	}

	void persistent_shader_cache::refresh_stats()
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::refresh_stats()");

		const u32 raw_shader_entries = count_files_with_extension(m_msl_path, ".msl");
		const u32 shader_source_entries = count_shader_source_metadata();
		if (raw_shader_entries > shader_source_entries)
		{
			rsx_log.warning("Metal shader cache has %u raw MSL files but only %u valid shader source metadata records",
				raw_shader_entries, shader_source_entries);
		}

		m_stats.available = true;
		m_stats.shader_entries = shader_source_entries;
		m_stats.translation_failure_entries = count_shader_translation_failure_metadata();
		count_shader_completion_metadata(
			m_stats.completion_entries,
			m_stats.completion_available_entries,
			m_stats.completion_gated_entries);
		count_pipeline_entry_metadata(
			m_stats.pipeline_entries,
			m_stats.pipeline_entry_available_entries,
			m_stats.pipeline_entry_gated_entries,
			m_stats.mesh_pipeline_entry_entries);
		count_pipeline_state_metadata(
			m_stats.pipeline_state_entries,
			m_stats.mesh_pipeline_state_entries);
		count_shader_library_metadata(
			m_stats.library_entries,
			m_stats.library_available_entries,
			m_stats.library_gated_entries);
		m_stats.archive_entries = count_pipeline_archive_metadata();
	}
}
