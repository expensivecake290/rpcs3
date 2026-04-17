#include "stdafx.h"
#include "MTLShaderCache.h"

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
	constexpr std::string_view s_manifest_text =
		"RPCS3 Metal shader cache\n"
		"backend=metal\n"
		"version=v1.0\n";

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
				count++;
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

	b8 is_pipeline_entry_stage(std::string_view stage)
	{
		rsx_log.trace("is_pipeline_entry_stage(stage=%s)", stage);
		return stage == "vp" || stage == "fp" || stage == "mesh";
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
			return pipeline_source_hash && !requirement_mask && !entry_point.empty() && !source_path.empty();
		}

		return requirement_mask && !entry_error.empty();
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
		if (!try_get_metadata_field(view, "record_version", field) || !try_parse_metadata_u32(field, metadata_u32) || metadata_u32 != 1)
		{
			return false;
		}

		if (!try_get_metadata_field(view, "stage", field) || !is_shader_library_stage(field))
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

		return fs::is_file(library_path);
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

	b8 is_valid_pipeline_archive_metadata(const std::string& path, const std::string& cache_version)
	{
		rsx_log.trace("is_valid_pipeline_archive_metadata(path=%s, cache_version=%s)", path, cache_version);

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
		if (!try_get_metadata_field(view, "record_version", field) || !try_parse_metadata_u32(field, metadata_u32) || metadata_u32 != 1)
		{
			return false;
		}

		std::string script_path;
		if (!try_get_metadata_field(view, "script_path", script_path) || script_path.empty())
		{
			return false;
		}

		std::string archive_path;
		if (!try_get_metadata_field(view, "archive_path", archive_path) || archive_path.empty())
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

		if (!try_get_metadata_field(view, "flushed_pipeline_count", field) || !try_parse_metadata_u32(field, metadata_u32) || !metadata_u32)
		{
			return false;
		}

		u64 actual_script_size = 0;
		u64 actual_archive_size = 0;
		return get_file_size(script_path, actual_script_size) &&
			get_file_size(archive_path, actual_archive_size) &&
			actual_script_size == script_size &&
			actual_archive_size == archive_size;
	}
}

namespace rsx::metal
{
	persistent_shader_cache::persistent_shader_cache(std::string version)
		: m_version(std::move(version))
	{
		rsx_log.notice("rsx::metal::persistent_shader_cache::persistent_shader_cache(version=%s)", m_version);
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
		rsx_log.notice("Metal shader cache entries: shaders=%u, libraries=%u, pipelines=%u, archives=%u",
			m_stats.shader_entries,
			m_stats.library_entries,
			m_stats.pipeline_entries,
			m_stats.archive_entries);
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
		if (!find_shader_source_metadata(stage, source_hash, source_text_hash, entry_point, source_path, loaded_metadata))
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
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& source_path,
		shader_source_metadata& metadata) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::find_shader_source_metadata(stage=%s, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, source_path=%s)",
			stage ? stage : "<null>", source_hash, source_text_hash, entry_point.c_str(), source_path.c_str());

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

		if (loaded_metadata.source_hash != source_hash ||
			loaded_metadata.source_text_hash != source_text_hash ||
			loaded_metadata.entry_point != entry_point ||
			loaded_metadata.source_path != source_path)
		{
			rsx_log.warning("Metal shader source metadata mismatch for '%s'", path);
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

		if (metadata.entry_available)
		{
			if (!metadata.pipeline_source_hash || metadata.requirement_mask || metadata.entry_point.empty() || metadata.source_path.empty())
			{
				fmt::throw_exception("Metal pipeline entry metadata '%s' marks an entry available without executable source data", path);
			}
		}
		else if (!metadata.requirement_mask || metadata.entry_error.empty())
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

	void persistent_shader_cache::store_shader_library_metadata(
		const char* stage,
		u32 shader_id,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& library_path)
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::store_shader_library_metadata(stage=%s, shader_id=%u, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, library_path=%s)",
			stage ? stage : "<null>", shader_id, source_hash, source_text_hash, entry_point.c_str(), library_path.c_str());

		if (!source_text_hash)
		{
			fmt::throw_exception("Metal shader library metadata requires a non-zero source text hash");
		}

		if (entry_point.empty())
		{
			fmt::throw_exception("Metal shader library metadata requires an entry point");
		}

		if (library_path.empty())
		{
			fmt::throw_exception("Metal shader library metadata requires a library path");
		}

		const std::string path = shader_library_metadata_path(stage, source_hash);
		const std::string metadata = fmt::format(
			"RPCS3 Metal shader library\n"
			"backend=metal\n"
			"cache_version=%s\n"
			"record_version=1\n"
			"stage=%s\n"
			"shader_id=%u\n"
			"source_hash=0x%llx\n"
			"source_text_hash=0x%llx\n"
			"entry_point=%s\n"
			"library_path=%s\n",
			m_version,
			stage,
			shader_id,
			source_hash,
			source_text_hash,
			metadata_value(entry_point),
			metadata_value(library_path));

		if (!fs::write_file(path, fs::rewrite, metadata))
		{
			fmt::throw_exception("Metal shader library metadata write failed for '%s' (%s)", path, fs::g_tls_error);
		}

		shader_library_metadata loaded_metadata;
		if (!find_shader_library_metadata(stage, source_hash, source_text_hash, entry_point, library_path, loaded_metadata))
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
		if (record_version != 1)
		{
			fmt::throw_exception("Metal shader library metadata '%s' has unsupported record version %u", path, record_version);
		}

		shader_library_metadata metadata =
		{
			.stage = get_metadata_field(view, "stage"),
			.shader_id = parse_metadata_u32(get_metadata_field(view, "shader_id"), "shader_id"),
			.source_hash = parse_metadata_u64(get_metadata_field(view, "source_hash"), "source_hash"),
			.source_text_hash = parse_metadata_u64(get_metadata_field(view, "source_text_hash"), "source_text_hash"),
			.entry_point = get_metadata_field(view, "entry_point"),
			.library_path = get_metadata_field(view, "library_path"),
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

		if (metadata.library_path.empty() || !fs::is_file(metadata.library_path))
		{
			fmt::throw_exception("Metal shader library metadata '%s' points to a missing library '%s'",
				path, metadata.library_path);
		}

		return metadata;
	}

	b8 persistent_shader_cache::find_shader_library_metadata(
		const char* stage,
		u64 source_hash,
		u64 source_text_hash,
		const std::string& entry_point,
		const std::string& library_path,
		shader_library_metadata& metadata) const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::find_shader_library_metadata(stage=%s, source_hash=0x%llx, source_text_hash=0x%llx, entry_point=%s, library_path=%s)",
			stage ? stage : "<null>", source_hash, source_text_hash, entry_point.c_str(), library_path.c_str());

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

		if (loaded_metadata.source_hash != source_hash ||
			loaded_metadata.source_text_hash != source_text_hash ||
			loaded_metadata.entry_point != entry_point ||
			loaded_metadata.library_path != library_path)
		{
			rsx_log.warning("Metal shader library metadata mismatch for '%s'", path);
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

		const std::string path = pipeline_archive_metadata_path();
		const std::string metadata = fmt::format(
			"RPCS3 Metal pipeline archive\n"
			"backend=metal\n"
			"cache_version=%s\n"
			"record_version=1\n"
			"script_path=%s\n"
			"archive_path=%s\n"
			"script_size=0x%llx\n"
			"archive_size=0x%llx\n"
			"flushed_pipeline_count=%u\n",
			m_version,
			metadata_value(script_path),
			metadata_value(archive_path),
			script_size,
			archive_size,
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
		if (record_version != 1)
		{
			fmt::throw_exception("Metal pipeline archive metadata '%s' has unsupported record version %u", path, record_version);
		}

		pipeline_archive_metadata metadata =
		{
			.script_path = get_metadata_field(view, "script_path"),
			.archive_path = get_metadata_field(view, "archive_path"),
			.script_size = parse_metadata_u64(get_metadata_field(view, "script_size"), "script_size"),
			.archive_size = parse_metadata_u64(get_metadata_field(view, "archive_size"), "archive_size"),
			.flushed_pipeline_count = parse_metadata_u32(get_metadata_field(view, "flushed_pipeline_count"), "flushed_pipeline_count"),
		};

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

		if (fs::file manifest{m_manifest_path, fs::read})
		{
			if (manifest.to_string() != s_manifest_text)
			{
				fmt::throw_exception("Metal shader cache metadata is incompatible at '%s'", m_manifest_path);
			}

			return;
		}

		if (!fs::write_file(m_manifest_path, fs::rewrite, s_manifest_text))
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

	std::string persistent_shader_cache::pipeline_archive_metadata_path() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::pipeline_archive_metadata_path()");
		return m_pipeline_archive_file_path + ".txt";
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
				count++;
			}
			else
			{
				rsx_log.warning("Ignoring invalid Metal shader source metadata '%s'", path);
			}
		}

		return count;
	}

	u32 persistent_shader_cache::count_pipeline_entry_metadata() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::count_pipeline_entry_metadata(path=%s)", m_pipeline_path);

		fs::dir dir(m_pipeline_path);
		if (!dir)
		{
			return 0;
		}

		u32 count = 0;

		for (const fs::dir_entry& entry : dir)
		{
			if (entry.is_directory || !std::string_view(entry.name).ends_with(".entry.txt"))
			{
				continue;
			}

			const std::string path = m_pipeline_path + entry.name;

			if (is_valid_pipeline_entry_metadata(path, m_version))
			{
				count++;
			}
			else
			{
				rsx_log.warning("Ignoring invalid Metal pipeline entry metadata '%s'", path);
			}
		}

		return count;
	}

	u32 persistent_shader_cache::count_shader_library_metadata() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::count_shader_library_metadata(path=%s)", m_library_path);

		fs::dir dir(m_library_path);
		if (!dir)
		{
			return 0;
		}

		u32 count = 0;

		for (const fs::dir_entry& entry : dir)
		{
			if (entry.is_directory || !std::string_view(entry.name).ends_with(".metallib.txt"))
			{
				continue;
			}

			const std::string path = m_library_path + entry.name;

			if (is_valid_shader_library_metadata(path, m_version))
			{
				count++;
			}
			else
			{
				rsx_log.warning("Ignoring invalid Metal shader library metadata '%s'", path);
			}
		}

		return count;
	}

	u32 persistent_shader_cache::count_pipeline_archive_metadata() const
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::count_pipeline_archive_metadata()");

		const std::string path = pipeline_archive_metadata_path();
		if (!fs::is_file(path))
		{
			return 0;
		}

		if (is_valid_pipeline_archive_metadata(path, m_version))
		{
			return 1;
		}

		rsx_log.warning("Ignoring invalid Metal pipeline archive metadata '%s'", path);
		return 0;
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
		m_stats.pipeline_entries = count_pipeline_entry_metadata();
		m_stats.library_entries = count_shader_library_metadata();
		m_stats.archive_entries = count_pipeline_archive_metadata();
	}
}
