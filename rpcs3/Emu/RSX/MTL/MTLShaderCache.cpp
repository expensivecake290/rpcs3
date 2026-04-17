#include "stdafx.h"
#include "MTLShaderCache.h"

#include "Emu/cache_utils.hpp"
#include "Emu/system_config.h"
#include "Utilities/File.h"

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

	void persistent_shader_cache::refresh_stats()
	{
		rsx_log.trace("rsx::metal::persistent_shader_cache::refresh_stats()");

		m_stats.available = true;
		m_stats.shader_entries = count_files_with_extension(m_msl_path, ".msl");
		m_stats.pipeline_entries = count_files_with_extension(m_pipeline_path, ".bin");
		m_stats.library_entries = count_files_with_extension(m_library_path, ".metallib");
		m_stats.archive_entries = count_files_with_extension(m_archive_path, ".mtl4archive");
	}
}
