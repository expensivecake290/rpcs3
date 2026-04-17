#pragma once

#include "util/types.hpp"

#include <string>

namespace rsx::metal
{
	struct shader_cache_stats
	{
		b8 available = false;
		u32 shader_entries = 0;
		u32 pipeline_entries = 0;
		u32 library_entries = 0;
		u32 archive_entries = 0;
	};

	class persistent_shader_cache
	{
	public:
		explicit persistent_shader_cache(std::string version);
		~persistent_shader_cache();

		persistent_shader_cache(const persistent_shader_cache&) = delete;
		persistent_shader_cache& operator=(const persistent_shader_cache&) = delete;

		void initialize();
		void report() const;

		const std::string& root_path() const;
		const std::string& raw_shader_path() const;
		const std::string& msl_path() const;
		const std::string& library_path() const;
		const std::string& pipeline_path() const;
		const std::string& archive_path() const;
		const std::string& pipeline_script_path() const;
		const std::string& pipeline_archive_file_path() const;
		const shader_cache_stats& stats() const;

	private:
		void create_directory(const std::string& path) const;
		void validate_manifest() const;
		void refresh_stats();

		std::string m_version;
		std::string m_root_path;
		std::string m_raw_shader_path;
		std::string m_msl_path;
		std::string m_library_path;
		std::string m_pipeline_path;
		std::string m_archive_path;
		std::string m_pipeline_script_path;
		std::string m_manifest_path;
		std::string m_pipeline_archive_file_path;
		shader_cache_stats m_stats{};
	};
}
