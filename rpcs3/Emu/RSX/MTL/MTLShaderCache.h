#pragma once

#include "util/types.hpp"

#include <string>

namespace rsx::metal
{
	struct pipeline_entry_metadata
	{
		std::string stage;
		u32 shader_id = 0;
		u64 source_hash = 0;
		u64 pipeline_source_hash = 0;
		std::string entry_point;
		std::string source_path;
		std::string entry_error;
		u32 requirement_mask = 0;
		b8 entry_available = false;
	};

	struct shader_library_metadata
	{
		std::string stage;
		u32 shader_id = 0;
		u64 source_hash = 0;
		u64 source_text_hash = 0;
		std::string entry_point;
		std::string library_path;
	};

	struct pipeline_archive_metadata
	{
		std::string script_path;
		std::string archive_path;
		u64 script_size = 0;
		u64 archive_size = 0;
		u32 flushed_pipeline_count = 0;
	};

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
		void store_pipeline_entry_metadata(
			const char* stage,
			u32 shader_id,
			u64 source_hash,
			u64 pipeline_source_hash,
			const std::string& entry_point,
			const std::string& source_path,
			const std::string& entry_error,
			u32 requirement_mask,
			b8 entry_available);
		pipeline_entry_metadata load_pipeline_entry_metadata(const std::string& path) const;
		b8 find_pipeline_entry_metadata(const char* stage, u64 source_hash, pipeline_entry_metadata& metadata) const;
		void store_shader_library_metadata(
			const char* stage,
			u32 shader_id,
			u64 source_hash,
			u64 source_text_hash,
			const std::string& entry_point,
			const std::string& library_path);
		shader_library_metadata load_shader_library_metadata(const std::string& path) const;
		b8 find_shader_library_metadata(
			const char* stage,
			u64 source_hash,
			u64 source_text_hash,
			const std::string& entry_point,
			const std::string& library_path,
			shader_library_metadata& metadata) const;
		void store_pipeline_archive_metadata(
			const std::string& script_path,
			const std::string& archive_path,
			u32 flushed_pipeline_count);
		pipeline_archive_metadata load_pipeline_archive_metadata(const std::string& path) const;
		b8 find_pipeline_archive_metadata(pipeline_archive_metadata& metadata) const;

		const std::string& root_path() const;
		const std::string& raw_shader_path() const;
		const std::string& msl_path() const;
		const std::string& library_path() const;
		const std::string& pipeline_path() const;
		const std::string& archive_path() const;
		const std::string& pipeline_script_path() const;
		const std::string& pipeline_script_file_path() const;
		const std::string& pipeline_archive_file_path() const;
		const shader_cache_stats& stats() const;

	private:
		void create_directory(const std::string& path) const;
		void validate_manifest() const;
		std::string pipeline_entry_metadata_path(const char* stage, u64 source_hash) const;
		std::string shader_library_metadata_path(const char* stage, u64 source_hash) const;
		std::string pipeline_archive_metadata_path() const;
		u32 count_pipeline_entry_metadata() const;
		u32 count_shader_library_metadata() const;
		u32 count_pipeline_archive_metadata() const;
		void refresh_stats();

		std::string m_version;
		std::string m_root_path;
		std::string m_raw_shader_path;
		std::string m_msl_path;
		std::string m_library_path;
		std::string m_pipeline_path;
		std::string m_archive_path;
		std::string m_pipeline_script_path;
		std::string m_pipeline_script_file_path;
		std::string m_manifest_path;
		std::string m_pipeline_archive_file_path;
		shader_cache_stats m_stats{};
	};
}
