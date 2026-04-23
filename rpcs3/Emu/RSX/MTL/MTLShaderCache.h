#pragma once

#include "util/types.hpp"

#include <string>
#include <vector>

namespace rsx::metal
{
	struct shader_source_metadata
	{
		std::string stage;
		u32 shader_id = 0;
		u64 source_hash = 0;
		u64 source_text_hash = 0;
		std::string entry_point;
		std::string source_path;
	};

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
		std::string requirement_description;
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
		u64 library_size = 0;
		u64 library_hash = 0;
		u32 pipeline_requirement_mask = 0;
		std::string pipeline_requirement_description;
		std::string pipeline_entry_error;
		b8 pipeline_entry_available = false;
	};

	struct shader_completion_metadata
	{
		std::string stage;
		u32 shader_id = 0;
		u64 source_hash = 0;
		u64 source_text_hash = 0;
		std::string entry_point;
		std::string source_path;
		std::vector<u32> fragment_constant_offsets;
		u64 pipeline_source_hash = 0;
		std::string pipeline_entry_point;
		std::string pipeline_source_path;
		std::string pipeline_entry_error;
		u32 pipeline_requirement_mask = 0;
		std::string pipeline_requirement_description;
		b8 pipeline_entry_available = false;
	};

	struct shader_translation_failure_metadata
	{
		std::string stage;
		u32 shader_id = 0;
		u64 source_hash = 0;
		std::string failure_reason;
	};

	struct pipeline_archive_metadata
	{
		std::string script_path;
		std::string archive_path;
		u64 script_size = 0;
		u64 archive_size = 0;
		u64 script_hash = 0;
		u64 archive_hash = 0;
		u32 flushed_pipeline_count = 0;
	};

	struct pipeline_state_metadata
	{
		std::string pipeline_type;
		u64 pipeline_hash = 0;
		u64 vertex_source_hash = 0;
		u64 fragment_source_hash = 0;
		u64 object_source_hash = 0;
		u64 mesh_source_hash = 0;
		u64 linked_library_hash = 0;
		u32 linked_library_count = 0;
		u32 shader_dependency_count = 0;
		u32 color_pixel_format = 0;
		u32 raster_sample_count = 0;
		b8 rasterization_enabled = false;
	};

	struct shader_cache_stats
	{
		b8 available = false;
		u32 shader_entries = 0;
		u32 completion_entries = 0;
		u32 completion_available_entries = 0;
		u32 completion_gated_entries = 0;
		u32 translation_failure_entries = 0;
		u32 pipeline_entries = 0;
		u32 pipeline_entry_available_entries = 0;
		u32 pipeline_entry_gated_entries = 0;
		u32 mesh_pipeline_entry_entries = 0;
		u32 pipeline_state_entries = 0;
		u32 mesh_pipeline_state_entries = 0;
		u32 library_entries = 0;
		u32 library_available_entries = 0;
		u32 library_gated_entries = 0;
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
		void store_shader_source_metadata(
			const char* stage,
			u32 shader_id,
			u64 source_hash,
			u64 source_text_hash,
			const std::string& entry_point,
			const std::string& source_path);
		shader_source_metadata load_shader_source_metadata(const std::string& path) const;
		b8 find_shader_source_metadata(
			const char* stage,
			u32 shader_id,
			u64 source_hash,
			u64 source_text_hash,
			const std::string& entry_point,
			const std::string& source_path,
			shader_source_metadata& metadata) const;
		b8 try_find_shader_source_metadata(
			const char* stage,
			u32 shader_id,
			u64 source_hash,
			u64 source_text_hash,
			const std::string& entry_point,
			const std::string& source_path,
			shader_source_metadata& metadata,
			std::string& error) const;
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
		b8 try_find_pipeline_entry_metadata(
			const char* stage,
			u64 source_hash,
			pipeline_entry_metadata& metadata,
			std::string& error) const;
		void store_shader_library_metadata(
			const char* stage,
			u32 shader_id,
			u64 source_hash,
			u64 source_text_hash,
			const std::string& entry_point,
			const std::string& library_path,
			const std::string& pipeline_entry_error,
			u32 pipeline_requirement_mask,
			b8 pipeline_entry_available);
		shader_library_metadata load_shader_library_metadata(const std::string& path) const;
		b8 find_shader_library_metadata(
			const char* stage,
			u32 shader_id,
			u64 source_hash,
			u64 source_text_hash,
			const std::string& entry_point,
			const std::string& library_path,
			const std::string& pipeline_entry_error,
			u32 pipeline_requirement_mask,
			b8 pipeline_entry_available,
			shader_library_metadata& metadata) const;
		b8 try_find_shader_library_metadata(
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
			std::string& error) const;
		void store_shader_completion_metadata(
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
			b8 pipeline_entry_available);
		shader_completion_metadata load_shader_completion_metadata(const std::string& path) const;
		b8 find_shader_completion_metadata(
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
			shader_completion_metadata& metadata) const;
		b8 try_find_shader_completion_metadata(
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
			std::string& error) const;
		b8 try_load_shader_completion_metadata(
			const char* stage,
			u32 shader_id,
			u64 source_hash,
			shader_completion_metadata& metadata,
			std::string& error) const;
		void store_shader_translation_failure_metadata(
			const char* stage,
			u32 shader_id,
			u64 source_hash,
			const std::string& failure_reason);
		shader_translation_failure_metadata load_shader_translation_failure_metadata(const std::string& path) const;
		b8 find_shader_translation_failure_metadata(const char* stage, u32 shader_id, u64 source_hash, shader_translation_failure_metadata& metadata) const;
		b8 try_find_shader_translation_failure_metadata(
			const char* stage,
			u32 shader_id,
			u64 source_hash,
			shader_translation_failure_metadata& metadata,
			std::string& error) const;
		void store_pipeline_archive_metadata(
			const std::string& script_path,
			const std::string& archive_path,
			u32 flushed_pipeline_count);
		pipeline_archive_metadata load_pipeline_archive_metadata(const std::string& path) const;
		b8 find_pipeline_archive_metadata(pipeline_archive_metadata& metadata) const;
		b8 try_find_pipeline_archive_metadata(pipeline_archive_metadata& metadata, std::string& error) const;
		void store_pipeline_state_metadata(
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
			b8 rasterization_enabled);
		pipeline_state_metadata load_pipeline_state_metadata(const std::string& path) const;
		b8 find_pipeline_state_metadata(
			const char* pipeline_type,
			u64 pipeline_hash,
			pipeline_state_metadata& metadata) const;
		b8 try_find_pipeline_state_metadata(
			const char* pipeline_type,
			u64 pipeline_hash,
			pipeline_state_metadata& metadata,
			std::string& error) const;

		const std::string& root_path() const;
		const std::string& raw_shader_path() const;
		const std::string& msl_path() const;
		const std::string& library_path() const;
		const std::string& completion_path() const;
		const std::string& pipeline_path() const;
		const std::string& archive_path() const;
		const std::string& pipeline_script_path() const;
		const std::string& pipeline_script_file_path() const;
		const std::string& pipeline_archive_file_path() const;
		const shader_cache_stats& stats() const;

	private:
		void create_directory(const std::string& path) const;
		void validate_manifest() const;
		std::string shader_source_metadata_path(const char* stage, u64 source_hash) const;
		std::string pipeline_entry_metadata_path(const char* stage, u64 source_hash) const;
		std::string shader_library_metadata_path(const char* stage, u64 source_hash) const;
		std::string shader_completion_metadata_path(const char* stage, u64 source_hash) const;
		std::string shader_translation_failure_metadata_path(const char* stage, u64 source_hash) const;
		std::string pipeline_archive_metadata_path() const;
		std::string pipeline_state_metadata_path(const char* pipeline_type, u64 pipeline_hash) const;
		u32 count_shader_source_metadata() const;
		void count_shader_completion_metadata(
			u32& total_entries,
			u32& available_entries,
			u32& gated_entries) const;
		u32 count_shader_translation_failure_metadata() const;
		void count_pipeline_entry_metadata(
			u32& total_entries,
			u32& available_entries,
			u32& gated_entries,
			u32& mesh_entries) const;
		void count_shader_library_metadata(
			u32& total_entries,
			u32& available_entries,
			u32& gated_entries) const;
		u32 count_pipeline_archive_metadata() const;
		void count_pipeline_state_metadata(
			u32& total_entries,
			u32& mesh_entries) const;
		void refresh_stats();

		std::string m_version;
		std::string m_root_path;
		std::string m_raw_shader_path;
		std::string m_msl_path;
		std::string m_library_path;
		std::string m_completion_path;
		std::string m_pipeline_path;
		std::string m_archive_path;
		std::string m_pipeline_script_path;
		std::string m_pipeline_script_file_path;
		std::string m_manifest_path;
		std::string m_pipeline_archive_file_path;
		shader_cache_stats m_stats{};
	};
}
