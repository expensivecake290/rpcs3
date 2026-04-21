#pragma once

#include "MTLShaderRecompiler.h"

#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
	class persistent_shader_cache;
	class shader_compiler;

	struct shader_library_record
	{
		shader_stage stage = shader_stage::vertex;
		u32 id = 0;
		u64 source_hash = 0;
		u64 source_text_hash = 0;
		std::string entry_point;
		std::string dynamic_library_path;
		void* dynamic_library_handle = nullptr;
		u32 pipeline_requirement_mask = 0;
		b8 pipeline_entry_available = false;
		b8 loaded_from_disk = false;
	};

	struct shader_library_cache_stats
	{
		u32 memory_hits = 0;
		u32 disk_probes = 0;
		u32 disk_file_misses = 0;
		u32 loaded_libraries = 0;
		u32 compiled_libraries = 0;
		u32 source_metadata_misses = 0;
		u32 source_metadata_invalid = 0;
		u32 completion_metadata_misses = 0;
		u32 completion_metadata_invalid = 0;
		u32 library_metadata_misses = 0;
		u32 library_metadata_invalid = 0;
		u32 disk_load_failures = 0;
		u32 source_compile_failures = 0;
		u32 dynamic_library_failures = 0;
		u32 serialization_failures = 0;
		u32 memory_validation_failures = 0;
		u32 retained_libraries = 0;
		u32 retained_library_records = 0;
		u32 disk_loaded_library_count = 0;
	};

	void validate_shader_library_record(const shader_library_record& record, b8 require_handle);
	std::string describe_shader_library_record(const shader_library_record& record);

	class shader_library_cache
	{
	public:
		shader_library_cache(shader_compiler& compiler, persistent_shader_cache& cache);
		~shader_library_cache();

		shader_library_cache(const shader_library_cache&) = delete;
		shader_library_cache& operator=(const shader_library_cache&) = delete;

		shader_library_record get_or_compile_dynamic_library(const translated_shader& shader);
		shader_library_cache_stats stats() const;
		void report() const;

	private:
		struct shader_library_cache_impl;
		std::unique_ptr<shader_library_cache_impl> m_impl;
	};
}
