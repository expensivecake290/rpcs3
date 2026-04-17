#pragma once

#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
	class persistent_shader_cache;
	class shader_compiler;

	struct pipeline_cache_stats
	{
		u32 pending_pipeline_count = 0;
		u32 flushed_pipeline_count = 0;
		u32 skipped_flush_count = 0;
		u32 successful_flush_count = 0;
		u32 serializer_missing_failures = 0;
		u32 script_serialization_failures = 0;
		u32 archive_serialization_failures = 0;
		std::string script_path;
		std::string archive_path;
	};

	class pipeline_cache
	{
	public:
		pipeline_cache(shader_compiler& compiler, persistent_shader_cache& cache);
		~pipeline_cache();

		pipeline_cache(const pipeline_cache&) = delete;
		pipeline_cache& operator=(const pipeline_cache&) = delete;

		void record_pipeline_compilation();
		void flush();
		pipeline_cache_stats stats() const;
		void report() const;

	private:
		struct pipeline_cache_impl;
		std::unique_ptr<pipeline_cache_impl> m_impl;
	};
}
