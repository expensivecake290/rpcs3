#pragma once

#include "util/types.hpp"

#include <memory>

namespace rsx::metal
{
	class persistent_shader_cache;
	class shader_compiler;

	class pipeline_cache
	{
	public:
		pipeline_cache(shader_compiler& compiler, persistent_shader_cache& cache);
		~pipeline_cache();

		pipeline_cache(const pipeline_cache&) = delete;
		pipeline_cache& operator=(const pipeline_cache&) = delete;

		void record_pipeline_compilation();
		void flush();
		void report() const;

	private:
		struct pipeline_cache_impl;
		std::unique_ptr<pipeline_cache_impl> m_impl;
	};
}
