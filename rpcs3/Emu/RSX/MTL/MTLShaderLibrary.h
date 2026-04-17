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
		std::string entry_point;
		std::string dynamic_library_path;
		void* dynamic_library_handle = nullptr;
		b8 loaded_from_disk = false;
	};

	class shader_library_cache
	{
	public:
		shader_library_cache(shader_compiler& compiler, persistent_shader_cache& cache);
		~shader_library_cache();

		shader_library_cache(const shader_library_cache&) = delete;
		shader_library_cache& operator=(const shader_library_cache&) = delete;

		shader_library_record get_or_compile_dynamic_library(const translated_shader& shader);
		void report() const;

	private:
		struct shader_library_cache_impl;
		std::unique_ptr<shader_library_cache_impl> m_impl;
	};
}
