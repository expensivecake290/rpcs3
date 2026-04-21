#pragma once

#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
	class device;
	class persistent_shader_cache;

	struct shader_compiler_stats
	{
		b8 compiler_ready = false;
		b8 pipeline_serializer_ready = false;
		b8 task_options_ready = false;
		b8 lookup_archive_configured = false;
		b8 archive_metadata_found = false;
		b8 archive_metadata_invalid = false;
		b8 archive_loaded = false;
		b8 archive_without_metadata = false;
		b8 archive_load_failed = false;
		std::string archive_path;
		std::string archive_metadata_error;
		std::string archive_load_error;
	};

	class shader_compiler
	{
	public:
		shader_compiler(device& metal_device, const persistent_shader_cache& cache);
		~shader_compiler();

		shader_compiler(const shader_compiler&) = delete;
		shader_compiler& operator=(const shader_compiler&) = delete;

		void report() const;
		shader_compiler_stats stats() const;
		void* compiler_handle() const;
		void* pipeline_serializer_handle() const;
		void* task_options_handle() const;
		void* archive_handle() const;

	private:
		struct shader_compiler_impl;
		std::unique_ptr<shader_compiler_impl> m_impl;
	};
}
