#pragma once

#include <memory>

namespace rsx::metal
{
	class device;
	class persistent_shader_cache;

	class shader_compiler
	{
	public:
		shader_compiler(device& metal_device, const persistent_shader_cache& cache);
		~shader_compiler();

		shader_compiler(const shader_compiler&) = delete;
		shader_compiler& operator=(const shader_compiler&) = delete;

		void report() const;
		void* compiler_handle() const;
		void* pipeline_serializer_handle() const;
		void* task_options_handle() const;
		void* archive_handle() const;

	private:
		struct shader_compiler_impl;
		std::unique_ptr<shader_compiler_impl> m_impl;
	};
}
