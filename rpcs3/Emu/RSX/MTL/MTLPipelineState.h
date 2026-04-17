#pragma once

#include "MTLShaderLibrary.h"

#include "util/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace rsx::metal
{
	class pipeline_cache;
	class persistent_shader_cache;
	class shader_compiler;

	struct render_pipeline_shader
	{
		u64 source_hash = 0;
		std::string source;
		std::string entry_point;
	};

	struct render_pipeline_desc
	{
		u64 pipeline_hash = 0;
		std::string label;
		render_pipeline_shader vertex;
		render_pipeline_shader fragment;
		std::vector<shader_library_record> linked_libraries;
		u32 color_pixel_format = 0;
		u32 raster_sample_count = 1;
		u32 input_primitive_topology = 0;
		b8 alpha_to_coverage = false;
		b8 alpha_to_one = false;
		b8 rasterization_enabled = true;
		b8 support_indirect_command_buffers = false;
	};

	struct render_pipeline_record
	{
		u64 pipeline_hash = 0;
		void* pipeline_handle = nullptr;
		b8 cached = false;
	};

	struct mesh_threadgroup_size
	{
		u32 width = 0;
		u32 height = 0;
		u32 depth = 0;
	};

	struct mesh_pipeline_desc
	{
		u64 pipeline_hash = 0;
		std::string label;
		render_pipeline_shader object;
		render_pipeline_shader mesh;
		render_pipeline_shader fragment;
		std::vector<shader_library_record> linked_libraries;
		u32 color_pixel_format = 0;
		u32 raster_sample_count = 1;
		u32 max_total_threads_per_object_threadgroup = 0;
		u32 max_total_threads_per_mesh_threadgroup = 0;
		mesh_threadgroup_size required_threads_per_object_threadgroup;
		mesh_threadgroup_size required_threads_per_mesh_threadgroup;
		u32 payload_memory_length = 0;
		u32 max_total_threadgroups_per_mesh_grid = 0;
		b8 object_threadgroup_size_is_multiple_of_thread_execution_width = false;
		b8 mesh_threadgroup_size_is_multiple_of_thread_execution_width = false;
		b8 alpha_to_coverage = false;
		b8 alpha_to_one = false;
		b8 rasterization_enabled = true;
		b8 support_indirect_command_buffers = false;
	};

	class render_pipeline_cache
	{
	public:
		render_pipeline_cache(shader_compiler& compiler, persistent_shader_cache& cache, pipeline_cache& pipelines);
		~render_pipeline_cache();

		render_pipeline_cache(const render_pipeline_cache&) = delete;
		render_pipeline_cache& operator=(const render_pipeline_cache&) = delete;

		render_pipeline_record get_or_compile_render_pipeline(const render_pipeline_desc& desc);
		render_pipeline_record get_or_compile_mesh_pipeline(const mesh_pipeline_desc& desc);
		void report() const;

	private:
		struct render_pipeline_cache_impl;
		std::unique_ptr<render_pipeline_cache_impl> m_impl;
	};
}
