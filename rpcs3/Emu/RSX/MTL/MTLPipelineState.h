#pragma once

#include "MTLShaderInterface.h"
#include "MTLShaderLibrary.h"

#include "Emu/RSX/gcm_enums.h"
#include "util/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace rsx::metal
{
	class argument_table;
	class command_frame;
	class pipeline_cache;
	class persistent_shader_cache;
	class shader_compiler;
	struct pipeline_entry_metadata;

	struct render_pipeline_shader
	{
		u64 source_hash = 0;
		std::string source;
		std::string entry_point;
		std::string entry_error;
		u32 requirement_mask = 0;
		b8 entry_available = false;
	};

	render_pipeline_shader make_render_pipeline_shader(const translated_shader& shader);
	render_pipeline_shader make_render_pipeline_shader(const pipeline_entry_metadata& metadata);
	u32 get_render_pipeline_topology_class(rsx::primitive_type primitive);

	struct render_pipeline_desc
	{
		u64 pipeline_hash = 0;
		std::string label;
		render_pipeline_shader vertex;
		render_pipeline_shader fragment;
		shader_interface_layout vertex_layout = make_vertex_shader_interface_layout();
		shader_interface_layout fragment_layout = make_fragment_shader_interface_layout(0, 0);
		std::vector<shader_library_record> linked_libraries;
		u32 color_pixel_format = 0;
		u32 raster_sample_count = 1;
		u32 input_primitive_topology = 0;
		b8 alpha_to_coverage = false;
		b8 alpha_to_one = false;
		u32 color_write_mask = 0xf;
		b8 blend_enabled = false;
		rsx::blend_factor source_rgb_blend_factor = rsx::blend_factor::one;
		rsx::blend_factor source_alpha_blend_factor = rsx::blend_factor::one;
		rsx::blend_factor destination_rgb_blend_factor = rsx::blend_factor::zero;
		rsx::blend_factor destination_alpha_blend_factor = rsx::blend_factor::zero;
		rsx::blend_equation rgb_blend_operation = rsx::blend_equation::add;
		rsx::blend_equation alpha_blend_operation = rsx::blend_equation::add;
		b8 rasterization_enabled = true;
		b8 support_indirect_command_buffers = false;
	};

	struct render_pipeline_record
	{
		u64 pipeline_hash = 0;
		void* pipeline_handle = nullptr;
		shader_interface_layout primary_layout = make_vertex_shader_interface_layout();
		shader_interface_layout fragment_layout = make_fragment_shader_interface_layout(0, 0);
		b8 cached = false;
		b8 mesh_pipeline = false;
		b8 has_fragment_layout = false;
	};

	void validate_pipeline_binding_record(const render_pipeline_record& record);
	void bind_render_pipeline_state(command_frame& frame, void* render_encoder_handle, const render_pipeline_record& record);
	void bind_pipeline_arguments(
		command_frame& frame,
		void* render_encoder_handle,
		const render_pipeline_record& record,
		const argument_table& primary_table,
		const argument_table* fragment_table = nullptr);

	struct render_pipeline_cache_stats
	{
		u32 compiled_render_pipeline_count = 0;
		u32 render_pipeline_cache_hit_count = 0;
		u32 retained_render_pipeline_count = 0;
		u32 render_pipeline_compile_failure_count = 0;
		u32 render_pipeline_metadata_hit_count = 0;
		u32 render_pipeline_metadata_miss_count = 0;
		u32 render_pipeline_metadata_mismatch_count = 0;
		u32 render_pipeline_metadata_invalid_count = 0;
		u32 compiled_mesh_pipeline_count = 0;
		u32 mesh_pipeline_cache_hit_count = 0;
		u32 retained_mesh_pipeline_count = 0;
		u32 mesh_pipeline_compile_failure_count = 0;
		u32 mesh_pipeline_metadata_hit_count = 0;
		u32 mesh_pipeline_metadata_miss_count = 0;
		u32 mesh_pipeline_metadata_mismatch_count = 0;
		u32 mesh_pipeline_metadata_invalid_count = 0;
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
		shader_interface_layout mesh_layout = make_mesh_shader_interface_layout();
		shader_interface_layout fragment_layout = make_fragment_shader_interface_layout(0, 0);
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
		render_pipeline_cache_stats stats() const;
		void report() const;

	private:
		struct render_pipeline_cache_impl;
		std::unique_ptr<render_pipeline_cache_impl> m_impl;
	};
}
