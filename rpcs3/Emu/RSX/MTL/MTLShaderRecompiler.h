#pragma once

#include "util/types.hpp"

#include <string>

struct RSXFragmentProgram;
struct RSXVertexProgram;

namespace rsx::metal
{
	class persistent_shader_cache;

	enum class shader_stage : u8
	{
		vertex,
		fragment,
		mesh,
	};

	enum class pipeline_entry_requirement : u32
	{
		argument_table_shader_binding = 1u << 0,
		vertex_input_fetch = 1u << 1,
		viewport_depth_transform = 1u << 2,
		stage_input_layout = 1u << 3,
		mrt_output_mapping = 1u << 4,
		depth_export_mapping = 1u << 5,
		mesh_object_mapping = 1u << 6,
		mesh_grid_mapping = 1u << 7,
	};

	struct translated_shader
	{
		shader_stage stage = shader_stage::vertex;
		u32 id = 0;
		u64 source_hash = 0;
		std::string entry_point;
		std::string source;
		std::string cache_path;
		u64 pipeline_source_hash = 0;
		std::string pipeline_entry_point;
		std::string pipeline_source;
		std::string pipeline_cache_path;
		std::string pipeline_entry_error;
		u32 pipeline_requirement_mask = 0;
		b8 pipeline_entry_available = false;
	};

	class shader_recompiler
	{
	public:
		explicit shader_recompiler(persistent_shader_cache& cache);
		~shader_recompiler();

		shader_recompiler(const shader_recompiler&) = delete;
		shader_recompiler& operator=(const shader_recompiler&) = delete;

		translated_shader translate_vertex_program(const RSXVertexProgram& program, u32 id);
		translated_shader translate_fragment_program(const RSXFragmentProgram& program, u32 id);
		void report() const;

	private:
		persistent_shader_cache& m_cache;
	};
}
