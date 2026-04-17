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

	struct translated_shader
	{
		shader_stage stage = shader_stage::vertex;
		u32 id = 0;
		u64 source_hash = 0;
		std::string entry_point;
		std::string source;
		std::string cache_path;
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
