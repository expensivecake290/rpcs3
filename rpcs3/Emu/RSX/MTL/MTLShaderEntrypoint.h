#pragma once

#include "MTLShaderRecompiler.h"

struct RSXFragmentProgram;

namespace rsx::metal
{
	void mark_vertex_pipeline_entry_status(translated_shader& shader);
	void mark_fragment_pipeline_entry_status(translated_shader& shader, const RSXFragmentProgram& program);
	void mark_mesh_pipeline_entry_status(translated_shader& shader);
}
