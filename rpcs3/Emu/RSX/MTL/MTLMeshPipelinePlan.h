#pragma once

#include "MTLShaderInterface.h"

#include "util/types.hpp"

#include <string>

namespace rsx::metal
{
	struct mesh_pipeline_plan
	{
		shader_interface_layout interface_layout;
		u32 requirement_mask = 0;
		std::string gated_reason;
		b8 requires_argument_table_binding = true;
		b8 requires_mesh_object_mapping = true;
		b8 requires_mesh_grid_mapping = true;
		b8 preferred_for_gpu_geometry_expansion = true;
		b8 allows_cpu_geometry_preprocessing = false;
		b8 replaces_traditional_vertex_path = false;
	};

	mesh_pipeline_plan make_mesh_pipeline_plan();
	void validate_mesh_pipeline_plan(const mesh_pipeline_plan& plan);
	std::string describe_mesh_pipeline_plan(const mesh_pipeline_plan& plan);
}
