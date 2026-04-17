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
	};

	mesh_pipeline_plan make_mesh_pipeline_plan();
	void validate_mesh_pipeline_plan(const mesh_pipeline_plan& plan);
	std::string describe_mesh_pipeline_plan(const mesh_pipeline_plan& plan);
}
