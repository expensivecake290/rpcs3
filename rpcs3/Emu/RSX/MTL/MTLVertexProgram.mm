#include "stdafx.h"
#include "MTLVertexProgram.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <sstream>

namespace mtl
{
	namespace
	{
		struct vertex_export
		{
			std::string_view name;
			std::string_view register_name;
			std::string_view mask;
			u32 output_mask;
		};

		constexpr std::array varying_exports = {
			vertex_export{"diff_color", "dst_reg1", "", CELL_GCM_ATTRIB_OUTPUT_MASK_FRONTDIFFUSE | CELL_GCM_ATTRIB_OUTPUT_MASK_BACKDIFFUSE},
			vertex_export{"spec_color", "dst_reg2", "", CELL_GCM_ATTRIB_OUTPUT_MASK_FRONTSPECULAR | CELL_GCM_ATTRIB_OUTPUT_MASK_BACKSPECULAR},
			vertex_export{"diff_color1", "dst_reg3", "", CELL_GCM_ATTRIB_OUTPUT_MASK_FRONTDIFFUSE | CELL_GCM_ATTRIB_OUTPUT_MASK_BACKDIFFUSE},
			vertex_export{"spec_color1", "dst_reg4", "", CELL_GCM_ATTRIB_OUTPUT_MASK_FRONTSPECULAR | CELL_GCM_ATTRIB_OUTPUT_MASK_BACKSPECULAR},
			vertex_export{"fogc", "dst_reg5", ".x", CELL_GCM_ATTRIB_OUTPUT_MASK_FOG},
			vertex_export{"tc0", "dst_reg7", "", CELL_GCM_ATTRIB_OUTPUT_MASK_TEX0},
			vertex_export{"tc1", "dst_reg8", "", CELL_GCM_ATTRIB_OUTPUT_MASK_TEX1},
			vertex_export{"tc2", "dst_reg9", "", CELL_GCM_ATTRIB_OUTPUT_MASK_TEX2},
			vertex_export{"tc3", "dst_reg10", "", CELL_GCM_ATTRIB_OUTPUT_MASK_TEX3},
			vertex_export{"tc4", "dst_reg11", "", CELL_GCM_ATTRIB_OUTPUT_MASK_TEX4},
			vertex_export{"tc5", "dst_reg12", "", CELL_GCM_ATTRIB_OUTPUT_MASK_TEX5},
			vertex_export{"tc6", "dst_reg13", "", CELL_GCM_ATTRIB_OUTPUT_MASK_TEX6},
			vertex_export{"tc7", "dst_reg14", "", CELL_GCM_ATTRIB_OUTPUT_MASK_TEX7},
			vertex_export{"tc8", "dst_reg15", "", CELL_GCM_ATTRIB_OUTPUT_MASK_TEX8},
			vertex_export{"tc9", "dst_reg6", "", CELL_GCM_ATTRIB_OUTPUT_MASK_TEX9},
		};

		std::string texture_type(const RSXVertexProgram& program, u32 unit, std::string_view declared_type)
		{
			msl_texture_type type;
			type.sample_type = "float";
			type.multisampled = !!(program.texture_state.multisampled_textures & (1u << unit));
			if (type.multisampled)
			{
				type.dimension = msl_texture_dimension::texture_2d;
			}
			else if (declared_type == "sampler1D")
			{
				type.dimension = msl_texture_dimension::texture_2d;
			}
			else if (declared_type == "sampler2D")
			{
				type.dimension = msl_texture_dimension::texture_2d;
			}
			else if (declared_type == "sampler3D")
			{
				type.dimension = msl_texture_dimension::texture_3d;
			}
			else if (declared_type == "samplerCube")
			{
				type.dimension = msl_texture_dimension::texture_cube;
			}
			else
			{
				fmt::throw_exception("Invalid vertex texture type '%s'", declared_type);
			}
			return get_msl_texture_type(type);
		}

		std::string find_texture_type(const ParamArray& parameters, u32 unit)
		{
			const std::string name = "vtex" + std::to_string(unit);
			for (const ParamType& type : parameters.params[PF_PARAM_UNIFORM])
			{
				for (const ParamItem& item : type.items)
				{
					if (item.name == name)
					{
						return type.type;
					}
				}
			}
			fmt::throw_exception("Vertex texture %u has no declared RSX sampler type", unit);
		}

		void append_resource_arguments(std::vector<std::string>& arguments,
			const MTLVertexProgram& program, const ParamArray& parameters, bool attributes)
		{
			const auto& bindings = program.bindings();
			const auto attribute = [attributes](std::string_view name, u32 index)
			{
				return attributes ? fmt::format(" [[%s(%u)]]", name, index) : std::string{};
			};

			arguments.push_back(fmt::format("device const uchar* persistent_input_stream%s",
				attribute("buffer", bindings.persistent_vertex_buffer.index)));
			arguments.push_back(fmt::format("device const uchar* volatile_input_stream%s",
				attribute("buffer", bindings.volatile_vertex_buffer.index)));
			arguments.push_back(fmt::format("constant draw_parameters_t* draw_parameters%s",
				attribute("buffer", bindings.draw_parameters_buffer.index)));
			arguments.push_back(fmt::format("constant vertex_context_t* vertex_contexts%s",
				attribute("buffer", bindings.context_buffer.index)));
			arguments.push_back(fmt::format("device const line_mapping_t* line_mappings%s",
				attribute("buffer", bindings.line_mapping_buffer.index)));
			if (bindings.sampler_state_buffer)
			{
				arguments.push_back(fmt::format("constant sampler_state_t* sampler_state%s",
					attribute("buffer", bindings.sampler_state_buffer.index)));
			}

			if (bindings.uses_conditional_rendering)
			{
				arguments.push_back(fmt::format("device const uint* conditional_render_predicate%s",
					attribute("buffer", bindings.conditional_render_predicate_buffer.index)));
			}
			if (bindings.uses_instanced_constants)
			{
				arguments.push_back(fmt::format("device const int* constants_addressing_lookup%s",
					attribute("buffer", bindings.instancing_lookup_buffer.index)));
				arguments.push_back(fmt::format("device const float4* instanced_constants_array%s",
					attribute("buffer", bindings.instancing_constants_buffer.index)));
			}
			else if (bindings.constants_buffer)
			{
				arguments.push_back(fmt::format("constant float4* vc%s",
					attribute("buffer", bindings.constants_buffer.index)));
			}

			for (u32 unit = 0; unit < vertex_texture_unit_count; unit++)
			{
				if (!(bindings.texture_mask & (1u << unit)))
				{
					continue;
				}
				const std::string declared_type = find_texture_type(parameters, unit);
				arguments.push_back(fmt::format("%s vtex%u%s",
					texture_type(program.rsx_program(), unit, declared_type), unit,
					attribute("texture", bindings.textures[unit].index)));
				if (!(program.rsx_program().texture_state.multisampled_textures & (1u << unit)))
				{
					arguments.push_back(fmt::format("sampler vtex%u_sampler%s", unit,
						attribute("sampler", bindings.samplers[unit].index)));
				}
			}
		}

		std::string resource_signature(const MTLVertexProgram& program, const ParamArray& parameters,
			bool attributes, bool include_registers)
		{
			std::vector<std::string> arguments;
			append_resource_arguments(arguments, program, parameters, attributes);
			arguments.push_back(attributes ? "uint vertex_id [[vertex_id]]" : "uint vertex_id");
			arguments.push_back(attributes ? "uint instance_id [[instance_id]]" : "uint instance_id");
			if (include_registers)
			{
				arguments.push_back("thread rsx_vertex_registers& registers");
			}
			return join_msl_arguments(arguments);
		}

		std::string resource_call(const MTLVertexProgram& program, bool include_registers,
			std::string_view vertex_id_name = "vertex_id",
			std::string_view register_name = "registers")
		{
			std::vector<std::string> arguments = {
				"persistent_input_stream", "volatile_input_stream", "draw_parameters", "vertex_contexts",
				"line_mappings"};
			const auto& bindings = program.bindings();
			if (bindings.sampler_state_buffer) arguments.emplace_back("sampler_state");
			if (bindings.uses_conditional_rendering)
			{
				arguments.emplace_back("conditional_render_predicate");
			}
			if (bindings.uses_instanced_constants)
			{
				arguments.emplace_back("constants_addressing_lookup");
				arguments.emplace_back("instanced_constants_array");
			}
			else if (bindings.constants_buffer)
			{
				arguments.emplace_back("vc");
			}
			for (u32 unit = 0; unit < vertex_texture_unit_count; unit++)
			{
				if (!(bindings.texture_mask & (1u << unit)))
				{
					continue;
				}
				arguments.push_back("vtex" + std::to_string(unit));
				if (!(program.rsx_program().texture_state.multisampled_textures & (1u << unit)))
				{
					arguments.push_back("vtex" + std::to_string(unit) + "_sampler");
				}
			}
			arguments.emplace_back(vertex_id_name);
			arguments.emplace_back("instance_id");
			if (include_registers)
			{
				arguments.emplace_back(register_name);
			}
			return join_msl_arguments(arguments);
		}

		u64 hash_source(std::string_view source)
		{
			u64 result = 14695981039346656037ull;
			for (const unsigned char value : source)
			{
				result = (result ^ value) * 1099511628211ull;
			}
			return result;
		}

		std::string native_diagnostic(NSError* error)
		{
			if (!error)
			{
				return "Metal returned no shader compiler diagnostic";
			}
			NSString* text = error.localizedDescription;
			return text ? text.UTF8String : "Metal returned an unreadable shader compiler diagnostic";
		}
	}

	struct MTLVertexProgram::impl
	{
		library_handle library;
		function_handle function;
	};

	void vertex_program_bindings::validate() const
	{
		const auto validate_buffer = [](const shader_binding_location& binding, u32 expected, std::string_view name)
		{
			if (!binding || binding.type != argument_binding_class::buffer || binding.index != expected ||
				!(binding.stages & argument_stage_vertex))
			{
				fmt::throw_exception("Invalid Metal vertex buffer binding for %s", name);
			}
		};

		validate_buffer(persistent_vertex_buffer, vertex_stage_binding_table::persistent_vertex_buffer, "persistent vertices");
		validate_buffer(volatile_vertex_buffer, vertex_stage_binding_table::volatile_vertex_buffer, "volatile vertices");
		validate_buffer(draw_parameters_buffer, vertex_stage_binding_table::draw_parameters_buffer, "draw parameters");
		validate_buffer(context_buffer, vertex_stage_binding_table::context_buffer, "vertex context");
		validate_buffer(line_mapping_buffer, vertex_stage_binding_table::line_mapping_buffer, "line mapping");
		if (uses_conditional_rendering)
		{
			validate_buffer(conditional_render_predicate_buffer,
				vertex_stage_binding_table::conditional_render_predicate_buffer, "conditional predicate");
		}
		if (uses_instanced_constants)
		{
			validate_buffer(instancing_lookup_buffer,
				vertex_stage_binding_table::instancing_lookup_buffer, "instancing lookup");
			validate_buffer(instancing_constants_buffer,
				vertex_stage_binding_table::instancing_constants_buffer, "instanced constants");
		}
		else if (constants_buffer)
		{
			validate_buffer(constants_buffer, vertex_stage_binding_table::constants_buffer, "vertex constants");
		}
		if (texture_mask)
		{
			validate_buffer(sampler_state_buffer, vertex_stage_binding_table::sampler_state_buffer,
				"sampler state");
		}

		for (u32 unit = 0; unit < vertex_texture_unit_count; unit++)
		{
			const bool used = !!(texture_mask & (1u << unit));
			if (used != static_cast<bool>(textures[unit]) || used != static_cast<bool>(samplers[unit]))
			{
				fmt::throw_exception("Incomplete Metal vertex texture binding for unit %u", unit);
			}
			if (used && (textures[unit] != vertex_stage_binding_table::texture(unit) ||
				samplers[unit] != vertex_stage_binding_table::sampler(unit)))
			{
				fmt::throw_exception("Unexpected Metal vertex texture binding for unit %u", unit);
			}
		}
	}

	u64 vertex_program_bindings::signature() const
	{
		u64 result = pipeline_binding_table::signature();
		result ^= static_cast<u64>(texture_mask) << 1;
		result ^= static_cast<u64>(uses_conditional_rendering) << 9;
		result ^= static_cast<u64>(uses_instanced_constants) << 10;
		result ^= static_cast<u64>(!!constants_buffer) << 11;
		result ^= static_cast<u64>(!!sampler_state_buffer) << 12;
		result ^= static_cast<u64>(line_mapping_buffer.index) << 16;
		return result;
	}

	MTLVertexDecompilerThread::MTLVertexDecompilerThread(const RSXVertexProgram& program,
		std::string& shader, MTLVertexProgram& destination, const vertex_compile_options& options)
		: VertexProgramDecompiler(program)
		, m_shader(shader)
		, m_destination(destination)
		, m_rsx_program(program)
		, m_options(options)
	{
	}

	std::string MTLVertexDecompilerThread::getFloatTypeName(usz element_count)
	{
		return get_msl_float_type(element_count);
	}

	std::string MTLVertexDecompilerThread::getIntTypeName(usz element_count)
	{
		return get_msl_int_type(element_count);
	}

	std::string MTLVertexDecompilerThread::getFunction(FUNCTION function)
	{
		const u32 unit = d2.tex_num;
		const std::string texture = "vtex" + std::to_string(unit);
		const std::string sampler_name = texture + "_sampler";
		switch (function)
		{
		case FUNCTION::VERTEX_TEXTURE_FETCH1D:
			return fmt::format("rsx_vertex_sample(%s, %s, $0.x, sampler_state[%u])",
				texture, sampler_name, unit);
		case FUNCTION::VERTEX_TEXTURE_FETCH2D:
			return fmt::format("rsx_vertex_sample(%s, %s, $0.xy, sampler_state[%u])",
				texture, sampler_name, unit);
		case FUNCTION::VERTEX_TEXTURE_FETCH3D:
			return fmt::format("rsx_vertex_sample(%s, %s, $0.xyz, sampler_state[%u])",
				texture, sampler_name, unit);
		case FUNCTION::VERTEX_TEXTURE_FETCHCUBE:
			return fmt::format("%s.sample(%s, $0.xyz, level(0.0f))", texture, sampler_name);
		case FUNCTION::VERTEX_TEXTURE_FETCH2DMS:
			return fmt::format("%s.read(uint2($0.xy * float2(%s.get_width(), %s.get_height())), 0u)",
				texture, texture, texture);
		default:
			return get_msl_function(function);
		}
	}

	std::string MTLVertexDecompilerThread::compareFunction(COMPARE comparison,
		std::string_view left, std::string_view right, bool scalar)
	{
		return get_msl_comparison(comparison, left, right, scalar);
	}

	void MTLVertexDecompilerThread::prepare_binding_table()
	{
		auto& bindings = m_destination.m_bindings;
		bindings = {};
		bindings.uses_conditional_rendering = m_options.emulate_conditional_rendering;
		bindings.uses_instanced_constants = !!(m_prog.ctrl & RSX_SHADER_CONTROL_INSTANCED_CONSTANTS);
		if (bindings.uses_conditional_rendering)
		{
			bindings.conditional_render_predicate_buffer = vertex_stage_binding_table::buffer(
				vertex_stage_binding_table::conditional_render_predicate_buffer);
		}

		for (const ParamType& type : m_parr.params[PF_PARAM_UNIFORM])
		{
			for (const ParamItem& item : type.items)
			{
				if (item.name.starts_with("vc["))
				{
					if (bindings.uses_instanced_constants)
					{
						bindings.instancing_lookup_buffer = vertex_stage_binding_table::buffer(
							vertex_stage_binding_table::instancing_lookup_buffer);
						bindings.instancing_constants_buffer = vertex_stage_binding_table::buffer(
							vertex_stage_binding_table::instancing_constants_buffer);
					}
					else
					{
						bindings.constants_buffer = vertex_stage_binding_table::buffer(
							vertex_stage_binding_table::constants_buffer);
					}
				}
				else if (type.type.starts_with("sampler"))
				{
					const u32 unit = static_cast<u32>(get_texture_index(item.name, vertex_texture_unit_count));
					bindings.texture_mask |= static_cast<u8>(1u << unit);
					bindings.textures[unit] = vertex_stage_binding_table::texture(unit);
					bindings.samplers[unit] = vertex_stage_binding_table::sampler(unit);
				}
			}
		}
		if (bindings.texture_mask)
		{
			bindings.sampler_state_buffer = vertex_stage_binding_table::buffer(
				vertex_stage_binding_table::sampler_state_buffer);
		}
		bindings.validate();

		m_inputs.clear();
		const auto add_buffer = [&](std::string name, shader_binding_location binding)
		{
			m_inputs.push_back({vertex_program_resource::buffer, binding, std::move(name), umax, argument_access::read});
		};
		add_buffer("persistent_input_stream", bindings.persistent_vertex_buffer);
		add_buffer("volatile_input_stream", bindings.volatile_vertex_buffer);
		add_buffer("draw_parameters", bindings.draw_parameters_buffer);
		add_buffer("vertex_contexts", bindings.context_buffer);
		add_buffer("line_mappings", bindings.line_mapping_buffer);
		if (bindings.sampler_state_buffer) add_buffer("sampler_state", bindings.sampler_state_buffer);
		if (bindings.uses_conditional_rendering)
		{
			add_buffer("conditional_render_predicate", bindings.conditional_render_predicate_buffer);
		}
		if (bindings.uses_instanced_constants)
		{
			add_buffer("constants_addressing_lookup", bindings.instancing_lookup_buffer);
			add_buffer("instanced_constants_array", bindings.instancing_constants_buffer);
		}
		else if (bindings.constants_buffer)
		{
			add_buffer("vc", bindings.constants_buffer);
		}
		for (u32 unit = 0; unit < vertex_texture_unit_count; unit++)
		{
			if (!(bindings.texture_mask & (1u << unit)))
			{
				continue;
			}
			m_inputs.push_back({vertex_program_resource::texture, bindings.textures[unit],
				"vtex" + std::to_string(unit), unit, argument_access::read});
			if (!(m_prog.texture_state.multisampled_textures & (1u << unit)))
			{
				m_inputs.push_back({vertex_program_resource::sampler, bindings.samplers[unit],
					"vtex" + std::to_string(unit) + "_sampler", unit, argument_access::read});
			}
		}
	}

	void MTLVertexDecompilerThread::insertHeader(std::stringstream& output)
	{
		prepare_binding_table();

		glsl::shader_properties properties{};
		properties.domain = glsl::glsl_vertex_program;
		properties.require_lit_emulation = this->properties.has_lit_op;
		properties.low_precision_tests = m_options.low_precision_comparisons;
		properties.supports_native_fp16 = m_options.use_native_half;
		output << generate_msl_prelude(msl_shader_stage::vertex, get_msl_helper_requirements(properties));
		output << R"MSL(
struct vertex_context_t
{
	float4x4 scale_offset_mat;
	uint user_clip_configuration_bits;
	uint transform_branch_bits;
	float point_size;
	float z_near;
	float z_far;
	float line_width;
	float viewport_width;
	float viewport_height;
};

struct draw_parameters_t
{
	uint vertex_base_index;
	uint vertex_index_offset;
	uint draw_id;
	uint xform_constants_offset;
	uint vs_context_offset;
	uint fs_constants_offset;
	uint fs_context_offset;
	uint fs_texture_base_index;
	uint fs_stipple_pattern_offset;
	uint reserved;
	uint2 attrib_data[16];
};

struct line_mapping_t
{
	uint vertex_id;
	uint other_vertex_id;
	float side;
	uint reserved;
};

struct sampler_state_t
{
	float4 border_color;
	float lod_bias;
	uint emulation_flags;
	uint address_modes;
	uint border_metadata;
};

inline float rsx_vertex_texture_coordinate(float coordinate, constant sampler_state_t& state,
	uint size, uint axis)
{
	if (state.emulation_flags & (1u << 5u)) coordinate /= float(max(size, 1u));
	const uint mode = (state.address_modes >> (axis * 4u)) & 0xfu;
	if (mode == 6u || mode == 7u) coordinate = abs(coordinate);
	if (mode == 4u || mode == 7u)
	{
		const float half_texel = 0.5f / float(max(size, 1u));
		coordinate = clamp(coordinate, -half_texel, 1.0f + half_texel);
	}
	return coordinate;
}

inline float4 rsx_vertex_sample(texture2d<float> texture_value, sampler sampler_value,
	float coordinate, constant sampler_state_t& state)
{
	const uint size = texture_value.get_width();
	const float transformed = rsx_vertex_texture_coordinate(coordinate, state, size, 0u);
	if ((state.emulation_flags & 1u) && (transformed < 0.0f || transformed > 1.0f))
		return state.border_color;
	return texture_value.sample(sampler_value, float2(transformed, 0.5f), level(0.0f));
}

inline float4 rsx_vertex_sample(texture2d<float> texture_value, sampler sampler_value,
	float2 coordinate, constant sampler_state_t& state)
{
	const uint2 size(texture_value.get_width(), texture_value.get_height());
	const float2 transformed(
		rsx_vertex_texture_coordinate(coordinate.x, state, size.x, 0u),
		rsx_vertex_texture_coordinate(coordinate.y, state, size.y, 1u));
	if ((state.emulation_flags & 1u) && any(transformed < 0.0f || transformed > 1.0f))
		return state.border_color;
	return texture_value.sample(sampler_value, transformed, level(0.0f));
}

inline float4 rsx_vertex_sample(texture3d<float> texture_value, sampler sampler_value,
	float3 coordinate, constant sampler_state_t& state)
{
	const uint3 size(texture_value.get_width(), texture_value.get_height(), texture_value.get_depth());
	const float3 transformed(
		rsx_vertex_texture_coordinate(coordinate.x, state, size.x, 0u),
		rsx_vertex_texture_coordinate(coordinate.y, state, size.y, 1u),
		rsx_vertex_texture_coordinate(coordinate.z, state, size.z, 2u));
	if ((state.emulation_flags & 1u) && any(transformed < 0.0f || transformed > 1.0f))
		return state.border_color;
	return texture_value.sample(sampler_value, transformed, level(0.0f));
}

inline float4 rsx_apply_zclip(float4 position, float near_plane, float far_plane)
{
	if (position.w == 0.0f)
	{
		return position;
	}
	const float real_near = min(far_plane, near_plane);
	const float real_far = max(far_plane, near_plane);
	const float depth_range = real_far - real_near;
	const float inverse_range = depth_range > 0.000001f ? 1.0f / (depth_range * position.w) : 0.0f;
	const float actual_depth = (position.z - real_near * position.w) * inverse_range;
	const float nearest_depth = floor(actual_depth + 0.5f);
	const float epsilon = (inverse_range * position.w) / 16777215.0f;
	const float depth = abs(actual_depth - nearest_depth) < epsilon ? nearest_depth : actual_depth;
	return float4(position.xy, depth * position.w, position.w);
}

inline uint rsx_extract_bits(uint value, uint offset, uint count)
{
	return (value >> offset) & ((1u << count) - 1u);
}

)MSL";
	}

	void MTLVertexDecompilerThread::insertInputs(std::stringstream& output,
		const std::vector<ParamType>&)
	{
		output << R"MSL(
struct rsx_attribute_description
{
	uint type;
	uint component_count;
	uint starting_offset;
	uint stride;
	uint frequency;
	bool swap_bytes;
	bool is_volatile;
	bool modulo;
};

inline rsx_attribute_description rsx_fetch_attribute_description(constant draw_parameters_t& draw, uint location)
{
	const uint2 packed = draw.attrib_data[location];
	rsx_attribute_description result;
	result.stride = rsx_extract_bits(packed.x, 0u, 8u);
	result.frequency = rsx_extract_bits(packed.x, 8u, 16u);
	result.type = rsx_extract_bits(packed.x, 24u, 3u);
	result.component_count = rsx_extract_bits(packed.x, 27u, 3u);
	result.starting_offset = rsx_extract_bits(packed.y, 0u, 29u);
	result.swap_bytes = ((packed.y >> 29u) & 1u) != 0u;
	result.is_volatile = ((packed.y >> 30u) & 1u) != 0u;
	result.modulo = ((packed.y >> 31u) & 1u) != 0u;
	return result;
}

inline uint rsx_load_vertex_component(device const uchar* stream, uint offset, uint size, bool swap_bytes)
{
	const uint x = stream[offset];
	if (size == 1u)
	{
		return x;
	}
	const uint y = stream[offset + 1u];
	if (size == 2u)
	{
		return swap_bytes ? (y | (x << 8u)) : (x | (y << 8u));
	}
	const uint z = stream[offset + 2u];
	const uint w = stream[offset + 3u];
	return swap_bytes ? (w | (z << 8u) | (y << 16u) | (x << 24u)) :
		(x | (y << 8u) | (z << 16u) | (w << 24u));
}

inline float rsx_sign_extend_16(uint value)
{
	return value < 0x8000u ? float(value) : float(int(value) - 65536);
}

inline float4 rsx_fetch_attribute(rsx_attribute_description description, int vertex_index,
	device const uchar* stream)
{
	uint element_size = 1u;
	if (description.type == 1u || description.type == 3u || description.type == 5u)
	{
		element_size = 2u;
	}
	else if (description.type == 2u || description.type == 6u)
	{
		element_size = 4u;
	}

	uint4 bits = uint4(0u);
	uint address = uint(vertex_index * int(description.stride) + int(description.starting_offset));
	for (uint component = 0u; component < min(description.component_count, 4u); component++)
	{
		bits[component] = rsx_load_vertex_component(stream, address, element_size, description.swap_bytes);
		address += element_size;
	}

	float scale = 1.0f;
	float4 result = float4(0.0f);
	if (description.type == 1u || description.type == 5u)
	{
		result = float4(rsx_sign_extend_16(bits.x), rsx_sign_extend_16(bits.y),
			rsx_sign_extend_16(bits.z), rsx_sign_extend_16(bits.w));
		if (description.type == 1u)
		{
			result = fma(float4(0.5f), float4(1.0f), result);
			scale = 32767.5f;
		}
	}
	else if (description.type == 2u)
	{
		result = as_type<float4>(bits);
	}
	else if (description.type == 3u)
	{
		const uint low = bits.x | (bits.y << 16u);
		const uint high = bits.z | (bits.w << 16u);
		result.xy = float2(as_type<half2>(low));
		result.zw = float2(as_type<half2>(high));
	}
	else if (description.type == 4u || description.type == 7u)
	{
		result = float4(bits);
		scale = description.type == 4u ? 255.0f : 1.0f;
	}
	else if (description.type == 6u)
	{
		const uint4 packed = uint4(rsx_extract_bits(bits.x, 0u, 11u),
			rsx_extract_bits(bits.x, 11u, 11u), rsx_extract_bits(bits.x, 22u, 10u), 32767u);
		result = float4(rsx_sign_extend_16(packed.x << 5u), rsx_sign_extend_16(packed.y << 5u),
			rsx_sign_extend_16(packed.z << 6u), float(packed.w));
		scale = 32767.0f;
	}
	if (description.component_count < 4u)
	{
		result.w = scale;
	}
	return result / scale;
}

inline float4 rsx_read_location(uint location, uint vertex_id, constant draw_parameters_t& draw,
	device const uchar* persistent_stream, device const uchar* volatile_stream)
{
	const rsx_attribute_description description = rsx_fetch_attribute_description(draw, location);
	int index = 0;
	if (description.frequency != 0u)
	{
		index = description.modulo ?
			int((vertex_id + draw.vertex_index_offset) % description.frequency) :
			(int(vertex_id) - int(draw.vertex_base_index)) / int(description.frequency);
	}
	return description.is_volatile ? rsx_fetch_attribute(description, index, volatile_stream) :
		rsx_fetch_attribute(description, index, persistent_stream);
}

)MSL";
	}

	void MTLVertexDecompilerThread::insertConstants(std::stringstream&,
		const std::vector<ParamType>&)
	{
		// Constants are supplied through the argument table emitted by insertInputs.
	}

	void MTLVertexDecompilerThread::insertOutputs(std::stringstream& output,
		const std::vector<ParamType>&)
	{
		output << "struct rsx_vertex_registers\n{\n";
		for (const ParamType& type : m_parr.params[PF_PARAM_OUT])
		{
			for (const ParamItem& item : type.items)
			{
				output << "\t" << type.type << " " << item.name << ";\n";
			}
		}
		output << "};\n\n";

		output << R"MSL(struct rsx_vertex_output
{
	float4 position [[position]];
	float point_size [[point_size]];
	float clip_distance [[clip_distance]] [6];
	float4 tc0 [[user(locn0)]];
	float4 tc1 [[user(locn1)]];
	float4 tc2 [[user(locn2)]];
	float4 tc3 [[user(locn3)]];
	float4 tc4 [[user(locn4)]];
	float4 tc5 [[user(locn5)]];
	float4 tc6 [[user(locn6)]];
	float4 tc7 [[user(locn7)]];
	float4 tc8 [[user(locn8)]];
	float4 tc9 [[user(locn9)]];
	float4 diff_color [[user(locn10)]];
	float4 diff_color1 [[user(locn11)]];
	float4 spec_color [[user(locn12)]];
	float4 spec_color1 [[user(locn13)]];
	float fogc [[user(locn14)]];
	uint4 draw_params_payload [[user(locn15)]];
};

)MSL";
	}

	void MTLVertexDecompilerThread::insertMainStart(std::stringstream& output)
	{
		const u32 constants_length = properties.has_indexed_constants ? 468u : static_cast<u32>(m_constant_ids.size());
		if (m_destination.m_bindings.uses_instanced_constants)
		{
			output << "#define _fetch_constant(index) instanced_constants_array[constants_addressing_lookup["
				"int(instance_id) * " << constants_length << " + int(index)]]\n";
		}
		else
		{
			output << "#define _fetch_constant(index) vc[xform_constants_offset + uint(index)]\n";
		}
		output << "#define _select(old_value, new_value, condition) select(old_value, new_value, condition)\n";
		output << "#define _builtin_lit(value) rsx_lit(value)\n\n";
		output << "inline void rsx_vertex_execute(" << resource_signature(m_destination, m_parr, false, true) << ")\n{\n";
		output << "\tconstant draw_parameters_t& draw = draw_parameters[0];\n";
		output << "\tconstant vertex_context_t& vertex_context = vertex_contexts[draw.vs_context_offset];\n";
		output << "\tconst uint xform_constants_offset = draw.xform_constants_offset;\n";
		output << "\tconst uint transform_branch_bits = vertex_context.transform_branch_bits;\n";

		for (const ParamType& type : m_parr.params[PF_PARAM_OUT])
		{
			for (const ParamItem& item : type.items)
			{
				output << "\tthread " << type.type << "& " << item.name << " = registers." << item.name << ";\n";
			}
		}
		for (const ParamType& type : m_parr.params[PF_PARAM_NONE])
		{
			for (const ParamItem& item : type.items)
			{
				if (item.name.starts_with("dst_reg"))
				{
					continue;
				}
				output << "\t" << type.type << " " << item.name;
				if (!item.value.empty())
				{
					output << " = " << item.value;
				}
				output << ";\n";
			}
		}
		for (const ParamType& type : m_parr.params[PF_PARAM_IN])
		{
			for (const ParamItem& item : type.items)
			{
				output << "\t" << type.type << " " << item.name << " = rsx_read_location("
					<< item.location << ", vertex_id, draw, persistent_input_stream, volatile_input_stream);\n";
			}
		}
		output << '\n';
	}

	void MTLVertexDecompilerThread::insertMainEnd(std::stringstream& output)
	{
		output << "}\n\n#undef _fetch_constant\n#undef _select\n#undef _builtin_lit\n\n";
		output << "vertex rsx_vertex_output rsx_vertex_main(" << resource_signature(m_destination, m_parr, true, false) << ")\n{\n";
		output << "\tconstant draw_parameters_t& draw = draw_parameters[0];\n";
		output << "\tconstant vertex_context_t& vertex_context = vertex_contexts[draw.vs_context_offset];\n";
		output << "\trsx_vertex_output output;\n";
		output << "\toutput.position = float4(0.0f, 0.0f, 0.0f, 1.0f);\n";
		output << "\toutput.point_size = vertex_context.point_size;\n";
		output << "\tfor (uint index = 0u; index < 6u; index++) output.clip_distance[index] = 0.5f;\n";
		for (const auto& varying : varying_exports)
		{
			output << "\toutput." << varying.name << " = " <<
				(varying.name == "fogc" ? "0.0f" : "float4(0.0f, 0.0f, 0.0f, 1.0f)") << ";\n";
		}
		output << "\toutput.draw_params_payload = uint4(draw.fs_constants_offset, draw.fs_context_offset, "
			"draw.fs_texture_base_index, draw.fs_stipple_pattern_offset);\n";

		if (m_destination.m_bindings.uses_conditional_rendering)
		{
			output << "\tif (conditional_render_predicate[0] == 0u)\n\t{\n";
			output << "\t\toutput.position = float4(0.0f, 0.0f, 0.0f, -1.0f);\n\t\treturn output;\n\t}\n";
		}
		output << "\tline_mapping_t line_mapping = {vertex_id, vertex_id, 0.0f, 0u};\n";
		output << "\tif (draw.reserved & 1u) line_mapping = line_mappings[vertex_id];\n";
		output << "\tconst uint source_vertex_id = line_mapping.vertex_id;\n";
		output << "\trsx_vertex_registers registers;\n";
		output << "\trsx_vertex_registers other_registers;\n";
		for (const ParamType& type : m_parr.params[PF_PARAM_OUT])
		{
			for (const ParamItem& item : type.items)
			{
				output << "\tregisters." << item.name << " = " <<
					(item.value.empty() ? type.type + "(0.0f)" : item.value) << ";\n";
				output << "\tother_registers." << item.name << " = registers." << item.name << ";\n";
			}
		}
		output << "\trsx_vertex_execute(" << resource_call(m_destination, true, "source_vertex_id") << ");\n";
		output << "\tif (draw.reserved & 1u) rsx_vertex_execute(";
		output << resource_call(m_destination, true, "line_mapping.other_vertex_id", "other_registers");
		output << ");\n";

		if (m_parr.HasParam(PF_PARAM_OUT, "float4", "dst_reg0"))
		{
			output << "\toutput.position = registers.dst_reg0 * vertex_context.scale_offset_mat;\n";
			output << "\toutput.position = rsx_apply_zclip(output.position, vertex_context.z_near, vertex_context.z_far);\n";
			output << "\tif (draw.reserved & 1u)\n\t{\n";
			output << "\t\tfloat4 other_position = other_registers.dst_reg0 * vertex_context.scale_offset_mat;\n";
			output << "\t\tother_position = rsx_apply_zclip(other_position, vertex_context.z_near, vertex_context.z_far);\n";
			output << "\t\tconst float2 viewport_size = max(float2(vertex_context.viewport_width, vertex_context.viewport_height), 1.0f);\n";
			output << "\t\tconst float first_w = abs(output.position.w) > 0.000001f ? output.position.w : copysign(0.000001f, output.position.w);\n";
			output << "\t\tconst float other_w = abs(other_position.w) > 0.000001f ? other_position.w : copysign(0.000001f, other_position.w);\n";
			output << "\t\tconst float2 delta = (other_position.xy / other_w - output.position.xy / first_w) * viewport_size;\n";
			output << "\t\tconst float delta_length = length(delta);\n";
			output << "\t\tconst float2 normal = delta_length > 0.000001f ? float2(-delta.y, delta.x) / delta_length : float2(0.0f, 1.0f);\n";
			output << "\t\toutput.position.xy += normal * line_mapping.side * vertex_context.line_width * output.position.w / viewport_size;\n";
			output << "\t}\n";
		}
		for (const auto& varying : varying_exports)
		{
			if ((m_rsx_program.output_mask & varying.output_mask) &&
				m_parr.HasParam(PF_PARAM_OUT, "float4", varying.register_name))
			{
				output << "\toutput." << varying.name << " = registers." << varying.register_name << varying.mask << ";\n";
			}
		}
		if ((m_rsx_program.output_mask & CELL_GCM_ATTRIB_OUTPUT_MASK_POINTSIZE) &&
			m_parr.HasParam(PF_PARAM_OUT, "float4", "dst_reg6"))
		{
			output << "\toutput.point_size = registers.dst_reg6.x;\n";
		}

		for (u32 index = 0; index < 6; index++)
		{
			const u32 mask = CELL_GCM_ATTRIB_OUTPUT_MASK_UC0 << index;
			const std::string register_name = index < 3 ? "dst_reg5" : "dst_reg6";
			const char component = "yzw"[index % 3];
			if ((m_rsx_program.output_mask & mask) && m_parr.HasParam(PF_PARAM_OUT, "float4", register_name))
			{
				output << "\t{ const uint mode = (vertex_context.user_clip_configuration_bits >> " << index * 2
					<< "u) & 3u; output.clip_distance[" << index << "] = mode != 1u ? registers."
					<< register_name << "." << component << " * (float(mode) - 1.0f) : 0.5f; }\n";
			}
		}
		output << "\treturn output;\n}\n";
	}

	void MTLVertexDecompilerThread::Task()
	{
		m_shader = Decompile();
		m_shader = fmt::replace_all(m_shader, "vec4", "float4");
		m_destination.m_inputs = m_inputs;
		m_destination.m_parameters = std::move(m_parr);
	}

	const std::vector<vertex_program_input>& MTLVertexDecompilerThread::inputs() const
	{
		return m_inputs;
	}

	MTLVertexProgram::MTLVertexProgram()
		: m_impl(std::make_unique<impl>())
	{
	}

	MTLVertexProgram::~MTLVertexProgram() = default;
	MTLVertexProgram::MTLVertexProgram(MTLVertexProgram&&) noexcept = default;
	MTLVertexProgram& MTLVertexProgram::operator=(MTLVertexProgram&&) noexcept = default;

	void MTLVertexProgram::Decompile(const RSXVertexProgram& program, const vertex_compile_options& options)
	{
		reset();
		if (program.data.empty() || program.data.size() % 4)
		{
			fmt::throw_exception("Invalid RSX vertex program length %u", program.data.size());
		}
		m_rsx_program = program;
		m_options = options;
		MTLVertexDecompilerThread decompiler(m_rsx_program, m_source, *this, m_options);
		decompiler.Task();

		has_indexed_constants = decompiler.properties.has_indexed_constants;
		constant_ids.assign(decompiler.m_constant_ids.begin(), decompiler.m_constant_ids.end());
		m_metadata.control = program.ctrl;
		m_metadata.output_mask = program.output_mask;
		m_metadata.referenced_textures_mask = m_bindings.texture_mask;
		m_metadata.instruction_count = static_cast<u32>(program.data.size() / 4);
		m_metadata.constant_count = static_cast<u32>(constant_ids.size());
		m_metadata.has_lit_op = decompiler.properties.has_lit_op;
		m_metadata.has_indexed_constants = has_indexed_constants;
		m_metadata.uses_instanced_constants = m_bindings.uses_instanced_constants;
		m_metadata.uses_conditional_rendering = m_bindings.uses_conditional_rendering;
		for (const ParamType& type : m_parameters.params[PF_PARAM_IN])
		{
			for (const ParamItem& item : type.items)
			{
				if (item.location >= 0 && item.location < 16)
				{
					m_metadata.referenced_inputs_mask |= static_cast<u16>(1u << item.location);
				}
			}
		}
		for (const auto& varying : varying_exports)
		{
			if (program.output_mask & varying.output_mask)
			{
				m_metadata.varying_mask |= static_cast<u16>(1u << get_varying_register_location(varying.name));
			}
		}
		m_metadata.varying_mask |= static_cast<u16>(1u << get_varying_register_location("usr"));
		m_metadata.source_hash = hash_source(m_source);
		m_state = vertex_compile_state::decompiled;
		if (m_options.log_source)
		{
			rsx_log.notice("Metal vertex program %u MSL:\n%s", id, m_source);
		}
	}

	void MTLVertexProgram::Compile(compiler_handle compiler)
	{
		if (m_state != vertex_compile_state::decompiled || !compiler)
		{
			fmt::throw_exception("Metal vertex program %u is not ready for compilation", id);
		}

		NSError* error = nil;
		MTLCompileOptions* compile_options = [MTLCompileOptions new];
		compile_options.languageVersion = MTLLanguageVersion4_0;
		MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
		const std::string label = m_options.label.empty() ? fmt::format("RPCS3 vertex program %u", id) : m_options.label;
		descriptor.name = [NSString stringWithUTF8String:label.c_str()];
		descriptor.source = [NSString stringWithUTF8String:m_source.c_str()];
		descriptor.options = compile_options;
		m_impl->library = [compiler newLibraryWithDescriptor:descriptor error:&error];
		if (!m_impl->library)
		{
			m_diagnostic = native_diagnostic(error);
			m_state = vertex_compile_state::failed;
			fmt::throw_exception("Metal vertex program %u compilation failed: %s", id, m_diagnostic);
		}
		m_impl->function = [m_impl->library newFunctionWithName:@"rsx_vertex_main"];
		if (!m_impl->function)
		{
			m_diagnostic = "The compiled Metal library does not contain rsx_vertex_main";
			m_state = vertex_compile_state::failed;
			fmt::throw_exception("Metal vertex program %u entry-point lookup failed", id);
		}
		m_diagnostic.clear();
		m_state = vertex_compile_state::compiled;
	}

	void MTLVertexProgram::reset()
	{
		m_impl = std::make_unique<impl>();
		m_rsx_program = {};
		m_parameters = {};
		m_source.clear();
		m_inputs.clear();
		m_bindings = {};
		m_metadata = {};
		m_options = {};
		m_state = vertex_compile_state::empty;
		m_diagnostic.clear();
		constant_ids.clear();
		has_indexed_constants = false;
	}

	MTLVertexProgram::operator bool() const
	{
		return m_state == vertex_compile_state::compiled && m_impl->function;
	}

	vertex_compile_state MTLVertexProgram::state() const
	{
		return m_state;
	}

	const RSXVertexProgram& MTLVertexProgram::rsx_program() const
	{
		return m_rsx_program;
	}

	const ParamArray& MTLVertexProgram::parameters() const
	{
		return m_parameters;
	}

	const std::string& MTLVertexProgram::source() const
	{
		return m_source;
	}

	const std::string& MTLVertexProgram::diagnostic() const
	{
		return m_diagnostic;
	}

	const vertex_program_bindings& MTLVertexProgram::bindings() const
	{
		return m_bindings;
	}

	const vertex_program_metadata& MTLVertexProgram::metadata() const
	{
		return m_metadata;
	}

	std::span<const vertex_program_input> MTLVertexProgram::inputs() const
	{
		return m_inputs;
	}

	library_handle MTLVertexProgram::library() const
	{
		return m_impl->library;
	}

	function_handle MTLVertexProgram::function() const
	{
		return m_impl->function;
	}
}
