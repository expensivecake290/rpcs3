#include "stdafx.h"
#include "MTLFragmentProgram.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <sstream>

namespace mtl
{
	namespace
	{
		std::string find_texture_type(const ParamArray& parameters, u32 unit)
		{
			const std::string name = "tex" + std::to_string(unit);
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
			fmt::throw_exception("Fragment texture %u has no declared RSX sampler type", unit);
		}

		msl_texture_dimension texture_dimension(std::string_view declared_type)
		{
			if (declared_type == "sampler1D") return msl_texture_dimension::texture_1d;
			if (declared_type == "sampler2D") return msl_texture_dimension::texture_2d;
			if (declared_type == "sampler3D") return msl_texture_dimension::texture_3d;
			if (declared_type == "samplerCube") return msl_texture_dimension::texture_cube;
			fmt::throw_exception("Invalid fragment texture type '%s'", declared_type);
		}

		std::string texture_type(const MTLFragmentProgram& program, const ParamArray& parameters,
			u32 unit, bool stencil)
		{
			msl_texture_type type;
			type.dimension = texture_dimension(find_texture_type(parameters, unit));
			type.sample_type = stencil ? "uint" : "float";
			type.multisampled = !!(program.rsx_program().texture_state.multisampled_textures & (1u << unit));
			if (type.multisampled)
			{
				type.dimension = msl_texture_dimension::texture_2d;
			}
			return get_msl_texture_type(type);
		}

		void append_resource_arguments(std::vector<std::string>& arguments,
			const MTLFragmentProgram& program, const ParamArray& parameters, bool attributes)
		{
			const auto& bindings = program.bindings();
			const auto attribute = [attributes](std::string_view name, u32 index)
			{
				return attributes ? fmt::format(" [[%s(%u)]]", name, index) : std::string{};
			};

			arguments.push_back(fmt::format("constant fragment_context_t* fragment_contexts%s",
				attribute("buffer", bindings.state_buffer.index)));
			if (bindings.constants_buffer)
			{
				arguments.push_back(fmt::format("constant float4* fc%s",
					attribute("buffer", bindings.constants_buffer.index)));
			}
			arguments.push_back(fmt::format("constant sampler_info* texture_parameters%s",
				attribute("buffer", bindings.texture_parameters_buffer.index)));
			arguments.push_back(fmt::format("device const uint4* rasterizer_environment%s",
				attribute("buffer", bindings.rasterizer_environment_buffer.index)));
			arguments.push_back(fmt::format("constant uint* sampler_state%s",
				attribute("buffer", bindings.sampler_state_buffer.index)));

			for (u32 unit = 0; unit < fragment_texture_unit_count; unit++)
			{
				const u16 bit = static_cast<u16>(1u << unit);
				if (!(bindings.texture_mask & bit))
				{
					continue;
				}
				arguments.push_back(fmt::format("%s tex%u%s", texture_type(program, parameters, unit, false),
					unit, attribute("texture", bindings.textures[unit].index)));
				if (bindings.stencil_texture_mask & bit)
				{
					arguments.push_back(fmt::format("%s tex%u_stencil%s", texture_type(program, parameters, unit, true),
						unit, attribute("texture", bindings.stencil_textures[unit].index)));
				}
				if (bindings.sampler_mask & bit)
				{
					arguments.push_back(fmt::format("sampler tex%u_sampler%s", unit,
						attribute("sampler", bindings.samplers[unit].index)));
				}
			}
			if (bindings.uses_depth_input)
			{
				const bool multisampled = !!(program.rsx_program().ctrl & RSX_SHADER_CONTROL_MULTISAMPLED_ZBUFFER);
				const std::string type = multisampled ? "texture2d_ms<float, access::read>" :
					"texture2d<float, access::read>";
				arguments.push_back(fmt::format("%s fragment_depth_input%s", type,
					attribute("texture", bindings.depth_input_texture.index)));
			}
		}

		std::string resource_signature(const MTLFragmentProgram& program, const ParamArray& parameters,
			bool attributes, bool include_registers)
		{
			std::vector<std::string> arguments;
			append_resource_arguments(arguments, program, parameters, attributes);
			arguments.push_back(attributes ? "rsx_fragment_input input [[stage_in]]" : "rsx_fragment_input input");
			arguments.push_back(attributes ? "bool front_facing [[front_facing]]" : "bool front_facing");
			arguments.push_back(attributes ? "float2 point_coord [[point_coord]]" : "float2 point_coord");
			arguments.push_back(attributes ? "uint sample_id [[sample_id]]" : "uint sample_id");
			if (include_registers)
			{
				arguments.emplace_back("thread rsx_fragment_registers& registers");
			}
			return join_msl_arguments(arguments);
		}

		std::string resource_call(const MTLFragmentProgram& program, bool include_registers)
		{
			std::vector<std::string> arguments = {"fragment_contexts"};
			const auto& bindings = program.bindings();
			if (bindings.constants_buffer) arguments.emplace_back("fc");
			arguments.emplace_back("texture_parameters");
			arguments.emplace_back("rasterizer_environment");
			arguments.emplace_back("sampler_state");
			for (u32 unit = 0; unit < fragment_texture_unit_count; unit++)
			{
				const u16 bit = static_cast<u16>(1u << unit);
				if (!(bindings.texture_mask & bit)) continue;
				arguments.push_back("tex" + std::to_string(unit));
				if (bindings.stencil_texture_mask & bit) arguments.push_back("tex" + std::to_string(unit) + "_stencil");
				if (bindings.sampler_mask & bit) arguments.push_back("tex" + std::to_string(unit) + "_sampler");
			}
			if (bindings.uses_depth_input) arguments.emplace_back("fragment_depth_input");
			arguments.emplace_back("input");
			arguments.emplace_back("front_facing");
			arguments.emplace_back("point_coord");
			arguments.emplace_back("sample_id");
			if (include_registers) arguments.emplace_back("registers");
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
			if (!error) return "Metal returned no shader compiler diagnostic";
			NSString* text = error.localizedDescription;
			return text ? text.UTF8String : "Metal returned an unreadable shader compiler diagnostic";
		}
	}

	struct MTLFragmentProgram::impl
	{
		library_handle library;
		function_handle function;
	};

	void fragment_program_bindings::validate() const
	{
		const auto validate_buffer = [](const shader_binding_location& binding, u32 expected, std::string_view name)
		{
			if (!binding || binding.type != argument_binding_class::buffer || binding.index != expected ||
				!(binding.stages & argument_stage_fragment))
			{
				fmt::throw_exception("Invalid Metal fragment buffer binding for %s", name);
			}
		};
		validate_buffer(state_buffer, fragment_stage_binding_table::state_buffer, "fragment state");
		validate_buffer(texture_parameters_buffer, fragment_stage_binding_table::texture_parameters_buffer, "texture parameters");
		validate_buffer(rasterizer_environment_buffer, fragment_stage_binding_table::rasterizer_environment_buffer, "rasterizer environment");
		validate_buffer(sampler_state_buffer, fragment_stage_binding_table::sampler_state_buffer, "sampler state");
		if (constants_buffer)
		{
			validate_buffer(constants_buffer, fragment_stage_binding_table::constants_buffer, "fragment constants");
		}
		for (u32 unit = 0; unit < fragment_texture_unit_count; unit++)
		{
			const u16 bit = static_cast<u16>(1u << unit);
			if (!!(texture_mask & bit) != static_cast<bool>(textures[unit]) ||
				!!(stencil_texture_mask & bit) != static_cast<bool>(stencil_textures[unit]) ||
				!!(sampler_mask & bit) != static_cast<bool>(samplers[unit]))
			{
				fmt::throw_exception("Incomplete Metal fragment texture binding for unit %u", unit);
			}
			if ((texture_mask & bit) && textures[unit] != fragment_stage_binding_table::texture(unit))
			{
				fmt::throw_exception("Unexpected Metal fragment texture binding for unit %u", unit);
			}
			if ((stencil_texture_mask & bit) && stencil_textures[unit] != fragment_stage_binding_table::stencil_texture(unit))
			{
				fmt::throw_exception("Unexpected Metal stencil texture binding for unit %u", unit);
			}
			if ((sampler_mask & bit) && samplers[unit] != fragment_stage_binding_table::sampler(unit))
			{
				fmt::throw_exception("Unexpected Metal fragment sampler binding for unit %u", unit);
			}
		}
		if (uses_depth_input && depth_input_texture != shader_binding_location{
			argument_binding_class::texture, fragment_stage_binding_table::depth_input_texture, argument_stage_fragment})
		{
			fmt::throw_exception("Invalid Metal fragment depth input binding");
		}
	}

	u64 fragment_program_bindings::signature() const
	{
		u64 result = pipeline_binding_table::signature();
		result ^= static_cast<u64>(texture_mask) << 1;
		result ^= static_cast<u64>(stencil_texture_mask) << 17;
		result ^= static_cast<u64>(sampler_mask) << 33;
		result ^= static_cast<u64>(!!constants_buffer) << 49;
		result ^= static_cast<u64>(uses_depth_input) << 50;
		return result;
	}

	MTLFragmentDecompilerThread::MTLFragmentDecompilerThread(std::string& shader,
		const RSXFragmentProgram& program, u32& size, MTLFragmentProgram& destination,
		const fragment_compile_options& options)
		: FragmentProgramDecompiler(program, size)
		, m_shader(shader)
		, m_destination(destination)
		, m_options(options)
	{
		device_props.has_native_half_support = options.use_native_half;
		device_props.emulate_depth_compare = options.emulate_shadow_compare;
		device_props.has_low_precision_rounding = options.low_precision_comparisons;
	}

	std::string MTLFragmentDecompilerThread::getFloatTypeName(usz element_count)
	{
		return get_msl_float_type(element_count);
	}

	std::string MTLFragmentDecompilerThread::getHalfTypeName(usz element_count)
	{
		return device_props.has_native_half_support ? get_msl_half_type(element_count) : get_msl_float_type(element_count);
	}

	std::string MTLFragmentDecompilerThread::getFunction(FUNCTION function)
	{
		return get_msl_function(function);
	}

	std::string MTLFragmentDecompilerThread::compareFunction(COMPARE comparison,
		std::string_view left, std::string_view right)
	{
		return get_msl_comparison(comparison, left, right);
	}

	void MTLFragmentDecompilerThread::prepare_binding_table()
	{
		auto& bindings = m_destination.m_bindings;
		bindings = {};
		if (!properties.constant_offsets.empty())
		{
			bindings.constants_buffer = fragment_stage_binding_table::buffer(fragment_stage_binding_table::constants_buffer);
		}
		for (const ParamType& type : m_parr.params[PF_PARAM_UNIFORM])
		{
			if (!type.type.starts_with("sampler")) continue;
			for (const ParamItem& item : type.items)
			{
				const u32 unit = static_cast<u32>(get_texture_index(item.name));
				const u16 bit = static_cast<u16>(1u << unit);
				bindings.texture_mask |= bit;
				bindings.textures[unit] = fragment_stage_binding_table::texture(unit);
				if (properties.redirected_sampler_mask & bit)
				{
					bindings.stencil_texture_mask |= bit;
					bindings.stencil_textures[unit] = fragment_stage_binding_table::stencil_texture(unit);
				}
				if (!(properties.multisampled_sampler_mask & bit))
				{
					bindings.sampler_mask |= bit;
					bindings.samplers[unit] = fragment_stage_binding_table::sampler(unit);
				}
			}
		}
		if (m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE)
		{
			bindings.uses_depth_input = true;
			bindings.depth_input_texture = {argument_binding_class::texture,
				fragment_stage_binding_table::depth_input_texture, argument_stage_fragment};
		}
		bindings.validate();

		m_inputs.clear();
		const auto add_buffer = [&](std::string name, shader_binding_location binding)
		{
			m_inputs.push_back({fragment_program_resource::buffer, binding, std::move(name), umax, argument_access::read});
		};
		add_buffer("fragment_contexts", bindings.state_buffer);
		if (bindings.constants_buffer) add_buffer("fc", bindings.constants_buffer);
		add_buffer("texture_parameters", bindings.texture_parameters_buffer);
		add_buffer("rasterizer_environment", bindings.rasterizer_environment_buffer);
		add_buffer("sampler_state", bindings.sampler_state_buffer);
		for (u32 unit = 0; unit < fragment_texture_unit_count; unit++)
		{
			const u16 bit = static_cast<u16>(1u << unit);
			if (!(bindings.texture_mask & bit)) continue;
			m_inputs.push_back({fragment_program_resource::texture, bindings.textures[unit],
				"tex" + std::to_string(unit), unit, argument_access::read});
			if (bindings.stencil_texture_mask & bit)
			{
				m_inputs.push_back({fragment_program_resource::texture, bindings.stencil_textures[unit],
					"tex" + std::to_string(unit) + "_stencil", unit, argument_access::read});
			}
			if (bindings.sampler_mask & bit)
			{
				m_inputs.push_back({fragment_program_resource::sampler, bindings.samplers[unit],
					"tex" + std::to_string(unit) + "_sampler", unit, argument_access::read});
			}
		}
		if (bindings.uses_depth_input)
		{
			m_inputs.push_back({fragment_program_resource::texture, bindings.depth_input_texture,
				"fragment_depth_input", umax, argument_access::read});
		}
	}

	void MTLFragmentDecompilerThread::insertHeader(std::stringstream& output)
	{
		prepare_binding_table();
		m_shader_properties.domain = glsl::glsl_fragment_program;
		m_shader_properties.require_lit_emulation = properties.has_lit_op;
		m_shader_properties.fp32_outputs = !!(m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS);
		m_shader_properties.require_wpos = !!(properties.in_register_mask & in_wpos);
		m_shader_properties.require_srgb_to_linear = true;
		m_shader_properties.require_linear_to_srgb = properties.has_pkg || !!(m_prog.ctrl & RSX_SHADER_CONTROL_SRGB_FRAMEBUFFER);
		m_shader_properties.require_fog_read = !!(properties.in_register_mask & in_fogc);
		m_shader_properties.emulate_shadow_compare = m_options.emulate_shadow_compare;
		m_shader_properties.emulate_depth_compare = !!(m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE);
		m_shader_properties.depth_buffer_multisampled = !!(m_prog.ctrl & RSX_SHADER_CONTROL_MULTISAMPLED_ZBUFFER);
		m_shader_properties.low_precision_tests = m_options.low_precision_comparisons;
		m_shader_properties.supports_native_fp16 = device_props.has_native_half_support;
		m_shader_properties.require_texture_ops = properties.has_tex_op;
		m_shader_properties.require_depth_conversion = properties.redirected_sampler_mask != 0;
		m_shader_properties.require_tex_shadow_ops = properties.shadow_sampler_mask != 0;
		m_shader_properties.require_msaa_ops = properties.multisampled_sampler_mask != 0;
		m_shader_properties.require_texture_expand = properties.has_exp_tex_op;
		m_shader_properties.ROP_output_rounding = m_options.round_8bit_outputs && !!(m_prog.ctrl & RSX_SHADER_CONTROL_8BIT_FRAMEBUFFER);
		m_shader_properties.ROP_sRGB_packing = !!(m_prog.ctrl & RSX_SHADER_CONTROL_SRGB_FRAMEBUFFER);
		m_shader_properties.ROP_alpha_test = !!(m_prog.ctrl & RSX_SHADER_CONTROL_ALPHA_TEST);
		m_shader_properties.ROP_alpha_to_coverage_test = !!(m_prog.ctrl & RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE);
		m_shader_properties.ROP_polygon_stipple_test = !!(m_prog.ctrl & RSX_SHADER_CONTROL_POLYGON_STIPPLE);
		m_shader_properties.ROP_discard = !!(m_prog.ctrl & RSX_SHADER_CONTROL_USES_KIL);
		m_shader_properties.require_alpha_kill = !!(m_prog.ctrl & RSX_SHADER_CONTROL_TEXTURE_ALPHA_KILL);
		m_shader_properties.require_color_format_convert = !!(m_prog.ctrl & RSX_SHADER_CONTROL_TEXTURE_FORMAT_CONVERT);

		output << generate_msl_prelude(msl_shader_stage::fragment,
			get_msl_helper_requirements(m_shader_properties));
		output << R"MSL(
struct sampler_info
{
	float scale_x, scale_y, scale_z;
	float bias_x, bias_y, bias_z;
	float clamp_min_x, clamp_min_y;
	float clamp_max_x, clamp_max_y;
	uint remap;
	uint flags;
};

struct fragment_context_t
{
	float fog_param0;
	float fog_param1;
	uint rop_control;
	float alpha_ref;
	uint fog_mode;
	float wpos_scale;
	float2 wpos_bias;
};

)MSL";
	}

	void MTLFragmentDecompilerThread::insertInputs(std::stringstream& output)
	{
		output << R"MSL(struct rsx_fragment_input
{
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
	uint4 draw_params_payload [[user(locn15), flat]];
	float4 position [[position]];
};

)MSL";
	}

	void MTLFragmentDecompilerThread::insertOutputs(std::stringstream& output)
	{
		output << "struct rsx_fragment_output\n{\n";
		for (u32 index = 0; index < m_prog.mrt_buffers_count && index < 4; index++)
		{
			output << "\tfloat4 color" << index << " [[color(" << index << ")]];\n";
			m_destination.m_metadata.output_color_masks[index] = umax;
		}
		if ((m_prog.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT) ||
			(m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE) ||
			(m_prog.ctrl & RSX_SHADER_CONTROL_DISABLE_EARLY_Z))
		{
			output << "\tfloat depth [[depth(any)]];\n";
		}
		output << "};\n\n";
	}

	void MTLFragmentDecompilerThread::insertConstants(std::stringstream&)
	{
	}

	void MTLFragmentDecompilerThread::insertGlobalFunctions(std::stringstream& output)
	{
		output << R"MSL(
inline uint floatBitsToUint(float value) { return as_type<uint>(value); }
inline uint4 floatBitsToUint(float4 value) { return as_type<uint4>(value); }
inline float uintBitsToFloat(uint value) { return as_type<float>(value); }
inline float4 uintBitsToFloat(uint4 value) { return as_type<float4>(value); }
inline uint packHalf2x16(float2 value) { return as_type<uint>(half2(value)); }
inline float2 unpackHalf2x16(uint value) { return float2(as_type<half2>(value)); }
inline uint packUnorm4x8(float4 value) { return as_type<uint>(uchar4(round(saturate(value) * 255.0f))); }
inline float4 unpackUnorm4x8(uint value) { return float4(as_type<uchar4>(value)) / 255.0f; }
inline uint packSnorm4x8(float4 value) { return as_type<uint>(char4(round(clamp(value, -1.0f, 1.0f) * 127.0f))); }
inline float4 unpackSnorm4x8(uint value) { return max(float4(as_type<char4>(value)) / 127.0f, -1.0f); }
inline uint packUnorm2x16(float2 value) { return as_type<uint>(ushort2(round(saturate(value) * 65535.0f))); }
inline float2 unpackUnorm2x16(uint value) { return float2(as_type<ushort2>(value)) / 65535.0f; }

inline float rsx_xform(float coordinate, sampler_info parameters)
{
	float result = fma(coordinate, parameters.scale_x, parameters.bias_x);
	return (parameters.flags & (1u << 21u)) ? clamp(result, parameters.clamp_min_x, parameters.clamp_max_x) : result;
}
inline float2 rsx_xform(float2 coordinate, sampler_info parameters)
{
	float2 result = fma(coordinate, float2(parameters.scale_x, parameters.scale_y),
		float2(parameters.bias_x, parameters.bias_y));
	return (parameters.flags & (1u << 21u)) ? clamp(result,
		float2(parameters.clamp_min_x, parameters.clamp_min_y),
		float2(parameters.clamp_max_x, parameters.clamp_max_y)) : result;
}
inline float3 rsx_xform(float3 coordinate, sampler_info parameters)
{
	return fma(coordinate, float3(parameters.scale_x, parameters.scale_y, parameters.scale_z),
		float3(parameters.bias_x, parameters.bias_y, parameters.bias_z));
}

inline float4 rsx_process_texel(float4 value, uint flags, bool expand_active)
{
	if ((flags & (1u << 4u)) && value.w < 0.000001f) discard_fragment();
	if (flags & (1u << 5u)) value = floor(value * 255.0f) / 255.0f;
	if (flags & 0x3c00u)
	{
		const int4 bits = int4(round(value * ((flags & (1u << 28u)) ? 65535.0f : 255.0f)));
		const float4 converted = (flags & (1u << 28u)) ? float4((bits << 16) >> 16) / 32767.0f :
			float4((bits << 24) >> 24) / 127.0f;
		value = select(value, converted, bool4(flags & (1u << 11u), flags & (1u << 12u),
			flags & (1u << 13u), flags & (1u << 10u)));
	}
	if (flags & 0x000fu)
	{
		const float3 linear = rsx_srgb_to_linear(value.rgb);
		value = select(value, float4(linear, value.w), bool4(flags & 2u, flags & 4u, flags & 8u, flags & 1u));
	}
	if ((flags & 0x03c0u) || expand_active)
	{
		const float4 converted = expand_active ? value * 2.0f - 1.0f :
			((flags & (1u << 28u)) ? (floor(value * 65535.0f + 0.5f) - 32768.0f) / 32767.0f :
			(floor(value * 255.0f + 0.5f) - 128.0f) / 127.0f);
		value = select(value, converted, expand_active ? bool4(true) : bool4(flags & (1u << 7u),
			flags & (1u << 8u), flags & (1u << 9u), flags & (1u << 6u)));
	}
	return value;
}

inline float4 rsx_decode_depth(float depth, uint stencil, sampler_info parameters)
{
	uint value = (parameters.flags & (1u << 14u)) ? ((as_type<uint>(depth) >> 7u) & 0x00ffffffu) :
		uint(depth * 16777215.0f);
	float4 result = float4(float((value >> 8u) & 255u), float(value & 255u), float(stencil & 255u),
		float((value >> 16u) & 255u)) / 255.0f;
	if (parameters.remap == 0x0000aae4u) return result;
	uint4 channels = ((uint4(parameters.remap) >> uint4(2u, 4u, 6u, 0u)) & 3u) + 3u;
	channels %= 4u;
	result = float4(result[channels.x], result[channels.y], result[channels.z], result[channels.w]);
	const uint4 controls = (uint4(parameters.remap) >> uint4(10u, 12u, 14u, 8u)) & 3u;
	return select(float4(controls), result, controls < 2u);
}

inline float4 rsx_sample_ms(texture2d_ms<float, access::read> texture_value, float2 coordinate,
	sampler_info parameters)
{
	const float2 transformed = rsx_xform(coordinate, parameters);
	const uint2 size = uint2(texture_value.get_width(), texture_value.get_height());
	const uint2 position = min(uint2(max(transformed, float2(0.0f)) * float2(size)), size - 1u);
	float4 result = float4(0.0f);
	const uint samples = texture_value.get_num_samples();
	for (uint sample = 0u; sample < samples; sample++) result += texture_value.read(position, sample);
	return result / float(max(samples, 1u));
}

#define RSX_TEXTURE_JOIN_(prefix, index) prefix##index
#define RSX_TEXTURE_JOIN(prefix, index) RSX_TEXTURE_JOIN_(prefix, index)
#define RSX_TEXTURE(index) RSX_TEXTURE_JOIN(tex, index)
#define RSX_STENCIL_NAME_(index) tex##index##_stencil
#define RSX_STENCIL(index) RSX_STENCIL_NAME_(index)
#define RSX_SAMPLER_NAME_(index) tex##index##_sampler
#define RSX_SAMPLER(index) RSX_SAMPLER_NAME_(index)
#define TEX_PARAM(index) texture_parameters[input.draw_params_payload.z + index]
#define TEX_FLAGS(index) TEX_PARAM(index).flags
#define TEX1D(index, coordinate) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index))), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX1D_BIAS(index, coordinate, value) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index)), bias(value)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX1D_LOD(index, coordinate, value) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index)), level(value)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX1D_GRAD(index, coordinate, dx, dy) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index)), gradient1d(dx, dy)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX1D_PROJ(index, coordinate) TEX1D(index, (coordinate).x / (coordinate).w)
#define TEX2D(index, coordinate) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index))), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX2D_BIAS(index, coordinate, value) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index)), bias(value)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX2D_LOD(index, coordinate, value) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index)), level(value)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX2D_GRAD(index, coordinate, dx, dy) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index)), gradient2d(dx, dy)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX2D_PROJ(index, coordinate) TEX2D(index, (coordinate).xy / (coordinate).w)
#define TEX3D(index, coordinate) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index))), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX3D_BIAS(index, coordinate, value) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index)), bias(value)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX3D_LOD(index, coordinate, value) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index)), level(value)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX3D_GRAD(index, coordinate, dx, dy) rsx_process_texel(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index)), gradient3d(dx, dy)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX3D_PROJ(index, coordinate) TEX3D(index, (coordinate).xyz / (coordinate).w)
#define TEX1D_SHADOW(index, coordinate) float4(float(rsx_shadow_compare(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform((coordinate).x, TEX_PARAM(index))).x, (coordinate).y, (TEX_FLAGS(index) >> 15u) & 7u)))
#define TEX1D_SHADOWPROJ(index, coordinate) TEX1D_SHADOW(index, float2((coordinate).x / (coordinate).w, (coordinate).y / (coordinate).w))
#define TEX2D_SHADOW(index, coordinate) float4(float(rsx_shadow_compare(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform((coordinate).xy, TEX_PARAM(index))).x, (coordinate).z, (TEX_FLAGS(index) >> 15u) & 7u)))
#define TEX2D_SHADOWPROJ(index, coordinate) TEX2D_SHADOW(index, (coordinate).xyz / (coordinate).w)
#define TEX3D_SHADOW(index, coordinate) float4(float(rsx_shadow_compare(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform((coordinate).xyz, TEX_PARAM(index))).x, (coordinate).w, (TEX_FLAGS(index) >> 15u) & 7u)))
#define TEX3D_SHADOWPROJ(index, coordinate) TEX3D_SHADOW(index, coordinate)
#define TEX2D_MS(index, coordinate) rsx_process_texel(rsx_sample_ms(RSX_TEXTURE(index), coordinate, TEX_PARAM(index)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX2D_SHADOW_MS(index, coordinate) float4(float(rsx_shadow_compare(rsx_sample_ms(RSX_TEXTURE(index), (coordinate).xy, TEX_PARAM(index)).x, (coordinate).z, (TEX_FLAGS(index) >> 15u) & 7u)))
#define TEX2D_SHADOWPROJ_MS(index, coordinate) TEX2D_SHADOW_MS(index, (coordinate).xyz / (coordinate).w)
#define TEX1D_Z24X8_RGBA8(index, coordinate) rsx_process_texel(rsx_decode_depth(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index))).x, RSX_STENCIL(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index))).x, TEX_PARAM(index)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX2D_Z24X8_RGBA8(index, coordinate) rsx_process_texel(rsx_decode_depth(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index))).x, RSX_STENCIL(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index))).x, TEX_PARAM(index)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX3D_Z24X8_RGBA8(index, coordinate) rsx_process_texel(rsx_decode_depth(RSX_TEXTURE(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index))).x, RSX_STENCIL(index).sample(RSX_SAMPLER(index), rsx_xform(coordinate, TEX_PARAM(index))).x, TEX_PARAM(index)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX1D_Z24X8_RGBA8_PROJ(index, coordinate) TEX1D_Z24X8_RGBA8(index, (coordinate).x / (coordinate).w)
#define TEX2D_Z24X8_RGBA8_PROJ(index, coordinate) TEX2D_Z24X8_RGBA8(index, (coordinate).xy / (coordinate).w)
#define TEX3D_Z24X8_RGBA8_PROJ(index, coordinate) TEX3D_Z24X8_RGBA8(index, (coordinate).xyz / (coordinate).w)
#define TEX2D_Z24X8_RGBA8_MS(index, coordinate) rsx_process_texel(rsx_decode_depth(rsx_sample_ms(RSX_TEXTURE(index), coordinate, TEX_PARAM(index)).x, 0u, TEX_PARAM(index)), TEX_FLAGS(index), rsx_texture_expand_active)
#define TEX2D_Z24X8_RGBA8_MS_PROJ(index, coordinate) TEX2D_Z24X8_RGBA8_MS(index, (coordinate).xy / (coordinate).w)
#define _enable_texture_expand(index) rsx_texture_expand_active = ((TEX_FLAGS(index) >> 26u) & 1u) != 0u
#define _disable_texture_expand() rsx_texture_expand_active = false
#define _select(old_value, new_value, condition) select(old_value, new_value, condition)
#define _saturate(value) saturate(value)
#define _builtin_lit(value) rsx_lit(value)
inline float4 _builtin_lif(float4 value) { return float4(1.0f, value.y, value.y > 0.0f ? exp2(value.w) : 0.0f, 1.0f); }

)MSL";

		if (properties.has_pkg)
		{
			output << "inline float4 _builtin_pkg(float4 value) { float4 converted = float4(rsx_linear_to_srgb(value.rgb), value.w); return float4(as_type<float>(packUnorm4x8(converted))); }\n";
		}
		if (properties.has_upg)
		{
			output << "inline float4 _builtin_upg(float value) { float4 raw = unpackUnorm4x8(as_type<uint>(value)); return float4(rsx_srgb_to_linear(raw.rgb), raw.w); }\n";
		}
		if (properties.has_divsq)
		{
			output << "inline float4 _builtin_divsq(float4 a, float b) { float4 value = a / sqrt(abs(b)); return select(a, value, abs(a) > 0.0f); }\n";
		}
		output << '\n';
	}

	void MTLFragmentDecompilerThread::insertMainStart(std::stringstream& output)
	{
		if (!properties.constant_offsets.empty())
		{
			output << "#define _fetch_constant(index) fc[input.draw_params_payload.x + uint(index)]\n";
		}
		output << "struct rsx_fragment_registers\n{\n";
		const bool fp32 = !!(m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS);
		const std::array<std::string_view, 4> output_names = fp32 ?
			std::array<std::string_view, 4>{"r0", "r2", "r3", "r4"} :
			std::array<std::string_view, 4>{"h0", "h4", "h6", "h8"};
		const std::string output_type = fp32 || !device_props.has_native_half_support ? "float4" : "half4";
		for (const auto name : output_names) output << "\t" << output_type << " " << name << ";\n";
		if (m_prog.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT) output << "\tfloat4 r1;\n";
		output << "};\n\n";
		output << "inline void rsx_fragment_execute(" << resource_signature(m_destination, m_parr, false, true) << ")\n{\n";
		output << "\tconst float4 fragment_position = input.position;\n";
		output << "\tbool rsx_texture_expand_active = false;\n";
		output << "\tconst float in_w = 1.0f / fragment_position.w;\n";
		output << "\tconst float4 ssa = front_facing ? float4(1.0f) : float4(-1.0f);\n";
		output << "\tconst float4 wpos = float4(fragment_position.xy, fragment_position.z, 1.0f / fragment_position.w);\n";
		output << "\tconst float4 fogc = float4(input.fogc, 0.0f, 0.0f, 0.0f);\n";
		output << "\tconst float4 diff_color = front_facing ? input.diff_color : input.diff_color1;\n";
		output << "\tconst float4 spec_color = front_facing ? input.spec_color : input.spec_color1;\n";
		for (u32 index = 0; index < 10; index++) output << "\tconst float4 tc" << index << " = input.tc" << index << ";\n";
		for (const ParamType& type : m_parr.params[PF_PARAM_NONE])
		{
			for (const ParamItem& item : type.items)
			{
				bool is_output = false;
				for (const auto name : output_names) is_output |= item.name == name;
				is_output |= item.name == "r1" && !!(m_prog.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT);
				if (is_output)
				{
					output << "\tthread " << type.type << "& " << item.name << " = registers." << item.name << ";\n";
				}
				else
				{
					output << "\t" << type.type << " " << item.name;
					if (!item.value.empty()) output << " = " << item.value;
					output << ";\n";
				}
			}
		}
		output << '\n';
	}

	void MTLFragmentDecompilerThread::insertMainEnd(std::stringstream& output)
	{
		output << "}\n\n";
		if (!properties.constant_offsets.empty()) output << "#undef _fetch_constant\n";
		output << "fragment rsx_fragment_output rsx_fragment_main(" << resource_signature(m_destination, m_parr, true, false) << ")\n{\n";
		output << "\trsx_fragment_output output;\n";
		output << "\trsx_fragment_registers registers;\n";
		const bool fp32 = !!(m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS);
		const std::array<std::string_view, 4> names = fp32 ?
			std::array<std::string_view, 4>{"r0", "r2", "r3", "r4"} :
			std::array<std::string_view, 4>{"h0", "h4", "h6", "h8"};
		for (const auto name : names) output << "\tregisters." << name << " = " <<
			(fp32 || !device_props.has_native_half_support ? "float4(0.0f);\n" : "half4(0.0h);\n");
		if (m_prog.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT) output << "\tregisters.r1 = float4(0.0f);\n";
		output << "\trsx_fragment_execute(" << resource_call(m_destination, true) << ");\n";
		output << "\tconst fragment_context_t context = fragment_contexts[input.draw_params_payload.y];\n";

		if (m_prog.ctrl & RSX_SHADER_CONTROL_POLYGON_STIPPLE)
		{
			output << "\tif (context.rop_control & 8u)\n\t{\n";
			output << "\t\tconst uint2 coordinate = uint2(input.position.xy) & 31u;\n";
			output << "\t\tconst uint address = coordinate.y * 32u + coordinate.x;\n";
			output << "\t\tconst uint4 word = rasterizer_environment[input.draw_params_payload.w + ((address >> 7u) & 7u)];\n";
			output << "\t\tif (((word[(address >> 5u) & 3u] >> (address & 31u)) & 1u) == 0u) discard_fragment();\n\t}\n";
		}

		std::array<std::string, 4> colors;
		for (u32 index = 0; index < 4; index++) colors[index] = "float4(registers." + std::string(names[index]) + ")";
		for (u32 index = 0; index < m_prog.mrt_buffers_count && index < 4; index++)
		{
			output << "\tfloat4 color" << index << " = " << colors[index] << ";\n";
			if (m_prog.ctrl & RSX_SHADER_CONTROL_SRGB_FRAMEBUFFER)
			{
				output << "\tif (context.rop_control & 2u) color" << index << ".rgb = rsx_linear_to_srgb(color" << index << ".rgb);\n";
			}
			if (m_shader_properties.ROP_output_rounding)
			{
				output << "\tcolor" << index << " = rsx_round_output(color" << index << ");\n";
			}
		}
		if ((m_prog.ctrl & RSX_SHADER_CONTROL_ALPHA_TEST) && m_prog.mrt_buffers_count)
		{
			output << "\tif ((context.rop_control & 1u) && !rsx_alpha_test(color0.w, context.alpha_ref, (context.rop_control >> 20u) & 7u)) discard_fragment();\n";
		}
		if ((m_prog.ctrl & RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE) && m_prog.mrt_buffers_count)
		{
			output << "\tconst uint noise = uint(input.position.x) * 1664525u + uint(input.position.y) * 1013904223u + sample_id * 747796405u;\n";
			output << "\tif (color0.w <= float(noise & 0xffffu) / 65535.0f) discard_fragment();\n";
		}
		for (u32 index = 0; index < m_prog.mrt_buffers_count && index < 4; index++)
		{
			output << "\toutput.color" << index << " = color" << index << ";\n";
		}

		const bool has_depth = (m_prog.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT) ||
			(m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE) ||
			(m_prog.ctrl & RSX_SHADER_CONTROL_DISABLE_EARLY_Z);
		if (has_depth)
		{
			output << "\tfloat fragment_depth = input.position.z;\n";
			if (m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE)
			{
				if (m_prog.ctrl & RSX_SHADER_CONTROL_MULTISAMPLED_ZBUFFER)
					output << "\tconst float destination_depth = fragment_depth_input.read(uint2(input.position.xy), sample_id).x;\n";
				else
					output << "\tconst float destination_depth = fragment_depth_input.read(uint2(input.position.xy), 0u).x;\n";
				output << "\tconst float depth_scale = (context.rop_control & (1u << 18u)) ? 16777215.0f : 65535.0f;\n";
				output << "\tif (abs(fragment_depth - destination_depth) < 1.0f / depth_scale) fragment_depth = destination_depth;\n";
			}
			if ((m_prog.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT) && m_parr.HasParam(PF_PARAM_NONE, "float4", "r1"))
			{
				output << "\tfragment_depth = saturate(registers.r1.z);\n";
			}
			output << "\toutput.depth = fragment_depth;\n";
		}
		output << "\treturn output;\n}\n";
	}

	void MTLFragmentDecompilerThread::Task()
	{
		m_shader = Decompile();
		const std::pair<std::string_view, std::string> replacements[] = {
			{"bvec4", "bool4"}, {"ivec4", "int4"}, {"uvec4", "uint4"},
			{"vec4", "float4"}, {"vec3", "float3"}, {"vec2", "float2"},
			{"gl_FragCoord", "fragment_position"}, {"gl_PointCoord", "point_coord"},
			{"gl_FrontFacing", "front_facing"}, {"discard;", "discard_fragment();"}};
		m_shader = fmt::replace_all(m_shader, replacements);
		m_destination.m_inputs = m_inputs;
		m_destination.m_parameters = std::move(m_parr);
	}

	const std::vector<fragment_program_input>& MTLFragmentDecompilerThread::inputs() const { return m_inputs; }
	const glsl::shader_properties& MTLFragmentDecompilerThread::shader_properties() const { return m_shader_properties; }

	MTLFragmentProgram::MTLFragmentProgram() : m_impl(std::make_unique<impl>()) {}
	MTLFragmentProgram::~MTLFragmentProgram() = default;
	MTLFragmentProgram::MTLFragmentProgram(MTLFragmentProgram&&) noexcept = default;
	MTLFragmentProgram& MTLFragmentProgram::operator=(MTLFragmentProgram&&) noexcept = default;

	void MTLFragmentProgram::Decompile(const RSXFragmentProgram& program, u32 id,
		const fragment_compile_options& options)
	{
		reset();
		if (!program.get_data() || !program.ucode_length)
		{
			fmt::throw_exception("Invalid RSX fragment program storage");
		}
		m_rsx_program = RSXFragmentProgram::clone(program);
		m_id = id;
		m_options = options;
		u32 size = 0;
		MTLFragmentDecompilerThread decompiler(m_source, m_rsx_program, size, *this, options);
		decompiler.Task();
		m_constant_offsets = std::move(decompiler.properties.constant_offsets);
		m_metadata.control = program.ctrl;
		m_metadata.input_mask = decompiler.properties.in_register_mask;
		m_metadata.common_texture_mask = decompiler.properties.common_access_sampler_mask;
		m_metadata.shadow_texture_mask = decompiler.properties.shadow_sampler_mask;
		m_metadata.redirected_texture_mask = decompiler.properties.redirected_sampler_mask;
		m_metadata.multisampled_texture_mask = decompiler.properties.multisampled_sampler_mask;
		m_metadata.instruction_bytes = size;
		m_metadata.constant_count = static_cast<u32>(m_constant_offsets.size());
		m_metadata.writes_depth = !!(program.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT);
		m_metadata.uses_discard = decompiler.properties.has_discard_op || !!(program.ctrl & RSX_SHADER_CONTROL_USES_KIL);
		m_metadata.uses_half = !(program.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS) && options.use_native_half;
		m_metadata.has_no_output = decompiler.properties.has_no_output;
		m_metadata.source_hash = hash_source(m_source);
		m_state = fragment_compile_state::decompiled;
		if (m_options.log_source) rsx_log.notice("Metal fragment program %u MSL:\n%s", m_id, m_source);
	}

	void MTLFragmentProgram::Compile(compiler_handle compiler)
	{
		if (m_state != fragment_compile_state::decompiled || !compiler)
		{
			fmt::throw_exception("Metal fragment program %u is not ready for compilation", m_id);
		}
		NSError* error = nil;
		MTLCompileOptions* options = [MTLCompileOptions new];
		options.languageVersion = MTLLanguageVersion4_0;
		MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
		const std::string label = m_options.label.empty() ? fmt::format("RPCS3 fragment program %u", m_id) : m_options.label;
		descriptor.name = [NSString stringWithUTF8String:label.c_str()];
		descriptor.source = [NSString stringWithUTF8String:m_source.c_str()];
		descriptor.options = options;
		m_impl->library = [compiler newLibraryWithDescriptor:descriptor error:&error];
		if (!m_impl->library)
		{
			m_diagnostic = native_diagnostic(error);
			m_state = fragment_compile_state::failed;
			fmt::throw_exception("Metal fragment program %u compilation failed: %s", m_id, m_diagnostic);
		}
		m_impl->function = [m_impl->library newFunctionWithName:@"rsx_fragment_main"];
		if (!m_impl->function)
		{
			m_diagnostic = "The compiled Metal library does not contain rsx_fragment_main";
			m_state = fragment_compile_state::failed;
			fmt::throw_exception("Metal fragment program %u entry-point lookup failed", m_id);
		}
		m_diagnostic.clear();
		m_state = fragment_compile_state::compiled;
	}

	void MTLFragmentProgram::reset()
	{
		m_impl = std::make_unique<impl>();
		m_rsx_program = {};
		m_parameters = {};
		m_source.clear();
		m_inputs.clear();
		m_constant_offsets.clear();
		m_bindings = {};
		m_metadata = {};
		m_options = {};
		m_state = fragment_compile_state::empty;
		m_diagnostic.clear();
		m_id = 0;
	}

	MTLFragmentProgram::operator bool() const { return m_state == fragment_compile_state::compiled && m_impl->function; }
	u32 MTLFragmentProgram::id() const { return m_id; }
	fragment_compile_state MTLFragmentProgram::state() const { return m_state; }
	const RSXFragmentProgram& MTLFragmentProgram::rsx_program() const { return m_rsx_program; }
	const ParamArray& MTLFragmentProgram::parameters() const { return m_parameters; }
	const std::string& MTLFragmentProgram::source() const { return m_source; }
	const std::string& MTLFragmentProgram::diagnostic() const { return m_diagnostic; }
	const fragment_program_bindings& MTLFragmentProgram::bindings() const { return m_bindings; }
	const fragment_program_metadata& MTLFragmentProgram::metadata() const { return m_metadata; }
	std::span<const fragment_program_input> MTLFragmentProgram::inputs() const { return m_inputs; }
	std::span<const u32> MTLFragmentProgram::constant_offsets() const { return m_constant_offsets; }
	library_handle MTLFragmentProgram::library() const { return m_impl->library; }
	function_handle MTLFragmentProgram::function() const { return m_impl->function; }
}
