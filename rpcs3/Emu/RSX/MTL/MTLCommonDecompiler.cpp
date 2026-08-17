#include "stdafx.h"
#include "MTLCommonDecompiler.h"

#include <array>
#include <cctype>
#include <limits>

namespace mtl
{
	namespace
	{
		constexpr std::array<msl_varying_location, 17> varying_registers = {{
			{"tc0", 0, 4, false},
			{"tc1", 1, 4, false},
			{"tc2", 2, 4, false},
			{"tc3", 3, 4, false},
			{"tc4", 4, 4, false},
			{"tc5", 5, 4, false},
			{"tc6", 6, 4, false},
			{"tc7", 7, 4, false},
			{"tc8", 8, 4, false},
			{"tc9", 9, 4, false},
			{"diff_color", 10, 4, false},
			{"diff_color1", 11, 4, false},
			{"spec_color", 12, 4, false},
			{"spec_color1", 13, 4, false},
			{"fogc", 14, 1, false},
			{"fog_c", 14, 1, false},
			{"usr", 15, 4, true},
		}};

		std::string vector_type(std::string_view scalar, usz components)
		{
			if (components < 1 || components > 4)
			{
				fmt::throw_exception("Invalid MSL vector width %u", components);
			}

			return components == 1 ? std::string(scalar) : fmt::format("%s%u", scalar, components);
		}

		std::string comparison_operator(COMPARE comparison)
		{
			switch (comparison)
			{
			case COMPARE::SEQ: return "==";
			case COMPARE::SGE: return ">=";
			case COMPARE::SGT: return ">";
			case COMPARE::SLE: return "<=";
			case COMPARE::SLT: return "<";
			case COMPARE::SNE: return "!=";
			}

			fmt::throw_exception("Unknown MSL comparison %d", static_cast<int>(comparison));
		}

		bool is_reserved_identifier(std::string_view name)
		{
			static constexpr std::array reserved = {
				"alignas", "alignof", "and", "and_eq", "asm", "atomic", "auto", "bitand", "bitor",
				"bool", "break", "case", "catch", "char", "class", "compl", "const", "constant",
				"constexpr", "continue", "decltype", "default", "delete", "device", "discard_fragment",
				"do", "double", "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false",
				"float", "for", "fragment", "friend", "goto", "half", "if", "inline", "int", "kernel",
				"long", "mutable", "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator",
				"or", "or_eq", "packed_float3", "private", "protected", "public", "register", "reinterpret_cast",
				"return", "sampler", "short", "signed", "sizeof", "static", "static_assert", "static_cast",
				"struct", "switch", "template", "texture", "thread", "threadgroup", "throw", "true", "try",
				"typedef", "typeid", "typename", "uint", "union", "unsigned", "using", "vertex", "virtual",
				"void", "volatile", "while", "xor", "xor_eq"};

			for (const std::string_view reserved_name : reserved)
			{
				if (name == reserved_name)
				{
					return true;
				}
			}
			return false;
		}
	}

	void msl_source_builder::append(std::string_view text)
	{
		m_source.append(text);
	}

	void msl_source_builder::line(std::string_view text)
	{
		m_source.append(m_indentation, '\t');
		m_source.append(text);
		m_source.push_back('\n');
	}

	void msl_source_builder::open_scope(std::string_view declaration)
	{
		if (!declaration.empty())
		{
			line(declaration);
		}
		line("{");
		m_indentation++;
	}

	void msl_source_builder::close_scope(std::string_view suffix)
	{
		if (!m_indentation)
		{
			fmt::throw_exception("Cannot close an unopened MSL source scope");
		}
		m_indentation--;
		line(fmt::format("}%s", suffix));
	}

	u32 msl_source_builder::indentation() const
	{
		return m_indentation;
	}

	const std::string& msl_source_builder::str() const
	{
		return m_source;
	}

	std::string msl_source_builder::take()
	{
		if (m_indentation)
		{
			fmt::throw_exception("Cannot finish MSL source with %u open scopes", m_indentation);
		}
		return std::move(m_source);
	}

	msl_varying_location get_varying_register(std::string_view name)
	{
		for (const auto& varying : varying_registers)
		{
			if (varying.canonical_name == name || (name == "fog_c" && varying.canonical_name == "fogc"))
			{
				return varying.canonical_name == "fog_c" ? varying_registers[14] : varying;
			}
		}

		fmt::throw_exception("Unknown MSL varying register: %s", name);
	}

	int get_varying_register_location(std::string_view name)
	{
		return static_cast<int>(get_varying_register(name).location);
	}

	int get_texture_index(std::string_view name, u32 maximum)
	{
		if (name.empty() || !maximum)
		{
			fmt::throw_exception("Invalid MSL texture name '%s'", name);
		}

		usz digits_end = std::string_view::npos;
		for (usz index = name.size(); index > 0; index--)
		{
			if (std::isdigit(static_cast<unsigned char>(name[index - 1])))
			{
				digits_end = index;
				break;
			}
		}
		if (digits_end == std::string_view::npos)
		{
			fmt::throw_exception("Invalid MSL texture name '%s'", name);
		}

		usz digits_begin = digits_end - 1;
		while (digits_begin && std::isdigit(static_cast<unsigned char>(name[digits_begin - 1])))
		{
			digits_begin--;
		}

		u64 value = 0;
		for (usz index = digits_begin; index < digits_end; index++)
		{
			value = value * 10 + static_cast<u64>(name[index] - '0');
			if (value > std::numeric_limits<u32>::max())
			{
				fmt::throw_exception("MSL texture index in '%s' is too large", name);
			}
		}
		if (value >= maximum)
		{
			fmt::throw_exception("MSL texture index %u exceeds the %u-slot binding table", value, maximum);
		}
		return static_cast<int>(value);
	}

	std::string sanitize_msl_identifier(std::string_view name)
	{
		std::string result;
		result.reserve(name.size() + 4);
		for (const unsigned char character : name)
		{
			result.push_back(std::isalnum(character) || character == '_' ? static_cast<char>(character) : '_');
		}

		if (result.empty())
		{
			return "rsx_value";
		}
		if (std::isdigit(static_cast<unsigned char>(result.front())) || is_reserved_identifier(result) || result.starts_with("__"))
		{
			result.insert(0, "rsx_");
		}
		return result;
	}

	std::string get_msl_float_type(usz components)
	{
		return vector_type("float", components);
	}

	std::string get_msl_half_type(usz components)
	{
		return vector_type("half", components);
	}

	std::string get_msl_int_type(usz components)
	{
		return vector_type("int", components);
	}

	std::string get_msl_uint_type(usz components)
	{
		return vector_type("uint", components);
	}

	std::string get_msl_texture_type(const msl_texture_type& type)
	{
		if (type.sample_type.empty())
		{
			fmt::throw_exception("An MSL texture sample type is required");
		}
		if (type.multisampled && (type.dimension != msl_texture_dimension::texture_2d || type.array))
		{
			fmt::throw_exception("MSL multisample textures must be non-array 2D textures");
		}
		if (type.depth && (type.dimension == msl_texture_dimension::texture_1d || type.dimension == msl_texture_dimension::texture_3d))
		{
			fmt::throw_exception("MSL depth textures must be 2D or cube textures");
		}
		if (type.writable && type.depth)
		{
			fmt::throw_exception("MSL depth textures cannot use read-write access");
		}

		std::string base;
		if (type.depth)
		{
			base = type.dimension == msl_texture_dimension::texture_cube ? "depthcube" : "depth2d";
		}
		else
		{
			switch (type.dimension)
			{
			case msl_texture_dimension::texture_1d: base = "texture1d"; break;
			case msl_texture_dimension::texture_2d: base = "texture2d"; break;
			case msl_texture_dimension::texture_3d: base = "texture3d"; break;
			case msl_texture_dimension::texture_cube: base = "texturecube"; break;
			}
		}

		if (type.multisampled)
		{
			base += "_ms";
		}
		else if (type.array)
		{
			base += "_array";
		}
		const std::string_view access = type.writable ? "read_write" : (type.multisampled ? "read" : "sample");
		return fmt::format("%s<%s, access::%s>", base, type.sample_type, access);
	}

	std::string get_msl_function(FUNCTION function)
	{
		switch (function)
		{
		case FUNCTION::DP2: return "$Ty(dot($0.xy, $1.xy))";
		case FUNCTION::DP2A: return "$Ty(dot($0.xy, $1.xy) + $2.x)";
		case FUNCTION::DP3: return "$Ty(dot($0.xyz, $1.xyz))";
		case FUNCTION::DP4: return "$Ty(dot($0, $1))";
		case FUNCTION::DPH: return "$Ty(dot(float4($0.xyz, 1.0f), $1))";
		case FUNCTION::SFL: return "$Ty(0.0f)";
		case FUNCTION::STR: return "$Ty(1.0f)";
		case FUNCTION::FRACT: return "fract($0)";
		case FUNCTION::DFDX: return "dfdx($0)";
		case FUNCTION::DFDY: return "dfdy($0)";
		case FUNCTION::REFL: return "reflect($0, $1)";
		case FUNCTION::TEXTURE_SAMPLE1D: return "TEX1D($_i, $0.x)";
		case FUNCTION::TEXTURE_SAMPLE1D_BIAS: return "TEX1D_BIAS($_i, $0.x, $1.x)";
		case FUNCTION::TEXTURE_SAMPLE1D_PROJ: return "TEX1D_PROJ($_i, $0)";
		case FUNCTION::TEXTURE_SAMPLE1D_LOD: return "TEX1D_LOD($_i, $0.x, $1.x)";
		case FUNCTION::TEXTURE_SAMPLE1D_GRAD: return "TEX1D_GRAD($_i, $0.x, $1.x, $2.x)";
		case FUNCTION::TEXTURE_SAMPLE1D_SHADOW: return "TEX1D_SHADOW($_i, $0.xy)";
		case FUNCTION::TEXTURE_SAMPLE1D_SHADOW_PROJ: return "TEX1D_SHADOWPROJ($_i, $0)";
		case FUNCTION::TEXTURE_SAMPLE1D_DEPTH_RGBA: return "TEX1D_Z24X8_RGBA8($_i, $0.x)";
		case FUNCTION::TEXTURE_SAMPLE1D_DEPTH_RGBA_PROJ: return "TEX1D_Z24X8_RGBA8($_i, $0.x / $0.w)";
		case FUNCTION::TEXTURE_SAMPLE2D: return "TEX2D($_i, $0.xy)";
		case FUNCTION::TEXTURE_SAMPLE2D_BIAS: return "TEX2D_BIAS($_i, $0.xy, $1.x)";
		case FUNCTION::TEXTURE_SAMPLE2D_PROJ: return "TEX2D_PROJ($_i, $0)";
		case FUNCTION::TEXTURE_SAMPLE2D_LOD: return "TEX2D_LOD($_i, $0.xy, $1.x)";
		case FUNCTION::TEXTURE_SAMPLE2D_GRAD: return "TEX2D_GRAD($_i, $0.xy, $1.xy, $2.xy)";
		case FUNCTION::TEXTURE_SAMPLE2D_SHADOW: return "TEX2D_SHADOW($_i, $0.xyz)";
		case FUNCTION::TEXTURE_SAMPLE2D_SHADOW_PROJ: return "TEX2D_SHADOWPROJ($_i, $0)";
		case FUNCTION::TEXTURE_SAMPLE2D_DEPTH_RGBA: return "TEX2D_Z24X8_RGBA8($_i, $0.xy)";
		case FUNCTION::TEXTURE_SAMPLE2D_DEPTH_RGBA_PROJ: return "TEX2D_Z24X8_RGBA8($_i, $0.xy / $0.w)";
		case FUNCTION::TEXTURE_SAMPLE3D: return "TEX3D($_i, $0.xyz)";
		case FUNCTION::TEXTURE_SAMPLE3D_BIAS: return "TEX3D_BIAS($_i, $0.xyz, $1.x)";
		case FUNCTION::TEXTURE_SAMPLE3D_PROJ: return "TEX3D_PROJ($_i, $0)";
		case FUNCTION::TEXTURE_SAMPLE3D_LOD: return "TEX3D_LOD($_i, $0.xyz, $1.x)";
		case FUNCTION::TEXTURE_SAMPLE3D_GRAD: return "TEX3D_GRAD($_i, $0.xyz, $1.xyz, $2.xyz)";
		case FUNCTION::TEXTURE_SAMPLE3D_SHADOW: return "TEX3D_SHADOW($_i, $0)";
		case FUNCTION::TEXTURE_SAMPLE3D_SHADOW_PROJ: return "TEX3D_SHADOWPROJ($_i, $0)";
		case FUNCTION::TEXTURE_SAMPLE3D_DEPTH_RGBA: return "TEX3D_Z24X8_RGBA8($_i, $0.xyz)";
		case FUNCTION::TEXTURE_SAMPLE3D_DEPTH_RGBA_PROJ: return "TEX3D_Z24X8_RGBA8($_i, $0.xyz / $0.w)";
		case FUNCTION::TEXTURE_SAMPLE2DMS:
		case FUNCTION::TEXTURE_SAMPLE2DMS_BIAS:
		case FUNCTION::TEXTURE_SAMPLE2DMS_LOD:
		case FUNCTION::TEXTURE_SAMPLE2DMS_GRAD: return "TEX2D_MS($_i, $0.xy)";
		case FUNCTION::TEXTURE_SAMPLE2DMS_PROJ: return "TEX2D_MS($_i, $0.xy / $0.w)";
		case FUNCTION::TEXTURE_SAMPLE2DMS_SHADOW: return "TEX2D_SHADOW_MS($_i, $0.xyz)";
		case FUNCTION::TEXTURE_SAMPLE2DMS_SHADOW_PROJ: return "TEX2D_SHADOWPROJ_MS($_i, $0)";
		case FUNCTION::TEXTURE_SAMPLE2DMS_DEPTH_RGBA: return "TEX2D_Z24X8_RGBA8_MS($_i, $0.xy)";
		case FUNCTION::TEXTURE_SAMPLE2DMS_DEPTH_RGBA_PROJ: return "TEX2D_Z24X8_RGBA8_MS($_i, $0.xy / $0.w)";
		case FUNCTION::VERTEX_TEXTURE_FETCH1D: return "$t.sample($s, $0.x, level(0.0f))";
		case FUNCTION::VERTEX_TEXTURE_FETCH2D: return "$t.sample($s, $0.xy, level(0.0f))";
		case FUNCTION::VERTEX_TEXTURE_FETCH3D:
		case FUNCTION::VERTEX_TEXTURE_FETCHCUBE: return "$t.sample($s, $0.xyz, level(0.0f))";
		case FUNCTION::VERTEX_TEXTURE_FETCH2DMS: return "$t.read(uint2($0.xy * float2($t.get_width(), $t.get_height())), 0)";
		default:
			fmt::throw_exception("Unexpected MSL function request: %d", static_cast<int>(function));
		}
	}

	std::string get_msl_comparison(COMPARE comparison, std::string_view left, std::string_view right, bool scalar)
	{
		const std::string operation = fmt::format("(rsx_cmp_fixup(%s) %s rsx_cmp_fixup(%s))",
			left, comparison_operator(comparison), right);
		return scalar ? operation : operation;
	}

	u32 vertex_texture_binding(u32 texture_index)
	{
		const auto binding = vertex_stage_binding_table::texture(texture_index);
		if (!binding)
		{
			fmt::throw_exception("Invalid vertex texture unit %u", texture_index);
		}
		return binding.index;
	}

	u32 fragment_texture_binding(u32 texture_index, bool stencil_view)
	{
		const auto binding = stencil_view ? fragment_stage_binding_table::stencil_texture(texture_index) :
			fragment_stage_binding_table::texture(texture_index);
		if (!binding)
		{
			fmt::throw_exception("Invalid fragment texture unit %u", texture_index);
		}
		return binding.index;
	}

	u32 vertex_sampler_binding(u32 texture_index)
	{
		const auto binding = vertex_stage_binding_table::sampler(texture_index);
		if (!binding)
		{
			fmt::throw_exception("Invalid vertex sampler unit %u", texture_index);
		}
		return binding.index;
	}

	u32 fragment_sampler_binding(u32 texture_index)
	{
		const auto binding = fragment_stage_binding_table::sampler(texture_index);
		if (!binding)
		{
			fmt::throw_exception("Invalid fragment sampler unit %u", texture_index);
		}
		return binding.index;
	}

	msl_helper_requirements get_msl_helper_requirements(const glsl::shader_properties& properties)
	{
		msl_helper_requirements result;
		result.domain = properties.domain;
		result.lit = properties.require_lit_emulation;
		result.comparison = properties.low_precision_tests || properties.ROP_alpha_test ||
			(properties.require_msaa_ops && properties.require_tex_shadow_ops);
		result.low_precision_comparisons = properties.low_precision_tests;
		result.fog = properties.require_fog_read;
		result.srgb_to_linear = properties.require_srgb_to_linear || properties.require_texture_ops;
		result.linear_to_srgb = properties.require_linear_to_srgb || properties.ROP_sRGB_packing;
		result.depth_conversion = properties.require_depth_conversion || properties.emulate_depth_compare;
		result.shadow_compare = properties.require_tex_shadow_ops || properties.emulate_shadow_compare;
		result.texture_expand = properties.require_texture_expand;
		result.output_rounding = properties.ROP_output_rounding;
		result.alpha_test = properties.ROP_alpha_test;
		result.polygon_stipple = properties.ROP_polygon_stipple_test;
		result.fp16 = properties.supports_native_fp16 && !properties.fp32_outputs;
		return result;
	}

	std::string generate_msl_prelude(msl_shader_stage stage, const msl_helper_requirements& requirements)
	{
		if (requirements.domain != glsl::glsl_invalid_program &&
			static_cast<u8>(requirements.domain) != static_cast<u8>(stage))
		{
			fmt::throw_exception("MSL helper domain does not match the requested shader stage");
		}

		msl_source_builder builder;
		builder.line("#include <metal_stdlib>");
		builder.line("using namespace metal;");
		builder.line();
		append_msl_helpers(builder, requirements);
		return builder.take();
	}

	void append_msl_helpers(msl_source_builder& builder, const msl_helper_requirements& requirements)
	{
		if (requirements.low_precision_comparisons)
		{
			builder.line("template <typename T> inline T rsx_cmp_fixup(T value) { return sign(value) * T(16.0f) + value; }");
		}
		else
		{
			builder.line("template <typename T> inline T rsx_cmp_fixup(T value) { return value; }");
		}
		builder.line("template <typename T> inline T rsx_saturate(T value) { return clamp(value, T(0.0f), T(1.0f)); }");

		if (requirements.lit)
		{
			builder.line("inline float4 rsx_lit(float4 value) { return float4(1.0f, max(value.x, 0.0f), value.x > 0.0f ? pow(max(value.y, 0.0f), clamp(value.w, -128.0f, 128.0f)) : 0.0f, 1.0f); }");
		}
		if (requirements.fog)
		{
			builder.line("inline float rsx_fog_factor(float coordinate, uint mode, float scale, float bias)");
			builder.open_scope({});
			builder.line("float value = (mode >= 3u) ? abs(coordinate) : coordinate;");
			builder.line("switch (mode % 3u) { case 0u: return saturate(value * scale + bias); case 1u: return saturate(exp2(-value * scale)); default: return saturate(exp2(-value * value * scale)); }");
			builder.close_scope();
		}
		if (requirements.srgb_to_linear)
		{
			builder.line("inline float3 rsx_srgb_to_linear(float3 value)");
			builder.open_scope({});
			builder.line("return select(pow((value + 0.055f) / 1.055f, float3(2.4f)), value / 12.92f, value <= 0.04045f);");
			builder.close_scope();
		}
		if (requirements.linear_to_srgb)
		{
			builder.line("inline float3 rsx_linear_to_srgb(float3 value)");
			builder.open_scope({});
			builder.line("value = max(value, float3(0.0f));");
			builder.line("return select(1.055f * pow(value, float3(1.0f / 2.4f)) - 0.055f, value * 12.92f, value <= 0.0031308f);");
			builder.close_scope();
		}
		if (requirements.depth_conversion)
		{
			builder.line("inline uint rsx_pack_z24(float depth) { return uint(round(saturate(depth) * 16777215.0f)); }");
			builder.line("inline float rsx_unpack_z24(uint depth) { return float(depth & 0x00ffffffu) / 16777215.0f; }");
			builder.line("inline float4 rsx_z24_to_rgba8(uint depth) { uint value = depth & 0x00ffffffu; return float4(float4((value >> 16u) & 255u, (value >> 8u) & 255u, value & 255u, 255u) / 255.0f); }");
		}
		if (requirements.shadow_compare || requirements.alpha_test)
		{
			builder.line("inline bool rsx_shadow_compare(float reference, float sample_value, uint operation)");
			builder.open_scope({});
			builder.line("switch (operation & 7u) { case 0u: return false; case 1u: return reference < sample_value; case 2u: return reference == sample_value; case 3u: return reference <= sample_value; case 4u: return reference > sample_value; case 5u: return reference != sample_value; case 6u: return reference >= sample_value; default: return true; }");
			builder.close_scope();
		}
		if (requirements.texture_expand)
		{
			builder.line("inline float4 rsx_expand_texture(float4 value, uint mask) { return select(value, value * 2.0f - 1.0f, bool4(mask & 1u, mask & 2u, mask & 4u, mask & 8u)); }");
		}
		if (requirements.output_rounding)
		{
			builder.line("inline float4 rsx_round_output(float4 value) { return rint(saturate(value) * 255.0f) / 255.0f; }");
		}
		if (requirements.alpha_test)
		{
			builder.line("inline bool rsx_alpha_test(float value, float reference, uint operation) { return rsx_shadow_compare(value, reference, operation); }");
		}
		if (requirements.polygon_stipple)
		{
			builder.line("inline bool rsx_polygon_stipple(uint2 position, uint row_pattern) { return ((row_pattern >> (position.x & 31u)) & 1u) != 0u; }");
		}
		if (requirements.fp16)
		{
			builder.line("inline half4 rsx_round_half(half4 value) { return half4(rint(float4(value) * 1024.0f) / 1024.0f); }");
		}
		builder.line();
	}

	std::string join_msl_arguments(std::span<const std::string> arguments)
	{
		std::string result;
		for (usz index = 0; index < arguments.size(); index++)
		{
			if (index)
			{
				result += ", ";
			}
			result += arguments[index];
		}
		return result;
	}
}
