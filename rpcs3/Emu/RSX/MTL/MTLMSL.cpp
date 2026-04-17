#include "stdafx.h"
#include "MTLMSL.h"

namespace rsx::metal::msl
{
	std::string get_float_type_name(usz element_count)
	{
		rsx_log.trace("rsx::metal::msl::get_float_type_name(element_count=%u)", static_cast<u32>(element_count));

		switch (element_count)
		{
		case 1: return "float";
		case 2: return "float2";
		case 3: return "float3";
		case 4: return "float4";
		default:
			fmt::throw_exception("Unsupported MSL float vector size: %u", static_cast<u32>(element_count));
		}
	}

	std::string get_half_type_name(usz element_count)
	{
		rsx_log.trace("rsx::metal::msl::get_half_type_name(element_count=%u)", static_cast<u32>(element_count));

		switch (element_count)
		{
		case 1: return "half";
		case 2: return "half2";
		case 3: return "half3";
		case 4: return "half4";
		default:
			fmt::throw_exception("Unsupported MSL half vector size: %u", static_cast<u32>(element_count));
		}
	}

	std::string get_int_type_name(usz element_count)
	{
		rsx_log.trace("rsx::metal::msl::get_int_type_name(element_count=%u)", static_cast<u32>(element_count));

		switch (element_count)
		{
		case 1: return "int";
		case 2: return "int2";
		case 3: return "int3";
		case 4: return "int4";
		default:
			fmt::throw_exception("Unsupported MSL integer vector size: %u", static_cast<u32>(element_count));
		}
	}

	std::string get_function(FUNCTION function)
	{
		rsx_log.trace("rsx::metal::msl::get_function(function=%u)", static_cast<u32>(function));

		switch (function)
		{
		case FUNCTION::DP2:
			return "$Ty(dot($0.xy, $1.xy))";
		case FUNCTION::DP2A:
			return "$Ty(dot($0.xy, $1.xy) + $2.x)";
		case FUNCTION::DP3:
			return "$Ty(dot($0.xyz, $1.xyz))";
		case FUNCTION::DP4:
			return "$Ty(dot($0, $1))";
		case FUNCTION::DPH:
			return "$Ty(dot(float4($0.xyz, 1.0f), $1))";
		case FUNCTION::SFL:
			return "$Ty(0.0f)";
		case FUNCTION::STR:
			return "$Ty(1.0f)";
		case FUNCTION::FRACT:
			return "fract($0)";
		case FUNCTION::REFL:
			return "reflect($0, $1)";
		default:
			fmt::throw_exception("Metal shader recompiler has no implemented MSL mapping for function %u", static_cast<u32>(function));
		}
	}

	std::string compare_function(COMPARE function, const std::string& op0, const std::string& op1, b8 scalar)
	{
		rsx_log.trace("rsx::metal::msl::compare_function(function=%u, scalar=%d)", static_cast<u32>(function), scalar);

		const char* op = nullptr;
		switch (function)
		{
		case COMPARE::SEQ: op = "=="; break;
		case COMPARE::SGE: op = ">="; break;
		case COMPARE::SGT: op = ">"; break;
		case COMPARE::SLE: op = "<="; break;
		case COMPARE::SLT: op = "<"; break;
		case COMPARE::SNE: op = "!="; break;
		default:
			fmt::throw_exception("Unknown Metal comparison function: %u", static_cast<u32>(function));
		}

		return scalar
			? fmt::format("(({}) {} ({}))", op0, op, op1)
			: fmt::format("(({}) {} ({}))", op0, op, op1);
	}

	void insert_header(std::stringstream& stream)
	{
		rsx_log.trace("rsx::metal::msl::insert_header()");

		stream <<
			"#include <metal_stdlib>\n"
			"using namespace metal;\n\n";
	}

	void insert_legacy_helpers(std::stringstream& stream)
	{
		rsx_log.trace("rsx::metal::msl::insert_legacy_helpers()");

		stream <<
			"float4 _select(float4 left, float4 right, bool4 condition)\n"
			"{\n"
			"	return select(left, right, condition);\n"
			"}\n\n"
			"float4 _builtin_div(float4 left, float right)\n"
			"{\n"
			"	return left / right;\n"
			"}\n\n";
	}
}
