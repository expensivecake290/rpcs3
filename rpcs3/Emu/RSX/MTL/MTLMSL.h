#pragma once

#include "Emu/RSX/Program/ShaderParam.h"

#include <sstream>
#include <string>

namespace rsx::metal::msl
{
	std::string get_float_type_name(usz element_count);
	std::string get_half_type_name(usz element_count);
	std::string get_int_type_name(usz element_count);
	std::string get_function(FUNCTION function);
	std::string compare_function(COMPARE function, const std::string& op0, const std::string& op1, b8 scalar);
	void insert_header(std::stringstream& stream);
	void insert_legacy_helpers(std::stringstream& stream);
}
