#include "stdafx.h"
#include "chip_class.h"

#include <algorithm>
#include <cctype>

namespace mtl
{
	namespace
	{
		bool contains_case_insensitive(std::string_view text, std::string_view pattern)
		{
			const auto equal_character = [](char lhs, char rhs)
			{
				return std::tolower(static_cast<unsigned char>(lhs)) == std::tolower(static_cast<unsigned char>(rhs));
			};

			return std::search(text.begin(), text.end(), pattern.begin(), pattern.end(), equal_character) != text.end();
		}

		chip_class classify_apple_family(u32 family)
		{
			switch (family)
			{
			case 0: return chip_class::apple_generic;
			case 7: return chip_class::apple_family7;
			case 8: return chip_class::apple_family8;
			case 9: return chip_class::apple_family9;
			case 10: return chip_class::apple_family10;
			default:
				return family > 10 ? chip_class::apple_family_future : chip_class::apple_generic;
			}
		}
	}

	chip_capabilities classify_device(const device_identity& identity)
	{
		chip_capabilities result;
		result.unified_memory = identity.unified_memory;
		result.low_power = identity.low_power;
		result.removable = identity.removable;

		if (identity.apple_gpu_family != 0 || contains_case_insensitive(identity.name, "apple"))
		{
			result.vendor = driver_vendor::apple;
			result.chip = classify_apple_family(identity.apple_gpu_family);
			result.tile_based = true;
			return result;
		}

		if (contains_case_insensitive(identity.name, "amd") || contains_case_insensitive(identity.name, "radeon"))
		{
			result.vendor = driver_vendor::amd;
			result.chip = chip_class::amd_generic;
			return result;
		}

		if (contains_case_insensitive(identity.name, "intel"))
		{
			result.vendor = driver_vendor::intel;
			result.chip = chip_class::intel_generic;
			return result;
		}

		return result;
	}

	const char* get_chip_class_name(chip_class chip)
	{
		switch (chip)
		{
		case chip_class::unknown: return "Unknown";
		case chip_class::apple_generic: return "Apple GPU";
		case chip_class::apple_family7: return "Apple GPU family 7";
		case chip_class::apple_family8: return "Apple GPU family 8";
		case chip_class::apple_family9: return "Apple GPU family 9";
		case chip_class::apple_family10: return "Apple GPU family 10";
		case chip_class::apple_family_future: return "Future Apple GPU family";
		case chip_class::amd_generic: return "AMD Metal GPU";
		case chip_class::intel_generic: return "Intel Metal GPU";
		case chip_class::_apple_max:
		case chip_class::_amd_max:
		case chip_class::_intel_max:
			break;
		}

		fmt::throw_exception("Invalid Metal chip class %u", static_cast<u8>(chip));
	}

	const char* get_driver_vendor_name(driver_vendor vendor)
	{
		switch (vendor)
		{
		case driver_vendor::unknown: return "Unknown";
		case driver_vendor::apple: return "Apple";
		case driver_vendor::amd: return "AMD";
		case driver_vendor::intel: return "Intel";
		}

		fmt::throw_exception("Invalid Metal driver vendor %u", static_cast<u8>(vendor));
	}
}
