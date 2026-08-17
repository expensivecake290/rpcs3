#pragma once

#include <string>
#include <string_view>

#include "util/types.hpp"

namespace mtl
{
	// Metal GPU families describe feature sets rather than marketing products.
	// Keep the ordering intact so feature/workaround checks can use ranges.
	enum class chip_class : u8
	{
		unknown,

		apple_generic,
		apple_family7,
		apple_family8,
		apple_family9,
		apple_family10,
		apple_family_future,
		_apple_max,

		amd_generic,
		_amd_max,

		intel_generic,
		_intel_max,
	};

	enum class driver_vendor : u8
	{
		unknown,
		apple,
		amd,
		intel,
	};

	struct device_identity
	{
		std::string name;
		u32 apple_gpu_family = 0;
		bool unified_memory = false;
		bool low_power = false;
		bool removable = false;
	};

	struct chip_capabilities
	{
		driver_vendor vendor = driver_vendor::unknown;
		chip_class chip = chip_class::unknown;
		bool tile_based = false;
		bool unified_memory = false;
		bool low_power = false;
		bool removable = false;
	};

	[[nodiscard]] chip_capabilities classify_device(const device_identity& identity);
	[[nodiscard]] const char* get_chip_class_name(chip_class chip);
	[[nodiscard]] const char* get_driver_vendor_name(driver_vendor vendor);

	[[nodiscard]] constexpr bool is_apple(chip_class chip)
	{
		return chip >= chip_class::apple_generic && chip < chip_class::_apple_max;
	}

	[[nodiscard]] constexpr bool is_amd(chip_class chip)
	{
		return chip >= chip_class::amd_generic && chip < chip_class::_amd_max;
	}

	[[nodiscard]] constexpr bool is_intel(chip_class chip)
	{
		return chip >= chip_class::intel_generic && chip < chip_class::_intel_max;
	}

	[[nodiscard]] constexpr bool is_apple(driver_vendor vendor)
	{
		return vendor == driver_vendor::apple;
	}

	[[nodiscard]] constexpr bool is_amd(driver_vendor vendor)
	{
		return vendor == driver_vendor::amd;
	}

	[[nodiscard]] constexpr bool is_intel(driver_vendor vendor)
	{
		return vendor == driver_vendor::intel;
	}
}
