#pragma once

#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
	class device;

	enum class sampler_filter : u8
	{
		nearest,
		linear,
	};

	enum class sampler_mip_filter : u8
	{
		not_mipmapped,
		nearest,
		linear,
	};

	enum class sampler_address_mode : u8
	{
		clamp_to_edge,
		mirror_clamp_to_edge,
		repeat,
		mirror_repeat,
		clamp_to_zero,
		clamp_to_border_color,
	};

	enum class sampler_border_color : u8
	{
		transparent_black,
		opaque_black,
		opaque_white,
	};

	enum class sampler_compare_function : u8
	{
		never,
		less,
		equal,
		less_equal,
		greater,
		not_equal,
		greater_equal,
		always,
	};

	enum class sampler_reduction_mode : u8
	{
		weighted_average,
		minimum,
		maximum,
	};

	struct sampler_desc
	{
		std::string label;
		sampler_filter min_filter = sampler_filter::nearest;
		sampler_filter mag_filter = sampler_filter::nearest;
		sampler_mip_filter mip_filter = sampler_mip_filter::not_mipmapped;
		sampler_address_mode s_address_mode = sampler_address_mode::clamp_to_edge;
		sampler_address_mode t_address_mode = sampler_address_mode::clamp_to_edge;
		sampler_address_mode r_address_mode = sampler_address_mode::clamp_to_edge;
		sampler_border_color border_color = sampler_border_color::transparent_black;
		sampler_compare_function compare_function = sampler_compare_function::never;
		sampler_reduction_mode reduction_mode = sampler_reduction_mode::weighted_average;
		u32 max_anisotropy = 1;
		f32 lod_min = 0.f;
		f32 lod_max = 1000.f;
		f32 lod_bias = 0.f;
		b8 normalized_coordinates = true;
		b8 support_argument_tables = true;
	};

	class sampler
	{
	public:
		sampler(device& dev, const sampler_desc& desc);
		~sampler();

		sampler(const sampler&) = delete;
		sampler& operator=(const sampler&) = delete;

		void* handle() const;
		u64 resource_id() const;

	private:
		struct sampler_impl;
		std::unique_ptr<sampler_impl> m_impl;
	};
}
