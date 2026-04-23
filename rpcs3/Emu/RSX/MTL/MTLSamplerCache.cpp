#include "stdafx.h"
#include "MTLSamplerCache.h"

#include "MTLDevice.h"

namespace
{
	rsx::metal::sampler_address_mode make_sampler_address_mode(rsx::texture_wrap_mode mode)
	{
		rsx_log.trace("make_sampler_address_mode(mode=%u)", static_cast<u32>(mode));

		switch (mode)
		{
		case rsx::texture_wrap_mode::wrap:
			return rsx::metal::sampler_address_mode::repeat;
		case rsx::texture_wrap_mode::mirror:
			return rsx::metal::sampler_address_mode::mirror_repeat;
		case rsx::texture_wrap_mode::clamp_to_edge:
		case rsx::texture_wrap_mode::clamp:
			return rsx::metal::sampler_address_mode::clamp_to_edge;
		case rsx::texture_wrap_mode::border:
			return rsx::metal::sampler_address_mode::clamp_to_border_color;
		case rsx::texture_wrap_mode::mirror_once_clamp_to_edge:
		case rsx::texture_wrap_mode::mirror_once_clamp:
			return rsx::metal::sampler_address_mode::mirror_clamp_to_edge;
		case rsx::texture_wrap_mode::mirror_once_border:
			rsx_log.todo("Metal mirror-once-border sampler mode is not implemented");
			break;
		}

		fmt::throw_exception("Unsupported Metal texture wrap mode: %u", static_cast<u32>(mode));
	}

	rsx::metal::sampler_filter make_min_filter(rsx::texture_minify_filter filter)
	{
		rsx_log.trace("make_min_filter(filter=%u)", static_cast<u32>(filter));

		switch (filter)
		{
		case rsx::texture_minify_filter::nearest:
		case rsx::texture_minify_filter::nearest_nearest:
		case rsx::texture_minify_filter::nearest_linear:
			return rsx::metal::sampler_filter::nearest;
		case rsx::texture_minify_filter::linear:
		case rsx::texture_minify_filter::linear_nearest:
		case rsx::texture_minify_filter::linear_linear:
		case rsx::texture_minify_filter::convolution_min:
			return rsx::metal::sampler_filter::linear;
		}

		fmt::throw_exception("Unsupported Metal minification filter: %u", static_cast<u32>(filter));
	}

	rsx::metal::sampler_mip_filter make_mip_filter(rsx::texture_minify_filter filter, b8 allow_mipmaps)
	{
		rsx_log.trace("make_mip_filter(filter=%u, allow_mipmaps=%u)",
			static_cast<u32>(filter),
			static_cast<u32>(allow_mipmaps));

		if (!allow_mipmaps)
		{
			return rsx::metal::sampler_mip_filter::not_mipmapped;
		}

		switch (filter)
		{
		case rsx::texture_minify_filter::nearest:
		case rsx::texture_minify_filter::linear:
			return rsx::metal::sampler_mip_filter::not_mipmapped;
		case rsx::texture_minify_filter::nearest_nearest:
		case rsx::texture_minify_filter::linear_nearest:
			return rsx::metal::sampler_mip_filter::nearest;
		case rsx::texture_minify_filter::nearest_linear:
		case rsx::texture_minify_filter::linear_linear:
		case rsx::texture_minify_filter::convolution_min:
			return rsx::metal::sampler_mip_filter::linear;
		}

		fmt::throw_exception("Unsupported Metal mip filter: %u", static_cast<u32>(filter));
	}

	rsx::metal::sampler_filter make_mag_filter(rsx::texture_magnify_filter filter)
	{
		rsx_log.trace("make_mag_filter(filter=%u)", static_cast<u32>(filter));

		switch (filter)
		{
		case rsx::texture_magnify_filter::nearest:
			return rsx::metal::sampler_filter::nearest;
		case rsx::texture_magnify_filter::linear:
		case rsx::texture_magnify_filter::convolution_mag:
			return rsx::metal::sampler_filter::linear;
		}

		fmt::throw_exception("Unsupported Metal magnification filter: %u", static_cast<u32>(filter));
	}

	u32 make_max_anisotropy(rsx::texture_max_anisotropy anisotropy)
	{
		rsx_log.trace("make_max_anisotropy(anisotropy=%u)", static_cast<u32>(anisotropy));

		switch (anisotropy)
		{
		case rsx::texture_max_anisotropy::x1:
			return 1;
		case rsx::texture_max_anisotropy::x2:
			return 2;
		case rsx::texture_max_anisotropy::x4:
			return 4;
		case rsx::texture_max_anisotropy::x6:
			return 6;
		case rsx::texture_max_anisotropy::x8:
			return 8;
		case rsx::texture_max_anisotropy::x10:
			return 10;
		case rsx::texture_max_anisotropy::x12:
			return 12;
		case rsx::texture_max_anisotropy::x16:
			return 16;
		}

		fmt::throw_exception("Unsupported Metal anisotropy level: %u", static_cast<u32>(anisotropy));
	}

	rsx::metal::sampler_border_color make_border_color(const color4f& color)
	{
		rsx_log.trace("make_border_color(r=%f, g=%f, b=%f, a=%f)", color.r, color.g, color.b, color.a);

		if (color.r == 0.f && color.g == 0.f && color.b == 0.f && color.a == 0.f)
		{
			return rsx::metal::sampler_border_color::transparent_black;
		}

		if (color.r == 0.f && color.g == 0.f && color.b == 0.f && color.a == 1.f)
		{
			return rsx::metal::sampler_border_color::opaque_black;
		}

		if (color.r == 1.f && color.g == 1.f && color.b == 1.f && color.a == 1.f)
		{
			return rsx::metal::sampler_border_color::opaque_white;
		}

		rsx_log.todo("Metal custom sampler border colors are not implemented");
		fmt::throw_exception("Metal backend custom sampler border colors are not implemented yet");
	}

	b8 uses_border_color(
		rsx::metal::sampler_address_mode s_address_mode,
		rsx::metal::sampler_address_mode t_address_mode,
		rsx::metal::sampler_address_mode r_address_mode)
	{
		return s_address_mode == rsx::metal::sampler_address_mode::clamp_to_border_color ||
			t_address_mode == rsx::metal::sampler_address_mode::clamp_to_border_color ||
			r_address_mode == rsx::metal::sampler_address_mode::clamp_to_border_color;
	}

	rsx::metal::sampler_desc make_fragment_sampler_desc(const rsx::fragment_texture& texture, b8 allow_mipmaps)
	{
		rsx_log.trace("make_fragment_sampler_desc(indexed_texture=0x%x, allow_mipmaps=%u)",
			texture.offset(),
			static_cast<u32>(allow_mipmaps));

		const rsx::metal::sampler_address_mode s_address_mode = make_sampler_address_mode(texture.wrap_s());
		const rsx::metal::sampler_address_mode t_address_mode = make_sampler_address_mode(texture.wrap_t());
		const rsx::metal::sampler_address_mode r_address_mode = make_sampler_address_mode(texture.wrap_r());
		const rsx::metal::sampler_border_color border_color = uses_border_color(s_address_mode, t_address_mode, r_address_mode)
			? make_border_color(texture.remapped_border_color(false))
			: rsx::metal::sampler_border_color::transparent_black;

		return
		{
			.label = "RPCS3 Metal fragment sampler",
			.min_filter = make_min_filter(texture.min_filter()),
			.mag_filter = make_mag_filter(texture.mag_filter()),
			.mip_filter = make_mip_filter(texture.min_filter(), allow_mipmaps),
			.s_address_mode = s_address_mode,
			.t_address_mode = t_address_mode,
			.r_address_mode = r_address_mode,
			.border_color = border_color,
			.compare_function = rsx::metal::sampler_compare_function::never,
			.reduction_mode = rsx::metal::sampler_reduction_mode::weighted_average,
			.max_anisotropy = make_max_anisotropy(texture.max_aniso()),
			.lod_min = allow_mipmaps ? texture.min_lod() : 0.f,
			.lod_max = allow_mipmaps ? texture.max_lod() : 0.f,
			.lod_bias = allow_mipmaps ? texture.bias() : 0.f,
			.normalized_coordinates = !(texture.format() & CELL_GCM_TEXTURE_UN),
			.support_argument_tables = true,
		};
	}

	b8 sampler_desc_equal(const rsx::metal::sampler_desc& lhs, const rsx::metal::sampler_desc& rhs)
	{
		return lhs.min_filter == rhs.min_filter &&
			lhs.mag_filter == rhs.mag_filter &&
			lhs.mip_filter == rhs.mip_filter &&
			lhs.s_address_mode == rhs.s_address_mode &&
			lhs.t_address_mode == rhs.t_address_mode &&
			lhs.r_address_mode == rhs.r_address_mode &&
			lhs.border_color == rhs.border_color &&
			lhs.compare_function == rhs.compare_function &&
			lhs.reduction_mode == rhs.reduction_mode &&
			lhs.max_anisotropy == rhs.max_anisotropy &&
			lhs.lod_min == rhs.lod_min &&
			lhs.lod_max == rhs.lod_max &&
			lhs.lod_bias == rhs.lod_bias &&
			lhs.normalized_coordinates == rhs.normalized_coordinates &&
			lhs.support_argument_tables == rhs.support_argument_tables;
	}
}

namespace rsx::metal
{
	sampler_cache::sampler_cache(device& dev)
		: m_device(dev)
	{
		rsx_log.notice("rsx::metal::sampler_cache::sampler_cache(device=*0x%x)", dev.handle());
	}

	sampler_cache::~sampler_cache()
	{
		rsx_log.notice("rsx::metal::sampler_cache::~sampler_cache(retained=%u)", stats().retained_sampler_count);
	}

	sampler& sampler_cache::get_fragment_sampler(const rsx::fragment_texture& texture, b8 allow_mipmaps)
	{
		rsx_log.trace("rsx::metal::sampler_cache::get_fragment_sampler(offset=0x%x, allow_mipmaps=%u)",
			texture.offset(),
			static_cast<u32>(allow_mipmaps));

		const sampler_desc desc = make_fragment_sampler_desc(texture, allow_mipmaps);
		for (sampler_record& record : m_samplers)
		{
			if (sampler_desc_equal(record.desc, desc))
			{
				m_sampler_cache_hit_count++;
				ensure(record.state);
				return *record.state;
			}
		}

		sampler_record record =
		{
			.desc = desc,
			.state = std::make_unique<sampler>(m_device, desc),
		};

		m_created_sampler_count++;
		m_samplers.emplace_back(std::move(record));
		return *m_samplers.back().state;
	}

	sampler_cache_stats sampler_cache::stats() const
	{
		rsx_log.trace("rsx::metal::sampler_cache::stats()");

		if (m_samplers.size() > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal sampler cache retained count overflow");
		}

		return
		{
			.retained_sampler_count = static_cast<u32>(m_samplers.size()),
			.created_sampler_count = m_created_sampler_count,
			.sampler_cache_hit_count = m_sampler_cache_hit_count,
		};
	}

	void sampler_cache::report() const
	{
		rsx_log.notice("rsx::metal::sampler_cache::report()");

		const sampler_cache_stats sampler_stats = stats();
		rsx_log.notice("Metal sampler cache: retained=%u, created=%u, hits=%u",
			sampler_stats.retained_sampler_count,
			sampler_stats.created_sampler_count,
			sampler_stats.sampler_cache_hit_count);
	}
}
