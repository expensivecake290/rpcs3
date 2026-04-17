#include "stdafx.h"
#include "MTLSampler.h"

#include "MTLDevice.h"

#import <Metal/Metal.h>

namespace
{
	NSString* make_ns_string(const std::string& value)
	{
		return [NSString stringWithUTF8String:value.c_str()];
	}

	MTLSamplerMinMagFilter make_filter(rsx::metal::sampler_filter filter)
	{
		rsx_log.trace("make_filter(filter=%u)", static_cast<u32>(filter));

		switch (filter)
		{
		case rsx::metal::sampler_filter::nearest:
			return MTLSamplerMinMagFilterNearest;
		case rsx::metal::sampler_filter::linear:
			return MTLSamplerMinMagFilterLinear;
		}

		fmt::throw_exception("Unknown Metal sampler filter %u", static_cast<u32>(filter));
	}

	MTLSamplerMipFilter make_mip_filter(rsx::metal::sampler_mip_filter filter)
	{
		rsx_log.trace("make_mip_filter(filter=%u)", static_cast<u32>(filter));

		switch (filter)
		{
		case rsx::metal::sampler_mip_filter::not_mipmapped:
			return MTLSamplerMipFilterNotMipmapped;
		case rsx::metal::sampler_mip_filter::nearest:
			return MTLSamplerMipFilterNearest;
		case rsx::metal::sampler_mip_filter::linear:
			return MTLSamplerMipFilterLinear;
		}

		fmt::throw_exception("Unknown Metal sampler mip filter %u", static_cast<u32>(filter));
	}

	MTLSamplerAddressMode make_address_mode(rsx::metal::sampler_address_mode mode)
	{
		rsx_log.trace("make_address_mode(mode=%u)", static_cast<u32>(mode));

		switch (mode)
		{
		case rsx::metal::sampler_address_mode::clamp_to_edge:
			return MTLSamplerAddressModeClampToEdge;
		case rsx::metal::sampler_address_mode::mirror_clamp_to_edge:
			return MTLSamplerAddressModeMirrorClampToEdge;
		case rsx::metal::sampler_address_mode::repeat:
			return MTLSamplerAddressModeRepeat;
		case rsx::metal::sampler_address_mode::mirror_repeat:
			return MTLSamplerAddressModeMirrorRepeat;
		case rsx::metal::sampler_address_mode::clamp_to_zero:
			return MTLSamplerAddressModeClampToZero;
		case rsx::metal::sampler_address_mode::clamp_to_border_color:
			return MTLSamplerAddressModeClampToBorderColor;
		}

		fmt::throw_exception("Unknown Metal sampler address mode %u", static_cast<u32>(mode));
	}

	MTLSamplerBorderColor make_border_color(rsx::metal::sampler_border_color color)
	{
		rsx_log.trace("make_border_color(color=%u)", static_cast<u32>(color));

		switch (color)
		{
		case rsx::metal::sampler_border_color::transparent_black:
			return MTLSamplerBorderColorTransparentBlack;
		case rsx::metal::sampler_border_color::opaque_black:
			return MTLSamplerBorderColorOpaqueBlack;
		case rsx::metal::sampler_border_color::opaque_white:
			return MTLSamplerBorderColorOpaqueWhite;
		}

		fmt::throw_exception("Unknown Metal sampler border color %u", static_cast<u32>(color));
	}

	MTLCompareFunction make_compare_function(rsx::metal::sampler_compare_function function)
	{
		rsx_log.trace("make_compare_function(function=%u)", static_cast<u32>(function));

		switch (function)
		{
		case rsx::metal::sampler_compare_function::never:
			return MTLCompareFunctionNever;
		case rsx::metal::sampler_compare_function::less:
			return MTLCompareFunctionLess;
		case rsx::metal::sampler_compare_function::equal:
			return MTLCompareFunctionEqual;
		case rsx::metal::sampler_compare_function::less_equal:
			return MTLCompareFunctionLessEqual;
		case rsx::metal::sampler_compare_function::greater:
			return MTLCompareFunctionGreater;
		case rsx::metal::sampler_compare_function::not_equal:
			return MTLCompareFunctionNotEqual;
		case rsx::metal::sampler_compare_function::greater_equal:
			return MTLCompareFunctionGreaterEqual;
		case rsx::metal::sampler_compare_function::always:
			return MTLCompareFunctionAlways;
		}

		fmt::throw_exception("Unknown Metal sampler compare function %u", static_cast<u32>(function));
	}

	MTLSamplerReductionMode make_reduction_mode(rsx::metal::sampler_reduction_mode mode)
	{
		rsx_log.trace("make_reduction_mode(mode=%u)", static_cast<u32>(mode));

		switch (mode)
		{
		case rsx::metal::sampler_reduction_mode::weighted_average:
			return MTLSamplerReductionModeWeightedAverage;
		case rsx::metal::sampler_reduction_mode::minimum:
			return MTLSamplerReductionModeMinimum;
		case rsx::metal::sampler_reduction_mode::maximum:
			return MTLSamplerReductionModeMaximum;
		}

		fmt::throw_exception("Unknown Metal sampler reduction mode %u", static_cast<u32>(mode));
	}
}

namespace rsx::metal
{
	struct sampler::sampler_impl
	{
		id<MTLSamplerState> m_sampler = nil;
	};

	sampler::sampler(device& dev, const sampler_desc& desc)
		: m_impl(std::make_unique<sampler_impl>())
	{
		rsx_log.notice("rsx::metal::sampler::sampler(device=*0x%x, min_filter=%u, mag_filter=%u, mip_filter=%u)",
			dev.handle(), static_cast<u32>(desc.min_filter), static_cast<u32>(desc.mag_filter), static_cast<u32>(desc.mip_filter));

		if (!dev.handle())
		{
			fmt::throw_exception("Metal sampler requires a valid device");
		}

		if (!desc.max_anisotropy)
		{
			fmt::throw_exception("Metal sampler requires max_anisotropy to be at least 1");
		}

		if (@available(macOS 26.0, *))
		{
			MTLSamplerDescriptor* sampler_desc = [MTLSamplerDescriptor new];
			sampler_desc.minFilter = make_filter(desc.min_filter);
			sampler_desc.magFilter = make_filter(desc.mag_filter);
			sampler_desc.mipFilter = make_mip_filter(desc.mip_filter);
			sampler_desc.sAddressMode = make_address_mode(desc.s_address_mode);
			sampler_desc.tAddressMode = make_address_mode(desc.t_address_mode);
			sampler_desc.rAddressMode = make_address_mode(desc.r_address_mode);
			sampler_desc.borderColor = make_border_color(desc.border_color);
			sampler_desc.compareFunction = make_compare_function(desc.compare_function);
			sampler_desc.reductionMode = make_reduction_mode(desc.reduction_mode);
			sampler_desc.maxAnisotropy = desc.max_anisotropy;
			sampler_desc.lodMinClamp = desc.lod_min;
			sampler_desc.lodMaxClamp = desc.lod_max;
			sampler_desc.lodBias = desc.lod_bias;
			sampler_desc.normalizedCoordinates = desc.normalized_coordinates;
			sampler_desc.supportArgumentBuffers = desc.support_argument_tables;

			if (!desc.label.empty())
			{
				sampler_desc.label = make_ns_string(desc.label);
			}

			id<MTLDevice> metal_device = (__bridge id<MTLDevice>)dev.handle();
			m_impl->m_sampler = [metal_device newSamplerStateWithDescriptor:sampler_desc];
			if (!m_impl->m_sampler)
			{
				fmt::throw_exception("Metal sampler creation failed");
			}
		}
		else
		{
			fmt::throw_exception("Metal sampler creation requires macOS 26.0 or newer");
		}
	}

	sampler::~sampler()
	{
		rsx_log.notice("rsx::metal::sampler::~sampler(sampler=*0x%x)", handle());
	}

	void* sampler::handle() const
	{
		rsx_log.trace("rsx::metal::sampler::handle()");
		return (__bridge void*)m_impl->m_sampler;
	}

	u64 sampler::resource_id() const
	{
		rsx_log.trace("rsx::metal::sampler::resource_id()");
		return m_impl->m_sampler.gpuResourceID._impl;
	}
}
