#include "stdafx.h"
#include "MTLDevice.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace
{
	std::string get_ns_string(NSString* value)
	{
		if (!value)
		{
			return {};
		}

		const char* text = [value UTF8String];
		return text ? std::string(text) : std::string();
	}

	std::string get_error_text(NSError* error)
	{
		if (!error)
		{
			return "unknown error";
		}

		return get_ns_string([error localizedDescription]);
	}

	b8 supports_metal4(id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		if (@available(macOS 26.0, *))
		{
			return [device supportsFamily:MTLGPUFamilyMetal4];
		}

		return false;
	}
}

namespace rsx::metal
{
	struct device::device_impl
	{
		id<MTLDevice> m_device = nil;
		id<MTLResidencySet> m_residency_set = nil;
		device_caps m_caps{};
	};

	b8 is_metal_supported()
	{
		rsx_log.trace("rsx::metal::is_metal_supported()");

		@autoreleasepool
		{
			id<MTLDevice> device = MTLCreateSystemDefaultDevice();
			return supports_metal4(device);
		}
	}

	device::device()
		: m_impl(std::make_unique<device_impl>())
	{
		rsx_log.notice("rsx::metal::device::device()");

		@autoreleasepool
		{
			id<MTLDevice> metal_device = MTLCreateSystemDefaultDevice();

			if (!metal_device)
			{
				fmt::throw_exception("Metal backend requires a Metal device, but MTLCreateSystemDefaultDevice returned nil");
			}

			if (!supports_metal4(metal_device))
			{
				fmt::throw_exception("Metal backend requires Metal 4 support");
			}

			if (@available(macOS 26.0, *))
			{
				MTLResidencySetDescriptor* residency_desc = [MTLResidencySetDescriptor new];
				residency_desc.label = @"RPCS3 Metal residency set";
				residency_desc.initialCapacity = 64;

				NSError* residency_error = nil;
				id<MTLResidencySet> residency_set = [metal_device newResidencySetWithDescriptor:residency_desc error:&residency_error];

				if (!residency_set)
				{
					fmt::throw_exception("Metal backend failed to create residency set: %s", get_error_text(residency_error));
				}

				[residency_set requestResidency];

				m_impl->m_device = metal_device;
				m_impl->m_residency_set = residency_set;
			}
			else
			{
				fmt::throw_exception("Metal backend requires macOS 26.0 or newer");
			}

			m_impl->m_caps.name = get_ns_string([metal_device name]);
			m_impl->m_caps.metal4_supported = true;
			m_impl->m_caps.metalfx_available = NSClassFromString(@"MTLFXSpatialScalerDescriptor") != nil;
			m_impl->m_caps.residency_sets_supported = true;
			m_impl->m_caps.frames_in_flight = 3;
			m_impl->m_caps.max_argument_table_buffers = 31;
			m_impl->m_caps.max_argument_table_textures = 128;
			m_impl->m_caps.max_argument_table_samplers = 16;
		}
	}

	device::~device()
	{
		rsx_log.notice("rsx::metal::device::~device()");

		if (m_impl && m_impl->m_residency_set)
		{
			[m_impl->m_residency_set endResidency];
		}
	}

	void* device::handle() const
	{
		rsx_log.trace("rsx::metal::device::handle()");
		return (__bridge void*)m_impl->m_device;
	}

	void* device::residency_set_handle() const
	{
		rsx_log.trace("rsx::metal::device::residency_set_handle()");
		return (__bridge void*)m_impl->m_residency_set;
	}

	const device_caps& device::caps() const
	{
		rsx_log.trace("rsx::metal::device::caps()");
		return m_impl->m_caps;
	}

	void device::report_capabilities() const
	{
		rsx_log.notice("Metal device: %s", m_impl->m_caps.name);
		rsx_log.notice("Metal 4 support: %s", m_impl->m_caps.metal4_supported ? "yes" : "no");
		rsx_log.notice("MetalFX framework availability: %s", m_impl->m_caps.metalfx_available ? "yes" : "no");
		rsx_log.notice("Residency sets: %s", m_impl->m_caps.residency_sets_supported ? "yes" : "no");
		rsx_log.notice("Frames in flight: %u", m_impl->m_caps.frames_in_flight);
		rsx_log.notice("MTL4ArgumentTable limits: buffers=%u, textures=%u, samplers=%u",
			m_impl->m_caps.max_argument_table_buffers,
			m_impl->m_caps.max_argument_table_textures,
			m_impl->m_caps.max_argument_table_samplers);
		rsx_log.warning("Metal shader cache, pipeline cache, binary archives, and MetalFX execution are not enabled in Phase 1");
	}
}
