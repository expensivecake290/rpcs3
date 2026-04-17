#include "stdafx.h"
#include "MTLDevice.h"

#include "MTLResidency.h"

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
		std::unique_ptr<residency_manager> m_residency;
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
				m_impl->m_device = metal_device;
				m_impl->m_residency = std::make_unique<residency_manager>((__bridge void*)metal_device, "RPCS3 Metal backend residency set", 64);
				m_impl->m_residency->request_residency();
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
	}

	void* device::handle() const
	{
		rsx_log.trace("rsx::metal::device::handle()");
		return (__bridge void*)m_impl->m_device;
	}

	void* device::residency_set_handle() const
	{
		rsx_log.trace("rsx::metal::device::residency_set_handle()");
		return m_impl->m_residency->handle();
	}

	void device::add_resident_allocation(void* allocation_handle)
	{
		rsx_log.trace("rsx::metal::device::add_resident_allocation(allocation_handle=*0x%x)", allocation_handle);
		m_impl->m_residency->add_allocation(allocation_handle);
	}

	void device::remove_resident_allocation(void* allocation_handle)
	{
		rsx_log.trace("rsx::metal::device::remove_resident_allocation(allocation_handle=*0x%x)", allocation_handle);
		m_impl->m_residency->remove_allocation(allocation_handle);
	}

	void device::commit_residency()
	{
		rsx_log.trace("rsx::metal::device::commit_residency()");
		m_impl->m_residency->commit();
	}

	u64 device::residency_allocated_size() const
	{
		rsx_log.trace("rsx::metal::device::residency_allocated_size()");
		return m_impl->m_residency->allocated_size();
	}

	u32 device::residency_allocation_count() const
	{
		rsx_log.trace("rsx::metal::device::residency_allocation_count()");
		return m_impl->m_residency->allocation_count();
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
		rsx_log.notice("Backend residency set: allocations=%u, allocated_size=0x%x",
			residency_allocation_count(), residency_allocated_size());
		rsx_log.notice("Frames in flight: %u", m_impl->m_caps.frames_in_flight);
		rsx_log.notice("MTL4ArgumentTable limits: buffers=%u, textures=%u, samplers=%u",
			m_impl->m_caps.max_argument_table_buffers,
			m_impl->m_caps.max_argument_table_textures,
			m_impl->m_caps.max_argument_table_samplers);
		rsx_log.warning("Metal RSX draw submission, argument tables, shader translation, pipeline creation, and MetalFX execution are not enabled yet");
	}
}
