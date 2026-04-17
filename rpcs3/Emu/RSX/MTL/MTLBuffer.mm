#include "stdafx.h"
#include "MTLBuffer.h"

#include "MTLDevice.h"

#import <Metal/Metal.h>

#include <cstring>

namespace
{
	NSString* make_ns_string(const std::string& value)
	{
		return [NSString stringWithUTF8String:value.c_str()];
	}

	MTLResourceOptions make_buffer_options(rsx::metal::buffer_storage storage, rsx::metal::buffer_cpu_cache_mode cpu_cache, rsx::metal::resource_hazard_tracking hazard_tracking)
	{
		rsx_log.trace("make_buffer_options(storage=%u, cpu_cache=%u, hazard_tracking=%u)",
			static_cast<u32>(storage), static_cast<u32>(cpu_cache), static_cast<u32>(hazard_tracking));

		MTLResourceOptions options = 0;

		switch (storage)
		{
		case rsx::metal::buffer_storage::shared:
			options |= MTLResourceStorageModeShared;
			break;
		case rsx::metal::buffer_storage::private_:
			options |= MTLResourceStorageModePrivate;
			break;
		}

		switch (cpu_cache)
		{
		case rsx::metal::buffer_cpu_cache_mode::default_:
			options |= MTLResourceCPUCacheModeDefaultCache;
			break;
		case rsx::metal::buffer_cpu_cache_mode::write_combined:
			options |= MTLResourceCPUCacheModeWriteCombined;
			break;
		}

		switch (hazard_tracking)
		{
		case rsx::metal::resource_hazard_tracking::tracked:
			options |= MTLResourceHazardTrackingModeTracked;
			break;
		case rsx::metal::resource_hazard_tracking::untracked:
			options |= MTLResourceHazardTrackingModeUntracked;
			break;
		}

		return options;
	}
}

namespace rsx::metal
{
	struct buffer::buffer_impl
	{
		device* m_device = nullptr;
		id<MTLBuffer> m_buffer = nil;
		buffer_desc m_desc{};
		b8 m_resident = false;
	};

	buffer::buffer(device& dev, buffer_desc desc)
		: m_impl(std::make_unique<buffer_impl>())
	{
		rsx_log.notice("rsx::metal::buffer::buffer(device=*0x%x, size=0x%llx, storage=%u, hazard_tracking=%u)",
			dev.handle(), desc.size, static_cast<u32>(desc.storage), static_cast<u32>(desc.hazard_tracking));

		if (!desc.size)
		{
			fmt::throw_exception("Metal buffer requires a non-zero size");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTLDevice> metal_device = (__bridge id<MTLDevice>)dev.handle();
			if (!metal_device)
			{
				fmt::throw_exception("Metal buffer requires a valid device");
			}

			const MTLResourceOptions options = make_buffer_options(desc.storage, desc.cpu_cache, desc.hazard_tracking);
			m_impl->m_buffer = [metal_device newBufferWithLength:static_cast<NSUInteger>(desc.size) options:options];
			if (!m_impl->m_buffer)
			{
				fmt::throw_exception("Metal buffer allocation failed for size=0x%llx", desc.size);
			}

			if (!desc.label.empty())
			{
				m_impl->m_buffer.label = make_ns_string(desc.label);
			}

			m_impl->m_device = &dev;
			m_impl->m_desc = std::move(desc);
			m_impl->m_device->add_resident_allocation((__bridge void*)m_impl->m_buffer);
			m_impl->m_device->commit_residency();
			m_impl->m_resident = true;
		}
		else
		{
			fmt::throw_exception("Metal buffer allocation requires macOS 26.0 or newer");
		}
	}

	buffer::~buffer()
	{
		rsx_log.notice("rsx::metal::buffer::~buffer(buffer=*0x%x)", handle());

		if (m_impl && m_impl->m_device && m_impl->m_buffer && m_impl->m_resident)
		{
			m_impl->m_device->remove_resident_allocation((__bridge void*)m_impl->m_buffer);
			m_impl->m_device->commit_residency();
			m_impl->m_resident = false;
		}
	}

	void* buffer::handle() const
	{
		rsx_log.trace("rsx::metal::buffer::handle()");
		return (__bridge void*)m_impl->m_buffer;
	}

	void* buffer::allocation_handle() const
	{
		rsx_log.trace("rsx::metal::buffer::allocation_handle()");
		return (__bridge void*)m_impl->m_buffer;
	}

	u64 buffer::length() const
	{
		rsx_log.trace("rsx::metal::buffer::length()");
		return static_cast<u64>(m_impl->m_buffer.length);
	}

	u64 buffer::allocated_size() const
	{
		rsx_log.trace("rsx::metal::buffer::allocated_size()");
		return static_cast<u64>(m_impl->m_buffer.allocatedSize);
	}

	u64 buffer::gpu_address() const
	{
		rsx_log.trace("rsx::metal::buffer::gpu_address()");
		return static_cast<u64>(m_impl->m_buffer.gpuAddress);
	}

	buffer_storage buffer::storage() const
	{
		rsx_log.trace("rsx::metal::buffer::storage()");
		return m_impl->m_desc.storage;
	}

	resource_hazard_tracking buffer::hazard_tracking() const
	{
		rsx_log.trace("rsx::metal::buffer::hazard_tracking()");
		return m_impl->m_desc.hazard_tracking;
	}

	void* buffer::map()
	{
		rsx_log.trace("rsx::metal::buffer::map(storage=%u)", static_cast<u32>(m_impl->m_desc.storage));

		if (m_impl->m_desc.storage != buffer_storage::shared)
		{
			fmt::throw_exception("Metal buffer CPU mapping is only valid for shared buffers");
		}

		void* ptr = [m_impl->m_buffer contents];
		if (!ptr)
		{
			fmt::throw_exception("Metal shared buffer returned a null CPU mapping");
		}

		return ptr;
	}

	void buffer::write(u64 offset, const void* data, u64 size)
	{
		rsx_log.trace("rsx::metal::buffer::write(offset=0x%llx, data=*0x%x, size=0x%llx)", offset, data, size);

		if (!size)
		{
			return;
		}

		if (!data)
		{
			fmt::throw_exception("Metal buffer write requires a valid source pointer");
		}

		if (offset > length() || size > length() - offset)
		{
			fmt::throw_exception("Metal buffer write out of bounds: offset=0x%llx, size=0x%llx, length=0x%llx", offset, size, length());
		}

		std::memcpy(static_cast<u8*>(map()) + offset, data, size);
	}
}
