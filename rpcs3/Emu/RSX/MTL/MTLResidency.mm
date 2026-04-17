#include "stdafx.h"
#include "MTLResidency.h"

#import <Metal/Metal.h>

#include <mutex>

namespace
{
	NSString* make_ns_string(std::string_view text)
	{
		return [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
	}

	std::string get_ns_string(NSString* value)
	{
		if (!value)
		{
			return {};
		}

		const char* text = [value UTF8String];
		return text ? std::string(text) : std::string();
	}
}

namespace rsx::metal
{
	struct residency_manager::residency_manager_impl
	{
		id<MTLResidencySet> m_residency_set = nil;
		std::mutex m_mutex;
		b8 m_resident = false;
	};

	residency_manager::residency_manager(void* device_handle, std::string_view label, u32 initial_capacity)
		: m_impl(std::make_unique<residency_manager_impl>())
	{
		const std::string label_text(label);
		rsx_log.notice("rsx::metal::residency_manager::residency_manager(device_handle=*0x%x, label=%s, initial_capacity=%u)",
			device_handle, label_text, initial_capacity);

		if (!device_handle)
		{
			fmt::throw_exception("Metal residency manager requires a valid device");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
			MTLResidencySetDescriptor* residency_desc = [MTLResidencySetDescriptor new];
			residency_desc.label = make_ns_string(label_text);
			residency_desc.initialCapacity = initial_capacity;

			NSError* residency_error = nil;
			m_impl->m_residency_set = [device newResidencySetWithDescriptor:residency_desc error:&residency_error];
			if (!m_impl->m_residency_set)
			{
				const std::string error = residency_error ? get_ns_string([residency_error localizedDescription]) : "unknown error";
				fmt::throw_exception("Metal residency set creation failed: %s", error);
			}
		}
		else
		{
			fmt::throw_exception("Metal residency manager requires macOS 26.0 or newer");
		}
	}

	residency_manager::~residency_manager()
	{
		rsx_log.notice("rsx::metal::residency_manager::~residency_manager()");
		end_residency();
	}

	void residency_manager::request_residency()
	{
		rsx_log.notice("rsx::metal::residency_manager::request_residency()");

		if (@available(macOS 26.0, *))
		{
			std::lock_guard lock(m_impl->m_mutex);
			[m_impl->m_residency_set requestResidency];
			m_impl->m_resident = true;
		}
		else
		{
			fmt::throw_exception("Metal residency request requires macOS 26.0 or newer");
		}
	}

	void residency_manager::end_residency()
	{
		rsx_log.notice("rsx::metal::residency_manager::end_residency()");

		if (@available(macOS 26.0, *))
		{
			std::lock_guard lock(m_impl->m_mutex);
			if (!m_impl || !m_impl->m_residency_set || !m_impl->m_resident)
			{
				return;
			}

			[m_impl->m_residency_set endResidency];
			m_impl->m_resident = false;
		}
		else
		{
			fmt::throw_exception("Metal residency end requires macOS 26.0 or newer");
		}
	}

	void residency_manager::add_allocation(void* allocation_handle)
	{
		rsx_log.trace("rsx::metal::residency_manager::add_allocation(allocation_handle=*0x%x)", allocation_handle);

		if (!allocation_handle)
		{
			fmt::throw_exception("Metal residency add requires a valid allocation");
		}

		if (@available(macOS 26.0, *))
		{
			std::lock_guard lock(m_impl->m_mutex);
			id<MTLAllocation> allocation = (__bridge id<MTLAllocation>)allocation_handle;
			[m_impl->m_residency_set addAllocation:allocation];
		}
		else
		{
			fmt::throw_exception("Metal residency add requires macOS 26.0 or newer");
		}
	}

	void residency_manager::remove_allocation(void* allocation_handle)
	{
		rsx_log.trace("rsx::metal::residency_manager::remove_allocation(allocation_handle=*0x%x)", allocation_handle);

		if (!allocation_handle)
		{
			fmt::throw_exception("Metal residency remove requires a valid allocation");
		}

		if (@available(macOS 26.0, *))
		{
			std::lock_guard lock(m_impl->m_mutex);
			id<MTLAllocation> allocation = (__bridge id<MTLAllocation>)allocation_handle;
			[m_impl->m_residency_set removeAllocation:allocation];
		}
		else
		{
			fmt::throw_exception("Metal residency remove requires macOS 26.0 or newer");
		}
	}

	void residency_manager::commit()
	{
		rsx_log.trace("rsx::metal::residency_manager::commit()");

		if (@available(macOS 26.0, *))
		{
			std::lock_guard lock(m_impl->m_mutex);
			[m_impl->m_residency_set commit];
		}
		else
		{
			fmt::throw_exception("Metal residency commit requires macOS 26.0 or newer");
		}
	}

	void* residency_manager::handle() const
	{
		rsx_log.trace("rsx::metal::residency_manager::handle()");
		return (__bridge void*)m_impl->m_residency_set;
	}

	u64 residency_manager::allocated_size() const
	{
		rsx_log.trace("rsx::metal::residency_manager::allocated_size()");

		if (@available(macOS 26.0, *))
		{
			std::lock_guard lock(m_impl->m_mutex);
			return m_impl->m_residency_set.allocatedSize;
		}

		fmt::throw_exception("Metal residency allocated size query requires macOS 26.0 or newer");
	}

	u32 residency_manager::allocation_count() const
	{
		rsx_log.trace("rsx::metal::residency_manager::allocation_count()");

		if (@available(macOS 26.0, *))
		{
			std::lock_guard lock(m_impl->m_mutex);
			return static_cast<u32>(m_impl->m_residency_set.allocationCount);
		}

		fmt::throw_exception("Metal residency allocation count query requires macOS 26.0 or newer");
	}
}
