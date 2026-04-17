#include "stdafx.h"
#include "MTLSynchronization.h"

#include "util/atomic.hpp"

#import <Metal/Metal.h>

namespace
{
	NSString* make_ns_string(std::string_view text)
	{
		return [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
	}
}

namespace rsx::metal
{
	struct shared_event::shared_event_impl
	{
		id<MTLSharedEvent> m_event = nil;
		atomic_t<u64> m_next_value = 0;
	};

	shared_event::shared_event(void* device_handle, std::string_view label)
		: m_impl(std::make_unique<shared_event_impl>())
	{
		const std::string label_text(label);
		rsx_log.notice("rsx::metal::shared_event::shared_event(device_handle=*0x%x, label=%s)", device_handle, label_text);

		if (!device_handle)
		{
			fmt::throw_exception("Metal shared event requires a valid device");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
			m_impl->m_event = [device newSharedEvent];
			if (!m_impl->m_event)
			{
				fmt::throw_exception("Metal shared event creation failed");
			}

			m_impl->m_event.label = make_ns_string(label_text);
			m_impl->m_event.signaledValue = 0;
		}
		else
		{
			fmt::throw_exception("Metal shared event requires macOS 26.0 or newer");
		}
	}

	shared_event::~shared_event()
	{
		rsx_log.notice("rsx::metal::shared_event::~shared_event()");
	}

	u64 shared_event::allocate_signal_value()
	{
		const u64 value = ++m_impl->m_next_value;
		rsx_log.trace("rsx::metal::shared_event::allocate_signal_value(value=0x%x)", value);
		return value;
	}

	void shared_event::signal_queue(void* queue_handle, u64 value) const
	{
		rsx_log.trace("rsx::metal::shared_event::signal_queue(queue_handle=*0x%x, value=0x%x)", queue_handle, value);

		if (!queue_handle)
		{
			fmt::throw_exception("Metal shared event signal requires a valid command queue");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTL4CommandQueue> queue = (__bridge id<MTL4CommandQueue>)queue_handle;
			[queue signalEvent:m_impl->m_event value:value];
		}
		else
		{
			fmt::throw_exception("Metal shared event signal requires macOS 26.0 or newer");
		}
	}

	void shared_event::notify(u64 value, std::function<void(u64)> callback) const
	{
		rsx_log.trace("rsx::metal::shared_event::notify(value=0x%x)", value);

		if (!callback)
		{
			fmt::throw_exception("Metal shared event notification requires a callback");
		}

		if (@available(macOS 26.0, *))
		{
			auto callback_ptr = std::make_shared<std::function<void(u64)>>(std::move(callback));
			[m_impl->m_event notifyListener:[MTLSharedEventListener sharedListener] atValue:value block:^(id<MTLSharedEvent>, uint64_t signaled_value)
			{
				(*callback_ptr)(signaled_value);
			}];
		}
		else
		{
			fmt::throw_exception("Metal shared event notification requires macOS 26.0 or newer");
		}
	}
}
