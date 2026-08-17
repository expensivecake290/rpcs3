#include "stdafx.h"
#include "sync.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <limits>

namespace mtl
{
	struct timeline_event::impl
	{
		id<MTLSharedEvent> event;
		MTLSharedEventListener* listener;
		dispatch_queue_t listener_queue;
		sync_domain domain = sync_domain::host;
	};

	struct fence::impl
	{
		id<MTLFence> fence;
	};

	namespace
	{
		MTLStages get_encoder_stages(id<MTL4CommandEncoder> encoder)
		{
			if ([encoder conformsToProtocol:@protocol(MTL4ComputeCommandEncoder)])
			{
				const MTLStages stages = [static_cast<id<MTL4ComputeCommandEncoder>>(encoder) stages];
				return stages ? stages : static_cast<MTLStages>(MTLStageDispatch | MTLStageBlit);
			}

			return static_cast<MTLStages>(MTLStageVertex | MTLStageFragment | MTLStageTile | MTLStageObject | MTLStageMesh);
		}
	}

	timeline_event::timeline_event()
		: m_impl(std::make_unique<impl>())
	{
	}

	timeline_event::~timeline_event()
	{
		destroy();
	}

	timeline_event::timeline_event(timeline_event&& other) noexcept
		: m_impl(std::move(other.m_impl))
	{
	}

	timeline_event& timeline_event::operator=(timeline_event&& other) noexcept
	{
		if (this != &other)
		{
			destroy();
			m_impl = std::move(other.m_impl);
		}
		return *this;
	}

	void timeline_event::create(const render_device& device, sync_domain domain, std::string_view label, u64 initial_value)
	{
		destroy();
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		m_impl->event = [device.native_handle() newSharedEvent];
		if (!m_impl->event)
		{
			fmt::throw_exception("Metal returned no shared event for '%s'", label);
		}

		const std::string queue_name = fmt::format("org.rpcs3.metal.event.%s", label);
		m_impl->listener_queue = dispatch_queue_create(queue_name.c_str(), DISPATCH_QUEUE_SERIAL);
		m_impl->listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:m_impl->listener_queue];
		if (!m_impl->listener)
		{
			m_impl->event = nil;
			m_impl->listener_queue = nullptr;
			fmt::throw_exception("Metal returned no shared-event listener for '%s'", label);
		}

		m_impl->event.label = [NSString stringWithUTF8String:std::string(label).c_str()];
		m_impl->event.signaledValue = initial_value;
		m_impl->domain = domain;
	}

	void timeline_event::destroy()
	{
		if (m_impl)
		{
			m_impl->listener = nil;
			m_impl->listener_queue = nullptr;
			m_impl->event = nil;
		}
	}

	void timeline_event::signal_host(u64 value)
	{
		if (!m_impl || !m_impl->event || m_impl->domain != sync_domain::host)
		{
			fmt::throw_exception("Cannot host-signal this Metal timeline event");
		}

		if (value < m_impl->event.signaledValue)
		{
			fmt::throw_exception("Metal timeline event values must be monotonic");
		}
		m_impl->event.signaledValue = value;
	}

	timeline_event::operator bool() const
	{
		return m_impl && m_impl->event;
	}

	shared_event_handle timeline_event::native_handle() const
	{
		return m_impl ? m_impl->event : nil;
	}

	u64 timeline_event::signaled_value() const
	{
		if (!m_impl || !m_impl->event)
		{
			fmt::throw_exception("Signaled value requested from an empty Metal timeline event");
		}
		return m_impl->event.signaledValue;
	}

	bool timeline_event::reached(u64 value) const
	{
		return m_impl && m_impl->event && m_impl->event.signaledValue >= value;
	}

	bool timeline_event::wait(u64 value, std::chrono::nanoseconds timeout) const
	{
		if (!m_impl || !m_impl->event || !m_impl->listener)
		{
			fmt::throw_exception("Cannot wait on an empty Metal timeline event");
		}
		if (reached(value))
		{
			return true;
		}
		if (timeout <= std::chrono::nanoseconds::zero())
		{
			return false;
		}

		dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
		[m_impl->event notifyListener:m_impl->listener atValue:value block:^(id<MTLSharedEvent>, uint64_t)
		{
			dispatch_semaphore_signal(semaphore);
		}];

		const auto nanoseconds = std::min<s64>(timeout.count(), std::numeric_limits<s64>::max());
		return dispatch_semaphore_wait(semaphore, dispatch_time(DISPATCH_TIME_NOW, nanoseconds)) == 0;
	}

	void timeline_event::wait(u64 value) const
	{
		if (!m_impl || !m_impl->event || !m_impl->listener)
		{
			fmt::throw_exception("Cannot wait on an empty Metal timeline event");
		}
		if (reached(value))
		{
			return;
		}

		dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
		[m_impl->event notifyListener:m_impl->listener atValue:value block:^(id<MTLSharedEvent>, uint64_t)
		{
			dispatch_semaphore_signal(semaphore);
		}];
		dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
	}

	event_operation timeline_event::wait_operation(u64 value) const
	{
		if (!m_impl || !m_impl->event)
		{
			fmt::throw_exception("Cannot create a wait operation from an empty Metal timeline event");
		}
		return {m_impl->event, value};
	}

	event_operation timeline_event::signal_operation(u64 value) const
	{
		if (!m_impl || !m_impl->event || value < m_impl->event.signaledValue)
		{
			fmt::throw_exception("Invalid Metal timeline signal operation");
		}
		return {m_impl->event, value};
	}

	fence::fence()
		: m_impl(std::make_unique<impl>())
	{
	}

	fence::~fence()
	{
		destroy();
	}

	fence::fence(fence&& other) noexcept
		: m_impl(std::move(other.m_impl))
	{
	}

	fence& fence::operator=(fence&& other) noexcept
	{
		if (this != &other)
		{
			destroy();
			m_impl = std::move(other.m_impl);
		}
		return *this;
	}

	void fence::create(const render_device& device, std::string_view label)
	{
		destroy();
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}
		m_impl->fence = [device.native_handle() newFence];
		if (!m_impl->fence)
		{
			fmt::throw_exception("Metal returned no fence for '%s'", label);
		}
		m_impl->fence.label = [NSString stringWithUTF8String:std::string(label).c_str()];
	}

	void fence::destroy()
	{
		if (m_impl)
		{
			m_impl->fence = nil;
		}
	}

	void fence::update(native_encoder_handle encoder)
	{
		if (!m_impl || !m_impl->fence || !encoder)
		{
			fmt::throw_exception("Cannot update an invalid Metal fence or encoder");
		}
		id<MTL4CommandEncoder> native_encoder = (__bridge id<MTL4CommandEncoder>)encoder;
		[native_encoder updateFence:m_impl->fence afterEncoderStages:get_encoder_stages(native_encoder)];
	}

	void fence::wait(native_encoder_handle encoder) const
	{
		if (!m_impl || !m_impl->fence || !encoder)
		{
			fmt::throw_exception("Cannot wait on an invalid Metal fence or encoder");
		}
		id<MTL4CommandEncoder> native_encoder = (__bridge id<MTL4CommandEncoder>)encoder;
		[native_encoder waitForFence:m_impl->fence beforeEncoderStages:get_encoder_stages(native_encoder)];
	}

	fence::operator bool() const
	{
		return m_impl && m_impl->fence;
	}

	void* fence::native_handle() const
	{
		return m_impl ? (__bridge void*)m_impl->fence : nullptr;
	}

	void submission_fence::reset()
	{
		if (m_submission && !m_submission.completed())
		{
			fmt::throw_exception("Cannot reset a pending Metal submission fence");
		}
		m_submission = {};
	}

	void submission_fence::signal(submission value)
	{
		if (!value || (m_submission && !m_submission.completed()))
		{
			fmt::throw_exception("Cannot signal a Metal submission fence in its current state");
		}
		m_submission = std::move(value);
	}

	bool submission_fence::signaled() const
	{
		return m_submission && m_submission.completed();
	}

	bool submission_fence::wait(std::chrono::nanoseconds timeout) const
	{
		return m_submission && m_submission.wait(timeout);
	}

	void submission_fence::wait() const
	{
		if (!m_submission)
		{
			fmt::throw_exception("Cannot wait on an empty Metal submission fence");
		}
		m_submission.wait();
	}

	u64 submission_fence::value() const
	{
		return m_submission.value();
	}

	debug_marker_scope::debug_marker_scope(command_buffer& commands, std::string_view message)
		: m_commands(&commands)
	{
		m_commands->push_debug_group(message);
	}

	debug_marker_scope::~debug_marker_scope()
	{
		if (m_commands)
		{
			m_commands->pop_debug_group();
		}
	}

	bool wait_for_submission(const submission& value, std::chrono::nanoseconds timeout)
	{
		return value && value.wait(timeout);
	}

	void wait_for_submission(const submission& value)
	{
		if (!value)
		{
			fmt::throw_exception("Cannot wait for an empty Metal submission");
		}
		value.wait();
	}
}
