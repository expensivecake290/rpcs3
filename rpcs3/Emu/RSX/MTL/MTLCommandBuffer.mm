#include "stdafx.h"
#include "MTLCommandBuffer.h"

#include "MTLCommandAllocator.h"
#include "MTLLifetime.h"
#include "MTLSynchronization.h"

#import <Metal/Metal.h>

#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

namespace
{
	NSString* make_label(const char* prefix, u32 frame_index)
	{
		return [NSString stringWithFormat:@"%s %u", prefix, frame_index];
	}
}

namespace rsx::metal
{
	struct command_frame::command_frame_impl
	{
		id<MTL4CommandBuffer> m_command_buffer = nil;
		std::unique_ptr<command_allocator> m_allocator;
		std::unique_ptr<shared_event> m_completion_event;
		std::unique_ptr<lifetime_tracker> m_lifetime;
		std::vector<void*> m_recorded_command_buffers;
		std::vector<std::function<void()>> m_completion_callbacks;
		std::mutex m_mutex;
		std::condition_variable m_available;
		u64 m_completion_value = 0;
		u32 m_frame_index = 0;
		b8 m_pending = false;
		b8 m_completing = false;
		b8 m_recording = false;
	};

	command_frame::command_frame(void* device_handle, u32 frame_index)
		: m_impl(std::make_unique<command_frame_impl>())
	{
		rsx_log.notice("rsx::metal::command_frame::command_frame(device_handle=*0x%x, frame_index=%u)", device_handle, frame_index);

		if (!device_handle)
		{
			fmt::throw_exception("Metal command frame requires a valid device");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;

			m_impl->m_command_buffer = [device newCommandBuffer];
			if (!m_impl->m_command_buffer)
			{
				fmt::throw_exception("Metal command buffer creation failed");
			}

			m_impl->m_command_buffer.label = make_label("RPCS3 Metal command buffer", frame_index);
		}
		else
		{
			fmt::throw_exception("Metal command frame requires macOS 26.0 or newer");
		}

		m_impl->m_allocator = std::make_unique<command_allocator>(device_handle, frame_index);
		m_impl->m_completion_event = std::make_unique<shared_event>(device_handle,
			fmt::format("RPCS3 Metal frame %u completion event", frame_index));
		m_impl->m_lifetime = std::make_unique<lifetime_tracker>(frame_index);
		m_impl->m_recorded_command_buffers.reserve(2);
		m_impl->m_frame_index = frame_index;
	}

	command_frame::~command_frame()
	{
		rsx_log.notice("rsx::metal::command_frame::~command_frame(frame_index=%u)", m_impl ? m_impl->m_frame_index : 0);
		wait_until_available();
	}

	void command_frame::begin()
	{
		rsx_log.trace("rsx::metal::command_frame::begin(frame_index=%u)", m_impl->m_frame_index);

		wait_until_available();

		if (@available(macOS 26.0, *))
		{
			m_impl->m_allocator->reset();
			m_impl->m_lifetime->clear();
			m_impl->m_recorded_command_buffers.clear();
			m_impl->m_completion_callbacks.clear();
			[m_impl->m_command_buffer beginCommandBufferWithAllocator:(__bridge id<MTL4CommandAllocator>)m_impl->m_allocator->handle()];
		}
		else
		{
			fmt::throw_exception("Metal command buffer recording requires macOS 26.0 or newer");
		}

		m_impl->m_recording = true;
	}

	void command_frame::end()
	{
		rsx_log.trace("rsx::metal::command_frame::end(frame_index=%u)", m_impl->m_frame_index);

		ensure(m_impl->m_recording);

		if (@available(macOS 26.0, *))
		{
			[m_impl->m_command_buffer endCommandBuffer];
			m_impl->m_recorded_command_buffers.emplace_back((__bridge void*)m_impl->m_command_buffer);
		}
		else
		{
			fmt::throw_exception("Metal command buffer recording requires macOS 26.0 or newer");
		}

		m_impl->m_recording = false;
	}

	void command_frame::wait_until_available()
	{
		rsx_log.trace("rsx::metal::command_frame::wait_until_available(frame_index=%u)", m_impl->m_frame_index);

		std::unique_lock lock(m_impl->m_mutex);
		m_impl->m_available.wait(lock, [this]()
		{
			return !m_impl->m_pending;
		});
	}

	u64 command_frame::arm_completion_signal()
	{
		rsx_log.trace("rsx::metal::command_frame::arm_completion_signal(frame_index=%u)", m_impl->m_frame_index);

		const u64 signal_value = m_impl->m_completion_event->allocate_signal_value();

		{
			std::lock_guard lock(m_impl->m_mutex);
			ensure(!m_impl->m_pending);
			ensure(!m_impl->m_recording);

			m_impl->m_completion_value = signal_value;
			m_impl->m_pending = true;
			m_impl->m_completing = false;
		}

		m_impl->m_completion_event->notify(signal_value, [this](u64 completed_value)
		{
			mark_completed(completed_value);
		});

		return signal_value;
	}

	void command_frame::signal_completion(void* queue_handle) const
	{
		rsx_log.trace("rsx::metal::command_frame::signal_completion(frame_index=%u, queue_handle=*0x%x)",
			m_impl->m_frame_index, queue_handle);

		m_impl->m_completion_event->signal_queue(queue_handle, completion_value());
	}

	void command_frame::mark_completed(u64 signal_value)
	{
		rsx_log.trace("rsx::metal::command_frame::mark_completed(frame_index=%u, signal_value=0x%x)", m_impl->m_frame_index, signal_value);

		std::vector<std::function<void()>> callbacks;

		{
			std::lock_guard lock(m_impl->m_mutex);
			if (!m_impl->m_pending || m_impl->m_completing || signal_value < m_impl->m_completion_value)
			{
				return;
			}

			m_impl->m_completing = true;
			callbacks = std::move(m_impl->m_completion_callbacks);
		}

		for (const std::function<void()>& callback : callbacks)
		{
			callback();
		}

		{
			std::lock_guard lock(m_impl->m_mutex);
			m_impl->m_lifetime->clear();
			m_impl->m_recorded_command_buffers.clear();
			m_impl->m_completion_callbacks.clear();
			m_impl->m_completion_value = 0;
			m_impl->m_pending = false;
			m_impl->m_completing = false;
		}

		m_impl->m_available.notify_all();
	}

	void command_frame::use_residency_set(void* residency_set_handle)
	{
		rsx_log.trace("rsx::metal::command_frame::use_residency_set(frame_index=%u, residency_set_handle=*0x%x)",
			m_impl->m_frame_index, residency_set_handle);

		if (!residency_set_handle)
		{
			return;
		}

		if (@available(macOS 26.0, *))
		{
			id<MTLResidencySet> residency_set = (__bridge id<MTLResidencySet>)residency_set_handle;
			[m_impl->m_command_buffer useResidencySet:residency_set];
		}
		else
		{
			fmt::throw_exception("Metal residency set binding requires macOS 26.0 or newer");
		}
	}

	void command_frame::track_object(void* object_handle)
	{
		rsx_log.trace("rsx::metal::command_frame::track_object(frame_index=%u, object_handle=*0x%x)",
			m_impl->m_frame_index, object_handle);

		if (!object_handle)
		{
			return;
		}

		std::lock_guard lock(m_impl->m_mutex);
		m_impl->m_lifetime->track_object(object_handle);
	}

	void command_frame::on_completed(std::function<void()> callback)
	{
		rsx_log.trace("rsx::metal::command_frame::on_completed(frame_index=%u)", m_impl->m_frame_index);

		if (!callback)
		{
			fmt::throw_exception("Metal command frame completion callback requires a valid callback");
		}

		std::lock_guard lock(m_impl->m_mutex);
		m_impl->m_completion_callbacks.emplace_back(std::move(callback));
	}

	void* command_frame::command_buffer_handle() const
	{
		rsx_log.trace("rsx::metal::command_frame::command_buffer_handle(frame_index=%u)", m_impl->m_frame_index);
		return (__bridge void*)m_impl->m_command_buffer;
	}

	std::span<void* const> command_frame::command_buffer_handles() const
	{
		rsx_log.trace("rsx::metal::command_frame::command_buffer_handles(frame_index=%u, count=%u)",
			m_impl->m_frame_index, static_cast<u32>(m_impl->m_recorded_command_buffers.size()));

		return std::span<void* const>(m_impl->m_recorded_command_buffers.data(), m_impl->m_recorded_command_buffers.size());
	}

	u32 command_frame::frame_index() const
	{
		rsx_log.trace("rsx::metal::command_frame::frame_index()");
		return m_impl->m_frame_index;
	}

	u64 command_frame::completion_value() const
	{
		rsx_log.trace("rsx::metal::command_frame::completion_value(frame_index=%u)", m_impl->m_frame_index);

		std::lock_guard lock(m_impl->m_mutex);
		return m_impl->m_completion_value;
	}
}
