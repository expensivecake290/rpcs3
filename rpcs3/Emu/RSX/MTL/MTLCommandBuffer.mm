#include "stdafx.h"
#include "MTLCommandBuffer.h"

#include "MTLCommandAllocator.h"
#include "MTLDevice.h"
#include "MTLLifetime.h"
#include "MTLResourceState.h"
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

	struct resident_allocation
	{
		rsx::metal::device* m_device = nullptr;
		void* m_allocation_handle = nullptr;
	};

	void release_resident_allocations(std::vector<resident_allocation> allocations)
	{
		rsx_log.trace("release_resident_allocations(count=%u)", static_cast<u32>(allocations.size()));

		for (const resident_allocation& allocation : allocations)
		{
			if (!allocation.m_device || !allocation.m_allocation_handle)
			{
				continue;
			}

			allocation.m_device->remove_resident_allocation(allocation.m_allocation_handle);
			allocation.m_device->commit_residency();
		}
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
		std::unique_ptr<resource_state_tracker> m_resource_state;
		std::vector<void*> m_recorded_command_buffers;
		std::vector<std::function<void()>> m_completion_callbacks;
		std::vector<resident_allocation> m_resident_allocations;
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
		m_impl->m_resource_state = std::make_unique<resource_state_tracker>(frame_index);
		m_impl->m_recorded_command_buffers.reserve(2);
		m_impl->m_resident_allocations.reserve(8);
		m_impl->m_frame_index = frame_index;
	}

	command_frame::~command_frame()
	{
		rsx_log.notice("rsx::metal::command_frame::~command_frame(frame_index=%u)", m_impl ? m_impl->m_frame_index : 0);
		wait_until_available();

		std::vector<resident_allocation> resident_allocations;
		{
			std::lock_guard lock(m_impl->m_mutex);
			resident_allocations = std::move(m_impl->m_resident_allocations);
		}

		release_resident_allocations(std::move(resident_allocations));
	}

	void command_frame::begin()
	{
		rsx_log.trace("rsx::metal::command_frame::begin(frame_index=%u)", m_impl->m_frame_index);

		wait_until_available();

		std::vector<resident_allocation> resident_allocations;
		{
			std::lock_guard lock(m_impl->m_mutex);
			ensure(!m_impl->m_pending);
			resident_allocations = std::move(m_impl->m_resident_allocations);
			m_impl->m_resident_allocations.reserve(8);
			m_impl->m_completion_callbacks.clear();
		}

		release_resident_allocations(std::move(resident_allocations));

		if (@available(macOS 26.0, *))
		{
			m_impl->m_allocator->reset();
			m_impl->m_lifetime->clear();
			m_impl->m_resource_state->reset();
			m_impl->m_recorded_command_buffers.clear();
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
		std::vector<resident_allocation> resident_allocations;

		{
			std::lock_guard lock(m_impl->m_mutex);
			if (!m_impl->m_pending || m_impl->m_completing || signal_value < m_impl->m_completion_value)
			{
				return;
			}

			m_impl->m_completing = true;
			callbacks = std::move(m_impl->m_completion_callbacks);
			resident_allocations = std::move(m_impl->m_resident_allocations);
		}

		for (const std::function<void()>& callback : callbacks)
		{
			callback();
		}

		release_resident_allocations(std::move(resident_allocations));

		{
			std::lock_guard lock(m_impl->m_mutex);
			m_impl->m_lifetime->clear();
			m_impl->m_resource_state->reset();
			m_impl->m_recorded_command_buffers.clear();
			m_impl->m_completion_callbacks.clear();
			m_impl->m_resident_allocations.reserve(8);
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

	void command_frame::track_resident_allocation(device& metal_device, void* allocation_handle)
	{
		rsx_log.trace("rsx::metal::command_frame::track_resident_allocation(frame_index=%u, allocation_handle=*0x%x)",
			m_impl->m_frame_index, allocation_handle);

		if (!allocation_handle)
		{
			fmt::throw_exception("Metal command frame resident allocation tracking requires a valid allocation");
		}

		track_object(allocation_handle);
		metal_device.add_resident_allocation(allocation_handle);
		metal_device.commit_residency();

		std::lock_guard lock(m_impl->m_mutex);
		m_impl->m_resident_allocations.push_back({ &metal_device, allocation_handle });
	}

	resource_barrier command_frame::track_resource_usage(const resource_usage& usage)
	{
		rsx_log.trace("rsx::metal::command_frame::track_resource_usage(frame_index=%u, resource_id=0x%x, stage=%s, access=%s, scope=%s)",
			m_impl->m_frame_index,
			usage.resource_id,
			describe_resource_stage(usage.stage),
			describe_resource_access(usage.access),
			describe_resource_barrier_scope(usage.scope));

		std::lock_guard lock(m_impl->m_mutex);
		return m_impl->m_resource_state->record_usage(usage);
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
