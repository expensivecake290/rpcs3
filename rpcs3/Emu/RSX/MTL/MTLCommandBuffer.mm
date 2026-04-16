#include "stdafx.h"
#include "MTLCommandBuffer.h"

#import <Metal/Metal.h>

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
		id<MTL4CommandAllocator> m_allocator = nil;
		id<MTL4CommandBuffer> m_command_buffer = nil;
		dispatch_semaphore_t m_available = nullptr;
		NSMutableArray* m_tracked_objects = nil;
		u32 m_frame_index = 0;
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

			NSError* allocator_error = nil;
			MTL4CommandAllocatorDescriptor* allocator_desc = [MTL4CommandAllocatorDescriptor new];
			allocator_desc.label = make_label("RPCS3 Metal command allocator", frame_index);

			m_impl->m_allocator = [device newCommandAllocatorWithDescriptor:allocator_desc error:&allocator_error];
			if (!m_impl->m_allocator)
			{
				fmt::throw_exception("Metal command allocator creation failed");
			}

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

		m_impl->m_available = dispatch_semaphore_create(1);
		m_impl->m_tracked_objects = [NSMutableArray arrayWithCapacity:16];
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
			[m_impl->m_allocator reset];
			[m_impl->m_tracked_objects removeAllObjects];
			[m_impl->m_command_buffer beginCommandBufferWithAllocator:m_impl->m_allocator];
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

		dispatch_semaphore_wait(m_impl->m_available, DISPATCH_TIME_FOREVER);
		dispatch_semaphore_signal(m_impl->m_available);
	}

	void command_frame::mark_submitted()
	{
		rsx_log.trace("rsx::metal::command_frame::mark_submitted(frame_index=%u)", m_impl->m_frame_index);
		dispatch_semaphore_wait(m_impl->m_available, DISPATCH_TIME_FOREVER);
	}

	void command_frame::mark_completed()
	{
		rsx_log.trace("rsx::metal::command_frame::mark_completed(frame_index=%u)", m_impl->m_frame_index);
		[m_impl->m_tracked_objects removeAllObjects];
		dispatch_semaphore_signal(m_impl->m_available);
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

		id object = (__bridge id)object_handle;
		[m_impl->m_tracked_objects addObject:object];
	}

	void* command_frame::command_buffer_handle() const
	{
		rsx_log.trace("rsx::metal::command_frame::command_buffer_handle(frame_index=%u)", m_impl->m_frame_index);
		return (__bridge void*)m_impl->m_command_buffer;
	}

	u32 command_frame::frame_index() const
	{
		rsx_log.trace("rsx::metal::command_frame::frame_index()");
		return m_impl->m_frame_index;
	}
}
