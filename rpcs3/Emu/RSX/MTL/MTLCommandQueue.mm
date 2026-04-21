#include "stdafx.h"
#include "MTLCommandQueue.h"

#import <Metal/Metal.h>

#include <vector>

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
}

namespace rsx::metal
{
	struct command_queue::command_queue_impl
	{
		id<MTL4CommandQueue> m_queue = nil;
		std::array<std::unique_ptr<command_frame>, 3> m_frames{};
		u32 m_frame_index = 0;
	};

	command_queue::command_queue(device& metal_device)
		: m_impl(std::make_unique<command_queue_impl>())
	{
		rsx_log.notice("rsx::metal::command_queue::command_queue()");

		if (@available(macOS 26.0, *))
		{
			id<MTLDevice> device = (__bridge id<MTLDevice>)metal_device.handle();
			NSError* queue_error = nil;

			MTL4CommandQueueDescriptor* queue_desc = [MTL4CommandQueueDescriptor new];
			queue_desc.label = @"RPCS3 Metal command queue";

			m_impl->m_queue = [device newMTL4CommandQueueWithDescriptor:queue_desc error:&queue_error];
			if (!m_impl->m_queue)
			{
				const std::string error = queue_error ? get_ns_string([queue_error localizedDescription]) : "unknown error";
				fmt::throw_exception("Metal command queue creation failed: %s", error);
			}

			[m_impl->m_queue addResidencySet:(__bridge id<MTLResidencySet>)metal_device.residency_set_handle()];
		}
		else
		{
			fmt::throw_exception("Metal command queue requires macOS 26.0 or newer");
		}

		for (u32 index = 0; index < m_impl->m_frames.size(); index++)
		{
			m_impl->m_frames[index] = std::make_unique<command_frame>(metal_device.handle(), index);
		}
	}

	command_queue::~command_queue()
	{
		rsx_log.notice("rsx::metal::command_queue::~command_queue()");
		wait_idle();
	}

	command_frame& command_queue::begin_frame()
	{
		rsx_log.trace("rsx::metal::command_queue::begin_frame()");

		command_frame& frame = *m_impl->m_frames[m_impl->m_frame_index];
		m_impl->m_frame_index = (m_impl->m_frame_index + 1) % m_impl->m_frames.size();
		frame.begin();
		return frame;
	}

	void command_queue::submit_frame(command_frame& frame, void* drawable_handle)
	{
		rsx_log.trace("rsx::metal::command_queue::submit_frame(frame_index=%u, drawable_handle=*0x%x)",
			frame.frame_index(), drawable_handle);

		id<MTLDrawable> drawable = (__bridge id<MTLDrawable>)drawable_handle;
		const resource_state_stats resource_stats = frame.resource_stats();

		if (drawable && !resource_stats.present_boundary_count)
		{
			fmt::throw_exception("Metal drawable submission requires a recorded present boundary");
		}

		if (!drawable && resource_stats.present_boundary_count)
		{
			fmt::throw_exception("Metal non-drawable submission recorded %u present boundaries",
				resource_stats.present_boundary_count);
		}

		if (@available(macOS 26.0, *))
		{
			std::vector<id<MTL4CommandBuffer>> command_buffers;
			for (void* command_buffer_handle : frame.command_buffer_handles())
			{
				command_buffers.emplace_back((__bridge id<MTL4CommandBuffer>)command_buffer_handle);
			}

			ensure(!command_buffers.empty());

			const u64 signal_value = frame.arm_completion_signal();

			MTL4CommitOptions* options = [MTL4CommitOptions new];
			command_frame* submitted_frame = &frame;

			[options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback)
			{
				if (feedback.error)
				{
					rsx_log.error("Metal command queue reported GPU execution error: %s",
						get_ns_string([feedback.error localizedDescription]));
					submitted_frame->mark_completed(signal_value);
				}
			}];

			if (drawable)
			{
				[m_impl->m_queue waitForDrawable:drawable];
			}

			[m_impl->m_queue commit:command_buffers.data() count:command_buffers.size() options:options];

			if (drawable)
			{
				[m_impl->m_queue signalDrawable:drawable];
			}

			frame.signal_completion((__bridge void*)m_impl->m_queue);

			if (drawable)
			{
				[drawable present];
			}
		}
		else
		{
			frame.mark_completed(frame.completion_value());
			fmt::throw_exception("Metal command queue submission requires macOS 26.0 or newer");
		}
	}

	void command_queue::wait_idle()
	{
		rsx_log.notice("rsx::metal::command_queue::wait_idle()");

		for (const std::unique_ptr<command_frame>& frame : m_impl->m_frames)
		{
			if (frame)
			{
				frame->wait_until_available();
			}
		}
	}

	void* command_queue::handle() const
	{
		rsx_log.trace("rsx::metal::command_queue::handle()");
		return (__bridge void*)m_impl->m_queue;
	}
}
