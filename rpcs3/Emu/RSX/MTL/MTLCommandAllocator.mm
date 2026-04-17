#include "stdafx.h"
#include "MTLCommandAllocator.h"

#import <Metal/Metal.h>

namespace
{
	NSString* make_label(const char* prefix, u32 frame_index)
	{
		return [NSString stringWithFormat:@"%s %u", prefix, frame_index];
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
	struct command_allocator::command_allocator_impl
	{
		id<MTL4CommandAllocator> m_allocator = nil;
		u32 m_frame_index = 0;
	};

	command_allocator::command_allocator(void* device_handle, u32 frame_index)
		: m_impl(std::make_unique<command_allocator_impl>())
	{
		rsx_log.notice("rsx::metal::command_allocator::command_allocator(device_handle=*0x%x, frame_index=%u)", device_handle, frame_index);

		if (!device_handle)
		{
			fmt::throw_exception("Metal command allocator requires a valid device");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
			MTL4CommandAllocatorDescriptor* allocator_desc = [MTL4CommandAllocatorDescriptor new];
			allocator_desc.label = make_label("RPCS3 Metal command allocator", frame_index);

			NSError* allocator_error = nil;
			m_impl->m_allocator = [device newCommandAllocatorWithDescriptor:allocator_desc error:&allocator_error];
			if (!m_impl->m_allocator)
			{
				const std::string error = allocator_error ? get_ns_string([allocator_error localizedDescription]) : "unknown error";
				fmt::throw_exception("Metal command allocator creation failed: %s", error);
			}
		}
		else
		{
			fmt::throw_exception("Metal command allocator requires macOS 26.0 or newer");
		}

		m_impl->m_frame_index = frame_index;
	}

	command_allocator::~command_allocator()
	{
		rsx_log.notice("rsx::metal::command_allocator::~command_allocator(frame_index=%u)", m_impl ? m_impl->m_frame_index : 0);
	}

	void command_allocator::reset()
	{
		rsx_log.trace("rsx::metal::command_allocator::reset(frame_index=%u)", m_impl->m_frame_index);

		if (@available(macOS 26.0, *))
		{
			[m_impl->m_allocator reset];
		}
		else
		{
			fmt::throw_exception("Metal command allocator reset requires macOS 26.0 or newer");
		}
	}

	void* command_allocator::handle() const
	{
		rsx_log.trace("rsx::metal::command_allocator::handle(frame_index=%u)", m_impl->m_frame_index);
		return (__bridge void*)m_impl->m_allocator;
	}
}
