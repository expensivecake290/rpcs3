#include "stdafx.h"
#include "MTLLifetime.h"

#import <Foundation/Foundation.h>

#include <limits>

namespace rsx::metal
{
	struct lifetime_tracker::lifetime_tracker_impl
	{
		NSMutableArray* m_objects = nil;
		u32 m_frame_index = 0;
	};

	lifetime_tracker::lifetime_tracker(u32 frame_index)
		: m_impl(std::make_unique<lifetime_tracker_impl>())
	{
		rsx_log.notice("rsx::metal::lifetime_tracker::lifetime_tracker(frame_index=%u)", frame_index);
		m_impl->m_objects = [NSMutableArray arrayWithCapacity:16];
		m_impl->m_frame_index = frame_index;
	}

	lifetime_tracker::~lifetime_tracker()
	{
		rsx_log.notice("rsx::metal::lifetime_tracker::~lifetime_tracker(frame_index=%u)", m_impl ? m_impl->m_frame_index : 0);
		clear();
	}

	void lifetime_tracker::track_object(void* object_handle)
	{
		rsx_log.trace("rsx::metal::lifetime_tracker::track_object(frame_index=%u, object_handle=*0x%x)",
			m_impl->m_frame_index, object_handle);

		if (!object_handle)
		{
			return;
		}

		if ([m_impl->m_objects count] >= static_cast<NSUInteger>(std::numeric_limits<u32>::max()))
		{
			fmt::throw_exception("Metal lifetime tracker object count overflow for frame %u", m_impl->m_frame_index);
		}

		id object = (__bridge id)object_handle;
		[m_impl->m_objects addObject:object];
	}

	void lifetime_tracker::clear()
	{
		rsx_log.trace("rsx::metal::lifetime_tracker::clear(frame_index=%u, count=%u)", m_impl->m_frame_index, count());
		[m_impl->m_objects removeAllObjects];
	}

	u32 lifetime_tracker::count() const
	{
		const NSUInteger object_count = [m_impl->m_objects count];
		if (object_count > static_cast<NSUInteger>(std::numeric_limits<u32>::max()))
		{
			fmt::throw_exception("Metal lifetime tracker object count exceeds u32 range for frame %u", m_impl->m_frame_index);
		}

		return static_cast<u32>(object_count);
	}
}
