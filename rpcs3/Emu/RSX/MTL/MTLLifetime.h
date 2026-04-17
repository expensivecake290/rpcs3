#pragma once

#include "util/types.hpp"

#include <memory>

namespace rsx::metal
{
	class lifetime_tracker
	{
	public:
		explicit lifetime_tracker(u32 frame_index);
		~lifetime_tracker();

		lifetime_tracker(const lifetime_tracker&) = delete;
		lifetime_tracker& operator=(const lifetime_tracker&) = delete;

		void track_object(void* object_handle);
		void clear();
		u32 count() const;

	private:
		struct lifetime_tracker_impl;
		std::unique_ptr<lifetime_tracker_impl> m_impl;
	};
}
