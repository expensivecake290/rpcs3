#pragma once

#include "util/types.hpp"

#include <functional>
#include <memory>
#include <string_view>

namespace rsx::metal
{
	class shared_event
	{
	public:
		shared_event(void* device_handle, std::string_view label);
		~shared_event();

		shared_event(const shared_event&) = delete;
		shared_event& operator=(const shared_event&) = delete;

		u64 allocate_signal_value();
		void signal_queue(void* queue_handle, u64 value) const;
		void notify(u64 value, std::function<void(u64)> callback) const;

	private:
		struct shared_event_impl;
		std::unique_ptr<shared_event_impl> m_impl;
	};
}
