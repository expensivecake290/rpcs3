#pragma once

#include "util/types.hpp"

#include <functional>
#include <memory>
#include <span>

namespace rsx::metal
{
	class command_frame
	{
	public:
		command_frame(void* device_handle, u32 frame_index);
		~command_frame();

		command_frame(const command_frame&) = delete;
		command_frame& operator=(const command_frame&) = delete;

		void begin();
		void end();
		void wait_until_available();
		u64 arm_completion_signal();
		void signal_completion(void* queue_handle) const;
		void mark_completed(u64 signal_value);
		void use_residency_set(void* residency_set_handle);
		void track_object(void* object_handle);
		void on_completed(std::function<void()> callback);

		void* command_buffer_handle() const;
		std::span<void* const> command_buffer_handles() const;
		u32 frame_index() const;
		u64 completion_value() const;

	private:
		struct command_frame_impl;
		std::unique_ptr<command_frame_impl> m_impl;

		friend class command_queue;
	};
}
