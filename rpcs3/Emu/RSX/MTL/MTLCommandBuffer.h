#pragma once

#include "util/types.hpp"

#include <memory>

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
		void mark_submitted();
		void mark_completed();
		void use_residency_set(void* residency_set_handle);
		void track_object(void* object_handle);

		void* command_buffer_handle() const;
		u32 frame_index() const;

	private:
		struct command_frame_impl;
		std::unique_ptr<command_frame_impl> m_impl;

		friend class command_queue;
	};
}
