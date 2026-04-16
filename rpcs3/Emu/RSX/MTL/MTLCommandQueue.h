#pragma once

#include "MTLCommandBuffer.h"
#include "MTLDevice.h"

#include <array>
#include <memory>

namespace rsx::metal
{
	class command_queue
	{
	public:
		explicit command_queue(device& metal_device);
		~command_queue();

		command_queue(const command_queue&) = delete;
		command_queue& operator=(const command_queue&) = delete;

		command_frame& begin_frame();
		void submit_frame(command_frame& frame, void* drawable_handle);
		void wait_idle();
		void* handle() const;

	private:
		struct command_queue_impl;
		std::unique_ptr<command_queue_impl> m_impl;
	};
}
