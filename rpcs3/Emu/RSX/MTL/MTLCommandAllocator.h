#pragma once

#include "util/types.hpp"

#include <memory>

namespace rsx::metal
{
	class command_allocator
	{
	public:
		command_allocator(void* device_handle, u32 frame_index);
		~command_allocator();

		command_allocator(const command_allocator&) = delete;
		command_allocator& operator=(const command_allocator&) = delete;

		void reset();
		void* handle() const;

	private:
		struct command_allocator_impl;
		std::unique_ptr<command_allocator_impl> m_impl;
	};
}
