#pragma once

#include "util/types.hpp"

#include <functional>
#include <memory>
#include <string>

namespace rsx::metal
{
	class buffer;
	class command_frame;
	class device;

	struct draw_buffer_binding
	{
		buffer* resource = nullptr;
		u64 offset = 0;
		u64 size = 0;
	};

	struct draw_resource_stats
	{
		u32 retained_buffer_count = 0;
		u32 created_buffer_count = 0;
		u32 reused_buffer_count = 0;
		u32 uploaded_buffer_count = 0;
		u64 uploaded_byte_count = 0;
	};

	class draw_resource_manager
	{
	public:
		explicit draw_resource_manager(device& dev);
		~draw_resource_manager();

		draw_resource_manager(const draw_resource_manager&) = delete;
		draw_resource_manager& operator=(const draw_resource_manager&) = delete;

		draw_buffer_binding upload_generated_buffer(command_frame& frame, u64 size, const std::string& label, const std::function<void(void*, u64)>& fill);
		buffer& zero_vertex_buffer() const;
		draw_resource_stats stats() const;
		void report() const;

	private:
		struct draw_resource_manager_impl;
		std::unique_ptr<draw_resource_manager_impl> m_impl;
	};
}
