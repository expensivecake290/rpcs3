#pragma once

#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
	class command_frame;

	struct heap_buffer_allocation
	{
		void* buffer_handle = nullptr;
		void* heap_handle = nullptr;
		u64 requested_size = 0;
		u64 required_size = 0;
		u64 required_alignment = 0;
		b8 created_heap = false;
		b8 heap_backed = false;
	};

	struct heap_texture_allocation
	{
		void* texture_handle = nullptr;
		void* heap_handle = nullptr;
		u64 required_size = 0;
		u64 required_alignment = 0;
		b8 created_heap = false;
		b8 heap_backed = false;
	};

	struct heap_manager_stats
	{
		u32 heap_count = 0;
		u32 buffer_allocation_count = 0;
		u32 texture_allocation_count = 0;
		u32 live_resource_count = 0;
		u32 active_resource_use_count = 0;
		u32 pending_aliasable_resource_count = 0;
		u32 aliasable_resource_count = 0;
		u64 total_heap_size = 0;
		u64 used_heap_size = 0;
		u64 current_allocated_size = 0;
	};

	class heap_manager
	{
	public:
		explicit heap_manager(void* device_handle);
		~heap_manager();

		heap_manager(const heap_manager&) = delete;
		heap_manager& operator=(const heap_manager&) = delete;

		heap_buffer_allocation allocate_private_buffer(u64 size, u64 resource_options, const std::string& label);
		heap_texture_allocation allocate_private_texture(void* texture_descriptor_handle, const std::string& label);
		void track_resource_use(command_frame& frame, void* resource_handle);
		void retire_resource(void* resource_handle);
		heap_manager_stats stats() const;
		void report() const;

	private:
		struct heap_manager_impl;
		std::unique_ptr<heap_manager_impl> m_impl;
	};
}
