#pragma once

#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
	class command_frame;
	class device;

	struct heap_resource_usage
	{
		device* metal_device = nullptr;
		void* resource_handle = nullptr;
	};

	struct device_caps
	{
		std::string name;
		b8 metal4_supported = false;
		b8 metalfx_available = false;
		b8 residency_sets_supported = false;
		u32 frames_in_flight = 0;
		u32 max_argument_table_buffers = 0;
		u32 max_argument_table_textures = 0;
		u32 max_argument_table_samplers = 0;
		u64 sparse_tile_size_in_bytes = 0;
		u64 private_buffer_heap_size = 0;
		u64 private_buffer_heap_alignment = 0;
		u64 private_texture_heap_size = 0;
		u64 private_texture_heap_alignment = 0;
		b8 placement_sparse_supported = false;
	};

	struct device_buffer_allocation
	{
		void* buffer_handle = nullptr;
		void* residency_allocation_handle = nullptr;
		b8 heap_backed = false;
	};

	struct device_texture_allocation
	{
		void* texture_handle = nullptr;
		void* residency_allocation_handle = nullptr;
		b8 heap_backed = false;
	};

	b8 is_metal_supported();

	class device
	{
	public:
		device();
		~device();

		device(const device&) = delete;
		device& operator=(const device&) = delete;

		void* handle() const;
		void* residency_set_handle() const;
		device_buffer_allocation create_buffer_allocation(u64 size, u64 resource_options, const std::string& label);
		device_texture_allocation create_texture_allocation(void* texture_descriptor_handle, const std::string& label);
		void track_heap_resource_use(command_frame& frame, void* resource_handle);
		void retire_heap_resource(void* resource_handle);
		void add_resident_allocation(void* allocation_handle);
		void remove_resident_allocation(void* allocation_handle);
		void commit_residency();
		u64 residency_allocated_size() const;
		u32 residency_allocation_count() const;
		void report_memory_usage() const;

		const device_caps& caps() const;
		void report_capabilities() const;

	private:
		struct device_impl;
		std::unique_ptr<device_impl> m_impl;
	};
}
