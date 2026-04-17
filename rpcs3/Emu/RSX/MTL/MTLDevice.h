#pragma once

#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
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
		void add_resident_allocation(void* allocation_handle);
		void remove_resident_allocation(void* allocation_handle);
		void commit_residency();
		u64 residency_allocated_size() const;
		u32 residency_allocation_count() const;

		const device_caps& caps() const;
		void report_capabilities() const;

	private:
		struct device_impl;
		std::unique_ptr<device_impl> m_impl;
	};
}
