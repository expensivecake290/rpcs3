#pragma once

#include "util/types.hpp"

#include <memory>
#include <string_view>

namespace rsx::metal
{
	class residency_manager
	{
	public:
		residency_manager(void* device_handle, std::string_view label, u32 initial_capacity);
		~residency_manager();

		residency_manager(const residency_manager&) = delete;
		residency_manager& operator=(const residency_manager&) = delete;

		void request_residency();
		void end_residency();
		void add_allocation(void* allocation_handle);
		void remove_allocation(void* allocation_handle);
		void commit();

		void* handle() const;
		u64 allocated_size() const;
		u32 allocation_count() const;

	private:
		struct residency_manager_impl;
		std::unique_ptr<residency_manager_impl> m_impl;
	};
}
