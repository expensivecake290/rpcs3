#pragma once

#include <memory>
#include <string>

#include "memory.h"
#include "unique_resource.h"

namespace mtl
{
	enum buffer_usage : u32
	{
		buffer_usage_none = 0,
		buffer_usage_vertex = 1 << 0,
		buffer_usage_index = 1 << 1,
		buffer_usage_constant = 1 << 2,
		buffer_usage_storage = 1 << 3,
		buffer_usage_indirect = 1 << 4,
		buffer_usage_texture_view = 1 << 5,
		buffer_usage_copy_source = 1 << 6,
		buffer_usage_copy_destination = 1 << 7,
		buffer_usage_query = 1 << 8,
	};

	struct buffer_create_info
	{
		u64 size = 0;
		u32 usage = buffer_usage_none;
		storage_mode storage = storage_mode::automatic;
		cpu_cache_mode cache = cpu_cache_mode::default_cache;
		cpu_access access = cpu_access::none;
		hazard_tracking hazards = hazard_tracking::tracked;
		allocation_pool pool = allocation_pool::system;
		std::string label;
		bool use_placement_heap = true;
		bool allow_failure = false;
		bool recover_on_failure = true;
	};

	class buffer : public unique_resource
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		buffer();
		buffer(memory_allocator& allocator, const buffer_create_info& info);
		~buffer();
		buffer(const buffer&) = delete;
		buffer& operator=(const buffer&) = delete;
		buffer(buffer&&) = delete;
		buffer& operator=(buffer&&) = delete;

		void create(memory_allocator& allocator, const buffer_create_info& info);
		void create_no_copy(const render_device& device, void* host_pointer, u64 size, u32 usage, std::string_view label);
		void destroy();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] buffer_handle native_handle() const;
		[[nodiscard]] u64 size() const;
		[[nodiscard]] u64 gpu_address() const;
		[[nodiscard]] u32 usage() const;
		[[nodiscard]] storage_mode storage() const;
		[[nodiscard]] bool is_cpu_visible() const;
		[[nodiscard]] bool in_range(u64 offset, u64 length) const;

		[[nodiscard]] void* map(u64 offset, u64 length);
		void unmap();
		void did_modify(u64 offset, u64 length);
		[[nodiscard]] const memory_allocation& allocation() const;
	};

	class buffer_view : public unique_resource
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		buffer_view();
		buffer_view(const buffer& source, u64 pixel_format, u64 offset, u64 size, u32 bytes_per_element);
		~buffer_view();
		buffer_view(const buffer_view&) = delete;
		buffer_view& operator=(const buffer_view&) = delete;
		buffer_view(buffer_view&&) = delete;
		buffer_view& operator=(buffer_view&&) = delete;

		void create(const buffer& source, u64 pixel_format, u64 offset, u64 size, u32 bytes_per_element);
		void destroy();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] texture_handle native_handle() const;
		[[nodiscard]] u64 source_uid() const;
		[[nodiscard]] u64 pixel_format() const;
		[[nodiscard]] u64 offset() const;
		[[nodiscard]] u64 size() const;
		[[nodiscard]] u32 element_count() const;
		[[nodiscard]] bool in_range(u64 address, u64 length, u64& relative_offset) const;
	};
}
