#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <type_traits>

#include "Utilities/StrFmt.h"
#include "buffer_object.h"
#include "commands.h"

namespace mtl
{
	enum data_heap_flag : u32
	{
		data_heap_default = 0,
		data_heap_low_latency = 1 << 0,
		data_heap_fixed_size = 1 << 1,
		data_heap_force_private_shadow = 1 << 2,
		data_heap_persistent_mapping = 1 << 3,
	};

	struct data_heap_create_info
	{
		u64 initial_size = 0;
		u64 maximum_size = 1024ull * 1024 * 1024;
		u64 growth_quantum = 64ull * 1024 * 1024;
		u64 guard_size = 64ull * 1024;
		u32 usage = buffer_usage_copy_source;
		u32 flags = data_heap_persistent_mapping;
		allocation_pool pool = allocation_pool::system;
		std::string label;
		std::function<void(u64 generation, u64 capacity)> growth_callback;
	};

	struct data_heap_slice
	{
		buffer_handle buffer = nullptr;
		u64 offset = 0;
		u64 size = 0;
		u64 gpu_address = 0;
		void* cpu_address = nullptr;
		u64 generation = 0;
		bool shadowed = false;

		[[nodiscard]] explicit operator bool() const
		{
			return buffer && size != 0 && generation != 0;
		}

		[[nodiscard]] u64 buffer_gpu_address() const
		{
			return gpu_address >= offset ? gpu_address - offset : 0;
		}
	};

	struct data_heap_window
	{
		buffer_handle buffer = nullptr;
		u64 offset = 0;
		u64 length = 0;
		u64 gpu_address = 0;
		u64 generation = 0;

		[[nodiscard]] explicit operator bool() const
		{
			return buffer && length != 0 && generation != 0;
		}
	};

	struct data_heap_statistics
	{
		u64 capacity = 0;
		u64 allocated = 0;
		u64 peak_allocated = 0;
		u64 generation = 0;
		u64 growth_count = 0;
		u64 dirty_bytes = 0;
		u64 pending_batches = 0;
		u64 retired_generations = 0;
		bool shadowed = false;
	};

	class data_heap
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		data_heap();
		~data_heap();
		data_heap(const data_heap&) = delete;
		data_heap& operator=(const data_heap&) = delete;
		data_heap(data_heap&&) = delete;
		data_heap& operator=(data_heap&&) = delete;

		void create(memory_allocator& allocator, const data_heap_create_info& info);
		void destroy();

		[[nodiscard]] data_heap_slice allocate(u64 size, u64 alignment, bool allow_growth = true);
		[[nodiscard]] void* map(const data_heap_slice& slice, u64 relative_offset = 0, u64 size = 0);
		void mark_modified(const data_heap_slice& slice, u64 relative_offset = 0, u64 size = 0);
		void write(const data_heap_slice& slice, std::span<const std::byte> data, u64 relative_offset = 0);
		void unmap(bool force = false);

		void flush(command_buffer& command);
		void seal(u64 submission_value);
		void reclaim(u64 completed_submission_value);
		void trim(memory_pressure pressure);

		[[nodiscard]] data_heap_window window(const data_heap_slice& slice, u64 required_length,
			u64 window_size, u64 alignment) const;
		[[nodiscard]] const buffer& target_buffer() const;
		[[nodiscard]] const buffer* staging_buffer() const;
		[[nodiscard]] bool is_dirty() const;
		[[nodiscard]] bool has_shadow() const;
		[[nodiscard]] u64 size() const;
		[[nodiscard]] data_heap_statistics statistics() const;

		template <typename T = std::byte>
			requires std::is_trivially_destructible_v<T>
		[[nodiscard]] std::pair<data_heap_slice, T*> allocate_and_map(usz count, u64 alignment)
		{
			if (count > std::numeric_limits<u64>::max() / sizeof(T))
			{
				fmt::throw_exception("Metal data-heap allocation size overflows");
			}
			data_heap_slice slice = allocate(static_cast<u64>(count) * sizeof(T), alignment);
			return {slice, static_cast<T*>(map(slice))};
		}
	};
}
