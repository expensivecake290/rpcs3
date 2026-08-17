#pragma once

#include <memory>
#include <string>

#include "buffer_object.h"
#include "commands.h"

namespace mtl
{
	inline constexpr u64 visibility_result_stride = sizeof(u64);

	enum class query_kind : u8
	{
		occlusion_boolean,
		occlusion_counting,
		timestamp,
	};

	enum class query_state : u8
	{
		free,
		allocated,
		active,
		ended,
		pending,
		ready,
	};

	enum class query_availability : u8
	{
		unavailable,
		partial,
		available,
		failed,
	};

	struct query_pool_create_info
	{
		query_kind kind = query_kind::occlusion_boolean;
		u32 capacity = 0;
		allocation_pool pool = allocation_pool::system;
		std::string label;
		bool cpu_readback = true;
	};

	struct query_handle
	{
		u32 index = umax;
		u32 generation = 0;

		[[nodiscard]] explicit operator bool() const
		{
			return index != umax && generation != 0;
		}

		[[nodiscard]] bool operator==(const query_handle&) const = default;
	};

	struct query_result
	{
		u64 value = 0;
		query_availability availability = query_availability::unavailable;
		u64 submission = 0;

		[[nodiscard]] explicit operator bool() const
		{
			return availability == query_availability::available ||
				availability == query_availability::partial;
		}
	};

	struct query_pool_statistics
	{
		u32 capacity = 0;
		u32 free_slots = 0;
		u32 allocated_slots = 0;
		u32 active_slots = 0;
		u32 pending_slots = 0;
		u32 ready_slots = 0;
		u64 last_submission = 0;
		u64 result_stride = 0;
	};

	class query_pool
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		query_pool();
		query_pool(render_device& device, memory_allocator& allocator, const query_pool_create_info& info);
		~query_pool();
		query_pool(const query_pool&) = delete;
		query_pool& operator=(const query_pool&) = delete;
		query_pool(query_pool&&) = delete;
		query_pool& operator=(query_pool&&) = delete;

		void create(render_device& device, memory_allocator& allocator, const query_pool_create_info& info);
		void destroy();

		[[nodiscard]] query_handle allocate();
		void release(query_handle query);
		void reset(query_handle query);
		void reset(u32 first, u32 count);

		void begin(command_buffer& command, query_handle query, bool accumulate = false);
		void end(command_buffer& command, query_handle query);
		void write_timestamp(command_buffer& command, query_handle query, u64 stages);
		void resolve(command_buffer& command, u32 first, u32 count);
		void mark_submitted(query_handle query, u64 submission_value);
		void notify_completed(u64 completed_submission_value, bool submission_succeeded = true);

		[[nodiscard]] query_state state(query_handle query) const;
		[[nodiscard]] bool available(query_handle query) const;
		[[nodiscard]] query_result result(query_handle query, bool allow_partial = false) const;
		void copy_result(command_buffer& command, query_handle query, buffer& destination,
			u64 destination_offset, bool sixty_four_bit = true, bool include_availability = false);

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] query_kind kind() const;
		[[nodiscard]] u32 size() const;
		[[nodiscard]] u64 result_stride() const;
		[[nodiscard]] const buffer* visibility_buffer() const;
		[[nodiscard]] const buffer* resolved_counter_buffer() const;
		[[nodiscard]] counter_heap_handle native_counter_heap() const;
		[[nodiscard]] u64 visibility_offset(query_handle query) const;
		[[nodiscard]] query_pool_statistics statistics() const;
	};
}
