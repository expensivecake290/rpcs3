#pragma once

#include <memory>
#include <span>

#include "MTLRenderPass.h"
#include "mtlutils/query_pool.hpp"

namespace mtl
{
	struct query_pool_manager_create_info
	{
		u32 capacity = 0;
		query_kind kind = query_kind::occlusion_counting;
		allocation_pool allocation = allocation_pool::system;
		std::string label;
		bool cpu_readback = true;
		bool allow_partial_results = true;
	};

	struct managed_query_status
	{
		query_state state = query_state::free;
		query_availability availability = query_availability::unavailable;
		u64 value = 0;
		u64 submission = 0;
		bool logically_active = false;
		bool visibility_enabled = false;

		[[nodiscard]] bool ready() const
		{
			return availability == query_availability::available ||
				availability == query_availability::failed;
		}
	};

	struct query_pool_manager_statistics
	{
		query_pool_statistics pool;
		u32 logically_active = 0;
		u32 suspended = 0;
		u64 allocations = 0;
		u64 releases = 0;
		u64 begins = 0;
		u64 resumes = 0;
		u64 suspends = 0;
		u64 ends = 0;
		u64 timestamp_writes = 0;
		u64 resolves = 0;
		u64 indirect_copies = 0;
	};

	class query_pool_manager final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		query_pool_manager();
		query_pool_manager(render_device& device, memory_allocator& allocator,
			const query_pool_manager_create_info& info);
		~query_pool_manager();
		query_pool_manager(const query_pool_manager&) = delete;
		query_pool_manager& operator=(const query_pool_manager&) = delete;
		query_pool_manager(query_pool_manager&&) = delete;
		query_pool_manager& operator=(query_pool_manager&&) = delete;

		void create(render_device& device, memory_allocator& allocator,
			const query_pool_manager_create_info& info);
		void destroy();

		[[nodiscard]] query_handle allocate_query();
		void release_query(query_handle query);
		void reset_query(query_handle query);

		void configure_render_pass(render_pass_configuration& configuration,
			visibility_result_behavior behavior = visibility_result_behavior::reset) const;
		void begin_query(render_pass& pass, query_handle query, bool accumulate = false);
		void suspend_query(render_pass& pass, query_handle query);
		void resume_query(render_pass& pass, query_handle query);
		void end_query(render_pass& pass, query_handle query);
		void write_timestamp(command_buffer& command, query_handle query, u64 stages = stage_all_gpu);
		void resolve_timestamps(command_buffer& command, u32 first, u32 count);

		void mark_submitted(query_handle query, u64 submission_value);
		void mark_submitted(std::span<const query_handle> queries, u64 submission_value);
		void notify_completed(u64 completed_submission_value, bool submission_succeeded = true);

		[[nodiscard]] managed_query_status check_query_status(query_handle query) const;
		[[nodiscard]] query_result get_query_result(query_handle query, bool allow_partial = false) const;
		void copy_query_result(render_pass* active_pass, command_buffer& command,
			query_handle query, buffer& destination, u64 destination_offset,
			bool sixty_four_bit = true, bool include_availability = false);

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] query_kind kind() const;
		[[nodiscard]] u32 capacity() const;
		[[nodiscard]] const buffer* visibility_buffer() const;
		[[nodiscard]] query_handle active_query() const;
		[[nodiscard]] bool is_suspended() const;
		[[nodiscard]] query_pool_manager_statistics statistics() const;
	};
}
