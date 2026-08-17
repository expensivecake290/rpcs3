#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "mtlutils/sync.h"

namespace mtl
{
	inline constexpr u32 maximum_async_command_slots = 256;

	enum class async_scheduler_mode : u8
	{
		gpu_timeline,
		host_synchronized,
	};

	struct async_scheduler_configuration
	{
		async_scheduler_mode mode = async_scheduler_mode::gpu_timeline;
		u32 maximum_command_slots = 32;
		queue_kind queue = queue_kind::transfer;
		std::string label = "RPCS3 Metal asynchronous scheduler";
	};

	struct async_sync_token
	{
		shared_event_handle event = nullptr;
		u64 value = 0;
		u64 submission = 0;
		u64 scheduler_generation = 0;

		[[nodiscard]] explicit operator bool() const
		{
			return event && value != 0 && submission != 0 && scheduler_generation != 0;
		}

		[[nodiscard]] event_operation wait_operation() const;
	};

	struct async_submission
	{
		mtl::submission work;
		async_sync_token synchronization;

		[[nodiscard]] explicit operator bool() const
		{
			return work && synchronization;
		}

		[[nodiscard]] bool completed() const;
		[[nodiscard]] bool succeeded() const;
		[[nodiscard]] bool wait(std::chrono::nanoseconds timeout) const;
		void wait() const;
	};

	struct async_scheduler_statistics
	{
		u64 generation = 0;
		u64 acquired_command_buffers = 0;
		u64 submissions = 0;
		u64 completed_submissions = 0;
		u64 failed_submissions = 0;
		u64 forced_host_waits = 0;
		u64 pool_exhaustion_waits = 0;
		u64 timeline_value = 0;
		u32 command_slots = 0;
		u32 pending_slots = 0;
		bool recording = false;
	};

	class async_task_scheduler
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		async_task_scheduler();
		~async_task_scheduler();
		async_task_scheduler(const async_task_scheduler&) = delete;
		async_task_scheduler& operator=(const async_task_scheduler&) = delete;

		void initialize(render_device& device, const async_scheduler_configuration& configuration = {});
		void destroy(bool device_is_idle = false);

		[[nodiscard]] command_buffer& get_current();
		[[nodiscard]] bool is_recording() const;
		[[nodiscard]] bool is_host_synchronized() const;

		[[nodiscard]] async_submission flush(submit_info info = {}, bool force_flush = false);
		[[nodiscard]] async_sync_token latest_sync_token() const;
		[[nodiscard]] event_operation graphics_wait_operation() const;

		void reclaim_completed();
		[[nodiscard]] bool wait_idle(std::chrono::nanoseconds timeout);
		void wait_idle();

		[[nodiscard]] async_scheduler_statistics statistics() const;
		[[nodiscard]] explicit operator bool() const;
	};
}
