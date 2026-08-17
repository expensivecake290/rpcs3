#pragma once

#include <memory>
#include <string_view>

#include "device.h"
#include "garbage_collector.h"
#include "memory.h"

namespace mtl
{
	class shared_state
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		shared_state();
		~shared_state();
		shared_state(const shared_state&) = delete;
		shared_state& operator=(const shared_state&) = delete;

		void initialize(std::string_view preferred_device);
		void shutdown();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] render_device& device();
		[[nodiscard]] const render_device& device() const;
		[[nodiscard]] memory_allocator& allocator();
		[[nodiscard]] residency_set& residency();
		[[nodiscard]] garbage_collector& garbage();

		[[nodiscard]] u64 begin_frame();
		[[nodiscard]] u64 next_submission();
		void notify_submission_completed(u64 value);

		[[nodiscard]] u64 current_frame() const;
		[[nodiscard]] u64 current_submission() const;
		[[nodiscard]] u64 completed_submission() const;
	};

	[[nodiscard]] shared_state& get_shared_state();
	[[nodiscard]] u64 get_frame_id();
	[[nodiscard]] u64 get_submission_id();
	[[nodiscard]] u64 get_completed_submission_id();
}
