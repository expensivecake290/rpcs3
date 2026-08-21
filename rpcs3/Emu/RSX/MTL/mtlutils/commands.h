#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "device.h"

namespace mtl
{
	enum class queue_kind : u8
	{
		graphics,
		transfer,
	};

	enum class encoder_kind : u8
	{
		none,
		render,
		compute,
	};

	enum command_buffer_flag : u32
	{
		command_has_occlusion_task = 0x01,
		command_has_blit_transfer = 0x02,
		command_has_dma_transfer = 0x04,
		command_has_open_query = 0x08,
		command_loads_occlusion_task = 0x10,
		command_has_conditional_render = 0x20,
		command_reload_dynamic_state = 0x40,
	};

	using native_encoder_handle = void*;
	using native_render_pass_descriptor = void*;

	struct event_operation
	{
		shared_event_handle event = nullptr;
		u64 value = 0;
	};

	struct submit_info
	{
		queue_kind queue = queue_kind::graphics;
		std::vector<event_operation> waits;
		std::vector<event_operation> signal_operations;
		bool wait_for_completion = false;
	};

	class submission
	{
		struct impl;
		std::shared_ptr<impl> m_impl;

		explicit submission(std::shared_ptr<impl> implementation);
		friend class command_buffer;

	public:
		submission() = default;

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] u64 value() const;
		[[nodiscard]] bool completed() const;
		[[nodiscard]] bool succeeded() const;
		[[nodiscard]] error failure() const;
		[[nodiscard]] f64 gpu_start_time() const;
		[[nodiscard]] f64 gpu_end_time() const;
		[[nodiscard]] bool wait(std::chrono::nanoseconds timeout) const;
		void wait() const;
	};

	class command_allocator
	{
		struct impl;
		std::unique_ptr<impl> m_impl;
		friend class command_buffer;

	public:
		command_allocator();
		~command_allocator();
		command_allocator(const command_allocator&) = delete;
		command_allocator& operator=(const command_allocator&) = delete;
		command_allocator(command_allocator&&) = delete;
		command_allocator& operator=(command_allocator&&) = delete;

		void create(render_device& device, std::string_view label);
		void destroy();
		void reset();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] render_device& owner() const;
		[[nodiscard]] command_allocator_handle native_handle() const;
		[[nodiscard]] u64 allocated_size() const;
	};

	class command_buffer
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		command_buffer();
		~command_buffer();
		command_buffer(const command_buffer&) = delete;
		command_buffer& operator=(const command_buffer&) = delete;
		command_buffer(command_buffer&&) noexcept;
		command_buffer& operator=(command_buffer&&) noexcept;

		void create(command_allocator& allocator, std::string_view label);
		void destroy();
		void begin();
		void end();
		void discard();

		[[nodiscard]] native_encoder_handle begin_render_encoding(native_render_pass_descriptor descriptor);
		[[nodiscard]] native_encoder_handle begin_compute_encoding();
		void end_encoding();
		void push_debug_group(std::string_view label);
		void pop_debug_group();
		void retain_native_object(void* object, bool make_resident = false);
		void notify_on_completion(std::function<void(bool)> callback);

		[[nodiscard]] submission submit(const submit_info& info);
		void reset_after_completion();

		void clear_flags();
		void set_flag(command_buffer_flag flag);
		void clear_flag(command_buffer_flag flag);
		[[nodiscard]] bool has_flag(command_buffer_flag flag) const;

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] bool is_recording() const;
		[[nodiscard]] bool is_pending() const;
		[[nodiscard]] encoder_kind active_encoder() const;
		[[nodiscard]] native_encoder_handle active_native_encoder() const;
		[[nodiscard]] command_buffer_handle native_handle() const;
		[[nodiscard]] command_allocator& allocator() const;
		[[nodiscard]] submission last_submission() const;
	};
}
