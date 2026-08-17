#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "commands.h"

namespace mtl
{
	enum class sync_domain : u8
	{
		host,
		gpu,
	};

	class timeline_event
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		timeline_event();
		~timeline_event();
		timeline_event(const timeline_event&) = delete;
		timeline_event& operator=(const timeline_event&) = delete;
		timeline_event(timeline_event&&) noexcept;
		timeline_event& operator=(timeline_event&&) noexcept;

		void create(const render_device& device, sync_domain domain, std::string_view label, u64 initial_value = 0);
		void destroy();
		void signal_host(u64 value);

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] shared_event_handle native_handle() const;
		[[nodiscard]] u64 signaled_value() const;
		[[nodiscard]] bool reached(u64 value) const;
		[[nodiscard]] bool wait(u64 value, std::chrono::nanoseconds timeout) const;
		void wait(u64 value) const;
		[[nodiscard]] event_operation wait_operation(u64 value) const;
		[[nodiscard]] event_operation signal_operation(u64 value) const;
	};

	class fence
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		fence();
		~fence();
		fence(const fence&) = delete;
		fence& operator=(const fence&) = delete;
		fence(fence&&) noexcept;
		fence& operator=(fence&&) noexcept;

		void create(const render_device& device, std::string_view label);
		void destroy();
		void update(native_encoder_handle encoder);
		void wait(native_encoder_handle encoder) const;

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] void* native_handle() const;
	};

	class submission_fence
	{
		submission m_submission;

	public:
		void reset();
		void signal(submission value);
		[[nodiscard]] bool signaled() const;
		[[nodiscard]] bool wait(std::chrono::nanoseconds timeout) const;
		void wait() const;
		[[nodiscard]] u64 value() const;
	};

	class debug_marker_scope
	{
		command_buffer* m_commands = nullptr;

	public:
		debug_marker_scope(command_buffer& commands, std::string_view message);
		~debug_marker_scope();
		debug_marker_scope(const debug_marker_scope&) = delete;
		debug_marker_scope& operator=(const debug_marker_scope&) = delete;
	};

	[[nodiscard]] bool wait_for_submission(const submission& value, std::chrono::nanoseconds timeout);
	void wait_for_submission(const submission& value);
}
