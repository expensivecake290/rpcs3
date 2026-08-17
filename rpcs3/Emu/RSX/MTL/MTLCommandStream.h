#pragma once

#include <chrono>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "mtlutils/data_heap.h"
#include "mtlutils/descriptors.h"

namespace mtl
{
	enum class command_stream_mode : u8
	{
		inline_submission,
		worker_submission,
	};

	enum class command_enqueue_policy : u8
	{
		wait_for_space,
		reject_when_full,
	};

	struct command_stream_configuration
	{
		command_stream_mode mode = command_stream_mode::worker_submission;
		command_enqueue_policy enqueue_policy = command_enqueue_policy::wait_for_space;
		u32 maximum_queued_submissions = 64;
		queue_kind queue = queue_kind::graphics;
		bool serialize_native_submissions = true;
		std::string label = "RPCS3 Metal command stream";
	};

	struct argument_state_snapshot
	{
		argument_table* table = nullptr;
		u64 mutation_serial = 0;
		u64 applied_serial = 0;
	};

	struct command_stream_result_state;

	class command_stream_ticket
	{
		std::shared_ptr<command_stream_result_state> m_state;
		explicit command_stream_ticket(std::shared_ptr<command_stream_result_state> state);
		friend class command_stream;

	public:
		command_stream_ticket() = default;

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] u64 sequence() const;
		[[nodiscard]] bool issued() const;
		[[nodiscard]] bool completed() const;
		[[nodiscard]] bool succeeded() const;
		[[nodiscard]] bool wait_until_issued(std::chrono::nanoseconds timeout) const;
		void wait_until_issued() const;
		[[nodiscard]] bool wait_until_completed(std::chrono::nanoseconds timeout) const;
		void wait_until_completed() const;
		[[nodiscard]] submission native_submission() const;
		[[nodiscard]] std::exception_ptr failure() const;
		void rethrow_if_failed() const;
	};

	struct command_stream_packet
	{
		u64 sequence = 0;
		command_buffer* commands = nullptr;
		submit_info submission_info;
		std::vector<argument_state_snapshot> argument_state;
		std::vector<data_heap*> heaps_to_seal;
		std::shared_ptr<command_stream_result_state> result;
		bool serialize_submission = true;

		command_stream_packet() = default;
		command_stream_packet(const command_stream_packet&) = delete;
		command_stream_packet& operator=(const command_stream_packet&) = delete;
		command_stream_packet(command_stream_packet&&) noexcept = default;
		command_stream_packet& operator=(command_stream_packet&&) noexcept = default;

		[[nodiscard]] explicit operator bool() const;
	};

	struct command_stream_statistics
	{
		u64 generation = 0;
		u64 enqueued = 0;
		u64 issued = 0;
		u64 completed = 0;
		u64 failed = 0;
		u64 rejected = 0;
		u64 queue_waits = 0;
		u64 maximum_queue_depth = 0;
		u64 last_sequence = 0;
		u64 last_submission = 0;
		u32 queued = 0;
		u32 in_flight = 0;
		bool worker_active = false;
	};

	class command_stream
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		command_stream();
		~command_stream();
		command_stream(const command_stream&) = delete;
		command_stream& operator=(const command_stream&) = delete;

		void initialize(const command_stream_configuration& configuration = {});
		void shutdown(bool wait_for_gpu_completion = true);

		[[nodiscard]] command_stream_ticket submit(command_buffer& commands, submit_info info = {},
			std::span<argument_table* const> argument_tables = {},
			std::span<data_heap* const> heaps_to_seal = {}, bool wait_for_issue = false);
		[[nodiscard]] command_stream_ticket enqueue(command_stream_packet packet, bool wait_for_issue = false);

		void collect_completed();
		[[nodiscard]] bool flush(std::chrono::nanoseconds timeout, bool wait_for_gpu_completion = false);
		void flush(bool wait_for_gpu_completion = false);

		[[nodiscard]] command_stream_statistics statistics() const;
		[[nodiscard]] explicit operator bool() const;
	};
}
