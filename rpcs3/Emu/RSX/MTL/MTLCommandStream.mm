#include "stdafx.h"
#include "MTLCommandStream.h"

#include "MTLHelpers.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace mtl
{
	struct command_stream_result_state
	{
		mutable std::mutex mutex;
		std::condition_variable condition;
		u64 sequence = 0;
		submission work;
		std::exception_ptr issue_failure;
		bool issue_finished = false;
		bool completion_accounted = false;
		bool failure_accounted = false;
	};

	namespace
	{
		std::atomic<u64> g_stream_generation = 0;

		u64 saturated_add(u64 first, u64 second)
		{
			return first > std::numeric_limits<u64>::max() - second ?
				std::numeric_limits<u64>::max() : first + second;
		}

		u64 allocate_generation()
		{
			u64 current = g_stream_generation.load(std::memory_order_relaxed);
			for (;;)
			{
				if (current == std::numeric_limits<u64>::max())
				{
					fmt::throw_exception("Metal command-stream generation overflowed");
				}
				if (g_stream_generation.compare_exchange_weak(current, current + 1, std::memory_order_acq_rel))
				{
					return current + 1;
				}
			}
		}

		std::exception_ptr gpu_failure(const submission& work)
		{
			if (!work || !work.completed() || work.succeeded())
			{
				return {};
			}
			const error failure = work.failure();
			return std::make_exception_ptr(std::runtime_error(
				fmt::format("%s: %s", failure.domain, failure.description)));
		}
	}

	command_stream_ticket::command_stream_ticket(std::shared_ptr<command_stream_result_state> state)
		: m_state(std::move(state))
	{
	}

	command_stream_ticket::operator bool() const
	{
		return m_state != nullptr;
	}

	u64 command_stream_ticket::sequence() const
	{
		if (!m_state) return 0;
		std::lock_guard lock(m_state->mutex);
		return m_state->sequence;
	}

	bool command_stream_ticket::issued() const
	{
		if (!m_state) return false;
		std::lock_guard lock(m_state->mutex);
		return m_state->issue_finished;
	}

	bool command_stream_ticket::completed() const
	{
		if (!m_state) return false;
		std::lock_guard lock(m_state->mutex);
		return m_state->issue_finished && (!m_state->work || m_state->work.completed());
	}

	bool command_stream_ticket::succeeded() const
	{
		if (!m_state) return false;
		std::lock_guard lock(m_state->mutex);
		return m_state->issue_finished && !m_state->issue_failure &&
			m_state->work && m_state->work.succeeded();
	}

	bool command_stream_ticket::wait_until_issued(std::chrono::nanoseconds timeout) const
	{
		if (!m_state || timeout < std::chrono::nanoseconds::zero()) return false;
		std::unique_lock lock(m_state->mutex);
		return m_state->condition.wait_for(lock, timeout, [&] { return m_state->issue_finished; });
	}

	void command_stream_ticket::wait_until_issued() const
	{
		if (!m_state)
		{
			fmt::throw_exception("Cannot wait on an empty Metal command-stream ticket");
		}
		std::unique_lock lock(m_state->mutex);
		m_state->condition.wait(lock, [&] { return m_state->issue_finished; });
	}

	bool command_stream_ticket::wait_until_completed(std::chrono::nanoseconds timeout) const
	{
		if (!m_state || timeout < std::chrono::nanoseconds::zero()) return false;
		const auto deadline = std::chrono::steady_clock::now() + timeout;
		std::unique_lock lock(m_state->mutex);
		if (!m_state->condition.wait_until(lock, deadline, [&] { return m_state->issue_finished; }))
		{
			return false;
		}
		const submission work = m_state->work;
		lock.unlock();
		if (!work) return true;
		if (work.completed()) return true;
		const auto now = std::chrono::steady_clock::now();
		return now < deadline && work.wait(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now));
	}

	void command_stream_ticket::wait_until_completed() const
	{
		wait_until_issued();
		const submission work = native_submission();
		if (work) work.wait();
	}

	submission command_stream_ticket::native_submission() const
	{
		if (!m_state)
		{
			return {};
		}
		std::lock_guard lock(m_state->mutex);
		if (!m_state->issue_finished)
		{
			fmt::throw_exception("Metal command-stream submission has not been issued yet");
		}
		return m_state->work;
	}

	std::exception_ptr command_stream_ticket::failure() const
	{
		if (!m_state) return {};
		std::lock_guard lock(m_state->mutex);
		if (m_state->issue_failure) return m_state->issue_failure;
		return gpu_failure(m_state->work);
	}

	void command_stream_ticket::rethrow_if_failed() const
	{
		if (const std::exception_ptr value = failure()) std::rethrow_exception(value);
	}

	command_stream_packet::operator bool() const
	{
		return commands != nullptr;
	}

	struct command_stream::impl
	{
		mutable std::mutex mutex;
		std::condition_variable work_available;
		std::condition_variable space_available;
		std::condition_variable idle;
		command_stream_configuration configuration;
		std::deque<command_stream_packet> queue;
		std::vector<std::shared_ptr<command_stream_result_state>> in_flight;
		std::thread worker;
		command_stream_statistics stats;
		u64 next_sequence = 1;
		u32 active_processors = 0;
		bool initialized = false;
		bool accepting = false;
		bool stopping = false;

		void finish_issue(const std::shared_ptr<command_stream_result_state>& result,
			submission work, std::exception_ptr failure)
		{
			{
				std::lock_guard state_lock(result->mutex);
				result->work = work;
				result->issue_failure = failure;
				result->issue_finished = true;
				result->failure_accounted = failure != nullptr;
			}
			result->condition.notify_all();

			std::lock_guard lock(mutex);
			stats.issued = saturated_add(stats.issued, 1);
			if (failure) stats.failed = saturated_add(stats.failed, 1);
			if (work)
			{
				stats.last_submission = work.value();
				in_flight.push_back(result);
			}
			ensure(active_processors != 0);
			--active_processors;
			if (queue.empty() && active_processors == 0) idle.notify_all();
			space_available.notify_all();
		}

		void process_packet(command_stream_packet packet)
		{
			submission work;
			std::exception_ptr failure;
			try
			{
				for (data_heap* heap : packet.heaps_to_seal)
				{
					if (!heap) fmt::throw_exception("Metal command stream contains a null data heap");
				}
				for (const argument_state_snapshot& snapshot : packet.argument_state)
				{
					if (!snapshot.table || !*snapshot.table || snapshot.table->dirty() ||
						snapshot.table->mutation_serial() != snapshot.mutation_serial ||
						snapshot.table->applied_serial() != snapshot.applied_serial ||
						snapshot.mutation_serial != snapshot.applied_serial)
					{
						fmt::throw_exception("Metal command-stream argument state changed after encoding");
					}
				}
				packet.submission_info.queue = configuration.queue;
				work = packet.serialize_submission ? submit_serialized(*packet.commands, packet.submission_info) :
					packet.commands->submit(packet.submission_info);
				try
				{
					for (data_heap* heap : packet.heaps_to_seal)
					{
						heap->seal(work.value());
					}
				}
				catch (...)
				{
					failure = std::current_exception();
				}
			}
			catch (...)
			{
				failure = std::current_exception();
			}
			finish_issue(packet.result, work, failure);
		}

		void worker_main()
		{
			for (;;)
			{
				command_stream_packet packet;
				{
					std::unique_lock lock(mutex);
					work_available.wait(lock, [&] { return stopping || !queue.empty(); });
					if (queue.empty() && stopping) break;
					packet = std::move(queue.front());
					queue.pop_front();
					++active_processors;
					stats.queued = queue.size();
					space_available.notify_all();
				}
				@autoreleasepool
				{
					process_packet(std::move(packet));
				}
			}
			std::lock_guard lock(mutex);
			stats.worker_active = false;
			idle.notify_all();
		}
	};

	command_stream::command_stream()
		: m_impl(std::make_unique<impl>())
	{
	}

	command_stream::~command_stream()
	{
		try
		{
			shutdown(true);
		}
		catch (...)
		{
		}
	}

	void command_stream::initialize(const command_stream_configuration& configuration)
	{
		shutdown(true);
		if (configuration.maximum_queued_submissions == 0 || configuration.label.empty())
		{
			fmt::throw_exception("Invalid Metal command-stream configuration");
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->configuration = configuration;
		m_impl->queue.clear();
		m_impl->in_flight.clear();
		m_impl->stats = {};
		m_impl->stats.generation = allocate_generation();
		m_impl->next_sequence = 1;
		m_impl->active_processors = 0;
		m_impl->initialized = true;
		m_impl->accepting = true;
		m_impl->stopping = false;
		if (configuration.mode == command_stream_mode::worker_submission)
		{
			try
			{
				m_impl->worker = std::thread([state = m_impl.get()] { state->worker_main(); });
				m_impl->stats.worker_active = true;
			}
			catch (...)
			{
				m_impl->initialized = false;
				m_impl->accepting = false;
				throw;
			}
		}
	}

	void command_stream::shutdown(bool wait_for_gpu_completion)
	{
		std::thread worker;
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->initialized) return;
			if (m_impl->worker.joinable() && m_impl->worker.get_id() == std::this_thread::get_id())
			{
				fmt::throw_exception("Metal command stream cannot shut down from its submission worker");
			}
			m_impl->accepting = false;
			m_impl->stopping = true;
			m_impl->work_available.notify_all();
			m_impl->space_available.notify_all();
			worker = std::move(m_impl->worker);
		}
		if (worker.joinable()) worker.join();
		flush(wait_for_gpu_completion);
		std::lock_guard lock(m_impl->mutex);
		m_impl->queue.clear();
		m_impl->in_flight.clear();
		m_impl->initialized = false;
		m_impl->stopping = false;
		m_impl->stats.queued = 0;
		m_impl->stats.in_flight = 0;
	}

	command_stream_ticket command_stream::submit(command_buffer& commands, submit_info info,
		std::span<argument_table* const> argument_tables,
		std::span<data_heap* const> heaps_to_seal, bool wait_for_issue)
	{
		command_stream_packet packet;
		packet.commands = &commands;
		packet.submission_info = std::move(info);
		packet.argument_state.reserve(argument_tables.size());
		for (argument_table* table : argument_tables)
		{
			if (!table || !*table || table->dirty() || table->mutation_serial() != table->applied_serial())
			{
				fmt::throw_exception("Metal command stream requires applied argument-table state");
			}
			packet.argument_state.push_back({table, table->mutation_serial(), table->applied_serial()});
		}
		packet.heaps_to_seal.assign(heaps_to_seal.begin(), heaps_to_seal.end());
		return enqueue(std::move(packet), wait_for_issue);
	}

	command_stream_ticket command_stream::enqueue(command_stream_packet packet, bool wait_for_issue)
	{
		if (!packet)
		{
			fmt::throw_exception("Invalid Metal command-stream packet");
		}
		auto result = std::make_shared<command_stream_result_state>();
		command_stream_ticket ticket(result);
		bool process_inline = false;
		{
			std::unique_lock lock(m_impl->mutex);
			if (!m_impl->initialized || !m_impl->accepting)
			{
				fmt::throw_exception("Metal command stream is not accepting submissions");
			}
			process_inline = m_impl->configuration.mode == command_stream_mode::inline_submission;
			if (!process_inline)
			{
				if (m_impl->queue.size() >= m_impl->configuration.maximum_queued_submissions)
				{
					if (m_impl->configuration.enqueue_policy == command_enqueue_policy::reject_when_full)
					{
						m_impl->stats.rejected = saturated_add(m_impl->stats.rejected, 1);
						fmt::throw_exception("Metal command-stream queue is full");
					}
					m_impl->stats.queue_waits = saturated_add(m_impl->stats.queue_waits, 1);
					m_impl->space_available.wait(lock, [&]
					{
						return !m_impl->accepting ||
							m_impl->queue.size() < m_impl->configuration.maximum_queued_submissions;
					});
					if (!m_impl->accepting)
					{
						fmt::throw_exception("Metal command stream stopped while waiting for queue space");
					}
				}
			}
			if (!m_impl->next_sequence)
			{
				fmt::throw_exception("Metal command-stream sequence overflowed");
			}
			packet.sequence = m_impl->next_sequence++;
			result->sequence = packet.sequence;
			packet.result = result;
			packet.serialize_submission = m_impl->configuration.serialize_native_submissions;
			m_impl->stats.enqueued = saturated_add(m_impl->stats.enqueued, 1);
			m_impl->stats.last_sequence = packet.sequence;
			if (process_inline)
			{
				++m_impl->active_processors;
			}
			else
			{
				m_impl->queue.push_back(std::move(packet));
				m_impl->stats.queued = m_impl->queue.size();
				m_impl->stats.maximum_queue_depth = std::max<u64>(m_impl->stats.maximum_queue_depth, m_impl->queue.size());
				m_impl->work_available.notify_one();
			}
		}
		if (process_inline)
		{
			@autoreleasepool
			{
				m_impl->process_packet(std::move(packet));
			}
		}
		if (wait_for_issue) ticket.wait_until_issued();
		return ticket;
	}

	void command_stream::collect_completed()
	{
		std::lock_guard lock(m_impl->mutex);
		for (auto iterator = m_impl->in_flight.begin(); iterator != m_impl->in_flight.end();)
		{
			auto& state = **iterator;
			std::lock_guard state_lock(state.mutex);
			if (!state.work.completed())
			{
				++iterator;
				continue;
			}
			if (!state.completion_accounted)
			{
				m_impl->stats.completed = saturated_add(m_impl->stats.completed, 1);
				if (!state.work.succeeded() && !state.failure_accounted)
				{
					m_impl->stats.failed = saturated_add(m_impl->stats.failed, 1);
					state.failure_accounted = true;
				}
				state.completion_accounted = true;
			}
			iterator = m_impl->in_flight.erase(iterator);
		}
		m_impl->stats.in_flight = m_impl->in_flight.size();
	}

	bool command_stream::flush(std::chrono::nanoseconds timeout, bool wait_for_gpu_completion)
	{
		if (timeout < std::chrono::nanoseconds::zero()) return false;
		const auto deadline = std::chrono::steady_clock::now() + timeout;
		std::vector<std::shared_ptr<command_stream_result_state>> in_flight;
		{
			std::unique_lock lock(m_impl->mutex);
			if (!m_impl->initialized) return true;
			if (!m_impl->idle.wait_until(lock, deadline, [&]
				{ return m_impl->queue.empty() && m_impl->active_processors == 0; }))
			{
				return false;
			}
			if (wait_for_gpu_completion) in_flight = m_impl->in_flight;
		}
		for (const auto& state : in_flight)
		{
			command_stream_ticket ticket(state);
			const auto now = std::chrono::steady_clock::now();
			if (now >= deadline ||
				!ticket.wait_until_completed(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now)))
			{
				return false;
			}
		}
		collect_completed();
		return true;
	}

	void command_stream::flush(bool wait_for_gpu_completion)
	{
		std::vector<std::shared_ptr<command_stream_result_state>> in_flight;
		{
			std::unique_lock lock(m_impl->mutex);
			if (!m_impl->initialized) return;
			m_impl->idle.wait(lock, [&] { return m_impl->queue.empty() && m_impl->active_processors == 0; });
			if (wait_for_gpu_completion) in_flight = m_impl->in_flight;
		}
		for (const auto& state : in_flight) command_stream_ticket(state).wait_until_completed();
		collect_completed();
	}

	command_stream_statistics command_stream::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		command_stream_statistics result = m_impl->stats;
		result.queued = m_impl->queue.size();
		result.in_flight = m_impl->in_flight.size();
		return result;
	}

	command_stream::operator bool() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->initialized;
	}
}
