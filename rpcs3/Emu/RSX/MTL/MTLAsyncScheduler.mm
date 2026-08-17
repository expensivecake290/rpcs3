#include "stdafx.h"
#include "MTLAsyncScheduler.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <mutex>

namespace mtl
{
	namespace
	{
		std::atomic<u64> g_scheduler_generation = 0;

		u64 next_generation()
		{
			u64 current = g_scheduler_generation.load(std::memory_order_relaxed);
			for (;;)
			{
				if (current == std::numeric_limits<u64>::max())
				{
					fmt::throw_exception("Metal asynchronous scheduler generation overflowed");
				}
				if (g_scheduler_generation.compare_exchange_weak(current, current + 1, std::memory_order_acq_rel))
				{
					return current + 1;
				}
			}
		}
	}

	event_operation async_sync_token::wait_operation() const
	{
		if (!*this)
		{
			fmt::throw_exception("Cannot wait on an empty Metal asynchronous synchronization token");
		}
		return {event, value};
	}

	bool async_submission::completed() const
	{
		return work && work.completed();
	}

	bool async_submission::succeeded() const
	{
		return work && work.succeeded();
	}

	bool async_submission::wait(std::chrono::nanoseconds timeout) const
	{
		return work && work.wait(timeout);
	}

	void async_submission::wait() const
	{
		if (!work)
		{
			fmt::throw_exception("Cannot wait on an empty Metal asynchronous submission");
		}
		work.wait();
	}

	struct async_task_scheduler::impl
	{
		struct command_slot
		{
			std::unique_ptr<command_allocator> allocator;
			std::unique_ptr<command_buffer> commands;
			submission pending;
			u64 sequence = 0;
			u64 timeline_value = 0;
			bool needs_reconstruction = false;
		};

		mutable std::mutex mutex;
		render_device* device = nullptr;
		async_scheduler_configuration configuration;
		std::vector<std::unique_ptr<command_slot>> slots;
		timeline_event timeline;
		command_slot* current = nullptr;
		async_sync_token latest;
		async_scheduler_statistics stats;
		u64 next_sequence = 1;
		bool initialized = false;
		bool shutting_down = false;

		std::unique_ptr<command_slot> create_slot(usz index)
		{
			auto result = std::make_unique<command_slot>();
			result->allocator = std::make_unique<command_allocator>();
			result->commands = std::make_unique<command_buffer>();
			const std::string allocator_label = fmt::format("%s allocator %u", configuration.label, index);
			const std::string command_label = fmt::format("%s commands %u", configuration.label, index);
			result->allocator->create(*device, allocator_label);
			try
			{
				result->commands->create(*result->allocator, command_label);
			}
			catch (...)
			{
				result->allocator->destroy();
				throw;
			}
			return result;
		}

		void reconstruct_slot(command_slot& slot, usz index)
		{
			auto replacement = create_slot(index);
			slot.commands->destroy();
			slot.allocator->destroy();
			slot = std::move(*replacement);
		}

		void reclaim_locked()
		{
			for (auto& slot : slots)
			{
				if (!slot->pending || !slot->pending.completed())
				{
					continue;
				}
				stats.completed_submissions++;
				if (!slot->pending.succeeded())
				{
					stats.failed_submissions++;
					slot->needs_reconstruction = true;
				}
				slot->pending = {};
			}
		}

		command_slot* available_slot_locked()
		{
			reclaim_locked();
			const auto found = std::find_if(slots.begin(), slots.end(), [](const auto& slot)
			{
				return !slot->pending;
			});
			return found == slots.end() ? nullptr : found->get();
		}
	};

	async_task_scheduler::async_task_scheduler()
		: m_impl(std::make_unique<impl>())
	{
	}

	async_task_scheduler::~async_task_scheduler()
	{
		try
		{
			destroy(false);
		}
		catch (...)
		{
		}
	}

	void async_task_scheduler::initialize(render_device& device,
		const async_scheduler_configuration& configuration)
	{
		destroy(false);
		if (!device || configuration.maximum_command_slots == 0 ||
			configuration.maximum_command_slots > maximum_async_command_slots || configuration.label.empty())
		{
			fmt::throw_exception("Invalid Metal asynchronous scheduler configuration");
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->device = &device;
		m_impl->configuration = configuration;
		m_impl->timeline.create(device, sync_domain::gpu, configuration.label + " timeline");
		m_impl->slots.clear();
		m_impl->slots.reserve(configuration.maximum_command_slots);
		m_impl->current = nullptr;
		m_impl->latest = {};
		m_impl->stats = {};
		m_impl->stats.generation = next_generation();
		m_impl->next_sequence = 1;
		m_impl->initialized = true;
		m_impl->shutting_down = false;
	}

	void async_task_scheduler::destroy(bool device_is_idle)
	{
		std::vector<submission> pending;
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->initialized)
			{
				return;
			}
			m_impl->shutting_down = true;
			if (m_impl->current)
			{
				m_impl->current->commands->discard();
				m_impl->current = nullptr;
			}
			for (const auto& slot : m_impl->slots)
			{
				if (slot->pending && !slot->pending.completed()) pending.push_back(slot->pending);
			}
		}

		if (!device_is_idle)
		{
			for (const submission& work : pending) work.wait();
		}
		else
		{
			// Feedback completion owns allocator lifetime even after the device is idle.
			for (const submission& work : pending)
			{
				if (!work.completed()) work.wait();
			}
		}

		std::lock_guard lock(m_impl->mutex);
		for (auto& slot : m_impl->slots)
		{
			slot->pending = {};
			slot->commands->destroy();
			slot->allocator->destroy();
		}
		m_impl->slots.clear();
		m_impl->timeline.destroy();
		m_impl->device = nullptr;
		m_impl->current = nullptr;
		m_impl->latest = {};
		m_impl->initialized = false;
		m_impl->shutting_down = false;
	}

	command_buffer& async_task_scheduler::get_current()
	{
		std::unique_lock lock(m_impl->mutex);
		if (!m_impl->initialized || m_impl->shutting_down)
		{
			fmt::throw_exception("Metal asynchronous scheduler is not available");
		}
		if (m_impl->current)
		{
			return *m_impl->current->commands;
		}

		impl::command_slot* selected = m_impl->available_slot_locked();
		if (!selected && m_impl->slots.size() < m_impl->configuration.maximum_command_slots)
		{
			auto slot = m_impl->create_slot(m_impl->slots.size());
			selected = slot.get();
			m_impl->slots.push_back(std::move(slot));
			m_impl->stats.command_slots = m_impl->slots.size();
		}
		if (!selected)
		{
			const auto oldest = std::min_element(m_impl->slots.begin(), m_impl->slots.end(), [](const auto& left, const auto& right)
			{
				return left->sequence < right->sequence;
			});
			if (oldest == m_impl->slots.end() || !(*oldest)->pending)
			{
				fmt::throw_exception("Metal asynchronous scheduler has no reusable command slot");
			}
			const submission wait_for = (*oldest)->pending;
			m_impl->stats.pool_exhaustion_waits++;
			lock.unlock();
			wait_for.wait();
			lock.lock();
			if (!m_impl->initialized || m_impl->shutting_down)
			{
				fmt::throw_exception("Metal asynchronous scheduler stopped while waiting for a command slot");
			}
			selected = m_impl->available_slot_locked();
			if (!selected)
			{
				fmt::throw_exception("Completed Metal command slot did not become reusable");
			}
		}

		if (selected->needs_reconstruction)
		{
			const auto found = std::find_if(m_impl->slots.begin(), m_impl->slots.end(), [&](const auto& candidate)
			{
				return candidate.get() == selected;
			});
			ensure(found != m_impl->slots.end());
			m_impl->reconstruct_slot(*selected, std::distance(m_impl->slots.begin(), found));
		}
		selected->commands->begin();
		m_impl->current = selected;
		m_impl->stats.acquired_command_buffers++;
		m_impl->stats.recording = true;
		return *selected->commands;
	}

	bool async_task_scheduler::is_recording() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->current != nullptr;
	}

	bool async_task_scheduler::is_host_synchronized() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->initialized && m_impl->configuration.mode == async_scheduler_mode::host_synchronized;
	}

	async_submission async_task_scheduler::flush(submit_info info, bool force_flush)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized || m_impl->shutting_down)
		{
			fmt::throw_exception("Metal asynchronous scheduler is not available");
		}
		if (!m_impl->current)
		{
			static_cast<void>(force_flush);
			return {};
		}

		impl::command_slot* slot = m_impl->current;
		command_buffer& commands = *slot->commands;
		if (m_impl->stats.timeline_value == std::numeric_limits<u64>::max() ||
			m_impl->next_sequence == std::numeric_limits<u64>::max())
		{
			fmt::throw_exception("Metal asynchronous scheduler counter overflowed");
		}
		const u64 signal_value = m_impl->stats.timeline_value + 1;
		info.queue = m_impl->configuration.queue;
		info.signals.push_back(m_impl->timeline.signal_operation(signal_value));
		if (m_impl->configuration.mode == async_scheduler_mode::host_synchronized)
		{
			info.wait_for_completion = true;
		}

		if (commands.active_encoder() != encoder_kind::none)
		{
			commands.end_encoding();
		}
		commands.end();
		m_impl->current = nullptr;
		m_impl->stats.recording = false;

		if (m_impl->configuration.mode == async_scheduler_mode::host_synchronized)
		{
			m_impl->stats.forced_host_waits++;
		}

		try
		{
			submission work = commands.submit(info);
			m_impl->stats.timeline_value = signal_value;
			m_impl->stats.submissions++;
			slot->pending = work;
			slot->sequence = m_impl->next_sequence++;
			slot->timeline_value = signal_value;
			m_impl->latest = {m_impl->timeline.native_handle(), signal_value,
				work.value(), m_impl->stats.generation};
			const async_submission result{work, m_impl->latest};
			m_impl->reclaim_locked();
			return result;
		}
		catch (...)
		{
			const std::exception_ptr failure = std::current_exception();
			const auto found = std::find_if(m_impl->slots.begin(), m_impl->slots.end(), [&](const auto& candidate)
			{
				return candidate.get() == slot;
			});
			if (found != m_impl->slots.end())
			{
				try
				{
					m_impl->reconstruct_slot(**found, std::distance(m_impl->slots.begin(), found));
				}
				catch (...)
				{
					m_impl->slots.erase(found);
					m_impl->stats.command_slots = m_impl->slots.size();
				}
			}
			std::rethrow_exception(failure);
		}
	}

	async_sync_token async_task_scheduler::latest_sync_token() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->latest;
	}

	event_operation async_task_scheduler::graphics_wait_operation() const
	{
		return latest_sync_token().wait_operation();
	}

	void async_task_scheduler::reclaim_completed()
	{
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->initialized) m_impl->reclaim_locked();
	}

	bool async_task_scheduler::wait_idle(std::chrono::nanoseconds timeout)
	{
		if (timeout < std::chrono::nanoseconds::zero())
		{
			fmt::throw_exception("Metal asynchronous scheduler timeout cannot be negative");
		}
		std::vector<submission> pending;
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->initialized) return true;
			if (m_impl->current)
			{
				fmt::throw_exception("Cannot wait for an idle Metal scheduler while commands are recording");
			}
			m_impl->reclaim_locked();
			for (const auto& slot : m_impl->slots)
			{
				if (slot->pending) pending.push_back(slot->pending);
			}
		}

		const auto deadline = std::chrono::steady_clock::now() + timeout;
		for (const submission& work : pending)
		{
			const auto now = std::chrono::steady_clock::now();
			if (now >= deadline || !work.wait(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now)))
			{
				return false;
			}
		}
		reclaim_completed();
		return true;
	}

	void async_task_scheduler::wait_idle()
	{
		std::vector<submission> pending;
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->initialized) return;
			if (m_impl->current)
			{
				fmt::throw_exception("Cannot wait for an idle Metal scheduler while commands are recording");
			}
			m_impl->reclaim_locked();
			for (const auto& slot : m_impl->slots)
			{
				if (slot->pending) pending.push_back(slot->pending);
			}
		}
		for (const submission& work : pending) work.wait();
		reclaim_completed();
	}

	async_scheduler_statistics async_task_scheduler::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		async_scheduler_statistics result = m_impl->stats;
		result.command_slots = m_impl->slots.size();
		result.pending_slots = std::count_if(m_impl->slots.begin(), m_impl->slots.end(), [](const auto& slot)
		{
			return slot->pending && !slot->pending.completed();
		});
		result.recording = m_impl->current != nullptr;
		return result;
	}

	async_task_scheduler::operator bool() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->initialized;
	}
}
