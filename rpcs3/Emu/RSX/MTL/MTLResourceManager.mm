#include "stdafx.h"
#include "MTLResourceManager.h"

#include "mtlutils/shared.h"

#include <array>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

namespace mtl
{
	namespace
	{
		struct retired_resource
		{
			disposable object;
			deferred_resource_info info;
			resource_retirement point;
		};

		std::atomic<u64> g_resource_event = 0;
		std::atomic<u64> g_completed_resource_event = 0;
		std::mutex g_event_completion_mutex;
		std::set<u64> g_ready_resource_events;

		u64 checked_add(u64 first, u64 second, const char* description)
		{
			if (first > std::numeric_limits<u64>::max() - second)
			{
				fmt::throw_exception("Metal resource-manager %s overflowed", description);
			}
			return first + second;
		}

		void run_deferred_callback(void* raw) noexcept
		{
			std::unique_ptr<std::function<void()>> callback(static_cast<std::function<void()>*>(raw));
			try
			{
				(*callback)();
			}
			catch (const std::exception& exception)
			{
				rsx_log.error("Deferred Metal resource callback failed: %s", exception.what());
			}
			catch (...)
			{
				rsx_log.error("Deferred Metal resource callback failed with an unknown exception");
			}
		}
	}

	struct resource_manager::impl
	{
		mutable std::mutex mutex;
		shared_state* shared = nullptr;
		std::map<u64, std::vector<retired_resource>> pending;
		std::unordered_map<u64, resource_debug_marker> markers;
		std::vector<std::function<void()>> shutdown_callbacks;
		resource_retirement current;
		resource_manager_statistics stats;
		std::array<u64, static_cast<usz>(managed_resource_class::count)> pending_by_class{};
		u64 next_marker_id = 1;
		bool initialized = false;
	};

	resource_manager::resource_manager()
		: m_impl(std::make_unique<impl>())
	{
	}

	resource_manager::~resource_manager()
	{
		try
		{
			destroy(std::numeric_limits<u64>::max(), true);
		}
		catch (...)
		{
		}
	}

	void resource_manager::initialize(shared_state& state)
	{
		if (!state)
		{
			fmt::throw_exception("Metal resource manager requires initialized shared state");
		}
		{
			std::lock_guard lock(m_impl->mutex);
			if (m_impl->initialized)
			{
				fmt::throw_exception("Metal resource manager is already initialized");
			}
			m_impl->shared = &state;
			m_impl->current = {current_resource_event(), state.current_frame()};
			m_impl->stats = {};
			m_impl->stats.current_submission = current_resource_event();
			m_impl->stats.completed_submission = last_completed_resource_event();
			m_impl->pending_by_class.fill(0);
			m_impl->next_marker_id = 1;
			m_impl->initialized = true;
		}
	}

	void resource_manager::destroy(u64 completed_submission, bool device_is_idle)
	{
		std::map<u64, std::vector<retired_resource>> pending;
		std::vector<std::function<void()>> callbacks;
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->initialized)
			{
				return;
			}
			const u64 effective_completion = device_is_idle ? std::numeric_limits<u64>::max() : completed_submission;
			if (!device_is_idle)
			{
				const auto first_unfinished = m_impl->pending.upper_bound(effective_completion);
				if (first_unfinished != m_impl->pending.end())
				{
					fmt::throw_exception("Cannot destroy Metal resource manager with %llu GPU-dependent resources pending",
						m_impl->stats.pending_count);
				}
			}
			pending.swap(m_impl->pending);
			callbacks.swap(m_impl->shutdown_callbacks);
			m_impl->markers.clear();
			m_impl->pending_by_class.fill(0);
			m_impl->stats.released_count = checked_add(m_impl->stats.released_count,
				m_impl->stats.pending_count, "released-resource count");
			m_impl->stats.pending_count = 0;
			m_impl->stats.pending_bytes = 0;
			m_impl->stats.marker_count = 0;
			m_impl->stats.shutdown_callbacks = callbacks.size();
			m_impl->shared = nullptr;
			m_impl->initialized = false;
		}

		pending.clear();
		for (auto iterator = callbacks.rbegin(); iterator != callbacks.rend(); ++iterator)
		{
			try
			{
				(*iterator)();
			}
			catch (const std::exception& exception)
			{
				rsx_log.error("Metal resource-manager shutdown callback failed: %s", exception.what());
			}
			catch (...)
			{
				rsx_log.error("Metal resource-manager shutdown callback failed with an unknown exception");
			}
		}
	}

	void resource_manager::set_retirement_point(resource_retirement point)
	{
		if (!point)
		{
			fmt::throw_exception("Metal resource retirement requires a nonzero submission");
		}
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized)
		{
			fmt::throw_exception("Metal resource manager is not initialized");
		}
		if (point.submission < m_impl->current.submission ||
			(point.submission == m_impl->current.submission && point.frame < m_impl->current.frame) ||
			point.submission < m_impl->stats.completed_submission)
		{
			fmt::throw_exception("Metal resource retirement points must advance monotonically");
		}
		m_impl->current = point;
		m_impl->stats.current_submission = point.submission;
	}

	resource_retirement resource_manager::retirement_point() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->current;
	}

	void resource_manager::retire(disposable&& object, const deferred_resource_info& info)
	{
		resource_retirement point;
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->initialized)
			{
				fmt::throw_exception("Metal resource manager is not initialized");
			}
			point = m_impl->current;
		}
		retire(std::move(object), point, info);
	}

	void resource_manager::retire(disposable&& object, resource_retirement point,
		const deferred_resource_info& info)
	{
		if (!object)
		{
			return;
		}
		if (!point || static_cast<usz>(info.resource_class) >= static_cast<usz>(managed_resource_class::count))
		{
			fmt::throw_exception("Invalid Metal deferred-resource retirement metadata");
		}

		deferred_resource_info stable_info = info;
		bool release_immediately = false;
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->initialized)
			{
				fmt::throw_exception("Metal resource manager is not initialized");
			}
			if (point.submission > m_impl->current.submission)
			{
				fmt::throw_exception("Metal resource cannot retire after the current submission point");
			}
			const u64 retired_count = checked_add(m_impl->stats.retired_count, 1, "retired-resource count");
			if (point.submission <= m_impl->stats.completed_submission)
			{
				const u64 released_count = checked_add(m_impl->stats.released_count, 1, "released-resource count");
				m_impl->stats.retired_count = retired_count;
				m_impl->stats.released_count = released_count;
				release_immediately = true;
			}
			else
			{
				const u64 pending_count = checked_add(m_impl->stats.pending_count, 1, "pending-resource count");
				const u64 pending_bytes = checked_add(m_impl->stats.pending_bytes, stable_info.bytes, "pending byte count");
				const usz class_index = static_cast<usz>(stable_info.resource_class);
				const u64 class_count = checked_add(m_impl->pending_by_class[class_index], 1,
					"per-class pending-resource count");
				auto [submission_iterator, inserted] = m_impl->pending.try_emplace(point.submission);
				auto& submission_resources = submission_iterator->second;
				try
				{
					submission_resources.reserve(submission_resources.size() + 1);
				}
				catch (...)
				{
					if (inserted)
					{
						m_impl->pending.erase(submission_iterator);
					}
					throw;
				}
				submission_resources.push_back({std::move(object), std::move(stable_info), point});
				m_impl->stats.retired_count = retired_count;
				m_impl->stats.pending_count = pending_count;
				m_impl->stats.pending_bytes = pending_bytes;
				m_impl->stats.peak_pending_bytes = std::max(m_impl->stats.peak_pending_bytes, m_impl->stats.pending_bytes);
				m_impl->pending_by_class[class_index] = class_count;
			}
		}
		if (release_immediately)
		{
			object.reset();
		}
	}

	void resource_manager::defer(std::function<void()> release, const deferred_resource_info& info)
	{
		if (!release)
		{
			fmt::throw_exception("Metal deferred release callback cannot be empty");
		}
		auto* callback = new std::function<void()>(std::move(release));
		retire(disposable::make(callback, run_deferred_callback), info);
	}

	void resource_manager::defer(std::function<void()> release, resource_retirement point,
		const deferred_resource_info& info)
	{
		if (!release)
		{
			fmt::throw_exception("Metal deferred release callback cannot be empty");
		}
		auto* callback = new std::function<void()>(std::move(release));
		retire(disposable::make(callback, run_deferred_callback), point, info);
	}

	u64 resource_manager::add_debug_marker(std::string label, std::string detail)
	{
		if (label.empty())
		{
			fmt::throw_exception("Metal resource debug marker requires a label");
		}
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized || !m_impl->current)
		{
			fmt::throw_exception("Metal resource manager has no active retirement point");
		}
		if (!m_impl->next_marker_id)
		{
			fmt::throw_exception("Metal resource debug-marker ID overflowed");
		}
		const u64 id = m_impl->next_marker_id++;
		m_impl->markers.emplace(id, resource_debug_marker{id, m_impl->current.submission,
			m_impl->current.frame, std::move(label), std::move(detail)});
		m_impl->stats.marker_count = m_impl->markers.size();
		return id;
	}

	void resource_manager::remove_debug_marker(u64 marker_id)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized)
		{
			return;
		}
		m_impl->markers.erase(marker_id);
		m_impl->stats.marker_count = m_impl->markers.size();
	}

	std::vector<resource_debug_marker> resource_manager::debug_markers() const
	{
		std::lock_guard lock(m_impl->mutex);
		std::vector<resource_debug_marker> result;
		result.reserve(m_impl->markers.size());
		for (const auto& [id, marker] : m_impl->markers)
		{
			static_cast<void>(id);
			result.push_back(marker);
		}
		std::sort(result.begin(), result.end(), [](const auto& left, const auto& right)
		{
			return left.id < right.id;
		});
		return result;
	}

	void resource_manager::complete(u64 completed_submission)
	{
		std::map<u64, std::vector<retired_resource>> ready;
		u64 released_count = 0;
		u64 released_bytes = 0;
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->initialized)
			{
				fmt::throw_exception("Metal resource manager is not initialized");
			}
			if (completed_submission < m_impl->stats.completed_submission ||
				completed_submission > m_impl->current.submission)
			{
				fmt::throw_exception("Invalid Metal completed resource submission %llu", completed_submission);
			}
			m_impl->stats.completed_submission = completed_submission;
			for (auto iterator = m_impl->pending.begin();
				iterator != m_impl->pending.end() && iterator->first <= completed_submission;)
			{
				for (const auto& entry : iterator->second)
				{
					++released_count;
					released_bytes = checked_add(released_bytes, entry.info.bytes, "released byte count");
					--m_impl->pending_by_class[static_cast<usz>(entry.info.resource_class)];
				}
				auto node = m_impl->pending.extract(iterator++);
				ready.insert(std::move(node));
			}
			m_impl->stats.pending_count -= released_count;
			m_impl->stats.pending_bytes -= released_bytes;
			m_impl->stats.released_count = checked_add(m_impl->stats.released_count,
				released_count, "released-resource count");
			for (auto iterator = m_impl->markers.begin(); iterator != m_impl->markers.end();)
			{
				if (iterator->second.submission <= completed_submission)
				{
					iterator = m_impl->markers.erase(iterator);
				}
				else
				{
					++iterator;
				}
			}
			m_impl->stats.marker_count = m_impl->markers.size();
		}
		ready.clear();
	}

	void resource_manager::flush_completed(u64 completed_submission)
	{
		complete(completed_submission);
	}

	void resource_manager::trim(memory_pressure pressure, u64 completed_submission)
	{
		complete(completed_submission);
		std::lock_guard lock(m_impl->mutex);
		++m_impl->stats.trim_count;
		if (pressure == memory_pressure::critical)
		{
			for (auto iterator = m_impl->markers.begin(); iterator != m_impl->markers.end();)
			{
				if (iterator->second.detail.empty())
				{
					iterator = m_impl->markers.erase(iterator);
				}
				else
				{
					++iterator;
				}
			}
			m_impl->stats.marker_count = m_impl->markers.size();
		}
	}

	void resource_manager::add_shutdown_callback(std::function<void()> callback)
	{
		if (!callback)
		{
			fmt::throw_exception("Metal resource-manager shutdown callback cannot be empty");
		}
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized)
		{
			fmt::throw_exception("Metal resource manager is not initialized");
		}
		m_impl->shutdown_callbacks.push_back(std::move(callback));
		m_impl->stats.shutdown_callbacks = m_impl->shutdown_callbacks.size();
	}

	resource_manager_statistics resource_manager::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->stats;
	}

	resource_manager::operator bool() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->initialized;
	}

	resource_manager& get_resource_manager()
	{
		static resource_manager manager;
		return manager;
	}

	u64 allocate_resource_event()
	{
		u64 current = g_resource_event.load(std::memory_order_relaxed);
		for (;;)
		{
			if (current == std::numeric_limits<u64>::max())
			{
				fmt::throw_exception("Metal resource event counter overflowed");
			}
			if (g_resource_event.compare_exchange_weak(current, current + 1, std::memory_order_acq_rel))
			{
				resource_manager& manager = get_resource_manager();
				if (manager)
				{
					manager.set_retirement_point({current + 1, get_frame_id()});
				}
				return current + 1;
			}
		}
	}

	u64 current_resource_event()
	{
		return g_resource_event.load(std::memory_order_acquire);
	}

	u64 last_completed_resource_event()
	{
		return g_completed_resource_event.load(std::memory_order_acquire);
	}

	void notify_resource_event_completed(u64 event)
	{
		if (!event || event > current_resource_event())
		{
			fmt::throw_exception("Invalid completed Metal resource event %llu", event);
		}
		u64 contiguous = 0;
		{
			std::lock_guard lock(g_event_completion_mutex);
			const u64 completed = g_completed_resource_event.load(std::memory_order_relaxed);
			if (event <= completed)
			{
				return;
			}
			g_ready_resource_events.insert(event);
			contiguous = completed;
			while (g_ready_resource_events.erase(contiguous + 1))
			{
				++contiguous;
			}
			g_completed_resource_event.store(contiguous, std::memory_order_release);
		}
		resource_manager& manager = get_resource_manager();
		if (manager && contiguous > manager.statistics().completed_submission)
		{
			manager.complete(contiguous);
		}
	}
}
