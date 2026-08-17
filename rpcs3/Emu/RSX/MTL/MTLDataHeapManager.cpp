#include "stdafx.h"
#include "MTLDataHeapManager.h"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <mutex>
#include <shared_mutex>

namespace mtl
{
	namespace
	{
		u64 saturated_add(u64 first, u64 second)
		{
			return first > std::numeric_limits<u64>::max() - second ?
				std::numeric_limits<u64>::max() : first + second;
		}

		bool valid_role(data_heap_role role)
		{
			return static_cast<usz>(role) < static_cast<usz>(data_heap_role::count);
		}
	}

	struct data_heap_manager::impl
	{
		struct entry
		{
			data_heap* heap = nullptr;
			managed_heap_registration registration;
			managed_heap_statistics stats;
		};

		mutable std::mutex mutex;
		mutable std::shared_mutex operations;
		std::vector<std::unique_ptr<entry>> entries;
		data_heap_manager_statistics stats;
		u64 next_id = 1;
		u64 next_registration = 1;
		u64 next_snapshot = 1;

		entry* find(managed_heap_id id) const
		{
			const auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& candidate)
			{
				return candidate->stats.id == id;
			});
			return found == entries.end() ? nullptr : found->get();
		}

		entry* find(data_heap* heap) const
		{
			const auto found = std::find_if(entries.begin(), entries.end(), [&](const auto& candidate)
			{
				return candidate->heap == heap;
			});
			return found == entries.end() ? nullptr : found->get();
		}

		std::vector<entry*> candidates(data_heap_role role, u64 size, u64 alignment, bool require_low_latency)
		{
			std::vector<entry*> result;
			for (const auto& candidate : entries)
			{
				if (candidate->registration.role != role ||
					(require_low_latency && !candidate->registration.prefer_low_latency))
				{
					continue;
				}
				const data_heap_statistics heap_stats = candidate->heap->statistics();
				const u64 padding = alignment - 1;
				const bool fits_without_growth = heap_stats.capacity >= heap_stats.allocated &&
					size <= heap_stats.capacity - heap_stats.allocated &&
					padding <= heap_stats.capacity - heap_stats.allocated - size;
				if (!candidate->registration.allow_growth && !fits_without_growth)
				{
					continue;
				}
				result.push_back(candidate.get());
			}
			std::sort(result.begin(), result.end(), [](const entry* left, const entry* right)
			{
				if (left->registration.priority != right->registration.priority)
				{
					return left->registration.priority > right->registration.priority;
				}
				if (left->registration.prefer_low_latency != right->registration.prefer_low_latency)
				{
					return left->registration.prefer_low_latency;
				}
				const auto left_stats = left->heap->statistics();
				const auto right_stats = right->heap->statistics();
				const long double left_load = left_stats.capacity ?
					static_cast<long double>(left_stats.allocated) / left_stats.capacity : 1.0L;
				const long double right_load = right_stats.capacity ?
					static_cast<long double>(right_stats.allocated) / right_stats.capacity : 1.0L;
				if (left_load != right_load)
				{
					return left_load < right_load;
				}
				return left->stats.id.value < right->stats.id.value;
			});
			return result;
		}
	};

	data_heap_manager::data_heap_manager()
		: m_impl(std::make_unique<impl>())
	{
	}

	data_heap_manager::~data_heap_manager() = default;

	managed_heap_id data_heap_manager::register_heap(data_heap& heap,
		const managed_heap_registration& registration)
	{
		if (!valid_role(registration.role) || registration.label.empty() || !heap.size())
		{
			fmt::throw_exception("Invalid Metal managed-heap registration");
		}
		std::unique_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->find(&heap))
		{
			fmt::throw_exception("Metal data heap is already registered");
		}
		if (!m_impl->next_id || !m_impl->next_registration)
		{
			fmt::throw_exception("Metal managed-heap identity counter overflowed");
		}
		auto result = std::make_unique<impl::entry>();
		result->heap = &heap;
		result->registration = registration;
		result->stats.id = {m_impl->next_id++, m_impl->next_registration++};
		result->stats.role = registration.role;
		result->stats.priority = registration.priority;
		result->stats.label = registration.label;
		const managed_heap_id id = result->stats.id;
		m_impl->entries.push_back(std::move(result));
		m_impl->stats.registered_heaps = m_impl->entries.size();
		return id;
	}

	std::vector<managed_heap_id> data_heap_manager::register_heaps(
		std::span<data_heap* const> heaps, const managed_heap_registration& registration)
	{
		if (heaps.empty() || !valid_role(registration.role) || registration.label.empty())
		{
			fmt::throw_exception("Invalid Metal managed-heap bulk registration");
		}
		std::unique_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		if (heaps.size() > std::numeric_limits<u64>::max() - m_impl->next_id ||
			heaps.size() > std::numeric_limits<u64>::max() - m_impl->next_registration)
		{
			fmt::throw_exception("Metal managed-heap identity counter overflowed");
		}
		for (usz index = 0; index < heaps.size(); ++index)
		{
			if (!heaps[index] || !heaps[index]->size() || m_impl->find(heaps[index]))
			{
				fmt::throw_exception("Metal managed-heap bulk registration contains an invalid or duplicate heap");
			}
			for (usz earlier = 0; earlier < index; ++earlier)
			{
				if (heaps[earlier] == heaps[index])
				{
					fmt::throw_exception("Metal managed-heap bulk registration repeats a heap");
				}
			}
		}

		std::vector<managed_heap_id> result;
		result.reserve(heaps.size());
		std::vector<std::unique_ptr<impl::entry>> additions;
		additions.reserve(heaps.size());
		m_impl->entries.reserve(m_impl->entries.size() + heaps.size());
		u64 next_id = m_impl->next_id;
		u64 next_registration = m_impl->next_registration;
		for (data_heap* heap : heaps)
		{
			auto addition = std::make_unique<impl::entry>();
			addition->heap = heap;
			addition->registration = registration;
			addition->stats.id = {next_id++, next_registration++};
			addition->stats.role = registration.role;
			addition->stats.priority = registration.priority;
			addition->stats.label = registration.label;
			result.push_back(addition->stats.id);
			additions.push_back(std::move(addition));
		}
		for (auto& addition : additions) m_impl->entries.push_back(std::move(addition));
		m_impl->next_id = next_id;
		m_impl->next_registration = next_registration;
		m_impl->stats.registered_heaps = m_impl->entries.size();
		return result;
	}

	void data_heap_manager::unregister_heap(managed_heap_id id)
	{
		if (!id)
		{
			fmt::throw_exception("Invalid Metal managed-heap identity");
		}
		std::unique_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		const auto found = std::find_if(m_impl->entries.begin(), m_impl->entries.end(), [&](const auto& candidate)
		{
			return candidate->stats.id == id;
		});
		if (found == m_impl->entries.end())
		{
			fmt::throw_exception("Metal managed-heap identity is stale or unknown");
		}
		const data_heap_statistics heap_stats = (*found)->heap->statistics();
		if (heap_stats.allocated || heap_stats.pending_batches || heap_stats.dirty_bytes)
		{
			fmt::throw_exception("Cannot unregister a Metal data heap with live allocations or pending writes");
		}
		m_impl->entries.erase(found);
		m_impl->stats.registered_heaps = m_impl->entries.size();
	}

	void data_heap_manager::clear_registrations(bool destroy_heaps, u64 completed_submission)
	{
		std::unique_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		for (const auto& entry : m_impl->entries)
		{
			entry->heap->reclaim(completed_submission);
			const data_heap_statistics heap_stats = entry->heap->statistics();
			if (heap_stats.allocated || heap_stats.pending_batches || heap_stats.dirty_bytes)
			{
				fmt::throw_exception("Cannot clear Metal data heaps before all allocations and writes retire");
			}
		}
		if (destroy_heaps)
		{
			for (const auto& entry : m_impl->entries)
			{
				entry->heap->destroy();
			}
		}
		m_impl->entries.clear();
		m_impl->stats.registered_heaps = 0;
		m_impl->stats.completed_submission = std::max(m_impl->stats.completed_submission, completed_submission);
	}

	data_heap& data_heap_manager::select(data_heap_role role, u64 size, bool require_low_latency)
	{
		if (!valid_role(role) || !size)
		{
			fmt::throw_exception("Invalid Metal managed-heap selection request");
		}
		std::shared_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		auto candidates = m_impl->candidates(role, size, 1, require_low_latency);
		if (candidates.empty())
		{
			fmt::throw_exception("No Metal data heap can satisfy the requested role and size");
		}
		candidates.front()->stats.selection_count = saturated_add(candidates.front()->stats.selection_count, 1);
		return *candidates.front()->heap;
	}

	managed_heap_allocation data_heap_manager::allocate(data_heap_role role, u64 size, u64 alignment,
		bool require_low_latency)
	{
		if (!valid_role(role) || !size || !alignment || (alignment & (alignment - 1)) != 0)
		{
			fmt::throw_exception("Invalid Metal managed-heap allocation request");
		}
		std::shared_lock operation_lock(m_impl->operations);
		std::vector<impl::entry*> candidates;
		{
			std::lock_guard lock(m_impl->mutex);
			candidates = m_impl->candidates(role, size, alignment, require_low_latency);
		}
		if (candidates.empty())
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->stats.failed_allocations = saturated_add(m_impl->stats.failed_allocations, 1);
			fmt::throw_exception("No Metal data heap can satisfy the allocation request");
		}

		std::exception_ptr last_failure;
		for (impl::entry* candidate : candidates)
		{
			try
			{
				data_heap_slice slice = candidate->heap->allocate(size, alignment,
					candidate->registration.allow_growth);
				std::lock_guard lock(m_impl->mutex);
				candidate->stats.selection_count = saturated_add(candidate->stats.selection_count, 1);
				candidate->stats.allocation_count = saturated_add(candidate->stats.allocation_count, 1);
				candidate->stats.allocated_bytes = saturated_add(candidate->stats.allocated_bytes, size);
				m_impl->stats.allocations = saturated_add(m_impl->stats.allocations, 1);
				m_impl->stats.allocated_bytes = saturated_add(m_impl->stats.allocated_bytes, size);
				return {candidate->stats.id, candidate->heap, slice};
			}
			catch (...)
			{
				last_failure = std::current_exception();
				std::lock_guard lock(m_impl->mutex);
				candidate->stats.failed_allocations = saturated_add(candidate->stats.failed_allocations, 1);
			}
		}
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->stats.failed_allocations = saturated_add(m_impl->stats.failed_allocations, 1);
		}
		std::rethrow_exception(last_failure);
	}

	void data_heap_manager::flush_all(command_buffer& command)
	{
		std::unique_lock operation_lock(m_impl->operations);
		std::vector<data_heap*> heaps_to_flush;
		{
			std::lock_guard lock(m_impl->mutex);
			for (const auto& entry : m_impl->entries) heaps_to_flush.push_back(entry->heap);
		}
		for (data_heap* heap : heaps_to_flush)
		{
			if (heap->is_dirty()) heap->flush(command);
		}
	}

	void data_heap_manager::seal_all(u64 submission)
	{
		if (!submission)
		{
			fmt::throw_exception("Metal managed heaps require a nonzero submission seal");
		}
		std::unique_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		if (submission < m_impl->stats.sealed_submission)
		{
			fmt::throw_exception("Metal managed-heap submissions must seal monotonically");
		}
		for (const auto& entry : m_impl->entries) entry->heap->seal(submission);
		m_impl->stats.sealed_submission = submission;
	}

	void data_heap_manager::reclaim_all(u64 completed_submission)
	{
		std::unique_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		if (completed_submission < m_impl->stats.completed_submission ||
			completed_submission > m_impl->stats.sealed_submission)
		{
			fmt::throw_exception("Invalid Metal managed-heap completion value");
		}
		for (const auto& entry : m_impl->entries) entry->heap->reclaim(completed_submission);
		m_impl->stats.completed_submission = completed_submission;
	}

	void data_heap_manager::trim_all(memory_pressure pressure, u64 completed_submission)
	{
		reclaim_all(completed_submission);
		std::unique_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		for (const auto& entry : m_impl->entries) entry->heap->trim(pressure);
	}

	managed_heap_snapshot data_heap_manager::capture_snapshot(u64 submission, bool seal_heaps)
	{
		if (!submission)
		{
			fmt::throw_exception("Metal managed-heap snapshot requires a nonzero submission");
		}
		std::unique_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		if (submission < m_impl->stats.sealed_submission || !m_impl->next_snapshot)
		{
			fmt::throw_exception("Invalid Metal managed-heap snapshot submission or serial");
		}
		if (seal_heaps)
		{
			for (const auto& entry : m_impl->entries) entry->heap->seal(submission);
			m_impl->stats.sealed_submission = submission;
		}
		else if (submission > m_impl->stats.sealed_submission)
		{
			fmt::throw_exception("An unsealed Metal heap snapshot cannot advance beyond the last seal");
		}

		managed_heap_snapshot result;
		result.serial = m_impl->next_snapshot++;
		result.submission = submission;
		result.heaps.reserve(m_impl->entries.size());
		for (const auto& entry : m_impl->entries)
		{
			const data_heap_statistics heap_stats = entry->heap->statistics();
			result.heaps.push_back({entry->stats.id, heap_stats.generation, heap_stats.capacity,
				heap_stats.allocated, heap_stats.dirty_bytes, heap_stats.pending_batches});
		}
		m_impl->stats.snapshots = saturated_add(m_impl->stats.snapshots, 1);
		return result;
	}

	void data_heap_manager::restore_snapshot(const managed_heap_snapshot& snapshot, u64 completed_submission)
	{
		if (!snapshot || completed_submission < snapshot.submission)
		{
			fmt::throw_exception("Invalid completed Metal managed-heap snapshot");
		}
		std::unique_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		if (completed_submission < m_impl->stats.completed_submission ||
			completed_submission > m_impl->stats.sealed_submission ||
			snapshot.heaps.size() != m_impl->entries.size())
		{
			fmt::throw_exception("Metal managed-heap snapshot does not match current completion state");
		}
		for (const managed_heap_snapshot_entry& snapshot_entry : snapshot.heaps)
		{
			impl::entry* current = m_impl->find(snapshot_entry.heap);
			if (!current || !snapshot_entry.buffer_generation)
			{
				fmt::throw_exception("Metal managed-heap snapshot contains a stale registration");
			}
		}
		for (const auto& entry : m_impl->entries) entry->heap->reclaim(completed_submission);
		m_impl->stats.completed_submission = completed_submission;
		m_impl->stats.restored_snapshots = saturated_add(m_impl->stats.restored_snapshots, 1);
	}

	std::vector<data_heap*> data_heap_manager::heaps() const
	{
		std::shared_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		std::vector<data_heap*> result;
		result.reserve(m_impl->entries.size());
		for (const auto& entry : m_impl->entries) result.push_back(entry->heap);
		return result;
	}

	std::vector<managed_heap_statistics> data_heap_manager::heap_statistics() const
	{
		std::shared_lock operation_lock(m_impl->operations);
		std::lock_guard lock(m_impl->mutex);
		std::vector<managed_heap_statistics> result;
		result.reserve(m_impl->entries.size());
		for (const auto& entry : m_impl->entries)
		{
			managed_heap_statistics stats = entry->stats;
			stats.heap = entry->heap->statistics();
			result.push_back(std::move(stats));
		}
		return result;
	}

	data_heap_manager_statistics data_heap_manager::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->stats;
	}

	data_heap_manager& get_data_heap_manager()
	{
		static data_heap_manager manager;
		return manager;
	}
}
