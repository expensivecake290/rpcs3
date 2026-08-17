#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "mtlutils/data_heap.h"

namespace mtl
{
	enum class data_heap_role : u8
	{
		upload,
		constants,
		vertex_data,
		index_data,
		texture_data,
		readback,
		transient,
		count,
	};

	struct managed_heap_id
	{
		u64 value = 0;
		u64 registration = 0;

		[[nodiscard]] explicit operator bool() const
		{
			return value != 0 && registration != 0;
		}

		[[nodiscard]] bool operator==(const managed_heap_id&) const = default;
	};

	struct managed_heap_registration
	{
		data_heap_role role = data_heap_role::transient;
		u32 priority = 0;
		std::string label;
		bool prefer_low_latency = false;
		bool allow_growth = true;
	};

	struct managed_heap_allocation
	{
		managed_heap_id heap;
		data_heap* owner = nullptr;
		data_heap_slice slice;

		[[nodiscard]] explicit operator bool() const
		{
			return heap && owner && slice;
		}
	};

	struct managed_heap_snapshot_entry
	{
		managed_heap_id heap;
		u64 buffer_generation = 0;
		u64 capacity = 0;
		u64 allocated = 0;
		u64 dirty_bytes = 0;
		u64 pending_batches = 0;
	};

	struct managed_heap_snapshot
	{
		u64 serial = 0;
		u64 submission = 0;
		std::vector<managed_heap_snapshot_entry> heaps;

		[[nodiscard]] explicit operator bool() const
		{
			return serial != 0 && submission != 0;
		}
	};

	struct managed_heap_statistics
	{
		managed_heap_id id;
		data_heap_role role = data_heap_role::transient;
		u32 priority = 0;
		std::string label;
		data_heap_statistics heap;
		u64 selection_count = 0;
		u64 allocation_count = 0;
		u64 allocated_bytes = 0;
		u64 failed_allocations = 0;
	};

	struct data_heap_manager_statistics
	{
		u64 registered_heaps = 0;
		u64 snapshots = 0;
		u64 restored_snapshots = 0;
		u64 sealed_submission = 0;
		u64 completed_submission = 0;
		u64 allocations = 0;
		u64 allocated_bytes = 0;
		u64 failed_allocations = 0;
	};

	class data_heap_manager
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		data_heap_manager();
		~data_heap_manager();
		data_heap_manager(const data_heap_manager&) = delete;
		data_heap_manager& operator=(const data_heap_manager&) = delete;

		[[nodiscard]] managed_heap_id register_heap(data_heap& heap,
			const managed_heap_registration& registration);
		[[nodiscard]] std::vector<managed_heap_id> register_heaps(
			std::span<data_heap* const> heaps, const managed_heap_registration& registration);
		void unregister_heap(managed_heap_id id);
		void clear_registrations(bool destroy_heaps, u64 completed_submission);

		[[nodiscard]] managed_heap_allocation allocate(data_heap_role role, u64 size, u64 alignment,
			bool require_low_latency = false);
		[[nodiscard]] data_heap& select(data_heap_role role, u64 size,
			bool require_low_latency = false);

		void flush_all(command_buffer& command);
		void seal_all(u64 submission);
		void reclaim_all(u64 completed_submission);
		void trim_all(memory_pressure pressure, u64 completed_submission);

		[[nodiscard]] managed_heap_snapshot capture_snapshot(u64 submission, bool seal_heaps = true);
		void restore_snapshot(const managed_heap_snapshot& snapshot, u64 completed_submission);

		[[nodiscard]] std::vector<data_heap*> heaps() const;
		[[nodiscard]] std::vector<managed_heap_statistics> heap_statistics() const;
		[[nodiscard]] data_heap_manager_statistics statistics() const;
	};

	[[nodiscard]] data_heap_manager& get_data_heap_manager();
}
