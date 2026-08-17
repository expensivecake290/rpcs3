#include "stdafx.h"
#include "mtlutils/memory.h"

#include <mutex>
#include <unordered_map>

namespace mtl
{
	namespace
	{
		struct allocation_record
		{
			u64 size;
			allocation_pool pool;
		};

		struct allocation_tracker
		{
			std::mutex mutex;
			std::unordered_map<native_allocation_handle, allocation_record> allocations;
			std::array<u64, static_cast<usz>(allocation_pool::count)> pools{};
			u64 total = 0;
			u64 peak = 0;

			~allocation_tracker()
			{
				report_leaks();
			}

			void report_leaks() const
			{
				if (allocations.empty())
				{
					return;
				}

				rsx_log.error("Leaking %u Metal memory allocations (%llu bytes)", allocations.size(), total);
				for (const auto& [handle, record] : allocations)
				{
					rsx_log.error("Metal allocation 0x%llx (%llu bytes) from pool %u was not freed",
						reinterpret_cast<uptr>(handle), record.size, static_cast<u8>(record.pool));
				}
			}
		};

		allocation_tracker& get_tracker()
		{
			static allocation_tracker tracker;
			return tracker;
		}
	}

	void notify_memory_allocated(native_allocation_handle handle, u64 size, allocation_pool pool)
	{
		if (!handle || size == 0 || pool >= allocation_pool::count)
		{
			fmt::throw_exception("Invalid Metal allocation tracking record");
		}

		auto& tracker = get_tracker();
		std::lock_guard lock(tracker.mutex);
		const auto [iterator, inserted] = tracker.allocations.emplace(handle, allocation_record{size, pool});
		if (!inserted)
		{
			fmt::throw_exception("Duplicate Metal allocation tracking handle 0x%llx", reinterpret_cast<uptr>(handle));
		}

		tracker.total += iterator->second.size;
		tracker.pools[static_cast<usz>(pool)] += iterator->second.size;
		tracker.peak = std::max(tracker.peak, tracker.total);
	}

	void notify_memory_freed(native_allocation_handle handle)
	{
		if (!handle)
		{
			return;
		}

		auto& tracker = get_tracker();
		std::lock_guard lock(tracker.mutex);
		const auto found = tracker.allocations.find(handle);
		if (found == tracker.allocations.end())
		{
			rsx_log.error("Unknown Metal allocation tracking handle 0x%llx was freed", reinterpret_cast<uptr>(handle));
			return;
		}

		ensure(tracker.total >= found->second.size);
		auto& pool_size = tracker.pools[static_cast<usz>(found->second.pool)];
		ensure(pool_size >= found->second.size);
		tracker.total -= found->second.size;
		pool_size -= found->second.size;
		tracker.allocations.erase(found);
	}

	void reset_memory_tracking()
	{
		auto& tracker = get_tracker();
		std::lock_guard lock(tracker.mutex);
		tracker.report_leaks();
		tracker.allocations.clear();
		tracker.pools.fill(0);
		tracker.total = 0;
		tracker.peak = 0;
	}

	u64 get_application_memory_usage()
	{
		auto& tracker = get_tracker();
		std::lock_guard lock(tracker.mutex);
		return tracker.total;
	}

	u64 get_application_pool_usage(allocation_pool pool)
	{
		if (pool >= allocation_pool::count)
		{
			fmt::throw_exception("Invalid Metal allocation pool %u", static_cast<u8>(pool));
		}

		auto& tracker = get_tracker();
		std::lock_guard lock(tracker.mutex);
		return tracker.pools[static_cast<usz>(pool)];
	}
}
