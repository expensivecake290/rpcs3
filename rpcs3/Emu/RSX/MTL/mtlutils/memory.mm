#include "stdafx.h"
#include "memory.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mtl
{
	namespace
	{
		constexpr u64 private_heap_size = 64ull * 1024 * 1024;
		constexpr u64 cpu_visible_heap_size = 16ull * 1024 * 1024;

		u64 align_up(u64 value, u64 alignment)
		{
			ensure(alignment != 0);
			const u64 remainder = value % alignment;
			return remainder ? value + alignment - remainder : value;
		}

		MTLStorageMode to_native_storage(storage_mode mode)
		{
			switch (mode)
			{
			case storage_mode::shared: return MTLStorageModeShared;
			case storage_mode::managed: return MTLStorageModeManaged;
			case storage_mode::private_: return MTLStorageModePrivate;
			case storage_mode::memoryless: return MTLStorageModeMemoryless;
			case storage_mode::automatic: break;
			}

			fmt::throw_exception("Automatic Metal storage mode was not resolved");
		}

		MTLCPUCacheMode to_native_cache(cpu_cache_mode mode)
		{
			return mode == cpu_cache_mode::write_combined ? MTLCPUCacheModeWriteCombined : MTLCPUCacheModeDefaultCache;
		}

		MTLHazardTrackingMode to_native_hazards(hazard_tracking mode)
		{
			return mode == hazard_tracking::untracked ? MTLHazardTrackingModeUntracked : MTLHazardTrackingModeTracked;
		}

		MTLResourceOptions make_resource_options(storage_mode storage, cpu_cache_mode cache, hazard_tracking hazards)
		{
			return static_cast<MTLResourceOptions>(
				static_cast<NSUInteger>(to_native_storage(storage)) << MTLResourceStorageModeShift |
				static_cast<NSUInteger>(to_native_cache(cache)) << MTLResourceCPUCacheModeShift |
				static_cast<NSUInteger>(to_native_hazards(hazards)) << MTLResourceHazardTrackingModeShift);
		}

		storage_mode resolve_storage(const render_device& device, const memory_allocation_request& request)
		{
			if (request.storage != storage_mode::automatic)
			{
				if (request.storage == storage_mode::managed && device.info().memory.unified)
				{
					return storage_mode::shared;
				}

				return request.storage;
			}

			if (request.access == cpu_access::none)
			{
				return storage_mode::private_;
			}

			return device.info().memory.unified ? storage_mode::shared : storage_mode::managed;
		}

		struct free_range
		{
			u64 offset;
			u64 size;
		};

		struct heap_state
		{
			id<MTLHeap> heap;
			storage_mode storage = storage_mode::private_;
			cpu_cache_mode cache = cpu_cache_mode::default_cache;
			hazard_tracking hazards = hazard_tracking::tracked;
			u64 size = 0;
			u64 live_allocations = 0;
			std::vector<free_range> free_ranges;

			bool compatible(storage_mode requested_storage, cpu_cache_mode requested_cache, hazard_tracking requested_hazards) const
			{
				return storage == requested_storage && cache == requested_cache && hazards == requested_hazards;
			}
		};

		struct allocator_state
		{
			const render_device* render = nullptr;
			id<MTLDevice> device;
			mutable std::mutex mutex;
			std::vector<std::shared_ptr<heap_state>> heaps;
			memory_usage stats;

			static void return_range(heap_state& heap, u64 offset, u64 size)
			{
				ensure(heap.live_allocations != 0);
				heap.live_allocations--;
				heap.free_ranges.push_back({offset, size});
				std::sort(heap.free_ranges.begin(), heap.free_ranges.end(), [](const free_range& lhs, const free_range& rhs)
				{
					return lhs.offset < rhs.offset;
				});

				std::vector<free_range> merged;
				for (const free_range range : heap.free_ranges)
				{
					if (!merged.empty() && merged.back().offset + merged.back().size == range.offset)
					{
						merged.back().size += range.size;
					}
					else
					{
						merged.push_back(range);
					}
				}
				heap.free_ranges = std::move(merged);
			}

			std::pair<std::shared_ptr<heap_state>, u64> reserve(
				u64 size,
				u64 alignment,
				storage_mode storage,
				cpu_cache_mode cache,
				hazard_tracking hazards)
			{
				std::lock_guard lock(mutex);
				for (const auto& candidate : heaps)
				{
					if (!candidate->compatible(storage, cache, hazards))
					{
						continue;
					}

					for (usz index = 0; index < candidate->free_ranges.size(); ++index)
					{
						const free_range range = candidate->free_ranges[index];
						const u64 aligned_offset = align_up(range.offset, alignment);
						const u64 padding = aligned_offset - range.offset;
						if (padding > range.size || size > range.size - padding)
						{
							continue;
						}

						candidate->free_ranges.erase(candidate->free_ranges.begin() + index);
						if (padding)
						{
							candidate->free_ranges.push_back({range.offset, padding});
						}

						const u64 tail_offset = aligned_offset + size;
						const u64 range_end = range.offset + range.size;
						if (tail_offset < range_end)
						{
							candidate->free_ranges.push_back({tail_offset, range_end - tail_offset});
						}

						candidate->live_allocations++;
						return {candidate, aligned_offset};
					}
				}

				const u64 default_size = storage == storage_mode::private_ ? private_heap_size : cpu_visible_heap_size;
				const u64 heap_size = align_up(std::max(default_size, size), std::max<u64>(alignment, 4096));
				MTLHeapDescriptor* descriptor = [MTLHeapDescriptor new];
				descriptor.size = heap_size;
				descriptor.type = MTLHeapTypePlacement;
				descriptor.storageMode = to_native_storage(storage);
				descriptor.cpuCacheMode = to_native_cache(cache);
				descriptor.hazardTrackingMode = to_native_hazards(hazards);

				id<MTLHeap> native_heap = [device newHeapWithDescriptor:descriptor];
				if (!native_heap)
				{
					return {nullptr, 0};
				}

				auto created = std::make_shared<heap_state>();
				created->heap = native_heap;
				created->storage = storage;
				created->cache = cache;
				created->hazards = hazards;
				created->size = heap_size;
				created->live_allocations = 1;
				if (size < heap_size)
				{
					created->free_ranges.push_back({size, heap_size - size});
				}
				heaps.push_back(created);
				stats.resident += heap_size;
				return {std::move(created), 0};
			}

			void release(const std::shared_ptr<heap_state>& heap, u64 offset, u64 reserved_size, u64 logical_size, allocation_pool pool)
			{
				std::lock_guard lock(mutex);
				stats.allocated -= logical_size;
				stats.pools[static_cast<usz>(pool)] -= logical_size;
				if (!heap)
				{
					stats.resident -= reserved_size;
					return;
				}

				return_range(*heap, offset, reserved_size);
			}

			void abandon(const std::shared_ptr<heap_state>& heap, u64 offset, u64 reserved_size)
			{
				std::lock_guard lock(mutex);
				return_range(*heap, offset, reserved_size);
			}

			void account(u64 size, allocation_pool pool, bool standalone)
			{
				std::lock_guard lock(mutex);
				stats.allocated += size;
				stats.pools[static_cast<usz>(pool)] += size;
				stats.peak = std::max(stats.peak, stats.allocated);
				if (standalone)
				{
					stats.resident += size;
				}
			}

			void trim(memory_pressure pressure)
			{
				std::lock_guard lock(mutex);
				for (auto iterator = heaps.begin(); iterator != heaps.end();)
				{
					const auto& heap = *iterator;
					if (heap->live_allocations == 0)
					{
						[heap->heap setPurgeableState:MTLPurgeableStateEmpty];
						stats.resident -= heap->size;
						iterator = heaps.erase(iterator);
						continue;
					}

					if (pressure == memory_pressure::critical)
					{
						[heap->heap setPurgeableState:MTLPurgeableStateNonVolatile];
					}
					++iterator;
				}
			}
		};
	}

	struct memory_allocation::impl
	{
		std::shared_ptr<allocator_state> owner;
		std::shared_ptr<heap_state> heap_state;
		id<MTLBuffer> native_buffer;
		u64 heap_offset = 0;
		u64 logical_size = 0;
		u64 reserved_size = 0;
		storage_mode resolved_storage = storage_mode::private_;
		allocation_pool allocation_pool_id = allocation_pool::system;
		bool heap_placed = false;
		bool purgeable_allowed = false;
		bool accounted = false;
		bool tracking_registered = false;
		u32 map_count = 0;

		~impl()
		{
			if (owner && accounted)
			{
				owner->release(heap_state, heap_offset, reserved_size, logical_size, allocation_pool_id);
			}
			else if (owner && heap_state)
			{
				owner->abandon(heap_state, heap_offset, reserved_size);
			}

			if (tracking_registered)
			{
				notify_memory_freed(this);
			}
		}
	};

	struct memory_allocator::impl
	{
		std::shared_ptr<allocator_state> state;
	};

	struct residency_set::impl
	{
		id<MTLResidencySet> set;
		std::unordered_map<void*, u32> references;
		std::vector<id<MTL4CommandQueue>> queues;
		std::mutex mutex;
		bool resident = false;
	};

	memory_allocation::memory_allocation(std::shared_ptr<impl> implementation)
		: m_impl(std::move(implementation))
	{
	}

	memory_allocation::operator bool() const
	{
		return m_impl && (m_impl->native_buffer || m_impl->heap_state);
	}

	u64 memory_allocation::size() const
	{
		return m_impl ? m_impl->logical_size : 0;
	}

	u64 memory_allocation::offset() const
	{
		return m_impl ? m_impl->heap_offset : 0;
	}

	storage_mode memory_allocation::storage() const
	{
		if (!m_impl)
		{
			fmt::throw_exception("Storage mode requested from an empty Metal allocation");
		}

		return m_impl->resolved_storage;
	}

	allocation_pool memory_allocation::pool() const
	{
		if (!m_impl)
		{
			fmt::throw_exception("Allocation pool requested from an empty Metal allocation");
		}

		return m_impl->allocation_pool_id;
	}

	bool memory_allocation::is_cpu_visible() const
	{
		return m_impl && (m_impl->resolved_storage == storage_mode::shared || m_impl->resolved_storage == storage_mode::managed);
	}

	bool memory_allocation::is_heap_placed() const
	{
		return m_impl && m_impl->heap_placed;
	}

	buffer_handle memory_allocation::buffer() const
	{
		return m_impl ? m_impl->native_buffer : nil;
	}

	native_heap_handle memory_allocation::heap() const
	{
		return m_impl && m_impl->heap_state ? (__bridge void*)m_impl->heap_state->heap : nullptr;
	}

	native_allocation_handle memory_allocation::native_allocation() const
	{
		if (!m_impl)
		{
			return nullptr;
		}

		if (m_impl->native_buffer)
		{
			return (__bridge void*)static_cast<id<MTLAllocation>>(m_impl->native_buffer);
		}

		return (__bridge void*)static_cast<id<MTLAllocation>>(m_impl->heap_state->heap);
	}

	void* memory_allocation::map(u64 offset, u64 size)
	{
		if (!m_impl || !m_impl->native_buffer)
		{
			fmt::throw_exception("Cannot map a Metal allocation without a buffer resource");
		}

		if (!is_cpu_visible())
		{
			fmt::throw_exception("Cannot map a private Metal buffer");
		}

		if (offset > m_impl->logical_size || (size && size > m_impl->logical_size - offset))
		{
			fmt::throw_exception("Metal buffer map range [%llu, %llu) exceeds allocation size %llu", offset, offset + size, m_impl->logical_size);
		}

		m_impl->map_count++;
		return static_cast<u8*>(m_impl->native_buffer.contents) + offset;
	}

	void memory_allocation::unmap()
	{
		if (!m_impl || m_impl->map_count == 0)
		{
			fmt::throw_exception("Unbalanced Metal buffer unmap");
		}

		m_impl->map_count--;
	}

	void memory_allocation::did_modify(u64 offset, u64 size)
	{
		if (!m_impl || !m_impl->native_buffer || offset > m_impl->logical_size || size > m_impl->logical_size - offset)
		{
			fmt::throw_exception("Invalid Metal buffer modification range");
		}

		if (m_impl->resolved_storage == storage_mode::managed)
		{
			[m_impl->native_buffer didModifyRange:NSMakeRange(offset, size)];
		}
	}

	bool memory_allocation::make_volatile()
	{
		if (!m_impl || !m_impl->purgeable_allowed)
		{
			return false;
		}

		if (m_impl->native_buffer)
		{
			return [m_impl->native_buffer setPurgeableState:MTLPurgeableStateVolatile] != MTLPurgeableStateEmpty;
		}

		return [m_impl->heap_state->heap setPurgeableState:MTLPurgeableStateVolatile] != MTLPurgeableStateEmpty;
	}

	bool memory_allocation::make_nonvolatile()
	{
		if (!m_impl || !m_impl->purgeable_allowed)
		{
			return false;
		}

		if (m_impl->native_buffer)
		{
			return [m_impl->native_buffer setPurgeableState:MTLPurgeableStateNonVolatile] != MTLPurgeableStateEmpty;
		}

		return [m_impl->heap_state->heap setPurgeableState:MTLPurgeableStateNonVolatile] != MTLPurgeableStateEmpty;
	}

	residency_set::residency_set()
		: m_impl(std::make_unique<impl>())
	{
	}

	residency_set::~residency_set()
	{
		destroy();
	}

	residency_set::residency_set(residency_set&&) noexcept = default;

	residency_set& residency_set::operator=(residency_set&& other) noexcept
	{
		if (this != &other)
		{
			destroy();
			m_impl = std::move(other.m_impl);
		}
		return *this;
	}

	void residency_set::create(const render_device& device, std::string_view label, u64 initial_capacity)
	{
		destroy();
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
		descriptor.label = [NSString stringWithUTF8String:std::string(label).c_str()];
		descriptor.initialCapacity = initial_capacity;
		NSError* native_error = nil;
		m_impl->set = [device.native_handle() newResidencySetWithDescriptor:descriptor error:&native_error];
		if (!m_impl->set)
		{
			error value;
			value.code = error_code::resource_creation_failed;
			value.native_code = native_error.code;
			value.domain = native_error.domain.UTF8String ?: "Metal";
			value.description = native_error.localizedDescription.UTF8String ?: "Metal returned no residency set";
			throw_error(value, "Metal residency-set creation");
		}
	}

	void residency_set::destroy()
	{
		if (!m_impl || !m_impl->set)
		{
			return;
		}

		std::lock_guard lock(m_impl->mutex);
		if (m_impl->resident)
		{
			[m_impl->set endResidency];
		}

		for (id<MTL4CommandQueue> queue : m_impl->queues)
		{
			[queue removeResidencySet:m_impl->set];
		}

		[m_impl->set removeAllAllocations];
		[m_impl->set commit];
		m_impl->queues.clear();
		m_impl->references.clear();
		m_impl->set = nil;
		m_impl->resident = false;
	}

	void residency_set::add(const memory_allocation& allocation)
	{
		if (!allocation)
		{
			fmt::throw_exception("Cannot add an invalid allocation to a Metal residency set");
		}
		add(allocation.native_allocation());
	}

	void residency_set::add(native_allocation_handle allocation)
	{
		if (!m_impl || !m_impl->set || !allocation)
		{
			fmt::throw_exception("Cannot add an invalid allocation to a Metal residency set");
		}

		std::lock_guard lock(m_impl->mutex);
		void* key = allocation;
		auto& references = m_impl->references[key];
		if (references++ == 0)
		{
			[m_impl->set addAllocation:(__bridge id<MTLAllocation>)key];
			[m_impl->set commit];
		}
	}

	void residency_set::remove(const memory_allocation& allocation)
	{
		if (!allocation)
		{
			fmt::throw_exception("Cannot remove an invalid allocation from a Metal residency set");
		}
		remove(allocation.native_allocation());
	}

	void residency_set::remove(native_allocation_handle allocation)
	{
		if (!m_impl || !m_impl->set || !allocation)
		{
			fmt::throw_exception("Cannot remove an invalid allocation from a Metal residency set");
		}

		std::lock_guard lock(m_impl->mutex);
		void* key = allocation;
		const auto found = m_impl->references.find(key);
		if (found == m_impl->references.end())
		{
			fmt::throw_exception("Metal allocation was not present in the residency set");
		}

		if (--found->second == 0)
		{
			[m_impl->set removeAllocation:(__bridge id<MTLAllocation>)key];
			[m_impl->set commit];
			m_impl->references.erase(found);
		}
	}

	void residency_set::attach(command_queue_handle queue)
	{
		if (!m_impl || !m_impl->set || !queue)
		{
			fmt::throw_exception("Cannot attach an invalid Metal residency set or queue");
		}

		std::lock_guard lock(m_impl->mutex);
		id<MTL4CommandQueue> native_queue = queue;
		if (std::find(m_impl->queues.begin(), m_impl->queues.end(), native_queue) == m_impl->queues.end())
		{
			[native_queue addResidencySet:m_impl->set];
			m_impl->queues.push_back(native_queue);
		}
	}

	void residency_set::detach(command_queue_handle queue)
	{
		if (!m_impl || !m_impl->set || !queue)
		{
			fmt::throw_exception("Cannot detach an invalid Metal residency set or queue");
		}

		std::lock_guard lock(m_impl->mutex);
		id<MTL4CommandQueue> native_queue = queue;
		const auto found = std::find(m_impl->queues.begin(), m_impl->queues.end(), native_queue);
		if (found != m_impl->queues.end())
		{
			[native_queue removeResidencySet:m_impl->set];
			m_impl->queues.erase(found);
		}
	}

	void residency_set::request_residency()
	{
		if (!m_impl || !m_impl->set)
		{
			fmt::throw_exception("Cannot request residency for an empty Metal residency set");
		}

		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->resident)
		{
			[m_impl->set requestResidency];
			m_impl->resident = true;
		}
	}

	void residency_set::end_residency()
	{
		if (!m_impl || !m_impl->set)
		{
			fmt::throw_exception("Cannot end residency for an empty Metal residency set");
		}

		std::lock_guard lock(m_impl->mutex);
		if (m_impl->resident)
		{
			[m_impl->set endResidency];
			m_impl->resident = false;
		}
	}

	residency_set::operator bool() const
	{
		return m_impl && m_impl->set;
	}

	native_allocation_handle residency_set::native_handle() const
	{
		return m_impl ? (__bridge void*)m_impl->set : nullptr;
	}

	memory_allocator::memory_allocator(const render_device& device)
		: m_impl(std::make_shared<impl>())
	{
		if (!device)
		{
			fmt::throw_exception("Cannot create a Metal memory allocator without a render device");
		}

		m_impl->state = std::make_shared<allocator_state>();
		m_impl->state->render = &device;
		m_impl->state->device = device.native_handle();
		m_impl->state->stats.budget = device.info().memory.recommended_working_set_size;
	}

	memory_allocator::~memory_allocator() = default;
	memory_allocator::memory_allocator(memory_allocator&&) noexcept = default;
	memory_allocator& memory_allocator::operator=(memory_allocator&&) noexcept = default;

	memory_allocation memory_allocator::allocate_buffer(const memory_allocation_request& request)
	{
		if (!m_impl || !m_impl->state || request.size == 0 || request.alignment == 0)
		{
			fmt::throw_exception("Invalid Metal buffer allocation request");
		}

		const storage_mode storage = resolve_storage(*m_impl->state->render, request);
		if (storage == storage_mode::memoryless)
		{
			fmt::throw_exception("Metal buffers cannot use memoryless storage");
		}

		const MTLResourceOptions options = make_resource_options(storage, request.cache, request.hazards);
		const MTLSizeAndAlign requirements = [m_impl->state->device heapBufferSizeAndAlignWithLength:request.size options:options];
		const u64 reserved_size = std::max<u64>(request.size, requirements.size);
		const u64 alignment = std::max<u64>(request.alignment, requirements.align);
		auto allocation = std::make_shared<memory_allocation::impl>();
		allocation->owner = m_impl->state;
		allocation->logical_size = request.size;
		allocation->reserved_size = request.use_placement_heap ? reserved_size : request.size;
		allocation->resolved_storage = storage;
		allocation->allocation_pool_id = request.pool;
		allocation->heap_placed = request.use_placement_heap;
		allocation->purgeable_allowed = request.purgeable && !request.use_placement_heap;

		if (request.use_placement_heap)
		{
			auto reserved = m_impl->state->reserve(reserved_size, alignment, storage, request.cache, request.hazards);
			if (!reserved.first && request.recover_on_failure)
			{
				m_impl->state->trim(memory_pressure::critical);
				reserved = m_impl->state->reserve(reserved_size, alignment, storage, request.cache, request.hazards);
			}

			if (reserved.first)
			{
				allocation->heap_state = reserved.first;
				allocation->heap_offset = reserved.second;
				allocation->native_buffer = [reserved.first->heap newBufferWithLength:request.size options:options offset:reserved.second];
			}
		}
		else
		{
			allocation->native_buffer = [m_impl->state->device newBufferWithLength:request.size options:options];
		}

		if (!allocation->native_buffer)
		{
			allocation.reset();
			if (!request.throw_on_failure)
			{
				return memory_allocation();
			}
			fmt::throw_exception("Failed to allocate %llu bytes for Metal buffer '%s'", request.size, request.label);
		}

		if (!request.label.empty())
		{
			allocation->native_buffer.label = [NSString stringWithUTF8String:request.label.c_str()];
		}

		m_impl->state->account(request.size, request.pool, !request.use_placement_heap);
		allocation->accounted = true;
		notify_memory_allocated(allocation.get(), request.size, request.pool);
		allocation->tracking_registered = true;
		return memory_allocation(std::move(allocation));
	}

	memory_allocation memory_allocator::allocate_placement(const memory_allocation_request& request)
	{
		if (!m_impl || !m_impl->state || request.size == 0 || request.alignment == 0)
		{
			fmt::throw_exception("Invalid Metal placement allocation request");
		}

		const storage_mode storage = resolve_storage(*m_impl->state->render, request);
		if (storage == storage_mode::memoryless)
		{
			fmt::throw_exception("Memoryless textures do not occupy placement-heap regions");
		}

		auto reserved = m_impl->state->reserve(request.size, request.alignment, storage, request.cache, request.hazards);
		if (!reserved.first && request.recover_on_failure)
		{
			m_impl->state->trim(memory_pressure::critical);
			reserved = m_impl->state->reserve(request.size, request.alignment, storage, request.cache, request.hazards);
		}

		if (!reserved.first)
		{
			if (!request.throw_on_failure)
			{
				return memory_allocation();
			}
			fmt::throw_exception("Failed to reserve %llu bytes in a Metal placement heap", request.size);
		}

		auto allocation = std::make_shared<memory_allocation::impl>();
		allocation->owner = m_impl->state;
		allocation->heap_state = reserved.first;
		allocation->heap_offset = reserved.second;
		allocation->logical_size = request.size;
		allocation->reserved_size = request.size;
		allocation->resolved_storage = storage;
		allocation->allocation_pool_id = request.pool;
		allocation->heap_placed = true;
		allocation->purgeable_allowed = false;
		m_impl->state->account(request.size, request.pool, false);
		allocation->accounted = true;
		notify_memory_allocated(allocation.get(), request.size, request.pool);
		allocation->tracking_registered = true;
		return memory_allocation(std::move(allocation));
	}

	void memory_allocator::trim(memory_pressure pressure)
	{
		if (m_impl && m_impl->state)
		{
			m_impl->state->trim(pressure);
		}
	}

	memory_usage memory_allocator::usage() const
	{
		if (!m_impl || !m_impl->state)
		{
			fmt::throw_exception("Memory usage requested from an empty Metal allocator");
		}

		std::lock_guard lock(m_impl->state->mutex);
		return m_impl->state->stats;
	}

	const render_device& memory_allocator::device() const
	{
		if (!m_impl || !m_impl->state || !m_impl->state->render)
		{
			fmt::throw_exception("Render device requested from an empty Metal allocator");
		}

		return *m_impl->state->render;
	}

	memory_block::memory_block(memory_allocator& allocator, const memory_allocation_request& request)
		: m_allocation(allocator.allocate_buffer(request))
	{
	}

	u64 memory_block::size() const
	{
		return m_allocation.size();
	}

	buffer_handle memory_block::buffer() const
	{
		return m_allocation.buffer();
	}

	void* memory_block::map(u64 offset, u64 size)
	{
		return m_allocation.map(offset, size);
	}

	void memory_block::unmap()
	{
		m_allocation.unmap();
	}

	void memory_block::did_modify(u64 offset, u64 size)
	{
		m_allocation.did_modify(offset, size);
	}

	const memory_allocation& memory_block::allocation() const
	{
		return m_allocation;
	}

	memory_pressure determine_memory_pressure(const memory_usage& usage)
	{
		const f32 ratio = usage.pressure_ratio();
		if (ratio >= 0.95f)
		{
			return memory_pressure::critical;
		}

		if (ratio >= 0.80f)
		{
			return memory_pressure::warning;
		}

		return memory_pressure::normal;
	}
}
