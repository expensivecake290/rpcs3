#include "stdafx.h"
#include "MTLDMA.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mtl
{
	namespace
	{
		using interval = std::pair<u32, u32>;
		constexpr u64 u32_address_space_size = 1ull << 32;

		std::atomic<u64> s_dma_generation{1};

		bool is_power_of_two(u64 value)
		{
			return value && !(value & (value - 1));
		}

		utils::address_range32 make_range(u32 address, u32 length)
		{
			if (!length || static_cast<u64>(address) + length > u32_address_space_size)
			{
				fmt::throw_exception("Invalid Metal DMA address range 0x%x + 0x%x", address, length);
			}
			return utils::address_range32::start_length(address, length);
		}

		u32 align_down(u32 value, u32 alignment)
		{
			return value & ~(alignment - 1);
		}

		u64 align_up(u64 value, u64 alignment)
		{
			return (value + alignment - 1) & ~(alignment - 1);
		}

		void add_interval(std::vector<interval>& intervals, u32 first, u32 last)
		{
			interval value{first, last};
			std::vector<interval> result;
			result.reserve(intervals.size() + 1);
			bool inserted = false;
			for (const auto& current : intervals)
			{
				if (static_cast<u64>(current.second) + 1 < value.first)
				{
					result.push_back(current);
				}
				else if (static_cast<u64>(value.second) + 1 < current.first)
				{
					if (!inserted)
					{
						result.push_back(value);
						inserted = true;
					}
					result.push_back(current);
				}
				else
				{
					value.first = std::min(value.first, current.first);
					value.second = std::max(value.second, current.second);
				}
			}
			if (!inserted)
			{
				result.push_back(value);
			}
			intervals = std::move(result);
		}

		void subtract_interval(std::vector<interval>& intervals, u32 first, u32 last)
		{
			std::vector<interval> result;
			result.reserve(intervals.size() + 1);
			for (const auto& current : intervals)
			{
				if (last < current.first || first > current.second)
				{
					result.push_back(current);
					continue;
				}
				if (first > current.first)
				{
					result.emplace_back(current.first, first - 1);
				}
				if (last < current.second)
				{
					result.emplace_back(last + 1, current.second);
				}
			}
			intervals = std::move(result);
		}

		bool intersects(const std::vector<interval>& intervals, u32 first, u32 last)
		{
			return std::any_of(intervals.begin(), intervals.end(), [&](const interval& value)
			{
				return value.first <= last && value.second >= first;
			});
		}
	}

	struct dma_block::impl
	{
		mutable std::mutex mutex;
		std::shared_ptr<dma_block> parent;
		std::unique_ptr<buffer> allocation;
		dma_host_callbacks host;
		dma_block_kind block_kind = dma_block_kind::staging;
		u32 base = 0;
		u64 length = 0;
		u64 block_generation = 0;
		u64 last_gpu_read = 0;
		u64 last_gpu_write = 0;
		u64 completed = 0;
		u64 load_count = 0;
		u64 flush_count = 0;
		u64 loaded_bytes = 0;
		u64 flushed_bytes = 0;
		std::vector<interval> host_dirty;
		std::vector<interval> gpu_dirty;
	};

	dma_block::dma_block()
		: m_impl(std::make_unique<impl>())
	{
	}

	dma_block::~dma_block() = default;

	void dma_block::create(memory_allocator& allocator, u32 base_address, u64 size,
		const dma_host_callbacks& host, const dma_pool_create_info& info)
	{
		destroy();
		if (!host || !size || static_cast<u64>(base_address) + size > u32_address_space_size ||
			!is_power_of_two(info.page_size) || !is_power_of_two(info.block_size) ||
			info.block_size < info.page_size || (base_address & (info.block_size - 1)) ||
			(size & (info.block_size - 1)))
		{
			fmt::throw_exception("Invalid Metal DMA block creation information");
		}

		std::lock_guard lock(m_impl->mutex);
		m_impl->host = host;
		m_impl->base = base_address;
		m_impl->length = size;
		m_impl->block_generation = s_dma_generation.fetch_add(1, std::memory_order_relaxed);
		if (!m_impl->block_generation)
		{
			m_impl->block_generation = s_dma_generation.fetch_add(1, std::memory_order_relaxed);
		}

		void* direct = info.allow_host_no_copy && host.direct_pointer
			? host.direct_pointer(base_address, size) : nullptr;
		if (direct && !(reinterpret_cast<uintptr_t>(direct) & (info.page_size - 1)) &&
			!(size & (info.page_size - 1)))
		{
			m_impl->allocation = std::make_unique<buffer>();
			m_impl->allocation->create_no_copy(allocator.device(), direct, size,
				buffer_usage_copy_source | buffer_usage_copy_destination | buffer_usage_storage,
				"RPCS3 Metal host DMA");
			m_impl->block_kind = dma_block_kind::host_no_copy;
			return;
		}

		buffer_create_info buffer_info;
		buffer_info.size = size;
		buffer_info.usage = buffer_usage_copy_source | buffer_usage_copy_destination | buffer_usage_storage;
		buffer_info.storage = storage_mode::shared;
		buffer_info.access = cpu_access::read_write;
		buffer_info.pool = info.allocation;
		buffer_info.label = "RPCS3 Metal staged DMA";
		buffer_info.use_placement_heap = false;
		m_impl->allocation = std::make_unique<buffer>(allocator, buffer_info);
		m_impl->block_kind = dma_block_kind::staging;
		if (info.preload_new_blocks)
		{
			void* destination = m_impl->allocation->map(0, size);
			if (!m_impl->host.read(base_address,
				std::span<u8>(static_cast<u8*>(destination), static_cast<usz>(size))))
			{
				m_impl->allocation->unmap();
				fmt::throw_exception("Metal DMA host read failed for 0x%x + 0x%llx", base_address, size);
			}
			m_impl->allocation->unmap();
			m_impl->allocation->did_modify(0, size);
			m_impl->load_count = 1;
			m_impl->loaded_bytes = size;
		}
		else
		{
			add_interval(m_impl->host_dirty, base_address,
				static_cast<u32>(static_cast<u64>(base_address) + size - 1));
		}
	}

	void dma_block::create_alias(std::shared_ptr<dma_block> parent, u32 base_address, u64 size)
	{
		destroy();
		if (!parent || !size || static_cast<u64>(base_address) + size > u32_address_space_size ||
			!parent->contains(utils::address_range32::start_length(base_address, static_cast<u32>(size))))
		{
			fmt::throw_exception("Invalid Metal DMA alias information");
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->parent = parent->head();
		m_impl->block_kind = dma_block_kind::alias;
		m_impl->base = base_address;
		m_impl->length = size;
		m_impl->block_generation = m_impl->parent->generation();
	}

	void dma_block::destroy()
	{
		if (!m_impl)
		{
			return;
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->parent.reset();
		m_impl->allocation.reset();
		m_impl->host = {};
		m_impl->block_kind = dma_block_kind::staging;
		m_impl->base = 0;
		m_impl->length = 0;
		m_impl->block_generation = 0;
		m_impl->last_gpu_read = 0;
		m_impl->last_gpu_write = 0;
		m_impl->completed = 0;
		m_impl->load_count = 0;
		m_impl->flush_count = 0;
		m_impl->loaded_bytes = 0;
		m_impl->flushed_bytes = 0;
		m_impl->host_dirty.clear();
		m_impl->gpu_dirty.clear();
	}

	void dma_block::extend(memory_allocator& allocator, u64 new_size,
		const dma_host_callbacks& host, const dma_pool_create_info& info)
	{
		if (kind() == dma_block_kind::alias)
		{
			fmt::throw_exception("Cannot extend a Metal DMA alias");
		}
		const u32 base_address = start();
		const u64 current_size = size();
		if (new_size <= current_size)
		{
			return;
		}
		flush(utils::address_range32::start_length(base_address, static_cast<u32>(current_size)));
		create(allocator, base_address, new_size, host, info);
	}

	dma_mapping_handle dma_block::map(const utils::address_range32& range)
	{
		auto owner = head();
		if (!owner->contains(range))
		{
			fmt::throw_exception("Metal DMA mapping lies outside its head block");
		}
		return {owner, owner->resource(), range.start - owner->start(), range.start,
			range.length(), owner->generation()};
	}

	void dma_block::load(const utils::address_range32& range)
	{
		auto owner = head();
		if (owner.get() != this)
		{
			owner->load(range);
			return;
		}
		std::lock_guard lock(m_impl->mutex);
		const bool inside = m_impl->length && range.start >= m_impl->base &&
			static_cast<u64>(range.end) < static_cast<u64>(m_impl->base) + m_impl->length;
		if (!inside || m_impl->last_gpu_read > m_impl->completed ||
			m_impl->last_gpu_write > m_impl->completed)
		{
			fmt::throw_exception("Metal DMA load overlaps unfinished GPU access or lies outside the block");
		}
		if (m_impl->block_kind == dma_block_kind::staging)
		{
			const u64 offset = range.start - m_impl->base;
			void* destination = m_impl->allocation->map(offset, range.length());
			if (!m_impl->host.read(range.start,
				std::span<u8>(static_cast<u8*>(destination), range.length())))
			{
				m_impl->allocation->unmap();
				fmt::throw_exception("Metal DMA host load failed at 0x%x", range.start);
			}
			m_impl->allocation->unmap();
			m_impl->allocation->did_modify(offset, range.length());
		}
		subtract_interval(m_impl->host_dirty, range.start, range.end);
		subtract_interval(m_impl->gpu_dirty, range.start, range.end);
		++m_impl->load_count;
		m_impl->loaded_bytes += range.length();
	}

	void dma_block::flush(const utils::address_range32& range)
	{
		auto owner = head();
		if (owner.get() != this)
		{
			owner->flush(range);
			return;
		}
		std::lock_guard lock(m_impl->mutex);
		const bool inside = m_impl->length && range.start >= m_impl->base &&
			static_cast<u64>(range.end) < static_cast<u64>(m_impl->base) + m_impl->length;
		if (!inside || m_impl->last_gpu_write > m_impl->completed)
		{
			fmt::throw_exception("Metal DMA flush overlaps unfinished GPU writes or lies outside the block");
		}
		if (m_impl->block_kind == dma_block_kind::staging &&
			intersects(m_impl->gpu_dirty, range.start, range.end))
		{
			const u64 offset = range.start - m_impl->base;
			void* source = m_impl->allocation->map(offset, range.length());
			const bool written = m_impl->host.write(range.start,
				std::span<const u8>(static_cast<const u8*>(source), range.length()));
			m_impl->allocation->unmap();
			if (!written)
			{
				fmt::throw_exception("Metal DMA host flush failed at 0x%x", range.start);
			}
		}
		subtract_interval(m_impl->gpu_dirty, range.start, range.end);
		++m_impl->flush_count;
		m_impl->flushed_bytes += range.length();
	}

	void dma_block::mark_host_modified(const utils::address_range32& range)
	{
		auto owner = head();
		if (owner.get() != this)
		{
			owner->mark_host_modified(range);
			return;
		}
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->length || range.start < m_impl->base ||
			static_cast<u64>(range.end) >= static_cast<u64>(m_impl->base) + m_impl->length)
		{
			fmt::throw_exception("Metal DMA host modification lies outside the block");
		}
		if (m_impl->block_kind == dma_block_kind::staging)
		{
			add_interval(m_impl->host_dirty, range.start, range.end);
		}
		else
		{
			m_impl->allocation->did_modify(range.start - m_impl->base, range.length());
		}
	}

	void dma_block::mark_gpu_access(const utils::address_range32& range,
		dma_access access, u64 submission_value)
	{
		auto owner = head();
		if (owner.get() != this)
		{
			owner->mark_gpu_access(range, access, submission_value);
			return;
		}
		std::lock_guard lock(m_impl->mutex);
		const bool inside = m_impl->length && range.start >= m_impl->base &&
			static_cast<u64>(range.end) < static_cast<u64>(m_impl->base) + m_impl->length;
		if (!inside || !submission_value || submission_value < m_impl->completed)
		{
			fmt::throw_exception("Invalid Metal DMA GPU access tracking");
		}
		if (access == dma_access::gpu_read || access == dma_access::gpu_read_write)
		{
			if (intersects(m_impl->host_dirty, range.start, range.end))
			{
				fmt::throw_exception("Metal DMA GPU read requires host changes to be loaded first");
			}
			m_impl->last_gpu_read = std::max(m_impl->last_gpu_read, submission_value);
		}
		if (access == dma_access::gpu_write || access == dma_access::gpu_read_write)
		{
			m_impl->last_gpu_write = std::max(m_impl->last_gpu_write, submission_value);
			subtract_interval(m_impl->host_dirty, range.start, range.end);
			add_interval(m_impl->gpu_dirty, range.start, range.end);
		}
	}

	void dma_block::notify_completed(u64 completed_submission_value)
	{
		auto owner = head();
		if (owner.get() != this)
		{
			owner->notify_completed(completed_submission_value);
			return;
		}
		std::lock_guard lock(m_impl->mutex);
		if (completed_submission_value < m_impl->completed)
		{
			fmt::throw_exception("Metal DMA completion value moved backwards");
		}
		m_impl->completed = completed_submission_value;
	}

	void dma_block::copy_to(command_buffer& command, buffer& destination, u64 destination_offset,
		const utils::address_range32& source_range)
	{
		auto owner = head();
		if (!owner->contains(source_range) || !destination.in_range(destination_offset, source_range.length()) ||
			!(destination.usage() & (buffer_usage_copy_destination | buffer_usage_storage)) ||
			!command.is_recording() || command.active_encoder() == encoder_kind::render)
		{
			fmt::throw_exception("Invalid Metal DMA source copy");
		}
		id<MTL4ComputeCommandEncoder> encoder = command.active_encoder() == encoder_kind::compute
			? (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder()
			: (__bridge id<MTL4ComputeCommandEncoder>)command.begin_compute_encoding();
		[encoder copyFromBuffer:owner->resource()->native_handle()
			sourceOffset:source_range.start - owner->start()
			toBuffer:destination.native_handle() destinationOffset:destination_offset size:source_range.length()];
		command.retain_native_object((__bridge void*)owner->resource()->native_handle(), true);
		command.retain_native_object((__bridge void*)destination.native_handle(), true);
		command.set_flag(command_has_dma_transfer);
	}

	void dma_block::copy_from(command_buffer& command, const buffer& source, u64 source_offset,
		const utils::address_range32& destination_range, u64 submission_value)
	{
		auto owner = head();
		if (!owner->contains(destination_range) || !source.in_range(source_offset, destination_range.length()) ||
			!(source.usage() & (buffer_usage_copy_source | buffer_usage_storage)) ||
			!command.is_recording() || command.active_encoder() == encoder_kind::render)
		{
			fmt::throw_exception("Invalid Metal DMA destination copy");
		}
		id<MTL4ComputeCommandEncoder> encoder = command.active_encoder() == encoder_kind::compute
			? (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder()
			: (__bridge id<MTL4ComputeCommandEncoder>)command.begin_compute_encoding();
		[encoder copyFromBuffer:source.native_handle() sourceOffset:source_offset
			toBuffer:owner->resource()->native_handle()
			destinationOffset:destination_range.start - owner->start() size:destination_range.length()];
		command.retain_native_object((__bridge void*)source.native_handle(), true);
		command.retain_native_object((__bridge void*)owner->resource()->native_handle(), true);
		command.set_flag(command_has_dma_transfer);
		owner->mark_gpu_access(destination_range, dma_access::gpu_write, submission_value);
	}

	bool dma_block::contains(const utils::address_range32& range) const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->length && range.start >= m_impl->base &&
			static_cast<u64>(range.end) < static_cast<u64>(m_impl->base) + m_impl->length;
	}

	bool dma_block::overlaps(const utils::address_range32& range) const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->length && range.start < static_cast<u64>(m_impl->base) + m_impl->length &&
			range.end >= m_impl->base;
	}

	u32 dma_block::start() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->base;
	}

	u32 dma_block::end() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->length ? static_cast<u32>(static_cast<u64>(m_impl->base) + m_impl->length - 1) : 0;
	}

	u64 dma_block::size() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->length;
	}

	u64 dma_block::generation() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->block_generation;
	}

	dma_block_kind dma_block::kind() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->block_kind;
	}

	buffer* dma_block::resource() const
	{
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->parent)
		{
			return m_impl->parent->resource();
		}
		return m_impl->allocation.get();
	}

	std::shared_ptr<dma_block> dma_block::head()
	{
		std::shared_ptr<dma_block> parent;
		{
			std::lock_guard lock(m_impl->mutex);
			parent = m_impl->parent;
		}
		return parent ? parent->head() : shared_from_this();
	}

	dma_block_statistics dma_block::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->parent)
		{
			return m_impl->parent->statistics();
		}
		return {
			.kind = m_impl->block_kind,
			.base_address = m_impl->base,
			.size = m_impl->length,
			.generation = m_impl->block_generation,
			.loads = m_impl->load_count,
			.flushes = m_impl->flush_count,
			.loaded_bytes = m_impl->loaded_bytes,
			.flushed_bytes = m_impl->flushed_bytes,
			.last_gpu_read_submission = m_impl->last_gpu_read,
			.last_gpu_write_submission = m_impl->last_gpu_write,
			.completed_submission = m_impl->completed,
			.host_dirty = !m_impl->host_dirty.empty(),
			.gpu_dirty = !m_impl->gpu_dirty.empty(),
		};
	}

	struct dma_pool::impl
	{
		mutable std::mutex mutex;
		memory_allocator* allocator = nullptr;
		dma_host_callbacks host;
		dma_pool_create_info creation;
		std::unordered_map<u32, std::shared_ptr<dma_block>> entries;
		std::unordered_map<dma_block*, u32> idle_checks;
		dma_pool_statistics stats;
	};

	dma_pool::dma_pool()
		: m_impl(std::make_unique<impl>())
	{
	}

	dma_pool::dma_pool(memory_allocator& allocator, dma_host_callbacks host,
		const dma_pool_create_info& info)
		: dma_pool()
	{
		create(allocator, std::move(host), info);
	}

	dma_pool::~dma_pool() = default;

	void dma_pool::create(memory_allocator& allocator, dma_host_callbacks host,
		const dma_pool_create_info& info)
	{
		destroy();
		if (!host || !is_power_of_two(info.block_size) || !is_power_of_two(info.page_size) ||
			info.block_size < info.page_size || !info.maximum_cached_bytes)
		{
			fmt::throw_exception("Invalid Metal DMA pool creation information");
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->allocator = &allocator;
		m_impl->host = std::move(host);
		m_impl->creation = info;
		m_impl->stats = {};
	}

	void dma_pool::destroy()
	{
		if (!m_impl)
		{
			return;
		}
		bool has_entries = false;
		{
			std::lock_guard lock(m_impl->mutex);
			has_entries = !m_impl->entries.empty();
		}
		if (has_entries)
		{
			clear();
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->idle_checks.clear();
		m_impl->allocator = nullptr;
		m_impl->host = {};
		m_impl->creation = {};
		m_impl->stats = {};
	}

	dma_mapping_handle dma_pool::map(u32 address, u32 length)
	{
		const auto requested = make_range(address, length);
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->allocator)
		{
			fmt::throw_exception("Cannot map an empty Metal DMA pool");
		}
		++m_impl->stats.mappings;
		const u32 block_size = m_impl->creation.block_size;
		const u32 requested_first = align_down(requested.start, block_size);
		const u64 requested_end_exclusive = static_cast<u64>(requested.end) + 1;
		const u64 requested_aligned_end = align_up(requested_end_exclusive, block_size);
		if (requested_aligned_end > u32_address_space_size)
		{
			fmt::throw_exception("Metal DMA mapping alignment exceeds the address space");
		}

		if (const auto found = m_impl->entries.find(requested_first); found != m_impl->entries.end())
		{
			auto head = found->second->head();
			if (head->contains(requested))
			{
				++m_impl->stats.cache_hits;
				m_impl->idle_checks[head.get()] = 0;
				return head->map(requested);
			}
		}

		++m_impl->stats.cache_misses;
		u32 merged_first = requested_first;
		u64 merged_end = requested_aligned_end;
		std::unordered_set<dma_block*> seen;
		std::vector<std::shared_ptr<dma_block>> overlapping;
		bool changed = true;
		while (changed)
		{
			changed = false;
			for (const auto& [base, entry] : m_impl->entries)
			{
				static_cast<void>(base);
				auto head = entry->head();
				if (seen.contains(head.get()))
				{
					continue;
				}
				const u64 head_end = static_cast<u64>(head->end()) + 1;
				if (head->start() < merged_end && head_end > merged_first)
				{
					seen.insert(head.get());
					overlapping.push_back(head);
					const u32 new_first = std::min(merged_first, head->start());
					const u64 new_end = std::max(merged_end, head_end);
					changed = changed || new_first != merged_first || new_end != merged_end;
					merged_first = new_first;
					merged_end = new_end;
				}
			}
		}

		for (const auto& old : overlapping)
		{
			old->flush(utils::address_range32::start_length(old->start(), static_cast<u32>(old->size())));
		}
		if (!overlapping.empty())
		{
			++m_impl->stats.merges;
		}

		const u64 merged_size = merged_end - merged_first;
		u64 retained_bytes = 0;
		std::unordered_set<dma_block*> retained_heads;
		for (const auto& [base, entry] : m_impl->entries)
		{
			static_cast<void>(base);
			auto existing_head = entry->head();
			if (!seen.contains(existing_head.get()) && retained_heads.insert(existing_head.get()).second)
			{
				retained_bytes += existing_head->size();
			}
		}
		if (merged_size > m_impl->creation.maximum_cached_bytes ||
			retained_bytes > m_impl->creation.maximum_cached_bytes - merged_size)
		{
			fmt::throw_exception("Metal DMA mapping exceeds the configured cache budget");
		}
		auto head = std::make_shared<dma_block>();
		head->create(*m_impl->allocator, merged_first, merged_size, m_impl->host, m_impl->creation);

		for (auto current = m_impl->entries.begin(); current != m_impl->entries.end();)
		{
			auto old_head = current->second->head();
			if (seen.contains(old_head.get()))
			{
				current = m_impl->entries.erase(current);
			}
			else
			{
				++current;
			}
		}
		for (u64 block = merged_first; block < merged_end; block += block_size)
		{
			const u32 block_address = static_cast<u32>(block);
			if (block == merged_first)
			{
				m_impl->entries[block_address] = head;
			}
			else
			{
				auto alias = std::make_shared<dma_block>();
				alias->create_alias(head, block_address, block_size);
				m_impl->entries[block_address] = std::move(alias);
			}
		}
		m_impl->idle_checks[head.get()] = 0;
		return head->map(requested);
	}

	void dma_pool::load(u32 address, u32 length)
	{
		const auto range = make_range(address, length);
		auto mapping = map(address, length);
		mapping.owner->load(range);
		std::lock_guard lock(m_impl->mutex);
		++m_impl->stats.loads;
	}

	void dma_pool::flush(u32 address, u32 length)
	{
		const auto range = make_range(address, length);
		std::shared_ptr<dma_block> owner;
		{
			std::lock_guard lock(m_impl->mutex);
			const auto found = m_impl->entries.find(align_down(address, m_impl->creation.block_size));
			if (found == m_impl->entries.end() || !(owner = found->second->head())->contains(range))
			{
				fmt::throw_exception("Metal DMA flush range is not mapped");
			}
		}
		owner->flush(range);
		std::lock_guard lock(m_impl->mutex);
		++m_impl->stats.flushes;
	}

	void dma_pool::mark_host_modified(u32 address, u32 length)
	{
		const auto range = make_range(address, length);
		auto mapping = map(address, length);
		mapping.owner->mark_host_modified(range);
	}

	void dma_pool::mark_gpu_access(const dma_mapping_handle& mapping,
		dma_access access, u64 submission_value)
	{
		if (!mapping || mapping.owner->generation() != mapping.generation)
		{
			fmt::throw_exception("Invalid or stale Metal DMA mapping handle");
		}
		mapping.owner->mark_gpu_access(make_range(mapping.address, mapping.length), access, submission_value);
	}

	void dma_pool::notify_completed(u64 completed_submission_value)
	{
		std::vector<std::shared_ptr<dma_block>> heads;
		{
			std::lock_guard lock(m_impl->mutex);
			if (completed_submission_value < m_impl->stats.completed_submission)
			{
				fmt::throw_exception("Metal DMA pool completion value moved backwards");
			}
			m_impl->stats.completed_submission = completed_submission_value;
			std::unordered_set<dma_block*> seen;
			for (const auto& [base, entry] : m_impl->entries)
			{
				static_cast<void>(base);
				auto head = entry->head();
				if (seen.insert(head.get()).second) heads.push_back(std::move(head));
			}
		}
		for (const auto& head : heads) head->notify_completed(completed_submission_value);
	}

	usz dma_pool::invalidate(u32 address, u32 length)
	{
		const auto range = make_range(address, length);
		std::lock_guard lock(m_impl->mutex);
		std::unordered_set<dma_block*> selected;
		std::vector<std::shared_ptr<dma_block>> heads;
		for (const auto& [base, entry] : m_impl->entries)
		{
			static_cast<void>(base);
			auto head = entry->head();
			if (head->overlaps(range) && selected.insert(head.get()).second)
			{
				heads.push_back(std::move(head));
			}
		}
		for (const auto& head : heads)
		{
			head->flush(utils::address_range32::start_length(head->start(), static_cast<u32>(head->size())));
		}
		usz removed = 0;
		for (auto current = m_impl->entries.begin(); current != m_impl->entries.end();)
		{
			if (selected.contains(current->second->head().get()))
			{
				current = m_impl->entries.erase(current);
				++removed;
			}
			else
			{
				++current;
			}
		}
		for (dma_block* head : selected) m_impl->idle_checks.erase(head);
		m_impl->stats.invalidations += removed;
		return removed;
	}

	usz dma_pool::trim(u32 required_idle_checks)
	{
		if (!required_idle_checks)
		{
			fmt::throw_exception("Metal DMA idle-check threshold must be nonzero");
		}
		std::lock_guard lock(m_impl->mutex);
		std::unordered_set<dma_block*> visited;
		std::unordered_set<dma_block*> selected;
		std::vector<std::shared_ptr<dma_block>> heads;
		for (const auto& [base, entry] : m_impl->entries)
		{
			static_cast<void>(base);
			auto head = entry->head();
			if (!visited.insert(head.get()).second) continue;
			u32& idle = m_impl->idle_checks[head.get()];
			if (++idle >= required_idle_checks)
			{
				selected.insert(head.get());
				heads.push_back(std::move(head));
			}
		}
		for (const auto& head : heads)
		{
			head->flush(utils::address_range32::start_length(head->start(), static_cast<u32>(head->size())));
		}
		usz removed = 0;
		for (auto current = m_impl->entries.begin(); current != m_impl->entries.end();)
		{
			if (selected.contains(current->second->head().get()))
			{
				current = m_impl->entries.erase(current);
				++removed;
			}
			else
			{
				++current;
			}
		}
		for (dma_block* head : selected) m_impl->idle_checks.erase(head);
		return removed;
	}

	void dma_pool::clear()
	{
		std::lock_guard lock(m_impl->mutex);
		std::unordered_set<dma_block*> seen;
		for (const auto& [base, entry] : m_impl->entries)
		{
			static_cast<void>(base);
			auto head = entry->head();
			if (seen.insert(head.get()).second)
			{
				head->flush(utils::address_range32::start_length(head->start(), static_cast<u32>(head->size())));
			}
		}
		m_impl->entries.clear();
		m_impl->idle_checks.clear();
	}

	dma_pool::operator bool() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->allocator && static_cast<bool>(m_impl->host);
	}

	dma_pool_statistics dma_pool::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		auto result = m_impl->stats;
		result.blocks = m_impl->entries.size();
		std::unordered_set<dma_block*> seen;
		for (const auto& [base, entry] : m_impl->entries)
		{
			static_cast<void>(base);
			auto head = entry->head();
			if (seen.insert(head.get()).second)
			{
				++result.head_blocks;
				result.allocated_bytes += head->size();
			}
		}
		return result;
	}

	namespace
	{
		std::mutex s_dma_pool_mutex;
		std::unique_ptr<dma_pool> s_dma_pool;
	}

	void initialize_dma_pool(memory_allocator& allocator, dma_host_callbacks host,
		const dma_pool_create_info& info)
	{
		std::lock_guard lock(s_dma_pool_mutex);
		if (s_dma_pool)
		{
			fmt::throw_exception("Metal DMA pool is already initialized");
		}
		s_dma_pool = std::make_unique<dma_pool>(allocator, std::move(host), info);
	}

	void shutdown_dma_pool()
	{
		std::lock_guard lock(s_dma_pool_mutex);
		if (s_dma_pool)
		{
			s_dma_pool->clear();
			s_dma_pool.reset();
		}
	}

	dma_pool& get_dma_pool()
	{
		std::lock_guard lock(s_dma_pool_mutex);
		if (!s_dma_pool)
		{
			fmt::throw_exception("Metal DMA pool is not initialized");
		}
		return *s_dma_pool;
	}

	dma_mapping_handle map_dma(u32 address, u32 length)
	{
		return get_dma_pool().map(address, length);
	}

	void load_dma(u32 address, u32 length)
	{
		get_dma_pool().load(address, length);
	}

	void flush_dma(u32 address, u32 length)
	{
		get_dma_pool().flush(address, length);
	}

	usz unmap_dma(u32 address, u32 length)
	{
		return get_dma_pool().invalidate(address, length);
	}

	void notify_dma_completed(u64 completed_submission_value)
	{
		get_dma_pool().notify_completed(completed_submission_value);
	}

	void clear_dma_resources()
	{
		get_dma_pool().clear();
	}
}
