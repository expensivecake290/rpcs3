#include "stdafx.h"
#include "data_heap.h"

#include "barriers.h"
#include "shared.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <mutex>

namespace mtl
{
	namespace
	{
		struct dirty_range
		{
			u64 offset = 0;
			u64 size = 0;
		};

		struct retirement_checkpoint
		{
			u64 submission = 0;
			u64 cursor = 0;
		};

		struct heap_generation
		{
			std::unique_ptr<buffer> target;
			std::unique_ptr<buffer> staging;
			void* mapped_base = nullptr;
			u64 capacity = 0;
			u64 usable_capacity = 0;
			u64 write_cursor = 0;
			u64 reclaim_cursor = 0;
			u64 identifier = 0;
			u64 last_submission = 0;
			std::deque<retirement_checkpoint> checkpoints;
			std::vector<dirty_range> dirty;
			bool has_unsealed_allocations = false;
		};

		bool is_power_of_two(u64 value)
		{
			return value && (value & (value - 1)) == 0;
		}

		u64 checked_add(u64 left, u64 right)
		{
			if (left > std::numeric_limits<u64>::max() - right)
			{
				fmt::throw_exception("Metal data-heap size overflows");
			}
			return left + right;
		}

		u64 align_up(u64 value, u64 alignment)
		{
			if (!is_power_of_two(alignment))
			{
				fmt::throw_exception("Metal data-heap alignment must be a power of two");
			}
			return checked_add(value, alignment - 1) & ~(alignment - 1);
		}

		void close_mapping(heap_generation& generation)
		{
			if (!generation.mapped_base)
			{
				return;
			}
			buffer* mapped_buffer = generation.staging ? generation.staging.get() : generation.target.get();
			mapped_buffer->unmap();
			generation.mapped_base = nullptr;
		}

		void* ensure_mapping(heap_generation& generation)
		{
			if (!generation.mapped_base)
			{
				buffer* mapped_buffer = generation.staging ? generation.staging.get() : generation.target.get();
				generation.mapped_base = mapped_buffer->map(0, generation.capacity);
			}
			return generation.mapped_base;
		}

		void add_dirty_range(heap_generation& generation, u64 offset, u64 size)
		{
			if (!generation.staging || size == 0)
			{
				return;
			}

			u64 first = offset;
			u64 last = checked_add(offset, size);
			for (auto iterator = generation.dirty.begin(); iterator != generation.dirty.end();)
			{
				const u64 range_last = iterator->offset + iterator->size;
				if (last < iterator->offset || first > range_last)
				{
					++iterator;
					continue;
				}
				first = std::min(first, iterator->offset);
				last = std::max(last, range_last);
				iterator = generation.dirty.erase(iterator);
			}
			generation.dirty.push_back({first, last - first});
			std::sort(generation.dirty.begin(), generation.dirty.end(), [](const dirty_range& left, const dirty_range& right)
			{
				return left.offset < right.offset;
			});
		}

		u64 used_bytes(const heap_generation& generation)
		{
			ensure(generation.write_cursor >= generation.reclaim_cursor);
			return generation.write_cursor - generation.reclaim_cursor;
		}

		std::optional<u64> reserve_range(heap_generation& generation, u64 size, u64 alignment)
		{
			const u64 cycle = generation.write_cursor / generation.usable_capacity;
			const u64 cycle_base = cycle * generation.usable_capacity;
			const u64 position = generation.write_cursor - cycle_base;
			u64 aligned_cursor = checked_add(cycle_base, align_up(position, alignment));
			if (aligned_cursor - cycle_base > generation.usable_capacity - size)
			{
				aligned_cursor = checked_add(cycle_base, generation.usable_capacity);
			}
			const u64 end_cursor = checked_add(aligned_cursor, size);
			if (end_cursor - generation.reclaim_cursor > generation.usable_capacity)
			{
				return std::nullopt;
			}
			generation.write_cursor = end_cursor;
			generation.has_unsealed_allocations = true;
			return aligned_cursor % generation.usable_capacity;
		}

		id<MTL4ComputeCommandEncoder> get_compute_encoder(command_buffer& command)
		{
			if (!command.is_recording())
			{
				fmt::throw_exception("Metal data-heap flush requires active command recording");
			}
			if (command.active_encoder() == encoder_kind::render)
			{
				command.end_encoding();
			}
			if (command.active_encoder() == encoder_kind::none)
			{
				return (__bridge id<MTL4ComputeCommandEncoder>)command.begin_compute_encoding();
			}
			return (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder();
		}
	}

	struct data_heap::impl
	{
		memory_allocator* allocator = nullptr;
		data_heap_create_info creation;
		std::unique_ptr<heap_generation> current;
		std::vector<std::unique_ptr<heap_generation>> retired;
		mutable std::mutex mutex;
		u64 next_generation = 1;
		u64 peak_allocated = 0;
		u64 growth_count = 0;
		u64 last_sealed_submission = 0;

		std::unique_ptr<heap_generation> make_generation(u64 capacity)
		{
			auto result = std::make_unique<heap_generation>();
			result->capacity = capacity;
			result->usable_capacity = capacity - creation.guard_size;
			result->identifier = next_generation++;

			const bool shadowed = (creation.flags & data_heap_force_private_shadow) != 0;
			buffer_create_info target_info;
			target_info.size = capacity;
			target_info.usage = creation.usage | (shadowed ? buffer_usage_copy_destination : 0);
			target_info.storage = shadowed ? storage_mode::private_ :
				((creation.flags & data_heap_low_latency) ? storage_mode::shared : storage_mode::automatic);
			target_info.cache = shadowed ? cpu_cache_mode::default_cache : cpu_cache_mode::write_combined;
			target_info.access = shadowed ? cpu_access::none : cpu_access::read_write;
			target_info.pool = creation.pool;
			target_info.label = fmt::format("%s generation %llu", creation.label, result->identifier);
			target_info.use_placement_heap = false;
			result->target = std::make_unique<buffer>(*allocator, target_info);

			if (shadowed)
			{
				buffer_create_info staging_info;
				staging_info.size = capacity;
				staging_info.usage = buffer_usage_copy_source;
				staging_info.storage = storage_mode::shared;
				staging_info.cache = cpu_cache_mode::write_combined;
				staging_info.access = cpu_access::write;
				staging_info.pool = creation.pool;
				staging_info.label = fmt::format("%s staging generation %llu", creation.label, result->identifier);
				staging_info.use_placement_heap = false;
				result->staging = std::make_unique<buffer>(*allocator, staging_info);
			}

			if (creation.flags & data_heap_persistent_mapping)
			{
				static_cast<void>(ensure_mapping(*result));
			}
			return result;
		}

		heap_generation* find_generation(u64 identifier) const
		{
			if (current && current->identifier == identifier) return current.get();
			for (const auto& generation : retired)
			{
				if (generation->identifier == identifier) return generation.get();
			}
			return nullptr;
		}

		u64 total_allocated() const
		{
			u64 result = current ? used_bytes(*current) : 0;
			for (const auto& generation : retired) result += used_bytes(*generation);
			return result;
		}

		void update_peak()
		{
			peak_allocated = std::max(peak_allocated, total_allocated());
		}

		void reclaim_generation(heap_generation& generation, u64 completed)
		{
			while (!generation.checkpoints.empty() && generation.checkpoints.front().submission <= completed)
			{
				generation.reclaim_cursor = generation.checkpoints.front().cursor;
				generation.checkpoints.pop_front();
			}
		}

		void reclaim_locked(u64 completed)
		{
			if (current) reclaim_generation(*current, completed);
			for (auto& generation : retired) reclaim_generation(*generation, completed);
			for (auto iterator = retired.begin(); iterator != retired.end();)
			{
				heap_generation& generation = **iterator;
				if (!generation.has_unsealed_allocations && generation.checkpoints.empty() &&
					generation.reclaim_cursor == generation.write_cursor)
				{
					close_mapping(generation);
					iterator = retired.erase(iterator);
					continue;
				}
				++iterator;
			}
		}
	};

	data_heap::data_heap()
		: m_impl(std::make_unique<impl>())
	{
	}

	data_heap::~data_heap()
	{
		destroy();
	}

	void data_heap::create(memory_allocator& allocator, const data_heap_create_info& info)
	{
		destroy();
		if (info.initial_size == 0 || info.maximum_size < info.initial_size || info.growth_quantum == 0 ||
			info.guard_size >= info.initial_size || info.usage == buffer_usage_none || info.label.empty() ||
			!is_power_of_two(info.growth_quantum))
		{
			fmt::throw_exception("Invalid Metal data-heap creation information");
		}
		if ((info.flags & ~(data_heap_low_latency | data_heap_fixed_size | data_heap_force_private_shadow |
			data_heap_persistent_mapping)) != 0)
		{
			fmt::throw_exception("Metal data heap contains unknown flags");
		}

		if (!m_impl) m_impl = std::make_unique<impl>();
		std::function<void(u64, u64)> callback;
		u64 generation = 0;
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->allocator = &allocator;
			m_impl->creation = info;
			m_impl->next_generation = 1;
			m_impl->peak_allocated = 0;
			m_impl->growth_count = 0;
			m_impl->last_sealed_submission = 0;
			m_impl->current = m_impl->make_generation(info.initial_size);
			callback = info.growth_callback;
			generation = m_impl->current->identifier;
		}
		if (callback) callback(generation, info.initial_size);
	}

	void data_heap::destroy()
	{
		if (!m_impl) return;
		std::unique_ptr<heap_generation> current;
		std::vector<std::unique_ptr<heap_generation>> retired;
		{
			std::lock_guard lock(m_impl->mutex);
			if (m_impl->current) close_mapping(*m_impl->current);
			for (auto& generation : m_impl->retired) close_mapping(*generation);
			current = std::move(m_impl->current);
			retired = std::move(m_impl->retired);
			m_impl->allocator = nullptr;
			m_impl->creation = {};
			m_impl->peak_allocated = 0;
			m_impl->growth_count = 0;
			m_impl->last_sealed_submission = 0;
		}
		current.reset();
		retired.clear();
	}

	data_heap_slice data_heap::allocate(u64 size, u64 alignment, bool allow_growth)
	{
		if (!m_impl || !m_impl->current || size == 0 || !is_power_of_two(alignment))
		{
			fmt::throw_exception("Invalid Metal data-heap allocation request");
		}

		std::function<void(u64, u64)> callback;
		u64 callback_generation = 0;
		u64 callback_capacity = 0;
		data_heap_slice result;
		{
			std::lock_guard lock(m_impl->mutex);
			heap_generation* generation = m_impl->current.get();
			if (size > generation->usable_capacity)
			{
				if (size > m_impl->creation.maximum_size - m_impl->creation.guard_size)
				{
					fmt::throw_exception("Metal data-heap request exceeds its maximum usable capacity");
				}
			}

			std::optional<u64> offset = size <= generation->usable_capacity
				? reserve_range(*generation, size, alignment) : std::nullopt;
			if (!offset)
			{
				if (!allow_growth)
				{
					fmt::throw_exception("Metal data heap has no immediately available range for this allocation");
				}
				u64 requested_capacity = checked_add(generation->capacity, size);
				requested_capacity = align_up(requested_capacity, m_impl->creation.growth_quantum);
				requested_capacity = std::max(requested_capacity,
					align_up(checked_add(size, m_impl->creation.guard_size), m_impl->creation.growth_quantum));
				if ((m_impl->creation.flags & data_heap_fixed_size) || requested_capacity > m_impl->creation.maximum_size)
				{
					requested_capacity = (m_impl->creation.flags & data_heap_fixed_size)
						? m_impl->creation.initial_size : m_impl->creation.maximum_size;
				}
				if (requested_capacity <= m_impl->creation.guard_size ||
					size > requested_capacity - m_impl->creation.guard_size)
				{
					fmt::throw_exception("Metal data heap cannot grow or rotate to satisfy an allocation");
				}

				auto replacement = m_impl->make_generation(requested_capacity);
				auto old = std::move(m_impl->current);
				m_impl->current = std::move(replacement);
				m_impl->retired.push_back(std::move(old));
				m_impl->growth_count++;
				generation = m_impl->current.get();
				offset = reserve_range(*generation, size, alignment);
				ensure(offset.has_value());
				callback = m_impl->creation.growth_callback;
				callback_generation = generation->identifier;
				callback_capacity = generation->capacity;
			}

			void* mapped = generation->mapped_base;
			result.buffer = generation->target->native_handle();
			result.offset = *offset;
			result.size = size;
			result.gpu_address = generation->target->gpu_address() + *offset;
			result.cpu_address = mapped ? static_cast<u8*>(mapped) + *offset : nullptr;
			result.generation = generation->identifier;
			result.shadowed = generation->staging != nullptr;
			m_impl->update_peak();
		}
		if (callback) callback(callback_generation, callback_capacity);
		return result;
	}

	void* data_heap::map(const data_heap_slice& slice, u64 relative_offset, u64 size)
	{
		if (!slice || relative_offset >= slice.size || (size && size > slice.size - relative_offset))
		{
			fmt::throw_exception("Invalid Metal data-heap map range");
		}
		const u64 mapped_size = size ? size : slice.size - relative_offset;
		std::lock_guard lock(m_impl->mutex);
		heap_generation* generation = m_impl->find_generation(slice.generation);
		if (!generation || generation->target->native_handle() != slice.buffer)
		{
			fmt::throw_exception("Metal data-heap slice generation is no longer alive");
		}
		void* base = ensure_mapping(*generation);
		add_dirty_range(*generation, slice.offset + relative_offset, mapped_size);
		return static_cast<u8*>(base) + slice.offset + relative_offset;
	}

	void data_heap::mark_modified(const data_heap_slice& slice, u64 relative_offset, u64 size)
	{
		if (!slice || relative_offset >= slice.size || (size && size > slice.size - relative_offset))
		{
			fmt::throw_exception("Invalid Metal data-heap modification range");
		}
		const u64 modified_size = size ? size : slice.size - relative_offset;
		std::lock_guard lock(m_impl->mutex);
		heap_generation* generation = m_impl->find_generation(slice.generation);
		if (!generation || generation->target->native_handle() != slice.buffer)
		{
			fmt::throw_exception("Metal data-heap slice generation is no longer alive");
		}
		buffer* cpu_buffer = generation->staging ? generation->staging.get() : generation->target.get();
		cpu_buffer->did_modify(slice.offset + relative_offset, modified_size);
		add_dirty_range(*generation, slice.offset + relative_offset, modified_size);
	}

	void data_heap::write(const data_heap_slice& slice, std::span<const std::byte> data, u64 relative_offset)
	{
		if (data.empty() || relative_offset > slice.size || data.size() > slice.size - relative_offset)
		{
			fmt::throw_exception("Invalid Metal data-heap write range");
		}
		void* destination = map(slice, relative_offset, data.size());
		std::memcpy(destination, data.data(), data.size());
		mark_modified(slice, relative_offset, data.size());
	}

	void data_heap::unmap(bool force)
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		if (!force && (m_impl->creation.flags & data_heap_persistent_mapping)) return;
		if (m_impl->current) close_mapping(*m_impl->current);
		for (auto& generation : m_impl->retired) close_mapping(*generation);
	}

	void data_heap::flush(command_buffer& command)
	{
		std::lock_guard lock(m_impl->mutex);
		std::vector<heap_generation*> generations;
		if (m_impl->current) generations.push_back(m_impl->current.get());
		for (auto& generation : m_impl->retired) generations.push_back(generation.get());

		id<MTL4ComputeCommandEncoder> encoder = nil;
		bool encoded = false;
		for (heap_generation* generation : generations)
		{
			if (!generation->staging || generation->dirty.empty()) continue;
			if (!encoder) encoder = get_compute_encoder(command);
			id<MTLBuffer> source = generation->staging->native_handle();
			id<MTLBuffer> destination = generation->target->native_handle();
			command.retain_native_object((__bridge void*)source, true);
			command.retain_native_object((__bridge void*)destination, true);
			for (const dirty_range& range : generation->dirty)
			{
				[encoder copyFromBuffer:source sourceOffset:range.offset
					toBuffer:destination destinationOffset:range.offset size:range.size];
			}
			generation->dirty.clear();
			encoded = true;
		}

		if (encoded)
		{
			barrier_plan visibility;
			visibility.scope = barrier_scope::between_encoders;
			visibility.after_stages = stage_blit;
			visibility.before_stages = stage_all_gpu;
			visibility.flush_caches = true;
			visibility.end_encoder = true;
			visibility.producer_barrier = true;
			encode_barrier(command.active_native_encoder(), visibility);
			command.end_encoding();
			command.set_flag(command_has_blit_transfer);
		}
	}

	void data_heap::seal(u64 submission_value)
	{
		if (!m_impl || submission_value == 0)
		{
			fmt::throw_exception("Invalid Metal data-heap submission seal");
		}
		std::lock_guard lock(m_impl->mutex);
		if (submission_value < m_impl->last_sealed_submission)
		{
			fmt::throw_exception("Metal data-heap submissions must be sealed monotonically");
		}
		auto seal_generation = [&](heap_generation& generation)
		{
			if (!generation.has_unsealed_allocations) return;
			generation.checkpoints.push_back({submission_value, generation.write_cursor});
			generation.last_submission = submission_value;
			generation.has_unsealed_allocations = false;
		};
		seal_generation(*m_impl->current);
		for (auto& generation : m_impl->retired) seal_generation(*generation);
		m_impl->last_sealed_submission = submission_value;
	}

	void data_heap::reclaim(u64 completed_submission_value)
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->reclaim_locked(completed_submission_value);
	}

	void data_heap::trim(memory_pressure pressure)
	{
		if (!m_impl || !m_impl->allocator) return;
		m_impl->allocator->trim(pressure);
		std::function<void(u64, u64)> callback;
		u64 generation_id = 0;
		u64 capacity = 0;
		{
			std::lock_guard lock(m_impl->mutex);
			const u64 completed = get_shared_state() ? get_completed_submission_id() : 0;
			m_impl->reclaim_locked(completed);
			if (pressure != memory_pressure::critical || !m_impl->current ||
				m_impl->current->capacity <= m_impl->creation.initial_size ||
				m_impl->current->has_unsealed_allocations || !m_impl->current->checkpoints.empty() ||
				used_bytes(*m_impl->current) != 0)
			{
				return;
			}
			close_mapping(*m_impl->current);
			m_impl->current = m_impl->make_generation(m_impl->creation.initial_size);
			m_impl->growth_count++;
			callback = m_impl->creation.growth_callback;
			generation_id = m_impl->current->identifier;
			capacity = m_impl->current->capacity;
		}
		if (callback) callback(generation_id, capacity);
	}

	data_heap_window data_heap::window(const data_heap_slice& slice, u64 required_length,
		u64 window_size, u64 alignment) const
	{
		if (!slice || required_length == 0 || required_length > slice.size || window_size == 0 || !is_power_of_two(alignment))
		{
			fmt::throw_exception("Invalid Metal data-heap binding window");
		}
		std::lock_guard lock(m_impl->mutex);
		const heap_generation* generation = m_impl->find_generation(slice.generation);
		if (!generation || generation->target->native_handle() != slice.buffer)
		{
			fmt::throw_exception("Metal data-heap window references a retired generation");
		}
		if (window_size >= generation->capacity)
		{
			return {slice.buffer, 0, generation->capacity, generation->target->gpu_address(), slice.generation};
		}

		const u64 aligned_window = window_size & ~(alignment - 1);
		if (aligned_window == 0)
		{
			fmt::throw_exception("Metal data-heap window is smaller than its alignment");
		}
		const u64 required_end = checked_add(slice.offset, required_length);
		const u64 first_partition = slice.offset / aligned_window;
		const u64 last_partition = (required_end - 1) / aligned_window;
		if (first_partition == last_partition)
		{
			const u64 offset = first_partition * aligned_window;
			const u64 length = std::min(aligned_window, generation->capacity - offset);
			return {slice.buffer, offset, length, generation->target->gpu_address() + offset, slice.generation};
		}
		return {slice.buffer, slice.offset, required_length,
			generation->target->gpu_address() + slice.offset, slice.generation};
	}

	const buffer& data_heap::target_buffer() const
	{
		if (!m_impl || !m_impl->current || !m_impl->current->target)
		{
			fmt::throw_exception("Target buffer requested from an empty Metal data heap");
		}
		return *m_impl->current->target;
	}

	const buffer* data_heap::staging_buffer() const
	{
		return m_impl && m_impl->current ? m_impl->current->staging.get() : nullptr;
	}

	bool data_heap::is_dirty() const
	{
		if (!m_impl) return false;
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->current && !m_impl->current->dirty.empty()) return true;
		return std::any_of(m_impl->retired.begin(), m_impl->retired.end(), [](const auto& generation)
		{
			return !generation->dirty.empty();
		});
	}

	bool data_heap::has_shadow() const
	{
		return m_impl && m_impl->current && m_impl->current->staging;
	}

	u64 data_heap::size() const
	{
		return m_impl && m_impl->current ? m_impl->current->capacity : 0;
	}

	data_heap_statistics data_heap::statistics() const
	{
		data_heap_statistics result;
		if (!m_impl) return result;
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->current)
		{
			result.capacity = m_impl->current->capacity;
			result.generation = m_impl->current->identifier;
			result.shadowed = m_impl->current->staging != nullptr;
		}
		result.allocated = m_impl->total_allocated();
		result.peak_allocated = m_impl->peak_allocated;
		result.growth_count = m_impl->growth_count;
		result.retired_generations = m_impl->retired.size();
		auto add_generation = [&](const heap_generation& generation)
		{
			result.pending_batches += generation.checkpoints.size();
			for (const dirty_range& range : generation.dirty) result.dirty_bytes += range.size;
		};
		if (m_impl->current) add_generation(*m_impl->current);
		for (const auto& generation : m_impl->retired) add_generation(*generation);
		return result;
	}
}
