#include "stdafx.h"
#include "MTLQueryPool.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

namespace mtl
{
	namespace
	{
		struct query_slot
		{
			u32 generation = 0;
			query_state state = query_state::free;
			u64 submission = 0;
			bool resolved = false;
			bool failed = false;
		};

		MTLRenderStages to_render_stages(u64 stages)
		{
			MTLRenderStages result = 0;
			if (stages & stage_vertex) result |= MTLRenderStageVertex;
			if (stages & stage_fragment) result |= MTLRenderStageFragment;
			if (stages & stage_tile) result |= MTLRenderStageTile;
			if (stages & stage_object) result |= MTLRenderStageObject;
			if (stages & stage_mesh) result |= MTLRenderStageMesh;
			return result ? result : MTLRenderStageFragment;
		}

		visibility_result_mode visibility_mode_for(query_kind kind)
		{
			switch (kind)
			{
			case query_kind::occlusion_boolean: return visibility_result_mode::boolean;
			case query_kind::occlusion_counting: return visibility_result_mode::counting;
			case query_kind::timestamp:
				fmt::throw_exception("Timestamp queries do not use Metal visibility mode");
			}
			fmt::throw_exception("Invalid Metal query kind %u", static_cast<u8>(kind));
		}
	}

	struct query_pool::impl
	{
		mutable std::mutex mutex;
		render_device* device = nullptr;
		query_pool_create_info creation;
		std::vector<query_slot> slots;
		std::deque<u32> available;
		std::unique_ptr<buffer> visibility;
		std::unique_ptr<buffer> resolved_counters;
		id<MTL4CounterHeap> counter_heap;
		u64 stride = visibility_result_stride;
		u64 last_submission = 0;
	};

	query_pool::query_pool()
		: m_impl(std::make_unique<impl>())
	{
	}

	query_pool::query_pool(render_device& device, memory_allocator& allocator,
		const query_pool_create_info& info)
		: query_pool()
	{
		create(device, allocator, info);
	}

	query_pool::~query_pool()
	{
		destroy();
	}

	void query_pool::create(render_device& device, memory_allocator& allocator,
		const query_pool_create_info& info)
	{
		destroy();
		if (!device || &allocator.device() != &device || !info.capacity)
		{
			fmt::throw_exception("Invalid Metal query pool creation information");
		}
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		m_impl->device = &device;
		m_impl->creation = info;
		m_impl->slots.resize(info.capacity);
		for (u32 index = 0; index < info.capacity; ++index)
		{
			m_impl->available.push_back(index);
		}

		if (info.kind == query_kind::timestamp)
		{
			if (!device.info().features.counter_sampling)
			{
				fmt::throw_exception("Metal device does not support timestamp counter sampling");
			}
			MTL4CounterHeapDescriptor* descriptor = [MTL4CounterHeapDescriptor new];
			descriptor.type = MTL4CounterHeapTypeTimestamp;
			descriptor.count = info.capacity;
			NSError* native_error = nil;
			m_impl->counter_heap = [device.native_handle() newCounterHeapWithDescriptor:descriptor error:&native_error];
			if (!m_impl->counter_heap)
			{
				fmt::throw_exception("Metal rejected timestamp counter heap creation: %s",
					native_error.localizedDescription.UTF8String ?: "unknown error");
			}
			m_impl->counter_heap.label = [NSString stringWithUTF8String:
				(info.label.empty() ? "RPCS3 timestamp queries" : info.label.c_str())];
			m_impl->stride = [device.native_handle() sizeOfCounterHeapEntry:MTL4CounterHeapTypeTimestamp];
			if (m_impl->stride < sizeof(MTL4TimestampHeapEntry))
			{
				fmt::throw_exception("Metal returned an invalid timestamp counter stride");
			}
			buffer_create_info buffer_info;
			buffer_info.size = m_impl->stride * info.capacity;
			buffer_info.usage = buffer_usage_query | buffer_usage_copy_source | buffer_usage_copy_destination;
			buffer_info.storage = storage_mode::shared;
			buffer_info.access = cpu_access::read_write;
			buffer_info.pool = info.pool;
			buffer_info.label = info.label.empty() ? "RPCS3 resolved timestamp queries" : info.label + " resolved";
			buffer_info.use_placement_heap = false;
			m_impl->resolved_counters = std::make_unique<buffer>(allocator, buffer_info);
		}
		else
		{
			buffer_create_info buffer_info;
			buffer_info.size = visibility_result_stride * info.capacity;
			buffer_info.usage = buffer_usage_query | buffer_usage_copy_source | buffer_usage_copy_destination;
			buffer_info.storage = storage_mode::shared;
			buffer_info.access = cpu_access::read_write;
			buffer_info.pool = info.pool;
			buffer_info.label = info.label.empty() ? "RPCS3 visibility queries" : info.label;
			buffer_info.use_placement_heap = false;
			m_impl->visibility = std::make_unique<buffer>(allocator, buffer_info);
		}
	}

	void query_pool::destroy()
	{
		if (!m_impl)
		{
			return;
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->counter_heap = nil;
		m_impl->visibility.reset();
		m_impl->resolved_counters.reset();
		m_impl->slots.clear();
		m_impl->available.clear();
		m_impl->device = nullptr;
		m_impl->creation = {};
		m_impl->stride = visibility_result_stride;
		m_impl->last_submission = 0;
	}

	query_handle query_pool::allocate()
	{
		if (!*this)
		{
			fmt::throw_exception("Cannot allocate from an empty Metal query pool");
		}
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->available.empty())
		{
			return {};
		}
		const u32 index = m_impl->available.front();
		m_impl->available.pop_front();
		auto& slot = m_impl->slots[index];
		slot.generation = slot.generation == umax ? 1 : slot.generation + 1;
		slot.state = query_state::allocated;
		slot.submission = 0;
		slot.resolved = false;
		slot.failed = false;
		return {index, slot.generation};
	}

	void query_pool::release(query_handle query)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!query || query.index >= m_impl->slots.size() ||
			m_impl->slots[query.index].generation != query.generation)
		{
			fmt::throw_exception("Invalid Metal query release handle");
		}
		auto& slot = m_impl->slots[query.index];
		if (slot.state == query_state::free || slot.state == query_state::active ||
			slot.state == query_state::pending)
		{
			fmt::throw_exception("Metal query cannot be released in its current state");
		}
		slot.state = query_state::free;
		slot.submission = 0;
		slot.resolved = false;
		slot.failed = false;
		m_impl->available.push_back(query.index);
	}

	void query_pool::reset(query_handle query)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!query || query.index >= m_impl->slots.size() ||
			m_impl->slots[query.index].generation != query.generation)
		{
			fmt::throw_exception("Invalid Metal query reset handle");
		}
		auto& slot = m_impl->slots[query.index];
		if (slot.state == query_state::free || slot.state == query_state::active ||
			slot.state == query_state::pending)
		{
			fmt::throw_exception("Metal query cannot be reset in its current state");
		}
		if (m_impl->counter_heap)
		{
			[m_impl->counter_heap invalidateCounterRange:NSMakeRange(query.index, 1)];
		}
		buffer* storage = m_impl->visibility ? m_impl->visibility.get() : m_impl->resolved_counters.get();
		void* destination = storage->map(query.index * m_impl->stride, m_impl->stride);
		std::memset(destination, 0, m_impl->stride);
		storage->unmap();
		storage->did_modify(query.index * m_impl->stride, m_impl->stride);
		slot.state = query_state::allocated;
		slot.submission = 0;
		slot.resolved = false;
		slot.failed = false;
	}

	void query_pool::reset(u32 first, u32 count)
	{
		if (!count || first >= size() || count > size() - first)
		{
			fmt::throw_exception("Invalid Metal query reset range");
		}
		std::vector<query_handle> handles;
		{
			std::lock_guard lock(m_impl->mutex);
			handles.reserve(count);
			for (u32 index = first; index < first + count; ++index)
			{
				if (m_impl->slots[index].state != query_state::free)
				{
					handles.push_back({index, m_impl->slots[index].generation});
				}
			}
		}
		for (const auto query : handles)
		{
			reset(query);
		}
	}

	void query_pool::begin(command_buffer& command, query_handle query, bool accumulate)
	{
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->creation.kind == query_kind::timestamp || !command.is_recording() ||
			!query || query.index >= m_impl->slots.size() ||
			m_impl->slots[query.index].generation != query.generation)
		{
			fmt::throw_exception("Invalid Metal visibility query begin");
		}
		auto& slot = m_impl->slots[query.index];
		if (slot.state != query_state::allocated && !(accumulate && slot.state == query_state::ended))
		{
			fmt::throw_exception("Metal visibility query is not ready to begin");
		}
		if (!accumulate)
		{
			void* destination = m_impl->visibility->map(query.index * visibility_result_stride,
				visibility_result_stride);
			std::memset(destination, 0, visibility_result_stride);
			m_impl->visibility->unmap();
			m_impl->visibility->did_modify(query.index * visibility_result_stride, visibility_result_stride);
		}
		slot.state = query_state::active;
		slot.submission = 0;
		slot.resolved = false;
		slot.failed = false;
		command.set_flag(command_has_open_query);
		command.retain_native_object((__bridge void*)m_impl->visibility->native_handle(), true);
	}

	void query_pool::end(command_buffer& command, query_handle query)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!command.is_recording() || !query || query.index >= m_impl->slots.size() ||
			m_impl->slots[query.index].generation != query.generation ||
			m_impl->slots[query.index].state != query_state::active)
		{
			fmt::throw_exception("Invalid Metal query end");
		}
		auto& slot = m_impl->slots[query.index];
		slot.state = query_state::ended;
		slot.resolved = true;
		command.clear_flag(command_has_open_query);
	}

	void query_pool::write_timestamp(command_buffer& command, query_handle query, u64 stages)
	{
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->creation.kind != query_kind::timestamp || !command.is_recording() ||
			!query || query.index >= m_impl->slots.size() ||
			m_impl->slots[query.index].generation != query.generation ||
			m_impl->slots[query.index].state != query_state::allocated)
		{
			fmt::throw_exception("Invalid Metal timestamp query write");
		}

		if (command.active_encoder() == encoder_kind::render)
		{
			id<MTL4RenderCommandEncoder> encoder =
				(__bridge id<MTL4RenderCommandEncoder>)command.active_native_encoder();
			[encoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
				afterStage:to_render_stages(stages) intoHeap:m_impl->counter_heap atIndex:query.index];
		}
		else if (command.active_encoder() == encoder_kind::compute)
		{
			id<MTL4ComputeCommandEncoder> encoder =
				(__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder();
			[encoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
				intoHeap:m_impl->counter_heap atIndex:query.index];
		}
		else
		{
			[command.native_handle() writeTimestampIntoHeap:m_impl->counter_heap atIndex:query.index];
		}
		command.retain_native_object((__bridge void*)m_impl->counter_heap, false);
		auto& slot = m_impl->slots[query.index];
		slot.state = query_state::ended;
		slot.resolved = false;
	}

	void query_pool::resolve(command_buffer& command, u32 first, u32 count)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!command.is_recording() || !count || first >= m_impl->slots.size() ||
			count > m_impl->slots.size() - first)
		{
			fmt::throw_exception("Invalid Metal query resolve range");
		}
		if (m_impl->creation.kind != query_kind::timestamp)
		{
			return;
		}
		if (command.active_encoder() != encoder_kind::none)
		{
			fmt::throw_exception("Metal timestamp query resolve requires no active encoder");
		}
		for (u32 index = first; index < first + count; ++index)
		{
			if (m_impl->slots[index].state != query_state::ended)
			{
				fmt::throw_exception("Metal timestamp resolve range contains an unfinished query");
			}
		}
		const u64 offset = first * m_impl->stride;
		const u64 length = count * m_impl->stride;
		const MTL4BufferRange destination = MTL4BufferRangeMake(
			m_impl->resolved_counters->gpu_address() + offset, length);
		[command.native_handle() resolveCounterHeap:m_impl->counter_heap
			withRange:NSMakeRange(first, count) intoBuffer:destination waitFence:nil updateFence:nil];
		command.retain_native_object((__bridge void*)m_impl->counter_heap, false);
		command.retain_native_object((__bridge void*)m_impl->resolved_counters->native_handle(), true);
		for (u32 index = first; index < first + count; ++index)
		{
			m_impl->slots[index].resolved = true;
		}
		command.set_flag(command_has_blit_transfer);
	}

	void query_pool::mark_submitted(query_handle query, u64 submission_value)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!submission_value || !query || query.index >= m_impl->slots.size() ||
			m_impl->slots[query.index].generation != query.generation ||
			m_impl->slots[query.index].state != query_state::ended ||
			!m_impl->slots[query.index].resolved)
		{
			fmt::throw_exception("Invalid Metal submitted query state");
		}
		auto& slot = m_impl->slots[query.index];
		slot.state = query_state::pending;
		slot.submission = submission_value;
		m_impl->last_submission = std::max(m_impl->last_submission, submission_value);
	}

	void query_pool::notify_completed(u64 completed_submission_value, bool submission_succeeded)
	{
		std::lock_guard lock(m_impl->mutex);
		for (auto& slot : m_impl->slots)
		{
			if (slot.state == query_state::pending && slot.submission <= completed_submission_value)
			{
				slot.state = query_state::ready;
				slot.failed = !submission_succeeded;
			}
		}
	}

	query_state query_pool::state(query_handle query) const
	{
		std::lock_guard lock(m_impl->mutex);
		if (!query || query.index >= m_impl->slots.size() ||
			m_impl->slots[query.index].generation != query.generation)
		{
			fmt::throw_exception("Invalid Metal query state handle");
		}
		return m_impl->slots[query.index].state;
	}

	bool query_pool::available(query_handle query) const
	{
		return state(query) == query_state::ready;
	}

	query_result query_pool::result(query_handle query, bool allow_partial) const
	{
		std::lock_guard lock(m_impl->mutex);
		if (!query || query.index >= m_impl->slots.size() ||
			m_impl->slots[query.index].generation != query.generation)
		{
			fmt::throw_exception("Invalid Metal query result handle");
		}
		const auto& slot = m_impl->slots[query.index];
		query_result result;
		result.submission = slot.submission;
		if (slot.state != query_state::ready)
		{
			result.availability = allow_partial && slot.state == query_state::ended
				? query_availability::partial : query_availability::unavailable;
			if (result.availability != query_availability::partial)
			{
				return result;
			}
		}
		else if (slot.failed)
		{
			result.availability = query_availability::failed;
			return result;
		}
		else
		{
			result.availability = query_availability::available;
		}

		buffer* storage = m_impl->visibility ? m_impl->visibility.get() : m_impl->resolved_counters.get();
		void* source = storage->map(query.index * m_impl->stride, sizeof(u64));
		std::memcpy(&result.value, source, sizeof(result.value));
		storage->unmap();
		if (m_impl->creation.kind == query_kind::occlusion_boolean)
		{
			result.value = result.value ? 1 : 0;
		}
		return result;
	}

	void query_pool::copy_result(command_buffer& command, query_handle query, buffer& destination,
		u64 destination_offset, bool sixty_four_bit, bool include_availability)
	{
		std::lock_guard lock(m_impl->mutex);
		const u64 value_size = sixty_four_bit ? sizeof(u64) : sizeof(u32);
		const u64 total_size = value_size * (include_availability ? 2 : 1);
		if (!command.is_recording() || command.active_encoder() == encoder_kind::render ||
			!query || query.index >= m_impl->slots.size() ||
			m_impl->slots[query.index].generation != query.generation ||
			(m_impl->slots[query.index].state != query_state::ended &&
			 m_impl->slots[query.index].state != query_state::pending &&
			 m_impl->slots[query.index].state != query_state::ready) ||
			!m_impl->slots[query.index].resolved ||
			!(destination.usage() & (buffer_usage_copy_destination | buffer_usage_storage)) ||
			!destination.in_range(destination_offset, total_size))
		{
			fmt::throw_exception("Invalid Metal indirect query result copy");
		}

		id<MTL4ComputeCommandEncoder> encoder = command.active_encoder() == encoder_kind::compute
			? (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder()
			: (__bridge id<MTL4ComputeCommandEncoder>)command.begin_compute_encoding();
		if (include_availability)
		{
			[encoder fillBuffer:destination.native_handle()
				range:NSMakeRange(destination_offset + value_size, value_size) value:0];
			[encoder fillBuffer:destination.native_handle()
				range:NSMakeRange(destination_offset + value_size, 1) value:1];
		}
		buffer* source = m_impl->visibility ? m_impl->visibility.get() : m_impl->resolved_counters.get();
		[encoder copyFromBuffer:source->native_handle() sourceOffset:query.index * m_impl->stride
			toBuffer:destination.native_handle() destinationOffset:destination_offset size:value_size];
		command.retain_native_object((__bridge void*)source->native_handle(), true);
		command.retain_native_object((__bridge void*)destination.native_handle(), true);
		command.set_flag(command_has_blit_transfer);
	}

	query_pool::operator bool() const
	{
		return m_impl && m_impl->device && !m_impl->slots.empty() &&
			(m_impl->visibility || m_impl->counter_heap);
	}

	query_kind query_pool::kind() const
	{
		return m_impl ? m_impl->creation.kind : query_kind::occlusion_boolean;
	}

	u32 query_pool::size() const
	{
		return m_impl ? static_cast<u32>(m_impl->slots.size()) : 0;
	}

	u64 query_pool::result_stride() const
	{
		return m_impl ? m_impl->stride : 0;
	}

	const buffer* query_pool::visibility_buffer() const
	{
		return m_impl ? m_impl->visibility.get() : nullptr;
	}

	const buffer* query_pool::resolved_counter_buffer() const
	{
		return m_impl ? m_impl->resolved_counters.get() : nullptr;
	}

	counter_heap_handle query_pool::native_counter_heap() const
	{
		return m_impl ? m_impl->counter_heap : nil;
	}

	u64 query_pool::visibility_offset(query_handle query) const
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->visibility || !query || query.index >= m_impl->slots.size() ||
			m_impl->slots[query.index].generation != query.generation)
		{
			fmt::throw_exception("Invalid Metal visibility query offset handle");
		}
		return query.index * visibility_result_stride;
	}

	query_pool_statistics query_pool::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		query_pool_statistics result;
		result.capacity = static_cast<u32>(m_impl->slots.size());
		result.free_slots = static_cast<u32>(m_impl->available.size());
		result.last_submission = m_impl->last_submission;
		result.result_stride = m_impl->stride;
		for (const auto& slot : m_impl->slots)
		{
			switch (slot.state)
			{
			case query_state::free: break;
			case query_state::allocated:
			case query_state::ended: ++result.allocated_slots; break;
			case query_state::active: ++result.active_slots; break;
			case query_state::pending: ++result.pending_slots; break;
			case query_state::ready: ++result.ready_slots; break;
			}
		}
		return result;
	}

	struct query_pool_manager::impl
	{
		mutable std::mutex mutex;
		std::unique_ptr<query_pool> pool;
		query_pool_manager_create_info creation;
		query_handle active;
		command_buffer* query_command = nullptr;
		bool suspended = false;
		query_pool_manager_statistics stats;
	};

	query_pool_manager::query_pool_manager()
		: m_impl(std::make_unique<impl>())
	{
	}

	query_pool_manager::query_pool_manager(render_device& device, memory_allocator& allocator,
		const query_pool_manager_create_info& info)
		: query_pool_manager()
	{
		create(device, allocator, info);
	}

	query_pool_manager::~query_pool_manager() = default;

	void query_pool_manager::create(render_device& device, memory_allocator& allocator,
		const query_pool_manager_create_info& info)
	{
		destroy();
		if (!info.capacity)
		{
			fmt::throw_exception("Metal managed query pool capacity must be nonzero");
		}
		query_pool_create_info native_info;
		native_info.kind = info.kind;
		native_info.capacity = info.capacity;
		native_info.pool = info.allocation;
		native_info.label = info.label;
		native_info.cpu_readback = info.cpu_readback;
		std::lock_guard lock(m_impl->mutex);
		m_impl->pool = std::make_unique<query_pool>(device, allocator, native_info);
		m_impl->creation = info;
		m_impl->stats = {};
	}

	void query_pool_manager::destroy()
	{
		if (!m_impl)
		{
			return;
		}
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->active)
		{
			fmt::throw_exception("Cannot destroy a Metal query manager with an active query");
		}
		m_impl->pool.reset();
		m_impl->creation = {};
		m_impl->query_command = nullptr;
		m_impl->suspended = false;
		m_impl->stats = {};
	}

	query_handle query_pool_manager::allocate_query()
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->pool)
		{
			fmt::throw_exception("Cannot allocate from an empty Metal query manager");
		}
		const auto result = m_impl->pool->allocate();
		if (result) ++m_impl->stats.allocations;
		return result;
	}

	void query_pool_manager::release_query(query_handle query)
	{
		std::lock_guard lock(m_impl->mutex);
		if (query == m_impl->active)
		{
			fmt::throw_exception("Cannot release the active Metal query");
		}
		m_impl->pool->release(query);
		++m_impl->stats.releases;
	}

	void query_pool_manager::reset_query(query_handle query)
	{
		std::lock_guard lock(m_impl->mutex);
		if (query == m_impl->active)
		{
			fmt::throw_exception("Cannot reset the active Metal query");
		}
		m_impl->pool->reset(query);
	}

	void query_pool_manager::configure_render_pass(render_pass_configuration& configuration,
		visibility_result_behavior behavior) const
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->pool || m_impl->pool->kind() == query_kind::timestamp)
		{
			fmt::throw_exception("Metal visibility pass configuration requires an occlusion query pool");
		}
		configuration.visibility_buffer = m_impl->pool->visibility_buffer();
		configuration.visibility_required = true;
		configuration.visibility_behavior = behavior;
	}

	void query_pool_manager::begin_query(render_pass& pass, query_handle query, bool accumulate)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->pool || m_impl->active || !pass.is_open() || !pass.command() ||
			pass.configuration().visibility_buffer != m_impl->pool->visibility_buffer() ||
			(accumulate && pass.configuration().visibility_behavior != visibility_result_behavior::accumulate))
		{
			fmt::throw_exception("Invalid Metal managed query begin");
		}
		m_impl->pool->begin(*pass.command(), query, accumulate);
		pass.set_visibility_result(visibility_mode_for(m_impl->pool->kind()),
			m_impl->pool->visibility_offset(query));
		m_impl->active = query;
		m_impl->query_command = pass.command();
		m_impl->suspended = false;
		++m_impl->stats.begins;
	}

	void query_pool_manager::suspend_query(render_pass& pass, query_handle query)
	{
		std::lock_guard lock(m_impl->mutex);
		if (query != m_impl->active || m_impl->suspended || !pass.is_open() ||
			pass.command() != m_impl->query_command)
		{
			fmt::throw_exception("Invalid Metal query suspension");
		}
		pass.set_visibility_result(visibility_result_mode::disabled);
		m_impl->suspended = true;
		++m_impl->stats.suspends;
	}

	void query_pool_manager::resume_query(render_pass& pass, query_handle query)
	{
		std::lock_guard lock(m_impl->mutex);
		if (query != m_impl->active || !m_impl->suspended || !pass.is_open() ||
			pass.command() != m_impl->query_command ||
			pass.configuration().visibility_buffer != m_impl->pool->visibility_buffer() ||
			pass.configuration().visibility_behavior != visibility_result_behavior::accumulate)
		{
			fmt::throw_exception("Invalid Metal query resume");
		}
		pass.set_visibility_result(visibility_mode_for(m_impl->pool->kind()),
			m_impl->pool->visibility_offset(query));
		m_impl->suspended = false;
		++m_impl->stats.resumes;
	}

	void query_pool_manager::end_query(render_pass& pass, query_handle query)
	{
		std::lock_guard lock(m_impl->mutex);
		if (query != m_impl->active || !m_impl->query_command || !m_impl->query_command->is_recording())
		{
			fmt::throw_exception("Invalid Metal managed query end");
		}
		if (!m_impl->suspended)
		{
			if (!pass.is_open() || pass.command() != m_impl->query_command)
			{
				fmt::throw_exception("Active Metal query render pass is unavailable");
			}
			pass.set_visibility_result(visibility_result_mode::disabled);
		}
		m_impl->pool->end(*m_impl->query_command, query);
		m_impl->active = {};
		m_impl->query_command = nullptr;
		m_impl->suspended = false;
		++m_impl->stats.ends;
	}

	void query_pool_manager::write_timestamp(command_buffer& command, query_handle query, u64 stages)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->pool || m_impl->pool->kind() != query_kind::timestamp || m_impl->active)
		{
			fmt::throw_exception("Invalid managed Metal timestamp write");
		}
		m_impl->pool->write_timestamp(command, query, stages);
		++m_impl->stats.timestamp_writes;
	}

	void query_pool_manager::resolve_timestamps(command_buffer& command, u32 first, u32 count)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->pool || m_impl->pool->kind() != query_kind::timestamp || m_impl->active)
		{
			fmt::throw_exception("Invalid managed Metal timestamp resolve");
		}
		m_impl->pool->resolve(command, first, count);
		++m_impl->stats.resolves;
	}

	void query_pool_manager::mark_submitted(query_handle query, u64 submission_value)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->pool->mark_submitted(query, submission_value);
	}

	void query_pool_manager::mark_submitted(
		std::span<const query_handle> queries, u64 submission_value)
	{
		std::lock_guard lock(m_impl->mutex);
		for (const auto query : queries)
		{
			m_impl->pool->mark_submitted(query, submission_value);
		}
	}

	void query_pool_manager::notify_completed(u64 completed_submission_value, bool submission_succeeded)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->pool->notify_completed(completed_submission_value, submission_succeeded);
	}

	managed_query_status query_pool_manager::check_query_status(query_handle query) const
	{
		std::lock_guard lock(m_impl->mutex);
		managed_query_status status;
		status.state = m_impl->pool->state(query);
		const auto result = m_impl->pool->result(query, m_impl->creation.allow_partial_results);
		status.availability = result.availability;
		status.value = result.value;
		status.submission = result.submission;
		status.logically_active = query == m_impl->active;
		status.visibility_enabled = status.logically_active && !m_impl->suspended;
		return status;
	}

	query_result query_pool_manager::get_query_result(query_handle query, bool allow_partial) const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->pool->result(query, allow_partial && m_impl->creation.allow_partial_results);
	}

	void query_pool_manager::copy_query_result(render_pass* active_pass, command_buffer& command,
		query_handle query, buffer& destination, u64 destination_offset,
		bool sixty_four_bit, bool include_availability)
	{
		std::lock_guard lock(m_impl->mutex);
		if (query == m_impl->active)
		{
			fmt::throw_exception("Cannot copy the active Metal query result");
		}
		if (active_pass && active_pass->is_open())
		{
			if (active_pass->command() != &command)
			{
				fmt::throw_exception("Metal query copy references a different active render pass");
			}
			active_pass->end();
		}
		else if (command.active_encoder() == encoder_kind::render)
		{
			fmt::throw_exception("Metal query copy requires ownership of the active render pass");
		}
		m_impl->pool->copy_result(command, query, destination, destination_offset,
			sixty_four_bit, include_availability);
		++m_impl->stats.indirect_copies;
	}

	query_pool_manager::operator bool() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->pool && static_cast<bool>(*m_impl->pool);
	}

	query_kind query_pool_manager::kind() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->pool ? m_impl->pool->kind() : query_kind::occlusion_boolean;
	}

	u32 query_pool_manager::capacity() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->pool ? m_impl->pool->size() : 0;
	}

	const buffer* query_pool_manager::visibility_buffer() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->pool ? m_impl->pool->visibility_buffer() : nullptr;
	}

	query_handle query_pool_manager::active_query() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->active;
	}

	bool query_pool_manager::is_suspended() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->active && m_impl->suspended;
	}

	query_pool_manager_statistics query_pool_manager::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		auto result = m_impl->stats;
		if (m_impl->pool)
		{
			result.pool = m_impl->pool->statistics();
		}
		result.logically_active = m_impl->active ? 1 : 0;
		result.suspended = m_impl->active && m_impl->suspended ? 1 : 0;
		return result;
	}
}
