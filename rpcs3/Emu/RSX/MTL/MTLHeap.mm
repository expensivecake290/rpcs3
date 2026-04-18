#include "stdafx.h"
#include "MTLHeap.h"

#include "MTLCommandBuffer.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <mutex>
#include <vector>

namespace
{
	constexpr u64 s_default_heap_size = 16ull * 1024ull * 1024ull;

	NSString* make_ns_string(const std::string& value)
	{
		return [NSString stringWithUTF8String:value.c_str()];
	}

	u64 align_up(u64 value, u64 alignment)
	{
		rsx_log.trace("align_up(value=0x%llx, alignment=0x%llx)", value, alignment);

		if (!alignment)
		{
			return value;
		}

		ensure((alignment & (alignment - 1)) == 0);
		return (value + alignment - 1) & ~(alignment - 1);
	}

	b8 is_private_untracked(MTLResourceOptions options)
	{
		rsx_log.trace("is_private_untracked(options=0x%llx)", static_cast<u64>(options));

		return (options & MTLResourceStorageModeMask) == MTLResourceStorageModePrivate &&
			(options & MTLResourceHazardTrackingModeMask) == MTLResourceHazardTrackingModeUntracked;
	}

	MTLHeapDescriptor* make_private_heap_descriptor(u64 heap_size)
	{
		rsx_log.trace("make_private_heap_descriptor(heap_size=0x%llx)", heap_size);

		MTLHeapDescriptor* desc = [MTLHeapDescriptor new];
		desc.size = static_cast<NSUInteger>(heap_size);
		desc.storageMode = MTLStorageModePrivate;
		desc.cpuCacheMode = MTLCPUCacheModeDefaultCache;
		desc.hazardTrackingMode = MTLHazardTrackingModeUntracked;
		desc.type = MTLHeapTypeAutomatic;
		return desc;
	}
}

namespace rsx::metal
{
	struct heap_record
	{
		id<MTLHeap> m_heap = nil;
	};

	struct heap_resource_record
	{
		void* m_resource_handle = nullptr;
		void* m_retained_resource_handle = nullptr;
		u32 m_active_uses = 0;
		b8 m_retired = false;
	};

	struct heap_manager::heap_manager_impl
	{
		id<MTLDevice> m_device = nil;
		std::vector<heap_record> m_heaps;
		std::vector<heap_resource_record> m_resources;
		mutable std::mutex m_mutex;
		u32 m_buffer_allocations = 0;
		u32 m_texture_allocations = 0;
		u32 m_aliasable_resources = 0;
	};

	std::vector<heap_resource_record>::iterator find_resource(std::vector<heap_resource_record>& resources, void* resource_handle)
	{
		rsx_log.trace("find_resource(resource_handle=*0x%x)", resource_handle);

		return std::find_if(resources.begin(), resources.end(), [resource_handle](const heap_resource_record& record)
		{
			return record.m_resource_handle == resource_handle;
		});
	}

	void make_resource_aliasable(id<MTLResource> resource)
	{
		rsx_log.trace("make_resource_aliasable(resource=*0x%x)", resource);

		if (!resource)
		{
			fmt::throw_exception("Metal heap aliasing requires a valid resource");
		}

		if (![resource isAliasable])
		{
			[resource makeAliasable];
		}
	}

	heap_manager::heap_manager(void* device_handle)
		: m_impl(std::make_unique<heap_manager_impl>())
	{
		rsx_log.notice("rsx::metal::heap_manager::heap_manager(device_handle=*0x%x)", device_handle);

		if (!device_handle)
		{
			fmt::throw_exception("Metal heap manager requires a valid device");
		}

		if (@available(macOS 26.0, *))
		{
			m_impl->m_device = (__bridge id<MTLDevice>)device_handle;
			m_impl->m_heaps.reserve(4);
		}
		else
		{
			fmt::throw_exception("Metal heap manager requires macOS 26.0 or newer");
		}
	}

	heap_manager::~heap_manager()
	{
		rsx_log.notice("rsx::metal::heap_manager::~heap_manager()");

		const heap_manager_stats heap_stats = stats();
		if (heap_stats.active_resource_use_count || heap_stats.pending_aliasable_resource_count)
		{
			rsx_log.error("Metal heap manager destroyed with active heap resources: active_uses=%u, pending_aliasable=%u, live_resources=%u",
				heap_stats.active_resource_use_count,
				heap_stats.pending_aliasable_resource_count,
				heap_stats.live_resource_count);
		}
	}

	heap_buffer_allocation heap_manager::allocate_private_buffer(u64 size, u64 resource_options, const std::string& label)
	{
		rsx_log.notice("rsx::metal::heap_manager::allocate_private_buffer(size=0x%llx, resource_options=0x%llx, label=%s)",
			size, resource_options, label);

		if (!size)
		{
			fmt::throw_exception("Metal heap buffer allocation requires a non-zero size");
		}

		if (@available(macOS 26.0, *))
		{
			const MTLResourceOptions options = static_cast<MTLResourceOptions>(resource_options);
			if (!is_private_untracked(options))
			{
				fmt::throw_exception("Metal heap buffer allocation requires private untracked resource options");
			}

			const MTLSizeAndAlign size_and_align = [m_impl->m_device heapBufferSizeAndAlignWithLength:static_cast<NSUInteger>(size) options:options];
			const u64 required_size = static_cast<u64>(size_and_align.size);
			const u64 required_alignment = static_cast<u64>(size_and_align.align);

			if (!required_size || !required_alignment)
			{
				fmt::throw_exception("Metal heap buffer sizing failed for size=0x%llx", size);
			}

			std::lock_guard lock(m_impl->m_mutex);

			for (heap_record& record : m_impl->m_heaps)
			{
				if ([record.m_heap maxAvailableSizeWithAlignment:static_cast<NSUInteger>(required_alignment)] < required_size)
				{
					continue;
				}

				id<MTLBuffer> buffer = [record.m_heap newBufferWithLength:static_cast<NSUInteger>(size) options:options];
				if (!buffer)
				{
					continue;
				}

				if (!label.empty())
				{
					buffer.label = make_ns_string(label);
				}

				m_impl->m_buffer_allocations++;
				m_impl->m_resources.push_back({ .m_resource_handle = (__bridge void*)buffer });

				return
				{
					.buffer_handle = (__bridge_retained void*)buffer,
					.heap_handle = (__bridge void*)record.m_heap,
					.requested_size = size,
					.required_size = required_size,
					.required_alignment = required_alignment,
					.created_heap = false,
					.heap_backed = true,
				};
			}

			const u64 heap_size = align_up(std::max(required_size, s_default_heap_size), required_alignment);
			MTLHeapDescriptor* heap_desc = make_private_heap_descriptor(heap_size);
			id<MTLHeap> heap = [m_impl->m_device newHeapWithDescriptor:heap_desc];
			if (!heap)
			{
				fmt::throw_exception("Metal private buffer heap allocation failed for heap_size=0x%llx", heap_size);
			}

			heap.label = make_ns_string(fmt::format("RPCS3 Metal private resource heap %u", static_cast<u32>(m_impl->m_heaps.size())));

			id<MTLBuffer> buffer = [heap newBufferWithLength:static_cast<NSUInteger>(size) options:options];
			if (!buffer)
			{
				fmt::throw_exception("Metal private heap failed to allocate initial buffer size=0x%llx", size);
			}

			if (!label.empty())
			{
				buffer.label = make_ns_string(label);
			}

			m_impl->m_heaps.push_back({ heap });
			m_impl->m_buffer_allocations++;
			m_impl->m_resources.push_back({ .m_resource_handle = (__bridge void*)buffer });

			return
			{
				.buffer_handle = (__bridge_retained void*)buffer,
				.heap_handle = (__bridge void*)heap,
				.requested_size = size,
				.required_size = required_size,
				.required_alignment = required_alignment,
				.created_heap = true,
				.heap_backed = true,
			};
		}

		fmt::throw_exception("Metal heap buffer allocation requires macOS 26.0 or newer");
	}

	heap_texture_allocation heap_manager::allocate_private_texture(void* texture_descriptor_handle, const std::string& label)
	{
		rsx_log.notice("rsx::metal::heap_manager::allocate_private_texture(texture_descriptor_handle=*0x%x, label=%s)",
			texture_descriptor_handle, label);

		if (!texture_descriptor_handle)
		{
			fmt::throw_exception("Metal heap texture allocation requires a valid texture descriptor");
		}

		if (@available(macOS 26.0, *))
		{
			MTLTextureDescriptor* texture_desc = (__bridge MTLTextureDescriptor*)texture_descriptor_handle;
			const MTLResourceOptions options = texture_desc.resourceOptions;
			if (!is_private_untracked(options))
			{
				fmt::throw_exception("Metal heap texture allocation requires private untracked resource options");
			}

			const MTLSizeAndAlign size_and_align = [m_impl->m_device heapTextureSizeAndAlignWithDescriptor:texture_desc];
			const u64 required_size = static_cast<u64>(size_and_align.size);
			const u64 required_alignment = static_cast<u64>(size_and_align.align);

			if (!required_size || !required_alignment)
			{
				fmt::throw_exception("Metal heap texture sizing failed");
			}

			std::lock_guard lock(m_impl->m_mutex);

			for (heap_record& record : m_impl->m_heaps)
			{
				if ([record.m_heap maxAvailableSizeWithAlignment:static_cast<NSUInteger>(required_alignment)] < required_size)
				{
					continue;
				}

				id<MTLTexture> texture = [record.m_heap newTextureWithDescriptor:texture_desc];
				if (!texture)
				{
					continue;
				}

				if (!label.empty())
				{
					texture.label = make_ns_string(label);
				}

				m_impl->m_texture_allocations++;
				m_impl->m_resources.push_back({ .m_resource_handle = (__bridge void*)texture });

				return
				{
					.texture_handle = (__bridge_retained void*)texture,
					.heap_handle = (__bridge void*)record.m_heap,
					.required_size = required_size,
					.required_alignment = required_alignment,
					.created_heap = false,
					.heap_backed = true,
				};
			}

			const u64 heap_size = align_up(std::max(required_size, s_default_heap_size), required_alignment);
			MTLHeapDescriptor* heap_desc = make_private_heap_descriptor(heap_size);
			id<MTLHeap> heap = [m_impl->m_device newHeapWithDescriptor:heap_desc];
			if (!heap)
			{
				fmt::throw_exception("Metal private texture heap allocation failed for heap_size=0x%llx", heap_size);
			}

			heap.label = make_ns_string(fmt::format("RPCS3 Metal private resource heap %u", static_cast<u32>(m_impl->m_heaps.size())));

			id<MTLTexture> texture = [heap newTextureWithDescriptor:texture_desc];
			if (!texture)
			{
				fmt::throw_exception("Metal private heap failed to allocate initial texture");
			}

			if (!label.empty())
			{
				texture.label = make_ns_string(label);
			}

			m_impl->m_heaps.push_back({ heap });
			m_impl->m_texture_allocations++;
			m_impl->m_resources.push_back({ .m_resource_handle = (__bridge void*)texture });

			return
			{
				.texture_handle = (__bridge_retained void*)texture,
				.heap_handle = (__bridge void*)heap,
				.required_size = required_size,
				.required_alignment = required_alignment,
				.created_heap = true,
				.heap_backed = true,
			};
		}

		fmt::throw_exception("Metal heap texture allocation requires macOS 26.0 or newer");
	}

	void heap_manager::track_resource_use(command_frame& frame, void* resource_handle)
	{
		rsx_log.trace("rsx::metal::heap_manager::track_resource_use(frame_index=%u, resource_handle=*0x%x)",
			frame.frame_index(), resource_handle);

		if (!resource_handle)
		{
			fmt::throw_exception("Metal heap resource usage tracking requires a valid resource");
		}

		{
			std::lock_guard lock(m_impl->m_mutex);
			auto found = find_resource(m_impl->m_resources, resource_handle);
			if (found == m_impl->m_resources.end())
			{
				fmt::throw_exception("Metal heap resource usage tracking received an unknown resource");
			}

			if (found->m_retired)
			{
				fmt::throw_exception("Metal heap resource usage tracking received a retired resource");
			}

			found->m_active_uses++;
		}

		frame.track_object(resource_handle);
		frame.on_completed([this, resource_handle]()
		{
			rsx_log.trace("rsx::metal::heap_manager::track_resource_use completion(resource_handle=*0x%x)", resource_handle);

			void* retained_resource_handle = nullptr;

			{
				std::lock_guard lock(m_impl->m_mutex);
				auto found = find_resource(m_impl->m_resources, resource_handle);
				if (found == m_impl->m_resources.end())
				{
					return;
				}

				ensure(found->m_active_uses > 0);
				found->m_active_uses--;

				if (found->m_active_uses || !found->m_retired)
				{
					return;
				}

				retained_resource_handle = found->m_retained_resource_handle;
				m_impl->m_resources.erase(found);
				m_impl->m_aliasable_resources++;
			}

			if (retained_resource_handle)
			{
				id<MTLResource> resource = (__bridge_transfer id<MTLResource>)retained_resource_handle;
				make_resource_aliasable(resource);
			}
		});
	}

	void heap_manager::retire_resource(void* resource_handle)
	{
		rsx_log.notice("rsx::metal::heap_manager::retire_resource(resource_handle=*0x%x)", resource_handle);

		if (!resource_handle)
		{
			fmt::throw_exception("Metal heap resource retirement requires a valid resource");
		}

		id<MTLResource> immediate_resource = nil;

		{
			std::lock_guard lock(m_impl->m_mutex);
			auto found = find_resource(m_impl->m_resources, resource_handle);
			if (found == m_impl->m_resources.end())
			{
				fmt::throw_exception("Metal heap resource retirement received an unknown resource");
			}

			if (found->m_retired)
			{
				rsx_log.warning("Metal heap resource was already retired: resource=*0x%x", resource_handle);
				return;
			}

			found->m_retired = true;

			if (found->m_active_uses)
			{
				id<MTLResource> resource = (__bridge id<MTLResource>)resource_handle;
				found->m_retained_resource_handle = (__bridge_retained void*)resource;
				return;
			}

			immediate_resource = (__bridge id<MTLResource>)resource_handle;
			m_impl->m_resources.erase(found);
			m_impl->m_aliasable_resources++;
		}

		make_resource_aliasable(immediate_resource);
	}

	heap_manager_stats heap_manager::stats() const
	{
		rsx_log.trace("rsx::metal::heap_manager::stats()");

		std::lock_guard lock(m_impl->m_mutex);

		heap_manager_stats result =
		{
			.heap_count = static_cast<u32>(m_impl->m_heaps.size()),
			.buffer_allocation_count = m_impl->m_buffer_allocations,
			.texture_allocation_count = m_impl->m_texture_allocations,
			.live_resource_count = static_cast<u32>(m_impl->m_resources.size()),
			.aliasable_resource_count = m_impl->m_aliasable_resources,
		};

		for (const heap_resource_record& record : m_impl->m_resources)
		{
			result.active_resource_use_count += record.m_active_uses;

			if (record.m_retired)
			{
				result.pending_aliasable_resource_count++;
			}
		}

		for (const heap_record& record : m_impl->m_heaps)
		{
			result.total_heap_size += static_cast<u64>(record.m_heap.size);
			result.used_heap_size += static_cast<u64>(record.m_heap.usedSize);
			result.current_allocated_size += static_cast<u64>(record.m_heap.currentAllocatedSize);
		}

		return result;
	}

	void heap_manager::report() const
	{
		rsx_log.notice("rsx::metal::heap_manager::report()");
		const heap_manager_stats heap_stats = stats();
		rsx_log.notice("Metal heaps: count=%u, buffer_allocations=%u, texture_allocations=%u, live_resources=%u, active_uses=%u, pending_aliasable=%u, aliasable=%u, total_size=0x%llx, used_size=0x%llx, allocated_size=0x%llx",
			heap_stats.heap_count,
			heap_stats.buffer_allocation_count,
			heap_stats.texture_allocation_count,
			heap_stats.live_resource_count,
			heap_stats.active_resource_use_count,
			heap_stats.pending_aliasable_resource_count,
			heap_stats.aliasable_resource_count,
			heap_stats.total_heap_size,
			heap_stats.used_heap_size,
			heap_stats.current_allocated_size);
		rsx_log.warning("Metal placement heap aliasing is disabled until explicit offset allocation and overlap barriers are implemented");
	}
}
