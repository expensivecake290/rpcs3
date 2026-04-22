#include "stdafx.h"
#include "MTLDevice.h"

#include "MTLCommandBuffer.h"
#include "MTLHeap.h"
#include "MTLResidency.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <vector>

namespace
{
	std::string get_ns_string(NSString* value)
	{
		if (!value)
		{
			return {};
		}

		const char* text = [value UTF8String];
		return text ? std::string(text) : std::string();
	}

	b8 supports_metal4(id<MTLDevice> device)
	{
		if (!device)
		{
			return false;
		}

		if (@available(macOS 26.0, *))
		{
			return [device supportsFamily:MTLGPUFamilyMetal4];
		}

		return false;
	}

	MTLResourceOptions private_untracked_resource_options()
	{
		rsx_log.trace("private_untracked_resource_options()");
		return MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked;
	}

	b8 can_allocate_from_private_heap(MTLResourceOptions options)
	{
		rsx_log.trace("can_allocate_from_private_heap(options=0x%llx)", static_cast<u64>(options));

		return (options & MTLResourceStorageModeMask) == MTLResourceStorageModePrivate &&
			(options & MTLResourceHazardTrackingModeMask) == MTLResourceHazardTrackingModeUntracked;
	}

	void set_resource_label(id<MTLResource> resource, const std::string& label)
	{
		rsx_log.trace("set_resource_label(resource=*0x%x, label=%s)", resource, label);

		if (!label.empty())
		{
			resource.label = [NSString stringWithUTF8String:label.c_str()];
		}
	}

	void query_heap_caps(id<MTLDevice> device, rsx::metal::device_caps& caps)
	{
		rsx_log.trace("query_heap_caps(device=*0x%x)", device);

		const MTLResourceOptions options = private_untracked_resource_options();
		const MTLSizeAndAlign buffer_size_and_align = [device heapBufferSizeAndAlignWithLength:4096 options:options];

		MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
			width:64
			height:64
			mipmapped:NO];
		texture_desc.resourceOptions = options;
		texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;

		const MTLSizeAndAlign texture_size_and_align = [device heapTextureSizeAndAlignWithDescriptor:texture_desc];

		caps.private_buffer_heap_size = static_cast<u64>(buffer_size_and_align.size);
		caps.private_buffer_heap_alignment = static_cast<u64>(buffer_size_and_align.align);
		caps.private_texture_heap_size = static_cast<u64>(texture_size_and_align.size);
		caps.private_texture_heap_alignment = static_cast<u64>(texture_size_and_align.align);
	}

	void query_sparse_caps(id<MTLDevice> device, rsx::metal::device_caps& caps)
	{
		rsx_log.trace("query_sparse_caps(device=*0x%x)", device);

		caps.sparse_tile_size_in_bytes = static_cast<u64>(device.sparseTileSizeInBytes);

		if (@available(macOS 26.4, *))
		{
			caps.placement_sparse_supported = device.supportsPlacementSparse;
		}
		else
		{
			caps.placement_sparse_supported = false;
		}
	}
}

namespace rsx::metal
{
	struct device::device_impl
	{
		id<MTLDevice> m_device = nil;
		std::unique_ptr<residency_manager> m_residency;
		std::unique_ptr<heap_manager> m_heaps;
		std::vector<void*> m_resident_heap_allocations;
		device_caps m_caps{};
	};

	b8 is_metal_supported()
	{
		rsx_log.trace("rsx::metal::is_metal_supported()");

		@autoreleasepool
		{
			id<MTLDevice> device = MTLCreateSystemDefaultDevice();
			return supports_metal4(device);
		}
	}

	device::device()
		: m_impl(std::make_unique<device_impl>())
	{
		rsx_log.notice("rsx::metal::device::device()");

		@autoreleasepool
		{
			id<MTLDevice> metal_device = MTLCreateSystemDefaultDevice();

			if (!metal_device)
			{
				fmt::throw_exception("Metal backend requires a Metal device, but MTLCreateSystemDefaultDevice returned nil");
			}

			if (!supports_metal4(metal_device))
			{
				fmt::throw_exception("Metal backend requires Metal 4 support");
			}

			if (@available(macOS 26.0, *))
			{
				m_impl->m_device = metal_device;
				m_impl->m_residency = std::make_unique<residency_manager>((__bridge void*)metal_device, "RPCS3 Metal backend residency set", 64);
				m_impl->m_residency->request_residency();
				m_impl->m_heaps = std::make_unique<heap_manager>((__bridge void*)metal_device);
			}
			else
			{
				fmt::throw_exception("Metal backend requires macOS 26.0 or newer");
			}

			m_impl->m_caps.name = get_ns_string([metal_device name]);
			m_impl->m_caps.metal4_supported = true;
			m_impl->m_caps.metalfx_available = NSClassFromString(@"MTLFXSpatialScalerDescriptor") != nil;
			m_impl->m_caps.residency_sets_supported = true;
			m_impl->m_caps.frames_in_flight = 3;
			m_impl->m_caps.max_argument_table_buffers = 31;
			m_impl->m_caps.max_argument_table_textures = 128;
			m_impl->m_caps.max_argument_table_samplers = 16;
			query_heap_caps(metal_device, m_impl->m_caps);
			query_sparse_caps(metal_device, m_impl->m_caps);
		}
	}

	device::~device()
	{
		rsx_log.notice("rsx::metal::device::~device()");

		if (m_impl && m_impl->m_residency)
		{
			for (void* allocation_handle : m_impl->m_resident_heap_allocations)
			{
				m_impl->m_residency->remove_allocation(allocation_handle);
			}

			if (!m_impl->m_resident_heap_allocations.empty())
			{
				m_impl->m_residency->commit();
				m_impl->m_resident_heap_allocations.clear();
			}

			m_impl->m_residency->end_residency();
		}
	}

	void* device::handle() const
	{
		rsx_log.trace("rsx::metal::device::handle()");
		return (__bridge void*)m_impl->m_device;
	}

	void* device::residency_set_handle() const
	{
		rsx_log.trace("rsx::metal::device::residency_set_handle()");
		return m_impl->m_residency->handle();
	}

	device_buffer_allocation device::create_buffer_allocation(u64 size, u64 resource_options, const std::string& label)
	{
		rsx_log.notice("rsx::metal::device::create_buffer_allocation(size=0x%llx, resource_options=0x%llx, label=%s)",
			size, resource_options, label);

		if (!size)
		{
			fmt::throw_exception("Metal device buffer allocation requires a non-zero size");
		}

		if (@available(macOS 26.0, *))
		{
			const MTLResourceOptions options = static_cast<MTLResourceOptions>(resource_options);
			if (can_allocate_from_private_heap(options))
			{
				const heap_buffer_allocation allocation = m_impl->m_heaps->allocate_private_buffer(size, resource_options, label);
				if (!allocation.buffer_handle || !allocation.heap_handle)
				{
					fmt::throw_exception("Metal heap buffer allocation returned an incomplete allocation");
				}

				rsx_log.trace("Metal heap buffer allocation: requested=0x%llx, required=0x%llx, alignment=0x%llx, created_heap=%u",
					allocation.requested_size,
					allocation.required_size,
					allocation.required_alignment,
					static_cast<u32>(allocation.created_heap));

				if (allocation.created_heap)
				{
					add_resident_allocation(allocation.heap_handle);
					commit_residency();
					m_impl->m_resident_heap_allocations.push_back(allocation.heap_handle);
				}

				return
				{
					.buffer_handle = allocation.buffer_handle,
					.residency_allocation_handle = nullptr,
					.heap_backed = allocation.heap_backed,
				};
			}

			id<MTLBuffer> buffer = [m_impl->m_device newBufferWithLength:static_cast<NSUInteger>(size) options:options];
			if (!buffer)
			{
				fmt::throw_exception("Metal standalone buffer allocation failed for size=0x%llx", size);
			}

			set_resource_label(buffer, label);

			return
			{
				.buffer_handle = (__bridge_retained void*)buffer,
				.residency_allocation_handle = (__bridge void*)buffer,
				.heap_backed = false,
			};
		}

		fmt::throw_exception("Metal device buffer allocation requires macOS 26.0 or newer");
	}

	device_texture_allocation device::create_texture_allocation(void* texture_descriptor_handle, const std::string& label)
	{
		rsx_log.notice("rsx::metal::device::create_texture_allocation(texture_descriptor_handle=*0x%x, label=%s)",
			texture_descriptor_handle, label);

		if (!texture_descriptor_handle)
		{
			fmt::throw_exception("Metal device texture allocation requires a valid texture descriptor");
		}

		if (@available(macOS 26.0, *))
		{
			MTLTextureDescriptor* texture_desc = (__bridge MTLTextureDescriptor*)texture_descriptor_handle;
			const MTLResourceOptions options = texture_desc.resourceOptions;
			if (can_allocate_from_private_heap(options))
			{
				const heap_texture_allocation allocation = m_impl->m_heaps->allocate_private_texture(texture_descriptor_handle, label);
				if (!allocation.texture_handle || !allocation.heap_handle)
				{
					fmt::throw_exception("Metal heap texture allocation returned an incomplete allocation");
				}

				rsx_log.trace("Metal heap texture allocation: required=0x%llx, alignment=0x%llx, created_heap=%u",
					allocation.required_size,
					allocation.required_alignment,
					static_cast<u32>(allocation.created_heap));

				if (allocation.created_heap)
				{
					add_resident_allocation(allocation.heap_handle);
					commit_residency();
					m_impl->m_resident_heap_allocations.push_back(allocation.heap_handle);
				}

				return
				{
					.texture_handle = allocation.texture_handle,
					.residency_allocation_handle = nullptr,
					.heap_backed = allocation.heap_backed,
				};
			}

			id<MTLTexture> texture = [m_impl->m_device newTextureWithDescriptor:texture_desc];
			if (!texture)
			{
				fmt::throw_exception("Metal standalone texture allocation failed");
			}

			set_resource_label(texture, label);

			return
			{
				.texture_handle = (__bridge_retained void*)texture,
				.residency_allocation_handle = (__bridge void*)texture,
				.heap_backed = false,
			};
		}

		fmt::throw_exception("Metal device texture allocation requires macOS 26.0 or newer");
	}

	void device::track_heap_resource_use(command_frame& frame, void* resource_handle)
	{
		rsx_log.trace("rsx::metal::device::track_heap_resource_use(frame_index=%u, resource_handle=*0x%x)",
			frame.frame_index(), resource_handle);
		m_impl->m_heaps->track_resource_use(frame, resource_handle);
	}

	void device::retire_heap_resource(void* resource_handle)
	{
		rsx_log.trace("rsx::metal::device::retire_heap_resource(resource_handle=*0x%x)", resource_handle);
		m_impl->m_heaps->retire_resource(resource_handle);
	}

	void device::add_resident_allocation(void* allocation_handle)
	{
		rsx_log.trace("rsx::metal::device::add_resident_allocation(allocation_handle=*0x%x)", allocation_handle);
		m_impl->m_residency->add_allocation(allocation_handle);
	}

	void device::remove_resident_allocation(void* allocation_handle)
	{
		rsx_log.trace("rsx::metal::device::remove_resident_allocation(allocation_handle=*0x%x)", allocation_handle);
		m_impl->m_residency->remove_allocation(allocation_handle);
	}

	void device::commit_residency()
	{
		rsx_log.trace("rsx::metal::device::commit_residency()");
		m_impl->m_residency->commit();
	}

	u64 device::residency_allocated_size() const
	{
		rsx_log.trace("rsx::metal::device::residency_allocated_size()");
		return m_impl->m_residency->allocated_size();
	}

	u32 device::residency_allocation_count() const
	{
		rsx_log.trace("rsx::metal::device::residency_allocation_count()");
		return m_impl->m_residency->allocation_count();
	}

	void device::report_memory_usage() const
	{
		rsx_log.notice("rsx::metal::device::report_memory_usage()");
		m_impl->m_heaps->report();
		rsx_log.notice("Metal residency set usage: allocations=%u, allocated_size=0x%llx",
			residency_allocation_count(), residency_allocated_size());
	}

	const device_caps& device::caps() const
	{
		rsx_log.trace("rsx::metal::device::caps()");
		return m_impl->m_caps;
	}

	void device::report_capabilities() const
	{
		rsx_log.notice("Metal device: %s", m_impl->m_caps.name);
		rsx_log.notice("Metal 4 support: %s", m_impl->m_caps.metal4_supported ? "yes" : "no");
		rsx_log.notice("MetalFX framework availability: %s", m_impl->m_caps.metalfx_available ? "yes" : "no");
		rsx_log.notice("Residency sets: %s", m_impl->m_caps.residency_sets_supported ? "yes" : "no");
		rsx_log.notice("Backend residency set: allocations=%u, allocated_size=0x%llx",
			residency_allocation_count(), residency_allocated_size());
		rsx_log.notice("Metal heap sizing: private_buffer_4k_size=0x%llx, private_buffer_alignment=0x%llx, private_texture_64x64_size=0x%llx, private_texture_alignment=0x%llx",
			m_impl->m_caps.private_buffer_heap_size,
			m_impl->m_caps.private_buffer_heap_alignment,
			m_impl->m_caps.private_texture_heap_size,
			m_impl->m_caps.private_texture_heap_alignment);
		rsx_log.notice("Metal sparse resources: tile_size=0x%llx, placement_sparse=%s",
			m_impl->m_caps.sparse_tile_size_in_bytes,
			m_impl->m_caps.placement_sparse_supported ? "yes" : "no");
		rsx_log.notice("Metal automatic heap aliasing: retired heap resources become aliasable only after tracked GPU use completes");
		rsx_log.warning("Metal placement heap aliasing remains disabled until explicit offset allocation and overlap barriers are implemented");
		report_memory_usage();
		rsx_log.notice("Frames in flight: %u", m_impl->m_caps.frames_in_flight);
		rsx_log.notice("MTL4ArgumentTable limits: buffers=%u, textures=%u, samplers=%u",
			m_impl->m_caps.max_argument_table_buffers,
			m_impl->m_caps.max_argument_table_textures,
			m_impl->m_caps.max_argument_table_samplers);
		rsx_log.warning("Metal RSX draw submission, argument tables, shader translation, pipeline creation, and MetalFX execution are not enabled yet");
	}
}
