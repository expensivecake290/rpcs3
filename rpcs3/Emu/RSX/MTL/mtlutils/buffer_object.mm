#include "stdafx.h"
#include "buffer_object.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <limits>
#include <unistd.h>

namespace mtl
{
	struct buffer::impl
	{
		memory_allocation memory;
		id<MTLBuffer> native_buffer;
		u64 buffer_size = 0;
		u32 buffer_usage_mask = buffer_usage_none;
		storage_mode resolved_storage = storage_mode::shared;
		void* external_pointer = nullptr;
		u32 external_map_count = 0;
	};

	struct buffer_view::impl
	{
		id<MTLBuffer> source_buffer;
		id<MTLTexture> texture;
		u64 source_resource_uid = 0;
		u64 native_pixel_format = 0;
		u64 source_offset = 0;
		u64 view_size = 0;
		u32 elements = 0;
	};

	buffer::buffer()
		: m_impl(std::make_unique<impl>())
	{
	}

	buffer::buffer(memory_allocator& allocator, const buffer_create_info& info)
		: buffer()
	{
		create(allocator, info);
	}

	buffer::~buffer()
	{
		destroy();
	}

	void buffer::create(memory_allocator& allocator, const buffer_create_info& info)
	{
		destroy();
		if (info.size == 0 || info.usage == buffer_usage_none)
		{
			fmt::throw_exception("Invalid Metal buffer creation information");
		}
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		memory_allocation_request request;
		request.size = info.size;
		request.alignment = allocator.device().info().limits.buffer_offset_alignment;
		request.storage = info.storage;
		request.cache = info.cache;
		request.access = info.access;
		request.hazards = info.hazards;
		request.pool = info.pool;
		request.label = info.label;
		request.use_placement_heap = info.use_placement_heap;
		request.throw_on_failure = !info.allow_failure;
		request.recover_on_failure = info.recover_on_failure;
		m_impl->memory = allocator.allocate_buffer(request);
		if (!m_impl->memory)
		{
			ensure(info.allow_failure);
			return;
		}

		m_impl->native_buffer = m_impl->memory.buffer();
		m_impl->buffer_size = info.size;
		m_impl->buffer_usage_mask = info.usage;
		m_impl->resolved_storage = m_impl->memory.storage();
	}

	void buffer::create_no_copy(const render_device& device, void* host_pointer, u64 size, u32 usage, std::string_view label)
	{
		destroy();
		if (!host_pointer || size == 0 || usage == buffer_usage_none)
		{
			fmt::throw_exception("Invalid no-copy Metal buffer creation information");
		}

		const u64 page_size = static_cast<u64>(getpagesize());
		if ((reinterpret_cast<uptr>(host_pointer) % page_size) != 0 || (size % page_size) != 0)
		{
			fmt::throw_exception("No-copy Metal buffers require page-aligned pointers and lengths");
		}
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		m_impl->native_buffer = [device.native_handle() newBufferWithBytesNoCopy:host_pointer
			length:size
			options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked
			deallocator:nil];
		if (!m_impl->native_buffer)
		{
			fmt::throw_exception("Metal rejected no-copy buffer '%s'", label);
		}

		m_impl->native_buffer.label = [NSString stringWithUTF8String:std::string(label).c_str()];
		m_impl->buffer_size = size;
		m_impl->buffer_usage_mask = usage;
		m_impl->resolved_storage = storage_mode::shared;
		m_impl->external_pointer = host_pointer;
	}

	void buffer::destroy()
	{
		if (!m_impl)
		{
			return;
		}
		ensure(m_impl->external_map_count == 0);
		m_impl->memory = {};
		m_impl->native_buffer = nil;
		m_impl->buffer_size = 0;
		m_impl->buffer_usage_mask = buffer_usage_none;
		m_impl->external_pointer = nullptr;
	}

	buffer::operator bool() const
	{
		return m_impl && m_impl->native_buffer;
	}

	buffer_handle buffer::native_handle() const
	{
		return m_impl ? m_impl->native_buffer : nil;
	}

	u64 buffer::size() const
	{
		return m_impl ? m_impl->buffer_size : 0;
	}

	u64 buffer::gpu_address() const
	{
		return m_impl && m_impl->native_buffer ? m_impl->native_buffer.gpuAddress : 0;
	}

	u32 buffer::usage() const
	{
		return m_impl ? m_impl->buffer_usage_mask : buffer_usage_none;
	}

	storage_mode buffer::storage() const
	{
		if (!m_impl || !m_impl->native_buffer)
		{
			fmt::throw_exception("Storage mode requested from an empty Metal buffer");
		}
		return m_impl->resolved_storage;
	}

	bool buffer::is_cpu_visible() const
	{
		return *this && (m_impl->resolved_storage == storage_mode::shared || m_impl->resolved_storage == storage_mode::managed);
	}

	bool buffer::in_range(u64 offset, u64 length) const
	{
		return *this && offset <= m_impl->buffer_size && length <= m_impl->buffer_size - offset;
	}

	void* buffer::map(u64 offset, u64 length)
	{
		if (!in_range(offset, length) || !is_cpu_visible())
		{
			fmt::throw_exception("Invalid Metal buffer map range");
		}

		if (m_impl->external_pointer)
		{
			m_impl->external_map_count++;
			return static_cast<u8*>(m_impl->external_pointer) + offset;
		}
		return m_impl->memory.map(offset, length);
	}

	void buffer::unmap()
	{
		if (!m_impl || !m_impl->native_buffer)
		{
			fmt::throw_exception("Cannot unmap an empty Metal buffer");
		}

		if (m_impl->external_pointer)
		{
			if (m_impl->external_map_count == 0)
			{
				fmt::throw_exception("Unbalanced no-copy Metal buffer unmap");
			}
			m_impl->external_map_count--;
			return;
		}
		m_impl->memory.unmap();
	}

	void buffer::did_modify(u64 offset, u64 length)
	{
		if (!in_range(offset, length))
		{
			fmt::throw_exception("Invalid Metal buffer modification range");
		}
		if (!m_impl->external_pointer)
		{
			m_impl->memory.did_modify(offset, length);
		}
	}

	const memory_allocation& buffer::allocation() const
	{
		if (!m_impl || !m_impl->memory)
		{
			fmt::throw_exception("Allocator-backed memory requested from a no-copy or empty Metal buffer");
		}
		return m_impl->memory;
	}

	buffer_view::buffer_view()
		: m_impl(std::make_unique<impl>())
	{
	}

	buffer_view::buffer_view(const buffer& source, u64 pixel_format, u64 offset, u64 size, u32 bytes_per_element)
		: buffer_view()
	{
		create(source, pixel_format, offset, size, bytes_per_element);
	}

	buffer_view::~buffer_view()
	{
		destroy();
	}

	void buffer_view::create(const buffer& source, u64 pixel_format, u64 offset, u64 size, u32 bytes_per_element)
	{
		destroy();
		if (!source || !(source.usage() & buffer_usage_texture_view) || pixel_format == 0 || bytes_per_element == 0 ||
			size == 0 || (size % bytes_per_element) != 0 || !source.in_range(offset, size))
		{
			fmt::throw_exception("Invalid Metal texture-buffer view");
		}
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		id<MTLBuffer> native_source = source.native_handle();
		id<MTLDevice> device = native_source.device;
		const MTLPixelFormat native_format = static_cast<MTLPixelFormat>(pixel_format);
		const u64 required_alignment = [device minimumTextureBufferAlignmentForPixelFormat:native_format];
		if (required_alignment == 0 || (offset % required_alignment) != 0)
		{
			fmt::throw_exception("Metal texture-buffer offset %llu is not aligned to %llu", offset, required_alignment);
		}

		const u64 elements = size / bytes_per_element;
		if (elements > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal texture-buffer view has too many elements");
		}

		MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
		descriptor.textureType = MTLTextureTypeTextureBuffer;
		descriptor.pixelFormat = native_format;
		descriptor.width = elements;
		descriptor.height = 1;
		descriptor.depth = 1;
		descriptor.mipmapLevelCount = 1;
		descriptor.arrayLength = 1;
		descriptor.sampleCount = 1;
		descriptor.storageMode = native_source.storageMode;
		descriptor.cpuCacheMode = native_source.cpuCacheMode;
		descriptor.usage = MTLTextureUsageShaderRead;
		if (source.usage() & buffer_usage_storage)
		{
			descriptor.usage |= MTLTextureUsageShaderWrite;
		}

		m_impl->texture = [native_source newTextureWithDescriptor:descriptor offset:offset bytesPerRow:size];
		if (!m_impl->texture)
		{
			fmt::throw_exception("Metal rejected texture-buffer view (format %llu, offset %llu, size %llu)", pixel_format, offset, size);
		}

		m_impl->source_buffer = native_source;
		m_impl->source_resource_uid = source.uid();
		m_impl->native_pixel_format = pixel_format;
		m_impl->source_offset = offset;
		m_impl->view_size = size;
		m_impl->elements = static_cast<u32>(elements);
	}

	void buffer_view::destroy()
	{
		if (m_impl)
		{
			m_impl->texture = nil;
			m_impl->source_buffer = nil;
			m_impl->source_resource_uid = 0;
			m_impl->native_pixel_format = 0;
			m_impl->source_offset = 0;
			m_impl->view_size = 0;
			m_impl->elements = 0;
		}
	}

	buffer_view::operator bool() const
	{
		return m_impl && m_impl->texture;
	}

	texture_handle buffer_view::native_handle() const
	{
		return m_impl ? m_impl->texture : nil;
	}

	u64 buffer_view::source_uid() const
	{
		return m_impl ? m_impl->source_resource_uid : 0;
	}

	u64 buffer_view::pixel_format() const
	{
		return m_impl ? m_impl->native_pixel_format : 0;
	}

	u64 buffer_view::offset() const
	{
		return m_impl ? m_impl->source_offset : 0;
	}

	u64 buffer_view::size() const
	{
		return m_impl ? m_impl->view_size : 0;
	}

	u32 buffer_view::element_count() const
	{
		return m_impl ? m_impl->elements : 0;
	}

	bool buffer_view::in_range(u64 address, u64 length, u64& relative_offset) const
	{
		if (!*this || address < m_impl->source_offset)
		{
			return false;
		}

		const u64 local_offset = address - m_impl->source_offset;
		if (local_offset > m_impl->view_size || length > m_impl->view_size - local_offset)
		{
			return false;
		}
		relative_offset = local_offset;
		return true;
	}
}
