#include "stdafx.h"
#include "MTLTextureCache.h"

#include "MTLBarrier.h"
#include "MTLBuffer.h"
#include "MTLCommandBuffer.h"
#include "MTLDevice.h"
#include "MTLTexture.h"

#import <Metal/Metal.h>

#include <cstring>
#include <limits>
#include <source_location>
#include <unordered_map>
#include <vector>

namespace vm
{
	extern u8* const g_base_addr;
}

namespace rsx
{
	u32 get_address(u32 offset, u32 location, u32 size_to_check, std::source_location src_loc);
}

namespace
{
	struct fragment_texture_key
	{
		u32 address = 0;
		u32 offset = 0;
		u32 location = 0;
		u32 pitch = 0;
		u16 width = 0;
		u16 height = 0;
		u8 format = 0;
		u8 mipmaps = 0;

		bool operator==(const fragment_texture_key& other) const
		{
			return address == other.address &&
				offset == other.offset &&
				location == other.location &&
				pitch == other.pitch &&
				width == other.width &&
				height == other.height &&
				format == other.format &&
				mipmaps == other.mipmaps;
		}
	};

	struct fragment_texture_key_hash
	{
		std::size_t operator()(const fragment_texture_key& key) const
		{
			std::size_t seed = 0;
			hash_combine(seed, key.address);
			hash_combine(seed, key.offset);
			hash_combine(seed, key.location);
			hash_combine(seed, key.pitch);
			hash_combine(seed, key.width);
			hash_combine(seed, key.height);
			hash_combine(seed, key.format);
			hash_combine(seed, key.mipmaps);
			return seed;
		}

	private:
		static void hash_combine(std::size_t& seed, u64 value)
		{
			seed ^= std::hash<u64>{}(value) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
		}
	};

	struct fragment_texture_upload_desc
	{
		fragment_texture_key key;
		u64 upload_size = 0;
		u64 bytes_per_row = 0;
		u32 metal_pixel_format = 0;
	};

	void increment_texture_counter(u32& counter, const char* name)
	{
		rsx_log.trace("increment_texture_counter(name=%s, counter=%u)", name, counter);

		if (counter == std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal sampled texture cache counter overflow: %s", name);
		}

		counter++;
	}

	void add_uploaded_texture_bytes(u64& counter, u64 size)
	{
		rsx_log.trace("add_uploaded_texture_bytes(counter=0x%llx, size=0x%llx)", counter, size);

		if (counter > std::numeric_limits<u64>::max() - size)
		{
			fmt::throw_exception("Metal sampled texture upload byte counter overflow");
		}

		counter += size;
	}

	u32 checked_retained_texture_count(usz count)
	{
		rsx_log.trace("checked_retained_texture_count(count=0x%llx)", static_cast<u64>(count));

		if (count > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal sampled texture retained count overflow");
		}

		return static_cast<u32>(count);
	}

	u32 checked_address_size(u64 size)
	{
		rsx_log.trace("checked_address_size(size=0x%llx)", size);

		if (size > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal sampled texture upload size exceeds RSX address-check range: 0x%llx", size);
		}

		return static_cast<u32>(size);
	}

	MTLPixelFormat make_metal_sampled_pixel_format(u8 format)
	{
		rsx_log.trace("make_metal_sampled_pixel_format(format=0x%x)", static_cast<u32>(format));

		switch (format & ~(CELL_GCM_TEXTURE_LN | CELL_GCM_TEXTURE_UN))
		{
		case CELL_GCM_TEXTURE_B8:
			return MTLPixelFormatR8Unorm;
		default:
			rsx_log.todo("Metal sampled texture format 0x%x is not implemented", static_cast<u32>(format));
			fmt::throw_exception("Metal backend sampled texture format 0x%x is not implemented yet", static_cast<u32>(format));
		}
	}

	fragment_texture_upload_desc make_fragment_texture_upload_desc(const rsx::fragment_texture& rsx_texture, u32 texture_index)
	{
		rsx_log.trace("make_fragment_texture_upload_desc(texture_index=%u, offset=0x%x, location=0x%x, format=0x%x)",
			texture_index,
			rsx_texture.offset(),
			rsx_texture.location(),
			rsx_texture.format());

		const u16 width = rsx_texture.width();
		const u16 height = rsx_texture.height();
		const u16 depth = rsx_texture.depth();
		const u32 pitch = rsx_texture.pitch();
		const u8 format = rsx_texture.format();
		const u8 base_format = format & ~(CELL_GCM_TEXTURE_LN | CELL_GCM_TEXTURE_UN);
		const u16 mipmaps = rsx_texture.get_exact_mipmap_count();

		if (!rsx_texture.enabled())
		{
			fmt::throw_exception("Metal sampled texture %u is not enabled", texture_index);
		}

		if (!width || !height)
		{
			fmt::throw_exception("Metal sampled texture %u has invalid dimensions: %ux%u", texture_index, width, height);
		}

		if (rsx_texture.get_extended_texture_dimension() != rsx::texture_dimension_extended::texture_dimension_2d || depth > 1)
		{
			rsx_log.todo("Metal sampled texture %u dimension 0x%x/depth=%u is not implemented",
				texture_index,
				static_cast<u32>(rsx_texture.get_extended_texture_dimension()),
				depth);
			fmt::throw_exception("Metal backend sampled texture %u requires a 2D texture for now", texture_index);
		}

		if (!(format & CELL_GCM_TEXTURE_LN))
		{
			rsx_log.todo("Metal sampled texture %u swizzled upload is not implemented", texture_index);
			fmt::throw_exception("Metal backend sampled texture %u requires a linear source layout for now", texture_index);
		}

		if (rsx_texture.border_type())
		{
			rsx_log.todo("Metal sampled texture %u border texel upload is not implemented", texture_index);
			fmt::throw_exception("Metal backend sampled texture %u border texel upload is not implemented yet", texture_index);
		}

		if (rsx_texture.is_compressed_format())
		{
			rsx_log.todo("Metal sampled texture %u compressed upload is not implemented", texture_index);
			fmt::throw_exception("Metal backend sampled texture %u compressed upload is not implemented yet", texture_index);
		}

		if (mipmaps != 1)
		{
			rsx_log.todo("Metal sampled texture %u mipmapped upload count %u is not implemented", texture_index, mipmaps);
			fmt::throw_exception("Metal backend sampled texture %u mipmapped upload is not implemented yet", texture_index);
		}

		if (base_format != CELL_GCM_TEXTURE_B8)
		{
			make_metal_sampled_pixel_format(format);
		}

		if (pitch < width)
		{
			fmt::throw_exception("Metal sampled texture %u pitch is too small: pitch=0x%x, width=0x%x",
				texture_index, pitch, width);
		}

		const u64 upload_size = static_cast<u64>(pitch) * height;
		if (!upload_size)
		{
			fmt::throw_exception("Metal sampled texture %u upload size is zero", texture_index);
		}

		const u32 address = rsx::get_address(rsx_texture.offset(), rsx_texture.location(), checked_address_size(upload_size), std::source_location::current());
		return
		{
			.key =
			{
				.address = address,
				.offset = rsx_texture.offset(),
				.location = rsx_texture.location(),
				.pitch = pitch,
				.width = width,
				.height = height,
				.format = format,
				.mipmaps = static_cast<u8>(mipmaps),
			},
			.upload_size = upload_size,
			.bytes_per_row = pitch,
			.metal_pixel_format = static_cast<u32>(make_metal_sampled_pixel_format(format)),
		};
	}

	MTLTextureDescriptor* make_texture_descriptor(const fragment_texture_upload_desc& desc)
	{
		rsx_log.trace("make_texture_descriptor(width=%u, height=%u, format=%u)",
			desc.key.width,
			desc.key.height,
			desc.metal_pixel_format);

		MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:static_cast<MTLPixelFormat>(desc.metal_pixel_format)
			width:desc.key.width
			height:desc.key.height
			mipmapped:NO];
		texture_desc.resourceOptions = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked;
		texture_desc.usage = MTLTextureUsageShaderRead;
		return texture_desc;
	}

	void track_heap_resource_use(rsx::metal::command_frame& frame, const rsx::metal::heap_resource_usage& usage)
	{
		rsx_log.trace("track_heap_resource_use(frame_index=%u, resource_handle=*0x%x)",
			frame.frame_index(), usage.resource_handle);

		if (!usage.metal_device && !usage.resource_handle)
		{
			return;
		}

		if (!usage.metal_device || !usage.resource_handle)
		{
			fmt::throw_exception("Metal sampled texture heap resource tracking received incomplete heap usage state");
		}

		usage.metal_device->track_heap_resource_use(frame, usage.resource_handle);
	}
}

namespace rsx::metal
{
	struct sampled_texture_record
	{
		std::unique_ptr<texture> image;
		std::vector<std::unique_ptr<buffer>> staging_buffers;
	};

	struct sampled_texture_cache::sampled_texture_cache_impl
	{
		device& m_device;
		std::unordered_map<fragment_texture_key, sampled_texture_record, fragment_texture_key_hash> m_textures;
		sampled_texture_cache_stats m_stats{};
		u32 m_frames_in_flight = 0;

		explicit sampled_texture_cache_impl(device& dev)
			: m_device(dev)
			, m_frames_in_flight(dev.caps().frames_in_flight)
		{
			if (!m_frames_in_flight)
			{
				fmt::throw_exception("Metal sampled texture cache requires at least one frame in flight");
			}
		}
	};

	sampled_texture_cache::sampled_texture_cache(device& dev)
		: m_impl(std::make_unique<sampled_texture_cache_impl>(dev))
	{
		rsx_log.notice("rsx::metal::sampled_texture_cache::sampled_texture_cache(device=*0x%x)", dev.handle());
	}

	sampled_texture_cache::~sampled_texture_cache()
	{
		rsx_log.notice("rsx::metal::sampled_texture_cache::~sampled_texture_cache(retained=%u)", stats().retained_texture_count);
	}

	texture& sampled_texture_cache::prepare_fragment_texture(command_frame& frame, const rsx::fragment_texture& rsx_texture, u32 texture_index)
	{
		rsx_log.trace("rsx::metal::sampled_texture_cache::prepare_fragment_texture(frame_index=%u, texture_index=%u)",
			frame.frame_index(),
			texture_index);

		const fragment_texture_upload_desc desc = make_fragment_texture_upload_desc(rsx_texture, texture_index);

		auto found = m_impl->m_textures.find(desc.key);
		if (found == m_impl->m_textures.end())
		{
			if (!@available(macOS 26.0, *))
			{
				fmt::throw_exception("Metal sampled texture allocation requires macOS 26.0 or newer");
			}

			MTLTextureDescriptor* texture_desc = make_texture_descriptor(desc);

			sampled_texture_record record;
			record.image = std::make_unique<texture>(m_impl->m_device, (__bridge void*)texture_desc,
				fmt::format("RPCS3 Metal sampled texture %u 0x%08x", texture_index, desc.key.address));
			record.staging_buffers.resize(m_impl->m_frames_in_flight);

			found = m_impl->m_textures.emplace(desc.key, std::move(record)).first;
			increment_texture_counter(m_impl->m_stats.created_texture_count, "created sampled texture");
			m_impl->m_stats.retained_texture_count = checked_retained_texture_count(m_impl->m_textures.size());
		}
		else
		{
			increment_texture_counter(m_impl->m_stats.reused_texture_count, "reused sampled texture");
		}

		sampled_texture_record& record = found->second;
		if (!record.image)
		{
			fmt::throw_exception("Metal sampled texture cache record has no texture");
		}

		const u32 frame_index = frame.frame_index();
		if (frame_index >= record.staging_buffers.size())
		{
			fmt::throw_exception("Metal sampled texture staging frame index %u exceeds frame count %u",
				frame_index,
				static_cast<u32>(record.staging_buffers.size()));
		}

		std::unique_ptr<buffer>& staging_buffer = record.staging_buffers[frame_index];
		if (!staging_buffer || staging_buffer->length() < desc.upload_size)
		{
			buffer_desc buffer_desc =
			{
				.size = desc.upload_size,
				.label = fmt::format("RPCS3 Metal sampled texture staging %u 0x%08x", texture_index, desc.key.address),
				.storage = buffer_storage::shared,
				.cpu_cache = buffer_cpu_cache_mode::write_combined,
				.hazard_tracking = resource_hazard_tracking::untracked,
			};

			staging_buffer = std::make_unique<buffer>(m_impl->m_device, std::move(buffer_desc));
		}

		std::memcpy(staging_buffer->map(), vm::g_base_addr + desc.key.address, static_cast<usz>(desc.upload_size));
		frame.track_object(staging_buffer->allocation_handle());
		frame.track_object(record.image->allocation_handle());
		track_heap_resource_use(frame, staging_buffer->heap_resource_usage_info());
		track_heap_resource_use(frame, record.image->heap_resource_usage_info());

		if (@available(macOS 26.0, *))
		{
			id<MTL4CommandBuffer> command_buffer = (__bridge id<MTL4CommandBuffer>)frame.command_buffer_handle();
			id<MTL4ComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
			if (!encoder)
			{
				fmt::throw_exception("Metal sampled texture upload failed to create a compute command encoder");
			}

			const resource_barrier source_barrier = frame.track_resource_usage(resource_usage
			{
				.resource_id = staging_buffer->gpu_address(),
				.stage = resource_stage::compute,
				.access = resource_access::read,
				.scope = resource_barrier_scope::buffers,
			});
			encode_consumer_barrier((__bridge void*)encoder, source_barrier);

			const resource_barrier destination_barrier = frame.track_resource_usage(resource_usage
			{
				.resource_id = record.image->resource_id(),
				.stage = resource_stage::compute,
				.access = resource_access::write,
				.scope = resource_barrier_scope::textures,
			});
			encode_consumer_barrier((__bridge void*)encoder, destination_barrier);

			[encoder copyFromBuffer:(__bridge id<MTLBuffer>)staging_buffer->handle()
			           sourceOffset:0
			      sourceBytesPerRow:static_cast<NSUInteger>(desc.bytes_per_row)
			    sourceBytesPerImage:0
			             sourceSize:MTLSizeMake(desc.key.width, desc.key.height, 1)
			              toTexture:(__bridge id<MTLTexture>)record.image->handle()
			       destinationSlice:0
			       destinationLevel:0
			      destinationOrigin:MTLOriginMake(0, 0, 0)];
			[encoder endEncoding];
		}
		else
		{
			fmt::throw_exception("Metal sampled texture upload requires macOS 26.0 or newer");
		}

		increment_texture_counter(m_impl->m_stats.uploaded_texture_count, "uploaded sampled texture");
		add_uploaded_texture_bytes(m_impl->m_stats.uploaded_byte_count, desc.upload_size);
		return *record.image;
	}

	texture& sampled_texture_cache::get_fragment_texture(const rsx::fragment_texture& rsx_texture, u32 texture_index)
	{
		rsx_log.trace("rsx::metal::sampled_texture_cache::get_fragment_texture(texture_index=%u)", texture_index);

		const fragment_texture_upload_desc desc = make_fragment_texture_upload_desc(rsx_texture, texture_index);
		auto found = m_impl->m_textures.find(desc.key);
		if (found == m_impl->m_textures.end() || !found->second.image)
		{
			fmt::throw_exception("Metal sampled texture %u was not prepared before argument binding", texture_index);
		}

		return *found->second.image;
	}

	sampled_texture_cache_stats sampled_texture_cache::stats() const
	{
		rsx_log.trace("rsx::metal::sampled_texture_cache::stats()");

		sampled_texture_cache_stats result = m_impl->m_stats;
		result.retained_texture_count = checked_retained_texture_count(m_impl->m_textures.size());
		return result;
	}

	void sampled_texture_cache::report() const
	{
		rsx_log.notice("rsx::metal::sampled_texture_cache::report()");

		const sampled_texture_cache_stats texture_stats = stats();
		rsx_log.notice("Metal sampled texture cache: retained=%u, created=%u, reused=%u, uploads=%u, uploaded_bytes=0x%llx",
			texture_stats.retained_texture_count,
			texture_stats.created_texture_count,
			texture_stats.reused_texture_count,
			texture_stats.uploaded_texture_count,
			texture_stats.uploaded_byte_count);
		rsx_log.warning("Metal sampled texture cache currently supports only raw linear CELL_GCM_TEXTURE_B8 uploads; swizzled, compressed, mipmapped, depth, and converted formats remain gated");
	}
}
