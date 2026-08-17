#include "stdafx.h"
#include "MTLTextureCache.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_set>

namespace mtl
{
	namespace
	{
		[[nodiscard]] u32 mip_dimension(u32 value, u32 level)
		{
			return std::max(value >> level, 1u);
		}

		[[nodiscard]] u64 image_bytes(const image& resource)
		{
			if (resource.allocation())
				return resource.allocation().size();
			const auto format = describe_native_format(resource.format());
			u64 result = 0;
			for (u32 level = 0; level < resource.mipmaps(); ++level)
			{
				const u64 blocks_x = utils::aligned_div(mip_dimension(resource.width(), level),
					static_cast<u32>(format.block_width));
				const u64 blocks_y = utils::aligned_div(mip_dimension(resource.height(), level),
					static_cast<u32>(format.block_height));
				result += blocks_x * blocks_y * mip_dimension(resource.depth(), level) *
					resource.layers() * format.bytes_per_block;
			}
			return result;
		}

		[[nodiscard]] subresource_range whole_range(const image& resource)
		{
			return {
				.first_mip = 0,
				.mip_count = resource.mipmaps(),
				.first_slice = 0,
				.slice_count = resource.layers(),
				.color = bool(resource.aspects() & texture_aspect_color),
				.depth = bool(resource.aspects() & texture_aspect_depth),
				.stencil = bool(resource.aspects() & texture_aspect_stencil),
			};
		}

		[[nodiscard]] texture_subresource make_subresource(const image& resource,
			u32 level = 0, u32 slice = 0)
		{
			return {
				.mip_level = level,
				.array_slice = slice,
				.aspects = resource.aspects(),
			};
		}

		[[nodiscard]] u32 native_texel_width(const image& resource)
		{
			const auto format = describe_native_format(resource.format());
			if (!format || format.block_width != 1 || format.block_height != 1)
				fmt::throw_exception("Metal texture-cache operation requires an uncompressed texel format");
			return format.bytes_per_block;
		}

		void clear_image(command_buffer& command, image& resource)
		{
			if (!command.is_recording() || resource.samples() != 1)
				fmt::throw_exception("Invalid Metal temporary-image clear");
			if (command.active_encoder() != encoder_kind::none)
				command.end_encoding();

			id<MTLTexture> texture = resource.native_handle();
			for (u32 level = 0; level < resource.mipmaps(); ++level)
			{
				const u32 slices = resource.type() == texture_type::texture_3d
					? mip_dimension(resource.depth(), level) : resource.layers();
				for (u32 slice = 0; slice < slices; ++slice)
				{
					MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
					if (resource.aspects() & texture_aspect_color)
					{
						pass.colorAttachments[0].texture = texture;
						pass.colorAttachments[0].level = level;
						pass.colorAttachments[0].slice = resource.type() == texture_type::texture_3d ? 0 : slice;
						pass.colorAttachments[0].depthPlane = resource.type() == texture_type::texture_3d ? slice : 0;
						pass.colorAttachments[0].loadAction = MTLLoadActionClear;
						pass.colorAttachments[0].storeAction = MTLStoreActionStore;
						pass.colorAttachments[0].clearColor = MTLClearColorMake(0., 0., 0., 0.);
					}
					if (resource.aspects() & texture_aspect_depth)
					{
						pass.depthAttachment.texture = texture;
						pass.depthAttachment.level = level;
						pass.depthAttachment.slice = slice;
						pass.depthAttachment.loadAction = MTLLoadActionClear;
						pass.depthAttachment.storeAction = MTLStoreActionStore;
						pass.depthAttachment.clearDepth = 1.;
					}
					if (resource.aspects() & texture_aspect_stencil)
					{
						pass.stencilAttachment.texture = texture;
						pass.stencilAttachment.level = level;
						pass.stencilAttachment.slice = slice;
						pass.stencilAttachment.loadAction = MTLLoadActionClear;
						pass.stencilAttachment.storeAction = MTLStoreActionStore;
						pass.stencilAttachment.clearStencil = 0;
					}
					static_cast<void>(command.begin_render_encoding((__bridge void*)pass));
					command.end_encoding();
				}
			}
			command.retain_native_object((__bridge void*)texture, true);
			resource.set_state({
				.queue = queue_kind::graphics,
				.stages = stage_fragment,
				.access = (resource.aspects() & texture_aspect_color) ? access_color_write : access_depth_stencil_write,
				.initialized = true,
			});
		}

		[[nodiscard]] image_conversion copy_conversion(const image& source, const image& destination)
		{
			const bool source_depth = bool(source.aspects() & texture_aspect_depth);
			const bool destination_depth = bool(destination.aspects() & texture_aspect_depth);
			image_conversion result;
			if (source_depth && destination_depth)
				result.kind = image_conversion_kind::none;
			else if (source_depth)
				result.kind = image_conversion_kind::depth_to_color;
			else if (destination_depth)
				result.kind = image_conversion_kind::color_to_depth;
			else
				result.kind = image_conversion_kind::color_to_color;
			return result;
		}

		[[nodiscard]] u64 align_transfer(u64 value)
		{
			return utils::align(value, 256ull);
		}

		struct texture_cache_blitter
		{
			void scale_image(command_buffer& command, image* source, image* destination,
				const areai& source_area, const areai& destination_area, bool interpolate,
				const rsx::typeless_xfer& transfer) const
			{
				ensure(source && destination);
				const u8 source_aspect = source->aspects() & texture_aspect_depth
					? texture_aspect_depth : texture_aspect_color;
				const u8 destination_aspect = destination->aspects() & texture_aspect_depth
					? texture_aspect_depth : texture_aspect_color;
				image_conversion conversion = copy_conversion(*source, *destination);
				if (!transfer.src_is_typeless && !transfer.dst_is_typeless &&
					source_aspect == destination_aspect && source->format() == destination->format())
				{
					conversion.kind = image_conversion_kind::none;
				}
				const image_scale_region region{
					.source = {.aspects = source_aspect},
					.destination = {.aspects = destination_aspect},
					.source_box = {source_area.x1, source_area.y1, 0,
						source_area.x2, source_area.y2, 1},
					.destination_box = {destination_area.x1, destination_area.y1, 0,
						destination_area.x2, destination_area.y2, 1},
				};
				mtl::scale_image(command, *source, *destination, region,
					interpolate ? image_filter::linear : image_filter::nearest, conversion);
			}
		};
	}

	struct cached_texture_section::dma_completion_state
	{
		std::mutex mutex;
		std::condition_variable condition;
		bool completed = false;
		bool succeeded = false;
	};

	void cached_texture_section::release_dma_resources()
	{
		m_dma_buffer.reset();
		m_dma_completion.reset();
		m_dma_offset = 0;
		m_dma_length = 0;
		m_dma_mapped = false;
	}

	void cached_texture_section::prepare_dma_completion(command_buffer& command)
	{
		auto completion = std::make_shared<dma_completion_state>();
		m_dma_completion = completion;
		command.notify_on_completion([completion](bool succeeded)
		{
			{
				std::lock_guard lock(completion->mutex);
				completion->completed = true;
				completion->succeeded = succeeded;
			}
			completion->condition.notify_all();
		});
	}

	void cached_texture_section::create(u16 new_width, u16 new_height, u16 new_depth,
		u16 new_mipmaps, image* resource, u32 pitch, bool managed, u32 format, bool swap_bytes)
	{
		auto* new_texture = dynamic_cast<viewable_image*>(resource);
		ensure(new_texture);
		ensure(!exists() || !is_managed() || m_vram_texture == new_texture);

		if (m_vram_texture != new_texture && !m_managed_texture &&
			get_protection() == utils::protection::no && m_vram_texture)
		{
			if (auto* previous = as_render_target(m_vram_texture))
				previous->on_swap_out();
			if (!managed)
			{
				if (auto* incoming = as_render_target(resource))
					incoming->on_swap_in(is_locked());
			}
		}

		m_vram_texture = new_texture;
		ensure(pitch);
		width = new_width;
		height = new_height;
		depth = new_depth;
		mipmaps = new_mipmaps;
		rsx_pitch = pitch;
		gcm_format = format;
		pack_unpack_swap_bytes = swap_bytes;
		m_native_format = resource->format();
		m_allocator = &static_cast<texture_cache*>(m_tex_cache)->allocator();

		if (managed)
			m_managed_texture.reset(new_texture);
		else
			ensure(!m_managed_texture);

		if (auto* target = as_render_target(resource))
			swizzled = target->raster_type != rsx::surface_raster_type::linear;

		if (synchronized)
		{
			release_dma_resources();
			synchronized = false;
			flushed = false;
			sync_timestamp = 0;
		}
		baseclass::on_section_resources_created();
	}

	void cached_texture_section::set_dimensions(u16 new_width, u16 new_height,
		u16 new_depth, u32 pitch)
	{
		ensure(!is_locked() && pitch);
		width = new_width;
		height = new_height;
		depth = new_depth;
		rsx_pitch = pitch;
	}

	void cached_texture_section::set_unpack_swap_bytes(bool swap_bytes)
	{
		pack_unpack_swap_bytes = swap_bytes;
	}

	void cached_texture_section::set_rsx_pitch(u32 pitch)
	{
		ensure(!is_locked() && pitch);
		rsx_pitch = pitch;
	}

	void cached_texture_section::dma_transfer(command_buffer& command, image* source,
		const areai& source_area, const utils::address_range32& valid_range, u32 pitch)
	{
		ensure(source && source->samples() == 1 && source_area.width() > 0 && source_area.height() > 0);
		if (!command.is_recording())
			fmt::throw_exception("Metal texture readback requires active command recording");
		if (!m_device)
			m_device = &command.allocator().owner();
		if (!m_allocator)
			m_allocator = &static_cast<texture_cache*>(m_tex_cache)->allocator();
		release_dma_resources();

		const u32 transfer_width = static_cast<u32>(source_area.width());
		const u32 transfer_height = static_cast<u32>(source_area.height());
		const auto format = describe_native_format(source->format());
		if (!format || format.block_width != 1 || format.block_height != 1)
			fmt::throw_exception("Metal texture readback cannot directly flush a compressed surface");

		const bool combined_depth_stencil =
			(source->aspects() & (texture_aspect_depth | texture_aspect_stencil)) ==
			(texture_aspect_depth | texture_aspect_stencil);
		const u32 output_bpp = combined_depth_stencil ? 4u : format.bytes_per_block;
		real_pitch = output_bpp * transfer_width;
		rsx_pitch = pitch;
		const u64 output_length = static_cast<u64>(real_pitch) * transfer_height;
		const u64 depth_offset = combined_depth_stencil ? align_transfer(output_length) : 0;
		const u64 stencil_offset = combined_depth_stencil
			? align_transfer(depth_offset + static_cast<u64>(transfer_width) * transfer_height * 4) : 0;
		const u64 required_size = combined_depth_stencil
			? align_transfer(stencil_offset + static_cast<u64>(transfer_width) * transfer_height)
			: align_transfer(output_length);
		m_dma_buffer = std::make_unique<buffer>(*m_allocator, buffer_create_info{
			.size = required_size,
			.usage = buffer_usage_storage | buffer_usage_copy_source | buffer_usage_copy_destination,
			.storage = storage_mode::shared,
			.cache = cpu_cache_mode::default_cache,
			.access = cpu_access::read_write,
			.hazards = hazard_tracking::tracked,
			.pool = allocation_pool::texture_cache,
			.label = "RPCS3 texture-cache readback",
		});

		if (combined_depth_stencil)
		{
			const std::array regions = {
				buffer_image_copy_region{
					.buffer_offset = depth_offset,
					.bytes_per_row = static_cast<u64>(transfer_width) * 4,
					.bytes_per_image = static_cast<u64>(transfer_width) * transfer_height * 4,
					.subresource = {.aspects = texture_aspect_depth},
					.origin = {static_cast<u32>(source_area.x1), static_cast<u32>(source_area.y1), 0},
					.extent = {transfer_width, transfer_height, 1},
				},
				buffer_image_copy_region{
					.buffer_offset = stencil_offset,
					.bytes_per_row = transfer_width,
					.bytes_per_image = static_cast<u64>(transfer_width) * transfer_height,
					.subresource = {.aspects = texture_aspect_stencil},
					.origin = {static_cast<u32>(source_area.x1), static_cast<u32>(source_area.y1), 0},
					.extent = {transfer_width, transfer_height, 1},
				},
			};
			download_image(command, *source, *m_dma_buffer, regions);
			const u32 pixels = transfer_width * transfer_height;
			if (source->format() == static_cast<u64>(MTLPixelFormatDepth32Float_Stencil8))
			{
				get_compute_task<cs_gather_d32x8<true, true>>()->run(command, *m_dma_buffer,
					0, pixels * 4, static_cast<u32>(depth_offset), static_cast<u32>(stencil_offset));
			}
			else
			{
				get_compute_task<cs_gather_d24x8<true>>()->run(command, *m_dma_buffer,
					0, pixels * 4, static_cast<u32>(depth_offset), static_cast<u32>(stencil_offset));
			}
		}
		else
		{
			const buffer_image_copy_region region{
				.buffer_offset = 0,
				.bytes_per_row = real_pitch,
				.bytes_per_image = output_length,
				.subresource = {.aspects = source->aspects()},
				.origin = {static_cast<u32>(source_area.x1), static_cast<u32>(source_area.y1), 0},
				.extent = {transfer_width, transfer_height, 1},
			};
			download_image(command, *source, *m_dma_buffer, std::span{&region, 1});
			if (pack_unpack_swap_bytes)
			{
				if (format.element_size == 2)
					get_compute_task<cs_shuffle_16>()->run(command, *m_dma_buffer, static_cast<u32>(output_length));
				else if (format.element_size == 4)
					get_compute_task<cs_shuffle_32>()->run(command, *m_dma_buffer, static_cast<u32>(output_length));
			}
		}

		m_dma_offset = 0;
		m_dma_length = output_length;
		prepare_dma_completion(command);
		command.set_flag(command_has_dma_transfer);

		if (context == rsx::texture_upload_context::dma)
			gcm_format = output_bpp == 2 ? CELL_GCM_TEXTURE_R5G6B5 : CELL_GCM_TEXTURE_A8R8G8B8;
		synchronized = true;
		sync_timestamp = rsx::get_shared_tag();
		(void)valid_range;
	}

	void cached_texture_section::copy_texture(command_buffer& command, bool miss)
	{
		ensure(exists());
		if (miss)
			baseclass::on_miss();
		else
			baseclass::on_speculative_flush();

		image* locked_resource = m_vram_texture;
		u32 transfer_width = width;
		u32 transfer_height = height;
		u32 transfer_x = 0;
		u32 transfer_y = 0;
		if (context == rsx::texture_upload_context::framebuffer_storage)
		{
			auto* surface = get_render_target();
			ensure(surface);
			surface->memory_barrier(command, rsx::surface_access::transfer_read);
			locked_resource = surface->get_surface(rsx::surface_access::transfer_read);
			transfer_width *= surface->samples_x;
			transfer_height *= surface->samples_y;
		}

		image* target = locked_resource;
		if (transfer_width != locked_resource->width() || transfer_height != locked_resource->height())
		{
			const image_create_info info{
				.type = texture_type::texture_2d,
				.formats = get_view_compatibility(locked_resource->format()),
				.width = transfer_width,
				.height = transfer_height,
				.depth = 1,
				.mip_levels = 1,
				.array_layers = 1,
				.sample_count = 1,
				.usage = texture_usage_shader_read | texture_usage_render_target |
					texture_usage_depth_stencil | texture_usage_copy_source | texture_usage_copy_destination |
					texture_usage_pixel_format_view,
				.aspects = locked_resource->aspects(),
				.storage = storage_mode::private_,
				.pool = allocation_pool::texture_cache,
				.label = "RPCS3 texture-cache scaled readback",
			};
			if (!m_scaled_texture || m_scaled_texture->width() != transfer_width ||
				m_scaled_texture->height() != transfer_height ||
				m_scaled_texture->format() != locked_resource->format())
			{
				m_scaled_texture = std::make_unique<viewable_image>(*m_allocator, info);
			}
			const image_scale_region scale{
				.source = make_subresource(*locked_resource),
				.destination = make_subresource(*m_scaled_texture),
				.source_box = {0, 0, 0, static_cast<s32>(locked_resource->width()),
					static_cast<s32>(locked_resource->height()), 1},
				.destination_box = {0, 0, 0, static_cast<s32>(transfer_width),
					static_cast<s32>(transfer_height), 1},
			};
			scale_image(command, *locked_resource, *m_scaled_texture, scale,
				is_depth_texture() ? image_filter::nearest : image_filter::linear);
			target = m_scaled_texture.get();
		}

		const auto valid_range = get_confirmed_range();
		if (const auto section_range = get_section_range(); section_range != valid_range)
		{
			if (const u32 offset = valid_range.start - get_section_base())
			{
				const u32 bytes_per_texel = context == rsx::texture_upload_context::framebuffer_storage
					? rsx::get_format_block_size_in_bytes(gcm_format) : native_texel_width(*target);
				transfer_y = offset / rsx_pitch;
				transfer_x = (offset % rsx_pitch) / bytes_per_texel;
				ensure(transfer_width >= transfer_x && transfer_height >= transfer_y);
				transfer_width -= transfer_x;
				transfer_height -= transfer_y;
			}
			if (const u32 tail = section_range.end - valid_range.end)
			{
				const u32 rows = tail / rsx_pitch;
				ensure(transfer_height >= rows);
				transfer_height -= rows;
			}
		}
		const areai source_area{
			static_cast<s32>(transfer_x), static_cast<s32>(transfer_y),
			static_cast<s32>(transfer_x + transfer_width),
			static_cast<s32>(transfer_y + transfer_height),
		};
		dma_transfer(command, target, source_area, valid_range, rsx_pitch);
	}

	void cached_texture_section::imp_flush()
	{
		AUDIT(synchronized);
		ensure(real_pitch);
		const auto valid_range = get_confirmed_range();
		ensure(valid_range.valid());
		const u32 valid_length = valid_range.length();
		const u32 valid_offset = valid_range.start - get_section_base();
		u32 mapped_offset;
		u32 mapped_length;
		if (real_pitch != rsx_pitch)
		{
			const u32 offset_x = valid_offset % rsx_pitch;
			const u32 offset_y = valid_offset / rsx_pitch;
			mapped_offset = offset_y * real_pitch + offset_x;
			const u32 available = (get_section_size() / rsx_pitch) * real_pitch +
				std::min<u32>(get_section_size() % rsx_pitch, real_pitch);
			mapped_length = std::min(available - mapped_offset, valid_length);
		}
		else
		{
			mapped_offset = valid_offset;
			mapped_length = valid_length;
		}
		auto* source = static_cast<u8*>(map_synchronized(mapped_offset, mapped_length));
		auto copy_filtered = [this](u32 destination_address, const u8* input, u32 length)
		{
			const auto copy_range = utils::address_range32::start_length(destination_address, length);
			if (flush_exclusions.empty() || !copy_range.overlaps(flush_exclusions))
			{
				std::memcpy(get_ptr(destination_address), input, length);
				return;
			}
			if (copy_range.inside(flush_exclusions))
				return;
			utils::address_range_vector32 included;
			included.merge(copy_range);
			included.exclude(flush_exclusions);
			for (const auto& part : included)
			{
				if (!part.valid())
					continue;
				const u32 offset = part.start - destination_address;
				std::memcpy(get_ptr(part.start), input + offset, part.length());
			}
		};
		if (real_pitch >= rsx_pitch || valid_length <= rsx_pitch)
		{
			copy_filtered(valid_range.start, source, valid_length);
		}
		else
		{
			u32 destination_address = valid_range.start;
			const u8* input = source;
			for (s32 remaining = valid_length; remaining > 0; remaining -= rsx_pitch)
			{
				const u32 length = std::min<u32>(real_pitch, remaining);
				copy_filtered(destination_address, input, length);
				input += real_pitch;
				destination_address += rsx_pitch;
			}
		}
		const auto range = context == rsx::texture_upload_context::framebuffer_storage
			? get_section_range() : get_confirmed_range();
		if (const auto tiled = rsx::get_current_renderer()->get_tiled_memory_region(range))
		{
			const u32 available = tiled.tile->size - (range.start - tiled.base_address);
			const u32 length = std::min<u32>(range.length(), available);
			auto* guest = get_ptr<u8>(range.start);
			rsx::simple_array<u8> encoded(length);
			std::memcpy(encoded.data(), guest, length);
			const u32 bytes = rsx::get_format_block_size_in_bytes(gcm_format);
			if (bytes == 2)
			{
				rsx::tile_texel_data<u16>(encoded.data(), guest, tiled.base_address,
					range.start - tiled.base_address, tiled.tile->size, tiled.tile->bank,
					tiled.tile->pitch, width, height);
			}
			else if (bytes == 4)
			{
				rsx::tile_texel_data<u32>(encoded.data(), guest, tiled.base_address,
					range.start - tiled.base_address, tiled.tile->size, tiled.tile->bank,
					tiled.tile->pitch, width, height);
			}
			else
			{
				fmt::throw_exception("Metal tiled texture readback requires two- or four-byte texels");
			}
			std::memcpy(guest, encoded.data(), length);
		}
		else if (is_swizzled())
		{
			auto* data = get_ptr(range.start);
			rsx::simple_array<u8> linear(rsx_pitch * height);
			std::memcpy(linear.data(), data, linear.size());
			switch (gcm_format)
			{
			case CELL_GCM_TEXTURE_A8R8G8B8:
			case CELL_GCM_TEXTURE_DEPTH24_D8:
				rsx::convert_linear_swizzle<u32, false>(linear.data(), data, width, height, rsx_pitch);
				break;
			case CELL_GCM_TEXTURE_R5G6B5:
			case CELL_GCM_TEXTURE_DEPTH16:
				rsx::convert_linear_swizzle<u16, false>(linear.data(), data, width, height, rsx_pitch);
				break;
			default:
				fmt::throw_exception("Unsupported swizzled Metal readback format 0x%x", gcm_format);
			}
		}
	}

	void cached_texture_section::dma_abort()
	{
		ensure(synchronized && !flushed);
		release_dma_resources();
	}

	void* cached_texture_section::map_synchronized(u32 offset, u32 size)
	{
		ensure(synchronized && m_dma_buffer && m_dma_completion);
		{
			std::unique_lock lock(m_dma_completion->mutex);
			m_dma_completion->condition.wait(lock, [this]
			{
				return m_dma_completion->completed;
			});
			if (!m_dma_completion->succeeded)
				fmt::throw_exception("Metal texture-cache readback command failed");
		}
		if (static_cast<u64>(offset) + size > m_dma_length)
			fmt::throw_exception("Metal texture-cache readback map exceeds the completed transfer");
		m_dma_mapped = true;
		return m_dma_buffer->map(m_dma_offset + offset, size);
	}

	void cached_texture_section::finish_flush()
	{
		if (m_dma_mapped)
		{
			m_dma_buffer->unmap();
			m_dma_mapped = false;
		}
	}

	void cached_texture_section::destroy()
	{
		if (!exists() && context != rsx::texture_upload_context::dma)
			return;
		m_tex_cache->on_section_destroyed(*this);
		m_vram_texture = nullptr;
		ensure(!m_managed_texture);
		m_scaled_texture.reset();
		release_dma_resources();
		baseclass::on_section_resources_destroyed();
	}

	bool cached_texture_section::exists() const
	{
		return m_vram_texture != nullptr || context == rsx::texture_upload_context::dma;
	}

	bool cached_texture_section::is_managed() const
	{
		return !m_vram_texture || bool(m_managed_texture);
	}

	bool cached_texture_section::is_flushed() const
	{
		return flushed;
	}

	bool cached_texture_section::is_depth_texture() const
	{
		return m_vram_texture && bool(m_vram_texture->aspects() & texture_aspect_depth);
	}

	bool cached_texture_section::has_compatible_format(image* resource) const
	{
		return resource && m_vram_texture && formats_are_bitcast_compatible(*resource, *m_vram_texture);
	}

	u64 cached_texture_section::get_format() const
	{
		return context == rsx::texture_upload_context::dma
			? static_cast<u64>(MTLPixelFormatR32Uint) : m_native_format;
	}

	image_view* cached_texture_section::get_view(const rsx::texture_channel_remap_t& remap)
	{
		ensure(m_vram_texture);
		component_mapping mapping;
		switch (get_view_flags())
		{
		case rsx::component_order::default_:
			mapping = apply_swizzle_remap(get_component_mapping(gcm_format), remap);
			break;
		case rsx::component_order::native:
			mapping = default_component_map;
			break;
		case rsx::component_order::swapped_native:
			mapping = {component_swizzle::alpha, component_swizzle::red,
				component_swizzle::green, component_swizzle::blue};
			break;
		default:
			fmt::throw_exception("Unknown Metal texture component order");
		}
		return m_vram_texture->get_view(m_vram_texture->format(), m_vram_texture->type(),
			mapping, whole_range(*m_vram_texture));
	}

	image_view* cached_texture_section::get_raw_view()
	{
		ensure(m_vram_texture);
		return m_vram_texture->get_view(m_vram_texture->format(), m_vram_texture->type(),
			default_component_map, whole_range(*m_vram_texture));
	}

	viewable_image* cached_texture_section::get_raw_texture() const
	{
		return m_managed_texture.get();
	}

	std::unique_ptr<viewable_image>& cached_texture_section::get_texture()
	{
		return m_managed_texture;
	}

	render_target* cached_texture_section::get_render_target() const
	{
		return as_render_target(m_vram_texture);
	}

	void cached_texture_section::sync_surface_memory(
		const rsx::simple_array<cached_texture_section*>& surfaces)
	{
		auto* target = get_render_target();
		ensure(target);
		target->sync_tag();
		for (auto* surface : surfaces)
		{
			if (surface)
				target->inherit_surface_contents(surface->get_render_target());
		}
	}

	u64 hash_image_properties(const image_create_info& info)
	{
		auto combine = [](u64 seed, u64 value)
		{
			return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
		};
		u64 result = info.formats.base_format;
		result = combine(result, static_cast<u8>(info.type));
		result = combine(result, info.width);
		result = combine(result, info.height);
		result = combine(result, info.depth);
		result = combine(result, info.mip_levels);
		result = combine(result, info.array_layers);
		result = combine(result, info.sample_count);
		result = combine(result, info.usage);
		result = combine(result, info.aspects);
		result = combine(result, static_cast<u8>(info.storage));
		result = combine(result, info.shareable);
		for (const u64 format : info.formats.view_formats)
			result = combine(result, format);
		return result;
	}

	texture_type get_texture_type(rsx::texture_dimension_extended type, u16 depth, u16 layers)
	{
		switch (type)
		{
		case rsx::texture_dimension_extended::texture_dimension_1d:
			return layers > 1 ? texture_type::texture_1d_array : texture_type::texture_1d;
		case rsx::texture_dimension_extended::texture_dimension_2d:
			return layers > 1 ? texture_type::texture_2d_array : texture_type::texture_2d;
		case rsx::texture_dimension_extended::texture_dimension_cubemap:
			return layers > 6 ? texture_type::texture_cube_array : texture_type::texture_cube;
		case rsx::texture_dimension_extended::texture_dimension_3d:
			ensure(depth > 1);
			return texture_type::texture_3d;
		default:
			fmt::throw_exception("Invalid RSX texture dimension %u", static_cast<u32>(type));
		}
	}

	texture_cache::cached_image_reference::cached_image_reference(texture_cache& cache,
		std::unique_ptr<viewable_image>& previous)
		: data(std::move(previous)), parent(&cache)
	{
		ensure(data);
	}

	texture_cache::cached_image_reference::~cached_image_reference()
	{
		if (!data || !parent)
			return;
		data->set_state({});
		data->clear_views();
		const u64 key = hash_image_properties(data->info());
		std::lock_guard lock(parent->m_cached_pool_lock);
		if (!parent->m_cache_is_exiting)
		{
			parent->m_cached_memory_size += image_bytes(*data);
			parent->m_cached_images.emplace_front(key, data);
		}
	}

	texture_cache::cached_image::cached_image(u64 new_key,
		std::unique_ptr<viewable_image>& resource)
		: key(new_key), data(std::move(resource))
	{
	}

	void texture_cache::on_section_destroyed(cached_texture_section& texture)
	{
		if (texture.is_managed() && texture.exists() && texture.get_texture())
		{
			auto reference = std::make_unique<cached_image_reference>(*this, texture.get_texture());
			get_resource_manager().retire(reference, {
				.resource_class = managed_resource_class::texture,
				.bytes = image_bytes(*reference->data),
				.label = "RPCS3 cached texture section",
			});
		}
	}

	void texture_cache::clear()
	{
		{
			std::lock_guard lock(m_cached_pool_lock);
			m_cache_is_exiting = true;
		}
		baseclass::clear();
		m_temporary_images.clear();
		m_cached_images.clear();
		m_cached_memory_size = 0;
	}

	component_mapping texture_cache::apply_component_mapping_flags(u32 format,
		rsx::component_order flags, const rsx::texture_channel_remap_t& remap) const
	{
		switch (format)
		{
		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			return {component_swizzle::red, component_swizzle::red,
				component_swizzle::red, component_swizzle::red};
		default:
			break;
		}
		switch (flags)
		{
		case rsx::component_order::default_:
			return apply_swizzle_remap(get_component_mapping(format), remap);
		case rsx::component_order::native:
			return default_component_map;
		case rsx::component_order::swapped_native:
			return {component_swizzle::alpha, component_swizzle::red,
				component_swizzle::green, component_swizzle::blue};
		default:
			fmt::throw_exception("Invalid Metal component-order flag");
		}
	}

	void texture_cache::copy_transfer_regions_impl(command_buffer& command, image* destination,
		const rsx::simple_array<copy_region_descriptor>& sections) const
	{
		ensure(destination);
		const u32 destination_bpp = native_texel_width(*destination);
		for (const auto& section : sections)
		{
			if (!section.src)
				continue;
			image* source = section.src;
			u32 source_x = section.src_x;
			u32 source_y = section.src_y;
			u32 source_width = section.src_w;
			u32 source_height = section.src_h;
			rsx::flags32_t transform = section.xform;
			if (transform & rsx::surface_transform::coordinate_transform)
			{
				const u32 source_bpp = native_texel_width(*source);
				source_x = source_x * destination_bpp / source_bpp;
				source_width = utils::aligned_div(source_width * destination_bpp, source_bpp);
				transform &= ~rsx::surface_transform::coordinate_transform;
			}
			if (auto* surface = as_render_target(source))
				surface->transform_samples_to_pixels(source_x, source_width, source_y, source_height);
			ensure(transform == rsx::surface_transform::identity);

			const texture_subresource source_subresource{
				.mip_level = 0,
				.array_slice = 0,
				.aspects = source->aspects() & texture_aspect_depth
					? texture_aspect_depth : texture_aspect_color,
			};
			texture_subresource destination_subresource{
				.mip_level = section.level,
				.array_slice = destination->type() == texture_type::texture_3d ? 0u : section.dst_z,
				.aspects = destination->aspects() & texture_aspect_depth
					? texture_aspect_depth : texture_aspect_color,
			};
			const bool direct = source_width == section.dst_w && source_height == section.dst_h &&
				formats_are_bitcast_compatible(*source, *destination);
			if (direct)
			{
				const image_copy_region region{
					.source = source_subresource,
					.destination = destination_subresource,
					.source_origin = {source_x, source_y, 0},
					.destination_origin = {section.dst_x, section.dst_y,
						destination->type() == texture_type::texture_3d ? section.dst_z : 0u},
					.extent = {source_width, source_height, 1},
				};
				copy_image(command, *source, *destination, std::span{&region, 1});
			}
			else
			{
				const image_scale_region region{
					.source = source_subresource,
					.destination = destination_subresource,
					.source_box = {static_cast<s32>(source_x), static_cast<s32>(source_y), 0,
						static_cast<s32>(source_x + source_width),
						static_cast<s32>(source_y + source_height), 1},
					.destination_box = {section.dst_x, section.dst_y,
						destination->type() == texture_type::texture_3d ? section.dst_z : 0,
						section.dst_x + section.dst_w, section.dst_y + section.dst_h,
						destination->type() == texture_type::texture_3d ? section.dst_z + 1 : 1},
				};
				scale_image(command, *source, *destination, region, image_filter::nearest,
					copy_conversion(*source, *destination));
			}
		}
	}

	image* texture_cache::get_template_from_collection_impl(
		const rsx::simple_array<copy_region_descriptor>& sections) const
	{
		image* result = nullptr;
		for (const auto& section : sections)
		{
			if (!section.src)
				continue;
			if (!result)
				result = section.src;
			else if (!formats_are_bitcast_compatible(describe_native_format(result->format()),
				describe_native_format(section.src->format()), false))
				return nullptr;
		}
		return result;
	}

	std::unique_ptr<viewable_image> texture_cache::find_cached_image(const image_create_info& info)
	{
		std::lock_guard lock(m_cached_pool_lock);
		const u64 desired_key = hash_image_properties(info);
		for (auto iterator = m_cached_images.begin(); iterator != m_cached_images.end(); ++iterator)
		{
			if (iterator->key == desired_key && (iterator->data->info().usage & info.usage) == info.usage)
			{
				auto result = std::move(iterator->data);
				m_cached_images.erase(iterator);
				m_cached_memory_size -= image_bytes(*result);
				return result;
			}
		}
		return {};
	}

	std::unique_ptr<viewable_image> texture_cache::create_temporary_subresource_storage(
		rsx::format_class format_class, u64 format, u16 width, u16 height, u16 depth,
		u16 layers, u8 mipmaps, texture_type type, u32 image_flags, u32 usage_flags)
	{
		(void)format_class;
		image_create_info info{
			.type = type,
			.formats = get_view_compatibility(format),
			.width = width,
			.height = height,
			.depth = depth,
			.mip_levels = mipmaps,
			.array_layers = layers,
			.sample_count = 1,
			.usage = texture_usage_copy_source | texture_usage_copy_destination |
				texture_usage_shader_read | texture_usage_pixel_format_view | usage_flags,
			.aspects = get_aspect_flags(format),
			.storage = storage_mode::private_,
			.hazards = hazard_tracking::tracked,
			.pool = allocation_pool::texture_cache,
			.label = "RPCS3 temporary texture-cache image",
			.shareable = bool(image_flags & texture_create_flag::shareable),
		};
		auto result = find_cached_image(info);
		if (!result)
		{
			info.allow_failure = true;
			result = std::make_unique<viewable_image>();
			result->create(*m_allocator, info);
			if (!*result)
				return {};
		}
		return result;
	}

	void texture_cache::dispose_reusable_image(std::unique_ptr<viewable_image>& resource)
	{
		if (!resource)
			return;
		auto reference = std::make_unique<cached_image_reference>(*this, resource);
		get_resource_manager().retire(reference, {
			.resource_class = managed_resource_class::texture,
			.bytes = image_bytes(*reference->data),
			.label = "RPCS3 reusable temporary texture",
		});
	}

	image_view* texture_cache::create_temporary_subresource_view_impl(command_buffer& command,
		image* source, texture_type image_type, texture_type view_type, u32 format,
		u16 x, u16 y, u16 width, u16 height, u16 depth, u8 mipmaps,
		const rsx::texture_channel_remap_t& remap, bool copy)
	{
		(void)image_type;
		const auto native_format = get_sampler_format(m_device->info(), format);
		if (!native_format)
			fmt::throw_exception("Unsupported Metal temporary texture format 0x%x", format);
		const u16 layers = view_type == texture_type::texture_cube ? 6 : 1;
		auto storage = create_temporary_subresource_storage(rsx::classify_format(format),
			native_format.pixel_format, width, height, depth, layers, mipmaps, view_type,
			view_type == texture_type::texture_cube ? texture_create_flag::mutable_format : 0,
			(native_format.aspects & texture_aspect_depth)
				? texture_usage_depth_stencil : texture_usage_render_target);
		if (!storage)
			return nullptr;

		storage->set_debug_name(fmt::format("Temporary texture-cache view 0x%x", format));
		const component_mapping mapping = apply_component_mapping_flags(format,
			rsx::component_order::default_, remap);
		const subresource_range range{
			.first_mip = 0,
			.mip_count = mipmaps,
			.first_slice = 0,
			.slice_count = layers,
			.color = bool(storage->aspects() & texture_aspect_color),
			.depth = bool(storage->aspects() & texture_aspect_depth),
			.stencil = bool(storage->aspects() & texture_aspect_stencil),
		};
		image_view* view = storage->get_view(storage->format(), view_type, mapping, range);
		if (copy)
		{
			ensure(source);
			rsx::simple_array<copy_region_descriptor> region = {{
				.src = source,
				.xform = rsx::surface_transform::coordinate_transform,
				.src_x = x,
				.src_y = y,
				.src_w = width,
				.src_h = height,
				.dst_w = width,
				.dst_h = height,
			}};
			copy_transfer_regions_impl(command, storage.get(), region);
		}
		image* key = storage.get();
		const auto [iterator, inserted] = m_temporary_images.emplace(key,
			active_temporary_image{std::move(storage), 1});
		ensure(inserted && iterator->second.data);
		return view;
	}

	image_view* texture_cache::create_temporary_subresource_view(command_buffer& command,
		const deferred_subresource& description)
	{
		ensure(description.external_handle);
		return create_temporary_subresource_view_impl(command, description.external_handle,
			description.external_handle->type(), texture_type::texture_2d, description.gcm_format,
			description.x, description.y, description.width, description.height, 1, 1,
			description.remap, true);
	}

	image_view* texture_cache::generate_cubemap_from_images(command_buffer& command,
		const deferred_subresource& description)
	{
		const u8 mip_count = 1 + description.sections_to_copy.reduce(0,
			FN(std::max<u8>(x, y.level)));
		image* pattern = get_template_from_collection_impl(description.sections_to_copy);
		auto* result = create_temporary_subresource_view_impl(command, pattern,
			texture_type::texture_2d, texture_type::texture_cube, description.gcm_format,
			0, 0, description.width, description.height, 1, mip_count,
			description.remap, false);
		if (!result)
			return nullptr;
		image* destination = result->image();
		if (description.force_bg_load)
			initialize_subresource_from_memory(command, destination, description,
				rsx::texture_dimension_extended::texture_dimension_cubemap);
		else
			clear_image(command, *destination);
		copy_transfer_regions_impl(command, destination, description.sections_to_copy);
		return result;
	}

	image_view* texture_cache::generate_3d_from_2d_images(command_buffer& command,
		const deferred_subresource& description)
	{
		image* pattern = get_template_from_collection_impl(description.sections_to_copy);
		auto* result = create_temporary_subresource_view_impl(command, pattern,
			texture_type::texture_3d, texture_type::texture_3d, description.gcm_format,
			0, 0, description.width, description.height, description.depth, 1,
			description.remap, false);
		if (!result)
			return nullptr;
		image* destination = result->image();
		if (description.force_bg_load)
			initialize_subresource_from_memory(command, destination, description,
				rsx::texture_dimension_extended::texture_dimension_3d);
		else
			clear_image(command, *destination);
		copy_transfer_regions_impl(command, destination, description.sections_to_copy);
		return result;
	}

	image_view* texture_cache::generate_atlas_from_images(command_buffer& command,
		const deferred_subresource& description)
	{
		image* pattern = get_template_from_collection_impl(description.sections_to_copy);
		auto* result = create_temporary_subresource_view_impl(command, pattern,
			texture_type::texture_2d, texture_type::texture_2d, description.gcm_format,
			0, 0, description.width, description.height, 1, 1,
			description.remap, false);
		if (!result)
			return nullptr;
		image* destination = result->image();
		if (description.force_bg_load)
			initialize_subresource_from_memory(command, destination, description,
				rsx::texture_dimension_extended::texture_dimension_2d);
		else
			clear_image(command, *destination);
		copy_transfer_regions_impl(command, destination, description.sections_to_copy);
		return result;
	}

	image_view* texture_cache::generate_2d_mipmaps_from_images(command_buffer& command,
		const deferred_subresource& description)
	{
		const u8 mipmaps = ::narrow<u8>(description.sections_to_copy.size());
		image* pattern = get_template_from_collection_impl(description.sections_to_copy);
		auto* result = create_temporary_subresource_view_impl(command, pattern,
			texture_type::texture_2d, texture_type::texture_2d, description.gcm_format,
			0, 0, description.width, description.height, 1, mipmaps,
			description.remap, false);
		if (!result)
			return nullptr;
		image* destination = result->image();
		if (description.force_bg_load)
			initialize_subresource_from_memory(command, destination, description,
				rsx::texture_dimension_extended::texture_dimension_2d);
		else
			clear_image(command, *destination);
		copy_transfer_regions_impl(command, destination, description.sections_to_copy);
		return result;
	}

	void texture_cache::release_temporary_subresource(image_view* view)
	{
		if (!view || !view->image())
			return;
		const auto found = m_temporary_images.find(view->image());
		ensure(found != m_temporary_images.end() && found->second.references);
		if (--found->second.references)
			return;
		auto resource = std::move(found->second.data);
		m_temporary_images.erase(found);
		dispose_reusable_image(resource);
	}

	void texture_cache::initialize_subresource_from_memory(command_buffer& command,
		image* destination, const deferred_subresource& description,
		rsx::texture_dimension_extended type) const
	{
		ensure(destination);
		const auto layouts = rsx::get_subresources_layout(description, type);
		mtl::upload_texture(command, *m_allocator, *destination, description.gcm_format,
			description.swizzled, layouts);
	}

	void texture_cache::update_image_contents(command_buffer& command,
		image_view* destination, image* source, u16 width, u16 height)
	{
		ensure(destination && destination->image() && source);
		rsx::simple_array<copy_region_descriptor> region = {{
			.src = source,
			.xform = rsx::surface_transform::identity,
			.src_w = width,
			.src_h = height,
			.dst_w = width,
			.dst_h = height,
		}};
		copy_transfer_regions_impl(command, destination->image(), region);
	}

	cached_texture_section* texture_cache::create_new_texture(command_buffer& command,
		const utils::address_range32& range, u16 width, u16 height, u16 depth,
		u16 mipmaps, u32 pitch, u32 format, rsx::texture_upload_context context,
		rsx::texture_dimension_extended dimension, bool swizzled,
		rsx::component_order component_order, rsx::flags32_t flags)
	{
		const u16 section_depth = depth;
		u16 layers = 1;
		texture_type native_type;
		switch (dimension)
		{
		case rsx::texture_dimension_extended::texture_dimension_1d:
			height = 1;
			depth = 1;
			native_type = texture_type::texture_1d;
			break;
		case rsx::texture_dimension_extended::texture_dimension_2d:
			depth = 1;
			native_type = texture_type::texture_2d;
			break;
		case rsx::texture_dimension_extended::texture_dimension_cubemap:
			depth = 1;
			layers = 6;
			native_type = texture_type::texture_cube;
			break;
		case rsx::texture_dimension_extended::texture_dimension_3d:
			native_type = texture_type::texture_3d;
			break;
		default:
			fmt::throw_exception("Invalid Metal texture dimension %u", static_cast<u32>(dimension));
		}

		const rsx::image_section_attributes_t search{
			.gcm_format = format,
			.width = width,
			.height = height,
			.depth = section_depth,
			.mipmaps = mipmaps,
		};
		const bool allow_dirty = context != rsx::texture_upload_context::framebuffer_storage;
		auto& region = *find_cached_texture(range, search, true, true, allow_dirty);
		ensure(!region.is_locked());

		const native_format_description native_format = get_sampler_format(m_device->info(), format);
		if (!native_format)
			fmt::throw_exception("Unsupported Metal sampler format 0x%x", format);
		const u32 usage = texture_usage_shader_read | texture_usage_copy_source |
			texture_usage_copy_destination | ((native_format.aspects & texture_aspect_depth)
				? texture_usage_depth_stencil : texture_usage_render_target) |
			texture_usage_pixel_format_view;
		image_create_info info{
			.type = native_type,
			.formats = get_view_compatibility(native_format.pixel_format),
			.width = width,
			.height = height,
			.depth = depth,
			.mip_levels = mipmaps,
			.array_layers = layers,
			.sample_count = 1,
			.usage = usage,
			.aspects = native_format.aspects,
			.storage = storage_mode::private_,
			.hazards = hazard_tracking::tracked,
			.pool = allocation_pool::texture_cache,
			.label = fmt::format("RPCS3 texture 0x%x", range.start),
			.shareable = bool(flags & texture_create_flag::shareable),
		};
		viewable_image* resource = nullptr;
		if (region.exists())
		{
			resource = region.get_raw_texture();
			const bool reusable = !(flags & texture_create_flag::do_not_reuse) && resource &&
				region.get_image_type() == dimension && resource->type() == native_type &&
				resource->format() == native_format.pixel_format && resource->width() == width &&
				resource->height() == height && resource->depth() == depth &&
				resource->mipmaps() == mipmaps && resource->layers() == layers &&
				(!info.shareable || resource->is_shareable());
			if (!reusable)
			{
				region.destroy();
				resource = nullptr;
			}
			else
			{
				region.set_dimensions(width, height, section_depth, pitch);
				if (flags & texture_create_flag::initialize_image_contents)
					clear_image(command, *resource);
			}
		}

		if (!resource)
		{
			auto storage = find_cached_image(info);
			if (!storage)
				storage = std::make_unique<viewable_image>(*m_allocator, info);
			resource = storage.release();
			region.reset(range);
			region.set_gcm_format(format);
			region.set_image_type(dimension);
			region.create(width, height, section_depth, mipmaps, resource, pitch, true, format);
			if (flags & texture_create_flag::initialize_image_contents)
				clear_image(command, *resource);
		}

		region.set_view_flags(component_order);
		region.set_context(context);
		region.set_swizzled(swizzled);
		region.set_dirty(false);
		switch (context)
		{
		case rsx::texture_upload_context::shader_read:
		case rsx::texture_upload_context::blit_engine_src:
			region.protect(utils::protection::ro);
			read_only_range = region.get_min_max(read_only_range, rsx::section_bounds::locked_range);
			break;
		case rsx::texture_upload_context::blit_engine_dst:
			region.set_unpack_swap_bytes(true);
			no_access_range = region.get_min_max(no_access_range, rsx::section_bounds::locked_range);
			break;
		default:
			fmt::throw_exception("Invalid Metal managed texture context %u", static_cast<u32>(context));
		}
		update_cache_tag();
		return &region;
	}

	cached_texture_section* texture_cache::create_nul_section(command_buffer& command,
		const utils::address_range32& range, const rsx::image_section_attributes_t& attributes,
		const rsx::GCM_tile_reference& tile, bool memory_load)
	{
		(void)command;
		auto& region = *find_cached_texture(range,
			rsx::image_section_attributes_t{.gcm_format = RSX_GCM_FORMAT_IGNORED},
			true, false, false);
		ensure(!region.is_locked());
		region.reset(range);
		region.create_dma_only(attributes.width, attributes.height, attributes.pitch);
		region.set_dirty(false);
		region.set_unpack_swap_bytes(true);
		if (memory_load && !tile && get_dma_pool())
		{
			static_cast<void>(map_dma(range.start, range.length()));
			load_dma(range.start, range.length());
		}
		no_access_range = region.get_min_max(no_access_range, rsx::section_bounds::locked_range);
		update_cache_tag();
		return &region;
	}

	cached_texture_section* texture_cache::upload_image_from_cpu(command_buffer& command,
		const utils::address_range32& range, u16 width, u16 height, u16 depth,
		u16 mipmaps, u32 pitch, u32 format, rsx::texture_upload_context context,
		const std::vector<rsx::subresource_layout>& layouts,
		rsx::texture_dimension_extended dimension, bool swizzled)
	{
		rsx::flags32_t create_flags = 0;
		if (context == rsx::texture_upload_context::shader_read &&
			!g_cfg.video.disable_hardware_texel_remapping)
		{
			create_flags |= texture_create_flag::mutable_format;
		}
		auto* section = create_new_texture(command, range, width, height, depth, mipmaps,
			pitch, format, context, dimension, swizzled,
			rsx::component_order::default_, create_flags);
		auto* resource = section->get_raw_texture();
		ensure(resource);
		resource->set_debug_name(fmt::format("Raw texture 0x%x", range.start));
		const bool input_swizzled = context == rsx::texture_upload_context::blit_engine_src
			? false : swizzled;
		const rsx::GCM_tile_reference tile =
			(context == rsx::texture_upload_context::blit_engine_src)
				? rsx::get_current_renderer()->get_tiled_memory_region(range)
				: rsx::GCM_tile_reference{};
		mtl::upload_texture(command, *m_allocator, *resource, format, input_swizzled,
			layouts, tile ? &tile : nullptr);
		section->last_write_tag = rsx::get_shared_tag();
		return section;
	}

	void texture_cache::set_component_order(cached_texture_section& section,
		u32 format, rsx::component_order expected)
	{
		if (section.get_view_flags() == expected)
			return;
		ensure(section.get_raw_texture());
		static_cast<void>(apply_component_mapping_flags(format, expected, rsx::default_remap_vector));
		section.set_view_flags(expected);
	}

	void texture_cache::insert_texture_barrier(command_buffer& command,
		image* resource, bool strong_ordering)
	{
		(void)strong_ordering;
		auto* target = as_render_target(resource);
		ensure(target);
		target->texture_barrier(command);
	}

	bool texture_cache::render_target_format_is_compatible(image* resource, u32 format)
	{
		if (!resource)
			return false;
		const auto expected = get_sampler_format(m_device->info(), format);
		if (!expected)
			return false;
		const auto actual = describe_native_format(resource->format());
		return formats_are_bitcast_compatible(actual, expected, false) ||
			((expected.aspects & texture_aspect_depth) && (actual.aspects & texture_aspect_depth) &&
				expected.source_bytes_per_block == actual.source_bytes_per_block);
	}

	void texture_cache::prepare_for_dma_transfers(command_buffer& command)
	{
		if (!command.is_recording())
			command.begin();
	}

	void texture_cache::cleanup_after_dma_transfers(command_buffer& command)
	{
		if (command.active_encoder() != encoder_kind::none)
			command.end_encoding();
		if (command.is_recording())
			command.end();
		const submission completed = command.submit({
			.queue = queue_kind::graphics,
			.wait_for_completion = true,
		});
		ensure(completed && completed.succeeded());
		command.reset_after_completion();
		command.begin();
	}

	void texture_cache::initialize(render_device& device, memory_allocator& allocator,
		surface_cache& surfaces)
	{
		if (m_device || m_allocator || m_surface_cache)
			fmt::throw_exception("Metal texture cache is already initialized");
		m_device = &device;
		m_allocator = &allocator;
		m_surface_cache = &surfaces;
		m_cache_is_exiting = false;
	}

	void texture_cache::destroy()
	{
		clear();
		m_surface_cache = nullptr;
		m_allocator = nullptr;
		m_device = nullptr;
	}

	bool texture_cache::is_depth_texture(u32 address, u32 size)
	{
		reader_lock lock(m_cache_mutex);
		auto& block = m_storage.block_for(address);
		if (!block.get_locked_count())
			return false;
		for (auto& texture : block)
		{
			if (texture.is_dirty() || !texture.overlaps(address, rsx::section_bounds::full_range))
				continue;
			if (address + size - texture.get_section_base() <= texture.get_section_size())
				return texture.is_depth_texture();
		}
		return false;
	}

	bool texture_cache::handle_memory_pressure(rsx::problem_severity severity)
	{
		bool released = baseclass::handle_memory_pressure(severity);
		if (severity <= rsx::problem_severity::low || !m_cached_memory_size)
			return released;
		constexpr u64 one_megabyte = 0x100000;
		if (severity <= rsx::problem_severity::moderate && m_cached_memory_size < 64 * one_megabyte)
			return released;
		std::unique_lock cache_lock(m_cache_mutex, std::defer_lock);
		if (!cache_lock.try_lock())
			return released;

		{
			std::lock_guard pool_lock(m_cached_pool_lock);
			released |= !m_cached_images.empty();
			m_cached_images.clear();
			m_cached_memory_size = 0;
		}
		released |= !m_temporary_subresource_cache.empty();
		for (auto& entry : m_temporary_subresource_cache)
		{
			if (entry.second.second)
				release_temporary_subresource(entry.second.second);
		}
		m_temporary_subresource_cache.clear();
		return released;
	}

	void texture_cache::on_frame_end()
	{
		trim_sections();
		if (m_storage.m_unreleased_texture_objects >= m_max_zombie_objects)
			purge_unreleased_sections();
		{
			std::lock_guard lock(m_cached_pool_lock);
			if (m_cached_images.size() > max_cached_image_pool_size ||
				m_cached_memory_size > 256ull * 0x100000)
			{
				const usz new_size = m_cached_images.size() / 2;
				for (usz index = new_size; index < m_cached_images.size(); ++index)
					m_cached_memory_size -= image_bytes(*m_cached_images[index].data);
				m_cached_images.resize(new_size);
			}
		}
		baseclass::on_frame_end();
		reset_frame_statistics();
	}

	viewable_image* texture_cache::upload_image_simple(command_buffer& command,
		u64 format, u32 address, u32 width, u32 height, u32 pitch)
	{
		const auto description = describe_native_format(format);
		if (!description || description.bytes_per_block != 4 || description.block_width != 1 ||
			description.block_height != 1 || pitch < width * 4)
		{
			return nullptr;
		}
		const u64 data_size = static_cast<u64>(width) * height * 4;
		auto staging = std::make_unique<buffer>(*m_allocator, buffer_create_info{
			.size = data_size,
			.usage = buffer_usage_copy_source,
			.storage = storage_mode::shared,
			.cache = cpu_cache_mode::write_combined,
			.access = cpu_access::write,
			.pool = allocation_pool::swapchain,
			.label = "RPCS3 simple texture upload",
		});
		auto* destination = static_cast<u32*>(staging->map(0, data_size));
		const u8* source = vm::_ptr<const u8>(address);
		for (u32 row = 0; row < height; ++row)
		{
			const auto* input = reinterpret_cast<const be_t<u32>*>(source + static_cast<u64>(row) * pitch);
			for (u32 column = 0; column < width; ++column)
				destination[static_cast<u64>(row) * width + column] = input[column];
		}
		staging->did_modify(0, data_size);
		staging->unmap();

		image_create_info info{
			.type = texture_type::texture_2d,
			.formats = get_view_compatibility(format),
			.width = width,
			.height = height,
			.depth = 1,
			.mip_levels = 1,
			.array_layers = 1,
			.sample_count = 1,
			.usage = texture_usage_shader_read | texture_usage_copy_destination |
				texture_usage_copy_source | texture_usage_pixel_format_view,
			.aspects = texture_aspect_color,
			.storage = storage_mode::private_,
			.pool = allocation_pool::swapchain,
			.label = "RPCS3 simple uploaded texture",
		};
		auto uploaded = std::make_unique<viewable_image>(*m_allocator, info);
		const buffer_image_copy_region region{
			.buffer_offset = 0,
			.bytes_per_row = static_cast<u64>(width) * 4,
			.bytes_per_image = data_size,
			.subresource = {.aspects = texture_aspect_color},
			.extent = {width, height, 1},
		};
		upload_image(command, *staging, *uploaded, std::span{&region, 1});
		auto* result = uploaded.get();
		get_resource_manager().retire(staging, {
			.resource_class = managed_resource_class::buffer,
			.bytes = data_size,
			.label = "RPCS3 simple texture staging",
		});
		get_resource_manager().retire(uploaded, {
			.resource_class = managed_resource_class::texture,
			.bytes = image_bytes(*result),
			.label = "RPCS3 simple uploaded texture",
		});
		return result;
	}

	bool texture_cache::blit(const rsx::blit_src_info& source,
		const rsx::blit_dst_info& destination, bool interpolate,
		surface_cache& surfaces, command_buffer& command)
	{
		texture_cache_blitter helper;
		const auto result = upload_scaled_image(source, destination, interpolate,
			command, surfaces, helper);
		if (!result.succeeded)
			return false;
		if (result.real_dst_size)
			flush_if_cache_miss_likely(command, result.to_address_range());
		return true;
	}

	u32 texture_cache::get_unreleased_textures_count() const
	{
		return baseclass::get_unreleased_textures_count() + ::size32(m_cached_images) +
			::size32(m_temporary_images);
	}

	u64 texture_cache::get_temporary_memory_in_use() const
	{
		return m_cached_memory_size;
	}

	bool texture_cache::is_overallocated() const
	{
		ensure(m_device && m_allocator);
		const u64 total = m_device->info().memory.recommended_working_set_size;
		u64 quota;
		if (total >= 2048ull * 0x100000)
			quota = std::min<u64>(3072ull * 0x100000, total * 40 / 100);
		else if (total >= 1024ull * 0x100000)
			quota = std::max<u64>(204ull * 0x100000, total * 30 / 100);
		else if (total >= 768ull * 0x100000)
			quota = 192ull * 0x100000;
		else
			quota = std::min<u64>(128ull * 0x100000, total / 2);
		return get_application_pool_usage(allocation_pool::texture_cache) > quota;
	}

	memory_allocator& texture_cache::allocator() const
	{
		ensure(m_allocator);
		return *m_allocator;
	}
}
