#include "stdafx.h"
#include "MTLRenderTargets.h"
#include "MTLResolveHelper.h"

#include "Emu/RSX/rsx_methods.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

namespace mtl
{
	namespace
	{
		[[nodiscard]] u64 checked_multiply(u64 left, u64 right, std::string_view operation)
		{
			if (left && right > std::numeric_limits<u64>::max() / left)
				fmt::throw_exception("Metal render-target %s size overflows", operation);
			return left * right;
		}

		[[nodiscard]] u64 checked_add(u64 left, u64 right, std::string_view operation)
		{
			if (right > std::numeric_limits<u64>::max() - left)
				fmt::throw_exception("Metal render-target %s range overflows", operation);
			return left + right;
		}

		[[nodiscard]] subresource_range full_range(const image& resource, u8 aspects = texture_aspect_none)
		{
			const u8 selected = aspects ? aspects : resource.aspects();
			return {0, resource.mipmaps(), 0, resource.layers(),
				bool(selected & texture_aspect_color), bool(selected & texture_aspect_depth),
				bool(selected & texture_aspect_stencil)};
		}

		[[nodiscard]] image_state access_state(const render_target& surface, rsx::surface_access access)
		{
			image_state state;
			state.initialized = true;
			if (access == rsx::surface_access::shader_read)
			{
				state.stages = stage_vertex | stage_fragment;
				state.access = access_shader_read;
			}
			else if (access == rsx::surface_access::shader_write)
			{
				state.stages = surface.is_depth_surface() ? stage_fragment : stage_fragment;
				state.access = surface.is_depth_surface() ?
					access_depth_stencil_read | access_depth_stencil_write :
					access_color_read | access_color_write;
			}
			else if (access == rsx::surface_access::transfer_read || access == rsx::surface_access::memory_read)
			{
				state.stages = stage_blit;
				state.access = access_blit_read;
			}
			else if (access == rsx::surface_access::transfer_write || access == rsx::surface_access::memory_write)
			{
				state.stages = stage_blit;
				state.access = access_blit_write;
			}
			else
			{
				state.stages = stage_all_gpu;
				state.access = access_shader_read;
			}
			return state;
		}

		[[nodiscard]] u32 sample_count(rsx::surface_antialiasing antialias)
		{
			return get_format_sample_count(antialias);
		}

		[[nodiscard]] u32 morton_index_2d(u32 x, u32 y, u32 width, u32 height)
		{
			u32 result = 0;
			u32 shift = 0;
			u32 width_bits = std::bit_width(width - 1);
			u32 height_bits = std::bit_width(height - 1);
			while (x || y)
			{
				if (width_bits) { result |= (x & 1) << shift++; x >>= 1; --width_bits; }
				if (height_bits) { result |= (y & 1) << shift++; y >>= 1; --height_bits; }
			}
			return result;
		}

		[[nodiscard]] image_create_info make_surface_info(const native_format_description& format,
			u32 width, u32 height, u32 samples, bool depth, std::string label)
		{
			image_create_info info;
			info.type = samples > 1 ? texture_type::texture_2d_multisample : texture_type::texture_2d;
			info.formats = get_view_compatibility(format.pixel_format);
			info.width = width;
			info.height = height;
			info.sample_count = samples;
			info.usage = texture_usage_shader_read | texture_usage_copy_source |
				texture_usage_copy_destination | texture_usage_pixel_format_view |
				(depth ? texture_usage_depth_stencil : texture_usage_render_target | texture_usage_shader_write);
			info.aspects = format.aspects;
			if (depth)
			{
				switch (static_cast<rsx::surface_depth_format2>(format.source_format))
				{
				case rsx::surface_depth_format2::z16_uint:
					info.format_class = rsx::RSX_FORMAT_CLASS_DEPTH16_UNORM;
					break;
				case rsx::surface_depth_format2::z16_float:
					info.format_class = rsx::RSX_FORMAT_CLASS_DEPTH16_FLOAT;
					break;
				case rsx::surface_depth_format2::z24s8_uint:
					info.format_class = rsx::RSX_FORMAT_CLASS_DEPTH24_UNORM_X8_PACK32;
					break;
				case rsx::surface_depth_format2::z24s8_float:
					info.format_class = rsx::RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32;
					break;
				}
			}
			else
			{
				info.format_class = rsx::RSX_FORMAT_CLASS_COLOR;
			}
			info.storage = storage_mode::private_;
			info.pool = allocation_pool::surface_cache;
			info.label = std::move(label);
			return info;
		}

		void end_encoder(command_buffer& command)
		{
			if (command.active_encoder() != encoder_kind::none) command.end_encoding();
		}
	}

	namespace surface_cache_utils
	{
		void dispose(surface_dma_buffer* resource)
		{
			delete resource;
		}
	}

	void image_reference_sync_barrier::on_insert_texture_barrier()
	{
		++m_texture_barrier_count;
		m_allow_skip_barrier = false;
	}

	void image_reference_sync_barrier::on_insert_draw_barrier()
	{
		m_draw_barrier_count = std::max(m_draw_barrier_count + 1, m_texture_barrier_count);
	}

	void image_reference_sync_barrier::allow_skip()
	{
		m_allow_skip_barrier = true;
	}

	void image_reference_sync_barrier::reset()
	{
		m_texture_barrier_count = 0;
		m_draw_barrier_count = 0;
		m_allow_skip_barrier = false;
	}

	bool image_reference_sync_barrier::can_skip() const { return m_allow_skip_barrier; }
	bool image_reference_sync_barrier::is_enabled() const { return m_texture_barrier_count != 0; }
	bool image_reference_sync_barrier::requires_post_loop_barrier() const
	{
		return is_enabled() && m_texture_barrier_count < m_draw_barrier_count;
	}

	surface_dma_buffer::surface_dma_buffer(render_device& device, u64 requested_size, u32 address,
		std::string_view label)
	{
		const u64 page_size = static_cast<u64>(getpagesize());
		if (!requested_size || requested_size > std::numeric_limits<usz>::max() || requested_size % page_size)
			fmt::throw_exception("Metal no-copy surface DMA buffers must be page-sized");
		if (posix_memalign(&m_host_memory, page_size, requested_size) || !m_host_memory)
			fmt::throw_exception("Metal surface DMA host allocation failed");
		std::memset(m_host_memory, 0, requested_size);
		if (address)
			std::memcpy(m_host_memory, vm::g_sudo_addr + address, requested_size);
		try
		{
			create_no_copy(device, m_host_memory, requested_size,
				buffer_usage_storage | buffer_usage_copy_source | buffer_usage_copy_destination, label);
		}
		catch (...)
		{
			std::free(m_host_memory);
			m_host_memory = nullptr;
			throw;
		}
		base_address = address;
	}

	surface_dma_buffer::~surface_dma_buffer()
	{
		destroy();
		std::free(m_host_memory);
	}

	render_target::render_target(memory_allocator& allocator, const image_create_info& info,
		const native_format_description& format, bool depth_surface)
		: viewable_image(allocator, info)
		, m_allocator(&allocator)
		, m_create_info(info)
		, m_format(format)
		, m_native_components(format.components)
		, m_depth_surface(depth_surface)
	{
		if (!format || depth_surface != bool(format.aspects & texture_aspect_depth))
			fmt::throw_exception("Invalid Metal render-target format classification");
	}

	render_target::~render_target()
	{
		if (!old_contents.empty()) clear_rw_barrier();
	}

	bool render_target::is_depth_surface() const { return m_depth_surface; }

	bool render_target::matches_dimensions(u16 requested_width, u16 requested_height) const
	{
		const auto [scaled_width, scaled_height] = rsx::apply_resolution_scale<true>(
			resolution_scaling_config, requested_width, requested_height);
		return width() == scaled_width && height() == scaled_height;
	}

	void render_target::reset_surface_counters()
	{
		frame_tag = 0;
		m_cyclic_reference.reset();
	}

	image_view* render_target::get_view(const rsx::texture_channel_remap_t& remap, u8 aspect_mask)
	{
		const auto channels = apply_swizzle_remap(
			{m_native_components.red, m_native_components.green, m_native_components.blue, m_native_components.alpha}, remap);
		return viewable_image::get_view(format(), type(), channels, full_range(*this, aspects() & aspect_mask));
	}

	image_view* render_target::get_view(u8 aspect_mask)
	{
		return viewable_image::get_view(format(), type(), m_native_components,
			full_range(*this, aspects() & aspect_mask));
	}

	void render_target::set_native_component_layout(component_mapping mapping) { m_native_components = mapping; }
	component_mapping render_target::native_component_layout() const { return m_native_components; }
	const native_format_description& render_target::native_format() const { return m_format; }
	memory_allocator& render_target::allocator() const { return *m_allocator; }
	bool render_target::memory_initialized() const { return m_memory_initialized; }
	bool render_target::spilled() const { return m_spilled_memory != nullptr && !static_cast<bool>(*this); }

	render_target* as_render_target(image* resource)
	{
		return dynamic_cast<render_target*>(resource);
	}

	const render_target* as_render_target(const image* resource)
	{
		return dynamic_cast<const render_target*>(resource);
	}


	void resolve_image(command_buffer& command, viewable_image& destination, viewable_image& source)
	{
		resolve_request request;
		request.multisampled = &source;
		request.expanded = &destination;
		request.direction = resolve_direction::multisample_to_expanded;
		const auto grid = resolve_sample_grid::from_sample_count(source.samples());
		request.multisampled_region = {0, 0, 0, 0, source.width(), source.height(), source.layers()};
		request.expanded_region = {0, 0, 0, 0, source.width() * grid.x,
			source.height() * grid.y, destination.layers()};
		if (const auto* surface = dynamic_cast<const render_target*>(&source);
			surface && (source.aspects() & texture_aspect_stencil))
		{
			request.stencil_initial_value = surface->stencil_init_flags & 0xff;
			request.stencil_contents_initialized = bool(surface->stencil_init_flags & 0xff00);
		}
		get_resolve_helper().run(command, request);
	}

	void unresolve_image(command_buffer& command, viewable_image& destination, viewable_image& source)
	{
		resolve_request request;
		request.multisampled = &destination;
		request.expanded = &source;
		request.direction = resolve_direction::expanded_to_multisample;
		const auto grid = resolve_sample_grid::from_sample_count(destination.samples());
		request.multisampled_region = {0, 0, 0, 0, destination.width(), destination.height(), destination.layers()};
		request.expanded_region = {0, 0, 0, 0, destination.width() * grid.x,
			destination.height() * grid.y, source.layers()};
		if (const auto* surface = dynamic_cast<const render_target*>(&destination);
			surface && (destination.aspects() & texture_aspect_stencil))
		{
			request.stencil_initial_value = surface->stencil_init_flags & 0xff;
			request.stencil_contents_initialized = bool(surface->stencil_init_flags & 0xff00);
		}
		get_resolve_helper().run(command, request);
	}

	viewable_image* render_target::get_resolve_target_safe(command_buffer& command)
	{
		if (!resolve_surface)
		{
			image_create_info info = m_create_info;
			info.type = texture_type::texture_2d;
			info.width = width() * samples_x;
			info.height = height() * samples_y;
			info.sample_count = 1;
			info.storage = storage_mode::private_;
			info.label = debug_name() + " planar resolve";
			resolve_surface = std::make_unique<viewable_image>(*m_allocator, info);
			transition_image(command, *resolve_surface,
				{queue_kind::graphics, stage_blit, access_blit_write, 0, true});
		}
		return resolve_surface.get();
	}

	void render_target::resolve(command_buffer& command)
	{
		if (!(msaa_flags & rsx::surface_state_flags::require_resolve)) return;
		auto* destination = get_resolve_target_safe(command);
		resolve_image(command, *destination, *this);
		msaa_flags &= ~rsx::surface_state_flags::require_resolve;
	}

	void render_target::unresolve(command_buffer& command)
	{
		if (!(msaa_flags & rsx::surface_state_flags::require_unresolve)) return;
		if (!resolve_surface) fmt::throw_exception("Metal render target cannot unresolve without planar data");
		unresolve_image(command, *this, *resolve_surface);
		msaa_flags &= ~rsx::surface_state_flags::require_unresolve;
	}

	void render_target::clear_memory(command_buffer& command, image& surface)
	{
		end_encoder(command);
		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		if (surface.aspects() & texture_aspect_color)
		{
			pass.colorAttachments[0].texture = surface.native_handle();
			pass.colorAttachments[0].loadAction = MTLLoadActionClear;
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
		}
		else
		{
			pass.depthAttachment.texture = surface.native_handle();
			pass.depthAttachment.loadAction = MTLLoadActionClear;
			pass.depthAttachment.storeAction = MTLStoreActionStore;
			pass.depthAttachment.clearDepth = 1.0;
			if (surface.aspects() & texture_aspect_stencil)
			{
				pass.stencilAttachment.texture = surface.native_handle();
				pass.stencilAttachment.loadAction = MTLLoadActionClear;
				pass.stencilAttachment.storeAction = MTLStoreActionStore;
				pass.stencilAttachment.clearStencil = 255;
			}
		}
		(void)command.begin_render_encoding((__bridge void*)pass);
		command.end_encoding();
		command.retain_native_object((__bridge void*)surface.native_handle(), true);
		surface.set_state({queue_kind::graphics,
			surface.aspects() & texture_aspect_color ? stage_fragment : stage_fragment,
			surface.aspects() & texture_aspect_color ? access_color_write : access_depth_stencil_write,
			0, true});
		if (&surface == this) state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
		m_memory_initialized = true;
	}

	std::vector<buffer_image_copy_region> render_target::build_spill_transfer_descriptors(image& target) const
	{
		std::vector<buffer_image_copy_region> result;
		buffer_image_copy_region depth_or_color;
		depth_or_color.subresource.aspects = target.aspects() &
			(texture_aspect_color | texture_aspect_depth);
		depth_or_color.extent = {target.width(), target.height(), 1};
		const u64 primary_bytes = target.aspects() & texture_aspect_depth ?
			(m_format.compatibility_class == native_format_class::depth16 ? 2 : 4) :
			(m_format.bytes_per_block ? m_format.bytes_per_block : 4);
		depth_or_color.bytes_per_row = checked_multiply(target.width(), primary_bytes, "spill row");
		depth_or_color.bytes_per_image = checked_multiply(depth_or_color.bytes_per_row,
			target.height(), "spill plane");
		result.push_back(depth_or_color);
		if (target.aspects() & texture_aspect_stencil)
		{
			buffer_image_copy_region stencil;
			stencil.buffer_offset = depth_or_color.bytes_per_image;
			stencil.bytes_per_row = target.width();
			stencil.bytes_per_image = checked_multiply(target.width(), target.height(), "stencil spill plane");
			stencil.subresource.aspects = texture_aspect_stencil;
			stencil.extent = depth_or_color.extent;
			result.push_back(stencil);
		}
		return result;
	}

	bool render_target::spill(command_buffer& command,
		std::vector<std::unique_ptr<viewable_image>>& resolve_cache)
	{
		if (spilled()) return true;
		if (!*this) return false;
		viewable_image* source = this;
		if (samples() > 1)
		{
			if (msaa_flags & rsx::surface_state_flags::require_resolve) resolve(command);
			source = get_resolve_target_safe(command);
		}
		const auto regions = build_spill_transfer_descriptors(*source);
		u64 size = 0;
		for (const auto& region : regions)
			size = std::max(size, checked_add(region.buffer_offset, region.bytes_per_image, "spill"));
		buffer_create_info info;
		info.size = size;
		info.usage = buffer_usage_copy_source | buffer_usage_copy_destination | buffer_usage_storage;
		info.storage = storage_mode::shared;
		info.access = cpu_access::read_write;
		info.pool = allocation_pool::system;
		info.label = debug_name() + " spill";
		m_spilled_memory = std::make_unique<buffer>(*m_allocator, info);
		download_image(command, *source, *m_spilled_memory, regions);
		m_spilled_row_bytes = regions.front().bytes_per_row;
		viewable_image::destroy();
		if (resolve_surface) resolve_cache.emplace_back(std::move(resolve_surface));
		spill_request_tag = 0;
		is_bound = false;
		return true;
	}

	void render_target::unspill(command_buffer& command)
	{
		if (*this) return;
		if (!m_spilled_memory) fmt::throw_exception("Metal render target has neither image nor spilled storage");
		viewable_image::create(*m_allocator, m_create_info);
		if (!(state_flags & rsx::surface_state_flags::erase_bkgnd))
		{
			viewable_image* destination = samples() > 1 ? get_resolve_target_safe(command) : this;
			const auto regions = build_spill_transfer_descriptors(*destination);
			upload_image(command, *m_spilled_memory, *destination, regions);
			if (samples() > 1)
			{
				msaa_flags &= ~rsx::surface_state_flags::require_resolve;
				msaa_flags |= rsx::surface_state_flags::require_unresolve;
			}
		}
		m_spilled_memory.reset();
		m_spilled_row_bytes = 0;
	}

	void render_target::load_memory(command_buffer& command)
	{
		const u32 guest_width = surface_width * samples_x;
		const u32 guest_height = surface_height * samples_y;
		const u32 source_bpp = m_format.source_bytes_per_block;
		if (!source_bpp) fmt::throw_exception("Metal surface format has no guest element size");
		const auto* guest_memory = vm::g_sudo_addr + base_addr;
		auto guest_element = [&](u32 x, u32 y)
		{
			const u64 offset = raster_type == rsx::surface_raster_type::swizzle ?
				checked_multiply(morton_index_2d(x, y, guest_width, guest_height), source_bpp, "swizzled upload") :
				checked_add(checked_multiply(y, rsx_pitch, "linear upload"),
					checked_multiply(x, source_bpp, "linear upload"), "linear upload");
			return guest_memory + offset;
		};
		std::vector<u8> staging;
		std::vector<buffer_image_copy_region> upload_regions;
		const bool expand_legacy = m_format.conversion_flags & format_conversion_expand_legacy;
		const bool expand_depth16 = m_format.conversion_flags & format_conversion_depth16_float;
		const bool expand_depth24 = m_format.conversion_flags & format_conversion_depth24;
		const bool split_depth_stencil = (m_format.aspects &
			(texture_aspect_depth | texture_aspect_stencil)) ==
			(texture_aspect_depth | texture_aspect_stencil);
		u32 staging_row_pitch = guest_width * m_format.bytes_per_block;

		if (split_depth_stencil)
		{
			staging_row_pitch = guest_width * 4;
			const u64 depth_size = checked_multiply(staging_row_pitch, guest_height, "depth upload");
			const u64 stencil_size = checked_multiply(guest_width, guest_height, "stencil upload");
			staging.resize(depth_size + stencil_size);
			for (u32 y = 0; y < guest_height; ++y)
			{
				for (u32 x = 0; x < guest_width; ++x)
				{
					u32 packed;
					std::memcpy(&packed, guest_element(x, y), 4);
					packed = __builtin_bswap32(packed);
					const u32 depth = packed >> 8;
					u32 native_depth;
					if (get_surface_depth_format() == rsx::surface_depth_format2::z24s8_float)
						native_depth = depth << 7;
					else if (expand_depth24)
					{
						const f32 value = static_cast<f32>(depth) / 16777215.f;
						std::memcpy(&native_depth, &value, 4);
					}
					else native_depth = depth;
					std::memcpy(staging.data() + y * staging_row_pitch + x * 4, &native_depth, 4);
					staging[depth_size + y * guest_width + x] = static_cast<u8>(packed);
				}
			}
			buffer_image_copy_region depth_region;
			depth_region.bytes_per_row = staging_row_pitch;
			depth_region.bytes_per_image = depth_size;
			depth_region.subresource.aspects = texture_aspect_depth;
			depth_region.extent = {guest_width, guest_height, 1};
			upload_regions.push_back(depth_region);
			buffer_image_copy_region stencil_region;
			stencil_region.buffer_offset = depth_size;
			stencil_region.bytes_per_row = guest_width;
			stencil_region.bytes_per_image = stencil_size;
			stencil_region.subresource.aspects = texture_aspect_stencil;
			stencil_region.extent = depth_region.extent;
			upload_regions.push_back(stencil_region);
		}
		else if (expand_depth16)
		{
			staging_row_pitch = guest_width * 4;
			staging.resize(checked_multiply(staging_row_pitch, guest_height, "depth16 upload"));
			for (u32 y = 0; y < guest_height; ++y)
			{
				for (u32 x = 0; x < guest_width; ++x)
				{
					u16 packed;
					std::memcpy(&packed, guest_element(x, y), 2);
					packed = __builtin_bswap16(packed);
					const u32 expanded = (static_cast<u32>(packed) << 11) + (120u << 23);
					std::memcpy(staging.data() + y * staging_row_pitch + x * 4, &expanded, 4);
				}
			}
		}
		else if (expand_legacy)
		{
			staging_row_pitch = guest_width * 4;
			staging.resize(checked_multiply(staging_row_pitch, guest_height, "legacy color upload"));
			for (u32 y = 0; y < guest_height; ++y)
			{
				for (u32 x = 0; x < guest_width; ++x)
				{
					u16 packed;
					std::memcpy(&packed, guest_element(x, y), 2);
					packed = __builtin_bswap16(packed);
					u8 red, green, blue;
					if (get_surface_color_format() == rsx::surface_color_format::r5g6b5)
					{
						red = static_cast<u8>(((packed >> 11) & 31) * 255 / 31);
						green = static_cast<u8>(((packed >> 5) & 63) * 255 / 63);
						blue = static_cast<u8>((packed & 31) * 255 / 31);
					}
					else
					{
						red = static_cast<u8>(((packed >> 10) & 31) * 255 / 31);
						green = static_cast<u8>(((packed >> 5) & 31) * 255 / 31);
						blue = static_cast<u8>((packed & 31) * 255 / 31);
					}
					u8* output = staging.data() + y * staging_row_pitch + x * 4;
					output[0] = blue; output[1] = green; output[2] = red;
					output[3] = m_format.conversion_flags & format_conversion_force_alpha_one ? 255 : 0;
				}
			}
		}
		else
		{
			staging_row_pitch = guest_width * source_bpp;
			staging.resize(checked_multiply(staging_row_pitch, guest_height, "surface upload"));
			for (u32 y = 0; y < guest_height; ++y)
			{
				if (raster_type == rsx::surface_raster_type::linear)
					std::memcpy(staging.data() + y * staging_row_pitch, guest_memory + y * rsx_pitch,
						staging_row_pitch);
				else
					for (u32 x = 0; x < guest_width; ++x)
						std::memcpy(staging.data() + y * staging_row_pitch + x * source_bpp,
							guest_element(x, y), source_bpp);
			}
			const u32 swap_size = m_format.conversion_flags & format_conversion_byte_swap_32 ? 4 :
				m_format.conversion_flags & format_conversion_byte_swap_16 ? 2 : 1;
			if (swap_size == 4)
				for (usz offset = 0; offset + 4 <= staging.size(); offset += 4)
				{
					u32 value; std::memcpy(&value, staging.data() + offset, 4);
					value = __builtin_bswap32(value); std::memcpy(staging.data() + offset, &value, 4);
				}
			else if (swap_size == 2)
				for (usz offset = 0; offset + 2 <= staging.size(); offset += 2)
				{
					u16 value; std::memcpy(&value, staging.data() + offset, 2);
					value = __builtin_bswap16(value); std::memcpy(staging.data() + offset, &value, 2);
				}
		}
		if (upload_regions.empty())
		{
			buffer_image_copy_region region;
			region.bytes_per_row = staging_row_pitch;
			region.bytes_per_image = staging.size();
			region.subresource.aspects = m_format.aspects;
			region.extent = {guest_width, guest_height, 1};
			upload_regions.push_back(region);
		}

		buffer_create_info buffer_info;
		buffer_info.size = staging.size();
		buffer_info.usage = buffer_usage_copy_source | buffer_usage_storage;
		buffer_info.storage = storage_mode::shared;
		buffer_info.cache = cpu_cache_mode::write_combined;
		buffer_info.access = cpu_access::write;
		buffer_info.pool = allocation_pool::system;
		buffer_info.label = debug_name() + " guest upload";
		buffer upload(*m_allocator, buffer_info);
		auto* destination_bytes = static_cast<u8*>(upload.map(0, staging.size()));
		std::memcpy(destination_bytes, staging.data(), staging.size());
		upload.unmap();
		upload.did_modify(0, staging.size());

		viewable_image* final_destination = samples() > 1 ? get_resolve_target_safe(command) : this;
		std::unique_ptr<viewable_image> intermediate;
		image* upload_destination = final_destination;
		if (final_destination->width() != guest_width || final_destination->height() != guest_height)
		{
			image_create_info info = m_create_info;
			info.type = texture_type::texture_2d;
			info.width = guest_width;
			info.height = guest_height;
			info.sample_count = 1;
			info.label = debug_name() + " unscaled upload";
			intermediate = std::make_unique<viewable_image>(*m_allocator, info);
			upload_destination = intermediate.get();
		}
		upload_image(command, upload, *upload_destination, upload_regions);
		if (intermediate)
		{
			image_scale_region scale;
			scale.source.aspects = intermediate->aspects();
			scale.destination.aspects = final_destination->aspects();
			scale.source_box = {0, 0, 0, static_cast<s32>(guest_width), static_cast<s32>(guest_height), 1};
			scale.destination_box = {0, 0, 0, static_cast<s32>(final_destination->width()),
				static_cast<s32>(final_destination->height()), 1};
			scale_image(command, *intermediate, *final_destination, scale,
				is_depth_surface() ? image_filter::nearest : image_filter::linear);
		}
		if (samples() > 1) msaa_flags = rsx::surface_state_flags::require_unresolve;
		state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
		m_memory_initialized = true;
	}

	void render_target::initialize_memory(command_buffer& command, rsx::surface_access access)
	{
		const bool read_guest = is_depth_surface() ? bool(g_cfg.video.read_depth_buffer) :
			bool(g_cfg.video.read_color_buffers);
		if (read_guest)
		{
			load_memory(command);
		}
		else
		{
			clear_memory(command, *this);
			if (samples() > 1 && access.is_transfer_or_read()) clear_memory(command, *get_resolve_target_safe(command));
			msaa_flags = rsx::surface_state_flags::ready;
		}
	}

	viewable_image* render_target::get_surface(rsx::surface_access access_type)
	{
		last_rw_access_tag = rsx::get_shared_tag();
		if (samples() == 1 || !access_type.is_transfer()) return this;
		if (!resolve_surface || (msaa_flags & rsx::surface_state_flags::require_resolve))
			fmt::throw_exception("Metal multisample transfer requested without a completed resolve");
		return resolve_surface.get();
	}

	void render_target::texture_barrier(command_buffer& command)
	{
		const bool depth_read_only = is_depth_surface() && !rsx::method_registers.depth_write_enabled();
		const image_state current = state();
		const u64 attachment_access = is_depth_surface() ?
			access_depth_stencil_read | (depth_read_only ? 0 : access_depth_stencil_write) :
			access_color_read | access_color_write;
		const image_state feedback{queue_kind::graphics, stage_fragment,
			attachment_access | access_shader_read, current.submission, true};
		if (m_cyclic_reference.can_skip() && current.access == feedback.access && depth_read_only) return;
		transition_image(command, *this, feedback, full_range(*this), true);
		m_cyclic_reference.on_insert_texture_barrier();
		if (depth_read_only) m_cyclic_reference.allow_skip();
	}

	void render_target::post_texture_barrier(command_buffer& command)
	{
		const bool depth_read_only = is_depth_surface() && !rsx::method_registers.depth_write_enabled();
		if (m_cyclic_reference.can_skip() && depth_read_only)
		{
			m_cyclic_reference.reset();
			return;
		}
		image_state next{queue_kind::graphics, stage_fragment,
			is_depth_surface() ? access_depth_stencil_read | access_depth_stencil_write :
				access_color_read | access_color_write,
			state().submission, true};
		transition_image(command, *this, next, full_range(*this), true);
		m_cyclic_reference.reset();
	}

	void render_target::memory_barrier(command_buffer& command, rsx::surface_access access)
	{
		if (access == rsx::surface_access::gpu_reference)
		{
			if (!*this) unspill(command);
			spill_request_tag = 0;
			return;
		}
		if (!*this) unspill(command);
		if (access == rsx::surface_access::shader_write && m_cyclic_reference.is_enabled())
		{
			m_cyclic_reference.on_insert_draw_barrier();
			if (m_cyclic_reference.requires_post_loop_barrier()) post_texture_barrier(command);
		}

		if (old_contents.empty())
		{
			if (state_flags & rsx::surface_state_flags::erase_bkgnd)
			{
				initialize_memory(command, access);
				on_write(rsx::get_shared_tag(), static_cast<rsx::surface_state_flags>(msaa_flags));
			}
			if ((msaa_flags & rsx::surface_state_flags::require_resolve) && access.is_transfer()) resolve(command);
			else if ((msaa_flags & rsx::surface_state_flags::require_unresolve) &&
				access == rsx::surface_access::shader_write) unresolve(command);
			transition_image(command, *this, access_state(*this, access));
			return;
		}

		viewable_image* destination = samples() > 1 ? get_resolve_target_safe(command) : this;
		const u32 first = prepare_rw_barrier_for_transfer(this);
		const bool accept_all = last_use_tag && test();
		u64 newest_tag = 0;
		bool full_overwrite = true;
		for (u32 index = first; index < old_contents.size(); ++index)
		{
			auto& section = old_contents[index];
			auto* source_surface = static_cast<render_target*>(section.source);
			source_surface->memory_barrier(command, rsx::surface_access::transfer_read);
			if (!accept_all && !source_surface->test()) continue;
			section.init_transfer(this);
			auto source_area = section.src_rect();
			auto destination_area = section.dst_rect();
			if (g_cfg.video.antialiasing_level != msaa_level::none)
			{
				source_surface->transform_pixels_to_samples(source_area);
				transform_pixels_to_samples(destination_area);
			}
			const bool overwrites_surface = destination_area.x1 == 0 && destination_area.y1 == 0 &&
				static_cast<u32>(destination_area.x2) == destination->width() &&
				static_cast<u32>(destination_area.y2) == destination->height();
			if (overwrites_surface)
			{
				state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
				msaa_flags = rsx::surface_state_flags::ready;
				stencil_init_flags = source_surface->stencil_init_flags;
			}
			else if (state_flags & rsx::surface_state_flags::erase_bkgnd)
			{
				initialize_memory(command, rsx::surface_access::memory_write);
				full_overwrite = false;
			}
			if (msaa_flags & rsx::surface_state_flags::require_resolve) resolve(command);
			if (source_surface->samples() > 1 &&
				(source_surface->msaa_flags & rsx::surface_state_flags::require_resolve))
				source_surface->resolve(command);
			viewable_image* source = source_surface->get_surface(rsx::surface_access::transfer_read);
			image_scale_region region;
			region.source.aspects = source->aspects();
			region.destination.aspects = destination->aspects();
			region.source_box = {source_area.x1, source_area.y1, 0, source_area.x2, source_area.y2, 1};
			region.destination_box = {destination_area.x1, destination_area.y1, 0,
				destination_area.x2, destination_area.y2, 1};
			image_conversion conversion;
			if (!formats_are_bitcast_compatible(*source, *destination))
			{
				conversion.kind = source->aspects() & texture_aspect_depth ?
					image_conversion_kind::depth_to_color :
					destination->aspects() & texture_aspect_depth ?
						image_conversion_kind::color_to_depth : image_conversion_kind::color_to_color;
			}
			scale_image(command, *source, *destination, region, image_filter::nearest, conversion);
			newest_tag = std::max(newest_tag, source_surface->last_use_tag);
			full_overwrite &= overwrites_surface;
		}
		if (!newest_tag)
		{
			clear_rw_barrier();
			state_flags |= rsx::surface_state_flags::erase_bkgnd;
			initialize_memory(command, access);
		}
		on_write_copy(newest_tag, full_overwrite);
		if (access == rsx::surface_access::shader_write && samples() > 1)
		{
			msaa_flags |= rsx::surface_state_flags::require_unresolve;
			unresolve(command);
		}
		transition_image(command, *this, access_state(*this, access));
	}

	void render_target::read_barrier(command_buffer& command)
	{
		memory_barrier(command, rsx::surface_access::shader_read);
	}

	void render_target::write_barrier(command_buffer& command)
	{
		memory_barrier(command, rsx::surface_access::shader_write);
	}

	std::unique_ptr<render_target> surface_cache_traits::create_new_surface(u32 address,
		rsx::surface_color_format color_format, usz requested_width, usz requested_height, usz pitch,
		rsx::surface_antialiasing antialias, const rsx::surface_scaling_config_t& scaling,
		memory_allocator& allocator, render_device& device, command_buffer& command)
	{
		if (&allocator.device() != &device || &command.allocator().owner() != &device)
			fmt::throw_exception("Metal color surface resources belong to different devices");
		const auto format = get_color_surface_format(device.info(), color_format);
		const u32 samples = sample_count(antialias);
		const auto [width, height] = rsx::apply_resolution_scale<true>(scaling,
			static_cast<u16>(requested_width), static_cast<u16>(requested_height));
		auto info = make_surface_info(format, width, height, samples, false,
			fmt::format("RPCS3 color surface @0x%x", address));
		auto result = std::make_unique<render_target>(allocator, info, format, false);
		result->set_format(color_format);
		result->set_aa_mode(antialias);
		result->set_resolution_scaling_config(scaling);
		result->sample_layout = samples > 1 ? rsx::surface_sample_layout::ps3 : rsx::surface_sample_layout::null;
		result->memory_usage_flags = rsx::surface_usage_flags::attachment;
		result->state_flags = rsx::surface_state_flags::erase_bkgnd;
		result->rsx_pitch = static_cast<u32>(pitch);
		const u32 bytes_per_pixel = format.source_bytes_per_block ? format.source_bytes_per_block : format.bytes_per_block;
		result->native_pitch = static_cast<u32>(requested_width) * bytes_per_pixel * result->samples_x;
		result->surface_width = static_cast<u16>(requested_width);
		result->surface_height = static_cast<u16>(requested_height);
		result->queue_tag(address);
		result->add_ref();
		transition_image(command, *result,
			{queue_kind::graphics, stage_fragment, access_color_write, 0, true});
		return result;
	}

	std::unique_ptr<render_target> surface_cache_traits::create_new_surface(u32 address,
		rsx::surface_depth_format2 depth_format, usz requested_width, usz requested_height, usz pitch,
		rsx::surface_antialiasing antialias, const rsx::surface_scaling_config_t& scaling,
		memory_allocator& allocator, render_device& device, command_buffer& command)
	{
		if (&allocator.device() != &device || &command.allocator().owner() != &device)
			fmt::throw_exception("Metal depth surface resources belong to different devices");
		const auto format = get_depth_surface_format(device.info(), depth_format);
		const u32 samples = sample_count(antialias);
		const auto [width, height] = rsx::apply_resolution_scale<true>(scaling,
			static_cast<u16>(requested_width), static_cast<u16>(requested_height));
		auto info = make_surface_info(format, width, height, samples, true,
			fmt::format("RPCS3 depth surface @0x%x", address));
		auto result = std::make_unique<render_target>(allocator, info, format, true);
		result->set_format(depth_format);
		result->set_aa_mode(antialias);
		result->set_resolution_scaling_config(scaling);
		result->sample_layout = samples > 1 ? rsx::surface_sample_layout::ps3 : rsx::surface_sample_layout::null;
		result->memory_usage_flags = rsx::surface_usage_flags::attachment;
		result->state_flags = rsx::surface_state_flags::erase_bkgnd;
		result->set_native_component_layout({component_swizzle::red, component_swizzle::red,
			component_swizzle::red, component_swizzle::red});
		result->rsx_pitch = static_cast<u32>(pitch);
		const u32 bytes_per_pixel = format.source_bytes_per_block ? format.source_bytes_per_block : format.bytes_per_block;
		result->native_pitch = static_cast<u32>(requested_width) * bytes_per_pixel * result->samples_x;
		result->surface_width = static_cast<u16>(requested_width);
		result->surface_height = static_cast<u16>(requested_height);
		result->queue_tag(address);
		result->add_ref();
		transition_image(command, *result,
			{queue_kind::graphics, stage_fragment, access_depth_stencil_write, 0, true});
		return result;
	}

	void surface_cache_traits::clone_surface(command_buffer& command,
		std::unique_ptr<render_target>& destination, render_target* source, u32 address,
		barrier_descriptor_t& previous, const rsx::surface_scaling_config_t& scaling)
	{
		if (!source) fmt::throw_exception("Metal surface clone requires a source");
		if (!destination)
		{
			const auto [width, height] = rsx::apply_resolution_scale<true>(scaling,
				previous.width, previous.height, source->surface_width, source->surface_height);
			image_create_info info = source->info();
			info.width = width;
			info.height = height;
			info.label = source->debug_name() + " inherited";
			destination = std::make_unique<render_target>(source->allocator(), info,
				source->native_format(), source->is_depth_surface());
			destination->set_spp(source->get_spp());
			destination->format_info = source->format_info;
			destination->sample_layout = source->sample_layout;
			destination->resolution_scaling_config = scaling;
			destination->memory_usage_flags = rsx::surface_usage_flags::storage;
			destination->state_flags = rsx::surface_state_flags::erase_bkgnd;
			destination->stencil_init_flags = source->stencil_init_flags;
			destination->set_native_component_layout(source->native_component_layout());
			destination->native_pitch = previous.width * source->get_bpp() * source->samples_x;
			destination->rsx_pitch = source->rsx_pitch;
			destination->surface_width = previous.width;
			destination->surface_height = previous.height;
			destination->queue_tag(address);
			destination->add_ref();
			transition_image(command, *destination,
				{queue_kind::graphics, stage_blit, access_blit_write, 0, true});
		}
		if (!destination->old_contents.empty()) destination->clear_rw_barrier();
		previous.target = destination.get();
		destination->set_old_contents_region(previous, false);
	}

	std::unique_ptr<render_target> surface_cache_traits::convert_pitch(command_buffer& command,
		std::unique_ptr<render_target>& source, usz output_pitch)
	{
		if (!source || !output_pitch || output_pitch > std::numeric_limits<u32>::max())
			fmt::throw_exception("Invalid Metal surface pitch conversion");
		const u32 bytes_per_sample_row = source->get_bpp() * source->samples_x;
		const u16 output_width = static_cast<u16>(output_pitch / bytes_per_sample_row);
		if (!output_width) fmt::throw_exception("Metal surface pitch cannot represent one pixel");
		image_create_info info = source->info();
		const auto [scaled_width, scaled_height] = rsx::apply_resolution_scale<true>(
			source->resolution_scaling_config, output_width, source->surface_height);
		info.width = scaled_width;
		info.height = scaled_height;
		info.label = source->debug_name() + " pitch conversion";
		auto destination = std::make_unique<render_target>(source->allocator(), info,
			source->native_format(), source->is_depth_surface());
		destination->format_info = source->format_info;
		destination->set_spp(source->get_spp());
		destination->sample_layout = source->sample_layout;
		destination->resolution_scaling_config = source->resolution_scaling_config;
		destination->rsx_pitch = static_cast<u32>(output_pitch);
		destination->native_pitch = output_width * source->get_bpp() * source->samples_x;
		destination->surface_width = output_width;
		destination->surface_height = source->surface_height;
		destination->queue_tag(source->base_addr);
		destination->state_flags = rsx::surface_state_flags::erase_bkgnd;
		destination->add_ref();
		rsx::deferred_clipped_region<render_target*> inherited;
		inherited.source = source.get();
		inherited.target = destination.get();
		inherited.width = std::min(source->surface_width, destination->surface_width);
		inherited.height = source->surface_height;
		destination->set_old_contents_region(inherited, false);
		destination->memory_barrier(command, rsx::surface_access::transfer_read);
		return destination;
	}

	bool surface_cache_traits::is_compatible_surface(const render_target* surface,
		const render_target* reference, u16 width, u16 height, u8 samples)
	{
		return surface && reference && surface->format() == reference->format() &&
			surface->get_spp() == samples && surface->surface_width == width && surface->surface_height == height;
	}

	void surface_cache_traits::prepare_surface_for_drawing(command_buffer& command, render_target* surface)
	{
		surface->memory_barrier(command, rsx::surface_access::gpu_reference);
		transition_image(command, *surface, access_state(*surface, rsx::surface_access::shader_write));
		surface->reset_surface_counters();
		surface->memory_usage_flags |= rsx::surface_usage_flags::attachment;
		surface->is_bound = true;
	}

	void surface_cache_traits::prepare_surface_for_sampling(command_buffer& command, render_target* surface)
	{
		surface->memory_barrier(command, rsx::surface_access::shader_read);
		surface->is_bound = false;
	}

	bool surface_cache_traits::surface_is_pitch_compatible(
		const std::unique_ptr<render_target>& surface, usz pitch)
	{
		return surface && surface->rsx_pitch == pitch;
	}

	void surface_cache_traits::invalidate_surface_contents(command_buffer& command, render_target* surface,
		rsx::surface_color_format format_value, u32 address, usz pitch)
	{
		(void)command;
		const auto format = get_color_surface_format(surface->allocator().device().info(), format_value);
		if (format.pixel_format != surface->format())
			fmt::throw_exception("Metal cannot invalidate a color surface with an incompatible native format");
		surface->set_format(format_value);
		surface->set_native_component_layout(format.components);
		surface->set_debug_name(fmt::format("RPCS3 color surface @0x%x", address));
		surface->rsx_pitch = static_cast<u32>(pitch);
		surface->queue_tag(address);
		surface->last_use_tag = 0;
		surface->stencil_init_flags = 0;
		surface->memory_usage_flags = rsx::surface_usage_flags::unknown;
		surface->raster_type = rsx::surface_raster_type::linear;
	}

	void surface_cache_traits::invalidate_surface_contents(command_buffer& command, render_target* surface,
		rsx::surface_depth_format2 format_value, u32 address, usz pitch)
	{
		(void)command;
		const auto format = get_depth_surface_format(surface->allocator().device().info(), format_value);
		if (format.pixel_format != surface->format())
			fmt::throw_exception("Metal cannot invalidate a depth surface with an incompatible native format");
		surface->set_format(format_value);
		surface->set_debug_name(fmt::format("RPCS3 depth surface @0x%x", address));
		surface->rsx_pitch = static_cast<u32>(pitch);
		surface->queue_tag(address);
		surface->last_use_tag = 0;
		surface->stencil_init_flags = 0;
		surface->memory_usage_flags = rsx::surface_usage_flags::unknown;
		surface->raster_type = rsx::surface_raster_type::linear;
	}

	void surface_cache_traits::notify_surface_invalidated(const std::unique_ptr<render_target>& surface)
	{
		surface->frame_tag = std::max<u64>(rsx::get_shared_tag(), 1);
		if (!surface->old_contents.empty()) surface->clear_rw_barrier();
		surface->release();
	}

	void surface_cache_traits::notify_surface_persist(const std::unique_ptr<render_target>& surface)
	{
		if (surface) surface->spill_request_tag = 0;
	}

	void surface_cache_traits::notify_surface_reused(const std::unique_ptr<render_target>& surface)
	{
		surface->state_flags |= rsx::surface_state_flags::erase_bkgnd;
		surface->add_ref();
	}

	bool surface_cache_traits::surface_matches_properties(const std::unique_ptr<render_target>& surface,
		rsx::surface_color_format format_value, usz width, usz height,
		rsx::surface_antialiasing antialias, const rsx::surface_scaling_config_t& scaling,
		bool check_references)
	{
		if (!surface || (check_references && surface->has_refs())) return false;
		const auto format = get_color_surface_format(surface->allocator().device().info(), format_value);
		return surface->format() == format.pixel_format && surface->get_spp() == sample_count(antialias) &&
			surface->matches_dimensions(static_cast<u16>(width), static_cast<u16>(height)) &&
			surface->resolution_scaling_config == scaling;
	}

	bool surface_cache_traits::surface_matches_properties(const std::unique_ptr<render_target>& surface,
		rsx::surface_depth_format2 format_value, usz width, usz height,
		rsx::surface_antialiasing antialias, const rsx::surface_scaling_config_t& scaling,
		bool check_references)
	{
		if (!surface || (check_references && surface->has_refs())) return false;
		const auto format = get_depth_surface_format(surface->allocator().device().info(), format_value);
		return surface->format() == format.pixel_format && surface->get_spp() == sample_count(antialias) &&
			surface->matches_dimensions(static_cast<u16>(width), static_cast<u16>(height)) &&
			surface->resolution_scaling_config == scaling;
	}

	void surface_cache_traits::spill_buffer(std::unique_ptr<surface_dma_buffer>& resource)
	{
		if (!resource || !*resource) fmt::throw_exception("Cannot spill an empty Metal surface DMA buffer");
		resource->did_modify(0, resource->size());
	}

	void surface_cache_traits::unspill_buffer(std::unique_ptr<surface_dma_buffer>& resource)
	{
		if (!resource || !*resource) fmt::throw_exception("Cannot restore an empty Metal surface DMA buffer");
	}

	void surface_cache_traits::write_render_target_to_memory(command_buffer& command,
		surface_dma_buffer* destination, render_target* surface, u64 destination_offset,
		u64 source_offset, u64 maximum_copy_length)
	{
		if (!destination || !surface || !maximum_copy_length ||
			!destination->in_range(destination_offset, maximum_copy_length))
			fmt::throw_exception("Invalid Metal render-target memory synchronization range");
		surface->memory_barrier(command, rsx::surface_access::transfer_read);
		viewable_image* source = surface->get_surface(rsx::surface_access::transfer_read);
		std::unique_ptr<viewable_image> unscaled;
		const u32 guest_width = surface->surface_width * surface->samples_x;
		const u32 guest_height = surface->surface_height * surface->samples_y;
		if (source->width() != guest_width || source->height() != guest_height)
		{
			image_create_info info = source->info();
			info.type = texture_type::texture_2d;
			info.width = guest_width;
			info.height = guest_height;
			info.sample_count = 1;
			info.label = surface->debug_name() + " readback scale";
			unscaled = std::make_unique<viewable_image>(surface->allocator(), info);
			image_scale_region scale;
			scale.source.aspects = source->aspects();
			scale.destination.aspects = unscaled->aspects();
			scale.source_box = {0, 0, 0, static_cast<s32>(source->width()), static_cast<s32>(source->height()), 1};
			scale.destination_box = {0, 0, 0, static_cast<s32>(guest_width), static_cast<s32>(guest_height), 1};
			scale_image(command, *source, *unscaled, scale,
				surface->is_depth_surface() ? image_filter::nearest : image_filter::linear);
			source = unscaled.get();
		}

		const u64 surface_length = surface->get_memory_range().length();
		if (source_offset >= surface_length)
			fmt::throw_exception("Metal render-target readback starts beyond the guest surface");
		u64 remaining = std::min(maximum_copy_length, surface_length - source_offset);
		u64 consumed = 0;
		const u32 bytes_per_pixel = surface->get_bpp();
		std::vector<buffer_image_copy_region> regions;
		while (remaining)
		{
			const u64 absolute = source_offset + consumed;
			const u32 row = static_cast<u32>(absolute / surface->rsx_pitch);
			const u32 row_offset = static_cast<u32>(absolute % surface->rsx_pitch);
			const u32 segment = static_cast<u32>(std::min<u64>(remaining, surface->rsx_pitch - row_offset));
			if (row_offset < surface->native_pitch)
			{
				const u32 image_bytes = std::min(segment, surface->native_pitch - row_offset);
				const u32 aligned_bytes = image_bytes - image_bytes % bytes_per_pixel;
				if (aligned_bytes)
				{
					buffer_image_copy_region region;
					region.buffer_offset = destination_offset + consumed;
					region.bytes_per_row = aligned_bytes;
					region.bytes_per_image = aligned_bytes;
					region.subresource.aspects = source->aspects();
					region.origin = {row_offset / bytes_per_pixel, row, 0};
					region.extent = {aligned_bytes / bytes_per_pixel, 1, 1};
					regions.push_back(region);
				}
			}
			consumed += segment;
			remaining -= segment;
		}
		if (!regions.empty()) download_image(command, *source, *destination, regions);
	}

	u64 surface_cache::get_surface_cache_memory_quota(u64 total_device_memory) const
	{
		const u64 megabytes = total_device_memory / 0x100000;
		u64 quota = 0;
		if (megabytes >= 2048) quota = std::min<u64>(6144, megabytes * 40 / 100);
		else if (megabytes >= 1024) quota = std::max<u64>(512, megabytes * 30 / 100);
		else if (megabytes >= 768) quota = 256;
		else quota = std::min<u64>(128, megabytes / 2);
		return quota * 0x100000;
	}

	void surface_cache::destroy()
	{
		invalidate_all();
		for (auto& surface : invalidated_resources)
			if (surface && !surface->old_contents.empty()) surface->clear_rw_barrier();
		invalidated_resources.clear();
		orphaned_surfaces.clear();
		superseded_surfaces.clear();
	}

	bool surface_cache::can_collapse_surface(const std::unique_ptr<render_target>& surface,
		rsx::problem_severity severity)
	{
		if (!surface) return true;
		if (severity < rsx::problem_severity::fatal) return true;
		std::function<bool(const render_target*)> can_collapse = [&](const render_target* target)
		{
			if (target->samples() > 1 && !target->resolve_surface) return false;
			for (const auto& region : target->old_contents)
				if (!can_collapse(static_cast<const render_target*>(region.source))) return false;
			return true;
		};
		return can_collapse(surface.get());
	}

	bool surface_cache::handle_memory_pressure(command_buffer& command, rsx::problem_severity severity)
	{
		bool released = rsx::surface_store<surface_cache_traits>::handle_memory_pressure(command, severity);
		if (severity < rsx::problem_severity::fatal) return released;
		std::vector<std::unique_ptr<viewable_image>> resolve_cache;
		auto spill_list = [&](auto& list, const rsx::address_range32& range)
		{
			for (auto iterator = list.begin_range(range); iterator != list.end(); ++iterator)
			{
				auto& surface = iterator->second;
				if (surface->is_bound) continue;
				if (surface->spill_request_tag && surface->spill_request_tag >= surface->last_rw_access_tag)
					released |= surface->spill(command, resolve_cache);
				else if (surface->resolve_surface)
				{
					resolve_cache.emplace_back(std::move(surface->resolve_surface));
					surface->msaa_flags |= rsx::surface_state_flags::require_resolve;
					released = true;
				}
			}
		};
		spill_list(m_render_targets_storage, m_render_targets_memory_range);
		spill_list(m_depth_stencil_storage, m_depth_stencil_memory_range);
		for (auto& surface : invalidated_resources)
		{
			if (!surface || surface->is_bound || !surface->has_refs()) continue;
			if (*surface) released |= surface->spill(command, resolve_cache);
		}
		return released;
	}

	bool surface_cache::spill_unused_memory(command_buffer& command,
		std::vector<std::unique_ptr<viewable_image>>& resolve_cache)
	{
		const u64 current = get_application_pool_usage(allocation_pool::surface_cache);
		const auto& device = command.allocator().owner();
		const u64 total = device.info().memory.recommended_working_set_size ?
			device.info().memory.recommended_working_set_size : device.info().limits.recommended_working_set_size;
		const u64 target = get_surface_cache_memory_quota(total);
		if (current <= target) return false;
		std::vector<render_target*> candidates;
		auto collect = [&](auto& list, const rsx::address_range32& range)
		{
			for (auto iterator = list.begin_range(range); iterator != list.end(); ++iterator)
				if (!iterator->second->is_bound && *iterator->second) candidates.push_back(iterator->second.get());
		};
		collect(m_render_targets_storage, m_render_targets_memory_range);
		collect(m_depth_stencil_storage, m_depth_stencil_memory_range);
		std::sort(candidates.begin(), candidates.end(), [](const render_target* left, const render_target* right)
		{
			return left->last_rw_access_tag < right->last_rw_access_tag;
		});
		u64 released_bytes = 0;
		const u64 requested = current - target;
		const u64 request_tag = rsx::get_shared_tag();
		for (render_target* surface : candidates)
		{
			const u64 allocation_size = surface->allocation().size();
			surface->spill_request_tag = request_tag;
			if (surface->spill(command, resolve_cache)) released_bytes += allocation_size;
			if (released_bytes >= requested) break;
		}
		return released_bytes != 0;
	}

	bool surface_cache::is_overallocated(const render_device& device) const
	{
		const u64 usage = get_application_pool_usage(allocation_pool::surface_cache);
		const u64 total = device.info().memory.recommended_working_set_size ?
			device.info().memory.recommended_working_set_size : device.info().limits.recommended_working_set_size;
		return usage > get_surface_cache_memory_quota(total);
	}

	void surface_cache::trim(command_buffer& command, rsx::problem_severity memory_pressure)
	{
		run_cleanup_internal(command, rsx::problem_severity::moderate, 300,
			[](command_buffer& active_command)
			{
				if (!active_command.is_recording()) active_command.begin();
			});
		const s32 threshold = memory_pressure == rsx::problem_severity::low ? 2 :
			memory_pressure == rsx::problem_severity::moderate ? 1 : -1;
		for (auto& surface : invalidated_resources)
		{
			if (!surface || surface->has_refs()) continue;
			if (!surface->old_contents.empty()) surface->clear_rw_barrier();
			if (threshold < 0 || surface->unused_check_count() >= threshold) surface.reset();
		}
		invalidated_resources.remove_if([](const auto& surface) { return !surface; });
	}
}
