#include "stdafx.h"
#include "MTLRenderTargetCache.h"

#include "MTLCommandQueue.h"
#include "MTLDevice.h"
#include "MTLTexture.h"

#include "Emu/RSX/Common/surface_store.h"
#include "Emu/RSX/Common/TextureUtils.h"

#import <Metal/Metal.h>

#include <unordered_map>

namespace
{
	struct color_surface_key
	{
		u32 address = 0;
		u32 pitch = 0;
		u32 width = 0;
		u32 height = 0;
		u32 surface_index = 0;
		rsx::surface_color_format format = rsx::surface_color_format::a8r8g8b8;
		rsx::surface_antialiasing aa_mode = rsx::surface_antialiasing::center_1_sample;
		rsx::surface_raster_type raster_type = rsx::surface_raster_type::linear;

		bool operator==(const color_surface_key& other) const
		{
			return address == other.address &&
				pitch == other.pitch &&
				width == other.width &&
				height == other.height &&
				surface_index == other.surface_index &&
				format == other.format &&
				aa_mode == other.aa_mode &&
				raster_type == other.raster_type;
		}
	};

	struct depth_stencil_surface_key
	{
		u32 address = 0;
		u32 pitch = 0;
		u32 width = 0;
		u32 height = 0;
		rsx::surface_depth_format2 format = rsx::surface_depth_format2::z16_uint;
		rsx::surface_antialiasing aa_mode = rsx::surface_antialiasing::center_1_sample;
		rsx::surface_raster_type raster_type = rsx::surface_raster_type::linear;

		bool operator==(const depth_stencil_surface_key& other) const
		{
			return address == other.address &&
				pitch == other.pitch &&
				width == other.width &&
				height == other.height &&
				format == other.format &&
				aa_mode == other.aa_mode &&
				raster_type == other.raster_type;
		}
	};

	struct color_surface_key_hash
	{
		std::size_t operator()(const color_surface_key& key) const
		{
			std::size_t seed = 0;
			hash_combine(seed, key.address);
			hash_combine(seed, key.pitch);
			hash_combine(seed, key.width);
			hash_combine(seed, key.height);
			hash_combine(seed, key.surface_index);
			hash_combine(seed, static_cast<u32>(key.format));
			hash_combine(seed, static_cast<u32>(key.aa_mode));
			hash_combine(seed, static_cast<u32>(key.raster_type));
			return seed;
		}

	private:
		static void hash_combine(std::size_t& seed, u64 value)
		{
			seed ^= std::hash<u64>{}(value) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
		}
	};

	struct depth_stencil_surface_key_hash
	{
		std::size_t operator()(const depth_stencil_surface_key& key) const
		{
			std::size_t seed = 0;
			hash_combine(seed, key.address);
			hash_combine(seed, key.pitch);
			hash_combine(seed, key.width);
			hash_combine(seed, key.height);
			hash_combine(seed, static_cast<u32>(key.format));
			hash_combine(seed, static_cast<u32>(key.aa_mode));
			hash_combine(seed, static_cast<u32>(key.raster_type));
			return seed;
		}

	private:
		static void hash_combine(std::size_t& seed, u64 value)
		{
			seed ^= std::hash<u64>{}(value) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
		}
	};

	color_surface_key make_color_surface_key(const rsx::framebuffer_layout& layout, u32 surface_index)
	{
		rsx_log.trace("make_color_surface_key(surface_index=%u)", surface_index);

		if (surface_index >= layout.color_addresses.size())
		{
			fmt::throw_exception("Metal color surface index %u is out of range", surface_index);
		}

		if (!layout.color_addresses[surface_index])
		{
			fmt::throw_exception("Metal color target cache requires a valid RSX color address");
		}

		if (!layout.actual_color_pitch[surface_index])
		{
			fmt::throw_exception("Metal color target cache requires a valid RSX color pitch");
		}

		if (!layout.width || !layout.height)
		{
			fmt::throw_exception("Metal color target cache requires a non-zero framebuffer size");
		}

		return
		{
			.address = layout.color_addresses[surface_index],
			.pitch = layout.actual_color_pitch[surface_index],
			.width = layout.width,
			.height = layout.height,
			.surface_index = surface_index,
			.format = layout.color_format,
			.aa_mode = layout.aa_mode,
			.raster_type = layout.raster_type,
		};
	}

	depth_stencil_surface_key make_depth_stencil_surface_key(const rsx::framebuffer_layout& layout)
	{
		rsx_log.trace("make_depth_stencil_surface_key()");

		if (!layout.zeta_address)
		{
			fmt::throw_exception("Metal depth/stencil target cache requires a valid RSX zeta address");
		}

		if (!layout.actual_zeta_pitch)
		{
			fmt::throw_exception("Metal depth/stencil target cache requires a valid RSX zeta pitch");
		}

		if (!layout.width || !layout.height)
		{
			fmt::throw_exception("Metal depth/stencil target cache requires a non-zero framebuffer size");
		}

		return
		{
			.address = layout.zeta_address,
			.pitch = layout.actual_zeta_pitch,
			.width = layout.width,
			.height = layout.height,
			.format = layout.depth_format,
			.aa_mode = layout.aa_mode,
			.raster_type = layout.raster_type,
		};
	}

	MTLPixelFormat get_color_target_pixel_format(rsx::surface_color_format format)
	{
		rsx_log.trace("get_color_target_pixel_format(format=0x%x)", static_cast<u32>(format));

		switch (format)
		{
		case rsx::surface_color_format::a8r8g8b8:
			return MTLPixelFormatBGRA8Unorm;
		case rsx::surface_color_format::a8b8g8r8:
			return MTLPixelFormatRGBA8Unorm;
		default:
			fmt::throw_exception("Metal color target format 0x%x is not implemented", static_cast<u32>(format));
		}
	}

	MTLPixelFormat get_depth_stencil_target_pixel_format(rsx::surface_depth_format2 format)
	{
		rsx_log.trace("get_depth_stencil_target_pixel_format(format=0x%x)", static_cast<u32>(format));

		switch (format)
		{
		case rsx::surface_depth_format2::z16_uint:
			return MTLPixelFormatDepth16Unorm;
		case rsx::surface_depth_format2::z16_float:
			return MTLPixelFormatDepth32Float;
		case rsx::surface_depth_format2::z24s8_uint:
		case rsx::surface_depth_format2::z24s8_float:
			return MTLPixelFormatDepth32Float_Stencil8;
		default:
			fmt::throw_exception("Metal depth/stencil target format 0x%x is not implemented", static_cast<u32>(format));
		}
	}

	std::string make_color_target_label(const color_surface_key& key)
	{
		rsx_log.trace("make_color_target_label(address=0x%x, surface_index=%u)", key.address, key.surface_index);
		return fmt::format("RPCS3 Metal color target %u 0x%08x", key.surface_index, key.address);
	}

	std::string make_depth_stencil_target_label(const depth_stencil_surface_key& key)
	{
		rsx_log.trace("make_depth_stencil_target_label(address=0x%x)", key.address);
		return fmt::format("RPCS3 Metal depth/stencil target 0x%08x", key.address);
	}
}

namespace rsx::metal
{
	struct render_target_cache::render_target_cache_impl
	{
		device& m_device;
		std::unordered_map<color_surface_key, std::unique_ptr<texture>, color_surface_key_hash> m_color_targets;
		std::unordered_map<depth_stencil_surface_key, std::unique_ptr<texture>, depth_stencil_surface_key_hash> m_depth_stencil_targets;
		render_target_cache_stats m_stats{};

		explicit render_target_cache_impl(device& metal_device)
			: m_device(metal_device)
		{
		}
	};

	render_target_cache::render_target_cache(device& metal_device)
		: m_impl(std::make_unique<render_target_cache_impl>(metal_device))
	{
		rsx_log.notice("rsx::metal::render_target_cache::render_target_cache(device=*0x%x)", metal_device.handle());
	}

	render_target_cache::~render_target_cache()
	{
		rsx_log.notice("rsx::metal::render_target_cache::~render_target_cache()");
	}

	texture& render_target_cache::get_color_target(const rsx::framebuffer_layout& layout, u32 surface_index)
	{
		rsx_log.trace("rsx::metal::render_target_cache::get_color_target(surface_index=%u)", surface_index);

		const color_surface_key key = make_color_surface_key(layout, surface_index);
		if (key.aa_mode != rsx::surface_antialiasing::center_1_sample)
		{
			fmt::throw_exception("Metal color target cache does not implement MSAA render targets yet");
		}

		if (key.raster_type != rsx::surface_raster_type::linear)
		{
			fmt::throw_exception("Metal color target cache does not implement swizzled render targets yet");
		}

		if (auto found = m_impl->m_color_targets.find(key); found != m_impl->m_color_targets.end())
		{
			m_impl->m_stats.color_target_reuse_count++;
			return *found->second;
		}

		if (!@available(macOS 26.0, *))
		{
			fmt::throw_exception("Metal color target cache requires macOS 26.0 or newer");
		}

		MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:get_color_target_pixel_format(key.format)
			width:key.width
			height:key.height
			mipmapped:NO];
		texture_desc.resourceOptions = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked;
		texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;

		auto color_texture = std::make_unique<texture>(m_impl->m_device, (__bridge void*)texture_desc, make_color_target_label(key));
		texture& result = *color_texture;
		m_impl->m_color_targets.emplace(key, std::move(color_texture));
		m_impl->m_stats.color_target_create_count++;
		m_impl->m_stats.color_target_count = ::size32(m_impl->m_color_targets);

		return result;
	}

	texture& render_target_cache::get_depth_stencil_target(const rsx::framebuffer_layout& layout)
	{
		rsx_log.trace("rsx::metal::render_target_cache::get_depth_stencil_target()");

		const depth_stencil_surface_key key = make_depth_stencil_surface_key(layout);
		if (key.aa_mode != rsx::surface_antialiasing::center_1_sample)
		{
			fmt::throw_exception("Metal depth/stencil target cache does not implement MSAA render targets yet");
		}

		if (key.raster_type != rsx::surface_raster_type::linear)
		{
			fmt::throw_exception("Metal depth/stencil target cache does not implement swizzled render targets yet");
		}

		if (auto found = m_impl->m_depth_stencil_targets.find(key); found != m_impl->m_depth_stencil_targets.end())
		{
			m_impl->m_stats.depth_stencil_target_reuse_count++;
			return *found->second;
		}

		if (!@available(macOS 26.0, *))
		{
			fmt::throw_exception("Metal depth/stencil target cache requires macOS 26.0 or newer");
		}

		MTLTextureDescriptor* texture_desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:get_depth_stencil_target_pixel_format(key.format)
			width:key.width
			height:key.height
			mipmapped:NO];
		texture_desc.resourceOptions = MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked;
		texture_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;

		auto depth_stencil_texture = std::make_unique<texture>(m_impl->m_device, (__bridge void*)texture_desc, make_depth_stencil_target_label(key));
		texture& result = *depth_stencil_texture;
		m_impl->m_depth_stencil_targets.emplace(key, std::move(depth_stencil_texture));
		m_impl->m_stats.depth_stencil_target_create_count++;
		m_impl->m_stats.depth_stencil_target_count = ::size32(m_impl->m_depth_stencil_targets);

		return result;
	}

	u32 render_target_cache::get_color_target_metal_pixel_format(const rsx::framebuffer_layout& layout, u32 surface_index) const
	{
		rsx_log.trace("rsx::metal::render_target_cache::get_color_target_metal_pixel_format(surface_index=%u)", surface_index);

		const color_surface_key key = make_color_surface_key(layout, surface_index);
		return static_cast<u32>(get_color_target_pixel_format(key.format));
	}

	draw_target_binding render_target_cache::prepare_draw_targets(const rsx::framebuffer_layout& layout)
	{
		rsx_log.trace("rsx::metal::render_target_cache::prepare_draw_targets(width=%u, height=%u, target=0x%x, zeta_address=0x%x)",
			layout.width, layout.height, static_cast<u32>(layout.target), layout.zeta_address);

		if (!layout.width || !layout.height)
		{
			fmt::throw_exception("Metal draw target preparation requires a valid framebuffer size");
		}

		draw_target_binding result =
		{
			.width = layout.width,
			.height = layout.height,
		};

		for (u8 index : rsx::utility::get_rtt_indexes(layout.target))
		{
			if (!layout.color_addresses[index])
			{
				continue;
			}

			texture& color_texture = get_color_target(layout, index);
			result.color_textures[index] = &color_texture;
			result.color_target_count++;
			m_impl->m_stats.draw_color_target_bind_count++;
		}

		if (layout.zeta_address)
		{
			texture& depth_stencil_texture = get_depth_stencil_target(layout);
			result.depth_stencil_texture = &depth_stencil_texture;
			result.stencil_attachment = is_depth_stencil_format(layout.depth_format);
			m_impl->m_stats.draw_depth_stencil_target_bind_count++;
		}

		if (!result.color_target_count && !result.depth_stencil_texture)
		{
			fmt::throw_exception("Metal draw target preparation found no active render targets");
		}

		m_impl->m_stats.draw_framebuffer_prepare_count++;
		return result;
	}

	void render_target_cache::clear_color_target(command_queue& queue, const rsx::framebuffer_layout& layout, u32 surface_index, clear_color color)
	{
		rsx_log.trace("rsx::metal::render_target_cache::clear_color_target(surface_index=%u, red=%f, green=%f, blue=%f, alpha=%f)",
			surface_index, color.red, color.green, color.blue, color.alpha);

		texture& color_texture = get_color_target(layout, surface_index);
		command_frame& frame = queue.begin_frame();
		frame.use_residency_set(m_impl->m_device.residency_set_handle());
		encode_clear_color_target(frame, color_texture, color);
		frame.end();
		queue.submit_frame(frame, nullptr);
		m_impl->m_stats.color_target_clear_count++;
	}

	void render_target_cache::clear_depth_stencil_target(command_queue& queue, const rsx::framebuffer_layout& layout, clear_depth_stencil clear)
	{
		rsx_log.trace("rsx::metal::render_target_cache::clear_depth_stencil_target(depth=%d, stencil=%d, depth_value=%f, stencil_value=0x%x)",
			clear.depth, clear.stencil, clear.depth_value, clear.stencil_value);

		texture& depth_stencil_texture = get_depth_stencil_target(layout);
		command_frame& frame = queue.begin_frame();
		frame.use_residency_set(m_impl->m_device.residency_set_handle());
		encode_clear_depth_stencil_target(frame, depth_stencil_texture, clear);
		frame.end();
		queue.submit_frame(frame, nullptr);
		m_impl->m_stats.depth_stencil_target_clear_count++;
	}

	render_target_cache_stats render_target_cache::stats() const
	{
		rsx_log.trace("rsx::metal::render_target_cache::stats()");
		return m_impl->m_stats;
	}

	void render_target_cache::report() const
	{
		rsx_log.notice("rsx::metal::render_target_cache::report()");
		rsx_log.notice("Metal render target cache: color_targets=%u, created=%u, reused=%u, clears=%u",
			m_impl->m_stats.color_target_count,
			m_impl->m_stats.color_target_create_count,
			m_impl->m_stats.color_target_reuse_count,
			m_impl->m_stats.color_target_clear_count);
		rsx_log.notice("Metal depth/stencil target cache: targets=%u, created=%u, reused=%u, clears=%u",
			m_impl->m_stats.depth_stencil_target_count,
			m_impl->m_stats.depth_stencil_target_create_count,
			m_impl->m_stats.depth_stencil_target_reuse_count,
			m_impl->m_stats.depth_stencil_target_clear_count);
		rsx_log.notice("Metal draw target preparation: framebuffers=%u, color_binds=%u, depth_stencil_binds=%u",
			m_impl->m_stats.draw_framebuffer_prepare_count,
			m_impl->m_stats.draw_color_target_bind_count,
			m_impl->m_stats.draw_depth_stencil_target_bind_count);
	}
}
