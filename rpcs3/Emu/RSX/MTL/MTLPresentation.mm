#include "stdafx.h"
#include "MTLPresentation.h"

#include "MTLRenderTargets.h"
#include "MTLTexture.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

namespace rsx::metal
{
	struct presentation_surface::presentation_surface_impl
	{
		device& m_device;
		native_window& m_window;
		CAMetalLayer* m_layer = nil;
		id<MTLResidencySet> m_layer_residency_set = nil;

		presentation_surface_impl(device& metal_device, native_window& window)
			: m_device(metal_device)
			, m_window(window)
		{
		}
	};

	presentation_surface::presentation_surface(device& metal_device, native_window& window, b8 use_vsync)
		: m_impl(std::make_unique<presentation_surface_impl>(metal_device, window))
	{
		rsx_log.notice("rsx::metal::presentation_surface::presentation_surface(use_vsync=%d)", use_vsync);

		CAMetalLayer* layer = (__bridge CAMetalLayer*)window.layer_handle();
		if (!layer)
		{
			fmt::throw_exception("Metal presentation requires a CAMetalLayer");
		}

		layer.device = (__bridge id<MTLDevice>)metal_device.handle();
		layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
		layer.framebufferOnly = YES;
		layer.maximumDrawableCount = 3;
		layer.allowsNextDrawableTimeout = YES;
		layer.displaySyncEnabled = use_vsync;

		window.update_drawable_size();

		m_impl->m_layer = layer;

		if (@available(macOS 26.0, *))
		{
			m_impl->m_layer_residency_set = layer.residencySet;
		}
		else
		{
			fmt::throw_exception("Metal presentation requires macOS 26.0 or newer");
		}

		if (!m_impl->m_layer_residency_set)
		{
			fmt::throw_exception("Metal presentation failed to acquire CAMetalLayer residency set");
		}
	}

	presentation_surface::~presentation_surface()
	{
		rsx_log.notice("rsx::metal::presentation_surface::~presentation_surface()");
	}

	void presentation_surface::present_clear_frame(command_queue& queue, f32 red, f32 green, f32 blue, f32 alpha)
	{
		rsx_log.trace("rsx::metal::presentation_surface::present_clear_frame(red=%f, green=%f, blue=%f, alpha=%f)",
			red, green, blue, alpha);

		m_impl->m_window.update_drawable_size();

		id<CAMetalDrawable> drawable = [m_impl->m_layer nextDrawable];
		if (!drawable)
		{
			rsx_log.warning("Metal presentation skipped frame because CAMetalLayer returned no drawable");
			return;
		}

		command_frame& frame = queue.begin_frame();
		frame.track_object((__bridge void*)drawable);
		frame.use_residency_set(m_impl->m_device.residency_set_handle());
		frame.use_residency_set((__bridge void*)m_impl->m_layer_residency_set);

		if (!@available(macOS 26.0, *))
		{
			fmt::throw_exception("Metal presentation requires macOS 26.0 or newer");
		}

		texture drawable_texture((__bridge void*)drawable.texture);
		frame.track_object(drawable_texture.handle());
		m_impl->m_device.add_resident_allocation(drawable_texture.handle());
		m_impl->m_device.commit_residency();
		device* metal_device = &m_impl->m_device;
		frame.on_completed([metal_device, texture_handle = drawable_texture.handle()]()
		{
			metal_device->remove_resident_allocation(texture_handle);
			metal_device->commit_residency();
		});

		drawable_render_target render_target(drawable_texture,
			drawable_texture.width(),
			drawable_texture.height(),
			{ red, green, blue, alpha });

		id<MTL4CommandBuffer> command_buffer = (__bridge id<MTL4CommandBuffer>)frame.command_buffer_handle();
		id<MTL4RenderCommandEncoder> encoder = [command_buffer renderCommandEncoderWithDescriptor:
			(__bridge MTL4RenderPassDescriptor*)render_target.render_pass_descriptor_handle()];

		if (!encoder)
		{
			fmt::throw_exception("Metal failed to create render command encoder for presentation clear");
		}

		[encoder endEncoding];
		frame.end();
		queue.submit_frame(frame, (__bridge void*)drawable);
	}
}
