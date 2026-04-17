#include "stdafx.h"
#include "MTLRenderTargets.h"

#include "MTLTexture.h"

#import <Metal/Metal.h>

namespace rsx::metal
{
	struct drawable_render_target::drawable_render_target_impl
	{
		MTL4RenderPassDescriptor* m_descriptor = nil;
		u32 m_width = 0;
		u32 m_height = 0;
	};

	drawable_render_target::drawable_render_target(texture& color_texture, u32 width, u32 height, clear_color color)
		: m_impl(std::make_unique<drawable_render_target_impl>())
	{
		rsx_log.trace("rsx::metal::drawable_render_target::drawable_render_target(color_texture=*0x%x, width=%u, height=%u)",
			color_texture.handle(), width, height);

		if (width == 0 || height == 0)
		{
			fmt::throw_exception("Metal drawable render target requires a non-zero size");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTLTexture> metal_texture = (__bridge id<MTLTexture>)color_texture.handle();
			MTL4RenderPassDescriptor* pass_desc = [MTL4RenderPassDescriptor new];
			MTLRenderPassColorAttachmentDescriptor* attachment = pass_desc.colorAttachments[0];

			attachment.texture = metal_texture;
			attachment.loadAction = MTLLoadActionClear;
			attachment.storeAction = MTLStoreActionStore;
			attachment.clearColor = MTLClearColorMake(color.red, color.green, color.blue, color.alpha);

			pass_desc.renderTargetWidth = width;
			pass_desc.renderTargetHeight = height;

			m_impl->m_descriptor = pass_desc;
		}
		else
		{
			fmt::throw_exception("Metal drawable render targets require macOS 26.0 or newer");
		}

		m_impl->m_width = width;
		m_impl->m_height = height;
	}

	drawable_render_target::~drawable_render_target()
	{
		rsx_log.trace("rsx::metal::drawable_render_target::~drawable_render_target()");
	}

	void* drawable_render_target::render_pass_descriptor_handle() const
	{
		rsx_log.trace("rsx::metal::drawable_render_target::render_pass_descriptor_handle()");
		return (__bridge void*)m_impl->m_descriptor;
	}

	u32 drawable_render_target::width() const
	{
		rsx_log.trace("rsx::metal::drawable_render_target::width()");
		return m_impl->m_width;
	}

	u32 drawable_render_target::height() const
	{
		rsx_log.trace("rsx::metal::drawable_render_target::height()");
		return m_impl->m_height;
	}
}
