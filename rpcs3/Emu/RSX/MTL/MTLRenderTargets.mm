#include "stdafx.h"
#include "MTLRenderTargets.h"

#include "MTLCommandBuffer.h"
#include "MTLDevice.h"
#include "MTLTexture.h"

#import <Metal/Metal.h>

namespace
{
	void track_heap_resource_use(rsx::metal::command_frame& frame, const rsx::metal::heap_resource_usage& usage)
	{
		rsx_log.trace("track_heap_resource_use(frame_index=%u, resource_handle=*0x%x)",
			frame.frame_index(), usage.resource_handle);

		if (usage.metal_device && usage.resource_handle)
		{
			usage.metal_device->track_heap_resource_use(frame, usage.resource_handle);
		}
	}
}

namespace rsx::metal
{
	struct drawable_render_target::drawable_render_target_impl
	{
		MTL4RenderPassDescriptor* m_descriptor = nil;
		resource_barrier m_color_barrier{};
		u32 m_width = 0;
		u32 m_height = 0;
	};

	drawable_render_target::drawable_render_target(command_frame& frame, texture& color_texture, u32 width, u32 height, clear_color color)
		: m_impl(std::make_unique<drawable_render_target_impl>())
	{
		rsx_log.trace("rsx::metal::drawable_render_target::drawable_render_target(frame_index=%u, color_texture=*0x%x, width=%u, height=%u)",
			frame.frame_index(), color_texture.handle(), width, height);

		if (width == 0 || height == 0)
		{
			fmt::throw_exception("Metal drawable render target requires a non-zero size");
		}

		track_heap_resource_use(frame, color_texture.heap_resource_usage_info());
		m_impl->m_color_barrier = frame.track_resource_usage(resource_usage
		{
			.resource_id = color_texture.resource_id(),
			.stage = resource_stage::render,
			.access = resource_access::write,
			.scope = resource_barrier_scope::render_targets
		});

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

	const resource_barrier& drawable_render_target::color_barrier() const
	{
		rsx_log.trace("rsx::metal::drawable_render_target::color_barrier()");
		return m_impl->m_color_barrier;
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
