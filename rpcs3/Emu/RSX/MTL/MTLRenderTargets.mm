#include "stdafx.h"
#include "MTLRenderTargets.h"

#include "MTLBarrier.h"
#include "MTLCommandBuffer.h"
#include "MTLDevice.h"
#include "MTLTexture.h"

#import <Metal/Metal.h>

namespace
{
	void track_heap_resource_use(rsx::metal::command_frame& frame, const rsx::metal::heap_resource_usage& usage);

	rsx::metal::resource_barrier track_render_target_write(rsx::metal::command_frame& frame, rsx::metal::texture& target)
	{
		rsx_log.trace("track_render_target_write(frame_index=%u, target=*0x%x)", frame.frame_index(), target.handle());

		frame.track_object(target.allocation_handle());
		track_heap_resource_use(frame, target.heap_resource_usage_info());
		return frame.track_resource_usage(rsx::metal::resource_usage
		{
			.resource_id = target.resource_id(),
			.stage = rsx::metal::resource_stage::render,
			.access = rsx::metal::resource_access::write,
			.scope = rsx::metal::resource_barrier_scope::render_targets
		});
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
			fmt::throw_exception("Metal render target heap resource tracking received incomplete heap usage state");
		}

		usage.metal_device->track_heap_resource_use(frame, usage.resource_handle);
	}
}

namespace rsx::metal
{
	struct draw_render_pass_descriptor::draw_render_pass_descriptor_impl
	{
		MTL4RenderPassDescriptor* m_descriptor = nil;
		u32 m_color_target_count = 0;
		b8 m_depth_stencil_target = false;
		b8 m_stencil_attachment = false;
		u32 m_width = 0;
		u32 m_height = 0;
	};

	draw_render_pass_descriptor::draw_render_pass_descriptor(const draw_render_pass_attachments& attachments)
		: m_impl(std::make_unique<draw_render_pass_descriptor_impl>())
	{
		rsx_log.trace("rsx::metal::draw_render_pass_descriptor::draw_render_pass_descriptor(color_target_count=%u, depth_stencil_texture=*0x%x, stencil_attachment=%d, width=%u, height=%u)",
			attachments.color_target_count,
			attachments.depth_stencil_texture ? attachments.depth_stencil_texture->handle() : nullptr,
			attachments.stencil_attachment,
			attachments.width,
			attachments.height);

		if (!attachments.width || !attachments.height)
		{
			fmt::throw_exception("Metal draw render pass descriptor requires a non-zero size");
		}

		if (!attachments.color_target_count && !attachments.depth_stencil_texture)
		{
			fmt::throw_exception("Metal draw render pass descriptor requires at least one attachment");
		}

		if (@available(macOS 26.0, *))
		{
			MTL4RenderPassDescriptor* pass_desc = [MTL4RenderPassDescriptor new];
			u32 color_target_count = 0;

			for (u32 index = 0; index < attachments.color_textures.size(); index++)
			{
				texture* color_texture = attachments.color_textures[index];
				if (!color_texture)
				{
					continue;
				}

				if (color_texture->width() != attachments.width || color_texture->height() != attachments.height)
				{
					fmt::throw_exception("Metal draw color attachment %u size mismatch: texture=%ux%u, pass=%ux%u",
						index,
						color_texture->width(),
						color_texture->height(),
						attachments.width,
						attachments.height);
				}

				MTLRenderPassColorAttachmentDescriptor* attachment = pass_desc.colorAttachments[index];
				attachment.texture = (__bridge id<MTLTexture>)color_texture->handle();
				attachment.loadAction = MTLLoadActionLoad;
				attachment.storeAction = MTLStoreActionStore;
				color_target_count++;
			}

			if (color_target_count != attachments.color_target_count)
			{
				fmt::throw_exception("Metal draw render pass color attachment count mismatch: counted=%u, expected=%u",
					color_target_count,
					attachments.color_target_count);
			}

			if (attachments.depth_stencil_texture)
			{
				if (attachments.depth_stencil_texture->width() != attachments.width ||
					attachments.depth_stencil_texture->height() != attachments.height)
				{
					fmt::throw_exception("Metal draw depth/stencil attachment size mismatch: texture=%ux%u, pass=%ux%u",
						attachments.depth_stencil_texture->width(),
						attachments.depth_stencil_texture->height(),
						attachments.width,
						attachments.height);
				}

				id<MTLTexture> depth_stencil_texture = (__bridge id<MTLTexture>)attachments.depth_stencil_texture->handle();
				MTLRenderPassDepthAttachmentDescriptor* depth_attachment = pass_desc.depthAttachment;
				depth_attachment.texture = depth_stencil_texture;
				depth_attachment.loadAction = MTLLoadActionLoad;
				depth_attachment.storeAction = MTLStoreActionStore;

				if (attachments.stencil_attachment)
				{
					MTLRenderPassStencilAttachmentDescriptor* stencil_attachment = pass_desc.stencilAttachment;
					stencil_attachment.texture = depth_stencil_texture;
					stencil_attachment.loadAction = MTLLoadActionLoad;
					stencil_attachment.storeAction = MTLStoreActionStore;
				}
			}

			pass_desc.renderTargetWidth = attachments.width;
			pass_desc.renderTargetHeight = attachments.height;

			m_impl->m_descriptor = pass_desc;
			m_impl->m_color_target_count = color_target_count;
			m_impl->m_depth_stencil_target = !!attachments.depth_stencil_texture;
			m_impl->m_stencil_attachment = attachments.stencil_attachment;
		}
		else
		{
			fmt::throw_exception("Metal draw render pass descriptors require macOS 26.0 or newer");
		}

		m_impl->m_width = attachments.width;
		m_impl->m_height = attachments.height;
	}

	draw_render_pass_descriptor::~draw_render_pass_descriptor()
	{
		rsx_log.trace("rsx::metal::draw_render_pass_descriptor::~draw_render_pass_descriptor()");
	}

	void* draw_render_pass_descriptor::handle() const
	{
		rsx_log.trace("rsx::metal::draw_render_pass_descriptor::handle()");
		return (__bridge void*)m_impl->m_descriptor;
	}

	u32 draw_render_pass_descriptor::color_target_count() const
	{
		rsx_log.trace("rsx::metal::draw_render_pass_descriptor::color_target_count()");
		return m_impl->m_color_target_count;
	}

	b8 draw_render_pass_descriptor::has_depth_stencil_target() const
	{
		rsx_log.trace("rsx::metal::draw_render_pass_descriptor::has_depth_stencil_target()");
		return m_impl->m_depth_stencil_target;
	}

	b8 draw_render_pass_descriptor::has_stencil_attachment() const
	{
		rsx_log.trace("rsx::metal::draw_render_pass_descriptor::has_stencil_attachment()");
		return m_impl->m_stencil_attachment;
	}

	u32 draw_render_pass_descriptor::width() const
	{
		rsx_log.trace("rsx::metal::draw_render_pass_descriptor::width()");
		return m_impl->m_width;
	}

	u32 draw_render_pass_descriptor::height() const
	{
		rsx_log.trace("rsx::metal::draw_render_pass_descriptor::height()");
		return m_impl->m_height;
	}

	struct draw_render_encoder_scope::draw_render_encoder_scope_impl
	{
		std::unique_ptr<draw_render_pass_descriptor> m_descriptor;
		std::array<resource_barrier, rsx::limits::color_buffers_count> m_color_barriers{};
		resource_barrier m_depth_stencil_barrier{};
		id<MTL4RenderCommandEncoder> m_encoder = nil;
		u32 m_color_target_count = 0;
		b8 m_depth_stencil_target = false;
		b8 m_stencil_attachment = false;
		b8 m_open = false;
	};

	draw_render_encoder_scope::draw_render_encoder_scope(command_frame& frame, const draw_render_pass_attachments& attachments)
		: m_impl(std::make_unique<draw_render_encoder_scope_impl>())
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::draw_render_encoder_scope(frame_index=%u, color_target_count=%u, depth_stencil_texture=*0x%x)",
			frame.frame_index(),
			attachments.color_target_count,
			attachments.depth_stencil_texture ? attachments.depth_stencil_texture->handle() : nullptr);

		m_impl->m_descriptor = std::make_unique<draw_render_pass_descriptor>(attachments);
		m_impl->m_color_target_count = m_impl->m_descriptor->color_target_count();
		m_impl->m_depth_stencil_target = m_impl->m_descriptor->has_depth_stencil_target();
		m_impl->m_stencil_attachment = m_impl->m_descriptor->has_stencil_attachment();

		for (u32 index = 0; index < attachments.color_textures.size(); index++)
		{
			texture* color_texture = attachments.color_textures[index];
			if (!color_texture)
			{
				continue;
			}

			m_impl->m_color_barriers[index] = track_render_target_write(frame, *color_texture);
		}

		if (attachments.depth_stencil_texture)
		{
			m_impl->m_depth_stencil_barrier = track_render_target_write(frame, *attachments.depth_stencil_texture);
		}

		if (@available(macOS 26.0, *))
		{
			id<MTL4CommandBuffer> command_buffer = (__bridge id<MTL4CommandBuffer>)frame.command_buffer_handle();
			m_impl->m_encoder = [command_buffer renderCommandEncoderWithDescriptor:
				(__bridge MTL4RenderPassDescriptor*)m_impl->m_descriptor->handle()];

			if (!m_impl->m_encoder)
			{
				fmt::throw_exception("Metal failed to create draw render command encoder");
			}

			for (const resource_barrier& barrier : m_impl->m_color_barriers)
			{
				encode_consumer_barrier((__bridge void*)m_impl->m_encoder, barrier);
			}

			encode_consumer_barrier((__bridge void*)m_impl->m_encoder, m_impl->m_depth_stencil_barrier);
			m_impl->m_open = true;
		}
		else
		{
			fmt::throw_exception("Metal draw render encoders require macOS 26.0 or newer");
		}
	}

	draw_render_encoder_scope::~draw_render_encoder_scope()
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::~draw_render_encoder_scope()");

		if (m_impl && m_impl->m_open && m_impl->m_encoder)
		{
			[m_impl->m_encoder endEncoding];
			m_impl->m_open = false;
		}
	}

	void* draw_render_encoder_scope::encoder_handle() const
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::encoder_handle()");
		return (__bridge void*)m_impl->m_encoder;
	}

	void draw_render_encoder_scope::end_encoding()
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::end_encoding(encoder=*0x%x)", encoder_handle());

		if (!m_impl->m_open || !m_impl->m_encoder)
		{
			fmt::throw_exception("Metal draw render encoder scope is not open");
		}

		[m_impl->m_encoder endEncoding];
		m_impl->m_open = false;
	}

	const resource_barrier& draw_render_encoder_scope::color_barrier(u32 index) const
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::color_barrier(index=%u)", index);

		if (index >= m_impl->m_color_barriers.size())
		{
			fmt::throw_exception("Metal draw render encoder color barrier index %u is out of range", index);
		}

		return m_impl->m_color_barriers[index];
	}

	const resource_barrier& draw_render_encoder_scope::depth_stencil_barrier() const
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::depth_stencil_barrier()");
		return m_impl->m_depth_stencil_barrier;
	}

	u32 draw_render_encoder_scope::color_target_count() const
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::color_target_count()");
		return m_impl->m_color_target_count;
	}

	b8 draw_render_encoder_scope::has_depth_stencil_target() const
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::has_depth_stencil_target()");
		return m_impl->m_depth_stencil_target;
	}

	b8 draw_render_encoder_scope::has_stencil_attachment() const
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::has_stencil_attachment()");
		return m_impl->m_stencil_attachment;
	}

	u32 draw_render_encoder_scope::width() const
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::width()");
		return m_impl->m_descriptor->width();
	}

	u32 draw_render_encoder_scope::height() const
	{
		rsx_log.trace("rsx::metal::draw_render_encoder_scope::height()");
		return m_impl->m_descriptor->height();
	}

	void encode_clear_color_target(command_frame& frame, texture& color_texture, clear_color color)
	{
		rsx_log.trace("rsx::metal::encode_clear_color_target(frame_index=%u, color_texture=*0x%x, red=%f, green=%f, blue=%f, alpha=%f)",
			frame.frame_index(), color_texture.handle(), color.red, color.green, color.blue, color.alpha);

		if (!@available(macOS 26.0, *))
		{
			fmt::throw_exception("Metal color target clears require macOS 26.0 or newer");
		}

		drawable_render_target render_target(frame,
			color_texture,
			color_texture.width(),
			color_texture.height(),
			color);

		id<MTL4CommandBuffer> command_buffer = (__bridge id<MTL4CommandBuffer>)frame.command_buffer_handle();
		id<MTL4RenderCommandEncoder> encoder = [command_buffer renderCommandEncoderWithDescriptor:
			(__bridge MTL4RenderPassDescriptor*)render_target.render_pass_descriptor_handle()];

		if (!encoder)
		{
			fmt::throw_exception("Metal failed to create render command encoder for color target clear");
		}

		encode_consumer_barrier((__bridge void*)encoder, render_target.color_barrier());
		[encoder endEncoding];
	}

	void encode_clear_depth_stencil_target(command_frame& frame, texture& depth_stencil_texture, clear_depth_stencil clear)
	{
		rsx_log.trace("rsx::metal::encode_clear_depth_stencil_target(frame_index=%u, depth_stencil_texture=*0x%x, depth=%d, stencil=%d, depth_value=%f, stencil_value=0x%x)",
			frame.frame_index(), depth_stencil_texture.handle(), clear.depth, clear.stencil, clear.depth_value, clear.stencil_value);

		if (!@available(macOS 26.0, *))
		{
			fmt::throw_exception("Metal depth/stencil target clears require macOS 26.0 or newer");
		}

		depth_stencil_render_target render_target(frame,
			depth_stencil_texture,
			depth_stencil_texture.width(),
			depth_stencil_texture.height(),
			clear);

		id<MTL4CommandBuffer> command_buffer = (__bridge id<MTL4CommandBuffer>)frame.command_buffer_handle();
		id<MTL4RenderCommandEncoder> encoder = [command_buffer renderCommandEncoderWithDescriptor:
			(__bridge MTL4RenderPassDescriptor*)render_target.render_pass_descriptor_handle()];

		if (!encoder)
		{
			fmt::throw_exception("Metal failed to create render command encoder for depth/stencil target clear");
		}

		encode_consumer_barrier((__bridge void*)encoder, render_target.barrier());
		[encoder endEncoding];
	}

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

		frame.track_object(color_texture.allocation_handle());
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

	struct depth_stencil_render_target::depth_stencil_render_target_impl
	{
		MTL4RenderPassDescriptor* m_descriptor = nil;
		resource_barrier m_barrier{};
		u32 m_width = 0;
		u32 m_height = 0;
	};

	depth_stencil_render_target::depth_stencil_render_target(command_frame& frame, texture& depth_stencil_texture, u32 width, u32 height, clear_depth_stencil clear)
		: m_impl(std::make_unique<depth_stencil_render_target_impl>())
	{
		rsx_log.trace("rsx::metal::depth_stencil_render_target::depth_stencil_render_target(frame_index=%u, depth_stencil_texture=*0x%x, width=%u, height=%u, depth=%d, stencil=%d)",
			frame.frame_index(), depth_stencil_texture.handle(), width, height, clear.depth, clear.stencil);

		if (width == 0 || height == 0)
		{
			fmt::throw_exception("Metal depth/stencil render target requires a non-zero size");
		}

		if (!clear.depth && !clear.stencil)
		{
			fmt::throw_exception("Metal depth/stencil render target clear requires at least one active aspect");
		}

		frame.track_object(depth_stencil_texture.allocation_handle());
		track_heap_resource_use(frame, depth_stencil_texture.heap_resource_usage_info());
		m_impl->m_barrier = frame.track_resource_usage(resource_usage
		{
			.resource_id = depth_stencil_texture.resource_id(),
			.stage = resource_stage::render,
			.access = resource_access::write,
			.scope = resource_barrier_scope::render_targets
		});

		if (@available(macOS 26.0, *))
		{
			id<MTLTexture> metal_texture = (__bridge id<MTLTexture>)depth_stencil_texture.handle();
			MTL4RenderPassDescriptor* pass_desc = [MTL4RenderPassDescriptor new];

			if (clear.depth)
			{
				MTLRenderPassDepthAttachmentDescriptor* depth_attachment = pass_desc.depthAttachment;
				depth_attachment.texture = metal_texture;
				depth_attachment.loadAction = MTLLoadActionClear;
				depth_attachment.storeAction = MTLStoreActionStore;
				depth_attachment.clearDepth = clear.depth_value;
			}

			if (clear.stencil)
			{
				MTLRenderPassStencilAttachmentDescriptor* stencil_attachment = pass_desc.stencilAttachment;
				stencil_attachment.texture = metal_texture;
				stencil_attachment.loadAction = MTLLoadActionClear;
				stencil_attachment.storeAction = MTLStoreActionStore;
				stencil_attachment.clearStencil = clear.stencil_value;
			}

			pass_desc.renderTargetWidth = width;
			pass_desc.renderTargetHeight = height;

			m_impl->m_descriptor = pass_desc;
		}
		else
		{
			fmt::throw_exception("Metal depth/stencil render targets require macOS 26.0 or newer");
		}

		m_impl->m_width = width;
		m_impl->m_height = height;
	}

	depth_stencil_render_target::~depth_stencil_render_target()
	{
		rsx_log.trace("rsx::metal::depth_stencil_render_target::~depth_stencil_render_target()");
	}

	void* depth_stencil_render_target::render_pass_descriptor_handle() const
	{
		rsx_log.trace("rsx::metal::depth_stencil_render_target::render_pass_descriptor_handle()");
		return (__bridge void*)m_impl->m_descriptor;
	}

	const resource_barrier& depth_stencil_render_target::barrier() const
	{
		rsx_log.trace("rsx::metal::depth_stencil_render_target::barrier()");
		return m_impl->m_barrier;
	}

	u32 depth_stencil_render_target::width() const
	{
		rsx_log.trace("rsx::metal::depth_stencil_render_target::width()");
		return m_impl->m_width;
	}

	u32 depth_stencil_render_target::height() const
	{
		rsx_log.trace("rsx::metal::depth_stencil_render_target::height()");
		return m_impl->m_height;
	}
}
