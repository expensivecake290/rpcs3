#include "stdafx.h"
#include "MTLRenderPass.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cmath>

namespace mtl
{
	namespace
	{
		MTLLoadAction to_native_load_action(attachment_load_action action)
		{
			switch (action)
			{
			case attachment_load_action::preserve: return MTLLoadActionLoad;
			case attachment_load_action::clear: return MTLLoadActionClear;
			case attachment_load_action::discard: return MTLLoadActionDontCare;
			}
			fmt::throw_exception("Invalid Metal attachment load action %u", static_cast<u8>(action));
		}

		MTLStoreAction to_native_store_action(attachment_store_action action)
		{
			switch (action)
			{
			case attachment_store_action::preserve: return MTLStoreActionStore;
			case attachment_store_action::discard: return MTLStoreActionDontCare;
			case attachment_store_action::resolve: return MTLStoreActionMultisampleResolve;
			case attachment_store_action::preserve_and_resolve: return MTLStoreActionStoreAndMultisampleResolve;
			}
			fmt::throw_exception("Invalid Metal attachment store action %u", static_cast<u8>(action));
		}

		MTLMultisampleDepthResolveFilter to_native_depth_resolve_filter(depth_resolve_filter filter)
		{
			switch (filter)
			{
			case depth_resolve_filter::sample_zero: return MTLMultisampleDepthResolveFilterSample0;
			case depth_resolve_filter::minimum: return MTLMultisampleDepthResolveFilterMin;
			case depth_resolve_filter::maximum: return MTLMultisampleDepthResolveFilterMax;
			}
			fmt::throw_exception("Invalid Metal depth resolve filter %u", static_cast<u8>(filter));
		}

		MTLVisibilityResultMode to_native_visibility_mode(visibility_result_mode mode)
		{
			switch (mode)
			{
			case visibility_result_mode::disabled: return MTLVisibilityResultModeDisabled;
			case visibility_result_mode::boolean: return MTLVisibilityResultModeBoolean;
			case visibility_result_mode::counting: return MTLVisibilityResultModeCounting;
			}
			fmt::throw_exception("Invalid Metal visibility result mode %u", static_cast<u8>(mode));
		}

		bool is_resolve_action(attachment_store_action action)
		{
			return action == attachment_store_action::resolve ||
				action == attachment_store_action::preserve_and_resolve;
		}

		void validate_actions(const attachment_reference& attachment,
			const render_attachment_actions& actions, const char* role)
		{
			if (!attachment)
			{
				if (is_resolve_action(actions.store))
				{
					fmt::throw_exception("Absent Metal %s attachment requests a resolve action", role);
				}
				return;
			}
			if (is_resolve_action(actions.store) && !attachment.has_resolve())
			{
				fmt::throw_exception("Metal %s attachment requests resolve without a resolve target", role);
			}
			if (attachment.memoryless && actions.load == attachment_load_action::preserve)
			{
				fmt::throw_exception("Memoryless Metal %s attachment cannot preserve prior contents", role);
			}
			if (attachment.memoryless && (actions.store == attachment_store_action::preserve ||
				actions.store == attachment_store_action::preserve_and_resolve))
			{
				fmt::throw_exception("Memoryless Metal %s attachment cannot store multisample contents", role);
			}
		}

		void configure_common_attachment(MTLRenderPassAttachmentDescriptor* descriptor,
			const attachment_reference& attachment, const render_attachment_actions& actions)
		{
			descriptor.texture = attachment.texture;
			descriptor.level = attachment.mip_level;
			descriptor.slice = attachment.array_slice;
			descriptor.depthPlane = attachment.depth_plane;
			descriptor.loadAction = to_native_load_action(actions.load);
			descriptor.storeAction = to_native_store_action(actions.store);
			if (attachment.has_resolve())
			{
				descriptor.resolveTexture = attachment.resolve_texture;
				descriptor.resolveLevel = attachment.resolve_mip_level;
				descriptor.resolveSlice = attachment.resolve_array_slice;
				descriptor.resolveDepthPlane = attachment.resolve_depth_plane;
			}
		}

		void retain_attachment(command_buffer& command, const attachment_reference& attachment)
		{
			if (!attachment)
			{
				return;
			}
			command.retain_native_object((__bridge void*)attachment.texture, true);
			if (attachment.has_resolve())
			{
				command.retain_native_object((__bridge void*)attachment.resolve_texture, true);
			}
		}

		MTL4RenderPassDescriptor* make_native_descriptor(command_buffer& command,
			const framebuffer& target, const render_pass_configuration& configuration)
		{
			const auto& attachments = target.attachments();
			MTL4RenderPassDescriptor* descriptor = [MTL4RenderPassDescriptor new];
			descriptor.renderTargetWidth = target.width();
			descriptor.renderTargetHeight = target.height();
			descriptor.renderTargetArrayLength = target.layers();
			descriptor.defaultRasterSampleCount = target.samples();
			descriptor.visibilityResultType = configuration.visibility_behavior == visibility_result_behavior::accumulate
				? MTLVisibilityResultTypeAccumulate : MTLVisibilityResultTypeReset;

			for (u32 index = 0; index < attachments.color_count(); ++index)
			{
				const auto& attachment = attachments.color(index);
				if (!attachment)
				{
					continue;
				}
				auto* native_attachment = descriptor.colorAttachments[index];
				configure_common_attachment(native_attachment, attachment, configuration.colors[index]);
				const auto& clear = configuration.colors[index].clear_color;
				native_attachment.clearColor = MTLClearColorMake(clear.red, clear.green, clear.blue, clear.alpha);
				retain_attachment(command, attachment);
			}

			if (attachments.depth())
			{
				configure_common_attachment(descriptor.depthAttachment, attachments.depth(), configuration.depth);
				descriptor.depthAttachment.clearDepth = configuration.depth.clear_depth;
				descriptor.depthAttachment.depthResolveFilter =
					to_native_depth_resolve_filter(configuration.depth.resolve_filter);
				retain_attachment(command, attachments.depth());
			}

			if (attachments.stencil())
			{
				configure_common_attachment(descriptor.stencilAttachment, attachments.stencil(), configuration.stencil);
				descriptor.stencilAttachment.clearStencil = configuration.stencil.clear_stencil;
				descriptor.stencilAttachment.stencilResolveFilter =
					configuration.stencil.resolve_filter == depth_resolve_filter::sample_zero
					? MTLMultisampleStencilResolveFilterSample0
					: MTLMultisampleStencilResolveFilterDepthResolvedSample;
				retain_attachment(command, attachments.stencil());
			}

			if (configuration.visibility_buffer)
			{
				descriptor.visibilityResultBuffer = configuration.visibility_buffer->native_handle();
				command.retain_native_object((__bridge void*)configuration.visibility_buffer->native_handle(), true);
			}
			return descriptor;
		}
	}

	void validate_render_pass_configuration(
		const framebuffer& target, const render_pass_configuration& configuration)
	{
		const auto& attachments = target.attachments();
		for (u32 index = 0; index < maximum_color_attachments; ++index)
		{
			validate_actions(attachments.color(index), configuration.colors[index], "color");
			const auto& clear = configuration.colors[index].clear_color;
			if (!std::isfinite(clear.red) || !std::isfinite(clear.green) ||
				!std::isfinite(clear.blue) || !std::isfinite(clear.alpha))
			{
				fmt::throw_exception("Metal color attachment %u has a non-finite clear value", index);
			}
		}
		validate_actions(attachments.depth(), configuration.depth, "depth");
		validate_actions(attachments.stencil(), configuration.stencil, "stencil");
		if (!std::isfinite(configuration.depth.clear_depth))
		{
			fmt::throw_exception("Metal depth attachment has a non-finite clear value");
		}
		if (configuration.visibility_required && !configuration.visibility_buffer)
		{
			fmt::throw_exception("Metal render pass requires a visibility result buffer");
		}
		if (configuration.visibility_buffer &&
			(!*configuration.visibility_buffer ||
			 !(configuration.visibility_buffer->usage() & buffer_usage_query) ||
			 configuration.visibility_buffer->size() < sizeof(u64)))
		{
			fmt::throw_exception("Invalid Metal render-pass visibility result buffer");
		}
	}

	struct render_pass::impl
	{
		command_buffer* active_command = nullptr;
		std::shared_ptr<framebuffer> active_target;
		render_pass_configuration active_configuration;
		native_encoder_handle encoder = nullptr;
		buffer_handle visibility_buffer = nullptr;
		u64 visibility_buffer_size = 0;
		visibility_result_mode visibility_mode = visibility_result_mode::disabled;
		u64 visibility_offset = 0;
		render_pass_statistics stats;
	};

	render_pass::render_pass()
		: m_impl(std::make_unique<impl>())
	{
	}

	render_pass::~render_pass()
	{
		if (is_open() && m_impl->active_command && m_impl->active_command->is_recording() &&
			m_impl->active_command->active_native_encoder() == m_impl->encoder)
		{
			try
			{
				m_impl->active_command->end_encoding();
			}
			catch (...)
			{
			}
		}
	}

	native_encoder_handle render_pass::begin(command_buffer& command,
		std::shared_ptr<framebuffer> target, const render_pass_configuration& configuration)
	{
		if (is_open() || !command.is_recording() || command.active_encoder() != encoder_kind::none || !target)
		{
			fmt::throw_exception("Invalid Metal render pass begin");
		}
		validate_render_pass_configuration(*target, configuration);
		MTL4RenderPassDescriptor* descriptor = make_native_descriptor(command, *target, configuration);
		const native_encoder_handle encoder = command.begin_render_encoding((__bridge void*)descriptor);
		id<MTL4RenderCommandEncoder> native_encoder = (__bridge id<MTL4RenderCommandEncoder>)encoder;
		native_encoder.label = [NSString stringWithUTF8String:
			(configuration.label.empty() ? "RPCS3 render pass" : configuration.label.c_str())];

		m_impl->active_command = &command;
		m_impl->active_target = std::move(target);
		m_impl->active_configuration = configuration;
		m_impl->encoder = encoder;
		m_impl->visibility_buffer = configuration.visibility_buffer
			? configuration.visibility_buffer->native_handle() : nullptr;
		m_impl->visibility_buffer_size = configuration.visibility_buffer
			? configuration.visibility_buffer->size() : 0;
		m_impl->visibility_mode = visibility_result_mode::disabled;
		m_impl->visibility_offset = 0;
		++m_impl->stats.opened;
		return encoder;
	}

	render_pass_transition render_pass::ensure(command_buffer& command,
		std::shared_ptr<framebuffer> target, const render_pass_configuration& configuration)
	{
		if (target && is_compatible(command, *target, configuration))
		{
			++m_impl->stats.reused;
			return render_pass_transition::reused;
		}
		if (is_open())
		{
			if (!target)
			{
				fmt::throw_exception("Metal render pass transition has no framebuffer target");
			}
			const visibility_result_mode prior_mode = m_impl->visibility_mode;
			const u64 prior_offset = m_impl->visibility_offset;
			end();
			static_cast<void>(begin(command, std::move(target), configuration));
			++m_impl->stats.restarted;
			if (prior_mode != visibility_result_mode::disabled)
			{
				set_visibility_result(prior_mode, prior_offset);
			}
			return render_pass_transition::restarted;
		}
		static_cast<void>(begin(command, std::move(target), configuration));
		return render_pass_transition::opened;
	}

	native_encoder_handle render_pass::restart(const render_pass_configuration& configuration)
	{
		if (!is_open())
		{
			fmt::throw_exception("Cannot restart a closed Metal render pass");
		}
		command_buffer* command = m_impl->active_command;
		auto target = m_impl->active_target;
		const visibility_result_mode prior_mode = m_impl->visibility_mode;
		const u64 prior_offset = m_impl->visibility_offset;
		end();
		const auto encoder = begin(*command, std::move(target), configuration);
		++m_impl->stats.restarted;
		if (prior_mode != visibility_result_mode::disabled)
		{
			set_visibility_result(prior_mode, prior_offset);
		}
		return encoder;
	}

	void render_pass::end()
	{
		if (!is_open() || !m_impl->active_command->is_recording() ||
			m_impl->active_command->active_encoder() != encoder_kind::render ||
			m_impl->active_command->active_native_encoder() != m_impl->encoder)
		{
			fmt::throw_exception("Cannot end an inactive or externally changed Metal render pass");
		}
		m_impl->active_command->end_encoding();
		m_impl->active_command = nullptr;
		m_impl->active_target.reset();
		m_impl->encoder = nullptr;
		m_impl->visibility_buffer = nullptr;
		m_impl->visibility_buffer_size = 0;
		m_impl->visibility_mode = visibility_result_mode::disabled;
		m_impl->visibility_offset = 0;
		++m_impl->stats.ended;
	}

	void render_pass::set_visibility_result(visibility_result_mode mode, u64 offset)
	{
		if (!is_open())
		{
			fmt::throw_exception("Cannot configure visibility on a closed Metal render pass");
		}
		if (mode == visibility_result_mode::disabled)
		{
			if (offset)
			{
				fmt::throw_exception("Disabled Metal visibility mode requires offset zero");
			}
		}
		else if (!m_impl->visibility_buffer || (offset & (sizeof(u64) - 1)) ||
			offset > m_impl->visibility_buffer_size - sizeof(u64))
		{
			fmt::throw_exception("Metal visibility result offset is invalid or has no backing buffer");
		}
		if (m_impl->visibility_mode == mode && m_impl->visibility_offset == offset)
		{
			return;
		}

		id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)m_impl->encoder;
		[encoder setVisibilityResultMode:to_native_visibility_mode(mode) offset:offset];
		m_impl->visibility_mode = mode;
		m_impl->visibility_offset = offset;
		++m_impl->stats.visibility_changes;
	}

	bool render_pass::is_compatible(const command_buffer& command, const framebuffer& target,
		const render_pass_configuration& configuration) const
	{
		return is_open() && m_impl->active_command == &command &&
			m_impl->active_target && m_impl->active_target->uid() == target.uid() &&
			m_impl->active_target->signature() == target.signature() &&
			m_impl->active_configuration == configuration && command.is_recording() &&
			command.active_encoder() == encoder_kind::render &&
			command.active_native_encoder() == m_impl->encoder;
	}

	bool render_pass::is_open() const
	{
		return m_impl && m_impl->active_command && m_impl->active_target && m_impl->encoder;
	}

	native_encoder_handle render_pass::native_encoder() const
	{
		return is_open() ? m_impl->encoder : nullptr;
	}

	command_buffer* render_pass::command() const
	{
		return is_open() ? m_impl->active_command : nullptr;
	}

	std::shared_ptr<framebuffer> render_pass::target() const
	{
		return is_open() ? m_impl->active_target : nullptr;
	}

	const render_pass_configuration& render_pass::configuration() const
	{
		return m_impl->active_configuration;
	}

	render_pass_statistics render_pass::statistics() const
	{
		return m_impl->stats;
	}
}
