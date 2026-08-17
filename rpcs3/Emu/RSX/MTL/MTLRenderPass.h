#pragma once

#include <array>
#include <memory>
#include <string>

#include "MTLFramebuffer.h"
#include "mtlutils/buffer_object.h"
#include "mtlutils/commands.h"

namespace mtl
{
	enum class attachment_load_action : u8
	{
		preserve,
		clear,
		discard,
	};

	enum class attachment_store_action : u8
	{
		preserve,
		discard,
		resolve,
		preserve_and_resolve,
	};

	enum class depth_resolve_filter : u8
	{
		sample_zero,
		minimum,
		maximum,
	};

	struct clear_color_value
	{
		f64 red = 0.0;
		f64 green = 0.0;
		f64 blue = 0.0;
		f64 alpha = 0.0;

		[[nodiscard]] bool operator==(const clear_color_value&) const = default;
	};

	struct render_attachment_actions
	{
		attachment_load_action load = attachment_load_action::preserve;
		attachment_store_action store = attachment_store_action::preserve;
		clear_color_value clear_color;
		f64 clear_depth = 1.0;
		u32 clear_stencil = 0;
		depth_resolve_filter resolve_filter = depth_resolve_filter::sample_zero;

		[[nodiscard]] bool operator==(const render_attachment_actions&) const = default;
	};

	enum class visibility_result_behavior : u8
	{
		reset,
		accumulate,
	};

	struct render_pass_configuration
	{
		std::array<render_attachment_actions, maximum_color_attachments> colors{};
		render_attachment_actions depth;
		render_attachment_actions stencil;
		const buffer* visibility_buffer = nullptr;
		std::string label;
		visibility_result_behavior visibility_behavior = visibility_result_behavior::reset;
		bool visibility_required = false;

		[[nodiscard]] bool operator==(const render_pass_configuration&) const = default;
	};

	enum class visibility_result_mode : u8
	{
		disabled,
		boolean,
		counting,
	};

	enum class render_pass_transition : u8
	{
		opened,
		reused,
		restarted,
	};

	struct render_pass_statistics
	{
		u64 opened = 0;
		u64 reused = 0;
		u64 restarted = 0;
		u64 ended = 0;
		u64 visibility_changes = 0;
	};

	void validate_render_pass_configuration(
		const framebuffer& target, const render_pass_configuration& configuration);

	class render_pass final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		render_pass();
		~render_pass();
		render_pass(const render_pass&) = delete;
		render_pass& operator=(const render_pass&) = delete;
		render_pass(render_pass&&) = delete;
		render_pass& operator=(render_pass&&) = delete;

		[[nodiscard]] native_encoder_handle begin(command_buffer& command,
			std::shared_ptr<framebuffer> target, const render_pass_configuration& configuration);
		[[nodiscard]] render_pass_transition ensure(command_buffer& command,
			std::shared_ptr<framebuffer> target, const render_pass_configuration& configuration);
		[[nodiscard]] native_encoder_handle restart(const render_pass_configuration& configuration);
		void end();

		void set_visibility_result(visibility_result_mode mode, u64 offset = 0);
		[[nodiscard]] bool is_compatible(const command_buffer& command, const framebuffer& target,
			const render_pass_configuration& configuration) const;
		[[nodiscard]] bool is_open() const;
		[[nodiscard]] native_encoder_handle native_encoder() const;
		[[nodiscard]] command_buffer* command() const;
		[[nodiscard]] std::shared_ptr<framebuffer> target() const;
		[[nodiscard]] const render_pass_configuration& configuration() const;
		[[nodiscard]] render_pass_statistics statistics() const;
	};
}
