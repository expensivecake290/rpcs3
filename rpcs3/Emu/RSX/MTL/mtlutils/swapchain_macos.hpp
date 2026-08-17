#pragma once

#include <memory>

#include "swapchain_core.h"

namespace mtl
{
	struct macos_surface_state
	{
		native_view_handle view = nullptr;
		metal_layer_handle layer = nullptr;
		device_handle device = nullptr;
		command_queue_handle queue = nullptr;
		drawable_size drawable_extent;
		f64 backing_scale = 1.0;
		bool visible = false;
		bool occluded = false;
		bool minimized = false;
		bool display_synchronized = true;
		bool extended_dynamic_range = false;

		[[nodiscard]] explicit operator bool() const
		{
			return view && layer && device && queue;
		}
	};

	class macos_swapchain final : public swapchain_interface
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		macos_swapchain();
		~macos_swapchain() override;
		macos_swapchain(const macos_swapchain&) = delete;
		macos_swapchain& operator=(const macos_swapchain&) = delete;
		macos_swapchain(macos_swapchain&&) = delete;
		macos_swapchain& operator=(macos_swapchain&&) = delete;

		void create(render_device& device, command_queue_handle queue,
			native_view_handle view, const swapchain_configuration& configuration) override;
		void destroy() override;
		void reconfigure(const swapchain_configuration& configuration) override;
		[[nodiscard]] drawable_acquire_status resize() override;

		[[nodiscard]] drawable_frame acquire_next_drawable(
			std::chrono::nanoseconds timeout) override;
		[[nodiscard]] drawable_present_status present(
			drawable_frame& frame, const presentation_timing& timing = {}) override;
		void discard(drawable_frame& frame) override;

		[[nodiscard]] explicit operator bool() const override;
		[[nodiscard]] const swapchain_configuration& configuration() const override;
		[[nodiscard]] metal_layer_handle layer() const override;
		[[nodiscard]] drawable_size size() const override;
		[[nodiscard]] u64 generation() const override;
		[[nodiscard]] swapchain_statistics statistics() const override;
		[[nodiscard]] macos_surface_state surface_state() const;
	};

	using native_swapchain = macos_swapchain;
}
