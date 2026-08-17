#pragma once

#include <chrono>
#include <string>

#include "commands.h"
#include "metal_layer.h"

namespace mtl
{
	enum class presentation_mode : u8
	{
		immediate,
		adaptive,
		synchronized,
	};

	enum class drawable_acquire_status : u8
	{
		success,
		timeout,
		unavailable,
		resized,
		surface_lost,
	};

	enum class drawable_present_status : u8
	{
		success,
		dropped,
		resized,
		surface_lost,
		failed,
	};

	struct swapchain_configuration
	{
		u64 pixel_format = 0;
		u32 width = 0;
		u32 height = 0;
		u32 maximum_drawables = 3;
		presentation_mode mode = presentation_mode::synchronized;
		std::chrono::nanoseconds acquire_timeout = std::chrono::seconds(1);
		std::string label;
		bool framebuffer_only = false;
		bool extended_dynamic_range = false;
		bool present_with_transaction = false;
		bool allow_acquire_timeout = true;

		[[nodiscard]] bool operator==(const swapchain_configuration&) const = default;
	};

	struct presentation_timing
	{
		f64 target_time = 0.0;
		f64 minimum_duration = 0.0;
		bool use_target_time = false;
		bool use_minimum_duration = false;
	};

	struct drawable_frame
	{
		drawable_handle drawable = nullptr;
		texture_handle texture = nullptr;
		drawable_size size;
		u64 frame_id = 0;
		u64 generation = 0;
		drawable_acquire_status status = drawable_acquire_status::unavailable;

		[[nodiscard]] explicit operator bool() const
		{
			return status == drawable_acquire_status::success && drawable && texture && size;
		}
	};

	struct swapchain_statistics
	{
		u64 generation = 0;
		u64 acquired_frames = 0;
		u64 presented_frames = 0;
		u64 dropped_frames = 0;
		u64 timed_out_acquires = 0;
		u64 resize_count = 0;
		u32 in_flight_drawables = 0;
		drawable_size size;
	};

	class swapchain_interface
	{
	public:
		virtual ~swapchain_interface() = default;

		virtual void create(render_device& device, command_queue_handle queue,
			native_view_handle view, const swapchain_configuration& configuration) = 0;
		virtual void destroy() = 0;
		virtual void reconfigure(const swapchain_configuration& configuration) = 0;
		virtual drawable_acquire_status resize() = 0;

		[[nodiscard]] virtual drawable_frame acquire_next_drawable(
			std::chrono::nanoseconds timeout) = 0;
		[[nodiscard]] virtual drawable_present_status present(
			drawable_frame& frame, const presentation_timing& timing = {}) = 0;
		virtual void discard(drawable_frame& frame) = 0;

		[[nodiscard]] virtual explicit operator bool() const = 0;
		[[nodiscard]] virtual const swapchain_configuration& configuration() const = 0;
		[[nodiscard]] virtual metal_layer_handle layer() const = 0;
		[[nodiscard]] virtual drawable_size size() const = 0;
		[[nodiscard]] virtual u64 generation() const = 0;
		[[nodiscard]] virtual swapchain_statistics statistics() const = 0;
	};
}
