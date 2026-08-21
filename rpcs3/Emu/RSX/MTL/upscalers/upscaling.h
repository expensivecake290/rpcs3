#pragma once

#include "../MTLHelpers.h"

#include <algorithm>
#include <limits>

namespace mtl
{
	enum upscaling_flag : u32
	{
		upscale_default_view = 1u << 0,
		upscale_left_view = 1u << 0,
		upscale_right_view = 1u << 1,
		upscale_and_commit = 1u << 2,
	};

	inline constexpr u32 upscale_view_mask = upscale_left_view | upscale_right_view;
	inline constexpr u32 upscale_known_mask = upscale_view_mask | upscale_and_commit;

	struct upscale_request
	{
		image_rectangle source_area;
		image_rectangle destination_area;

		[[nodiscard]] u32 source_width() const
		{
			const s64 difference = static_cast<s64>(source_area.x1) - source_area.x0;
			return static_cast<u32>(difference < 0 ? -difference : difference);
		}

		[[nodiscard]] u32 source_height() const
		{
			const s64 difference = static_cast<s64>(source_area.y1) - source_area.y0;
			return static_cast<u32>(difference < 0 ? -difference : difference);
		}

		[[nodiscard]] u32 destination_width() const
		{
			const s64 difference = static_cast<s64>(destination_area.x1) - destination_area.x0;
			return static_cast<u32>(difference < 0 ? -difference : difference);
		}

		[[nodiscard]] u32 destination_height() const
		{
			const s64 difference = static_cast<s64>(destination_area.y1) - destination_area.y0;
			return static_cast<u32>(difference < 0 ? -difference : difference);
		}

		[[nodiscard]] bool operator==(const upscale_request& other) const
		{
			return source_area.x0 == other.source_area.x0 &&
				source_area.y0 == other.source_area.y0 &&
				source_area.x1 == other.source_area.x1 &&
				source_area.y1 == other.source_area.y1 &&
				destination_area.x0 == other.destination_area.x0 &&
				destination_area.y0 == other.destination_area.y0 &&
				destination_area.x1 == other.destination_area.x1 &&
				destination_area.y1 == other.destination_area.y1;
		}
	};

	inline void validate_upscale_request(const command_buffer& command,
		const viewable_image& source, const viewable_image* destination,
		const upscale_request& request, rsx::flags32_t mode)
	{
		if (!command.is_recording() || !source || source.samples() != 1 ||
			source.type() != texture_type::texture_2d ||
			!(source.info().usage & texture_usage_shader_read))
		{
			fmt::throw_exception("Invalid Metal upscale source or command state");
		}
		if (mode & ~upscale_known_mask)
			fmt::throw_exception("Metal upscale request contains unknown mode bits 0x%x", mode);
		const u32 view = mode & upscale_view_mask;
		if (view != upscale_left_view && view != upscale_right_view)
			fmt::throw_exception("Metal upscale request must select exactly one output view");
		const bool commit = (mode & upscale_and_commit) != 0;
		if (commit != (destination != nullptr))
			fmt::throw_exception("Metal upscale destination does not match commit mode");
		if (destination && (!*destination || destination->samples() != 1 ||
			destination->type() != texture_type::texture_2d ||
			!(destination->info().usage & texture_usage_render_target)))
		{
			fmt::throw_exception("Invalid Metal upscale destination");
		}

		const auto within = [](const image_rectangle& area, u32 width, u32 height)
		{
			const s64 minimum_x = std::min<s64>(area.x0, area.x1);
			const s64 maximum_x = std::max<s64>(area.x0, area.x1);
			const s64 minimum_y = std::min<s64>(area.y0, area.y1);
			const s64 maximum_y = std::max<s64>(area.y0, area.y1);
			return minimum_x >= 0 && minimum_y >= 0 &&
				maximum_x <= width && maximum_y <= height;
		};
		if (!request.source_width() || !request.source_height() ||
			!request.destination_width() || !request.destination_height() ||
			!within(request.source_area, source.width(), source.height()) ||
			(destination && !within(request.destination_area,
				destination->width(), destination->height())))
		{
			fmt::throw_exception("Metal upscale request is empty or outside its image bounds");
		}
	}

	class upscaler
	{
	public:
		virtual ~upscaler() = default;
		upscaler(const upscaler&) = delete;
		upscaler& operator=(const upscaler&) = delete;
		upscaler(upscaler&&) = delete;
		upscaler& operator=(upscaler&&) = delete;

		virtual viewable_image* scale_output(command_buffer& command,
			viewable_image& source, viewable_image* destination,
			const upscale_request& request, rsx::flags32_t mode) = 0;

	protected:
		upscaler() = default;
	};
}
