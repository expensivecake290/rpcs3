#pragma once

#include "../MTLFormats.h"
#include "upscaling.h"

namespace mtl
{
	class nearest_upscale_pass final : public upscaler
	{
	public:
		viewable_image* scale_output(command_buffer& command,
			viewable_image& source, viewable_image* destination,
			const upscale_request& request, rsx::flags32_t mode) override
		{
			validate_upscale_request(command, source, destination, request, mode);
			if (!(mode & upscale_and_commit)) return &source;

			image_conversion conversion;
			const bool compatible = formats_are_bitcast_compatible(source, *destination);
			if (!compatible)
				conversion.kind = image_conversion_kind::color_to_color;
			copy_scaled_image(command, source, *destination,
				request.source_area, request.destination_area, 1, compatible,
				image_filter::nearest, conversion);
			return nullptr;
		}
	};
}
