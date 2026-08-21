#pragma once

#include "upscaling.h"

#include <memory>

namespace mtl
{
	enum class metalfx_color_processing : u8
	{
		perceptual,
		linear,
		high_dynamic_range,
	};

	class metalfx_upscale_pass final : public upscaler
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		metalfx_upscale_pass(render_device& device, memory_allocator& allocator,
			metalfx_color_processing color_processing = metalfx_color_processing::perceptual);
		~metalfx_upscale_pass() override;

		[[nodiscard]] static bool supported(const render_device& device);
		[[nodiscard]] metalfx_color_processing color_processing() const;

		viewable_image* scale_output(command_buffer& command,
			viewable_image& source, viewable_image* destination,
			const upscale_request& request, rsx::flags32_t mode) override;
	};
}
