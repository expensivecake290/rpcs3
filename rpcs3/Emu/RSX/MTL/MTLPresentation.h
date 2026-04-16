#pragma once

#include "MTLCommandQueue.h"
#include "MTLDevice.h"
#include "MTLNativeWindow.h"

#include <memory>

namespace rsx::metal
{
	class presentation_surface
	{
	public:
		presentation_surface(device& metal_device, native_window& window, b8 use_vsync);
		~presentation_surface();

		presentation_surface(const presentation_surface&) = delete;
		presentation_surface& operator=(const presentation_surface&) = delete;

		void present_clear_frame(command_queue& queue, f32 red, f32 green, f32 blue, f32 alpha);

	private:
		struct presentation_surface_impl;
		std::unique_ptr<presentation_surface_impl> m_impl;
	};
}
