#pragma once

#include "../MetalAPI.h"

namespace mtl
{
	using native_view_handle = void*;
	using metal_layer_handle = void*;

	struct drawable_size
	{
		u32 width = 0;
		u32 height = 0;

		explicit operator bool() const
		{
			return width != 0 && height != 0;
		}
	};

	// Returns the view's CAMetalLayer, or null when the view has not been prepared
	// for Metal presentation. The returned pointer is borrowed from the NSView.
	[[nodiscard]] metal_layer_handle get_metal_layer(native_view_handle view);

	// Creates and installs a CAMetalLayer when necessary. The NSView owns the
	// installed layer and the returned pointer remains borrowed.
	[[nodiscard]] metal_layer_handle ensure_metal_layer(native_view_handle view, device_handle device);

	// Updates drawableSize from the view's backing-pixel dimensions. This must be
	// called after window moves and scale or size changes.
	[[nodiscard]] drawable_size update_metal_layer_size(native_view_handle view, metal_layer_handle layer);
}
