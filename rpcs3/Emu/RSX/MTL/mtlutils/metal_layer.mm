#include "stdafx.h"
#include "metal_layer.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <cmath>

namespace mtl
{
	namespace
	{
		template <typename F>
		void on_main_thread(F&& function)
		{
			if (NSThread.isMainThread)
			{
				function();
				return;
			}

			dispatch_sync(dispatch_get_main_queue(), std::forward<F>(function));
		}

		NSView* get_native_view(native_view_handle view)
		{
			if (!view)
			{
				fmt::throw_exception("Cannot create a Metal layer without a native view");
			}

			NSView* native_view = (__bridge NSView*)view;
			if (![native_view isKindOfClass:NSView.class])
			{
				fmt::throw_exception("Metal presentation requires an NSView");
			}

			return native_view;
		}

		CAMetalLayer* get_native_layer(metal_layer_handle layer)
		{
			if (!layer)
			{
				fmt::throw_exception("Cannot configure a null CAMetalLayer");
			}

			CAMetalLayer* native_layer = (__bridge CAMetalLayer*)layer;
			if (![native_layer isKindOfClass:CAMetalLayer.class])
			{
				fmt::throw_exception("Metal presentation layer is not a CAMetalLayer");
			}

			return native_layer;
		}
	}

	metal_layer_handle get_metal_layer(native_view_handle view)
	{
		NSView* native_view = get_native_view(view);
		CAMetalLayer* result = nil;

		on_main_thread([&]
		{
			if ([native_view.layer isKindOfClass:CAMetalLayer.class])
			{
				result = static_cast<CAMetalLayer*>(native_view.layer);
			}
		});

		return (__bridge void*)result;
	}

	metal_layer_handle ensure_metal_layer(native_view_handle view, device_handle device)
	{
		NSView* native_view = get_native_view(view);
		if (!device)
		{
			fmt::throw_exception("Cannot create a CAMetalLayer without a Metal device");
		}

		CAMetalLayer* result = nil;
		on_main_thread([&]
		{
			native_view.wantsLayer = YES;
			if ([native_view.layer isKindOfClass:CAMetalLayer.class])
			{
				result = static_cast<CAMetalLayer*>(native_view.layer);
			}
			else
			{
				result = [CAMetalLayer layer];
				native_view.layer = result;
			}

			result.device = device;
			result.framebufferOnly = NO;
			result.opaque = YES;
		});

		const auto size = update_metal_layer_size(view, (__bridge void*)result);
		static_cast<void>(size);
		return (__bridge void*)result;
	}

	drawable_size update_metal_layer_size(native_view_handle view, metal_layer_handle layer)
	{
		NSView* native_view = get_native_view(view);
		CAMetalLayer* native_layer = get_native_layer(layer);
		drawable_size result;

		on_main_thread([&]
		{
			const NSRect backing_bounds = [native_view convertRectToBacking:native_view.bounds];
			const CGFloat width = std::max<CGFloat>(0.0, std::ceil(backing_bounds.size.width));
			const CGFloat height = std::max<CGFloat>(0.0, std::ceil(backing_bounds.size.height));
			const CGFloat scale = native_view.window ? native_view.window.backingScaleFactor : NSScreen.mainScreen.backingScaleFactor;

			native_layer.contentsScale = scale > 0.0 ? scale : 1.0;
			native_layer.drawableSize = CGSizeMake(width, height);
			result.width = static_cast<u32>(width);
			result.height = static_cast<u32>(height);
		});

		return result;
	}
}
