#include "stdafx.h"
#include "swapchain_macos.hpp"

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <mutex>

namespace mtl
{
	namespace
	{
		template <typename Function>
		void on_main_thread(Function&& function)
		{
			if (NSThread.isMainThread)
			{
				function();
				return;
			}
			dispatch_sync(dispatch_get_main_queue(), std::forward<Function>(function));
		}

		NSView* native_view(native_view_handle view)
		{
			NSView* result = (__bridge NSView*)view;
			if (!result || ![result isKindOfClass:NSView.class])
			{
				fmt::throw_exception("Metal swapchain requires a valid NSView");
			}
			return result;
		}

		CAMetalLayer* native_layer(metal_layer_handle layer)
		{
			CAMetalLayer* result = (__bridge CAMetalLayer*)layer;
			if (!result || ![result isKindOfClass:CAMetalLayer.class])
			{
				fmt::throw_exception("Metal swapchain requires a valid CAMetalLayer");
			}
			return result;
		}

		bool supported_layer_format(u64 format)
		{
			switch (static_cast<MTLPixelFormat>(format))
			{
			case MTLPixelFormatBGRA8Unorm:
			case MTLPixelFormatBGRA8Unorm_sRGB:
			case MTLPixelFormatRGBA16Float:
			case MTLPixelFormatBGRA10_XR:
			case MTLPixelFormatBGRA10_XR_sRGB:
				return true;
			default:
				return false;
			}
		}

		void configure_layer(CAMetalLayer* layer, id<MTLDevice> device,
			const swapchain_configuration& configuration)
		{
			if (!supported_layer_format(configuration.pixel_format) ||
				configuration.maximum_drawables < 2 || configuration.maximum_drawables > 3)
			{
				fmt::throw_exception("Unsupported Metal presentation format or drawable count");
			}
			on_main_thread([&]
			{
				layer.device = device;
				layer.pixelFormat = static_cast<MTLPixelFormat>(configuration.pixel_format);
				layer.maximumDrawableCount = configuration.maximum_drawables;
				layer.framebufferOnly = configuration.framebuffer_only;
				layer.presentsWithTransaction = configuration.present_with_transaction;
				layer.displaySyncEnabled = configuration.mode != presentation_mode::immediate;
				layer.allowsNextDrawableTimeout = configuration.allow_acquire_timeout;
				layer.wantsExtendedDynamicRangeContent = configuration.extended_dynamic_range;
				layer.colorspace = configuration.extended_dynamic_range
					? NSColorSpace.extendedSRGBColorSpace.CGColorSpace
					: NSColorSpace.sRGBColorSpace.CGColorSpace;
			});
		}

		macos_surface_state query_surface_state(NSView* view, CAMetalLayer* layer,
			id<MTLDevice> device, id<MTL4CommandQueue> queue)
		{
			macos_surface_state result;
			result.view = (__bridge void*)view;
			result.layer = (__bridge void*)layer;
			result.device = device;
			result.queue = queue;
			result.drawable_extent = {static_cast<u32>(layer.drawableSize.width),
				static_cast<u32>(layer.drawableSize.height)};
			result.display_synchronized = layer.displaySyncEnabled;
			result.extended_dynamic_range = layer.wantsExtendedDynamicRangeContent;
			on_main_thread([&]
			{
				NSWindow* window = view.window;
				result.backing_scale = window ? window.backingScaleFactor : NSScreen.mainScreen.backingScaleFactor;
				result.minimized = window ? window.miniaturized : false;
				result.visible = window && window.visible && !view.hidden && !result.minimized;
				result.occluded = window && !(window.occlusionState & NSWindowOcclusionStateVisible);
			});
			return result;
		}
	}

	struct macos_swapchain::impl
	{
		render_device* owner = nullptr;
		id<MTL4CommandQueue> queue = nil;
		NSView* view = nil;
		CAMetalLayer* layer = nil;
		id<MTLResidencySet> layer_residency = nil;
		swapchain_core core;
		mutable std::mutex mutex;
		bool surface_lost = false;
	};

	macos_swapchain::macos_swapchain()
		: m_impl(std::make_unique<impl>())
	{
	}

	macos_swapchain::~macos_swapchain()
	{
		destroy();
	}

	void macos_swapchain::create(render_device& device, command_queue_handle queue,
		native_view_handle view, const swapchain_configuration& configuration)
	{
		destroy();
		if (!device || !queue || !view || !configuration.pixel_format || !configuration.width ||
			!configuration.height || configuration.label.empty())
		{
			fmt::throw_exception("Invalid Metal macOS swapchain creation request");
		}
		NSView* view_object = native_view(view);
		CAMetalLayer* layer_object = native_layer(ensure_metal_layer(view, device.native_handle()));
		configure_layer(layer_object, device.native_handle(), configuration);
		const drawable_size actual_size = update_metal_layer_size(view, (__bridge void*)layer_object);
		if (!actual_size)
		{
			fmt::throw_exception("Metal presentation view has an empty drawable extent");
		}
		id<MTL4CommandQueue> native_queue = queue;
		id<MTLResidencySet> residency = layer_object.residencySet;
		if (!residency)
		{
			fmt::throw_exception("CAMetalLayer did not provide its Metal 4 residency set");
		}
		[native_queue addResidencySet:residency];
		try
		{
			m_impl->core.initialize(configuration, actual_size);
		}
		catch (...)
		{
			[native_queue removeResidencySet:residency];
			throw;
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->owner = &device;
		m_impl->queue = native_queue;
		m_impl->view = view_object;
		m_impl->layer = layer_object;
		m_impl->layer_residency = residency;
		m_impl->surface_lost = false;
	}

	void macos_swapchain::destroy()
	{
		if (!m_impl) return;
		id<MTL4CommandQueue> queue = nil;
		id<MTLResidencySet> residency = nil;
		CAMetalLayer* layer = nil;
		{
			std::lock_guard lock(m_impl->mutex);
			queue = m_impl->queue;
			residency = m_impl->layer_residency;
			layer = m_impl->layer;
			m_impl->owner = nullptr;
			m_impl->queue = nil;
			m_impl->view = nil;
			m_impl->layer = nil;
			m_impl->layer_residency = nil;
			m_impl->surface_lost = false;
		}
		m_impl->core.shutdown();
		if (queue && residency) [queue removeResidencySet:residency];
		if (layer)
		{
			on_main_thread([&]
			{
				layer.device = nil;
			});
		}
	}

	void macos_swapchain::reconfigure(const swapchain_configuration& configuration)
	{
		render_device* owner = nullptr;
		NSView* view = nil;
		CAMetalLayer* layer = nil;
		{
			std::lock_guard lock(m_impl->mutex);
			owner = m_impl->owner;
			view = m_impl->view;
			layer = m_impl->layer;
			if (m_impl->surface_lost) owner = nullptr;
		}
		if (!owner || !view || !layer)
		{
			fmt::throw_exception("Cannot reconfigure an unavailable Metal swapchain");
		}
		configure_layer(layer, owner->native_handle(), configuration);
		const drawable_size actual = update_metal_layer_size((__bridge void*)view, (__bridge void*)layer);
		if (!actual) fmt::throw_exception("Metal swapchain reconfiguration produced an empty drawable extent");
		m_impl->core.reconfigure(configuration, actual);
	}

	drawable_acquire_status macos_swapchain::resize()
	{
		NSView* view = nil;
		CAMetalLayer* layer = nil;
		{
			std::lock_guard lock(m_impl->mutex);
			view = m_impl->view;
			layer = m_impl->layer;
			if (!m_impl->owner || m_impl->surface_lost)
			{
				view = nil;
				layer = nil;
			}
		}
		if (!view || !layer)
		{
			return drawable_acquire_status::surface_lost;
		}
		const drawable_size actual = update_metal_layer_size((__bridge void*)view, (__bridge void*)layer);
		if (!actual) return drawable_acquire_status::unavailable;
		return m_impl->core.resize(actual) ? drawable_acquire_status::resized : drawable_acquire_status::success;
	}

	drawable_frame macos_swapchain::acquire_next_drawable(std::chrono::nanoseconds timeout)
	{
		if (timeout < std::chrono::nanoseconds::zero())
		{
			fmt::throw_exception("Metal drawable timeout cannot be negative");
		}
		const drawable_acquire_status resize_status = resize();
		if (resize_status == drawable_acquire_status::surface_lost ||
			resize_status == drawable_acquire_status::unavailable)
		{
			if (m_impl->core.initialized()) m_impl->core.note_acquire_failure(resize_status);
			return {.status = resize_status};
		}

		render_device* owner = nullptr;
		id<MTL4CommandQueue> queue = nil;
		NSView* view = nil;
		CAMetalLayer* layer = nil;
		{
			std::lock_guard lock(m_impl->mutex);
			owner = m_impl->owner;
			queue = m_impl->queue;
			view = m_impl->view;
			layer = m_impl->layer;
			if (m_impl->surface_lost) layer = nil;
		}
		if (!owner || !layer || !queue)
		{
			return {.status = drawable_acquire_status::surface_lost};
		}
		if (timeout == std::chrono::nanoseconds::zero() &&
			m_impl->core.statistics().in_flight_drawables >= m_impl->core.configuration().maximum_drawables)
		{
			m_impl->core.note_acquire_failure(drawable_acquire_status::unavailable);
			return {.status = drawable_acquire_status::unavailable};
		}

		layer.allowsNextDrawableTimeout = timeout != std::chrono::nanoseconds::max();
		id<CAMetalDrawable> drawable = [layer nextDrawable];
		if (!drawable)
		{
			const macos_surface_state surface = query_surface_state(view, layer,
				owner->native_handle(), queue);
			const drawable_acquire_status status = surface.visible && !surface.occluded
				? drawable_acquire_status::timeout : drawable_acquire_status::unavailable;
			m_impl->core.note_acquire_failure(status);
			return {.status = status};
		}
		[queue waitForDrawable:drawable];
		const drawable_size actual = {static_cast<u32>(drawable.texture.width),
			static_cast<u32>(drawable.texture.height)};
		return m_impl->core.acquire(drawable, drawable.texture, actual);
	}

	drawable_present_status macos_swapchain::present(drawable_frame& frame, const presentation_timing& timing)
	{
		if (!m_impl->core.is_current(frame))
		{
			fmt::throw_exception("Cannot present an unknown Metal drawable frame");
		}
		if ((timing.use_target_time && timing.target_time < 0.0) ||
			(timing.use_minimum_duration && timing.minimum_duration < 0.0))
		{
			fmt::throw_exception("Metal presentation timing cannot be negative");
		}
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->queue || m_impl->surface_lost)
		{
			return m_impl->core.complete(frame, drawable_present_status::surface_lost);
		}
		id<MTLDrawable> drawable = frame.drawable;
		[m_impl->queue signalDrawable:drawable];
		if (timing.use_target_time)
		{
			[drawable presentAtTime:timing.target_time];
		}
		else if (timing.use_minimum_duration)
		{
			[drawable presentAfterMinimumDuration:timing.minimum_duration];
		}
		else
		{
			[drawable present];
		}
		return m_impl->core.complete(frame, drawable_present_status::success);
	}

	void macos_swapchain::discard(drawable_frame& frame)
	{
		m_impl->core.discard(frame);
	}

	macos_swapchain::operator bool() const
	{
		if (!m_impl) return false;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->owner && m_impl->queue && m_impl->view && m_impl->layer &&
			!m_impl->surface_lost && m_impl->core.initialized();
	}

	const swapchain_configuration& macos_swapchain::configuration() const
	{
		return m_impl->core.configuration();
	}

	metal_layer_handle macos_swapchain::layer() const
	{
		if (!m_impl) return nullptr;
		std::lock_guard lock(m_impl->mutex);
		return (__bridge void*)m_impl->layer;
	}

	drawable_size macos_swapchain::size() const
	{
		return m_impl ? m_impl->core.size() : drawable_size{};
	}

	u64 macos_swapchain::generation() const
	{
		return m_impl ? m_impl->core.generation() : 0;
	}

	swapchain_statistics macos_swapchain::statistics() const
	{
		return m_impl ? m_impl->core.statistics() : swapchain_statistics{};
	}

	macos_surface_state macos_swapchain::surface_state() const
	{
		if (!m_impl) return {};
		render_device* owner = nullptr;
		id<MTL4CommandQueue> queue = nil;
		NSView* view = nil;
		CAMetalLayer* layer = nil;
		{
			std::lock_guard lock(m_impl->mutex);
			owner = m_impl->owner;
			queue = m_impl->queue;
			view = m_impl->view;
			layer = m_impl->layer;
		}
		if (!owner || !view || !layer || !queue) return {};
		return query_surface_state(view, layer, owner->native_handle(), queue);
	}
}
