#include "stdafx.h"
#include "MTLNativeWindow.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

#include <functional>

namespace
{
	void run_on_main_thread(const std::function<void()>& func)
	{
		if ([NSThread isMainThread])
		{
			func();
			return;
		}

		dispatch_sync(dispatch_get_main_queue(), ^
		{
			func();
		});
	}

	NSString* make_ns_string(std::string_view text)
	{
		return [[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
	}
}

namespace rsx::metal
{
	struct native_window::native_window_impl
	{
		NSWindow* m_window = nil;
		NSView* m_view = nil;
		CAMetalLayer* m_layer = nil;
		u32 m_drawable_width = 1;
		u32 m_drawable_height = 1;
		f64 m_refresh_rate = 60.;

		void update_drawable_size()
		{
			if (!m_window || !m_view || !m_layer)
			{
				return;
			}

			NSScreen* screen = [m_window screen] ?: [NSScreen mainScreen];
			const CGFloat scale = screen ? [screen backingScaleFactor] : 1.0;
			const NSRect bounds = [m_view bounds];
			const u32 width = static_cast<u32>(std::max<CGFloat>(1.0, std::floor(bounds.size.width * scale)));
			const u32 height = static_cast<u32>(std::max<CGFloat>(1.0, std::floor(bounds.size.height * scale)));

			m_layer.contentsScale = scale;
			m_layer.drawableSize = CGSizeMake(width, height);
			m_drawable_width = width;
			m_drawable_height = height;

			if (screen)
			{
				m_refresh_rate = std::max<f64>(20., [screen maximumFramesPerSecond]);
			}
		}
	};

	native_window::native_window(u32 width, u32 height)
		: m_impl(std::make_unique<native_window_impl>())
	{
		rsx_log.notice("rsx::metal::native_window::native_window(width=%u, height=%u)", width, height);

		run_on_main_thread([this, width, height]()
		{
			const NSRect rect = NSMakeRect(0.0, 0.0, width, height);
			const NSWindowStyleMask style =
				NSWindowStyleMaskTitled |
				NSWindowStyleMaskClosable |
				NSWindowStyleMaskMiniaturizable |
				NSWindowStyleMaskResizable;

			m_impl->m_window = [[NSWindow alloc] initWithContentRect:rect
				styleMask:style
				backing:NSBackingStoreBuffered
				defer:NO];

			m_impl->m_window.releasedWhenClosed = NO;
			m_impl->m_window.title = @"RPCS3 Metal";
			m_impl->m_window.collectionBehavior = NSWindowCollectionBehaviorFullScreenPrimary;

			m_impl->m_view = [[NSView alloc] initWithFrame:rect];
			m_impl->m_view.wantsLayer = YES;
			m_impl->m_layer = [CAMetalLayer layer];
			m_impl->m_view.layer = m_impl->m_layer;
			m_impl->m_view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
			m_impl->m_window.contentView = m_impl->m_view;
			[m_impl->m_window center];
			m_impl->update_drawable_size();
		});
	}

	native_window::~native_window()
	{
		rsx_log.notice("rsx::metal::native_window::~native_window()");
		close();
	}

	void native_window::show()
	{
		rsx_log.notice("rsx::metal::native_window::show()");

		run_on_main_thread([this]()
		{
			[m_impl->m_window makeKeyAndOrderFront:nil];
		});
	}

	void native_window::close()
	{
		rsx_log.notice("rsx::metal::native_window::close()");

		run_on_main_thread([this]()
		{
			if (m_impl->m_window)
			{
				[m_impl->m_window close];
				m_impl->m_window = nil;
				m_impl->m_view = nil;
				m_impl->m_layer = nil;
			}
		});
	}

	void native_window::set_title(std::string_view title)
	{
		const std::string title_text(title);
		rsx_log.trace("rsx::metal::native_window::set_title(title=%s)", title_text);

		run_on_main_thread([this, title_text]()
		{
			NSString* ns_title = make_ns_string(title_text);
			m_impl->m_window.title = ns_title;
		});
	}

	void native_window::update_drawable_size()
	{
		rsx_log.trace("rsx::metal::native_window::update_drawable_size()");

		run_on_main_thread([this]()
		{
			m_impl->update_drawable_size();
		});
	}

	u32 native_window::drawable_width() const
	{
		rsx_log.trace("rsx::metal::native_window::drawable_width()");
		return m_impl->m_drawable_width;
	}

	u32 native_window::drawable_height() const
	{
		rsx_log.trace("rsx::metal::native_window::drawable_height()");
		return m_impl->m_drawable_height;
	}

	f64 native_window::refresh_rate() const
	{
		rsx_log.trace("rsx::metal::native_window::refresh_rate()");
		return m_impl->m_refresh_rate;
	}

	void* native_window::layer_handle() const
	{
		rsx_log.trace("rsx::metal::native_window::layer_handle()");
		return (__bridge void*)m_impl->m_layer;
	}
}
