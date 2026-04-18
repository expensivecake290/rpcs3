#include "stdafx.h"
#include "MTLTextureViewPool.h"

#include "MTLDevice.h"
#include "MTLTexture.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <vector>

namespace
{
	NSString* make_ns_string(const std::string& value)
	{
		return [NSString stringWithUTF8String:value.c_str()];
	}

	std::string get_ns_error(NSError* error)
	{
		if (!error)
		{
			return {};
		}

		NSString* description = [error localizedDescription];
		const char* text = description ? [description UTF8String] : nullptr;
		return text ? std::string(text) : std::string();
	}
}

namespace rsx::metal
{
	struct texture_view_pool::texture_view_pool_impl
	{
		id<MTLTextureViewPool> m_pool = nil;
		std::vector<texture_view_binding> m_views;
		u32 m_view_count = 0;
	};

	texture_view_pool::texture_view_pool(device& dev, u32 view_count, std::string label)
		: m_impl(std::make_unique<texture_view_pool_impl>())
	{
		rsx_log.notice("rsx::metal::texture_view_pool::texture_view_pool(device=*0x%x, view_count=%u, label=%s)",
			dev.handle(), view_count, label.c_str());

		if (!dev.handle())
		{
			fmt::throw_exception("Metal texture view pool requires a valid device");
		}

		if (!view_count)
		{
			fmt::throw_exception("Metal texture view pool requires at least one view");
		}

		if (@available(macOS 26.0, *))
		{
			MTLResourceViewPoolDescriptor* desc = [MTLResourceViewPoolDescriptor new];
			desc.resourceViewCount = view_count;

			if (!label.empty())
			{
				desc.label = make_ns_string(label);
			}

			NSError* error = nil;
			id<MTLDevice> metal_device = (__bridge id<MTLDevice>)dev.handle();
			m_impl->m_pool = [metal_device newTextureViewPoolWithDescriptor:desc error:&error];
			if (!m_impl->m_pool)
			{
				const std::string message = get_ns_error(error);
				if (message.empty())
				{
					fmt::throw_exception("Metal texture view pool creation failed");
				}

				fmt::throw_exception("Metal texture view pool creation failed: %s", message);
			}
		}
		else
		{
			fmt::throw_exception("Metal texture view pools require macOS 26.0 or newer");
		}

		m_impl->m_view_count = view_count;
		m_impl->m_views.resize(view_count);
	}

	texture_view_pool::~texture_view_pool()
	{
		u32 assigned_view_count = 0;
		for (const texture_view_binding& binding : m_impl->m_views)
		{
			if (binding.view_resource_id)
			{
				assigned_view_count++;
			}
		}

		rsx_log.notice("rsx::metal::texture_view_pool::~texture_view_pool(pool=*0x%x, view_count=%u, assigned=%u)",
			m_impl ? (__bridge void*)m_impl->m_pool : nullptr,
			m_impl ? m_impl->m_view_count : 0,
			assigned_view_count);
	}

	void* texture_view_pool::handle() const
	{
		rsx_log.trace("rsx::metal::texture_view_pool::handle()");
		return (__bridge void*)m_impl->m_pool;
	}

	texture_view_binding texture_view_pool::set_default_texture_view(u32 index, const texture& tex)
	{
		rsx_log.trace("rsx::metal::texture_view_pool::set_default_texture_view(index=%u, texture=*0x%x)",
			index, tex.handle());

		if (index >= m_impl->m_view_count)
		{
			fmt::throw_exception("Metal texture view pool index out of range: index=%u, view_count=%u",
				index, m_impl->m_view_count);
		}

		const u64 source_resource_id = tex.resource_id();
		if (!source_resource_id)
		{
			fmt::throw_exception("Metal texture view pool requires a non-zero source texture resource ID");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTLTexture> metal_texture = (__bridge id<MTLTexture>)tex.handle();
			const MTLResourceID view_resource_id = [m_impl->m_pool setTextureView:metal_texture atIndex:index];
			if (!view_resource_id._impl)
			{
				fmt::throw_exception("Metal texture view pool returned a zero resource ID");
			}

			texture_view_binding binding =
			{
				.source_resource_id = source_resource_id,
				.view_resource_id = view_resource_id._impl,
				.index = index,
			};

			m_impl->m_views[index] = binding;
			return binding;
		}

		fmt::throw_exception("Metal texture view pool updates require macOS 26.0 or newer");
	}
}
