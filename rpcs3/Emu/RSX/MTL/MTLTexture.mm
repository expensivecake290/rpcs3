#include "stdafx.h"
#include "MTLTexture.h"

#import <Metal/Metal.h>

namespace rsx::metal
{
	struct texture::texture_impl
	{
		id<MTLTexture> m_texture = nil;
	};

	texture::texture(void* texture_handle)
		: m_impl(std::make_unique<texture_impl>())
	{
		rsx_log.trace("rsx::metal::texture::texture(texture_handle=*0x%x)", texture_handle);

		if (!texture_handle)
		{
			fmt::throw_exception("Metal texture requires a valid texture handle");
		}

		m_impl->m_texture = (__bridge id<MTLTexture>)texture_handle;
	}

	texture::~texture()
	{
		rsx_log.trace("rsx::metal::texture::~texture()");
	}

	void* texture::handle() const
	{
		rsx_log.trace("rsx::metal::texture::handle()");
		return (__bridge void*)m_impl->m_texture;
	}

	u64 texture::resource_id() const
	{
		rsx_log.trace("rsx::metal::texture::resource_id()");
		return m_impl->m_texture.gpuResourceID._impl;
	}

	u32 texture::width() const
	{
		rsx_log.trace("rsx::metal::texture::width()");
		return static_cast<u32>(m_impl->m_texture.width);
	}

	u32 texture::height() const
	{
		rsx_log.trace("rsx::metal::texture::height()");
		return static_cast<u32>(m_impl->m_texture.height);
	}
}
