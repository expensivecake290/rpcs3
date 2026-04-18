#include "stdafx.h"
#include "MTLTexture.h"

#include "MTLDevice.h"

#import <Metal/Metal.h>

namespace rsx::metal
{
	struct texture::texture_impl
	{
		device* m_device = nullptr;
		id<MTLTexture> m_texture = nil;
		void* m_residency_allocation_handle = nullptr;
		b8 m_resident = false;
		b8 m_heap_backed = false;
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

	texture::texture(device& dev, void* texture_descriptor_handle, std::string label)
		: m_impl(std::make_unique<texture_impl>())
	{
		rsx_log.notice("rsx::metal::texture::texture(device=*0x%x, texture_descriptor_handle=*0x%x, label=%s)",
			dev.handle(), texture_descriptor_handle, label);

		if (!texture_descriptor_handle)
		{
			fmt::throw_exception("Metal texture allocation requires a valid texture descriptor");
		}

		const device_texture_allocation allocation = dev.create_texture_allocation(texture_descriptor_handle, label);
		m_impl->m_texture = (__bridge_transfer id<MTLTexture>)allocation.texture_handle;
		if (!m_impl->m_texture)
		{
			fmt::throw_exception("Metal texture allocation returned a null texture");
		}

		m_impl->m_device = &dev;
		m_impl->m_residency_allocation_handle = allocation.residency_allocation_handle;
		m_impl->m_heap_backed = allocation.heap_backed;

		if (m_impl->m_residency_allocation_handle)
		{
			m_impl->m_device->add_resident_allocation(m_impl->m_residency_allocation_handle);
			m_impl->m_device->commit_residency();
			m_impl->m_resident = true;
		}
	}

	texture::~texture()
	{
		rsx_log.trace("rsx::metal::texture::~texture()");

		if (m_impl && m_impl->m_device && m_impl->m_texture && m_impl->m_heap_backed)
		{
			m_impl->m_device->retire_heap_resource((__bridge void*)m_impl->m_texture);
		}

		if (m_impl && m_impl->m_device && m_impl->m_residency_allocation_handle && m_impl->m_resident)
		{
			m_impl->m_device->remove_resident_allocation(m_impl->m_residency_allocation_handle);
			m_impl->m_device->commit_residency();
			m_impl->m_resident = false;
		}
	}

	void* texture::handle() const
	{
		rsx_log.trace("rsx::metal::texture::handle()");
		return (__bridge void*)m_impl->m_texture;
	}

	void* texture::allocation_handle() const
	{
		rsx_log.trace("rsx::metal::texture::allocation_handle()");
		return m_impl->m_residency_allocation_handle ? m_impl->m_residency_allocation_handle : (__bridge void*)m_impl->m_texture;
	}

	heap_resource_usage texture::heap_resource_usage_info() const
	{
		rsx_log.trace("rsx::metal::texture::heap_resource_usage_info()");

		if (!m_impl->m_heap_backed)
		{
			return {};
		}

		return
		{
			.metal_device = m_impl->m_device,
			.resource_handle = (__bridge void*)m_impl->m_texture,
		};
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
