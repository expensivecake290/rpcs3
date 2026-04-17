#include "stdafx.h"
#include "MTLArgumentTable.h"

#include "MTLBuffer.h"
#include "MTLDevice.h"
#include "MTLSampler.h"
#include "MTLTexture.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace
{
	constexpr u32 supported_render_stage_mask =
		static_cast<u32>(rsx::metal::argument_table_render_stage::vertex) |
		static_cast<u32>(rsx::metal::argument_table_render_stage::fragment) |
		static_cast<u32>(rsx::metal::argument_table_render_stage::tile) |
		static_cast<u32>(rsx::metal::argument_table_render_stage::object) |
		static_cast<u32>(rsx::metal::argument_table_render_stage::mesh);

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

	MTLResourceID make_resource_id(u64 value)
	{
		rsx_log.trace("make_resource_id(value=0x%llx)", value);

		MTLResourceID resource_id{};
		resource_id._impl = value;
		return resource_id;
	}

	MTLRenderStages make_render_stages(u32 stages)
	{
		rsx_log.trace("make_render_stages(stages=0x%x)", stages);

		if (!stages)
		{
			fmt::throw_exception("Metal argument table render binding requires at least one render stage");
		}

		if (stages & ~supported_render_stage_mask)
		{
			fmt::throw_exception("Metal argument table render binding has unsupported stage mask: 0x%x", stages);
		}

		MTLRenderStages result = 0;

		if (stages & static_cast<u32>(rsx::metal::argument_table_render_stage::vertex))
		{
			result |= MTLRenderStageVertex;
		}

		if (stages & static_cast<u32>(rsx::metal::argument_table_render_stage::fragment))
		{
			result |= MTLRenderStageFragment;
		}

		if (stages & static_cast<u32>(rsx::metal::argument_table_render_stage::tile))
		{
			result |= MTLRenderStageTile;
		}

		if (stages & static_cast<u32>(rsx::metal::argument_table_render_stage::object))
		{
			result |= MTLRenderStageObject;
		}

		if (stages & static_cast<u32>(rsx::metal::argument_table_render_stage::mesh))
		{
			result |= MTLRenderStageMesh;
		}

		return result;
	}
}

namespace rsx::metal
{
	struct argument_table::argument_table_impl
	{
		id<MTL4ArgumentTable> m_table = nil;
		argument_table_desc m_desc{};
	};

	argument_table::argument_table(device& dev, const argument_table_desc& desc)
		: m_impl(std::make_unique<argument_table_impl>())
	{
		rsx_log.notice("rsx::metal::argument_table::argument_table(device=*0x%x, max_buffers=%u, max_textures=%u, max_samplers=%u)",
			dev.handle(), desc.max_buffers, desc.max_textures, desc.max_samplers);

		if (!dev.handle())
		{
			fmt::throw_exception("Metal argument table requires a valid device");
		}

		if (!desc.max_buffers && !desc.max_textures && !desc.max_samplers)
		{
			fmt::throw_exception("Metal argument table requires at least one binding slot");
		}

		const device_caps& caps = dev.caps();
		if (desc.max_buffers > caps.max_argument_table_buffers)
		{
			fmt::throw_exception("Metal argument table buffer count exceeds device limit: requested=%u, limit=%u",
				desc.max_buffers, caps.max_argument_table_buffers);
		}

		if (desc.max_textures > caps.max_argument_table_textures)
		{
			fmt::throw_exception("Metal argument table texture count exceeds device limit: requested=%u, limit=%u",
				desc.max_textures, caps.max_argument_table_textures);
		}

		if (desc.max_samplers > caps.max_argument_table_samplers)
		{
			fmt::throw_exception("Metal argument table sampler count exceeds device limit: requested=%u, limit=%u",
				desc.max_samplers, caps.max_argument_table_samplers);
		}

		if (@available(macOS 26.0, *))
		{
			MTL4ArgumentTableDescriptor* table_desc = [MTL4ArgumentTableDescriptor new];
			table_desc.maxBufferBindCount = desc.max_buffers;
			table_desc.maxTextureBindCount = desc.max_textures;
			table_desc.maxSamplerStateBindCount = desc.max_samplers;
			table_desc.initializeBindings = desc.initialize_bindings;
			table_desc.supportAttributeStrides = desc.support_attribute_strides;

			if (!desc.label.empty())
			{
				table_desc.label = make_ns_string(desc.label);
			}

			NSError* error = nil;
			id<MTLDevice> metal_device = (__bridge id<MTLDevice>)dev.handle();
			m_impl->m_table = [metal_device newArgumentTableWithDescriptor:table_desc error:&error];
			if (!m_impl->m_table)
			{
				const std::string message = get_ns_error(error);
				if (message.empty())
				{
					fmt::throw_exception("Metal argument table creation failed");
				}

				fmt::throw_exception("Metal argument table creation failed: %s", message);
			}

			m_impl->m_desc = desc;
		}
		else
		{
			fmt::throw_exception("Metal argument table creation requires macOS 26.0 or newer");
		}
	}

	argument_table::~argument_table()
	{
		rsx_log.notice("rsx::metal::argument_table::~argument_table(table=*0x%x)", handle());
	}

	void* argument_table::handle() const
	{
		rsx_log.trace("rsx::metal::argument_table::handle()");
		return (__bridge void*)m_impl->m_table;
	}

	u32 argument_table::max_buffers() const
	{
		rsx_log.trace("rsx::metal::argument_table::max_buffers()");
		return m_impl->m_desc.max_buffers;
	}

	u32 argument_table::max_textures() const
	{
		rsx_log.trace("rsx::metal::argument_table::max_textures()");
		return m_impl->m_desc.max_textures;
	}

	u32 argument_table::max_samplers() const
	{
		rsx_log.trace("rsx::metal::argument_table::max_samplers()");
		return m_impl->m_desc.max_samplers;
	}

	b8 argument_table::supports_attribute_strides() const
	{
		rsx_log.trace("rsx::metal::argument_table::supports_attribute_strides()");
		return m_impl->m_desc.support_attribute_strides;
	}

	void argument_table::bind_buffer_address(u32 index, const buffer& buf, u64 offset)
	{
		rsx_log.trace("rsx::metal::argument_table::bind_buffer_address(index=%u, buffer=*0x%x, offset=0x%llx)",
			index, buf.handle(), offset);

		if (index >= m_impl->m_desc.max_buffers)
		{
			fmt::throw_exception("Metal argument table buffer binding out of range: index=%u, max=%u",
				index, m_impl->m_desc.max_buffers);
		}

		const u64 length = buf.length();
		if (offset > length)
		{
			fmt::throw_exception("Metal argument table buffer binding offset out of range: offset=0x%llx, length=0x%llx",
				offset, length);
		}

		const u64 address = buf.gpu_address();
		if (!address)
		{
			fmt::throw_exception("Metal argument table buffer binding requires a non-zero GPU address");
		}

		if (@available(macOS 26.0, *))
		{
			[m_impl->m_table setAddress:static_cast<MTLGPUAddress>(address + offset) atIndex:index];
		}
		else
		{
			fmt::throw_exception("Metal argument table buffer binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::bind_vertex_buffer_address(u32 index, const buffer& buf, u64 offset, u32 stride)
	{
		rsx_log.trace("rsx::metal::argument_table::bind_vertex_buffer_address(index=%u, buffer=*0x%x, offset=0x%llx, stride=0x%x)",
			index, buf.handle(), offset, stride);

		if (!m_impl->m_desc.support_attribute_strides)
		{
			fmt::throw_exception("Metal argument table was not created with attribute stride support");
		}

		if (!stride)
		{
			fmt::throw_exception("Metal argument table vertex buffer binding requires a non-zero stride");
		}

		if (index >= m_impl->m_desc.max_buffers)
		{
			fmt::throw_exception("Metal argument table vertex buffer binding out of range: index=%u, max=%u",
				index, m_impl->m_desc.max_buffers);
		}

		const u64 length = buf.length();
		if (offset > length)
		{
			fmt::throw_exception("Metal argument table vertex buffer offset out of range: offset=0x%llx, length=0x%llx",
				offset, length);
		}

		const u64 address = buf.gpu_address();
		if (!address)
		{
			fmt::throw_exception("Metal argument table vertex buffer binding requires a non-zero GPU address");
		}

		if (@available(macOS 26.0, *))
		{
			[m_impl->m_table setAddress:static_cast<MTLGPUAddress>(address + offset)
				attributeStride:stride
				atIndex:index];
		}
		else
		{
			fmt::throw_exception("Metal argument table vertex buffer binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::bind_texture(u32 index, const texture& tex)
	{
		rsx_log.trace("rsx::metal::argument_table::bind_texture(index=%u, texture=*0x%x)", index, tex.handle());

		if (index >= m_impl->m_desc.max_textures)
		{
			fmt::throw_exception("Metal argument table texture binding out of range: index=%u, max=%u",
				index, m_impl->m_desc.max_textures);
		}

		const u64 resource_id = tex.resource_id();
		if (!resource_id)
		{
			fmt::throw_exception("Metal argument table texture binding requires a non-zero resource ID");
		}

		if (@available(macOS 26.0, *))
		{
			[m_impl->m_table setTexture:make_resource_id(resource_id) atIndex:index];
		}
		else
		{
			fmt::throw_exception("Metal argument table texture binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::bind_sampler(u32 index, const sampler& sampler_state)
	{
		rsx_log.trace("rsx::metal::argument_table::bind_sampler(index=%u, sampler=*0x%x)", index, sampler_state.handle());

		if (index >= m_impl->m_desc.max_samplers)
		{
			fmt::throw_exception("Metal argument table sampler binding out of range: index=%u, max=%u",
				index, m_impl->m_desc.max_samplers);
		}

		const u64 resource_id = sampler_state.resource_id();
		if (!resource_id)
		{
			fmt::throw_exception("Metal argument table sampler binding requires a non-zero resource ID");
		}

		if (@available(macOS 26.0, *))
		{
			[m_impl->m_table setSamplerState:make_resource_id(resource_id) atIndex:index];
		}
		else
		{
			fmt::throw_exception("Metal argument table sampler binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::bind_to_render_encoder(void* render_encoder_handle, u32 stages) const
	{
		rsx_log.trace("rsx::metal::argument_table::bind_to_render_encoder(render_encoder_handle=*0x%x, stages=0x%x)",
			render_encoder_handle, stages);

		if (!render_encoder_handle)
		{
			fmt::throw_exception("Metal argument table render binding requires a valid render encoder");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)render_encoder_handle;
			[encoder setArgumentTable:m_impl->m_table atStages:make_render_stages(stages)];
		}
		else
		{
			fmt::throw_exception("Metal argument table render binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::bind_to_compute_encoder(void* compute_encoder_handle) const
	{
		rsx_log.trace("rsx::metal::argument_table::bind_to_compute_encoder(compute_encoder_handle=*0x%x)",
			compute_encoder_handle);

		if (!compute_encoder_handle)
		{
			fmt::throw_exception("Metal argument table compute binding requires a valid compute encoder");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTL4ComputeCommandEncoder> encoder = (__bridge id<MTL4ComputeCommandEncoder>)compute_encoder_handle;
			[encoder setArgumentTable:m_impl->m_table];
		}
		else
		{
			fmt::throw_exception("Metal argument table compute binding requires macOS 26.0 or newer");
		}
	}
}
