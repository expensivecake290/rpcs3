#include "stdafx.h"
#include "MTLArgumentTable.h"

#include "MTLBarrier.h"
#include "MTLBuffer.h"
#include "MTLCommandBuffer.h"
#include "MTLDevice.h"
#include "MTLSampler.h"
#include "MTLShaderInterface.h"
#include "MTLTexture.h"
#include "MTLTextureViewPool.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <vector>

namespace
{
	constexpr u32 supported_render_stage_mask =
		static_cast<u32>(rsx::metal::argument_table_render_stage::vertex) |
		static_cast<u32>(rsx::metal::argument_table_render_stage::fragment) |
		static_cast<u32>(rsx::metal::argument_table_render_stage::tile) |
		static_cast<u32>(rsx::metal::argument_table_render_stage::object) |
		static_cast<u32>(rsx::metal::argument_table_render_stage::mesh);
	constexpr u32 traditional_render_stage_mask =
		static_cast<u32>(rsx::metal::argument_table_render_stage::vertex) |
		static_cast<u32>(rsx::metal::argument_table_render_stage::fragment) |
		static_cast<u32>(rsx::metal::argument_table_render_stage::tile);
	constexpr u32 mesh_render_stage_mask =
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

	template <typename T>
	u32 checked_binding_count(const std::vector<T>& resources, const char* resource_kind)
	{
		rsx_log.trace("checked_binding_count(resource_kind=%s, count=0x%llx)",
			resource_kind ? resource_kind : "<null>",
			static_cast<u64>(resources.size()));

		if (resources.size() > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal argument table %s binding count exceeds u32 range",
				resource_kind ? resource_kind : "resource");
		}

		return static_cast<u32>(resources.size());
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

	void track_object_once(rsx::metal::command_frame& frame, std::vector<void*>& tracked_objects, void* object_handle)
	{
		rsx_log.trace("track_object_once(frame_index=%u, object_handle=*0x%x)",
			frame.frame_index(), object_handle);

		if (!object_handle)
		{
			return;
		}

		for (void* tracked_object : tracked_objects)
		{
			if (tracked_object == object_handle)
			{
				return;
			}
		}

		tracked_objects.push_back(object_handle);
		frame.track_object(object_handle);
	}

	void track_heap_resource_use(
		rsx::metal::command_frame& frame,
		std::vector<void*>& tracked_heap_resources,
		const rsx::metal::heap_resource_usage& usage)
	{
		rsx_log.trace("track_heap_resource_use(frame_index=%u, resource_handle=*0x%x)",
			frame.frame_index(), usage.resource_handle);

		if (!usage.metal_device && !usage.resource_handle)
		{
			return;
		}

		if (!usage.metal_device || !usage.resource_handle)
		{
			fmt::throw_exception("Metal argument table heap resource tracking received incomplete heap usage state");
		}

		for (void* tracked_heap_resource : tracked_heap_resources)
		{
			if (tracked_heap_resource == usage.resource_handle)
			{
				return;
			}
		}

		tracked_heap_resources.push_back(usage.resource_handle);
		usage.metal_device->track_heap_resource_use(frame, usage.resource_handle);
	}

	rsx::metal::resource_stage binding_resource_stage_for_render_stages(u32 stages)
	{
		rsx_log.trace("binding_resource_stage_for_render_stages(stages=0x%x)", stages);

		if (!stages)
		{
			fmt::throw_exception("Metal argument table resource tracking requires at least one render stage");
		}

		if (stages & ~supported_render_stage_mask)
		{
			fmt::throw_exception("Metal argument table resource tracking has unsupported render stages: 0x%x", stages);
		}

		const b8 uses_mesh_stages = !!(stages & mesh_render_stage_mask);
		const b8 uses_traditional_stages = !!(stages & traditional_render_stage_mask);
		if (uses_mesh_stages && uses_traditional_stages)
		{
			fmt::throw_exception("Metal argument table resource tracking requires separate bindings for mesh/object stages and traditional render stages");
		}

		return uses_mesh_stages ? rsx::metal::resource_stage::mesh : rsx::metal::resource_stage::render;
	}

	void validate_argument_table_resource_access(rsx::metal::resource_access access, const char* resource_kind)
	{
		rsx_log.trace("validate_argument_table_resource_access(access=%u, resource_kind=%s)",
			static_cast<u32>(access),
			resource_kind ? resource_kind : "<null>");

		switch (access)
		{
		case rsx::metal::resource_access::read:
		case rsx::metal::resource_access::write:
			return;
		}

		fmt::throw_exception("Metal argument table %s binding received an invalid resource access mode: %u",
			resource_kind ? resource_kind : "resource",
			static_cast<u32>(access));
	}

	void track_bound_resource_usage(
		rsx::metal::command_frame& frame,
		void* encoder_handle,
		u64 resource_id,
		rsx::metal::resource_stage stage,
		rsx::metal::resource_access access,
		rsx::metal::resource_barrier_scope scope)
	{
		rsx_log.trace("track_bound_resource_usage(frame_index=%u, encoder_handle=*0x%x, resource_id=0x%llx, stage=%s, access=%s, scope=%s)",
			frame.frame_index(),
			encoder_handle,
			resource_id,
			rsx::metal::describe_resource_stage(stage),
			rsx::metal::describe_resource_access(access),
			rsx::metal::describe_resource_barrier_scope(scope));

		if (!resource_id)
		{
			return;
		}

		const rsx::metal::resource_barrier barrier = frame.track_resource_usage(rsx::metal::resource_usage
		{
			.resource_id = resource_id,
			.stage = stage,
			.access = access,
			.scope = scope
		});

		rsx::metal::encode_consumer_barrier(encoder_handle, barrier);
	}
}

namespace rsx::metal
{
	struct argument_table_use_state
	{
		std::mutex m_mutex;
		u32 m_in_flight_use_count = 0;
	};

	struct argument_table_bound_resource
	{
		void* m_object_handle = nullptr;
		heap_resource_usage m_heap_usage{};
		u64 m_resource_id = 0;
		u32 m_slot = 0;
		resource_access m_access = resource_access::read;
	};

	struct argument_table::argument_table_impl
	{
		id<MTL4ArgumentTable> m_table = nil;
		std::unique_ptr<texture_view_pool> m_texture_views;
		std::shared_ptr<argument_table_use_state> m_use_state = std::make_shared<argument_table_use_state>();
		std::vector<argument_table_bound_resource> m_bound_buffers;
		std::vector<argument_table_bound_resource> m_bound_textures;
		std::vector<void*> m_bound_samplers;
		argument_table_desc m_desc{};
	};

	void validate_table_mutable(const argument_table_use_state& use_state, const char* operation)
	{
		rsx_log.trace("validate_table_mutable(operation=%s, in_flight_use_count=%u)",
			operation, use_state.m_in_flight_use_count);

		if (use_state.m_in_flight_use_count)
		{
			fmt::throw_exception("Metal argument table cannot %s while %u frame uses are pending",
				operation, use_state.m_in_flight_use_count);
		}
	}

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

		m_impl->m_bound_buffers.resize(desc.max_buffers);
		m_impl->m_bound_textures.resize(desc.max_textures);
		m_impl->m_bound_samplers.resize(desc.max_samplers);

		if (desc.max_textures)
		{
			m_impl->m_texture_views = std::make_unique<texture_view_pool>(dev, desc.max_textures,
				desc.label.empty() ? "RPCS3 Metal argument texture views" : desc.label + " texture views");
		}
	}

	argument_table::~argument_table()
	{
		rsx_log.notice("rsx::metal::argument_table::~argument_table(table=*0x%x)", handle());

		std::lock_guard lock(m_impl->m_use_state->m_mutex);
		if (m_impl->m_use_state->m_in_flight_use_count)
		{
			rsx_log.error("Metal argument table destroyed while %u frame uses are pending",
				m_impl->m_use_state->m_in_flight_use_count);
		}
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

	b8 argument_table::is_in_flight() const
	{
		rsx_log.trace("rsx::metal::argument_table::is_in_flight(table=*0x%x)", handle());

		std::lock_guard lock(m_impl->m_use_state->m_mutex);
		return !!m_impl->m_use_state->m_in_flight_use_count;
	}

	void argument_table::begin_update()
	{
		rsx_log.trace("rsx::metal::argument_table::begin_update(table=*0x%x)", handle());

		std::lock_guard lock(m_impl->m_use_state->m_mutex);
		validate_table_mutable(*m_impl->m_use_state, "begin a binding update");
		std::fill(m_impl->m_bound_buffers.begin(), m_impl->m_bound_buffers.end(), argument_table_bound_resource{});
		std::fill(m_impl->m_bound_textures.begin(), m_impl->m_bound_textures.end(), argument_table_bound_resource{});
		std::fill(m_impl->m_bound_samplers.begin(), m_impl->m_bound_samplers.end(), nullptr);
	}

	void argument_table::bind_buffer_address(u32 index, const buffer& buf, u64 offset, resource_access access)
	{
		rsx_log.trace("rsx::metal::argument_table::bind_buffer_address(index=%u, buffer=*0x%x, offset=0x%llx, access=%s)",
			index, buf.handle(), offset, describe_resource_access(access));

		std::lock_guard lock(m_impl->m_use_state->m_mutex);
		validate_table_mutable(*m_impl->m_use_state, "bind a buffer");
		validate_argument_table_resource_access(access, "buffer");

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
			m_impl->m_bound_buffers[index] =
			{
				.m_object_handle = buf.handle(),
				.m_heap_usage = buf.heap_resource_usage_info(),
				.m_resource_id = address,
				.m_slot = index,
				.m_access = access,
			};
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

		std::lock_guard lock(m_impl->m_use_state->m_mutex);
		validate_table_mutable(*m_impl->m_use_state, "bind a vertex buffer");

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
			m_impl->m_bound_buffers[index] =
			{
				.m_object_handle = buf.handle(),
				.m_heap_usage = buf.heap_resource_usage_info(),
				.m_resource_id = address,
				.m_slot = index,
				.m_access = resource_access::read,
			};
		}
		else
		{
			fmt::throw_exception("Metal argument table vertex buffer binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::bind_texture(u32 index, const texture& tex, resource_access access)
	{
		rsx_log.trace("rsx::metal::argument_table::bind_texture(index=%u, texture=*0x%x, access=%s)",
			index, tex.handle(), describe_resource_access(access));

		std::lock_guard lock(m_impl->m_use_state->m_mutex);
		validate_table_mutable(*m_impl->m_use_state, "bind a texture");
		validate_argument_table_resource_access(access, "texture");

		if (index >= m_impl->m_desc.max_textures)
		{
			fmt::throw_exception("Metal argument table texture binding out of range: index=%u, max=%u",
				index, m_impl->m_desc.max_textures);
		}

		if (!m_impl->m_texture_views)
		{
			fmt::throw_exception("Metal argument table texture binding requires a texture view pool");
		}

		const texture_view_binding view = m_impl->m_texture_views->set_default_texture_view(index, tex);
		if (!view.view_resource_id || !view.source_resource_id)
		{
			fmt::throw_exception("Metal argument table texture binding requires a non-zero resource ID");
		}

		if (@available(macOS 26.0, *))
		{
			[m_impl->m_table setTexture:make_resource_id(view.view_resource_id) atIndex:index];
			m_impl->m_bound_textures[index] =
			{
				.m_object_handle = tex.allocation_handle(),
				.m_heap_usage = tex.heap_resource_usage_info(),
				.m_resource_id = view.source_resource_id,
				.m_slot = index,
				.m_access = access,
			};

			if (!m_impl->m_bound_textures[index].m_resource_id)
			{
				fmt::throw_exception("Metal argument table texture binding requires a non-zero underlying texture resource ID");
			}
		}
		else
		{
			fmt::throw_exception("Metal argument table texture binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::bind_sampler(u32 index, const sampler& sampler_state)
	{
		rsx_log.trace("rsx::metal::argument_table::bind_sampler(index=%u, sampler=*0x%x)", index, sampler_state.handle());

		std::lock_guard lock(m_impl->m_use_state->m_mutex);
		validate_table_mutable(*m_impl->m_use_state, "bind a sampler");

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
			m_impl->m_bound_samplers[index] = sampler_state.handle();
		}
		else
		{
			fmt::throw_exception("Metal argument table sampler binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::validate_shader_layout(const shader_interface_layout& layout) const
	{
		rsx_log.trace("rsx::metal::argument_table::validate_shader_layout(table=*0x%x, layout_stage=%u)",
			handle(), static_cast<u32>(layout.stage));

		std::lock_guard lock(m_impl->m_use_state->m_mutex);
		validate_shader_layout_locked(layout);
	}

	void argument_table::validate_shader_bindings(const shader_interface_layout& layout) const
	{
		rsx_log.trace("rsx::metal::argument_table::validate_shader_bindings(table=*0x%x, layout_stage=%u)",
			handle(), static_cast<u32>(layout.stage));

		std::lock_guard lock(m_impl->m_use_state->m_mutex);
		validate_shader_bindings_locked(layout);
	}

	void argument_table::bind_to_render_encoder(command_frame& frame, void* render_encoder_handle, const shader_interface_layout& layout) const
	{
		rsx_log.trace("rsx::metal::argument_table::bind_to_render_encoder(frame_index=%u, render_encoder_handle=*0x%x, layout_stage=%u, stages=0x%x)",
			frame.frame_index(), render_encoder_handle, static_cast<u32>(layout.stage), layout.render_stage_mask);

		if (!render_encoder_handle)
		{
			fmt::throw_exception("Metal argument table render binding requires a valid render encoder");
		}

		if (@available(macOS 26.0, *))
		{
			std::lock_guard lock(m_impl->m_use_state->m_mutex);

			validate_shader_bindings_locked(layout);

			id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)render_encoder_handle;
			track_bound_resources(frame, render_encoder_handle, binding_resource_stage_for_render_stages(layout.render_stage_mask));
			[encoder setArgumentTable:m_impl->m_table atStages:make_render_stages(layout.render_stage_mask)];
			retain_bound_table_locked(frame);
		}
		else
		{
			fmt::throw_exception("Metal argument table render binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::bind_to_render_encoder(command_frame& frame, void* render_encoder_handle, u32 stages) const
	{
		rsx_log.trace("rsx::metal::argument_table::bind_to_render_encoder(frame_index=%u, render_encoder_handle=*0x%x, stages=0x%x)",
			frame.frame_index(), render_encoder_handle, stages);

		if (!render_encoder_handle)
		{
			fmt::throw_exception("Metal argument table render binding requires a valid render encoder");
		}

		if (@available(macOS 26.0, *))
		{
			std::lock_guard lock(m_impl->m_use_state->m_mutex);

			id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)render_encoder_handle;
			track_bound_resources(frame, render_encoder_handle, binding_resource_stage_for_render_stages(stages));
			[encoder setArgumentTable:m_impl->m_table atStages:make_render_stages(stages)];
			retain_bound_table_locked(frame);
		}
		else
		{
			fmt::throw_exception("Metal argument table render binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::bind_to_compute_encoder(command_frame& frame, void* compute_encoder_handle) const
	{
		rsx_log.trace("rsx::metal::argument_table::bind_to_compute_encoder(frame_index=%u, compute_encoder_handle=*0x%x)",
			frame.frame_index(), compute_encoder_handle);

		if (!compute_encoder_handle)
		{
			fmt::throw_exception("Metal argument table compute binding requires a valid compute encoder");
		}

		if (@available(macOS 26.0, *))
		{
			std::lock_guard lock(m_impl->m_use_state->m_mutex);

			id<MTL4ComputeCommandEncoder> encoder = (__bridge id<MTL4ComputeCommandEncoder>)compute_encoder_handle;
			track_bound_resources(frame, compute_encoder_handle, resource_stage::compute);
			[encoder setArgumentTable:m_impl->m_table];
			retain_bound_table_locked(frame);
		}
		else
		{
			fmt::throw_exception("Metal argument table compute binding requires macOS 26.0 or newer");
		}
	}

	void argument_table::validate_shader_layout_locked(const shader_interface_layout& layout) const
	{
		rsx_log.trace("rsx::metal::argument_table::validate_shader_layout_locked(table=*0x%x, layout_stage=%u, stages=0x%x)",
			handle(), static_cast<u32>(layout.stage), layout.render_stage_mask);

		validate_shader_interface_layout(layout);

		const argument_table_desc& expected = layout.argument_table;
		if (m_impl->m_desc.max_buffers != expected.max_buffers ||
			m_impl->m_desc.max_textures != expected.max_textures ||
			m_impl->m_desc.max_samplers != expected.max_samplers)
		{
			fmt::throw_exception("Metal argument table descriptor mismatch for shader stage %u: table buffers=%u textures=%u samplers=%u, layout buffers=%u textures=%u samplers=%u",
				static_cast<u32>(layout.stage),
				m_impl->m_desc.max_buffers,
				m_impl->m_desc.max_textures,
				m_impl->m_desc.max_samplers,
				expected.max_buffers,
				expected.max_textures,
				expected.max_samplers);
		}

		if (m_impl->m_desc.support_attribute_strides != expected.support_attribute_strides)
		{
			fmt::throw_exception("Metal argument table attribute-stride support mismatch for shader stage %u", static_cast<u32>(layout.stage));
		}
	}

	void argument_table::validate_shader_bindings_locked(const shader_interface_layout& layout) const
	{
		rsx_log.trace("rsx::metal::argument_table::validate_shader_bindings_locked(table=*0x%x, layout_stage=%u, stages=0x%x)",
			handle(), static_cast<u32>(layout.stage), layout.render_stage_mask);

		validate_shader_layout_locked(layout);

		const auto validate_buffer_slot = [this, &layout](u32 index, const char* role)
		{
			if (index >= m_impl->m_bound_buffers.size())
			{
				fmt::throw_exception("Metal shader stage %u %s buffer slot is out of range: index=%u, buffers=%u",
					static_cast<u32>(layout.stage),
					role,
					index,
					static_cast<u32>(m_impl->m_bound_buffers.size()));
			}

			const argument_table_bound_resource& binding = m_impl->m_bound_buffers[index];
			if (!binding.m_resource_id || !binding.m_object_handle)
			{
				fmt::throw_exception("Metal shader stage %u requires %s buffer slot %u to be bound before argument table submission",
					static_cast<u32>(layout.stage),
					role,
					index);
			}
		};

		const auto validate_texture_slot = [this, &layout](u32 index)
		{
			if (index >= m_impl->m_bound_textures.size())
			{
				fmt::throw_exception("Metal shader stage %u texture slot is out of range: index=%u, textures=%u",
					static_cast<u32>(layout.stage),
					index,
					static_cast<u32>(m_impl->m_bound_textures.size()));
			}

			const argument_table_bound_resource& binding = m_impl->m_bound_textures[index];
			if (!binding.m_resource_id || !binding.m_object_handle)
			{
				fmt::throw_exception("Metal shader stage %u requires texture slot %u to be bound before argument table submission",
					static_cast<u32>(layout.stage),
					index);
			}
		};

		const auto validate_sampler_slot = [this, &layout](u32 index)
		{
			if (index >= m_impl->m_bound_samplers.size())
			{
				fmt::throw_exception("Metal shader stage %u sampler slot is out of range: index=%u, samplers=%u",
					static_cast<u32>(layout.stage),
					index,
					static_cast<u32>(m_impl->m_bound_samplers.size()));
			}

			if (!m_impl->m_bound_samplers[index])
			{
				fmt::throw_exception("Metal shader stage %u requires sampler slot %u to be bound before argument table submission",
					static_cast<u32>(layout.stage),
					index);
			}
		};

		if (layout.context_buffer_index != shader_binding_none)
		{
			validate_buffer_slot(layout.context_buffer_index, "context");
		}

		if (layout.constants_buffer_index != shader_binding_none)
		{
			validate_buffer_slot(layout.constants_buffer_index, "constants");
		}

		if (layout.vertex_layout_buffer_index != shader_binding_none)
		{
			validate_buffer_slot(layout.vertex_layout_buffer_index, "vertex layout");
		}

		if (layout.persistent_vertex_buffer_index != shader_binding_none)
		{
			validate_buffer_slot(layout.persistent_vertex_buffer_index, "persistent vertex stream");
		}

		if (layout.volatile_vertex_buffer_index != shader_binding_none)
		{
			validate_buffer_slot(layout.volatile_vertex_buffer_index, "volatile vertex stream");
		}

		for (u32 index = 0; index < layout.vertex_buffer_count; index++)
		{
			validate_buffer_slot(layout.vertex_buffer_base_index + index, "vertex input");
		}

		for (u32 index = 0; index < layout.texture_count; index++)
		{
			validate_texture_slot(layout.texture_base_index + index);
		}

		for (u32 index = 0; index < layout.sampler_count; index++)
		{
			validate_sampler_slot(layout.sampler_base_index + index);
		}
	}

	void argument_table::validate_bound_resource_conflicts(resource_stage stage) const
	{
		rsx_log.trace("rsx::metal::argument_table::validate_bound_resource_conflicts(table=*0x%x, stage=%s)",
			handle(),
			describe_resource_stage(stage));

		const auto validate_resources = [stage](const std::vector<argument_table_bound_resource>& resources, const char* resource_kind)
		{
			const u32 resource_count = checked_binding_count(resources, resource_kind);
			for (u32 lhs_index = 0; lhs_index < resource_count; lhs_index++)
			{
				const argument_table_bound_resource& lhs = resources[lhs_index];
				if (!lhs.m_resource_id)
				{
					continue;
				}

				for (u32 rhs_index = lhs_index + 1; rhs_index < resource_count; rhs_index++)
				{
					const argument_table_bound_resource& rhs = resources[rhs_index];
					if (lhs.m_resource_id != rhs.m_resource_id)
					{
						continue;
					}

					if (lhs.m_access == resource_access::read && rhs.m_access == resource_access::read)
					{
						continue;
					}

					fmt::throw_exception("Metal argument table %s resource_id=0x%llx has conflicting access in slots %u and %u for %s stage",
						resource_kind,
						lhs.m_resource_id,
						lhs.m_slot,
						rhs.m_slot,
						describe_resource_stage(stage));
				}
			}
		};

		validate_resources(m_impl->m_bound_buffers, "buffer");
		validate_resources(m_impl->m_bound_textures, "texture");
	}

	void argument_table::track_bound_resources(command_frame& frame, void* encoder_handle, resource_stage stage) const
	{
		rsx_log.trace("rsx::metal::argument_table::track_bound_resources(frame_index=%u, encoder_handle=*0x%x, table=*0x%x, stage=%s)",
			frame.frame_index(),
			encoder_handle,
			handle(),
			describe_resource_stage(stage));

		validate_bound_resource_conflicts(stage);

		std::vector<void*> tracked_objects;
		std::vector<void*> tracked_heap_resources;
		tracked_objects.reserve(1 + m_impl->m_bound_buffers.size() + m_impl->m_bound_textures.size() + m_impl->m_bound_samplers.size());
		tracked_heap_resources.reserve(m_impl->m_bound_buffers.size() + m_impl->m_bound_textures.size());

		const auto track_bound_resource = [&frame, encoder_handle, stage, &tracked_objects, &tracked_heap_resources](
			const argument_table_bound_resource& resource,
			const char* resource_kind,
			resource_barrier_scope scope)
		{
			if (!resource.m_resource_id)
			{
				return;
			}

			if (!resource.m_object_handle)
			{
				fmt::throw_exception("Metal argument table %s slot %u has a resource ID without a lifetime-tracked object",
					resource_kind,
					resource.m_slot);
			}

			track_object_once(frame, tracked_objects, resource.m_object_handle);
			track_heap_resource_use(frame, tracked_heap_resources, resource.m_heap_usage);
			track_bound_resource_usage(frame, encoder_handle, resource.m_resource_id, stage, resource.m_access, scope);
		};

		track_object_once(frame, tracked_objects, handle());

		if (m_impl->m_texture_views)
		{
			track_object_once(frame, tracked_objects, m_impl->m_texture_views->handle());
		}

		for (const argument_table_bound_resource& buffer : m_impl->m_bound_buffers)
		{
			track_bound_resource(buffer, "buffer", resource_barrier_scope::buffers);
		}

		for (const argument_table_bound_resource& texture : m_impl->m_bound_textures)
		{
			track_bound_resource(texture, "texture", resource_barrier_scope::textures);
		}

		for (void* sampler_handle : m_impl->m_bound_samplers)
		{
			track_object_once(frame, tracked_objects, sampler_handle);
		}
	}

	void argument_table::retain_bound_table_locked(command_frame& frame) const
	{
		rsx_log.trace("rsx::metal::argument_table::retain_bound_table_locked(frame_index=%u, table=*0x%x)",
			frame.frame_index(), handle());

		std::shared_ptr<argument_table_use_state> use_state = m_impl->m_use_state;
		frame.on_completed([use_state]()
		{
			std::lock_guard lock(use_state->m_mutex);
			ensure(use_state->m_in_flight_use_count);
			use_state->m_in_flight_use_count--;
		});

		if (use_state->m_in_flight_use_count == std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal argument table in-flight use counter overflow for table=*0x%x", handle());
		}

		use_state->m_in_flight_use_count++;
	}

	struct argument_table_pool::argument_table_pool_impl
	{
		device& m_device;
		argument_table_desc m_desc{};
		std::vector<std::unique_ptr<argument_table>> m_tables;
		mutable std::mutex m_mutex;

		argument_table_pool_impl(device& dev, argument_table_desc desc)
			: m_device(dev)
			, m_desc(std::move(desc))
		{
		}
	};

	argument_table_pool::argument_table_pool(device& dev, argument_table_desc desc)
		: m_impl(std::make_unique<argument_table_pool_impl>(dev, std::move(desc)))
	{
		rsx_log.notice("rsx::metal::argument_table_pool::argument_table_pool(device=*0x%x, buffers=%u, textures=%u, samplers=%u)",
			dev.handle(),
			m_impl->m_desc.max_buffers,
			m_impl->m_desc.max_textures,
			m_impl->m_desc.max_samplers);
	}

	argument_table_pool::~argument_table_pool()
	{
		rsx_log.notice("rsx::metal::argument_table_pool::~argument_table_pool(retained=%u)", retained_table_count());
	}

	argument_table& argument_table_pool::acquire()
	{
		rsx_log.trace("rsx::metal::argument_table_pool::acquire(retained=%u)", retained_table_count());

		std::lock_guard lock(m_impl->m_mutex);
		for (std::unique_ptr<argument_table>& table : m_impl->m_tables)
		{
			if (table->is_in_flight())
			{
				continue;
			}

			table->begin_update();
			return *table;
		}

		auto table = std::make_unique<argument_table>(m_impl->m_device, m_impl->m_desc);
		argument_table& result = *table;
		result.begin_update();
		m_impl->m_tables.emplace_back(std::move(table));
		return result;
	}

	u32 argument_table_pool::retained_table_count() const
	{
		rsx_log.trace("rsx::metal::argument_table_pool::retained_table_count()");

		std::lock_guard lock(m_impl->m_mutex);
		if (m_impl->m_tables.size() > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal argument table pool retained table count overflow");
		}

		return static_cast<u32>(m_impl->m_tables.size());
	}
}
