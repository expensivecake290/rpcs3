#include "stdafx.h"
#include "MTLDrawCommands.h"

#include "MTLBarrier.h"
#include "MTLBuffer.h"
#include "MTLCommandBuffer.h"
#include "MTLDevice.h"

#import <Metal/Metal.h>

#include <limits>

namespace
{
	MTLPrimitiveType make_primitive_type(rsx::metal::draw_primitive_type primitive)
	{
		rsx_log.trace("make_primitive_type(primitive=%u)", static_cast<u32>(primitive));

		switch (primitive)
		{
		case rsx::metal::draw_primitive_type::points:
			return MTLPrimitiveTypePoint;
		case rsx::metal::draw_primitive_type::lines:
			return MTLPrimitiveTypeLine;
		case rsx::metal::draw_primitive_type::line_strip:
			return MTLPrimitiveTypeLineStrip;
		case rsx::metal::draw_primitive_type::triangles:
			return MTLPrimitiveTypeTriangle;
		case rsx::metal::draw_primitive_type::triangle_strip:
			return MTLPrimitiveTypeTriangleStrip;
		}

		fmt::throw_exception("Unsupported Metal draw primitive type: %u", static_cast<u32>(primitive));
	}

	MTLIndexType make_index_type(rsx::metal::draw_index_type type)
	{
		rsx_log.trace("make_index_type(type=%u)", static_cast<u32>(type));

		switch (type)
		{
		case rsx::metal::draw_index_type::uint16:
			return MTLIndexTypeUInt16;
		case rsx::metal::draw_index_type::uint32:
			return MTLIndexTypeUInt32;
		}

		fmt::throw_exception("Unsupported Metal draw index type: %u", static_cast<u32>(type));
	}

	u32 get_index_type_size(rsx::metal::draw_index_type type)
	{
		rsx_log.trace("get_index_type_size(type=%u)", static_cast<u32>(type));

		switch (type)
		{
		case rsx::metal::draw_index_type::uint16:
			return sizeof(u16);
		case rsx::metal::draw_index_type::uint32:
			return sizeof(u32);
		}

		fmt::throw_exception("Unsupported Metal draw index type size request: %u", static_cast<u32>(type));
	}

	void validate_index_buffer(const rsx::metal::draw_index_buffer& index)
	{
		rsx_log.trace("validate_index_buffer(buffer=*0x%x, offset=0x%llx, length=0x%llx, index_count=%u, type=%u)",
			index.resource,
			index.offset,
			index.length,
			index.index_count,
			static_cast<u32>(index.type));

		if (!index.resource)
		{
			fmt::throw_exception("Metal indexed draw requires a valid index buffer");
		}

		if (!index.index_count)
		{
			fmt::throw_exception("Metal indexed draw requires a non-zero index count");
		}

		const u32 index_size = get_index_type_size(index.type);
		if (index.offset % index_size)
		{
			fmt::throw_exception("Metal indexed draw buffer offset is not aligned: offset=0x%llx, index_size=%u",
				index.offset,
				index_size);
		}

		if (index.length % index_size)
		{
			fmt::throw_exception("Metal indexed draw buffer length is not aligned: length=0x%llx, index_size=%u",
				index.length,
				index_size);
		}

		if (index.index_count > (std::numeric_limits<u64>::max() / index_size))
		{
			fmt::throw_exception("Metal indexed draw byte count overflow: index_count=%u, index_size=%u",
				index.index_count,
				index_size);
		}

		const u64 required_length = static_cast<u64>(index.index_count) * index_size;
		if (index.length < required_length)
		{
			fmt::throw_exception("Metal indexed draw buffer is shorter than the command requires: length=0x%llx, required=0x%llx",
				index.length,
				required_length);
		}

		const u64 buffer_length = index.resource->length();
		if (index.offset > buffer_length || index.length > buffer_length - index.offset)
		{
			fmt::throw_exception("Metal indexed draw buffer range is out of bounds: offset=0x%llx, length=0x%llx, buffer_length=0x%llx",
				index.offset,
				index.length,
				buffer_length);
		}
	}
}

namespace rsx::metal
{
	draw_primitive_type get_draw_primitive_type(rsx::primitive_type primitive)
	{
		rsx_log.trace("rsx::metal::get_draw_primitive_type(primitive=%u)", static_cast<u32>(primitive));

		switch (primitive)
		{
		case rsx::primitive_type::points:
			return draw_primitive_type::points;
		case rsx::primitive_type::lines:
			return draw_primitive_type::lines;
		case rsx::primitive_type::line_strip:
			return draw_primitive_type::line_strip;
		case rsx::primitive_type::triangles:
			return draw_primitive_type::triangles;
		case rsx::primitive_type::triangle_strip:
			return draw_primitive_type::triangle_strip;
		case rsx::primitive_type::line_loop:
		case rsx::primitive_type::triangle_fan:
		case rsx::primitive_type::quads:
		case rsx::primitive_type::quad_strip:
		case rsx::primitive_type::polygon:
			rsx_log.todo("Metal draw primitive %u requires a GPU-only expansion path before it can be encoded", static_cast<u32>(primitive));
			break;
		}

		fmt::throw_exception("Metal backend does not support RSX primitive type %u without GPU-side expansion",
			static_cast<u32>(primitive));
	}

	draw_index_type get_draw_index_type(rsx::index_array_type type)
	{
		rsx_log.trace("rsx::metal::get_draw_index_type(type=%u)", static_cast<u32>(type));

		switch (type)
		{
		case rsx::index_array_type::u16:
			return draw_index_type::uint16;
		case rsx::index_array_type::u32:
			return draw_index_type::uint32;
		}

		fmt::throw_exception("Unsupported Metal draw index array type: %u", static_cast<u32>(type));
	}

	void validate_prepared_draw_command(const prepared_draw_command& command)
	{
		rsx_log.trace("rsx::metal::validate_prepared_draw_command(primitive=%u, vertex_start=%u, vertex_count=%u, instance_count=%u, base_instance=%u, base_vertex=%lld, indexed=%u)",
			static_cast<u32>(command.primitive),
			command.vertex_start,
			command.vertex_count,
			command.instance_count,
			command.base_instance,
			command.base_vertex,
			static_cast<u32>(command.indexed));

		if (!command.instance_count)
		{
			fmt::throw_exception("Metal draw requires a non-zero instance count");
		}

		const u64 last_instance = static_cast<u64>(command.base_instance) + command.instance_count - 1;
		if (last_instance > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal draw instance range overflows: base_instance=%u, instance_count=%u",
				command.base_instance,
				command.instance_count);
		}

		if (command.indexed)
		{
			validate_index_buffer(command.index);

			if (command.base_vertex < static_cast<s64>(std::numeric_limits<NSInteger>::min()) ||
				command.base_vertex > static_cast<s64>(std::numeric_limits<NSInteger>::max()))
			{
				fmt::throw_exception("Metal indexed draw base vertex is outside NSInteger range: base_vertex=%lld",
					command.base_vertex);
			}
		}
		else
		{
			if (!command.vertex_count)
			{
				fmt::throw_exception("Metal non-indexed draw requires a non-zero vertex count");
			}

			const u64 last_vertex = static_cast<u64>(command.vertex_start) + command.vertex_count - 1;
			if (last_vertex > std::numeric_limits<u32>::max())
			{
				fmt::throw_exception("Metal draw vertex range overflows: vertex_start=%u, vertex_count=%u",
					command.vertex_start,
					command.vertex_count);
			}
		}
	}

	void encode_draw_command(command_frame& frame, void* render_encoder_handle, const prepared_draw_command& command)
	{
		rsx_log.trace("rsx::metal::encode_draw_command(frame_index=%u, render_encoder_handle=*0x%x, primitive=%u, vertex_count=%u, index_count=%u, instances=%u, indexed=%u)",
			frame.frame_index(),
			render_encoder_handle,
			static_cast<u32>(command.primitive),
			command.vertex_count,
			command.index.index_count,
			command.instance_count,
			static_cast<u32>(command.indexed));

		if (!render_encoder_handle)
		{
			fmt::throw_exception("Metal draw encoding requires a valid render encoder");
		}

		validate_prepared_draw_command(command);

		if (@available(macOS 26.0, *))
		{
			id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)render_encoder_handle;
			const MTLPrimitiveType primitive = make_primitive_type(command.primitive);

			if (command.indexed)
			{
				buffer& index_buffer = *command.index.resource;
				frame.track_object(index_buffer.handle());

				const heap_resource_usage usage = index_buffer.heap_resource_usage_info();
				if (usage.metal_device && usage.resource_handle)
				{
					usage.metal_device->track_heap_resource_use(frame, usage.resource_handle);
				}

				const resource_barrier barrier = frame.track_resource_usage(resource_usage
				{
					.resource_id = index_buffer.gpu_address(),
					.stage = resource_stage::render,
					.access = resource_access::read,
					.scope = resource_barrier_scope::buffers,
				});
				encode_consumer_barrier(render_encoder_handle, barrier);

				[encoder drawIndexedPrimitives:primitive
					indexCount:command.index.index_count
					indexType:make_index_type(command.index.type)
					indexBuffer:static_cast<MTLGPUAddress>(index_buffer.gpu_address() + command.index.offset)
					indexBufferLength:static_cast<NSUInteger>(command.index.length)
					instanceCount:command.instance_count
					baseVertex:static_cast<NSInteger>(command.base_vertex)
					baseInstance:command.base_instance];
				return;
			}

			[encoder drawPrimitives:primitive
				vertexStart:command.vertex_start
				vertexCount:command.vertex_count
				instanceCount:command.instance_count
				baseInstance:command.base_instance];
			return;
		}

		fmt::throw_exception("Metal draw encoding requires macOS 26.0 or newer");
	}
}
