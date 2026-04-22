#pragma once

#include "Emu/RSX/gcm_enums.h"
#include "util/types.hpp"

namespace rsx::metal
{
	class buffer;
	class command_frame;

	enum class draw_primitive_type : u8
	{
		points,
		lines,
		line_strip,
		triangles,
		triangle_strip,
	};

	enum class draw_index_type : u8
	{
		uint16,
		uint32,
	};

	struct draw_index_buffer
	{
		buffer* resource = nullptr;
		u64 offset = 0;
		u64 length = 0;
		u32 index_count = 0;
		draw_index_type type = draw_index_type::uint16;
	};

	struct prepared_draw_command
	{
		draw_primitive_type primitive = draw_primitive_type::triangles;
		u32 vertex_start = 0;
		u32 vertex_count = 0;
		u32 instance_count = 1;
		u32 base_instance = 0;
		s64 base_vertex = 0;
		draw_index_buffer index{};
		b8 indexed = false;
	};

	draw_primitive_type get_draw_primitive_type(rsx::primitive_type primitive);
	draw_index_type get_draw_index_type(rsx::index_array_type type);
	void validate_prepared_draw_command(const prepared_draw_command& command);
	void encode_draw_command(command_frame& frame, void* render_encoder_handle, const prepared_draw_command& command);
}
