#pragma once

#include "descriptors.h"

namespace mtl
{
	inline constexpr u8 invalid_binding_index = 0xff;
	inline constexpr u32 fragment_texture_unit_count = 16;
	inline constexpr u32 vertex_texture_unit_count = 4;

	enum class argument_binding_class : u8
	{
		buffer,
		texture,
		sampler,
	};

	struct shader_binding_location
	{
		argument_binding_class type = argument_binding_class::buffer;
		u8 index = invalid_binding_index;
		u8 stages = argument_stage_none;

		[[nodiscard]] constexpr explicit operator bool() const
		{
			return index != invalid_binding_index && stages != argument_stage_none;
		}

		[[nodiscard]] constexpr bool operator==(const shader_binding_location&) const = default;
	};

	struct vertex_stage_binding_table
	{
		static constexpr u8 persistent_vertex_buffer = 0;
		static constexpr u8 volatile_vertex_buffer = 1;
		static constexpr u8 draw_parameters_buffer = 2;
		static constexpr u8 context_buffer = 3;
		static constexpr u8 conditional_render_predicate_buffer = 4;
		static constexpr u8 constants_buffer = 5;
		static constexpr u8 instancing_lookup_buffer = 6;
		static constexpr u8 instancing_constants_buffer = 7;
		static constexpr u8 sampler_state_buffer = 8;
		static constexpr u8 line_mapping_buffer = 10;
		static constexpr u8 buffer_count = 11;

		static constexpr u8 textures_first = 0;
		static constexpr u8 samplers_first = 0;
		static constexpr u8 texture_count = vertex_texture_unit_count;
		static constexpr u8 sampler_count = vertex_texture_unit_count;

		[[nodiscard]] static constexpr shader_binding_location buffer(u8 index)
		{
			return {argument_binding_class::buffer, index, argument_stage_vertex};
		}

		[[nodiscard]] static constexpr shader_binding_location texture(u32 unit)
		{
			return unit < texture_count ? shader_binding_location{
				argument_binding_class::texture, static_cast<u8>(textures_first + unit), argument_stage_vertex} :
				shader_binding_location{};
		}

		[[nodiscard]] static constexpr shader_binding_location sampler(u32 unit)
		{
			return unit < sampler_count ? shader_binding_location{
				argument_binding_class::sampler, static_cast<u8>(samplers_first + unit), argument_stage_vertex} :
				shader_binding_location{};
		}

		[[nodiscard]] static constexpr argument_table_layout layout()
		{
			return {buffer_count, texture_count, sampler_count, true};
		}
	};

	struct fragment_stage_binding_table
	{
		static constexpr u8 state_buffer = 0;
		static constexpr u8 constants_buffer = 1;
		static constexpr u8 texture_parameters_buffer = 2;
		static constexpr u8 rasterizer_environment_buffer = 3;
		static constexpr u8 sampler_state_buffer = 4;
		static constexpr u8 buffer_count = 5;

		static constexpr u8 textures_first = 0;
		static constexpr u8 stencil_textures_first = textures_first + fragment_texture_unit_count;
		static constexpr u8 depth_input_texture = stencil_textures_first + fragment_texture_unit_count;
		static constexpr u8 texture_count = depth_input_texture + 1;
		static constexpr u8 samplers_first = 0;
		static constexpr u8 sampler_count = fragment_texture_unit_count;

		[[nodiscard]] static constexpr shader_binding_location buffer(u8 index)
		{
			return {argument_binding_class::buffer, index, argument_stage_fragment};
		}

		[[nodiscard]] static constexpr shader_binding_location texture(u32 unit)
		{
			return unit < fragment_texture_unit_count ? shader_binding_location{
				argument_binding_class::texture, static_cast<u8>(textures_first + unit), argument_stage_fragment} :
				shader_binding_location{};
		}

		[[nodiscard]] static constexpr shader_binding_location stencil_texture(u32 unit)
		{
			return unit < fragment_texture_unit_count ? shader_binding_location{
				argument_binding_class::texture, static_cast<u8>(stencil_textures_first + unit), argument_stage_fragment} :
				shader_binding_location{};
		}

		[[nodiscard]] static constexpr shader_binding_location sampler(u32 unit)
		{
			return unit < fragment_texture_unit_count ? shader_binding_location{
				argument_binding_class::sampler, static_cast<u8>(samplers_first + unit), argument_stage_fragment} :
				shader_binding_location{};
		}

		[[nodiscard]] static constexpr argument_table_layout layout()
		{
			return {buffer_count, texture_count, sampler_count, false};
		}
	};

	struct pipeline_binding_table
	{
		vertex_stage_binding_table vertex;
		fragment_stage_binding_table fragment;

		[[nodiscard]] static constexpr shader_binding_location vertex_params()
		{
			return vertex_stage_binding_table::buffer(vertex_stage_binding_table::context_buffer);
		}

		[[nodiscard]] static constexpr shader_binding_location vertex_constants()
		{
			return vertex_stage_binding_table::buffer(vertex_stage_binding_table::constants_buffer);
		}

		[[nodiscard]] static constexpr shader_binding_location fragment_constants()
		{
			return fragment_stage_binding_table::buffer(fragment_stage_binding_table::constants_buffer);
		}

		[[nodiscard]] static constexpr shader_binding_location fragment_state()
		{
			return fragment_stage_binding_table::buffer(fragment_stage_binding_table::state_buffer);
		}

		[[nodiscard]] static constexpr shader_binding_location fragment_texture_parameters()
		{
			return fragment_stage_binding_table::buffer(fragment_stage_binding_table::texture_parameters_buffer);
		}

		[[nodiscard]] static constexpr u64 signature()
		{
			constexpr u64 vertex_signature = vertex_stage_binding_table::buffer_count |
				(static_cast<u64>(vertex_stage_binding_table::texture_count) << 8) |
				(static_cast<u64>(vertex_stage_binding_table::sampler_count) << 16) |
				(1ull << 24);
			constexpr u64 fragment_signature = fragment_stage_binding_table::buffer_count |
				(static_cast<u64>(fragment_stage_binding_table::texture_count) << 8) |
				(static_cast<u64>(fragment_stage_binding_table::sampler_count) << 16);
			return vertex_signature | (fragment_signature << 32);
		}
	};

	static_assert(vertex_stage_binding_table::buffer_count <= maximum_argument_buffers);
	static_assert(vertex_stage_binding_table::texture_count <= maximum_argument_textures);
	static_assert(vertex_stage_binding_table::sampler_count <= maximum_argument_samplers);
	static_assert(fragment_stage_binding_table::buffer_count <= maximum_argument_buffers);
	static_assert(fragment_stage_binding_table::texture_count <= maximum_argument_textures);
	static_assert(fragment_stage_binding_table::sampler_count <= maximum_argument_samplers);
}
