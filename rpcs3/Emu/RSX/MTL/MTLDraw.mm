#include "stdafx.h"
#include "MTLGSRender.h"

#include "Emu/RSX/Common/BufferUtils.h"
#include "Emu/RSX/Program/GLSLCommon.h"
#include "Emu/RSX/rsx_methods.h"
#include "mtlutils/image_helpers.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <span>

namespace
{
	using native_render_encoder = id<MTL4RenderCommandEncoder>;

	mtl::texture_type texture_type(rsx::texture_dimension_extended dimension)
	{
		switch (dimension)
		{
		case rsx::texture_dimension_extended::texture_dimension_1d:
			return mtl::texture_type::texture_2d;
		case rsx::texture_dimension_extended::texture_dimension_2d:
			return mtl::texture_type::texture_2d;
		case rsx::texture_dimension_extended::texture_dimension_cubemap:
			return mtl::texture_type::texture_cube;
		case rsx::texture_dimension_extended::texture_dimension_3d:
			return mtl::texture_type::texture_3d;
		}
		fmt::throw_exception("Invalid RSX texture dimension 0x%x", static_cast<u32>(dimension));
	}

	u32 interpreter_texture_offset(rsx::texture_dimension_extended dimension)
	{
		switch (dimension)
		{
		case rsx::texture_dimension_extended::texture_dimension_1d: return 0;
		case rsx::texture_dimension_extended::texture_dimension_2d: return 16;
		case rsx::texture_dimension_extended::texture_dimension_cubemap: return 32;
		case rsx::texture_dimension_extended::texture_dimension_3d: return 48;
		}
		fmt::throw_exception("Invalid RSX interpreter texture dimension 0x%x",
			static_cast<u32>(dimension));
	}

	MTLPrimitiveType primitive_type(mtl::primitive_topology topology)
	{
		switch (topology)
		{
		case mtl::primitive_topology::point: return MTLPrimitiveTypePoint;
		case mtl::primitive_topology::line: return MTLPrimitiveTypeLine;
		case mtl::primitive_topology::line_strip: return MTLPrimitiveTypeLineStrip;
		case mtl::primitive_topology::triangle: return MTLPrimitiveTypeTriangle;
		case mtl::primitive_topology::triangle_strip: return MTLPrimitiveTypeTriangleStrip;
		}
		fmt::throw_exception("Invalid Metal primitive topology 0x%x", static_cast<u32>(topology));
	}

	MTLIndexType index_type(mtl::index_element_type type)
	{
		return type == mtl::index_element_type::u16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
	}

	u32 index_stride(mtl::index_element_type type)
	{
		return type == mtl::index_element_type::u16 ? sizeof(u16) : sizeof(u32);
	}

	u64 stencil_view_format(u64 format)
	{
		switch (static_cast<MTLPixelFormat>(format))
		{
		case MTLPixelFormatDepth32Float_Stencil8:
		case MTLPixelFormatX32_Stencil8:
			return MTLPixelFormatX32_Stencil8;
		case MTLPixelFormatDepth24Unorm_Stencil8:
		case MTLPixelFormatX24_Stencil8:
			return MTLPixelFormatX24_Stencil8;
		case MTLPixelFormatStencil8:
			return MTLPixelFormatStencil8;
		default:
			fmt::throw_exception("Metal texture format 0x%llx has no stencil sampling view", format);
		}
	}

	mtl::sampler_compare_function sampler_compare(rsx::comparison_function operation,
		bool reverse_direction)
	{
		switch (operation)
		{
		case rsx::comparison_function::never: return mtl::sampler_compare_function::never;
		case rsx::comparison_function::greater:
			return reverse_direction ? mtl::sampler_compare_function::less :
				mtl::sampler_compare_function::greater;
		case rsx::comparison_function::less:
			return reverse_direction ? mtl::sampler_compare_function::greater :
				mtl::sampler_compare_function::less;
		case rsx::comparison_function::less_or_equal:
			return reverse_direction ? mtl::sampler_compare_function::greater_equal :
				mtl::sampler_compare_function::less_equal;
		case rsx::comparison_function::greater_or_equal:
			return reverse_direction ? mtl::sampler_compare_function::less_equal :
				mtl::sampler_compare_function::greater_equal;
		case rsx::comparison_function::equal: return mtl::sampler_compare_function::equal;
		case rsx::comparison_function::not_equal: return mtl::sampler_compare_function::not_equal;
		case rsx::comparison_function::always: return mtl::sampler_compare_function::always;
		}
		fmt::throw_exception("Invalid RSX sampler comparison 0x%x", static_cast<u32>(operation));
	}

	void transition_for_sampling(mtl::command_buffer& command, mtl::image_view& view,
		u64 stage, const rsx::sampled_image_descriptor_base& description)
	{
		mtl::image* resource = view.image();
		if (!resource) fmt::throw_exception("Metal sampled view has no image");
		if (description.is_cyclic_reference)
		{
			if (auto* target = dynamic_cast<mtl::render_target*>(resource))
			{
				target->texture_barrier(command);
				return;
			}
		}
		mtl::image_state state;
		state.queue = mtl::queue_kind::graphics;
		state.stages = stage;
		state.access = mtl::access_shader_read;
		state.initialized = true;
		mtl::transition_image(command, *resource, state, true);
	}

	template <rsx::Texture Texture, typename SampledImage>
	mtl::sampler_description make_sampler_description(const Texture& texture,
		const SampledImage& sampled, const mtl::device_info& device, bool vertex_stage)
	{
		mtl::sampler_description result;
		result.address_s = mtl::get_wrap_mode(texture.wrap_s());
		result.address_t = mtl::get_wrap_mode(texture.wrap_t());
		result.address_r = vertex_stage ? mtl::sampler_address_mode::wrap :
			mtl::get_wrap_mode(texture.wrap_r());
		result.normalized_coordinates = !(texture.format() & CELL_GCM_TEXTURE_UN);
		result.min_lod = texture.min_lod();
		result.max_lod = texture.max_lod();
		result.border = mtl::border_color::opaque_black();
		if (rsx::is_border_clamped_texture(texture))
		{
			const bool signed_conversion =
				(sampled.format_ex.texel_remap_control & rsx::texture_control_bits::SEXT_MASK) != 0;
			color4f color = texture.remapped_border_color(signed_conversion);
			if constexpr (requires { texture.argb_signed(); })
			{
				if (sampled.format_ex.host_snorm_format_active())
				{
					const f32 bias = 128.f / 255.f;
					const f32 scale = 255.f / 127.f;
					const u8 mask = texture.argb_signed();
					if (mask & 1) color.a = (color.a - bias) * scale;
					if (mask & 2) color.r = (color.r - bias) * scale;
					if (mask & 4) color.g = (color.g - bias) * scale;
					if (mask & 8) color.b = (color.b - bias) * scale;
				}
			}
			result.border = mtl::border_color::custom(
				{color.r, color.g, color.b, color.a}, 0, mtl::texture_aspect_color);
		}
		if (vertex_stage)
		{
			result.min_filter = mtl::sampler_filter::nearest;
			result.mag_filter = mtl::sampler_filter::nearest;
			result.mip_filter = mtl::sampler_mip_filter::nearest;
			return result;
		}

		const auto minimum = mtl::get_min_filter(texture.min_filter());
		result.min_filter = minimum.filter;
		result.mag_filter = mtl::get_mag_filter(texture.mag_filter());
		result.mip_filter = minimum.mip_filter;
		const bool signed_conversion =
			(sampled.format_ex.texel_remap_control & rsx::texture_control_bits::SEXT_MASK) != 0;
		u64 native_format = sampled.image_handle ? sampled.image_handle->format() : 0;
		if (!native_format && sampled.external_subresource_desc.gcm_format)
			native_format = mtl::get_sampler_format(device,
				sampled.external_subresource_desc.gcm_format).pixel_format;
		const bool filterable = native_format &&
			(mtl::describe_native_format(native_format).capabilities & mtl::format_capability_filterable);
		if (signed_conversion || !filterable)
		{
			result.min_filter = mtl::sampler_filter::nearest;
			result.mag_filter = mtl::sampler_filter::nearest;
			result.mip_filter = mtl::sampler_mip_filter::nearest;
		}
		if constexpr (requires { texture.max_aniso(); })
		{
			result.max_anisotropy = std::max(1u,
				static_cast<u32>(std::lround(mtl::get_max_anisotropy(texture.max_aniso()))));
		}
		const u32 format = sampled.format_ex.format();
		const bool depth = format >= CELL_GCM_TEXTURE_DEPTH24_D8 &&
			format <= CELL_GCM_TEXTURE_DEPTH16_FLOAT;
		if constexpr (requires { texture.zfunc(); })
		{
			if (depth)
			{
				result.compare_enabled = true;
				result.compare = sampler_compare(texture.zfunc(), true);
			}
		}

		f32 available_mips = static_cast<f32>(texture.get_exact_mipmap_count());
		if (sampled.upload_context != rsx::texture_upload_context::shader_read)
			available_mips = 1.f;
		if (minimum.sample_mipmaps && available_mips > 1.f)
		{
			result.min_lod = std::min(result.min_lod, available_mips - 1.f);
			result.max_lod = std::min(result.max_lod, available_mips - 1.f);
			result.lod_bias = texture.bias();
			if (result.mip_filter == mtl::sampler_mip_filter::nearest)
				result.lod_bias = std::floor(result.lod_bias * 2.f + .5f) * .5f;
		}
		else
		{
			result.min_lod = 0.f;
			result.max_lod = 0.f;
			result.lod_bias = 0.f;
			result.mip_filter = mtl::sampler_mip_filter::nearest;
		}
		return result;
	}

	mtl::image_view* alternate_texture_view(mtl::image_view* source, u64 format)
	{
		if (!source || !format || source->format() == format) return source;
		auto* image = dynamic_cast<mtl::viewable_image*>(source->image());
		if (!image) return source;
		return image->get_view(format, source->type(), source->mapping(), source->range());
	}
}

void MTLGSRender::load_texture_environment()
{
	bool cyclic_reference = false;
	std::lock_guard lock(m_sampler_mutex);

	for (u32 mask = current_fp_metadata.referenced_textures_mask, index = 0; mask; mask >>= 1, ++index)
	{
		if (!(mask & 1)) continue;
		if (!fs_sampler_state[index])
			fs_sampler_state[index] = std::make_unique<mtl::texture_cache::sampled_image_descriptor>();
		auto* sampled = static_cast<mtl::texture_cache::sampled_image_descriptor*>(
			fs_sampler_state[index].get());
		const auto& texture = rsx::method_registers.fragment_textures[index];
		const auto previous_class = sampled->format_class;
		const bool texture_dirty = m_textures_dirty[index];
		if (!m_samplers_dirty && !texture_dirty &&
			!m_texture_cache.test_if_descriptor_expired(*m_current_command_buffer,
				m_render_targets, sampled, texture))
		{
			cyclic_reference |= sampled->is_cyclic_reference;
			continue;
		}
		m_textures_dirty[index] = false;
		if (!texture.enabled())
		{
			*sampled = {};
			m_fragment_samplers[index] = nullptr;
			continue;
		}
		*sampled = m_texture_cache.upload_texture(*m_current_command_buffer,
			texture, m_render_targets);
		if (!sampled->validate())
		{
			m_fragment_samplers[index] = nullptr;
			continue;
		}
		cyclic_reference |= sampled->is_cyclic_reference;
		if (!texture_dirty && sampled->format_class != previous_class)
			m_graphics_state |= rsx::fragment_program_state_dirty;
		sampled->format_ex = texture.format_ex();

		if (sampled->format_ex.texel_remap_control && sampled->image_handle &&
			sampled->upload_context == rsx::texture_upload_context::shader_read &&
			!(current_fp_metadata.bx2_texture_reads_mask & (1u << index)) &&
			!g_cfg.video.disable_hardware_texel_remapping)
		{
			u64 replacement = 0;
			rsx::flags32_t erase = 0;
			rsx::flags32_t features = 0;
			if (sampled->format_ex.hw_SNORM_possible())
			{
				replacement = mtl::get_snorm_format(sampled->image_handle->format());
				erase = rsx::texture_control_bits::SEXT_MASK;
				features = rsx::RSX_HOST_FORMAT_FEATURE_SNORM;
			}
			else if (sampled->format_ex.hw_SRGB_possible())
			{
				replacement = mtl::get_srgb_format(sampled->image_handle->format());
				erase = rsx::texture_control_bits::GAMMA_CTRL_MASK;
				features = rsx::RSX_HOST_FORMAT_FEATURE_SRGB;
			}
			if (replacement && replacement != sampled->image_handle->format())
			{
				sampled->image_handle = alternate_texture_view(sampled->image_handle, replacement);
				sampled->format_ex.texel_remap_control &= ~erase;
				sampled->format_ex.host_features |= features;
			}
		}

		const mtl::sampler_description description =
			make_sampler_description(texture, *sampled, m_device->info(), false);
		m_fragment_samplers[index] = m_resources.samplers().get(description,
			fmt::format("RPCS3 fragment sampler %u", index));
	}

	for (u32 mask = current_vp_metadata.referenced_textures_mask, index = 0; mask; mask >>= 1, ++index)
	{
		if (!(mask & 1)) continue;
		if (!vs_sampler_state[index])
			vs_sampler_state[index] = std::make_unique<mtl::texture_cache::sampled_image_descriptor>();
		auto* sampled = static_cast<mtl::texture_cache::sampled_image_descriptor*>(
			vs_sampler_state[index].get());
		const auto& texture = rsx::method_registers.vertex_textures[index];
		const auto previous_class = sampled->format_class;
		const bool texture_dirty = m_vertex_textures_dirty[index];
		if (!m_samplers_dirty && !texture_dirty &&
			!m_texture_cache.test_if_descriptor_expired(*m_current_command_buffer,
				m_render_targets, sampled, texture))
		{
			cyclic_reference |= sampled->is_cyclic_reference;
			continue;
		}
		m_vertex_textures_dirty[index] = false;
		if (!texture.enabled())
		{
			*sampled = {};
			m_vertex_samplers[index] = nullptr;
			continue;
		}
		*sampled = m_texture_cache.upload_texture(*m_current_command_buffer,
			texture, m_render_targets);
		if (!sampled->validate())
		{
			m_vertex_samplers[index] = nullptr;
			continue;
		}
		cyclic_reference |= sampled->is_cyclic_reference ||
			sampled->external_subresource_desc.do_not_cache;
		if (!texture_dirty && sampled->format_class != previous_class)
			m_graphics_state |= rsx::vertex_program_state_dirty;
		const mtl::sampler_description description =
			make_sampler_description(texture, *sampled, m_device->info(), true);
		m_vertex_samplers[index] = m_resources.samplers().get(description,
			fmt::format("RPCS3 vertex sampler %u", index));
	}

	m_samplers_dirty.store(false);
	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE)
	{
		auto* depth = std::get<1>(m_render_targets.m_bound_depth_stencil);
		if (!depth) fmt::throw_exception("Metal depth-compare emulation has no depth surface");
		depth->texture_barrier(*m_current_command_buffer);
		cyclic_reference = true;
	}
	if (cyclic_reference)
	{
		close_render_pass();
		m_encoder_bindings.invalidate();
	}
}

bool MTLGSRender::bind_texture_environment()
{
	if (!m_program || !m_fragment_bindings || !m_vertex_bindings)
		fmt::throw_exception("Metal recompiled texture binding has no binding table");
	bool allocation_failed = false;
	auto& scratch = m_resources.scratch();
	const mtl::sampler& fallback_sampler = scratch.null_sampler();
	std::array<mtl::sampler_shader_state, rsx::limits::fragment_textures_count> sampler_states{};
	std::array<mtl::sampler_shader_state, rsx::limits::vertex_textures_count> vertex_sampler_states{};

	for (u32 mask = current_fp_metadata.referenced_textures_mask, index = 0; mask; mask >>= 1, ++index)
	{
		if (!(mask & 1)) continue;
		auto* sampled = static_cast<mtl::texture_cache::sampled_image_descriptor*>(
			fs_sampler_state[index].get());
		mtl::image_view* view = nullptr;
		if (sampled && rsx::method_registers.fragment_textures[index].enabled() && sampled->validate())
		{
			view = sampled->image_handle;
			if (!view)
			{
				view = m_texture_cache.create_temporary_subresource(*m_current_command_buffer,
					sampled->external_subresource_desc);
				allocation_failed |= !view;
			}
			if (view) transition_for_sampling(*m_current_command_buffer, *view,
				mtl::stage_fragment, *sampled);
		}
		const auto dimension = sampled && sampled->validate() ? sampled->image_type :
			current_fragment_program.get_texture_dimension(index);
		if (!view) view = &scratch.null_image_view(*m_current_command_buffer, texture_type(dimension));
		mtl::sampler* sampler = m_fragment_samplers[index].get();
		if (!sampler) sampler = const_cast<mtl::sampler*>(&fallback_sampler);
		m_program->set_texture(mtl::msl_shader_stage::fragment,
			m_fragment_bindings->textures[index].index, *view);
		if (m_fragment_bindings->samplers[index])
			m_program->set_sampler(mtl::msl_shader_stage::fragment,
				m_fragment_bindings->samplers[index].index, *sampler);
		sampler_states[index] = sampler->shader_state();

		if (m_fragment_bindings->stencil_textures[index])
		{
			mtl::image_view* stencil = nullptr;
			if (view->image() && (view->image()->aspects() & mtl::texture_aspect_stencil))
			{
				if (auto* root = dynamic_cast<mtl::viewable_image*>(view->image()))
				{
					mtl::subresource_range range = view->range();
					range.color = false;
					range.depth = false;
					range.stencil = true;
					stencil = root->get_view(stencil_view_format(view->image()->format()),
						view->type(), {}, range);
				}
			}
			if (!stencil)
				stencil = &scratch.null_image_view(*m_current_command_buffer,
					texture_type(dimension), MTLPixelFormatR8Uint);
			m_program->set_texture(mtl::msl_shader_stage::fragment,
				m_fragment_bindings->stencil_textures[index].index, *stencil);
		}
	}

	for (u32 mask = current_vp_metadata.referenced_textures_mask, index = 0; mask; mask >>= 1, ++index)
	{
		if (!(mask & 1)) continue;
		auto* sampled = static_cast<mtl::texture_cache::sampled_image_descriptor*>(
			vs_sampler_state[index].get());
		mtl::image_view* view = nullptr;
		if (sampled && rsx::method_registers.vertex_textures[index].enabled() && sampled->validate())
		{
			view = sampled->image_handle;
			if (!view)
			{
				view = m_texture_cache.create_temporary_subresource(*m_current_command_buffer,
					sampled->external_subresource_desc);
				allocation_failed |= !view;
			}
			if (view) transition_for_sampling(*m_current_command_buffer, *view,
				mtl::stage_vertex, *sampled);
		}
		const auto dimension = sampled && sampled->validate() ? sampled->image_type :
			current_vertex_program.get_texture_dimension(index);
		if (!view) view = &scratch.null_image_view(*m_current_command_buffer, texture_type(dimension));
		mtl::sampler* sampler = m_vertex_samplers[index].get();
		if (!sampler) sampler = const_cast<mtl::sampler*>(&fallback_sampler);
		vertex_sampler_states[index] = sampler->shader_state();
		m_program->set_texture(mtl::msl_shader_stage::vertex,
			m_vertex_bindings->textures[index].index, *view);
		m_program->set_sampler(mtl::msl_shader_stage::vertex,
			m_vertex_bindings->samplers[index].index, *sampler);
	}

	if (m_fragment_bindings->uses_depth_input)
	{
		auto* depth = std::get<1>(m_render_targets.m_bound_depth_stencil);
		if (!depth) fmt::throw_exception("Metal fragment depth input has no surface");
		mtl::image_view* view = depth->get_view(mtl::texture_aspect_depth);
		mtl::image_state state;
		state.queue = mtl::queue_kind::graphics;
		state.stages = mtl::stage_fragment;
		state.access = mtl::access_shader_read;
		state.initialized = true;
		mtl::transition_image(*m_current_command_buffer, *view->image(), state, true);
		m_program->set_texture(mtl::msl_shader_stage::fragment,
			m_fragment_bindings->depth_input_texture.index, *view);
	}

	const mtl::data_heap_slice state_slice = m_fragment_texture_parameters_heap.allocate(
		sizeof(sampler_states), 256);
	std::memcpy(m_fragment_texture_parameters_heap.map(state_slice), sampler_states.data(),
		sizeof(sampler_states));
	m_fragment_texture_parameters_heap.mark_modified(state_slice);
	m_fragment_texture_parameters_heap.unmap();
	m_sampler_state_binding = {.resource = state_slice.buffer, .gpu_address = state_slice.buffer_gpu_address(),
		.offset = state_slice.offset, .length = state_slice.size};
	m_program->set_buffer(mtl::msl_shader_stage::fragment,
		m_fragment_bindings->sampler_state_buffer.index, m_sampler_state_binding);
	if (m_vertex_bindings->sampler_state_buffer)
	{
		const mtl::data_heap_slice vertex_state_slice = m_fragment_texture_parameters_heap.allocate(
			sizeof(vertex_sampler_states), 256);
		std::memcpy(m_fragment_texture_parameters_heap.map(vertex_state_slice),
			vertex_sampler_states.data(), sizeof(vertex_sampler_states));
		m_fragment_texture_parameters_heap.mark_modified(vertex_state_slice);
		m_fragment_texture_parameters_heap.unmap();
		m_vertex_sampler_state_binding = {.resource = vertex_state_slice.buffer,
			.gpu_address = vertex_state_slice.buffer_gpu_address(), .offset = vertex_state_slice.offset,
			.length = vertex_state_slice.size};
		m_program->set_buffer(mtl::msl_shader_stage::vertex,
			m_vertex_bindings->sampler_state_buffer.index, m_vertex_sampler_state_binding);
	}
	return allocation_failed;
}

bool MTLGSRender::bind_interpreter_texture_environment()
{
	bool allocation_failed = false;
	auto& scratch = m_resources.scratch();
	const mtl::sampler& fallback_sampler = scratch.null_sampler();
	std::array<mtl::argument_texture_binding, 96> textures{};
	std::array<mtl::argument_sampler_binding, 16> samplers{};
	std::array<mtl::sampler_shader_state, rsx::limits::fragment_textures_count> fragment_sampler_states{};
	std::array<mtl::sampler_shader_state, rsx::limits::vertex_textures_count> vertex_sampler_states{};
	for (u32 block = 0; block < 4; ++block)
	{
		const std::array types{mtl::texture_type::texture_2d, mtl::texture_type::texture_2d,
			mtl::texture_type::texture_cube, mtl::texture_type::texture_3d};
		mtl::image_view& fallback = scratch.null_image_view(*m_current_command_buffer, types[block]);
		for (u32 index = 0; index < 16; ++index)
			textures[block * 16 + index] = {fallback.native_handle(), mtl::argument_access::read};
	}
	mtl::image_view& multisample_fallback = scratch.null_image_view(*m_current_command_buffer,
		mtl::texture_type::texture_2d_multisample);
	mtl::image_view& stencil_fallback = scratch.null_image_view(*m_current_command_buffer,
		mtl::texture_type::texture_2d, MTLPixelFormatR8Uint);
	for (u32 index = 0; index < 16; ++index)
	{
		textures[64 + index] = {multisample_fallback.native_handle(), mtl::argument_access::read};
		textures[80 + index] = {stencil_fallback.native_handle(), mtl::argument_access::read};
	}
	for (u32 index = 0; index < 16; ++index)
	{
		samplers[index] = {fallback_sampler.native_handle()};
		fragment_sampler_states[index] = fallback_sampler.shader_state();
	}

	for (u32 mask = current_fp_metadata.referenced_textures_mask, index = 0; mask; mask >>= 1, ++index)
	{
		if (!(mask & 1)) continue;
		auto* sampled = static_cast<mtl::texture_cache::sampled_image_descriptor*>(
			fs_sampler_state[index].get());
		if (!sampled || !rsx::method_registers.fragment_textures[index].enabled() ||
			!sampled->validate()) continue;
		mtl::image_view* view = sampled->image_handle;
		if (!view)
		{
			view = m_texture_cache.create_temporary_subresource(*m_current_command_buffer,
				sampled->external_subresource_desc);
			allocation_failed |= !view;
		}
		if (!view) continue;
		transition_for_sampling(*m_current_command_buffer, *view, mtl::stage_fragment, *sampled);
		const bool multisampled = (current_fragment_program.texture_state.multisampled_textures &
			(1u << index)) != 0;
		textures[(multisampled ? 64u : interpreter_texture_offset(sampled->image_type)) + index] =
			{view->native_handle(), mtl::argument_access::read};
		const bool redirected = (current_fragment_program.texture_state.redirected_textures &
			(1u << index)) != 0;
		if (redirected)
		{
			if (view->image() && (view->image()->aspects() & mtl::texture_aspect_stencil))
			{
				if (auto* root = dynamic_cast<mtl::viewable_image*>(view->image()))
				{
					mtl::subresource_range range = view->range();
					range.color = false;
					range.depth = false;
					range.stencil = true;
					if (mtl::image_view* stencil = root->get_view(
						stencil_view_format(view->image()->format()),
						mtl::texture_type::texture_2d, {}, range))
						textures[80 + index] = {stencil->native_handle(), mtl::argument_access::read};
				}
			}
		}
		if (m_fragment_samplers[index])
		{
			samplers[index] = {m_fragment_samplers[index]->native_handle()};
			fragment_sampler_states[index] = m_fragment_samplers[index]->shader_state();
		}
		if (multisampled) fragment_sampler_states[index].border_metadata |= 1u << 9;
		if (redirected) fragment_sampler_states[index].border_metadata |= 1u << 10;
	}
	for (u32 index = 0; index < textures.size(); ++index)
		m_program->set_texture(mtl::msl_shader_stage::fragment, index, textures[index]);
	for (u32 index = 0; index < samplers.size(); ++index)
		m_program->set_sampler(mtl::msl_shader_stage::fragment, index, samplers[index]);

	for (u32 index = 0; index < rsx::limits::vertex_textures_count; ++index)
	{
		mtl::image_view* view = &scratch.null_image_view(*m_current_command_buffer,
			mtl::texture_type::texture_2d);
		mtl::sampler* sampler = const_cast<mtl::sampler*>(&fallback_sampler);
		if (current_vp_metadata.referenced_textures_mask & (1u << index))
		{
			auto* sampled = static_cast<mtl::texture_cache::sampled_image_descriptor*>(
				vs_sampler_state[index].get());
			if (sampled && rsx::method_registers.vertex_textures[index].enabled() && sampled->validate())
			{
				view = sampled->image_handle;
				if (!view)
				{
					view = m_texture_cache.create_temporary_subresource(*m_current_command_buffer,
						sampled->external_subresource_desc);
					allocation_failed |= !view;
				}
				if (view) transition_for_sampling(*m_current_command_buffer, *view,
					mtl::stage_vertex, *sampled);
				else view = &scratch.null_image_view(*m_current_command_buffer,
					mtl::texture_type::texture_2d);
			}
			if (m_vertex_samplers[index]) sampler = m_vertex_samplers[index].get();
		}
		m_program->set_texture(mtl::msl_shader_stage::vertex, index, *view);
		m_program->set_sampler(mtl::msl_shader_stage::vertex, index, *sampler);
		vertex_sampler_states[index] = sampler->shader_state();
	}
	const auto upload_sampler_states = [&](std::span<const mtl::sampler_shader_state> states)
	{
		const mtl::data_heap_slice slice = m_fragment_texture_parameters_heap.allocate(
			states.size_bytes(), 256);
		std::memcpy(m_fragment_texture_parameters_heap.map(slice), states.data(), states.size_bytes());
		m_fragment_texture_parameters_heap.mark_modified(slice);
		m_fragment_texture_parameters_heap.unmap();
		return mtl::argument_buffer_binding{.resource = slice.buffer, .gpu_address = slice.buffer_gpu_address(),
			.offset = slice.offset, .length = slice.size};
	};
	m_sampler_state_binding = upload_sampler_states(fragment_sampler_states);
	m_program->set_buffer(mtl::msl_shader_stage::fragment, 6, m_sampler_state_binding);
	m_vertex_sampler_state_binding = upload_sampler_states(vertex_sampler_states);
	m_program->set_buffer(mtl::msl_shader_stage::vertex, 8, m_vertex_sampler_state_binding);
	return allocation_failed;
}

void MTLGSRender::update_draw_state()
{
	if (!m_render_pass.is_open()) fmt::throw_exception("Metal dynamic state requires a render pass");
	native_render_encoder encoder =
		(__bridge native_render_encoder)m_render_pass.native_encoder();
	const auto blend = rsx::get_constant_blend_colors();
	[encoder setBlendColorRed:blend[0] green:blend[1] blue:blend[2] alpha:blend[3]];
	[encoder setStencilFrontReferenceValue:rsx::method_registers.stencil_func_ref()
		backReferenceValue:rsx::method_registers.two_sided_stencil_test_enabled() ?
			rsx::method_registers.back_stencil_func_ref() :
			rsx::method_registers.stencil_func_ref()];
	const bool line_expansion = m_pipeline_properties.state.render.emulation_flags &
		mtl::pipeline_emulation_wide_lines;
	const bool depth_bias = line_expansion ? rsx::method_registers.poly_offset_line_enabled() :
		m_pipeline_properties.state.render.topology == mtl::primitive_topology::point ?
			rsx::method_registers.poly_offset_point_enabled() :
			rsx::method_registers.poly_offset_fill_enabled();
	[encoder setDepthBias:depth_bias ? rsx::method_registers.poly_offset_bias() : 0.f
		slopeScale:depth_bias ? rsx::method_registers.poly_offset_scale() : 0.f clamp:0.f];
	const bool depth_bounds = rsx::method_registers.depth_bounds_test_enabled();
	[encoder setDepthTestMinBound:depth_bounds ? rsx::method_registers.depth_bounds_min() :
		std::min(0.f, rsx::method_registers.clip_min())
		maxBound:depth_bounds ? rsx::method_registers.depth_bounds_max() :
		std::max(1.f, rsx::method_registers.clip_max())];
	bind_viewport();
	m_current_command_buffer->clear_flag(mtl::command_reload_dynamic_state);
}

void MTLGSRender::emit_geometry(u32 sub_index)
{
	auto& draw = rsx::method_registers.current_draw_clause;
	m_profiler.start();
	const rsx::flags32_t vertex_mask = rsx::vertex_base_changed | rsx::vertex_arrays_changed;
	const rsx::flags32_t state = sub_index == 0 ? rsx::vertex_arrays_changed :
		draw.execute_pipeline_dependencies(m_ctx);
	if (state & rsx::vertex_arrays_changed)
		m_draw_processor.analyse_inputs_interleaved(m_vertex_layout, current_vp_metadata);
	else if (state & rsx::vertex_base_changed)
	{
		for (auto& block : m_vertex_layout.interleaved_blocks)
		{
			block->vertex_range.second = 0;
			const u32 base = rsx::method_registers.vertex_data_base_offset();
			block->real_offset_address = rsx::get_address(
				rsx::get_vertex_offset_from_base(base, block->base_offset), block->memory_location);
		}
	}
	else
	{
		for (auto& block : m_vertex_layout.interleaved_blocks) block->vertex_range.second = 0;
	}
	if ((state & vertex_mask) && !m_vertex_layout.validate())
	{
		do draw.execute_pipeline_dependencies(m_ctx); while (draw.next());
		draw.end();
		return;
	}

	const mtl::vertex_upload_info upload = upload_vertex_data();
	if (!upload.vertex_draw_count) return;
	m_frame_stats.vertex_upload_time += m_profiler.duration();
	if (m_current_draw.subdraw_id)
	{
		m_program_instance = m_program->create_binding_instance(true);
		m_program = m_program_instance.get();
	}
	update_vertex_environment(sub_index, upload);
	load_program_environment();
	// Private heap generations are populated through their CPU-visible shadows. Those copies must
	// precede the render encoder that consumes them; flushing only at submission records the copies
	// after the draw and leaves every per-draw vertex/context slice unreadable for that draw.
	mtl::get_data_heap_manager().flush_all(*m_current_command_buffer);

	begin_render_pass();
	m_program->bind(*m_current_command_buffer);
	if (!m_current_draw.subdraw_id++ ||
		m_current_command_buffer->has_flag(mtl::command_reload_dynamic_state))
		update_draw_state();

	native_render_encoder encoder =
		(__bridge native_render_encoder)m_render_pass.native_encoder();
	const MTLPrimitiveType primitive = primitive_type(
		m_pipeline_properties.state.render.topology);
	const u32 instances = draw.is_trivial_instanced_draw ? draw.pass_count() : 1;
	if (!upload.index_info)
	{
		if (upload.line_expansion || draw.is_trivial_instanced_draw || draw.is_single_draw())
		{
			[encoder drawPrimitives:primitive vertexStart:0 vertexCount:upload.vertex_draw_count
				instanceCount:instances baseInstance:0];
		}
		else
		{
			u32 first = 0;
			for (const auto& range : draw.get_subranges())
			{
				[encoder drawPrimitives:primitive vertexStart:first vertexCount:range.count];
				first += range.count;
			}
		}
	}
	else
	{
		const auto [offset, element_type] = *upload.index_info;
		const mtl::buffer& index_buffer = m_index_heap.target_buffer();
		// Metal 4 indexed draws consume a raw GPU address, so the index allocation is not
		// discovered through an argument table and must be made resident explicitly.
		m_current_command_buffer->retain_native_object(
			(__bridge void*)index_buffer.native_handle(), true);
		const MTLGPUAddress indices = index_buffer.gpu_address() + offset;
		const MTLIndexType native_index_type = index_type(element_type);
		if (draw.is_trivial_instanced_draw || draw.is_single_draw())
		{
			[encoder drawIndexedPrimitives:primitive indexCount:upload.vertex_draw_count
				indexType:native_index_type indexBuffer:indices
				indexBufferLength:static_cast<NSUInteger>(upload.vertex_draw_count) * index_stride(element_type)
				instanceCount:instances baseVertex:0 baseInstance:0];
		}
		else
		{
			u32 first = 0;
			for (const auto& range : draw.get_subranges())
			{
				const u32 count = get_index_count(draw.primitive, range.count);
				[encoder drawIndexedPrimitives:primitive indexCount:count
					indexType:native_index_type
					indexBuffer:indices + static_cast<u64>(first) * index_stride(element_type)
					indexBufferLength:static_cast<NSUInteger>(count) * index_stride(element_type)];
				first += count;
			}
		}
	}
	m_frame_stats.draw_exec_time += m_profiler.duration();
}

void MTLGSRender::begin()
{
	m_interpreter_state = m_graphics_state.load() & rsx::pipeline_state::invalidate_pipeline_bits;
	rsx::thread::begin();
	if (skip_current_frame || m_swapchain_unavailable || cond_render_ctrl.disable_rendering()) return;
	initialize_buffers(rsx::framebuffer_creation_context::context_draw);
	if (m_graphics_state & rsx::pipeline_state::invalidate_pipeline_bits)
	{
		m_program_instance.reset();
		m_program = nullptr;
		m_program_template = nullptr;
		m_program_interpreted = false;
	}
}

void MTLGSRender::end()
{
	if (skip_current_frame || !m_graphics_state.test(rsx::rtt_config_valid) ||
		m_swapchain_unavailable || cond_render_ctrl.disable_rendering())
	{
		execute_nop_draw();
		rsx::thread::end();
		return;
	}
	m_profiler.start();
	if (m_current_frame->flags & frame_context_dirty)
	{
		check_present_status();
		if (m_current_frame->swap_command_buffer)
		{
			m_auxiliary_frame_context.grab_resources(*m_current_frame);
			m_current_frame = &m_auxiliary_frame_context;
		}
		if (m_current_frame->swap_command_buffer)
			fmt::throw_exception("Metal draw frame still owns a presentation command buffer");
		m_current_frame->flags &= ~frame_context_dirty;
	}

	analyse_current_rsx_pipeline();
	m_frame_stats.setup_time += m_profiler.duration();
	load_texture_environment();
	m_frame_stats.textures_upload_time += m_profiler.duration();
	if (!load_program())
	{
		std::this_thread::yield();
		execute_nop_draw();
		rsx::thread::end();
		return;
	}

	for (auto& color : m_render_targets.m_bound_render_targets)
		if (auto* target = std::get<1>(color)) target->write_barrier(*m_current_command_buffer);
	if (auto* depth = std::get<1>(m_render_targets.m_bound_depth_stencil))
		depth->write_barrier(*m_current_command_buffer);
	m_graphics_state.clear(rsx::zeta_address_cyclic_barrier);

	for (u32 retry = 0; retry < 3; ++retry)
	{
		if (retry && m_samplers_dirty)
		{
			load_texture_environment();
			m_graphics_state |= rsx::vertex_program_state_dirty |
				rsx::fragment_program_state_dirty;
			get_current_fragment_program(fs_sampler_state);
			get_current_vertex_program(vs_sampler_state);
			m_graphics_state.clear(rsx::pipeline_state::invalidate_pipeline_bits);
		}
		const bool allocation_failed = m_program_interpreted
			? bind_interpreter_texture_environment() : bind_texture_environment();
		if (!allocation_failed || !on_vram_exhausted(rsx::problem_severity::fatal)) break;
	}
	m_texture_cache.release_uncached_temporary_subresources();
	m_frame_stats.textures_upload_time += m_profiler.duration();

	m_current_draw.subdraw_id = 0;
	auto& draw = rsx::method_registers.current_draw_clause;
	draw.begin();
	u32 sub_index = 0;
	do
	{
		emit_geometry(sub_index++);
		if (draw.is_trivial_instanced_draw) draw.end();
	}
	while (draw.next());
	m_render_targets.on_write(m_framebuffer_layout.color_write_enabled,
		m_framebuffer_layout.zeta_write_enabled);
	rsx::thread::end();
}
