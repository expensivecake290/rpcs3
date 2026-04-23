#include "stdafx.h"
#include "MTLGSRender.h"

#include "MTLArgumentTable.h"
#include "MTLDrawCommands.h"
#include "MTLDrawResources.h"
#include "MTLRenderState.h"
#include "MTLShaderInterface.h"

#include "Emu/System.h"
#include "Emu/RSX/Common/BufferUtils.h"
#include "Emu/RSX/Common/surface_store.h"
#include "Emu/RSX/Common/TextureUtils.h"
#include "Emu/RSX/Program/ProgramStateCache.h"
#include "Emu/RSX/color_utils.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/RSX/rsx_utils.h"
#include "Emu/system_config.h"
#include "util/fnv_hash.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <span>
#include <tuple>

namespace
{
	struct metal_index_buffer_binding
	{
		rsx::metal::draw_buffer_binding buffer{};
		rsx::index_array_type index_type = rsx::index_array_type::u16;
		u32 index_count = 0;
		u32 index_size = 0;
		u32 min_index = 0;
		u32 max_index = 0;
		b8 valid = false;
	};

	struct metal_vertex_input_state
	{
		u32 vertex_draw_count = 0;
		u32 allocated_vertex_count = 0;
		u32 first_vertex = 0;
		u32 vertex_index_base = 0;
		u32 vertex_index_offset = 0;
		s64 base_vertex = 0;
		metal_index_buffer_binding index_buffer{};
		b8 indexed = false;
	};

	u64 append_pipeline_hash_u32(u64 hash, u32 value)
	{
		for (u32 index = 0; index < sizeof(value); index++)
		{
			hash = rpcs3::hash64(hash, static_cast<u8>((value >> (index * 8)) & 0xff));
		}

		return hash;
	}

	u64 append_pipeline_hash_u64(u64 hash, u64 value)
	{
		for (u32 index = 0; index < sizeof(value); index++)
		{
			hash = rpcs3::hash64(hash, static_cast<u8>((value >> (index * 8)) & 0xff));
		}

		return hash;
	}

	u64 make_draw_pipeline_hash(
		const rsx::metal::translated_shader& vertex,
		const rsx::metal::translated_shader& fragment,
		u32 color_pixel_format,
		u32 sample_count,
		u32 topology_class,
		u32 fragment_texture_slots,
		u32 color_write_mask,
		b8 blend_enabled,
		rsx::blend_factor source_rgb_blend_factor,
		rsx::blend_factor source_alpha_blend_factor,
		rsx::blend_factor destination_rgb_blend_factor,
		rsx::blend_factor destination_alpha_blend_factor,
		rsx::blend_equation rgb_blend_operation,
		rsx::blend_equation alpha_blend_operation)
	{
		rsx_log.trace("make_draw_pipeline_hash(vertex=0x%llx, fragment=0x%llx, color_pixel_format=%u, sample_count=%u, topology_class=%u, fragment_textures=%u, color_mask=0x%x, blend=%u)",
			vertex.source_hash,
			fragment.source_hash,
			color_pixel_format,
			sample_count,
			topology_class,
			fragment_texture_slots,
			color_write_mask,
			static_cast<u32>(blend_enabled));

		u64 hash = rpcs3::fnv_seed;
		hash = append_pipeline_hash_u64(hash, vertex.source_hash);
		hash = append_pipeline_hash_u64(hash, vertex.pipeline_source_hash);
		hash = append_pipeline_hash_u64(hash, fragment.source_hash);
		hash = append_pipeline_hash_u64(hash, fragment.pipeline_source_hash);
		hash = append_pipeline_hash_u32(hash, vertex.pipeline_requirement_mask);
		hash = append_pipeline_hash_u32(hash, fragment.pipeline_requirement_mask);
		hash = append_pipeline_hash_u32(hash, color_pixel_format);
		hash = append_pipeline_hash_u32(hash, sample_count);
		hash = append_pipeline_hash_u32(hash, topology_class);
		hash = append_pipeline_hash_u32(hash, fragment_texture_slots);
		hash = append_pipeline_hash_u32(hash, color_write_mask);
		hash = append_pipeline_hash_u32(hash, static_cast<u32>(blend_enabled));
		hash = append_pipeline_hash_u32(hash, static_cast<u32>(source_rgb_blend_factor));
		hash = append_pipeline_hash_u32(hash, static_cast<u32>(source_alpha_blend_factor));
		hash = append_pipeline_hash_u32(hash, static_cast<u32>(destination_rgb_blend_factor));
		hash = append_pipeline_hash_u32(hash, static_cast<u32>(destination_alpha_blend_factor));
		hash = append_pipeline_hash_u32(hash, static_cast<u32>(rgb_blend_operation));
		hash = append_pipeline_hash_u32(hash, static_cast<u32>(alpha_blend_operation));
		return hash ? hash : 1;
	}

	u32 make_metal_color_write_mask(u32 color_index, rsx::surface_color_format format)
	{
		rsx_log.trace("make_metal_color_write_mask(color_index=%u, format=0x%x)",
			color_index,
			static_cast<u32>(format));

		bool color_mask_r = rsx::method_registers.color_mask_r(color_index);
		bool color_mask_g = rsx::method_registers.color_mask_g(color_index);
		bool color_mask_b = rsx::method_registers.color_mask_b(color_index);
		bool color_mask_a = rsx::method_registers.color_mask_a(color_index);

		switch (format)
		{
		case rsx::surface_color_format::b8:
			rsx::get_b8_colormask(color_mask_r, color_mask_g, color_mask_b, color_mask_a);
			break;
		case rsx::surface_color_format::g8b8:
			rsx::get_g8b8_r8g8_colormask(color_mask_r, color_mask_g, color_mask_b, color_mask_a);
			break;
		default:
			break;
		}

		const std::array<bool, 4> host_write_mask = rsx::get_write_output_mask(format);
		u32 mask = 0;
		if (color_mask_r && host_write_mask[0])
		{
			mask |= 0x1;
		}

		if (color_mask_g && host_write_mask[1])
		{
			mask |= 0x2;
		}

		if (color_mask_b && host_write_mask[2])
		{
			mask |= 0x4;
		}

		if (color_mask_a && host_write_mask[3])
		{
			mask |= 0x8;
		}

		return mask;
	}

	b8 get_metal_blend_enabled(u32 color_index)
	{
		rsx_log.trace("get_metal_blend_enabled(color_index=%u)", color_index);

		switch (color_index)
		{
		case 0:
			return rsx::method_registers.blend_enabled();
		case 1:
			return rsx::method_registers.blend_enabled_surface_1();
		case 2:
			return rsx::method_registers.blend_enabled_surface_2();
		case 3:
			return rsx::method_registers.blend_enabled_surface_3();
		}

		fmt::throw_exception("Metal blend state active color index out of range: index=%u", color_index);
	}

	u32 get_fragment_texture_slot_count(u32 referenced_textures_mask)
	{
		rsx_log.trace("get_fragment_texture_slot_count(referenced_textures_mask=0x%x)", referenced_textures_mask);

		u32 slot_count = 0;
		for (u32 mask = referenced_textures_mask; mask; mask >>= 1)
		{
			slot_count++;
		}

		if (slot_count > rsx::limits::fragment_textures_count)
		{
			fmt::throw_exception("Metal fragment texture slot count exceeds RSX limits: count=%u, limit=%u",
				slot_count,
				static_cast<u32>(rsx::limits::fragment_textures_count));
		}

		return slot_count;
	}

	metal_vertex_input_state get_array_vertex_input_state(const rsx::draw_array_command&)
	{
		rsx_log.trace("get_array_vertex_input_state()");

		const u32 vertex_count = rsx::method_registers.current_draw_clause.get_elements_count();
		const u32 min_index = rsx::method_registers.current_draw_clause.min_index();
		if (!vertex_count)
		{
			return {};
		}

		if (min_index > std::numeric_limits<u32>::max() - (vertex_count - 1))
		{
			fmt::throw_exception("Metal vertex input array range overflow: min=%u, count=%u",
				min_index,
				vertex_count);
		}

		return
		{
			.vertex_draw_count = vertex_count,
			.allocated_vertex_count = vertex_count,
			.first_vertex = min_index,
		};
	}

	metal_vertex_input_state get_indexed_vertex_input_state(
		const rsx::draw_indexed_array_command& command,
		rsx::metal::command_frame& frame,
		rsx::metal::draw_resource_manager& draw_resources)
	{
		rsx_log.trace("get_indexed_vertex_input_state()");

		const rsx::primitive_type primitive = rsx::method_registers.current_draw_clause.primitive;
		if (!is_primitive_native(primitive))
		{
			rsx_log.todo("get_indexed_vertex_input_state() non-native indexed primitive 0x%x requires GPU expansion", static_cast<u32>(primitive));
			fmt::throw_exception("Metal backend non-native indexed primitive 0x%x requires a GPU expansion path that is not implemented yet",
				static_cast<u32>(primitive));
		}

		if (rsx::method_registers.restart_index_enabled())
		{
			rsx_log.todo("get_indexed_vertex_input_state() primitive restart is not implemented");
			fmt::throw_exception("Metal backend indexed primitive restart is not implemented yet");
		}

		const rsx::index_array_type index_type = rsx::method_registers.current_draw_clause.is_immediate_draw
			? rsx::index_array_type::u32
			: rsx::method_registers.index_type();
		const u32 index_size = get_index_type_size(index_type);
		const u32 index_count = rsx::method_registers.current_draw_clause.get_elements_count();
		if (!index_count)
		{
			return {};
		}

		if (index_count > std::numeric_limits<u32>::max() / index_size)
		{
			fmt::throw_exception("Metal index buffer upload size overflow: count=%u, index_size=%u",
				index_count,
				index_size);
		}

		const u32 upload_size = index_count * index_size;
		if (command.raw_index_buffer.size() < upload_size)
		{
			fmt::throw_exception("Metal indexed draw raw index buffer is too small: required=0x%x, available=0x%x",
				upload_size,
				static_cast<u32>(command.raw_index_buffer.size()));
		}

		u32 min_index = 0;
		u32 max_index = 0;
		u32 written_index_count = 0;
		rsx::metal::draw_buffer_binding index_buffer = draw_resources.upload_generated_buffer(
			frame,
			upload_size,
			"RPCS3 Metal index stream",
			[&command, index_type, primitive, upload_size, &min_index, &max_index, &written_index_count](void* data, u64 size)
			{
				if (size != upload_size)
				{
					fmt::throw_exception("Metal index buffer upload size mismatch: expected=0x%x, size=0x%llx",
						upload_size,
						size);
				}

				std::span<std::byte> dst(static_cast<std::byte*>(data), upload_size);
				std::tie(min_index, max_index, written_index_count) = write_index_array_data_to_buffer(
					dst,
					command.raw_index_buffer,
					index_type,
					primitive,
					false,
					0,
					[](rsx::primitive_type)
					{
						return false;
					});
			});

		if (!written_index_count)
		{
			return {};
		}

		if (min_index > max_index)
		{
			fmt::throw_exception("Metal indexed draw produced an invalid index range: min=%u, max=%u",
				min_index,
				max_index);
		}

		const u32 allocated_vertex_count = (max_index - min_index) + 1;
		const u32 vertex_base = rsx::get_index_from_base(min_index, rsx::method_registers.vertex_data_base_index());

		return
		{
			.vertex_draw_count = written_index_count,
			.allocated_vertex_count = allocated_vertex_count,
			.first_vertex = vertex_base,
			.vertex_index_base = min_index,
			.vertex_index_offset = rsx::method_registers.vertex_data_base_index(),
			.base_vertex = -static_cast<s64>(min_index),
			.index_buffer =
			{
				.buffer = index_buffer,
				.index_type = index_type,
				.index_count = written_index_count,
				.index_size = index_size,
				.min_index = min_index,
				.max_index = max_index,
				.valid = true,
			},
			.indexed = true,
		};
	}

	metal_vertex_input_state get_inlined_vertex_input_state(const rsx::draw_inlined_array&, const rsx::vertex_input_layout& layout)
	{
		rsx_log.trace("get_inlined_vertex_input_state()");

		if (layout.interleaved_blocks.empty() || !layout.interleaved_blocks[0]->attribute_stride)
		{
			fmt::throw_exception("Metal inlined vertex input requires a valid interleaved vertex block");
		}

		const u64 stream_size = rsx::method_registers.current_draw_clause.inline_vertex_array.size() * sizeof(u32);
		const u32 stride = layout.interleaved_blocks[0]->attribute_stride;
		if (stream_size % stride)
		{
			fmt::throw_exception("Metal inlined vertex stream size is not aligned to stride: size=0x%llx, stride=0x%x",
				stream_size,
				stride);
		}

		const u64 vertex_count = stream_size / stride;
		if (vertex_count > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal inlined vertex count exceeds u32 range: count=0x%llx", vertex_count);
		}

		return
		{
			.vertex_draw_count = static_cast<u32>(vertex_count),
			.allocated_vertex_count = static_cast<u32>(vertex_count),
			.first_vertex = 0,
		};
	}

	struct metal_vertex_input_state_visitor
	{
		const rsx::vertex_input_layout& layout;
		rsx::metal::command_frame& frame;
		rsx::metal::draw_resource_manager& draw_resources;

		metal_vertex_input_state operator()(const rsx::draw_array_command& command) const
		{
			return get_array_vertex_input_state(command);
		}

		metal_vertex_input_state operator()(const rsx::draw_indexed_array_command& command) const
		{
			return get_indexed_vertex_input_state(command, frame, draw_resources);
		}

		metal_vertex_input_state operator()(const rsx::draw_inlined_array& command) const
		{
			return get_inlined_vertex_input_state(command, layout);
		}
	};

	rsx::metal::prepared_draw_command make_prepared_draw_command(const metal_vertex_input_state& vertex_state)
	{
		rsx_log.trace("make_prepared_draw_command(draw_vertices=%u, allocated_vertices=%u, indexed=%u, index_count=%u)",
			vertex_state.vertex_draw_count,
			vertex_state.allocated_vertex_count,
			static_cast<u32>(vertex_state.indexed),
			vertex_state.index_buffer.index_count);

		const auto& draw_call = rsx::method_registers.current_draw_clause;

		rsx::metal::prepared_draw_command command{};
		command.primitive = rsx::metal::get_draw_primitive_type(draw_call.primitive);
		command.vertex_start = 0;
		command.vertex_count = vertex_state.vertex_draw_count;
		command.instance_count = draw_call.pass_count();
		command.base_instance = 0;
		command.base_vertex = vertex_state.base_vertex;
		command.indexed = vertex_state.indexed;

		if (vertex_state.indexed)
		{
			if (!vertex_state.index_buffer.valid || !vertex_state.index_buffer.buffer.resource)
			{
				fmt::throw_exception("Metal indexed draw state is missing a valid index buffer");
			}

			command.index =
			{
				.resource = vertex_state.index_buffer.buffer.resource,
				.offset = vertex_state.index_buffer.buffer.offset,
				.length = static_cast<u64>(vertex_state.index_buffer.index_count) * vertex_state.index_buffer.index_size,
				.index_count = vertex_state.index_buffer.index_count,
				.type = rsx::metal::get_draw_index_type(vertex_state.index_buffer.index_type),
			};
		}

		return command;
	}
}

u64 MTLGSRender::get_cycles()
{
	return thread_ctrl::get_cycles(static_cast<named_thread<MTLGSRender>&>(*this));
}

MTLGSRender::MTLGSRender(utils::serial* ar) noexcept
	: rsx::thread(ar)
{
	backend_config.supports_multidraw = false;
	backend_config.supports_hw_a2c = false;
	backend_config.supports_hw_a2c_1spp = false;
	backend_config.supports_hw_renormalization = false;
	backend_config.supports_hw_msaa = false;
	backend_config.supports_hw_a2one = false;
	backend_config.supports_hw_conditional_render = false;
	backend_config.supports_passthrough_dma = false;
	backend_config.supports_asynchronous_compute = false;
	backend_config.supports_host_gpu_labels = false;
	backend_config.supports_normalized_barycentrics = true;
}

MTLGSRender::~MTLGSRender()
{
	rsx_log.notice("MTLGSRender::~MTLGSRender()");
}

void MTLGSRender::on_init_thread()
{
	rsx_log.notice("MTLGSRender::on_init_thread()");

	constexpr u32 initial_width = 1280;
	constexpr u32 initial_height = 720;

	m_device = std::make_unique<rsx::metal::device>();
	m_device->report_capabilities();
	m_shader_cache = std::make_unique<rsx::metal::persistent_shader_cache>("v1.0");
	m_shader_cache->initialize();
	m_shader_cache->report();
	m_shader_compiler = std::make_unique<rsx::metal::shader_compiler>(*m_device, *m_shader_cache);
	m_shader_compiler->report();
	m_shader_library_cache = std::make_unique<rsx::metal::shader_library_cache>(*m_shader_compiler, *m_shader_cache);
	m_shader_library_cache->report();
	m_shader_recompiler = std::make_unique<rsx::metal::shader_recompiler>(*m_shader_cache);
	m_shader_recompiler->report();
	m_pipeline_cache = std::make_unique<rsx::metal::pipeline_cache>(*m_shader_compiler, *m_shader_cache);
	m_pipeline_cache->report();
	m_render_pipeline_cache = std::make_unique<rsx::metal::render_pipeline_cache>(*m_shader_compiler, *m_shader_cache, *m_pipeline_cache);
	m_render_pipeline_cache->report();
	m_render_target_cache = std::make_unique<rsx::metal::render_target_cache>(*m_device);
	m_render_target_cache->report();
	m_render_state_cache = std::make_unique<rsx::metal::render_state_cache>(*m_device);
	m_sampler_cache = std::make_unique<rsx::metal::sampler_cache>(*m_device);
	m_sampler_cache->report();
	m_texture_cache = std::make_unique<rsx::metal::sampled_texture_cache>(*m_device);
	m_texture_cache->report();
	m_draw_resources = std::make_unique<rsx::metal::draw_resource_manager>(*m_device);
	m_draw_resources->report();

	const rsx::metal::shader_interface_layout vertex_layout = rsx::metal::make_vertex_shader_interface_layout();
	const rsx::metal::shader_interface_layout fragment_layout = rsx::metal::make_fragment_shader_interface_layout(0, 0);
	m_vertex_argument_tables = std::make_unique<rsx::metal::argument_table_pool>(*m_device, vertex_layout.argument_table);
	get_fragment_argument_table_pool(fragment_layout);
	report_backend_state();

	m_window = std::make_unique<rsx::metal::native_window>(initial_width, initial_height);
	m_queue = std::make_unique<rsx::metal::command_queue>(*m_device);
	m_presentation = std::make_unique<rsx::metal::presentation_surface>(*m_device, *m_window, g_cfg.video.vsync != vsync_mode::off);

	m_window->set_title(Emu.GetFormattedTitle(0));
	m_window->show();
	present_clear_frame();
}

void MTLGSRender::on_exit()
{
	rsx_log.notice("MTLGSRender::on_exit()");

	if (m_queue)
	{
		m_queue->wait_idle();
	}

	if (m_pipeline_cache)
	{
		m_pipeline_cache->flush();
	}

	m_presentation.reset();
	m_render_target_cache.reset();
	m_render_state_cache.reset();
	m_sampler_cache.reset();
	m_texture_cache.reset();
	m_draw_resources.reset();
	m_vertex_argument_tables.reset();
	m_fragment_argument_table_pools.clear();
	m_queue.reset();
	m_window.reset();
	m_render_pipeline_cache.reset();
	m_pipeline_cache.reset();
	m_shader_recompiler.reset();
	m_shader_library_cache.reset();
	m_shader_compiler.reset();
	m_shader_cache.reset();
	m_device.reset();

	rsx::thread::on_exit();
}

void MTLGSRender::end()
{
	rsx_log.trace("MTLGSRender::end()");

	if (skip_current_frame)
	{
		return;
	}

	if (!m_device || !m_queue || !m_render_target_cache || !m_render_state_cache)
	{
		fmt::throw_exception("Metal backend draw requested before command queue, render target cache, and render state initialization");
	}

	m_graphics_state.clear(
		rsx::rtt_config_dirty |
		rsx::rtt_config_contested |
		rsx::rtt_config_valid |
		rsx::rtt_cache_state_dirty);

	get_framebuffer_layout(rsx::framebuffer_creation_context::context_draw, m_framebuffer_layout);
	if (!m_graphics_state.test(rsx::rtt_config_valid))
	{
		return;
	}

	const rsx::metal::draw_target_binding binding = m_render_target_cache->prepare_draw_targets(m_framebuffer_layout);
	update_draw_target_state(binding);
	on_framebuffer_layout_updated();
	const rsx::metal::render_pipeline_record pipeline = get_current_render_pipeline(binding);
	if (pipeline.mesh_pipeline)
	{
		rsx_log.todo("MTLGSRender::end() mesh draw encoding is not implemented");
		fmt::throw_exception("Metal backend mesh draw encoding is not implemented yet");
	}

	rsx::metal::command_frame& frame = m_queue->begin_frame();
	frame.use_residency_set(m_device->residency_set_handle());
	if (pipeline.has_fragment_layout)
	{
		prepare_fragment_textures(frame, pipeline.fragment_layout);
	}

	{
		rsx::metal::draw_render_encoder_scope draw_encoder(frame, binding);
		rsx::metal::bind_render_pipeline_state(frame, draw_encoder.encoder_handle(), pipeline);
		const rsx::metal::prepared_draw_command draw_command = preflight_draw_argument_tables(frame, draw_encoder.encoder_handle(), pipeline);
		rsx::metal::validate_prepared_draw_command(draw_command);
		const rsx::metal::dynamic_render_state_desc render_state = get_current_dynamic_render_state(draw_encoder);
		m_render_state_cache->bind_dynamic_render_state(frame, draw_encoder.encoder_handle(), render_state);
		rsx::metal::encode_draw_command(frame, draw_encoder.encoder_handle(), draw_command);
		rsx_log.trace("MTLGSRender::end() encoded draw command (encoder=*0x%x, pipeline=0x%llx, vertices=%u, indices=%u, indexed=%u, color_targets=%u, depth_stencil=%d)",
			draw_encoder.encoder_handle(),
			pipeline.pipeline_hash,
			draw_command.vertex_count,
			draw_command.index.index_count,
			static_cast<u32>(draw_command.indexed),
			draw_encoder.color_target_count(),
			draw_encoder.has_depth_stencil_target());
		draw_encoder.end_encoding();
	}

	frame.end();
	m_queue->submit_frame(frame, nullptr);
	rsx::thread::end();
}

void MTLGSRender::clear_surface(u32 arg)
{
	rsx_log.trace("MTLGSRender::clear_surface(arg=0x%x)", arg);

	if (skip_current_frame)
	{
		return;
	}

	if (!rsx::method_registers.stencil_mask())
	{
		arg &= ~RSX_GCM_CLEAR_STENCIL_BIT;
	}

	if (!(arg & RSX_GCM_CLEAR_ANY_MASK))
	{
		return;
	}

	const u32 color_mask = arg & RSX_GCM_CLEAR_COLOR_RGBA_MASK;
	if (color_mask && color_mask != RSX_GCM_CLEAR_COLOR_RGBA_MASK)
	{
		rsx_log.todo("MTLGSRender::clear_surface(arg=0x%x) partial color mask is not implemented", arg);
		fmt::throw_exception("Metal backend partial color surface clear is not implemented yet");
	}

	const u32 depth_stencil_mask = arg & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK;

	if (!m_queue || !m_render_target_cache)
	{
		fmt::throw_exception("Metal backend clear requested before command queue and render target cache initialization");
	}

	u8 context = rsx::framebuffer_creation_context::context_draw;
	if (color_mask)
	{
		context |= rsx::framebuffer_creation_context::context_clear_color;
	}

	if (depth_stencil_mask)
	{
		context |= rsx::framebuffer_creation_context::context_clear_depth;
	}

	get_framebuffer_layout(rsx::framebuffer_creation_context{context}, m_framebuffer_layout);
	if (!m_graphics_state.test(rsx::rtt_config_valid))
	{
		return;
	}

	const bool depth_stencil_format = is_depth_stencil_format(m_framebuffer_layout.depth_format);
	const bool clear_depth = !!(arg & RSX_GCM_CLEAR_DEPTH_BIT) && !!m_framebuffer_layout.zeta_address;
	const bool clear_stencil = !!(arg & RSX_GCM_CLEAR_STENCIL_BIT) && depth_stencil_format && !!m_framebuffer_layout.zeta_address;
	const bool has_depth_stencil_work = clear_depth || clear_stencil;

	if (clear_stencil && rsx::method_registers.stencil_mask() != 0xff)
	{
		rsx_log.todo("MTLGSRender::clear_surface(arg=0x%x) partial stencil mask is not implemented", arg);
		fmt::throw_exception("Metal backend partial stencil surface clear is not implemented yet");
	}

	if (has_depth_stencil_work)
	{
		if (m_framebuffer_layout.aa_mode != rsx::surface_antialiasing::center_1_sample)
		{
			rsx_log.todo("MTLGSRender::clear_surface(arg=0x%x) MSAA depth/stencil clear is not implemented", arg);
			fmt::throw_exception("Metal backend MSAA depth/stencil surface clear is not implemented yet");
		}

		if (m_framebuffer_layout.raster_type != rsx::surface_raster_type::linear)
		{
			rsx_log.todo("MTLGSRender::clear_surface(arg=0x%x) swizzled depth/stencil clear is not implemented", arg);
			fmt::throw_exception("Metal backend swizzled depth/stencil surface clear is not implemented yet");
		}
	}

	const auto rtt_indexes = rsx::utility::get_rtt_indexes(m_framebuffer_layout.target);
	const bool has_color_work = color_mask && !rtt_indexes.empty();
	if (!has_color_work && !has_depth_stencil_work)
	{
		return;
	}

	if (rsx::method_registers.scissor_origin_x() != 0 ||
		rsx::method_registers.scissor_origin_y() != 0 ||
		rsx::method_registers.scissor_width() < m_framebuffer_layout.width ||
		rsx::method_registers.scissor_height() < m_framebuffer_layout.height)
	{
		rsx_log.todo("MTLGSRender::clear_surface(arg=0x%x) scissored clear is not implemented", arg);
		fmt::throw_exception("Metal backend scissored surface clear is not implemented yet");
	}

	if (has_color_work && rtt_indexes.size() != 1)
	{
		rsx_log.todo("MTLGSRender::clear_surface(arg=0x%x) MRT color clear is not implemented", arg);
		fmt::throw_exception("Metal backend MRT color surface clear is not implemented yet");
	}

	u32 surface_index = umax;
	if (has_color_work)
	{
		surface_index = rtt_indexes[0];
		if (!m_framebuffer_layout.color_addresses[surface_index])
		{
			if (!has_depth_stencil_work)
			{
				return;
			}
		}
	}

	u8 clear_a = rsx::method_registers.clear_color_a();
	u8 clear_r = rsx::method_registers.clear_color_r();
	u8 clear_g = rsx::method_registers.clear_color_g();
	u8 clear_b = rsx::method_registers.clear_color_b();

	if (has_color_work && m_framebuffer_layout.color_addresses[surface_index])
	{
		switch (m_framebuffer_layout.color_format)
		{
		case rsx::surface_color_format::a8r8g8b8:
			break;
		case rsx::surface_color_format::a8b8g8r8:
			rsx::get_abgr8_clear_color(clear_r, clear_g, clear_b, clear_a);
			break;
		default:
			rsx_log.todo("MTLGSRender::clear_surface(arg=0x%x) color format 0x%x is not implemented",
				arg, static_cast<u32>(m_framebuffer_layout.color_format));
			fmt::throw_exception("Metal backend color surface clear format 0x%x is not implemented yet",
				static_cast<u32>(m_framebuffer_layout.color_format));
		}
	}

	if (has_depth_stencil_work)
	{
		const u32 clear_depth_value = rsx::method_registers.z_clear_value(depth_stencil_format);
		const u32 max_depth_value = get_max_depth_value(m_framebuffer_layout.depth_format);
		const rsx::metal::clear_depth_stencil clear =
		{
			.depth = clear_depth,
			.stencil = clear_stencil,
			.depth_value = static_cast<f64>(clear_depth_value) / static_cast<f64>(max_depth_value),
			.stencil_value = rsx::method_registers.stencil_clear_value(),
		};

		m_render_target_cache->clear_depth_stencil_target(*m_queue, m_framebuffer_layout, clear);

		m_depth_surface_info.address = m_framebuffer_layout.zeta_address;
		m_depth_surface_info.pitch = m_framebuffer_layout.actual_zeta_pitch;
		m_depth_surface_info.width = m_framebuffer_layout.width;
		m_depth_surface_info.height = m_framebuffer_layout.height;
		m_depth_surface_info.depth_format = m_framebuffer_layout.depth_format;
		m_depth_surface_info.bpp = get_format_block_size_in_bytes(m_framebuffer_layout.depth_format);
		m_depth_surface_info.samples = get_format_sample_count(m_framebuffer_layout.aa_mode);
	}

	if (has_color_work && m_framebuffer_layout.color_addresses[surface_index])
	{
		const rsx::metal::clear_color color =
		{
			.red = static_cast<f32>(clear_r) / 255.f,
			.green = static_cast<f32>(clear_g) / 255.f,
			.blue = static_cast<f32>(clear_b) / 255.f,
			.alpha = static_cast<f32>(clear_a) / 255.f,
		};

		m_render_target_cache->clear_color_target(*m_queue, m_framebuffer_layout, surface_index, color);

		m_surface_info[surface_index].address = m_framebuffer_layout.color_addresses[surface_index];
		m_surface_info[surface_index].pitch = m_framebuffer_layout.actual_color_pitch[surface_index];
		m_surface_info[surface_index].width = m_framebuffer_layout.width;
		m_surface_info[surface_index].height = m_framebuffer_layout.height;
		m_surface_info[surface_index].color_format = m_framebuffer_layout.color_format;
		m_surface_info[surface_index].bpp = get_format_block_size_in_bytes(m_framebuffer_layout.color_format);
		m_surface_info[surface_index].samples = get_format_sample_count(m_framebuffer_layout.aa_mode);
	}
}

void MTLGSRender::flip(const rsx::display_flip_info_t& info)
{
	rsx_log.trace("MTLGSRender::flip(buffer=%u, skip_frame=%d, emu_flip=%d, in_progress=%d)",
		info.buffer, info.skip_frame, info.emu_flip, info.in_progress);

	if (!info.skip_frame)
	{
		present_clear_frame();
	}

	rsx::thread::flip(info);
}

f64 MTLGSRender::get_display_refresh_rate() const
{
	rsx_log.trace("MTLGSRender::get_display_refresh_rate()");

	if (m_window)
	{
		return m_window->refresh_rate();
	}

	return 60.;
}

void MTLGSRender::report_backend_state() const
{
	rsx_log.notice("MTLGSRender::report_backend_state()");

	if (!m_device || !m_shader_cache || !m_shader_compiler || !m_shader_library_cache || !m_pipeline_cache || !m_render_pipeline_cache || !m_render_target_cache || !m_render_state_cache || !m_draw_resources)
	{
		rsx_log.warning("Metal backend state report skipped because initialization is incomplete");
		return;
	}

	const rsx::metal::device_caps& caps = m_device->caps();
	const rsx::metal::shader_cache_stats& cache_stats = m_shader_cache->stats();
	const rsx::metal::shader_compiler_stats compiler_stats = m_shader_compiler->stats();
	const rsx::metal::shader_library_cache_stats library_stats = m_shader_library_cache->stats();
	const rsx::metal::pipeline_cache_stats pipeline_stats = m_pipeline_cache->stats();
	const rsx::metal::render_pipeline_cache_stats render_pipeline_stats = m_render_pipeline_cache->stats();
	const rsx::metal::render_target_cache_stats render_target_stats = m_render_target_cache->stats();
	const rsx::metal::draw_resource_stats draw_resource_stats = m_draw_resources->stats();

	rsx_log.notice("Metal backend state: device=%s, metal4=%u, frames_in_flight=%u, residency_allocations=%u, residency_size=0x%llx",
		caps.name,
		static_cast<u32>(caps.metal4_supported),
		caps.frames_in_flight,
		m_device->residency_allocation_count(),
		m_device->residency_allocated_size());
	rsx_log.notice("Metal backend cache state: source_metadata=%u, completions=%u, translation_failures=%u, pipeline_entries=%u, pipeline_states=%u, dynamic_libraries=%u, pipeline_archives=%u",
		cache_stats.shader_entries,
		cache_stats.completion_entries,
		cache_stats.translation_failure_entries,
		cache_stats.pipeline_entries,
		cache_stats.pipeline_state_entries,
		cache_stats.library_entries,
		cache_stats.archive_entries);
	rsx_log.notice("Metal backend shader completion cache: available=%u, gated=%u",
		cache_stats.completion_available_entries,
		cache_stats.completion_gated_entries);
	rsx_log.notice("Metal backend pipeline entry cache: available=%u, gated=%u, mesh=%u",
		cache_stats.pipeline_entry_available_entries,
		cache_stats.pipeline_entry_gated_entries,
		cache_stats.mesh_pipeline_entry_entries);
	rsx_log.notice("Metal backend compiler/archive state: compiler_ready=%u, serializer_ready=%u, task_options_ready=%u, lookup_archive=%u, archive_metadata=%u, archive_metadata_invalid=%u, archive_loaded=%u, archive_load_failed=%u, archive_without_metadata=%u",
		static_cast<u32>(compiler_stats.compiler_ready),
		static_cast<u32>(compiler_stats.pipeline_serializer_ready),
		static_cast<u32>(compiler_stats.task_options_ready),
		static_cast<u32>(compiler_stats.lookup_archive_configured),
		static_cast<u32>(compiler_stats.archive_metadata_found),
		static_cast<u32>(compiler_stats.archive_metadata_invalid),
		static_cast<u32>(compiler_stats.archive_loaded),
		static_cast<u32>(compiler_stats.archive_load_failed),
		static_cast<u32>(compiler_stats.archive_without_metadata));
	rsx_log.notice("Metal backend shader-library workflow: memory_hits=%u, disk_hits=%u, disk_file_misses=%u, compiled=%u, retained=%u",
		library_stats.memory_hits,
		library_stats.loaded_libraries,
		library_stats.disk_file_misses,
		library_stats.compiled_libraries,
		library_stats.retained_libraries);
	rsx_log.notice("Metal backend shader-library metadata: source_misses=%u, source_invalid=%u, completion_misses=%u, completion_invalid=%u, library_misses=%u, library_invalid=%u",
		library_stats.source_metadata_misses,
		library_stats.source_metadata_invalid,
		library_stats.completion_metadata_misses,
		library_stats.completion_metadata_invalid,
		library_stats.library_metadata_misses,
		library_stats.library_metadata_invalid);
	rsx_log.notice("Metal backend shader-library failures: disk_load=%u, source_compile=%u, dynamic_library=%u, serialization=%u, memory_validation=%u, records=%u, disk_loaded=%u",
		library_stats.disk_load_failures,
		library_stats.source_compile_failures,
		library_stats.dynamic_library_failures,
		library_stats.serialization_failures,
		library_stats.memory_validation_failures,
		library_stats.retained_library_records,
		library_stats.disk_loaded_library_count);
	rsx_log.notice("Metal backend pipeline cache workflow: pending=%u, flushed=%u, archived=%u, archive_metadata_misses=%u, archive_metadata_invalid=%u, successful_flushes=%u, skipped_flushes=%u, serializer_failures=%u, script_failures=%u, archive_failures=%u",
		pipeline_stats.pending_pipeline_count,
		pipeline_stats.flushed_pipeline_count,
		pipeline_stats.archived_pipeline_count,
		pipeline_stats.archive_metadata_miss_count,
		static_cast<u32>(pipeline_stats.archive_metadata_invalid),
		pipeline_stats.successful_flush_count,
		pipeline_stats.skipped_flush_count,
		pipeline_stats.serializer_missing_failures,
		pipeline_stats.script_serialization_failures,
		pipeline_stats.archive_serialization_failures);
	rsx_log.notice("Metal backend render pipeline state cache: render_compiled=%u, render_hits=%u, render_retained=%u, mesh_compiled=%u, mesh_hits=%u, mesh_retained=%u",
		render_pipeline_stats.compiled_render_pipeline_count,
		render_pipeline_stats.render_pipeline_cache_hit_count,
		render_pipeline_stats.retained_render_pipeline_count,
		render_pipeline_stats.compiled_mesh_pipeline_count,
		render_pipeline_stats.mesh_pipeline_cache_hit_count,
		render_pipeline_stats.retained_mesh_pipeline_count);
	rsx_log.notice("Metal backend render pipeline metadata: hits=%u, misses=%u, mismatches=%u, invalid=%u",
		render_pipeline_stats.render_pipeline_metadata_hit_count,
		render_pipeline_stats.render_pipeline_metadata_miss_count,
		render_pipeline_stats.render_pipeline_metadata_mismatch_count,
		render_pipeline_stats.render_pipeline_metadata_invalid_count);
	rsx_log.notice("Metal backend mesh pipeline metadata: hits=%u, misses=%u, mismatches=%u, invalid=%u",
		render_pipeline_stats.mesh_pipeline_metadata_hit_count,
		render_pipeline_stats.mesh_pipeline_metadata_miss_count,
		render_pipeline_stats.mesh_pipeline_metadata_mismatch_count,
		render_pipeline_stats.mesh_pipeline_metadata_invalid_count);
	rsx_log.notice("Metal backend render pipeline compile failures: render=%u, mesh=%u",
		render_pipeline_stats.render_pipeline_compile_failure_count,
		render_pipeline_stats.mesh_pipeline_compile_failure_count);
	rsx_log.notice("Metal backend render target cache: color_targets=%u, created=%u, reused=%u, clears=%u",
		render_target_stats.color_target_count,
		render_target_stats.color_target_create_count,
		render_target_stats.color_target_reuse_count,
		render_target_stats.color_target_clear_count);
	rsx_log.notice("Metal backend depth/stencil target cache: targets=%u, created=%u, reused=%u, clears=%u",
		render_target_stats.depth_stencil_target_count,
		render_target_stats.depth_stencil_target_create_count,
		render_target_stats.depth_stencil_target_reuse_count,
		render_target_stats.depth_stencil_target_clear_count);
	rsx_log.notice("Metal backend draw target preparation: framebuffers=%u, color_binds=%u, depth_stencil_binds=%u",
		render_target_stats.draw_framebuffer_prepare_count,
		render_target_stats.draw_color_target_bind_count,
		render_target_stats.draw_depth_stencil_target_bind_count);
	rsx_log.notice("Metal backend draw resource uploads: retained=%u, created=%u, reused=%u, uploads=%u, bytes=0x%llx",
		draw_resource_stats.retained_buffer_count,
		draw_resource_stats.created_buffer_count,
		draw_resource_stats.reused_buffer_count,
		draw_resource_stats.uploaded_buffer_count,
		draw_resource_stats.uploaded_byte_count);
	rsx_log.notice("Metal backend draw argument table pools: vertex=%u, fragment=%u, fragment_variants=%u",
		m_vertex_argument_tables ? m_vertex_argument_tables->retained_table_count() : 0,
		retained_fragment_argument_table_count(),
		static_cast<u32>(m_fragment_argument_table_pools.size()));
	rsx_log.notice("Metal backend render state cache: depth_stencil_states=%u",
		m_render_state_cache ? m_render_state_cache->retained_depth_stencil_state_count() : 0);
	if (m_sampler_cache)
	{
		m_sampler_cache->report();
	}
	if (m_texture_cache)
	{
		m_texture_cache->report();
	}
	rsx_log.warning("Metal backend render pipeline entrypoints remain gated; helper MSL, shader source metadata, dynamic libraries, and pipeline archive plumbing are initialized only for validated Phase 4 paths");
}

void MTLGSRender::present_clear_frame()
{
	rsx_log.trace("MTLGSRender::present_clear_frame()");

	if (!m_presentation || !m_queue)
	{
		return;
	}

	m_presentation->present_clear_frame(*m_queue, 0.015f, 0.016f, 0.018f, 1.0f);
	m_frame_count++;
}

rsx::metal::render_pipeline_record MTLGSRender::get_current_render_pipeline(const rsx::metal::draw_target_binding& binding)
{
	rsx_log.trace("MTLGSRender::get_current_render_pipeline(color_target_count=%u, depth_stencil_target=%d, width=%u, height=%u)",
		binding.color_target_count,
		!!binding.depth_stencil_texture,
		binding.width,
		binding.height);

	if (!m_shader_recompiler || !m_render_pipeline_cache || !m_render_target_cache)
	{
		fmt::throw_exception("Metal backend draw pipeline requested before shader and pipeline caches are initialized");
	}

	if (binding.color_target_count != 1)
	{
		rsx_log.todo("MTLGSRender::get_current_render_pipeline(color_target_count=%u) MRT and depth-only pipeline binding are not implemented",
			binding.color_target_count);
		fmt::throw_exception("Metal backend draw pipeline binding requires exactly one color target for now");
	}

	u32 active_color_index = umax;
	u32 active_color_count = 0;
	for (u32 index = 0; index < rsx::limits::color_buffers_count; index++)
	{
		if (!binding.color_textures[index])
		{
			continue;
		}

		active_color_index = index;
		active_color_count++;
	}

	if (active_color_count != 1 || active_color_index == umax)
	{
		fmt::throw_exception("Metal backend draw pipeline color target binding is inconsistent: target_count=%u, active_count=%u",
			binding.color_target_count,
			active_color_count);
	}

	if (rsx::method_registers.msaa_alpha_to_coverage_enabled())
	{
		rsx_log.todo("MTLGSRender::get_current_render_pipeline() alpha-to-coverage is not implemented");
		fmt::throw_exception("Metal backend alpha-to-coverage pipeline state is not implemented yet");
	}

	if (rsx::method_registers.msaa_alpha_to_one_enabled())
	{
		rsx_log.todo("MTLGSRender::get_current_render_pipeline() alpha-to-one is not implemented");
		fmt::throw_exception("Metal backend alpha-to-one pipeline state is not implemented yet");
	}

	get_current_fragment_program(fs_sampler_state);
	ensure(current_fragment_program.valid);
	get_current_vertex_program(vs_sampler_state);

	const u32 vertex_id = static_cast<u32>(program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(current_vertex_program));
	const u32 fragment_id = static_cast<u32>(program_hash_util::fragment_program_utils::get_fragment_program_ucode_hash(current_fragment_program));
	const rsx::metal::translated_shader vertex_shader = m_shader_recompiler->translate_vertex_program(current_vertex_program, vertex_id);
	const rsx::metal::translated_shader fragment_shader = m_shader_recompiler->translate_fragment_program(current_fragment_program, fragment_id);

	const u32 fragment_texture_slots = get_fragment_texture_slot_count(current_fp_metadata.referenced_textures_mask);
	const rsx::metal::shader_interface_layout fragment_layout = rsx::metal::make_fragment_shader_interface_layout(fragment_texture_slots, fragment_texture_slots);
	const u32 color_pixel_format = m_render_target_cache->get_color_target_metal_pixel_format(m_framebuffer_layout, active_color_index);
	const u32 sample_count = get_format_sample_count(m_framebuffer_layout.aa_mode);
	const u32 topology_class = rsx::metal::get_render_pipeline_topology_class(rsx::method_registers.current_draw_clause.primitive);
	const u32 color_write_mask = make_metal_color_write_mask(active_color_index, m_framebuffer_layout.color_format);
	const b8 blend_enabled = !rsx::method_registers.logic_op_enabled() && get_metal_blend_enabled(active_color_index);
	const rsx::blend_factor source_rgb_blend_factor = blend_enabled ? rsx::method_registers.blend_func_sfactor_rgb() : rsx::blend_factor::one;
	const rsx::blend_factor source_alpha_blend_factor = blend_enabled ? rsx::method_registers.blend_func_sfactor_a() : rsx::blend_factor::one;
	const rsx::blend_factor destination_rgb_blend_factor = blend_enabled ? rsx::method_registers.blend_func_dfactor_rgb() : rsx::blend_factor::zero;
	const rsx::blend_factor destination_alpha_blend_factor = blend_enabled ? rsx::method_registers.blend_func_dfactor_a() : rsx::blend_factor::zero;
	const rsx::blend_equation rgb_blend_operation = blend_enabled ? rsx::method_registers.blend_equation_rgb() : rsx::blend_equation::add;
	const rsx::blend_equation alpha_blend_operation = blend_enabled ? rsx::method_registers.blend_equation_a() : rsx::blend_equation::add;
	const u64 pipeline_hash = make_draw_pipeline_hash(
		vertex_shader,
		fragment_shader,
		color_pixel_format,
		sample_count,
		topology_class,
		fragment_texture_slots,
		color_write_mask,
		blend_enabled,
		source_rgb_blend_factor,
		source_alpha_blend_factor,
		destination_rgb_blend_factor,
		destination_alpha_blend_factor,
		rgb_blend_operation,
		alpha_blend_operation);

	const rsx::metal::render_pipeline_desc desc =
	{
		.pipeline_hash = pipeline_hash,
		.label = fmt::format("RPCS3 Metal draw pipeline 0x%llx", pipeline_hash),
		.vertex = rsx::metal::make_render_pipeline_shader(vertex_shader),
		.fragment = rsx::metal::make_render_pipeline_shader(fragment_shader),
		.vertex_layout = rsx::metal::make_vertex_shader_interface_layout(),
		.fragment_layout = fragment_layout,
		.color_pixel_format = color_pixel_format,
		.raster_sample_count = sample_count,
		.input_primitive_topology = topology_class,
		.color_write_mask = color_write_mask,
		.blend_enabled = blend_enabled,
		.source_rgb_blend_factor = source_rgb_blend_factor,
		.source_alpha_blend_factor = source_alpha_blend_factor,
		.destination_rgb_blend_factor = destination_rgb_blend_factor,
		.destination_alpha_blend_factor = destination_alpha_blend_factor,
		.rgb_blend_operation = rgb_blend_operation,
		.alpha_blend_operation = alpha_blend_operation,
		.rasterization_enabled = true,
	};

	return m_render_pipeline_cache->get_or_compile_render_pipeline(desc);
}

rsx::metal::dynamic_render_state_desc MTLGSRender::get_current_dynamic_render_state(const rsx::metal::draw_render_encoder_scope& draw_encoder)
{
	rsx_log.trace("MTLGSRender::get_current_dynamic_render_state(width=%u, height=%u, color_targets=%u, depth_stencil=%u)",
		draw_encoder.width(),
		draw_encoder.height(),
		draw_encoder.color_target_count(),
		static_cast<u32>(draw_encoder.has_depth_stencil_target()));

	const auto [clip_width, clip_height] = rsx::apply_resolution_scale<true>(
		resolution_scaling_config,
		rsx::method_registers.surface_clip_width(),
		rsx::method_registers.surface_clip_height());

	areau scissor_region{};
	if (get_scissor(scissor_region, true))
	{
		m_draw_scissor =
		{
			.x = scissor_region.x1,
			.y = scissor_region.y1,
			.width = scissor_region.width(),
			.height = scissor_region.height(),
		};
		m_draw_scissor_valid = true;
	}

	if (!m_draw_scissor_valid)
	{
		m_draw_scissor =
		{
			.x = 0,
			.y = 0,
			.width = clip_width,
			.height = clip_height,
		};
		m_draw_scissor_valid = true;
	}

	u32 active_color_index = 0;
	for (u32 index = 0; index < rsx::limits::color_buffers_count; index++)
	{
		if (!m_framebuffer_layout.color_addresses[index])
		{
			continue;
		}

		active_color_index = index;
		break;
	}

	const u32 color_write_mask = draw_encoder.color_target_count() == 0 ? 0xf : make_metal_color_write_mask(active_color_index, m_framebuffer_layout.color_format);
	const b8 color_write_all = draw_encoder.color_target_count() == 0 || color_write_mask == 0xf;
	const b8 blend_enabled = draw_encoder.color_target_count() != 0 && !rsx::method_registers.logic_op_enabled() && get_metal_blend_enabled(active_color_index);
	const std::array<float, 4> blend_colors = rsx::get_constant_blend_colors();

	const b8 has_depth_stencil_target = draw_encoder.has_depth_stencil_target();
	const b8 has_stencil_attachment = draw_encoder.has_stencil_attachment();

	return
	{
		.viewport =
		{
			.x = 0.,
			.y = 0.,
			.width = static_cast<f64>(clip_width),
			.height = static_cast<f64>(clip_height),
			.z_near = rsx::method_registers.clip_min(),
			.z_far = rsx::method_registers.clip_max(),
		},
		.scissor = m_draw_scissor,
		.render_width = draw_encoder.width(),
		.render_height = draw_encoder.height(),
		.primitive = rsx::method_registers.current_draw_clause.primitive,
		.front_face = rsx::method_registers.front_face_mode(),
		.cull_face = rsx::method_registers.cull_face_mode(),
		.front_polygon_mode = rsx::method_registers.polygon_mode_front(),
		.back_polygon_mode = rsx::method_registers.polygon_mode_back(),
		.depth_func = rsx::method_registers.depth_func(),
		.stencil_func = rsx::method_registers.stencil_func(),
		.back_stencil_func = rsx::method_registers.back_stencil_func(),
		.stencil_fail = rsx::method_registers.stencil_op_fail(),
		.stencil_zfail = rsx::method_registers.stencil_op_zfail(),
		.stencil_zpass = rsx::method_registers.stencil_op_zpass(),
		.back_stencil_fail = rsx::method_registers.back_stencil_op_fail(),
		.back_stencil_zfail = rsx::method_registers.back_stencil_op_zfail(),
		.back_stencil_zpass = rsx::method_registers.back_stencil_op_zpass(),
		.stencil_ref = rsx::method_registers.stencil_func_ref(),
		.back_stencil_ref = rsx::method_registers.back_stencil_func_ref(),
		.stencil_read_mask = rsx::method_registers.stencil_func_mask(),
		.back_stencil_read_mask = rsx::method_registers.back_stencil_func_mask(),
		.stencil_write_mask = rsx::method_registers.stencil_mask(),
		.back_stencil_write_mask = rsx::method_registers.back_stencil_mask(),
		.depth_bounds_min = rsx::method_registers.depth_bounds_min(),
		.depth_bounds_max = rsx::method_registers.depth_bounds_max(),
		.blend_color_red = blend_colors[0],
		.blend_color_green = blend_colors[1],
		.blend_color_blue = blend_colors[2],
		.blend_color_alpha = blend_colors[3],
		.polygon_offset_scale = rsx::method_registers.poly_offset_scale(),
		.polygon_offset_bias = rsx::method_registers.poly_offset_bias(),
		.line_width = rsx::method_registers.line_width() * resolution_scaling_config.scale_factor(),
		.has_depth_stencil_target = has_depth_stencil_target,
		.has_stencil_attachment = has_stencil_attachment,
		.cull_face_enabled = rsx::method_registers.cull_face_enabled(),
		.depth_clip_enabled = rsx::method_registers.depth_clip_enabled(),
		.depth_clamp_enabled = rsx::method_registers.depth_clamp_enabled(),
		.depth_bounds_test_enabled = has_depth_stencil_target && rsx::method_registers.depth_bounds_test_enabled(),
		.depth_test_enabled = has_depth_stencil_target && rsx::method_registers.depth_test_enabled(),
		.depth_write_enabled = has_depth_stencil_target && rsx::method_registers.depth_write_enabled(),
		.stencil_test_enabled = has_stencil_attachment && rsx::method_registers.stencil_test_enabled(),
		.two_sided_stencil_test_enabled = rsx::method_registers.two_sided_stencil_test_enabled(),
		.polygon_offset_fill_enabled = rsx::method_registers.poly_offset_fill_enabled(),
		.polygon_offset_line_enabled = rsx::method_registers.poly_offset_line_enabled(),
		.polygon_offset_point_enabled = rsx::method_registers.poly_offset_point_enabled(),
		.blend_enabled = blend_enabled,
		.logic_op_enabled = rsx::method_registers.logic_op_enabled(),
		.alpha_test_enabled = rsx::method_registers.alpha_test_enabled(),
		.dither_enabled = rsx::method_registers.dither_enabled(),
		.line_smooth_enabled = rsx::method_registers.line_smooth_enabled(),
		.poly_smooth_enabled = rsx::method_registers.poly_smooth_enabled(),
		.polygon_stipple_enabled = rsx::method_registers.polygon_stipple_enabled(),
		.color_write_all = color_write_all,
	};
}

rsx::metal::prepared_draw_command MTLGSRender::prepare_draw_vertex_inputs(rsx::metal::command_frame& frame, rsx::metal::argument_table& vertex_table, const rsx::metal::shader_interface_layout& layout)
{
	rsx_log.trace("MTLGSRender::prepare_draw_vertex_inputs(frame_index=%u, layout_stage=%u, vertex_layout=%u, persistent=%u, volatile=%u)",
		frame.frame_index(),
		static_cast<u32>(layout.stage),
		layout.vertex_layout_buffer_index,
		layout.persistent_vertex_buffer_index,
		layout.volatile_vertex_buffer_index);

	if (!m_draw_resources)
	{
		fmt::throw_exception("Metal backend draw resources are not initialized");
	}

	if (layout.vertex_layout_buffer_index == rsx::metal::shader_binding_none ||
		layout.persistent_vertex_buffer_index == rsx::metal::shader_binding_none ||
		layout.volatile_vertex_buffer_index == rsx::metal::shader_binding_none)
	{
		fmt::throw_exception("Metal vertex input binding requires layout, persistent stream, and volatile stream slots");
	}

	if (layout.vertex_buffer_count)
	{
		fmt::throw_exception("Metal vertex input binding uses packed RSX fetch streams, found %u per-attribute slots",
			layout.vertex_buffer_count);
	}

	m_draw_processor.analyse_inputs_interleaved(m_vertex_layout, current_vp_metadata);
	if (!m_vertex_layout.validate())
	{
		rsx_log.todo("MTLGSRender::prepare_draw_vertex_inputs() no valid vertex inputs are available");
		fmt::throw_exception("Metal backend draw without valid vertex inputs is not implemented yet");
	}

	const metal_vertex_input_state vertex_state = std::visit(
		metal_vertex_input_state_visitor{ m_vertex_layout, frame, *m_draw_resources },
		m_draw_processor.get_draw_command(rsx::method_registers));

	if (!vertex_state.vertex_draw_count || !vertex_state.allocated_vertex_count)
	{
		rsx_log.todo("MTLGSRender::prepare_draw_vertex_inputs() empty vertex input draw is not implemented");
		fmt::throw_exception("Metal backend empty vertex input draw is not implemented yet");
	}

	const std::pair<u32, u32> required = calculate_memory_requirements(
		m_vertex_layout,
		vertex_state.first_vertex,
		vertex_state.allocated_vertex_count);

	rsx::metal::draw_buffer_binding persistent_stream{};
	if (required.first)
	{
		persistent_stream = m_draw_resources->upload_generated_buffer(
			frame,
			required.first,
			"RPCS3 Metal persistent vertex stream",
			[this, first_vertex = vertex_state.first_vertex, vertex_count = vertex_state.allocated_vertex_count](void* data, u64 size)
			{
				if (size > std::numeric_limits<u32>::max())
				{
					fmt::throw_exception("Metal persistent vertex stream upload exceeds u32 range: size=0x%llx", size);
				}

				m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, first_vertex, vertex_count, data, nullptr);
			});
	}

	rsx::metal::draw_buffer_binding volatile_stream{};
	if (required.second)
	{
		volatile_stream = m_draw_resources->upload_generated_buffer(
			frame,
			required.second,
			"RPCS3 Metal volatile vertex stream",
			[this, first_vertex = vertex_state.first_vertex, vertex_count = vertex_state.allocated_vertex_count](void* data, u64 size)
			{
				if (size > std::numeric_limits<u32>::max())
				{
					fmt::throw_exception("Metal volatile vertex stream upload exceeds u32 range: size=0x%llx", size);
				}

				m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, first_vertex, vertex_count, nullptr, data);
			});
	}

	rsx::metal::buffer& zero_vertex_buffer = m_draw_resources->zero_vertex_buffer();
	constexpr u64 vertex_layout_state_size = sizeof(s32) * 2 * rsx::metal::shader_vertex_input_count;
	const rsx::metal::draw_buffer_binding vertex_layout_state = m_draw_resources->upload_generated_buffer(
		frame,
		vertex_layout_state_size,
		"RPCS3 Metal vertex fetch layout",
		[this, first_vertex = vertex_state.first_vertex, vertex_count = vertex_state.allocated_vertex_count](void* data, u64 size)
		{
			if (size != vertex_layout_state_size)
			{
				fmt::throw_exception("Metal vertex fetch layout buffer size mismatch: size=0x%llx", size);
			}

			std::memset(data, 0, static_cast<usz>(size));
			m_draw_processor.fill_vertex_layout_state(
				m_vertex_layout,
				current_vp_metadata,
				first_vertex,
				vertex_count,
				static_cast<s32*>(data),
				0,
				0);
		});

	if (!vertex_layout_state.resource)
	{
		fmt::throw_exception("Metal vertex fetch layout upload returned an empty buffer");
	}

	if (required.first && !persistent_stream.resource)
	{
		fmt::throw_exception("Metal persistent vertex stream upload returned an empty buffer");
	}

	if (required.second && !volatile_stream.resource)
	{
		fmt::throw_exception("Metal volatile vertex stream upload returned an empty buffer");
	}

	vertex_table.bind_buffer_address(layout.vertex_layout_buffer_index, *vertex_layout_state.resource, vertex_layout_state.offset);

	vertex_table.bind_buffer_address(
		layout.persistent_vertex_buffer_index,
		required.first ? *persistent_stream.resource : zero_vertex_buffer,
		required.first ? persistent_stream.offset : 0);

	vertex_table.bind_buffer_address(
		layout.volatile_vertex_buffer_index,
		required.second ? *volatile_stream.resource : zero_vertex_buffer,
		required.second ? volatile_stream.offset : 0);

	rsx_log.trace("Metal vertex input upload: draw_vertices=%u allocated_vertices=%u first_vertex=%u persistent=0x%x volatile=0x%x indexed=%u index_count=%u index_type=%u index_range=%u..%u base_vertex=%lld",
		vertex_state.vertex_draw_count,
		vertex_state.allocated_vertex_count,
		vertex_state.first_vertex,
		required.first,
		required.second,
		static_cast<u32>(vertex_state.indexed),
		vertex_state.index_buffer.index_count,
		static_cast<u32>(vertex_state.index_buffer.index_type),
		vertex_state.index_buffer.min_index,
		vertex_state.index_buffer.max_index,
		vertex_state.base_vertex);

	return make_prepared_draw_command(vertex_state);
}

rsx::metal::prepared_draw_command MTLGSRender::preflight_draw_argument_tables(rsx::metal::command_frame& frame, void* render_encoder_handle, const rsx::metal::render_pipeline_record& pipeline)
{
	rsx_log.trace("MTLGSRender::preflight_draw_argument_tables(frame_index=%u, render_encoder_handle=*0x%x, pipeline_hash=0x%llx, mesh_pipeline=%u, has_fragment_layout=%u)",
		frame.frame_index(),
		render_encoder_handle,
		pipeline.pipeline_hash,
		static_cast<u32>(pipeline.mesh_pipeline),
		static_cast<u32>(pipeline.has_fragment_layout));

	if (!render_encoder_handle)
	{
		fmt::throw_exception("Metal backend draw argument table preflight requires a valid render encoder");
	}

	if (!m_vertex_argument_tables)
	{
		fmt::throw_exception("Metal backend vertex argument table pool is not initialized");
	}

	if (!m_draw_resources)
	{
		fmt::throw_exception("Metal backend draw resources are not initialized");
	}

	if (pipeline.mesh_pipeline)
	{
		rsx_log.todo("MTLGSRender::preflight_draw_argument_tables() mesh argument table binding is not implemented");
		fmt::throw_exception("Metal backend mesh argument table binding is not implemented yet");
	}

	rsx::metal::argument_table& vertex_table = m_vertex_argument_tables->acquire();
	vertex_table.validate_shader_layout(pipeline.primary_layout);
	if (pipeline.primary_layout.constants_buffer_index == rsx::metal::shader_binding_none)
	{
		fmt::throw_exception("Metal vertex draw argument layout requires a constants buffer slot");
	}

	constexpr u64 vertex_context_size = 96;
	const rsx::metal::draw_buffer_binding vertex_context = m_draw_resources->upload_generated_buffer(
		frame,
		vertex_context_size,
		"RPCS3 Metal vertex context",
		[this](void* data, u64 size)
		{
			if (size != vertex_context_size)
			{
				fmt::throw_exception("Metal vertex context buffer size mismatch: size=0x%llx", size);
			}

			u8* bytes = static_cast<u8*>(data);
			m_draw_processor.fill_scale_offset_data(bytes, false);
			m_draw_processor.fill_user_clip_data(bytes + 64);
			*(reinterpret_cast<u32*>(bytes + 68)) = rsx::method_registers.transform_branch_bits();
			*(reinterpret_cast<f32*>(bytes + 72)) = rsx::method_registers.point_size() * resolution_scaling_config.scale_factor();
			*(reinterpret_cast<f32*>(bytes + 76)) = rsx::method_registers.clip_min();
			*(reinterpret_cast<f32*>(bytes + 80)) = rsx::method_registers.clip_max();
		});
	vertex_table.bind_buffer_address(pipeline.primary_layout.constants_buffer_index, *vertex_context.resource, vertex_context.offset);
	const rsx::metal::prepared_draw_command draw_command = prepare_draw_vertex_inputs(frame, vertex_table, pipeline.primary_layout);

	rsx::metal::argument_table* fragment_table_ptr = nullptr;
	if (pipeline.has_fragment_layout)
	{
		rsx::metal::argument_table_pool& fragment_pool = get_fragment_argument_table_pool(pipeline.fragment_layout);
		rsx::metal::argument_table& fragment_table = fragment_pool.acquire();
		fragment_table_ptr = &fragment_table;
		fragment_table.validate_shader_layout(pipeline.fragment_layout);
		if (pipeline.fragment_layout.constants_buffer_index == rsx::metal::shader_binding_none)
		{
			fmt::throw_exception("Metal fragment draw argument layout requires a constants buffer slot");
		}

		constexpr u64 fragment_context_size = 32;
		const rsx::metal::draw_buffer_binding fragment_context = m_draw_resources->upload_generated_buffer(
			frame,
			fragment_context_size,
			"RPCS3 Metal fragment context",
			[this](void* data, u64 size)
			{
				if (size != fragment_context_size)
				{
					fmt::throw_exception("Metal fragment context buffer size mismatch: size=0x%llx", size);
				}

				m_draw_processor.fill_fragment_state_buffer(data, current_fragment_program);
			});
		fragment_table.bind_buffer_address(pipeline.fragment_layout.constants_buffer_index, *fragment_context.resource, fragment_context.offset);
		bind_fragment_textures(fragment_table, pipeline.fragment_layout);
		bind_fragment_samplers(fragment_table, pipeline.fragment_layout);
	}

	rsx::metal::bind_pipeline_arguments(frame, render_encoder_handle, pipeline, vertex_table, fragment_table_ptr);
	return draw_command;
}

rsx::metal::argument_table_pool& MTLGSRender::get_fragment_argument_table_pool(const rsx::metal::shader_interface_layout& layout)
{
	rsx_log.trace("MTLGSRender::get_fragment_argument_table_pool(stage=%u, buffers=%u, textures=%u, samplers=%u)",
		static_cast<u32>(layout.stage),
		layout.argument_table.max_buffers,
		layout.argument_table.max_textures,
		layout.argument_table.max_samplers);

	rsx::metal::validate_shader_interface_layout(layout);
	if (layout.stage != rsx::metal::shader_stage::fragment)
	{
		fmt::throw_exception("Metal fragment argument table pool requested for shader stage %u",
			static_cast<u32>(layout.stage));
	}

	if (!m_device)
	{
		fmt::throw_exception("Metal fragment argument table pool requested before device initialization");
	}

	const rsx::metal::argument_table_desc& desc = layout.argument_table;
	for (fragment_argument_table_pool_record& record : m_fragment_argument_table_pools)
	{
		if (record.max_buffers == desc.max_buffers &&
			record.max_textures == desc.max_textures &&
			record.max_samplers == desc.max_samplers &&
			record.support_attribute_strides == desc.support_attribute_strides)
		{
			ensure(record.pool);
			return *record.pool;
		}
	}

	fragment_argument_table_pool_record record =
	{
		.max_buffers = desc.max_buffers,
		.max_textures = desc.max_textures,
		.max_samplers = desc.max_samplers,
		.support_attribute_strides = desc.support_attribute_strides,
		.pool = std::make_unique<rsx::metal::argument_table_pool>(*m_device, desc),
	};

	m_fragment_argument_table_pools.emplace_back(std::move(record));
	return *m_fragment_argument_table_pools.back().pool;
}

void MTLGSRender::prepare_fragment_textures(rsx::metal::command_frame& frame, const rsx::metal::shader_interface_layout& layout)
{
	rsx_log.trace("MTLGSRender::prepare_fragment_textures(frame_index=%u, texture_count=%u, referenced=0x%x)",
		frame.frame_index(),
		layout.texture_count,
		current_fp_metadata.referenced_textures_mask);

	rsx::metal::validate_shader_interface_layout(layout);
	if (layout.stage != rsx::metal::shader_stage::fragment)
	{
		fmt::throw_exception("Metal fragment texture preparation requested for shader stage %u",
			static_cast<u32>(layout.stage));
	}

	if (!layout.texture_count)
	{
		return;
	}

	if (!m_texture_cache)
	{
		fmt::throw_exception("Metal fragment texture preparation requested before texture cache initialization");
	}

	for (u32 texture_index = 0; texture_index < layout.texture_count; texture_index++)
	{
		const u32 reference_mask = 1u << texture_index;
		if ((current_fp_metadata.referenced_textures_mask & reference_mask) == 0)
		{
			rsx_log.todo("MTLGSRender::prepare_fragment_textures() sparse fragment texture layouts are not implemented");
			fmt::throw_exception("Metal backend sparse fragment texture layout is not implemented yet");
		}

		const rsx::fragment_texture& texture = rsx::method_registers.fragment_textures[texture_index];
		if (!texture.enabled())
		{
			rsx_log.todo("MTLGSRender::prepare_fragment_textures() disabled referenced fragment texture %u is not implemented", texture_index);
			fmt::throw_exception("Metal backend referenced fragment texture %u is disabled", texture_index);
		}

		m_texture_cache->prepare_fragment_texture(frame, texture, texture_index);
	}
}

void MTLGSRender::bind_fragment_textures(rsx::metal::argument_table& fragment_table, const rsx::metal::shader_interface_layout& layout)
{
	rsx_log.trace("MTLGSRender::bind_fragment_textures(texture_count=%u, referenced=0x%x)",
		layout.texture_count,
		current_fp_metadata.referenced_textures_mask);

	rsx::metal::validate_shader_interface_layout(layout);
	if (layout.stage != rsx::metal::shader_stage::fragment)
	{
		fmt::throw_exception("Metal fragment texture binding requested for shader stage %u",
			static_cast<u32>(layout.stage));
	}

	if (!layout.texture_count)
	{
		return;
	}

	if (layout.texture_base_index == rsx::metal::shader_binding_none)
	{
		fmt::throw_exception("Metal fragment texture binding requires a valid texture base index");
	}

	if (!m_texture_cache)
	{
		fmt::throw_exception("Metal fragment texture binding requested before texture cache initialization");
	}

	for (u32 texture_index = 0; texture_index < layout.texture_count; texture_index++)
	{
		const u32 reference_mask = 1u << texture_index;
		if ((current_fp_metadata.referenced_textures_mask & reference_mask) == 0)
		{
			rsx_log.todo("MTLGSRender::bind_fragment_textures() sparse fragment texture layouts are not implemented");
			fmt::throw_exception("Metal backend sparse fragment texture layout is not implemented yet");
		}

		const rsx::fragment_texture& texture = rsx::method_registers.fragment_textures[texture_index];
		if (!texture.enabled())
		{
			rsx_log.todo("MTLGSRender::bind_fragment_textures() disabled referenced fragment texture %u is not implemented", texture_index);
			fmt::throw_exception("Metal backend referenced fragment texture %u is disabled", texture_index);
		}

		rsx::metal::texture& sampled_texture = m_texture_cache->get_fragment_texture(texture, texture_index);
		fragment_table.bind_texture(layout.texture_base_index + texture_index, sampled_texture);
	}
}

void MTLGSRender::bind_fragment_samplers(rsx::metal::argument_table& fragment_table, const rsx::metal::shader_interface_layout& layout)
{
	rsx_log.trace("MTLGSRender::bind_fragment_samplers(texture_count=%u, sampler_count=%u, referenced=0x%x)",
		layout.texture_count,
		layout.sampler_count,
		current_fp_metadata.referenced_textures_mask);

	rsx::metal::validate_shader_interface_layout(layout);
	if (layout.stage != rsx::metal::shader_stage::fragment)
	{
		fmt::throw_exception("Metal fragment sampler binding requested for shader stage %u",
			static_cast<u32>(layout.stage));
	}

	if (!layout.sampler_count)
	{
		return;
	}

	if (layout.sampler_base_index == rsx::metal::shader_binding_none)
	{
		fmt::throw_exception("Metal fragment sampler binding requires a valid sampler base index");
	}

	if (!m_sampler_cache)
	{
		fmt::throw_exception("Metal fragment sampler binding requested before sampler cache initialization");
	}

	for (u32 texture_index = 0; texture_index < layout.sampler_count; texture_index++)
	{
		const u32 reference_mask = 1u << texture_index;
		if ((current_fp_metadata.referenced_textures_mask & reference_mask) == 0)
		{
			rsx_log.todo("MTLGSRender::bind_fragment_samplers() sparse fragment sampler layouts are not implemented");
			fmt::throw_exception("Metal backend sparse fragment sampler layout is not implemented yet");
		}

		const rsx::fragment_texture& texture = rsx::method_registers.fragment_textures[texture_index];
		if (!texture.enabled())
		{
			rsx_log.todo("MTLGSRender::bind_fragment_samplers() disabled referenced fragment texture %u is not implemented", texture_index);
			fmt::throw_exception("Metal backend referenced fragment texture %u is disabled", texture_index);
		}

		rsx::metal::sampler& sampler = m_sampler_cache->get_fragment_sampler(texture, texture.get_exact_mipmap_count() > 1);
		fragment_table.bind_sampler(layout.sampler_base_index + texture_index, sampler);
	}
}

u32 MTLGSRender::retained_fragment_argument_table_count() const
{
	rsx_log.trace("MTLGSRender::retained_fragment_argument_table_count()");

	u32 count = 0;
	for (const fragment_argument_table_pool_record& record : m_fragment_argument_table_pools)
	{
		if (!record.pool)
		{
			continue;
		}

		const u32 pool_count = record.pool->retained_table_count();
		if (count > std::numeric_limits<u32>::max() - pool_count)
		{
			fmt::throw_exception("Metal fragment argument table retained count overflow");
		}

		count += pool_count;
	}

	return count;
}

void MTLGSRender::update_draw_target_state(const rsx::metal::draw_target_binding& binding)
{
	rsx_log.trace("MTLGSRender::update_draw_target_state(color_target_count=%u, depth_stencil_target=%d, width=%u, height=%u)",
		binding.color_target_count, !!binding.depth_stencil_texture, binding.width, binding.height);

	const u32 color_bpp = get_format_block_size_in_bytes(m_framebuffer_layout.color_format);
	const u8 samples = get_format_sample_count(m_framebuffer_layout.aa_mode);

	for (u8 index = 0; index < rsx::limits::color_buffers_count; index++)
	{
		m_surface_info[index].address = 0;
		m_surface_info[index].pitch = 0;
		m_surface_info[index].width = binding.width;
		m_surface_info[index].height = binding.height;
		m_surface_info[index].color_format = m_framebuffer_layout.color_format;
		m_surface_info[index].bpp = color_bpp;
		m_surface_info[index].samples = samples;

		if (!binding.color_textures[index])
		{
			continue;
		}

		m_surface_info[index].address = m_framebuffer_layout.color_addresses[index];
		m_surface_info[index].pitch = m_framebuffer_layout.actual_color_pitch[index];
	}

	m_depth_surface_info.address = 0;
	m_depth_surface_info.pitch = 0;
	m_depth_surface_info.width = binding.width;
	m_depth_surface_info.height = binding.height;
	m_depth_surface_info.depth_format = m_framebuffer_layout.depth_format;
	m_depth_surface_info.bpp = get_format_block_size_in_bytes(m_framebuffer_layout.depth_format);
	m_depth_surface_info.samples = samples;

	if (binding.depth_stencil_texture)
	{
		m_depth_surface_info.address = m_framebuffer_layout.zeta_address;
		m_depth_surface_info.pitch = m_framebuffer_layout.actual_zeta_pitch;
	}
}
