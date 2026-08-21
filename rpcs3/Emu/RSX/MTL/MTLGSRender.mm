#include "stdafx.h"
#include "MTLGSRender.h"

#include "Emu/Memory/vm_locking.h"
#include "Emu/RSX/NV47/HW/context_accessors.define.h"
#include "Emu/RSX/Overlays/Shaders/shader_loading_dialog_native.h"
#include "Emu/RSX/rsx_methods.h"
#include "mtlutils/swapchain_macos.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <set>

namespace mtl
{
	namespace
	{
		[[nodiscard]] compare_function comparison(rsx::comparison_function value)
		{
			switch (value)
			{
			case rsx::comparison_function::never: return compare_function::never;
			case rsx::comparison_function::less: return compare_function::less;
			case rsx::comparison_function::equal: return compare_function::equal;
			case rsx::comparison_function::less_or_equal: return compare_function::less_equal;
			case rsx::comparison_function::greater: return compare_function::greater;
			case rsx::comparison_function::not_equal: return compare_function::not_equal;
			case rsx::comparison_function::greater_or_equal: return compare_function::greater_equal;
			case rsx::comparison_function::always: return compare_function::always;
			}
			fmt::throw_exception("Invalid RSX comparison function 0x%x", static_cast<u32>(value));
		}

		[[nodiscard]] blend_factor blend(rsx::blend_factor value)
		{
			switch (value)
			{
			case rsx::blend_factor::zero: return blend_factor::zero;
			case rsx::blend_factor::one: return blend_factor::one;
			case rsx::blend_factor::src_color: return blend_factor::source_color;
			case rsx::blend_factor::one_minus_src_color: return blend_factor::one_minus_source_color;
			case rsx::blend_factor::src_alpha: return blend_factor::source_alpha;
			case rsx::blend_factor::one_minus_src_alpha: return blend_factor::one_minus_source_alpha;
			case rsx::blend_factor::dst_color: return blend_factor::destination_color;
			case rsx::blend_factor::one_minus_dst_color: return blend_factor::one_minus_destination_color;
			case rsx::blend_factor::dst_alpha: return blend_factor::destination_alpha;
			case rsx::blend_factor::one_minus_dst_alpha: return blend_factor::one_minus_destination_alpha;
			case rsx::blend_factor::src_alpha_saturate: return blend_factor::source_alpha_saturated;
			case rsx::blend_factor::constant_color: return blend_factor::blend_color;
			case rsx::blend_factor::one_minus_constant_color: return blend_factor::one_minus_blend_color;
			case rsx::blend_factor::constant_alpha: return blend_factor::blend_alpha;
			case rsx::blend_factor::one_minus_constant_alpha: return blend_factor::one_minus_blend_alpha;
			}
			fmt::throw_exception("Invalid RSX blend factor 0x%x", static_cast<u32>(value));
		}

		[[nodiscard]] blend_operation blend_equation(rsx::blend_equation value)
		{
			switch (value)
			{
			case rsx::blend_equation::add:
			case rsx::blend_equation::add_signed:
			case rsx::blend_equation::reverse_add_signed: return blend_operation::add;
			case rsx::blend_equation::subtract: return blend_operation::subtract;
			case rsx::blend_equation::reverse_subtract:
			case rsx::blend_equation::reverse_subtract_signed: return blend_operation::reverse_subtract;
			case rsx::blend_equation::min: return blend_operation::minimum;
			case rsx::blend_equation::max: return blend_operation::maximum;
			}
			fmt::throw_exception("Invalid RSX blend equation 0x%x", static_cast<u32>(value));
		}

		[[nodiscard]] stencil_operation stencil(rsx::stencil_op value)
		{
			switch (value)
			{
			case rsx::stencil_op::keep: return stencil_operation::keep;
			case rsx::stencil_op::zero: return stencil_operation::zero;
			case rsx::stencil_op::replace: return stencil_operation::replace;
			case rsx::stencil_op::incr: return stencil_operation::increment_clamp;
			case rsx::stencil_op::decr: return stencil_operation::decrement_clamp;
			case rsx::stencil_op::invert: return stencil_operation::invert;
			case rsx::stencil_op::incr_wrap: return stencil_operation::increment_wrap;
			case rsx::stencil_op::decr_wrap: return stencil_operation::decrement_wrap;
			}
			fmt::throw_exception("Invalid RSX stencil operation 0x%x", static_cast<u32>(value));
		}

		[[nodiscard]] logic_operation logic(rsx::logic_op value)
		{
			switch (value)
			{
			case rsx::logic_op::logic_clear: return logic_operation::clear;
			case rsx::logic_op::logic_and: return logic_operation::and_;
			case rsx::logic_op::logic_and_reverse: return logic_operation::and_reverse;
			case rsx::logic_op::logic_copy: return logic_operation::copy;
			case rsx::logic_op::logic_and_inverted: return logic_operation::and_inverted;
			case rsx::logic_op::logic_noop: return logic_operation::no_op;
			case rsx::logic_op::logic_xor: return logic_operation::xor_;
			case rsx::logic_op::logic_or: return logic_operation::or_;
			case rsx::logic_op::logic_nor: return logic_operation::nor;
			case rsx::logic_op::logic_equiv: return logic_operation::equivalent;
			case rsx::logic_op::logic_invert: return logic_operation::invert;
			case rsx::logic_op::logic_or_reverse: return logic_operation::or_reverse;
			case rsx::logic_op::logic_copy_inverted: return logic_operation::copy_inverted;
			case rsx::logic_op::logic_or_inverted: return logic_operation::or_inverted;
			case rsx::logic_op::logic_nand: return logic_operation::nand;
			case rsx::logic_op::logic_set: return logic_operation::set;
			}
			fmt::throw_exception("Invalid RSX logic operation 0x%x", static_cast<u32>(value));
		}

		[[nodiscard]] front_face winding(rsx::front_face value)
		{
			switch (value)
			{
			case rsx::front_face::cw: return front_face::clockwise;
			case rsx::front_face::ccw: return front_face::counter_clockwise;
			}
			fmt::throw_exception("Invalid RSX front-face mode 0x%x", static_cast<u32>(value));
		}

		[[nodiscard]] cull_mode culling(rsx::cull_face value)
		{
			switch (value)
			{
			case rsx::cull_face::front: return cull_mode::front;
			case rsx::cull_face::back: return cull_mode::back;
			case rsx::cull_face::front_and_back: return cull_mode::none;
			}
			fmt::throw_exception("Invalid RSX cull mode 0x%x", static_cast<u32>(value));
		}

		struct vertex_input_state
		{
			primitive_topology topology = primitive_topology::triangle;
			bool restart = false;
			bool discard_all_primitives = false;
			bool line_expansion = false;
		};

		[[nodiscard]] vertex_input_state decode_vertex_input_state()
		{
			const auto& draw = rsx::method_registers.current_draw_clause;
			const primitive_mapping mapping = get_primitive_mapping(draw.primitive);
			const bool line_expansion = draw.primitive == rsx::primitive_type::lines ||
				draw.primitive == rsx::primitive_type::line_loop ||
				draw.primitive == rsx::primitive_type::line_strip;
			vertex_input_state result{line_expansion ? primitive_topology::triangle : mapping.topology,
				false, false, line_expansion};
			if (line_expansion) return result;
			const bool polygon_topology = mapping.topology == primitive_topology::triangle ||
				mapping.topology == primitive_topology::triangle_strip;
			if (polygon_topology && rsx::method_registers.cull_face_enabled() &&
				rsx::method_registers.cull_face_mode() == rsx::cull_face::front_and_back)
				result.discard_all_primitives = true;
			if (rsx::method_registers.restart_index_enabled() && !draw.is_disjoint_primitive &&
				draw.command == rsx::draw_command::indexed && !mapping.requires_index_emulation &&
				!emulate_primitive_restart(mapping.topology))
				result.restart = true;
			return result;
		}

		[[nodiscard]] graphics_pipeline_configuration decode_pipeline_state(
			const vertex_input_state& vertex_input, render_target* depth_target,
			const rsx::backend_configuration& backend, std::span<const u8> draw_buffers,
			u32 sample_count, bool framebuffer_fetch)
		{
			graphics_pipeline_configuration result;
			auto& state = result.state;
			state.set_primitive_type(vertex_input.topology);
			state.enable_primitive_restart(vertex_input.restart);
			if (vertex_input.line_expansion)
				state.render.emulation_flags |= pipeline_emulation_wide_lines;
			state.render.rasterization_enabled = !vertex_input.discard_all_primitives;
			state.render.color_attachment_count = static_cast<u32>(draw_buffers.size());
			state.dynamic.winding = winding(rsx::method_registers.front_face_mode());
			state.enable_depth_clamp(rsx::method_registers.depth_clamp_enabled() ||
				!rsx::method_registers.depth_clip_enabled());
			state.enable_depth_bias(true);
			const bool depth_bias_enabled = vertex_input.line_expansion
				? rsx::method_registers.poly_offset_line_enabled()
				: vertex_input.topology == primitive_topology::point
					? rsx::method_registers.poly_offset_point_enabled()
					: rsx::method_registers.poly_offset_fill_enabled();
			state.dynamic.depth_bias = depth_bias_enabled
				? rsx::method_registers.poly_offset_bias() : 0.f;
			state.dynamic.depth_bias_slope = depth_bias_enabled
				? rsx::method_registers.poly_offset_scale() : 0.f;
			state.dynamic.depth_bias_clamp = 0.f;
			const auto blend_color = rsx::get_constant_blend_colors();
			std::copy(blend_color.begin(), blend_color.end(), state.dynamic.blend_color.begin());
			state.enable_depth_bounds_test(rsx::method_registers.depth_bounds_test_enabled());
			state.dynamic.minimum_depth_bounds = rsx::method_registers.depth_bounds_test_enabled()
				? rsx::method_registers.depth_bounds_min() :
				std::min(0.f, rsx::method_registers.clip_min());
			state.dynamic.maximum_depth_bounds = rsx::method_registers.depth_bounds_test_enabled()
				? rsx::method_registers.depth_bounds_max() :
				std::max(1.f, rsx::method_registers.clip_max());
			if (rsx::method_registers.depth_test_enabled())
			{
				state.set_depth_mask(rsx::method_registers.depth_write_enabled());
				state.enable_depth_test(comparison(rsx::method_registers.depth_func()));
			}
			if (rsx::method_registers.cull_face_enabled() && !vertex_input.discard_all_primitives &&
				!vertex_input.line_expansion)
				state.dynamic.cull = culling(rsx::method_registers.cull_face_mode());
			if (!vertex_input.line_expansion && (vertex_input.topology == primitive_topology::triangle ||
				vertex_input.topology == primitive_topology::triangle_strip))
			{
				bool front_visible = true;
				bool back_visible = true;
				if (rsx::method_registers.cull_face_enabled())
				{
					const auto cull = rsx::method_registers.cull_face_mode();
					front_visible = cull != rsx::cull_face::front &&
						cull != rsx::cull_face::front_and_back;
					back_visible = cull != rsx::cull_face::back &&
						cull != rsx::cull_face::front_and_back;
				}
				const bool emulate_front = front_visible &&
					rsx::method_registers.polygon_mode_front() != rsx::polygon_mode::fill;
				const bool emulate_back = back_visible &&
					rsx::method_registers.polygon_mode_back() != rsx::polygon_mode::fill;
				if (emulate_front || emulate_back)
					state.render.emulation_flags |= pipeline_emulation_polygon_mode;
			}

			const auto host_mask = rsx::get_write_output_mask(rsx::method_registers.surface_color());
			for (u32 output = 0; output < draw_buffers.size(); ++output)
			{
				const u32 index = draw_buffers[output];
				bool blue = rsx::method_registers.color_mask_b(index);
				bool green = rsx::method_registers.color_mask_g(index);
				bool red = rsx::method_registers.color_mask_r(index);
				bool alpha = rsx::method_registers.color_mask_a(index);
				switch (rsx::method_registers.surface_color())
				{
				case rsx::surface_color_format::b8:
					rsx::get_b8_colormask(red, green, blue, alpha);
					break;
				case rsx::surface_color_format::g8b8:
					rsx::get_g8b8_r8g8_colormask(red, green, blue, alpha);
					break;
				default: break;
				}
				state.set_color_mask(output, red && host_mask[0], green && host_mask[1],
					blue && host_mask[2], alpha && host_mask[3]);
			}

			if (rsx::method_registers.logic_op_enabled())
				state.enable_logic_op(logic(rsx::method_registers.logic_operation()));
			else
			{
				const std::array enabled{
					rsx::method_registers.blend_enabled(),
					rsx::method_registers.blend_enabled_surface_1(),
					rsx::method_registers.blend_enabled_surface_2(),
					rsx::method_registers.blend_enabled_surface_3(),
				};
				const auto rgb_equation = rsx::method_registers.blend_equation_rgb();
				const auto alpha_equation = rsx::method_registers.blend_equation_a();
				if (rgb_equation == rsx::blend_equation::add_signed ||
					alpha_equation == rsx::blend_equation::add_signed)
					state.render.emulation_flags |= pipeline_emulation_signed_blend;
				if (rgb_equation == rsx::blend_equation::reverse_add_signed ||
					alpha_equation == rsx::blend_equation::reverse_add_signed)
					state.render.emulation_flags |= pipeline_emulation_reverse_signed_blend;
				const bool programmable_blend =
					rgb_equation == rsx::blend_equation::add_signed ||
					rgb_equation == rsx::blend_equation::reverse_add_signed ||
					alpha_equation == rsx::blend_equation::add_signed ||
					alpha_equation == rsx::blend_equation::reverse_add_signed;
				if (programmable_blend && !framebuffer_fetch)
					fmt::throw_exception("Exact RSX signed blending requires Metal framebuffer fetch support");
				for (u32 output = 0; output < draw_buffers.size(); ++output)
				{
					if (!enabled[draw_buffers[output]]) continue;
					if (!programmable_blend)
						state.enable_blend(output,
							blend(rsx::method_registers.blend_func_sfactor_rgb()),
							blend(rsx::method_registers.blend_func_sfactor_a()),
							blend(rsx::method_registers.blend_func_dfactor_rgb()),
							blend(rsx::method_registers.blend_func_dfactor_a()),
							blend_equation(rgb_equation), blend_equation(alpha_equation));
				}
			}

			if (rsx::method_registers.stencil_test_enabled())
			{
				state.enable_stencil_test(
					stencil(rsx::method_registers.stencil_op_fail()),
					stencil(rsx::method_registers.stencil_op_zfail()),
					stencil(rsx::method_registers.stencil_op_zpass()),
					comparison(rsx::method_registers.stencil_func()),
					rsx::method_registers.stencil_func_mask(),
					rsx::method_registers.stencil_func_ref());
				state.set_stencil_mask(rsx::method_registers.stencil_mask());
				if (rsx::method_registers.two_sided_stencil_test_enabled())
				{
					state.enable_stencil_test_separate(true,
						stencil(rsx::method_registers.back_stencil_op_fail()),
						stencil(rsx::method_registers.back_stencil_op_zfail()),
						stencil(rsx::method_registers.back_stencil_op_zpass()),
						comparison(rsx::method_registers.back_stencil_func()),
						rsx::method_registers.back_stencil_func_mask(),
						rsx::method_registers.back_stencil_func_ref());
					state.set_stencil_mask_separate(true,
						rsx::method_registers.back_stencil_mask());
				}
				if (depth_target && depth_target->samples() > 1 &&
					!(depth_target->stencil_init_flags & 0xff00))
					depth_target->stencil_init_flags |= 0x100;
			}

			if (backend.supports_hw_a2c || sample_count > 1)
			{
				state.set_multisample_state(sample_count, rsx::method_registers.msaa_sample_mask(),
					rsx::method_registers.msaa_enabled(),
					rsx::method_registers.msaa_alpha_to_coverage_enabled(),
					rsx::method_registers.msaa_alpha_to_one_enabled() && backend.supports_hw_a2one);
				state.set_multisample_shading_rate(1.f);
			}
			return result;
		}

		void create_heap(data_heap& heap, memory_allocator& allocator, data_heap_role role,
			u64 size, u32 usage, allocation_pool pool, std::string_view label,
			bool low_latency = false)
		{
			data_heap_create_info information;
			information.initial_size = size;
			information.maximum_size = 1024ull * 1024 * 1024;
			information.growth_quantum = std::max<u64>(size, 16ull * 1024 * 1024);
			information.guard_size = std::min<u64>(size / 16, 4ull * 1024 * 1024);
			information.usage = usage;
			information.flags = data_heap_persistent_mapping |
				(low_latency ? data_heap_low_latency : data_heap_default);
			information.pool = pool;
			information.label = std::string(label);
			heap.create(allocator, information);
			static_cast<void>(get_data_heap_manager().register_heap(heap,
				{.role = role, .priority = low_latency ? 10u : 0u,
					.label = std::string(label), .prefer_low_latency = low_latency}));
		}

		void encode_buffer_update(command_buffer& command, buffer& destination, u64 offset,
			std::span<const std::byte> bytes)
		{
			if (bytes.empty() || !destination.in_range(offset, bytes.size()))
				fmt::throw_exception("Invalid Metal buffer update range");
			if (command.active_encoder() != encoder_kind::none) command.end_encoding();
			id<MTLDevice> device = command.allocator().owner().native_handle();
			id<MTLBuffer> source = [device newBufferWithLength:std::max<NSUInteger>(bytes.size(), 256)
				options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
			if (!source) fmt::throw_exception("Metal buffer-update staging allocation failed");
			std::memcpy(source.contents, bytes.data(), bytes.size());
			id<MTL4ComputeCommandEncoder> encoder = (__bridge id<MTL4ComputeCommandEncoder>)
				command.begin_compute_encoding();
			[encoder copyFromBuffer:source sourceOffset:0 toBuffer:destination.native_handle()
				destinationOffset:offset size:bytes.size()];
			command.retain_native_object((__bridge void*)source, true);
			command.retain_native_object((__bridge void*)destination.native_handle(), true);
			command.end_encoding();
		}

		void encode_viewport_scissor(native_encoder_handle native_encoder,
			const viewport& viewport_state, const scissor_rectangle& scissor_state)
		{
			id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)native_encoder;
			if (!encoder) fmt::throw_exception("Metal viewport update requires a render encoder");
			[encoder setViewport:MTLViewport{viewport_state.x, viewport_state.y,
				viewport_state.width, viewport_state.height, viewport_state.minimum_depth,
				viewport_state.maximum_depth}];
			[encoder setScissorRect:MTLScissorRect{scissor_state.x, scissor_state.y,
				scissor_state.width, scissor_state.height}];
		}

		struct logic_attachment_encoding
		{
			u32 type = 0;
			std::array<u32, 4> scale{255, 255, 255, 255};
		};

		struct alignas(16) fragment_environment
		{
			std::array<u8, 32> base{};
			u32 logic_operation = 0;
			u32 rop_emulation = 0;
			std::array<u32, 4> logic_types{};
			std::array<u32, 2> padding{};
			std::array<std::array<u32, 4>, 4> logic_scales{};
			std::array<f32, 4> blend_constants{};
			u32 blend_equations = 0;
			u32 blend_factors_alpha = 0;
			u32 blend_factors_rgb = 0;
			u32 programmable_blend_mask = 0;
			u32 sample_mask = 0xffffffffu;
			std::array<u32, 3> sample_padding{};
			u32 polygon_modes = 0;
			f32 polygon_line_width = 1.f;
			f32 polygon_point_size = 1.f;
			u32 polygon_padding = 0;
		};

		static_assert(sizeof(fragment_environment) == 192);

		logic_attachment_encoding logic_attachment(rsx::surface_color_format format)
		{
			switch (format)
			{
			case rsx::surface_color_format::r5g6b5:
				return {0, {31, 63, 31, 1}};
			case rsx::surface_color_format::x1r5g5b5_o1r5g5b5:
			case rsx::surface_color_format::x1r5g5b5_z1r5g5b5:
				return {0, {31, 31, 31, 1}};
			case rsx::surface_color_format::w16z16y16x16:
				return {1, {0xffff, 0xffff, 0xffff, 0xffff}};
			case rsx::surface_color_format::w32z32y32x32:
			case rsx::surface_color_format::x32:
				return {2, {umax, umax, umax, umax}};
			case rsx::surface_color_format::b8:
				return {0, {255, 1, 1, 1}};
			case rsx::surface_color_format::g8b8:
				return {0, {255, 255, 1, 1}};
			default:
				return {};
			}
		}
	}
}

u64 MTLGSRender::get_cycles()
{
	return thread_ctrl::get_cycles(static_cast<named_thread<MTLGSRender>&>(*this));
}

MTLGSRender::MTLGSRender(utils::serial* archive) noexcept
	: GSRender(archive)
{
	try
	{
		g_fxo->need<rsx::dma_manager>();
		m_shared_state = &mtl::get_shared_state();
		m_shared_state->initialize(g_cfg.video.mtl.adapter.get());
		m_device = &m_shared_state->device();
		m_allocator = &m_shared_state->allocator();
		mtl::set_current_renderer(*m_device);
		mtl::get_resource_manager().initialize(*m_shared_state);
		m_resources.initialize(*m_shared_state);
		mtl::initialize_dma_pool(*m_allocator, {
			.read = [](u32 address, std::span<u8> destination)
			{
				std::fill(destination.begin(), destination.end(), 0);
				for (u32 offset = 0; offset < destination.size();)
				{
					const u32 length = std::min<u32>(4096 - ((address + offset) & 4095),
						static_cast<u32>(destination.size() - offset));
					if (rsx::get_location(address + offset) == CELL_GCM_LOCATION_LOCAL ||
						vm::check_addr(address + offset, 0, length))
						std::memcpy(destination.data() + offset,
							vm::get_super_ptr<u8>(address + offset), length);
					offset += length;
				}
				return true;
			},
			.write = [](u32 address, std::span<const u8> source)
			{
				for (u32 offset = 0; offset < source.size();)
				{
					const u32 length = std::min<u32>(4096 - ((address + offset) & 4095),
						static_cast<u32>(source.size() - offset));
					if (rsx::get_location(address + offset) == CELL_GCM_LOCATION_LOCAL ||
						vm::check_addr(address + offset, 0, length))
						std::memcpy(vm::get_super_ptr<u8>(address + offset),
							source.data() + offset, length);
					offset += length;
				}
				return true;
			},
			.direct_pointer = [](u32 address, u64 length) -> void*
			{
				if (length > umax) return nullptr;
				if (rsx::get_location(address) != CELL_GCM_LOCATION_LOCAL &&
					!vm::check_addr(address, 0, static_cast<u32>(length)))
					return nullptr;
				return vm::get_super_ptr<u8>(address);
			},
		}, {.allow_host_no_copy = m_device->info().memory.unified});
		static_cast<void>(m_shared_state->begin_frame());

		m_swapchain = std::make_unique<mtl::native_swapchain>();
		const mtl::swapchain_configuration swap_configuration{
			.pixel_format = MTLPixelFormatBGRA8Unorm,
			.width = static_cast<u32>(std::max(m_frame->client_width(), 1)),
			.height = static_cast<u32>(std::max(m_frame->client_height(), 1)),
			.maximum_drawables = 3,
			.mode = m_vsync_mode == vsync_mode::off ? mtl::presentation_mode::immediate :
				mtl::presentation_mode::synchronized,
			.acquire_timeout = mtl::frame_present_timeout,
			.label = "RPCS3 Metal presentation",
		};
		m_swapchain->create(*m_device, m_device->graphics_queue(), m_frame->handle(), swap_configuration);
		const auto drawable_dimensions = m_swapchain->size();
		m_swapchain_dimensions = {drawable_dimensions.width, drawable_dimensions.height};

		m_primary_commands.create(*m_device, "RPCS3 primary command buffer");
		m_secondary_commands.create(*m_device, "RPCS3 secondary command buffer");
		m_current_command_buffer = m_primary_commands.get();
		m_current_command_buffer->begin();

		m_pipeline_compiler.initialize(*m_device);
		mtl::initialize_compute_tasks(*m_device, m_pipeline_compiler);
		m_command_stream.initialize({.mode = mtl::command_stream_mode::worker_submission,
			.maximum_queued_submissions = 64, .queue = mtl::queue_kind::graphics,
			.label = "RPCS3 Metal graphics stream"});
		if (m_device->info().features.shared_events &&
			m_device->graphics_queue() != m_device->transfer_queue())
		{
			m_async_scheduler.initialize(*m_device, {.mode = mtl::async_scheduler_mode::gpu_timeline,
				.maximum_command_slots = 32, .queue = mtl::queue_kind::transfer,
				.label = "RPCS3 Metal transfer scheduler"});
			backend_config.supports_asynchronous_compute = true;
		}

		m_occlusion_query_manager = std::make_unique<mtl::query_pool_manager>(*m_device,
			*m_allocator, mtl::query_pool_manager_create_info{
				.capacity = rsx::reports::occlusion_query_count,
				.kind = mtl::query_kind::occlusion_counting,
				.allocation = mtl::allocation_pool::system,
				.label = "RPCS3 occlusion queries",
				.cpu_readback = true,
				.allow_partial_results = true,
			});
		m_occlusion_map.resize(rsx::reports::occlusion_query_count);
		for (u32 index = 0; index < rsx::reports::occlusion_query_count; ++index)
			m_occlusion_query_data[index].driver_handle = index;

		using enum mtl::buffer_usage;
		mtl::create_heap(m_attribute_heap, *m_allocator, mtl::data_heap_role::vertex_data,
			mtl::attribute_ring_buffer_size, buffer_usage_vertex | buffer_usage_texture_view |
				buffer_usage_copy_source, mtl::allocation_pool::system, "RPCS3 vertex attributes");
		mtl::create_heap(m_fragment_constants_heap, *m_allocator, mtl::data_heap_role::constants,
			mtl::fragment_constants_buffer_size, buffer_usage_constant | buffer_usage_storage,
			mtl::allocation_pool::system, "RPCS3 fragment constants", true);
		mtl::create_heap(m_transform_constants_heap, *m_allocator, mtl::data_heap_role::constants,
			mtl::transform_constants_buffer_size, buffer_usage_constant | buffer_usage_storage,
			mtl::allocation_pool::system, "RPCS3 transform constants");
		mtl::create_heap(m_fragment_environment_heap, *m_allocator, mtl::data_heap_role::constants,
			mtl::uniform_ring_buffer_size, buffer_usage_constant, mtl::allocation_pool::system,
			"RPCS3 fragment environment", true);
		mtl::create_heap(m_vertex_environment_heap, *m_allocator, mtl::data_heap_role::constants,
			mtl::uniform_ring_buffer_size, buffer_usage_constant, mtl::allocation_pool::system,
			"RPCS3 vertex environment");
		mtl::create_heap(m_fragment_texture_parameters_heap, *m_allocator,
			mtl::data_heap_role::constants, mtl::uniform_ring_buffer_size,
			buffer_usage_constant, mtl::allocation_pool::system,
			"RPCS3 fragment texture parameters", true);
		mtl::create_heap(m_vertex_layout_heap, *m_allocator, mtl::data_heap_role::vertex_data,
			mtl::uniform_ring_buffer_size, buffer_usage_storage, mtl::allocation_pool::system,
			"RPCS3 vertex layout", true);
		mtl::create_heap(m_index_heap, *m_allocator, mtl::data_heap_role::index_data,
			mtl::index_ring_buffer_size, buffer_usage_index | buffer_usage_copy_source,
			mtl::allocation_pool::system, "RPCS3 index data");
		mtl::create_heap(m_texture_upload_heap, *m_allocator, mtl::data_heap_role::texture_data,
			mtl::texture_upload_ring_buffer_size, buffer_usage_copy_source,
			mtl::allocation_pool::texture_cache, "RPCS3 texture uploads");
		mtl::create_heap(m_raster_environment_heap, *m_allocator, mtl::data_heap_role::constants,
			mtl::uniform_ring_buffer_size, buffer_usage_storage, mtl::allocation_pool::system,
			"RPCS3 raster environment", true);
		mtl::create_heap(m_instancing_heap, *m_allocator, mtl::data_heap_role::vertex_data,
			mtl::transform_constants_buffer_size, buffer_usage_storage,
			mtl::allocation_pool::system, "RPCS3 instancing data");

		const shader_mode shader_mode_setting = g_cfg.video.shadermode.get();
		if (shader_mode_setting == shader_mode::async_with_interpreter ||
			shader_mode_setting == shader_mode::interpreter_only)
		{
			mtl::create_heap(m_vertex_instructions_heap, *m_allocator,
				mtl::data_heap_role::constants, 64ull * 1024 * 1024,
				buffer_usage_storage, mtl::allocation_pool::system,
				"RPCS3 vertex instructions", true);
			mtl::create_heap(m_fragment_instructions_heap, *m_allocator,
				mtl::data_heap_role::constants, 64ull * 1024 * 1024,
				buffer_usage_storage, mtl::allocation_pool::system,
				"RPCS3 fragment instructions", true);
		}

		for (mtl::data_heap* heap : mtl::get_data_heap_manager().heaps())
			if (heap->has_shadow()) m_flushable_heaps.push_back(heap);

		m_maximum_async_frames = m_swapchain->configuration().maximum_drawables;
		m_frame_context_storage.resize(m_maximum_async_frames);
		for (u32 index = 0; index < m_maximum_async_frames; ++index)
			m_frame_context_storage[index].initialize(index + 1);
		m_current_frame = &m_frame_context_storage.front();

		m_null_buffer = std::make_unique<mtl::buffer>(*m_allocator, mtl::buffer_create_info{
			.size = 256, .usage = buffer_usage_constant | buffer_usage_texture_view,
			.storage = mtl::storage_mode::private_, .access = mtl::cpu_access::none,
			.pool = mtl::allocation_pool::system, .label = "RPCS3 null buffer"});
		m_null_buffer_view = std::make_unique<mtl::buffer_view>(*m_null_buffer,
			MTLPixelFormatR8Uint, 0, 256, 1);
		m_conditional_render_buffer = std::make_unique<mtl::buffer>(*m_allocator,
			mtl::buffer_create_info{.size = 256,
				.usage = buffer_usage_copy_destination | buffer_usage_storage,
				.storage = mtl::storage_mode::shared, .access = mtl::cpu_access::read_write,
				.pool = mtl::allocation_pool::system,
				.label = "RPCS3 conditional-render predicate"});
		const std::array<u32, 2> initial_predicates{0u, umax};
		std::memcpy(m_conditional_render_buffer->map(0, sizeof(initial_predicates)),
			initial_predicates.data(), sizeof(initial_predicates));
		m_conditional_render_buffer->did_modify(0, sizeof(initial_predicates));
		m_conditional_render_buffer->unmap();

		m_program_buffer = std::make_unique<mtl::MTLProgramBuffer>();
		m_program_buffer->initialize(m_pipeline_compiler,
			[this](const mtl::pipeline_properties& properties,
				const RSXVertexProgram& vertex, const RSXFragmentProgram& fragment)
			{
				if (m_shader_cache) m_shader_cache->store(properties, vertex, fragment);
			});
		m_shader_cache = std::make_unique<mtl::shader_cache>(*m_program_buffer, "metal4", "v3");
		m_vertex_cache = g_cfg.video.disable_vertex_cache
			? std::unique_ptr<mtl::vertex_cache>(std::make_unique<mtl::null_vertex_cache>())
			: std::unique_ptr<mtl::vertex_cache>(std::make_unique<mtl::weak_vertex_cache>());

		m_texture_cache.initialize(*m_device, *m_allocator, m_render_targets);
		m_overlay_passes.initialize(*m_device, *m_allocator, m_pipeline_compiler);
		m_overlay_passes.ui().initialize_resources(*m_current_command_buffer, m_texture_upload_heap);
		if (shader_mode_setting == shader_mode::async_with_interpreter ||
			shader_mode_setting == shader_mode::interpreter_only)
			m_shader_interpreter.initialize(*m_device, m_pipeline_compiler);

		backend_config.supports_multidraw = true;
		backend_config.supports_hw_instanced_rendering = true;
		backend_config.supports_normalized_barycentrics =
			m_device->info().features.barycentric_coordinates;
		backend_config.supports_hw_msaa = true;
		backend_config.supports_hw_a2c = true;
		backend_config.supports_hw_a2c_1spp = true;
		backend_config.supports_hw_a2one = true;
		backend_config.supports_hw_conditional_render = true;
		backend_config.supports_passthrough_dma = m_device->info().memory.unified;
		backend_config.supports_host_gpu_labels = g_cfg.video.host_label_synchronization &&
			backend_config.supports_passthrough_dma;
		backend_config.supports_hw_renormalization = false;

		if (backend_config.supports_host_gpu_labels)
		{
			m_host_object_data = std::make_unique<mtl::buffer>(*m_allocator, mtl::buffer_create_info{
				.size = 0x10000, .usage = buffer_usage_copy_destination | buffer_usage_storage,
				.storage = mtl::storage_mode::shared, .access = mtl::cpu_access::read_write,
				.pool = mtl::allocation_pool::system, .label = "RPCS3 host GPU context"});
			m_host_dma_ctrl = std::make_unique<rsx::RSXDMAWriter>(m_host_object_data->map(0, 0x10000));
		}

		rsx_log.notice("Metal 4 renderer initialized on '%s'", m_device->gpu().name());
	}
	catch (const std::exception& error)
	{
		rsx_log.fatal("Metal renderer initialization failed: %s", error.what());
		m_device = nullptr;
		m_allocator = nullptr;
	}
}

MTLGSRender::~MTLGSRender()
{
	if (!m_shared_state || !*m_shared_state) return;
	try
	{
		if (m_current_command_buffer && m_current_command_buffer->is_recording())
		{
			if (m_command_stream)
			{
				const mtl::submission final_submission = close_and_submit_command_buffer({}, {}, true);
				if (final_submission) final_submission.wait();
			}
			else
			{
				close_render_pass();
				if (m_current_command_buffer->has_flag(mtl::command_has_open_query) &&
					m_occlusion_query_manager && m_active_query_info)
				{
					auto& data = m_occlusion_map[m_active_query_info->driver_handle];
					if (!data.indices.empty())
						m_occlusion_query_manager->end_query(m_render_pass, data.indices.back());
					m_current_command_buffer->clear_flag(mtl::command_has_open_query);
				}
				m_current_command_buffer->end();
				m_current_command_buffer->tag();
				static_cast<void>(m_current_command_buffer->submit({.wait_for_completion = true}));
			}
		}
		if (m_async_scheduler) m_async_scheduler.wait_idle();
		if (m_command_stream) m_command_stream.flush(true);
		m_primary_commands.wait_all();
		m_secondary_commands.wait_all();
		if (m_pipeline_compiler) m_pipeline_compiler.wait_idle();
		mtl::reset_compute_tasks();
		const u64 completed = m_shared_state->completed_submission();
		m_overlay_passes.destroy();
		m_shader_interpreter.destroy();
		m_shader_cache.reset();
		m_program_buffer.reset();
		m_upscaler.reset();
		m_present_temporary_buffers.clear();
		m_present_temporary_images.clear();
		m_overlay_recording_image.reset();
		m_texture_cache.destroy();
		m_render_targets.destroy();
		mtl::shutdown_dma_pool();
		m_occlusion_query_manager.reset();
		m_conditional_render_buffer.reset();
		m_host_dma_ctrl.reset();
		if (m_host_object_data) m_host_object_data->unmap();
		m_host_object_data.reset();
		m_null_buffer_view.reset();
		m_null_buffer.reset();
		mtl::get_data_heap_manager().clear_registrations(true, completed);
		m_resources.destroy();
		if (m_async_scheduler) m_async_scheduler.destroy(true);
		if (m_command_stream) m_command_stream.shutdown(true);
		m_primary_commands.destroy();
		m_secondary_commands.destroy();
		if (m_swapchain) m_swapchain->destroy();
		mtl::get_resource_manager().destroy(completed, true);
		if (m_device) mtl::clear_current_renderer(*m_device);
		m_shared_state->shutdown();
	}
	catch (const std::exception& error)
	{
		rsx_log.fatal("Metal renderer shutdown failed: %s", error.what());
	}
	m_device = nullptr;
	m_allocator = nullptr;
	m_shared_state = nullptr;
}

std::pair<const mtl::vertex_program_bindings*, const mtl::fragment_program_bindings*>
MTLGSRender::get_binding_tables() const
{
	if (!m_program) fmt::throw_exception("Metal binding tables require a current program");
	if (!m_program_interpreted)
	{
		if (!m_vertex_program || !m_fragment_program)
			fmt::throw_exception("Metal recompiled program has no shader metadata");
		return {&m_vertex_program->bindings(), &m_fragment_program->bindings()};
	}
	return {nullptr, nullptr};
}

bool MTLGSRender::is_current_program_interpreted() const
{
	return m_program && m_program_interpreted;
}

bool MTLGSRender::load_program()
{
	const shader_mode mode = g_cfg.video.shadermode.get();
	const mtl::vertex_input_state vertex_input = mtl::decode_vertex_input_state();
	if (m_graphics_state & rsx::pipeline_state::invalidate_pipeline_bits)
	{
		get_current_fragment_program(fs_sampler_state);
		if (!current_fragment_program.valid)
			fmt::throw_exception("Metal draw has an invalid fragment program");
		get_current_vertex_program(vs_sampler_state);
		m_graphics_state.clear(rsx::pipeline_state::invalidate_pipeline_bits);
	}

	mtl::render_target* depth_target = std::get<1>(m_render_targets.m_bound_depth_stencil);
	const u32 sample_count = depth_target ? depth_target->samples() :
		(!m_framebuffer_images.empty() ? m_framebuffer_images.front()->samples() : 1);
	mtl::pipeline_properties properties = mtl::decode_pipeline_state(vertex_input, depth_target,
		backend_config, m_draw_buffers, sample_count, m_device->info().features.framebuffer_fetch);
	for (u32 output = 0; output < m_draw_buffers.size(); ++output)
	{
		mtl::render_target* target = std::get<1>(
			m_render_targets.m_bound_render_targets[m_draw_buffers[output]]);
		if (!target) fmt::throw_exception("Metal draw framebuffer has an empty color attachment");
		properties.state.render.color_attachments[output].pixel_format = target->format();
	}
	if (depth_target)
	{
		if (depth_target->aspects() & mtl::texture_aspect_depth)
			properties.state.depth_stencil.depth_pixel_format = depth_target->format();
		if (depth_target->aspects() & mtl::texture_aspect_stencil)
			properties.state.depth_stencil.stencil_pixel_format = depth_target->format();
	}
	properties.label = "RPCS3 RSX graphics pipeline";

	if (m_program_template && !(m_graphics_state & rsx::pipeline_state::pipeline_config_dirty) &&
		m_pipeline_properties == properties)
	{
		m_program_instance = m_program_template->create_binding_instance();
		m_program = m_program_instance.get();
		return true;
	}
	m_pipeline_properties = properties;
	m_graphics_state.clear(rsx::pipeline_state::pipeline_config_dirty);
	m_program_instance.reset();
	m_program = nullptr;
	m_program_template = nullptr;
	m_program_interpreted = false;
	m_vertex_program = nullptr;
	m_fragment_program = nullptr;

	if (mode != shader_mode::interpreter_only)
	{
		mtl::enter_uninterruptible();
		auto [pipeline, vertex, fragment] = m_program_buffer->get_graphics_pipeline(
			&m_program_cache_hint, current_vertex_program, current_fragment_program,
			m_pipeline_properties, mode != shader_mode::recompiler, true);
		mtl::leave_uninterruptible();
		m_program_template = pipeline;
		m_vertex_program = vertex && vertex->program ? vertex->program.get() : nullptr;
		m_fragment_program = fragment && fragment->program ? fragment->program.get() : nullptr;
	}

	if (!m_program && (mode == shader_mode::async_with_interpreter ||
		mode == shader_mode::interpreter_only))
	{
		mtl::interpreter_program_state state;
		state.fragment_metadata.referenced_textures_mask = current_fp_metadata.referenced_textures_mask;
		state.fragment_metadata.has_pack_instructions = current_fp_metadata.has_pack_instructions;
		state.fragment_metadata.has_branch_instructions = current_fp_metadata.has_branch_instructions;
		state.vertex_metadata.referenced_textures_mask = current_vp_metadata.referenced_textures_mask;
		state.vertex_control = current_vertex_program.ctrl;
		state.fragment_control = current_fragment_program.ctrl;
		state.polygon_stipple = rsx::method_registers.polygon_stipple_enabled();
		auto pipeline = m_shader_interpreter.get(m_pipeline_properties, state, false);
		m_program_template = pipeline.get();
		m_program_interpreted = true;
		m_interpreter_state = rsx::invalidate_pipeline_bits;
	}

	if (!m_program_template) return false;
	m_program_instance = m_program_template->create_binding_instance();
	m_program = m_program_instance.get();
	if (!m_program_interpreted)
		std::tie(m_vertex_bindings, m_fragment_bindings) = get_binding_tables();
	else
	{
		m_vertex_bindings = nullptr;
		m_fragment_bindings = nullptr;
	}
	return true;
}

void MTLGSRender::upload_transform_constants(const rsx::io_buffer& destination)
{
	const bool interpreter = m_program_interpreted;
	const usz size = interpreter || (m_vertex_program && m_vertex_program->metadata().has_indexed_constants)
		? 8192 : (m_vertex_program ? m_vertex_program->constant_ids.size() * 16 : 0);
	if (!size) return;
	destination.reserve(size);
	const std::span<const u16> identifiers = size == 8192 ? std::span<const u16>{} :
		std::span<const u16>(m_vertex_program->constant_ids);
	m_draw_processor.fill_vertex_program_constants_data(destination.data(), identifiers);
}

void MTLGSRender::load_program_environment()
{
	if (!m_program) fmt::throw_exception("Metal program environment requires a pipeline");
	const auto& context = REGS(m_ctx);
	const bool interpreter = m_program_interpreted;
	const bool update_vertex = true;
	const bool update_fragment = true;
	const bool update_transform = m_graphics_state & rsx::pipeline_state::transform_constants_dirty;
	const bool update_constants = m_graphics_state & rsx::pipeline_state::fragment_constants_dirty;
	const bool update_textures = m_graphics_state & rsx::pipeline_state::fragment_texture_state_dirty;

	auto binding = [](const mtl::data_heap_slice& slice) -> mtl::argument_buffer_binding
	{
		return {.resource = slice.buffer, .gpu_address = slice.buffer_gpu_address(),
			.offset = slice.offset, .length = slice.size};
	};
	const mtl::argument_buffer_binding null_binding{
		.resource = m_null_buffer->native_handle(),
		.gpu_address = m_null_buffer->gpu_address(),
		.offset = 0,
		.length = m_null_buffer->size(),
	};

	if (update_vertex)
	{
		const auto slice = m_vertex_environment_heap.allocate(96, 256);
		auto* bytes = static_cast<u8*>(m_vertex_environment_heap.map(slice));
		// Metal's viewport maps normalized Y to its top-left framebuffer origin.
		// Invert the RSX viewport transform here to avoid applying the Y flip twice.
		m_draw_processor.fill_scale_offset_data(bytes, true);
		m_draw_processor.fill_user_clip_data(bytes + 64);
		*reinterpret_cast<u32*>(bytes + 68) = context->transform_branch_bits();
		*reinterpret_cast<f32*>(bytes + 72) = context->point_size() * resolution_scaling_config.scale_factor();
		*reinterpret_cast<f32*>(bytes + 76) = context->clip_min();
		*reinterpret_cast<f32*>(bytes + 80) = context->clip_max();
		*reinterpret_cast<f32*>(bytes + 84) = context->line_width() *
			resolution_scaling_config.scale_factor() * 0.5f;
		*reinterpret_cast<f32*>(bytes + 88) = std::abs(m_viewport.width);
		*reinterpret_cast<f32*>(bytes + 92) = std::abs(m_viewport.height);
		m_vertex_environment_heap.mark_modified(slice);
		m_vertex_environment_heap.unmap();
		m_vertex_environment_binding = binding(slice);
		m_vertex_environment_offset = slice.offset;
	}

	if (context->current_draw_clause.is_trivial_instanced_draw && !interpreter)
	{
		mtl::data_heap_slice indirection;
		mtl::data_heap_slice constants;
		rsx::io_buffer indirection_buffer([&](usz size) -> std::pair<void*, usz>
		{
			indirection = m_instancing_heap.allocate(size, 256);
			return {m_instancing_heap.map(indirection), size};
		});
		rsx::io_buffer constants_buffer([&](usz size) -> std::pair<void*, usz>
		{
			constants = m_instancing_heap.allocate(size, 256);
			return {m_instancing_heap.map(constants), size};
		});
		m_draw_processor.fill_constants_instancing_buffer(indirection_buffer, constants_buffer,
			interpreter ? nullptr : m_vertex_program);
		if (indirection)
		{
			m_instancing_heap.mark_modified(indirection);
			m_instancing_indirection_binding = binding(indirection);
		}
		if (constants)
		{
			m_instancing_heap.mark_modified(constants);
			m_instancing_constants_binding = binding(constants);
		}
		m_instancing_heap.unmap();
	}
	else if (update_transform)
	{
		mtl::data_heap_slice slice;
		rsx::io_buffer transform([&](usz size) -> std::pair<void*, usz>
		{
			slice = m_transform_constants_heap.allocate(size, 256);
			return {m_transform_constants_heap.map(slice), size};
		});
		upload_transform_constants(transform);
		if (slice)
		{
			m_transform_constants_heap.mark_modified(slice);
			m_transform_constants_heap.unmap();
			m_vertex_constants_binding = binding(slice);
			m_transform_constants_offset = slice.offset;
		}
	}

	if (update_constants && !interpreter && m_fragment_program)
	{
		const usz size = m_fragment_program->constant_offsets().size() * 16;
		if (size)
		{
			const auto slice = m_fragment_constants_heap.allocate(size, 256);
			auto* output = static_cast<f32*>(m_fragment_constants_heap.map(slice));
			m_program_buffer->fill_fragment_constants_buffer({output, size / sizeof(f32)},
				mtl::cached_fragment_program{m_fragment_program->id(),
					std::shared_ptr<mtl::MTLFragmentProgram>(),
					std::vector<u32>(m_fragment_program->constant_offsets().begin(),
						m_fragment_program->constant_offsets().end())},
				current_fragment_program, true);
			m_fragment_constants_heap.mark_modified(slice);
			m_fragment_constants_heap.unmap();
			m_fragment_constants_binding = binding(slice);
			m_fragment_constants_offset = slice.offset;
		}
	}

	if (update_fragment)
	{
		const auto slice = m_fragment_environment_heap.allocate(
			sizeof(mtl::fragment_environment), 256);
		auto* environment = static_cast<mtl::fragment_environment*>(
			m_fragment_environment_heap.map(slice));
		*environment = {};
		m_draw_processor.fill_fragment_state_buffer(environment->base.data(),
			current_fragment_program);
		environment->logic_operation = static_cast<u32>(
			mtl::logic(rsx::method_registers.logic_operation()));
		environment->rop_emulation = rsx::method_registers.logic_op_enabled() &&
			m_device->info().features.framebuffer_fetch ? 1u : 0u;
		if (rsx::method_registers.logic_op_enabled() && !m_device->info().features.framebuffer_fetch)
			fmt::throw_exception("Exact RSX logic operations require Metal framebuffer fetch support");
		const mtl::logic_attachment_encoding encoding =
			mtl::logic_attachment(rsx::method_registers.surface_color());
		for (u32 output = 0; output < 4; ++output)
		{
			environment->logic_types[output] = encoding.type;
			environment->logic_scales[output] = encoding.scale;
		}
		environment->blend_constants = rsx::get_constant_blend_colors();
		environment->blend_equations =
			static_cast<u32>(static_cast<u16>(rsx::method_registers.blend_equation_rgb())) |
			(static_cast<u32>(static_cast<u16>(rsx::method_registers.blend_equation_a())) << 16);
		environment->blend_factors_alpha =
			static_cast<u32>(static_cast<u16>(rsx::method_registers.blend_func_sfactor_a())) |
			(static_cast<u32>(static_cast<u16>(rsx::method_registers.blend_func_dfactor_a())) << 16);
		environment->blend_factors_rgb =
			static_cast<u32>(static_cast<u16>(rsx::method_registers.blend_func_sfactor_rgb())) |
			(static_cast<u32>(static_cast<u16>(rsx::method_registers.blend_func_dfactor_rgb())) << 16);
		environment->sample_mask = rsx::method_registers.msaa_enabled()
			? rsx::method_registers.msaa_sample_mask() : 0xffffffffu;
		const auto encode_polygon_mode = [](rsx::polygon_mode mode) -> u32
		{
			switch (mode)
			{
			case rsx::polygon_mode::fill: return 0u;
			case rsx::polygon_mode::line: return 1u;
			case rsx::polygon_mode::point: return 2u;
			}
			fmt::throw_exception("Invalid RSX polygon mode");
		};
		const bool polygon_emulation = m_pipeline_properties.state.render.emulation_flags &
			mtl::pipeline_emulation_polygon_mode;
		environment->polygon_modes = polygon_emulation ? (1u << 31) |
			encode_polygon_mode(rsx::method_registers.polygon_mode_front()) |
			(encode_polygon_mode(rsx::method_registers.polygon_mode_back()) << 2) : 0u;
		if (m_pipeline_properties.state.render.emulation_flags & mtl::pipeline_emulation_wide_lines)
			environment->polygon_modes |= 1u << 30;
		environment->polygon_line_width = context->line_width() *
			resolution_scaling_config.scale_factor();
		environment->polygon_point_size = context->point_size() *
			resolution_scaling_config.scale_factor();
		if (!rsx::method_registers.logic_op_enabled() && m_device->info().features.framebuffer_fetch)
		{
			const auto rgb_equation = rsx::method_registers.blend_equation_rgb();
			const auto alpha_equation = rsx::method_registers.blend_equation_a();
			const bool programmable =
				rgb_equation == rsx::blend_equation::add_signed ||
				rgb_equation == rsx::blend_equation::reverse_add_signed ||
				alpha_equation == rsx::blend_equation::add_signed ||
				alpha_equation == rsx::blend_equation::reverse_add_signed;
			if (programmable)
			{
				const std::array enabled{
					rsx::method_registers.blend_enabled(),
					rsx::method_registers.blend_enabled_surface_1(),
					rsx::method_registers.blend_enabled_surface_2(),
					rsx::method_registers.blend_enabled_surface_3(),
				};
				for (u32 output = 0; output < m_draw_buffers.size(); ++output)
					if (enabled[m_draw_buffers[output]]) environment->programmable_blend_mask |= 1u << output;
				environment->rop_emulation |= environment->programmable_blend_mask ? 2u : 0u;
			}
		}
		m_fragment_environment_heap.mark_modified(slice);
		m_fragment_environment_heap.unmap();
		m_fragment_environment_binding = binding(slice);
		m_fragment_environment_offset = slice.offset;
	}

	if (update_textures)
	{
		const auto slice = m_fragment_texture_parameters_heap.allocate(768, 256);
		current_fragment_program.texture_params.write_to(
			m_fragment_texture_parameters_heap.map(slice), current_fp_metadata.referenced_textures_mask);
		m_fragment_texture_parameters_heap.mark_modified(slice);
		m_fragment_texture_parameters_heap.unmap();
		m_fragment_texture_parameters_binding = binding(slice);
		m_texture_parameters_offset = slice.offset;
	}

	if (context->polygon_stipple_enabled() &&
		(m_graphics_state & rsx::pipeline_state::polygon_stipple_pattern_dirty))
	{
		const auto slice = m_raster_environment_heap.allocate(128, 128);
		std::memcpy(m_raster_environment_heap.map(slice), context->polygon_stipple_pattern(), 128);
		m_raster_environment_heap.mark_modified(slice);
		m_raster_environment_heap.unmap();
		m_raster_environment_binding = binding(slice);
		m_stipple_array_offset = slice.offset;
		m_graphics_state.clear(rsx::pipeline_state::polygon_stipple_pattern_dirty);
	}

	if (interpreter && m_interpreter_state)
	{
		if (m_interpreter_state & rsx::vertex_program_dirty)
		{
			const usz size = current_vp_metadata.ucode_length + 16;
			const auto slice = m_vertex_instructions_heap.allocate(size, 256);
			auto* output = static_cast<u8*>(m_vertex_instructions_heap.map(slice));
			auto* configuration = reinterpret_cast<u32*>(output);
			configuration[0] = current_vertex_program.base_address;
			configuration[1] = current_vertex_program.entry;
			configuration[2] = current_vertex_program.output_mask;
			configuration[3] = context->two_side_light_en() ? 1u : 0u;
			std::memcpy(output + 16, current_vertex_program.data.data(), current_vp_metadata.ucode_length);
			m_vertex_instructions_heap.mark_modified(slice);
			m_vertex_instructions_heap.unmap();
			m_vertex_instructions_binding = binding(slice);
		}
		if (m_interpreter_state & rsx::fragment_program_dirty)
		{
			const usz size = current_fp_metadata.program_ucode_length + 16;
			const auto slice = m_fragment_instructions_heap.allocate(size, 256);
			auto* output = static_cast<u8*>(m_fragment_instructions_heap.map(slice));
			auto* configuration = reinterpret_cast<u32*>(output);
			configuration[0] = context->shader_control();
			configuration[1] = current_fragment_program.texture_state.texture_dimensions;
			configuration[2] = current_fp_metadata.program_ucode_length / 16;
			configuration[3] = context->two_side_light_en() ? 1u : 0u;
			std::memcpy(output + 16, current_fragment_program.get_data(),
				current_fragment_program.ucode_length);
			m_fragment_instructions_heap.mark_modified(slice);
			m_fragment_instructions_heap.unmap();
			m_fragment_instructions_binding = binding(slice);
		}
		m_interpreter_state = 0;
	}
	if (interpreter)
	{
		if (!m_vertex_instructions_binding || !m_fragment_instructions_binding)
			fmt::throw_exception("Metal interpreter has no instruction buffers");
		m_program->set_buffer(mtl::msl_shader_stage::vertex, 9,
			m_vertex_instructions_heap.target_buffer(), m_vertex_instructions_binding.offset,
			m_vertex_instructions_binding.length);
		m_program->set_buffer(mtl::msl_shader_stage::fragment, 5,
			m_fragment_instructions_heap.target_buffer(), m_fragment_instructions_binding.offset,
			m_fragment_instructions_binding.length);
	}

	m_program->set_buffer(mtl::msl_shader_stage::vertex,
		mtl::vertex_stage_binding_table::context_buffer, m_vertex_environment_binding);
	m_program->set_buffer(mtl::msl_shader_stage::vertex,
		mtl::vertex_stage_binding_table::draw_parameters_buffer, m_vertex_layout_binding);
	if (m_conditional_render_buffer)
	{
		const u64 predicate_offset = cond_render_ctrl.hw_cond_active ? 0 : sizeof(u32);
		m_program->set_buffer(mtl::msl_shader_stage::vertex,
			mtl::vertex_stage_binding_table::conditional_render_predicate_buffer,
			*m_conditional_render_buffer, predicate_offset, sizeof(u32));
	}
	m_program->set_buffer(mtl::msl_shader_stage::fragment,
		mtl::fragment_stage_binding_table::state_buffer,
		m_fragment_environment_binding ? m_fragment_environment_binding : null_binding);
	m_program->set_buffer(mtl::msl_shader_stage::fragment,
		mtl::fragment_stage_binding_table::texture_parameters_buffer,
		m_fragment_texture_parameters_binding ? m_fragment_texture_parameters_binding : null_binding);
	m_program->set_buffer(mtl::msl_shader_stage::fragment,
		mtl::fragment_stage_binding_table::rasterizer_environment_buffer,
		m_raster_environment_binding ? m_raster_environment_binding : null_binding);
	if (m_vertex_constants_binding)
		m_program->set_buffer(mtl::msl_shader_stage::vertex,
			mtl::vertex_stage_binding_table::constants_buffer, m_vertex_constants_binding);
	if (!interpreter && m_fragment_constants_binding)
		m_program->set_buffer(mtl::msl_shader_stage::fragment,
			mtl::fragment_stage_binding_table::constants_buffer, m_fragment_constants_binding);
	if (context->current_draw_clause.is_trivial_instanced_draw && !interpreter)
	{
		m_program->set_buffer(mtl::msl_shader_stage::vertex,
			mtl::vertex_stage_binding_table::instancing_lookup_buffer,
			m_instancing_indirection_binding);
		m_program->set_buffer(mtl::msl_shader_stage::vertex,
			mtl::vertex_stage_binding_table::instancing_constants_buffer,
			m_instancing_constants_binding);
	}

	rsx::flags32_t handled = rsx::pipeline_state::fragment_state_dirty |
		rsx::pipeline_state::vertex_state_dirty |
		rsx::pipeline_state::fragment_texture_state_dirty;
	if (!context->current_draw_clause.is_trivial_instanced_draw)
		handled |= rsx::pipeline_state::transform_constants_dirty;
	if (update_constants && !interpreter)
		handled |= rsx::pipeline_state::fragment_constants_dirty;
	m_graphics_state.clear(handled);
}

mtl::submission MTLGSRender::close_and_submit_command_buffer(
	std::span<const mtl::event_operation> waits,
	std::span<const mtl::event_operation> signal_operations, bool wait_for_completion)
{
	if (!m_current_command_buffer || !m_current_command_buffer->is_recording())
		fmt::throw_exception("Metal command submission requires active recording");
	if (m_queue_status.test_and_set(flush_queue_active))
		fmt::throw_exception("Recursive Metal command submission");
	try
	{
		rsx::mm_flush();
		static_cast<void>(g_fxo->get<rsx::dma_manager>().sync());
		close_render_pass();
		mtl::get_data_heap_manager().flush_all(*m_current_command_buffer);
		if (m_current_command_buffer->has_flag(mtl::command_has_open_query))
		{
			if (!m_active_query_info)
				fmt::throw_exception("Metal command buffer has an untracked open query");
			auto& data = m_occlusion_map[m_active_query_info->driver_handle];
			if (data.indices.empty())
				fmt::throw_exception("Metal open query has no allocated result slot");
			m_occlusion_query_manager->end_query(m_render_pass, data.indices.back());
			m_current_command_buffer->clear_flag(mtl::command_has_open_query);
		}

		if (m_host_dma_ctrl && m_host_dma_ctrl->host_ctx()->needs_label_release())
		{
			const u64 event = m_host_dma_ctrl->host_ctx()->last_label_acquire_event;
			mtl::encode_buffer_update(*m_current_command_buffer, *m_host_object_data,
				::offset32(&mtl::host_data::commands_complete_event),
				std::as_bytes(std::span{&event, 1}));
			m_host_dma_ctrl->host_ctx()->on_label_release();
		}

		std::vector<mtl::event_operation> effective_waits(waits.begin(), waits.end());
		if (m_async_scheduler && m_async_scheduler.is_recording())
		{
			const mtl::async_submission asynchronous = m_async_scheduler.flush({}, true);
			if (asynchronous) effective_waits.push_back(asynchronous.synchronization.wait_operation());
		}
		m_current_command_buffer->end();
		m_current_command_buffer->tag();
		mtl::submit_info information;
		information.queue = mtl::queue_kind::graphics;
		information.waits = std::move(effective_waits);
		information.signal_operations.assign(signal_operations.begin(), signal_operations.end());
		information.wait_for_completion = wait_for_completion;
		const std::vector<mtl::data_heap*> heaps = mtl::get_data_heap_manager().heaps();
		mtl::command_stream_ticket ticket = m_command_stream.submit(*m_current_command_buffer,
			std::move(information), {}, heaps, true);
		ticket.rethrow_if_failed();
		mtl::submission result = ticket.native_submission();
		if (!result) fmt::throw_exception("Metal graphics stream produced no submission");
		mtl::get_data_heap_manager().seal_all(result.value());
		for (auto& query_data : m_occlusion_map)
		{
			for (const mtl::query_handle query : query_data.indices)
			{
				const auto status = m_occlusion_query_manager->check_query_status(query);
				if (status.state == mtl::query_state::ended)
					m_occlusion_query_manager->mark_submitted(query, result.value());
			}
		}
		m_queue_status.clear(flush_queue_active);
		return result;
	}
	catch (...)
	{
		m_queue_status.clear(flush_queue_active);
		throw;
	}
}

void MTLGSRender::reclaim_completed_resources()
{
	if (!m_shared_state || !*m_shared_state) return;
	const u64 completed = m_shared_state->completed_submission();
	mtl::get_data_heap_manager().reclaim_all(completed);
	mtl::notify_dma_completed(completed);
	if (m_occlusion_query_manager) m_occlusion_query_manager->notify_completed(completed, true);
	if (const u64 resource_event = mtl::last_completed_resource_event(); resource_event)
		m_overlay_passes.reclaim(resource_event);
	m_resources.reset(completed);
	m_command_stream.collect_completed();
	if (m_async_scheduler) m_async_scheduler.reclaim_completed();
}

void MTLGSRender::flush_command_queue(bool hard_sync, bool preserve_current)
{
	mtl::submission submitted = close_and_submit_command_buffer({}, {}, hard_sync);
	if (hard_sync)
	{
		submitted.wait();
		if (!submitted.succeeded())
			fmt::throw_exception("Metal hard synchronization submission failed");
		reclaim_completed_resources();
		m_primary_commands.poke_all();
		while (!m_queued_frames.empty())
			check_present_status();
		m_flush_requests.clear_pending_flag();
	}
	if (preserve_current)
	{
		if (!hard_sync)
			fmt::throw_exception("Metal command preservation requires hard synchronization");
		m_current_command_buffer->reset();
	}
	else
	{
		m_current_command_buffer = m_primary_commands.next();
	}
	if (m_occlusion_query_active)
		m_current_command_buffer->set_flag(mtl::command_loads_occlusion_task);
	m_current_command_buffer->begin();
}

void MTLGSRender::close_render_pass()
{
	if (!m_render_pass.is_open()) return;
	if (m_current_command_buffer &&
		m_current_command_buffer->has_flag(mtl::command_has_open_query) &&
		m_active_query_info)
	{
		auto& data = m_occlusion_map[m_active_query_info->driver_handle];
		if (!data.indices.empty())
			m_occlusion_query_manager->suspend_query(m_render_pass, data.indices.back());
	}
	m_render_pass.end();
}

void MTLGSRender::begin_render_pass()
{
	if (!m_draw_framebuffer) fmt::throw_exception("Metal render pass has no framebuffer");
	m_render_pass_configuration.label = "RPCS3 RSX draw";
	if (m_occlusion_query_manager)
		m_occlusion_query_manager->configure_render_pass(m_render_pass_configuration,
			mtl::visibility_result_behavior::accumulate);
	static_cast<void>(m_render_pass.ensure(*m_current_command_buffer,
		m_draw_framebuffer, m_render_pass_configuration));
	if (m_occlusion_query_active &&
		m_current_command_buffer->has_flag(mtl::command_loads_occlusion_task))
	{
		auto& data = m_occlusion_map[m_active_query_info->driver_handle];
		if (data.indices.empty() || !m_occlusion_query_manager->is_suspended())
		{
			const bool accumulate = !data.indices.empty();
			const mtl::query_handle query = m_occlusion_query_manager->allocate_query();
			data.indices.push_back(query);
			data.set_sync_command_buffer(m_current_command_buffer);
			m_occlusion_query_manager->begin_query(m_render_pass, query, accumulate);
		}
		else
			m_occlusion_query_manager->resume_query(m_render_pass, data.indices.back());
		m_current_command_buffer->set_flag(mtl::command_has_open_query);
		m_current_command_buffer->clear_flag(mtl::command_loads_occlusion_task);
	}
}

void MTLGSRender::begin_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	if (!query || m_occlusion_query_active)
		fmt::throw_exception("Invalid Metal occlusion-query begin");
	if (query->driver_handle >= m_occlusion_map.size())
		fmt::throw_exception("Metal occlusion-query handle %u is out of range", query->driver_handle);
	query->result = 0;
	m_active_query_info = query;
	m_occlusion_query_active = true;
	m_current_command_buffer->set_flag(mtl::command_has_occlusion_task);
	m_current_command_buffer->set_flag(mtl::command_loads_occlusion_task);
}

void MTLGSRender::end_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	if (!query || query != m_active_query_info || !m_occlusion_query_active)
		fmt::throw_exception("Invalid Metal occlusion-query end");
	if (m_current_command_buffer->has_flag(mtl::command_has_open_query))
	{
		if (m_render_pass.is_open()) close_render_pass();
		auto& data = m_occlusion_map[query->driver_handle];
		if (data.indices.empty())
			fmt::throw_exception("Metal active occlusion query has no result slot");
		m_occlusion_query_manager->end_query(m_render_pass, data.indices.back());
		m_current_command_buffer->clear_flag(mtl::command_has_open_query);
	}
	m_current_command_buffer->clear_flag(mtl::command_loads_occlusion_task);
	m_occlusion_query_active = false;
	m_active_query_info = nullptr;
}

bool MTLGSRender::check_occlusion_query_status(rsx::reports::occlusion_query_info* query)
{
	if (!query || query->driver_handle >= m_occlusion_map.size())
		fmt::throw_exception("Invalid Metal occlusion-query status request");
	if (!query->num_draws) return true;
	auto& data = m_occlusion_map[query->driver_handle];
	if (data.indices.empty()) return true;
	if (data.is_current(m_current_command_buffer)) return false;
	if (m_shared_state && *m_shared_state)
		m_occlusion_query_manager->notify_completed(m_shared_state->completed_submission(), true);
	for (const mtl::query_handle handle : data.indices)
	{
		if (!m_occlusion_query_manager->check_query_status(handle).ready()) return false;
	}
	return true;
}

void MTLGSRender::get_occlusion_query_result(rsx::reports::occlusion_query_info* query)
{
	if (!query || query->driver_handle >= m_occlusion_map.size())
		fmt::throw_exception("Invalid Metal occlusion-query result request");
	auto& data = m_occlusion_map[query->driver_handle];
	if (data.indices.empty()) return;
	if (data.is_current(m_current_command_buffer))
	{
		std::lock_guard lock(m_flush_queue_mutex);
		flush_command_queue();
		m_flush_requests.clear_pending_flag();
		rsx_log.warning("[Performance warning] Metal ZCULL read caused a GPU synchronization");
	}
	data.sync();
	m_occlusion_query_manager->notify_completed(m_shared_state->completed_submission(), true);
	query->result = 0;
	if (query->num_draws)
	{
		for (const mtl::query_handle handle : data.indices)
		{
			const mtl::query_result result = m_occlusion_query_manager->get_query_result(handle);
			if (result.availability == mtl::query_availability::failed)
				fmt::throw_exception("Metal occlusion-query submission failed");
			if (!result)
				fmt::throw_exception("Metal occlusion-query result is not available");
			query->result += static_cast<u32>(result.value);
			if (query->result && !g_cfg.video.precise_zpass_count) break;
		}
	}
	for (const mtl::query_handle handle : data.indices)
		m_occlusion_query_manager->release_query(handle);
	data.indices.clear();
}

void MTLGSRender::discard_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	if (!query || query->driver_handle >= m_occlusion_map.size())
		fmt::throw_exception("Invalid Metal occlusion-query discard request");
	if (m_active_query_info == query) end_occlusion_query(query);
	auto& data = m_occlusion_map[query->driver_handle];
	if (data.indices.empty()) return;
	if (data.is_current(m_current_command_buffer)) flush_command_queue();
	data.sync();
	m_occlusion_query_manager->notify_completed(m_shared_state->completed_submission(), true);
	for (const mtl::query_handle handle : data.indices)
		m_occlusion_query_manager->release_query(handle);
	data.indices.clear();
}

void MTLGSRender::emergency_query_cleanup(mtl::command_buffer* commands)
{
	if (commands != m_current_command_buffer)
		fmt::throw_exception("Metal query cleanup received a foreign command buffer");
	if (!m_current_command_buffer->has_flag(mtl::command_has_open_query)) return;
	if (!m_active_query_info)
		fmt::throw_exception("Metal query cleanup has no logical query");
	if (m_render_pass.is_open()) close_render_pass();
	auto& data = m_occlusion_map[m_active_query_info->driver_handle];
	if (data.indices.empty())
		fmt::throw_exception("Metal query cleanup has no result slot");
	m_occlusion_query_manager->end_query(m_render_pass, data.indices.back());
	m_current_command_buffer->clear_flag(mtl::command_has_open_query);
	if (m_occlusion_query_active)
		m_current_command_buffer->set_flag(mtl::command_loads_occlusion_task);
}

void MTLGSRender::begin_conditional_rendering(
	const std::vector<rsx::reports::occlusion_query_info*>& sources)
{
	if (sources.empty() || !m_conditional_render_buffer)
		fmt::throw_exception("Invalid Metal conditional-render request");
	if (m_current_command_buffer->has_flag(mtl::command_has_open_query))
		emergency_query_cleanup(m_current_command_buffer);

	bool requires_submission = false;
	for (const auto* source : sources)
	{
		if (!source || source->driver_handle >= m_occlusion_map.size())
			fmt::throw_exception("Metal conditional rendering has an invalid query source");
		requires_submission |= m_occlusion_map[source->driver_handle].is_current(
			m_current_command_buffer);
	}
	if (requires_submission) flush_command_queue();

	u32 predicate = 0;
	for (const auto* source : sources)
	{
		auto& data = m_occlusion_map[source->driver_handle];
		if (data.indices.empty()) continue;
		data.sync();
		m_occlusion_query_manager->notify_completed(m_shared_state->completed_submission(), true);
		for (const mtl::query_handle handle : data.indices)
		{
			const mtl::query_result result = m_occlusion_query_manager->get_query_result(handle);
			if (result.availability == mtl::query_availability::failed)
				fmt::throw_exception("Metal conditional-render query submission failed");
			if (!result)
				fmt::throw_exception("Metal conditional-render query is unavailable");
			predicate |= result.value != 0;
			if (predicate) break;
		}
		if (predicate) break;
	}
	mtl::encode_buffer_update(*m_current_command_buffer, *m_conditional_render_buffer, 0,
		std::as_bytes(std::span{&predicate, 1}));
	m_conditional_render_sync_tag = sources.front()->sync_tag;
	m_current_command_buffer->set_flag(mtl::command_has_conditional_render);
	rsx::thread::begin_conditional_rendering(sources);
}

void MTLGSRender::end_conditional_rendering()
{
	rsx::thread::end_conditional_rendering();
	if (m_current_command_buffer)
		m_current_command_buffer->clear_flag(mtl::command_has_conditional_render);
}

std::pair<volatile mtl::host_data*, mtl::buffer_handle>
MTLGSRender::map_host_object_data() const
{
	if (!m_host_dma_ctrl || !m_host_object_data)
		fmt::throw_exception("Metal host-GPU synchronization is unavailable");
	return {m_host_dma_ctrl->host_ctx(), m_host_object_data->native_handle()};
}

bool MTLGSRender::release_GCM_label(u32 type, u32 address, u32 data)
{
	if (!backend_config.supports_host_gpu_labels || !m_host_dma_ctrl || !m_host_object_data)
		return false;
	volatile mtl::host_data* context = m_host_dma_ctrl->host_ctx();
	if (type == NV4097_TEXTURE_READ_SEMAPHORE_RELEASE && context->texture_loads_completed())
	{
		m_host_dma_ctrl->drain_label_queue();
		return false;
	}
	const mtl::dma_mapping_handle mapping = mtl::map_dma(address, sizeof(u32));
	if (!mapping || mapping.owner->head()->kind() != mtl::dma_block_kind::host_no_copy)
	{
		rsx_log.warning("Metal host label update at 0x%x cannot use direct guest memory", address);
		m_host_dma_ctrl->drain_label_queue();
		return false;
	}
	static_cast<void>(context->on_label_acquire());
	const u32 encoded = std::bit_cast<u32, be_t<u32>>(data);
	close_render_pass();
	mtl::encode_buffer_update(*m_current_command_buffer, *mapping.resource, mapping.offset,
		std::as_bytes(std::span{&encoded, 1}));
	m_current_command_buffer->set_flag(mtl::command_has_dma_transfer);
	flush_command_queue();
	return true;
}

void MTLGSRender::on_guest_texture_read(const mtl::command_buffer& command)
{
	if (!backend_config.supports_host_gpu_labels || !m_host_dma_ctrl || !m_host_object_data)
		return;
	volatile mtl::host_data* context = m_host_dma_ctrl->host_ctx();
	const u64 event = context->on_texture_load_acquire();
	if (&command == m_current_command_buffer) close_render_pass();
	auto& mutable_command = const_cast<mtl::command_buffer&>(command);
	mtl::encode_buffer_update(mutable_command, *m_host_object_data,
		::offset32(&mtl::host_data::texture_load_complete_event),
		std::as_bytes(std::span{&event, 1}));
}

void MTLGSRender::write_barrier(u32 address, u32 range)
{
	if (!is_current_thread())
		fmt::throw_exception("Metal render-target write barrier was requested off the RSX thread");
	m_render_targets.invalidate_range(utils::address_range32::start_length(address, range));
}

void MTLGSRender::sync_hint(rsx::FIFO::interrupt_hint hint,
	rsx::reports::sync_hint_payload_t payload)
{
	rsx::thread::sync_hint(hint, payload);
	if (!m_current_command_buffer->has_flag(mtl::command_has_occlusion_task)) return;
	switch (hint)
	{
	case rsx::FIFO::interrupt_hint::conditional_render_eval:
	{
		if (m_flush_requests.pending()) return;
		const u32 address = static_cast<u32>(payload.address);
		if (!zcull_ctrl->is_query_result_urgent(address)) return;
		const u64 now = get_system_time();
		if (now - m_last_conditional_render_evaluation > 50)
		{
			m_flush_requests.post(false);
			m_flush_requests.remove_one();
		}
		m_last_conditional_render_evaluation = now;
		break;
	}
	case rsx::FIFO::interrupt_hint::zcull_sync:
	{
		if (!payload.query || payload.query->driver_handle >= m_occlusion_map.size()) return;
		auto& query = m_occlusion_map[payload.query->driver_handle];
		if (!query.is_current(m_current_command_buffer) || query.indices.empty()) return;
		std::lock_guard lock(m_flush_queue_mutex);
		flush_command_queue();
		m_flush_requests.clear_pending_flag();
		break;
	}
	}
}

void MTLGSRender::invalidate_render_pass()
{
	close_render_pass();
	m_draw_framebuffer.reset();
	m_encoder_bindings.invalidate();
}

void MTLGSRender::set_viewport()
{
	const auto [width, height] = rsx::apply_resolution_scale<true>(resolution_scaling_config,
		rsx::method_registers.surface_clip_width(), rsx::method_registers.surface_clip_height());
	m_viewport = {0.0, 0.0, static_cast<f64>(width), static_cast<f64>(height),
		static_cast<f64>(rsx::method_registers.clip_min()),
		static_cast<f64>(rsx::method_registers.clip_max())};
	m_encoder_bindings.viewport_valid = false;
	m_graphics_state.clear(rsx::pipeline_state::zclip_config_state_dirty);
}

void MTLGSRender::set_scissor(bool clip_viewport)
{
	areau region;
	if (!get_scissor(region, clip_viewport)) return;
	m_scissor = {region.x1, region.y1, region.width(), region.height()};
	m_encoder_bindings.scissor_valid = false;
}

void MTLGSRender::bind_viewport()
{
	if (!m_render_pass.is_open()) fmt::throw_exception("Metal viewport binding requires a render pass");
	if (m_graphics_state & rsx::pipeline_state::zclip_config_state_dirty)
	{
		m_viewport.minimum_depth = rsx::method_registers.clip_min();
		m_viewport.maximum_depth = rsx::method_registers.clip_max();
		m_graphics_state.clear(rsx::pipeline_state::zclip_config_state_dirty);
	}
	mtl::encode_viewport_scissor(m_render_pass.native_encoder(), m_viewport, m_scissor);
	m_encoder_bindings.viewport_valid = true;
	m_encoder_bindings.scissor_valid = true;
}

void MTLGSRender::initialize_buffers(rsx::framebuffer_creation_context context, bool)
{
	prepare_render_targets(context);
}

void MTLGSRender::prepare_render_targets(rsx::framebuffer_creation_context context)
{
	const bool clipped_scissor = context == rsx::framebuffer_creation_context::context_draw;
	if (m_current_framebuffer_context == context &&
		!m_graphics_state.test(rsx::rtt_config_dirty) && m_draw_framebuffer)
	{
		set_scissor(clipped_scissor);
		return;
	}

	invalidate_render_pass();
	m_graphics_state.clear(rsx::rtt_config_dirty | rsx::rtt_config_contested |
		rsx::rtt_config_valid | rsx::rtt_cache_state_dirty);
	get_framebuffer_layout(context, m_framebuffer_layout);
	if (!m_graphics_state.test(rsx::rtt_config_valid)) return;
	if (m_draw_framebuffer && m_framebuffer_layout.ignore_change)
	{
		set_scissor(clipped_scissor);
		return;
	}

	m_render_targets.prepare_render_target(*m_current_command_buffer,
		m_framebuffer_layout.color_format, m_framebuffer_layout.depth_format,
		m_framebuffer_layout.width, m_framebuffer_layout.height,
		m_framebuffer_layout.target, m_framebuffer_layout.aa_mode,
		m_framebuffer_layout.raster_type, m_framebuffer_layout.color_addresses,
		m_framebuffer_layout.zeta_address, m_framebuffer_layout.actual_color_pitch,
		m_framebuffer_layout.actual_zeta_pitch, resolution_scaling_config,
		*m_allocator, *m_device, *m_current_command_buffer);

	const u32 color_bytes = get_format_block_size_in_bytes(m_framebuffer_layout.color_format);
	const u32 samples = get_format_sample_count(m_framebuffer_layout.aa_mode);
	for (u32 index = 0; index < rsx::limits::color_buffers_count; ++index)
	{
		if (m_surface_info[index].pitch && g_cfg.video.write_color_buffers)
		{
			const auto range = m_surface_info[index].get_memory_range();
			m_texture_cache.set_memory_read_flags(range, rsx::memory_read_flags::flush_once);
			m_texture_cache.flush_if_cache_miss_likely(*m_current_command_buffer, range);
		}
		m_surface_info[index].address = 0;
		m_surface_info[index].pitch = 0;
		m_surface_info[index].width = m_framebuffer_layout.width;
		m_surface_info[index].height = m_framebuffer_layout.height;
		m_surface_info[index].color_format = m_framebuffer_layout.color_format;
		m_surface_info[index].bpp = color_bytes;
		m_surface_info[index].samples = samples;
	}
	if (m_depth_surface_info.pitch && g_cfg.video.write_depth_buffer)
	{
		const auto range = m_depth_surface_info.get_memory_range();
		m_texture_cache.set_memory_read_flags(range, rsx::memory_read_flags::flush_once);
		m_texture_cache.flush_if_cache_miss_likely(*m_current_command_buffer, range);
	}
	m_depth_surface_info.address = 0;
	m_depth_surface_info.pitch = 0;
	m_depth_surface_info.width = m_framebuffer_layout.width;
	m_depth_surface_info.height = m_framebuffer_layout.height;
	m_depth_surface_info.depth_format = m_framebuffer_layout.depth_format;
	m_depth_surface_info.bpp = get_format_block_size_in_bytes(m_framebuffer_layout.depth_format);
	m_depth_surface_info.samples = samples;

	m_draw_buffers.clear();
	m_framebuffer_images.clear();
	for (const u8 index : rsx::utility::get_rtt_indexes(m_framebuffer_layout.target))
	{
		mtl::render_target* surface = std::get<1>(m_render_targets.m_bound_render_targets[index]);
		if (!surface) continue;
		m_framebuffer_images.push_back(surface);
		m_surface_info[index].address = m_framebuffer_layout.color_addresses[index];
		m_surface_info[index].pitch = m_framebuffer_layout.actual_color_pitch[index];
		if (surface->rsx_pitch != m_framebuffer_layout.actual_color_pitch[index])
			fmt::throw_exception("Metal color surface pitch does not match RSX state");
		m_texture_cache.notify_surface_changed(
			m_surface_info[index].get_memory_range(m_framebuffer_layout.aa_factors));
		m_draw_buffers.push_back(index);
	}
	if (mtl::render_target* depth = std::get<1>(m_render_targets.m_bound_depth_stencil))
	{
		m_framebuffer_images.push_back(depth);
		m_depth_surface_info.address = m_framebuffer_layout.zeta_address;
		m_depth_surface_info.pitch = m_framebuffer_layout.actual_zeta_pitch;
		if (depth->rsx_pitch != m_framebuffer_layout.actual_zeta_pitch)
			fmt::throw_exception("Metal depth surface pitch does not match RSX state");
		m_texture_cache.notify_surface_changed(
			m_depth_surface_info.get_memory_range(m_framebuffer_layout.aa_factors));
	}

	if (m_current_command_buffer->has_flag(mtl::command_has_dma_transfer))
		flush_command_queue();
	for (auto& surface : m_render_targets.superseded_surfaces)
		m_texture_cache.discard_framebuffer_memory_region(*m_current_command_buffer,
			surface->get_memory_range());
	m_render_targets.superseded_surfaces.clear();

	for (auto& [address, surface] : m_render_targets.orphaned_surfaces)
	{
		const bool writeback = surface->is_depth_surface() ?
			static_cast<bool>(g_cfg.video.write_depth_buffer) :
			static_cast<bool>(g_cfg.video.write_color_buffers);
		if (!writeback || !surface->is_locked())
		{
			m_texture_cache.commit_framebuffer_memory_region(*m_current_command_buffer,
				surface->get_memory_range());
			continue;
		}
		u32 format = 0;
		bool swap_bytes = false;
		if (surface->is_depth_surface())
		{
			format = surface->get_surface_depth_format() == rsx::surface_depth_format::z16
				? CELL_GCM_TEXTURE_DEPTH16 : CELL_GCM_TEXTURE_DEPTH24_D8;
			swap_bytes = true;
		}
		else
		{
			std::tie(format, swap_bytes) = get_compatible_gcm_format(surface->get_surface_color_format());
		}
		m_texture_cache.lock_memory_region(*m_current_command_buffer, surface,
			surface->get_memory_range(), false,
			surface->get_surface_width<rsx::surface_metrics::pixels>(),
			surface->get_surface_height<rsx::surface_metrics::pixels>(),
			surface->get_rsx_pitch(), format, swap_bytes);
		static_cast<void>(address);
	}
	m_render_targets.orphaned_surfaces.clear();

	const auto color_format = get_compatible_gcm_format(m_framebuffer_layout.color_format);
	for (const u8 index : m_draw_buffers)
	{
		if (!m_surface_info[index].address || !m_surface_info[index].pitch) continue;
		const auto range = m_surface_info[index].get_memory_range();
		if (g_cfg.video.write_color_buffers)
			m_texture_cache.lock_memory_region(*m_current_command_buffer,
				std::get<1>(m_render_targets.m_bound_render_targets[index]), range, true,
				m_surface_info[index].width, m_surface_info[index].height,
				m_framebuffer_layout.actual_color_pitch[index], color_format.first, color_format.second);
		else
			m_texture_cache.commit_framebuffer_memory_region(*m_current_command_buffer, range);
	}
	if (m_depth_surface_info.address && m_depth_surface_info.pitch)
	{
		const auto range = m_depth_surface_info.get_memory_range();
		if (g_cfg.video.write_depth_buffer)
		{
			const u32 format = m_depth_surface_info.depth_format == rsx::surface_depth_format::z16
				? CELL_GCM_TEXTURE_DEPTH16 : CELL_GCM_TEXTURE_DEPTH24_D8;
			m_texture_cache.lock_memory_region(*m_current_command_buffer,
				std::get<1>(m_render_targets.m_bound_depth_stencil), range, true,
				m_depth_surface_info.width, m_depth_surface_info.height,
				m_framebuffer_layout.actual_zeta_pitch, format, true);
		}
		else
			m_texture_cache.commit_framebuffer_memory_region(*m_current_command_buffer, range);
	}

	mtl::framebuffer_request framebuffer;
	std::tie(framebuffer.width, framebuffer.height) = rsx::apply_resolution_scale<true>(
		resolution_scaling_config, m_framebuffer_layout.width, m_framebuffer_layout.height);
	framebuffer.samples = samples;
	for (u32 output = 0; output < m_draw_buffers.size(); ++output)
	{
		mtl::render_target* surface = std::get<1>(
			m_render_targets.m_bound_render_targets[m_draw_buffers[output]]);
		framebuffer.colors[output].render.resource = surface;
		framebuffer.colors[output].render.subresources = {0, 1, 0, 1, true, false, false};
	}
	if (mtl::render_target* depth = std::get<1>(m_render_targets.m_bound_depth_stencil))
	{
		if (depth->aspects() & mtl::texture_aspect_depth)
		{
			framebuffer.depth.render.resource = depth;
			framebuffer.depth.render.subresources = {0, 1, 0, 1, false, true, false};
		}
		if (depth->aspects() & mtl::texture_aspect_stencil)
		{
			framebuffer.stencil.render.resource = depth;
			framebuffer.stencil.render.subresources = {0, 1, 0, 1, false, false, true};
		}
	}
	m_draw_framebuffer = mtl::get_framebuffer(framebuffer);
	set_viewport();
	set_scissor(clipped_scissor);
	on_framebuffer_layout_updated();
	check_zcull_status(true);
}

std::shared_ptr<mtl::framebuffer> MTLGSRender::get_framebuffer()
{
	if (!m_draw_framebuffer) prepare_render_targets(rsx::framebuffer_creation_context::context_draw);
	if (!m_draw_framebuffer) fmt::throw_exception("Metal draw has no framebuffer");
	return m_draw_framebuffer;
}

void MTLGSRender::clear_surface(u32 mask)
{
	if (skip_current_frame || m_swapchain_unavailable) return;
	if (!rsx::method_registers.stencil_mask()) mask &= ~RSX_GCM_CLEAR_STENCIL_BIT;
	if (!(mask & RSX_GCM_CLEAR_ANY_MASK)) return;
	u8 creation = rsx::framebuffer_creation_context::context_draw;
	if (mask & RSX_GCM_CLEAR_COLOR_RGBA_MASK)
		creation |= rsx::framebuffer_creation_context::context_clear_color;
	if (mask & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK)
		creation |= rsx::framebuffer_creation_context::context_clear_depth;
	initialize_buffers(rsx::framebuffer_creation_context{creation});
	if (!m_graphics_state.test(rsx::rtt_config_valid) || !m_draw_framebuffer) return;

	u16 x = static_cast<u16>(m_scissor.x);
	u16 y = static_cast<u16>(m_scissor.y);
	u16 width = static_cast<u16>(m_scissor.width);
	u16 height = static_cast<u16>(m_scissor.height);
	std::tie(x, y, width, height) = rsx::clip_region<u16>(m_draw_framebuffer->width(),
		m_draw_framebuffer->height(), x, y, width, height, true);
	if (!width || !height) return;
	const areau rectangle{x, y, static_cast<u32>(x + width), static_cast<u32>(y + height)};
	const bool full_frame = x == 0 && y == 0 && width == m_draw_framebuffer->width() &&
		height == m_draw_framebuffer->height();
	mtl::render_pass_configuration fast_clear;
	fast_clear.label = "RPCS3 attachment clear";
	bool has_fast_clear = false;
	bool update_color = false;
	bool update_depth = false;

	u32 color_mask = mask & RSX_GCM_CLEAR_COLOR_RGBA_MASK;
	if (color_mask && !m_draw_buffers.empty())
	{
		u8 alpha = rsx::method_registers.clear_color_a();
		u8 red = rsx::method_registers.clear_color_r();
		u8 green = rsx::method_registers.clear_color_g();
		u8 blue = rsx::method_registers.clear_color_b();
		bool all_native_channels = color_mask == RSX_GCM_CLEAR_COLOR_RGBA_MASK;
		switch (rsx::method_registers.surface_color())
		{
		case rsx::surface_color_format::x32:
		case rsx::surface_color_format::w16z16y16x16:
		case rsx::surface_color_format::w32z32y32x32:
			color_mask = 0;
			break;
		case rsx::surface_color_format::b8:
			rsx::get_b8_clear_color(red, green, blue, alpha);
			color_mask = rsx::get_b8_clearmask(color_mask);
			all_native_channels = color_mask & RSX_GCM_CLEAR_RED_BIT;
			break;
		case rsx::surface_color_format::g8b8:
			rsx::get_g8b8_clear_color(red, green, blue, alpha);
			color_mask = rsx::get_g8b8_r8g8_clearmask(color_mask);
			all_native_channels = (color_mask & RSX_GCM_CLEAR_COLOR_RG_MASK) ==
				RSX_GCM_CLEAR_COLOR_RG_MASK;
			break;
		case rsx::surface_color_format::r5g6b5:
			rsx::get_rgb565_clear_color(red, green, blue, alpha);
			all_native_channels = (color_mask & RSX_GCM_CLEAR_COLOR_RGB_MASK) ==
				RSX_GCM_CLEAR_COLOR_RGB_MASK;
			break;
		case rsx::surface_color_format::x1r5g5b5_o1r5g5b5:
			rsx::get_a1rgb555_clear_color(red, green, blue, alpha, 255);
			break;
		case rsx::surface_color_format::x1r5g5b5_z1r5g5b5:
			rsx::get_a1rgb555_clear_color(red, green, blue, alpha, 0);
			break;
		case rsx::surface_color_format::a8b8g8r8:
		case rsx::surface_color_format::x8b8g8r8_o8b8g8r8:
		case rsx::surface_color_format::x8b8g8r8_z8b8g8r8:
			rsx::get_abgr8_clear_color(red, green, blue, alpha);
			color_mask = rsx::get_abgr8_clearmask(color_mask);
			break;
		default: break;
		}
		if (color_mask)
		{
			const mtl::clear_color_value clear{
				static_cast<f64>(red) / 255.0, static_cast<f64>(green) / 255.0,
				static_cast<f64>(blue) / 255.0, static_cast<f64>(alpha) / 255.0};
			if (all_native_channels && full_frame)
			{
				for (u32 output = 0; output < m_draw_buffers.size(); ++output)
				{
					fast_clear.colors[output].load = mtl::attachment_load_action::clear;
					fast_clear.colors[output].clear_color = clear;
				}
				has_fast_clear = true;
			}
			else
			{
				const color4f clear_color{static_cast<f32>(clear.red), static_cast<f32>(clear.green),
					static_cast<f32>(clear.blue), static_cast<f32>(clear.alpha)};
				for (const u8 index : m_draw_buffers)
				{
					mtl::render_target* target = std::get<1>(
						m_render_targets.m_bound_render_targets[index]);
					m_overlay_passes.color_clear().run(*m_current_command_buffer,
						{target, 0, 0, mtl::texture_aspect_color, true}, rectangle,
						color_mask, clear_color);
				}
			}
			update_color = true;
		}
	}

	if (mtl::render_target* depth = std::get<1>(m_render_targets.m_bound_depth_stencil);
		depth && (mask & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK))
	{
		if (mask & RSX_GCM_CLEAR_DEPTH_BIT)
		{
			const u32 maximum = get_max_depth_value(m_framebuffer_layout.depth_format);
			const u32 raw = rsx::method_registers.z_clear_value(
				is_depth_stencil_format(m_framebuffer_layout.depth_format));
			const f32 value = static_cast<f32>(raw) / maximum;
			if (full_frame)
			{
				fast_clear.depth.load = mtl::attachment_load_action::clear;
				fast_clear.depth.clear_depth = value;
				has_fast_clear = true;
			}
			else
				m_overlay_passes.depth_clear().run(*m_current_command_buffer, *depth, rectangle, value);
			update_depth = true;
		}
		if ((mask & RSX_GCM_CLEAR_STENCIL_BIT) &&
			(depth->aspects() & mtl::texture_aspect_stencil))
		{
			const u32 value = rsx::method_registers.stencil_clear_value();
			const u32 write_mask = rsx::method_registers.stencil_mask();
			if (full_frame && write_mask == 0xff)
			{
				fast_clear.stencil.load = mtl::attachment_load_action::clear;
				fast_clear.stencil.clear_stencil = value;
				depth->stencil_init_flags = value | 0x100;
				has_fast_clear = true;
			}
			else
				m_overlay_passes.stencil_clear().run(*m_current_command_buffer, *depth,
					rectangle, value, write_mask);
			update_depth = true;
		}
	}

	if (has_fast_clear)
	{
		close_render_pass();
		static_cast<void>(m_render_pass.begin(*m_current_command_buffer,
			m_draw_framebuffer, fast_clear));
		m_render_pass.end();
	}
	if (update_color || update_depth)
		m_render_targets.on_write({update_color, update_color, update_color, update_color}, update_depth);
}

void MTLGSRender::patch_transform_constants(rsx::context*, u32 index, u32 count)
{
	if (!m_program || !m_vertex_program || !m_vertex_program->overlaps_constants_range(index, count))
	{
		m_graphics_state |= rsx::pipeline_state::transform_constants_dirty;
		return;
	}
	mtl::data_heap_slice slice;
	rsx::io_buffer destination([&](usz size) -> std::pair<void*, usz>
	{
		slice = m_transform_constants_heap.allocate(size, 256);
		return {m_transform_constants_heap.map(slice), size};
	});
	upload_transform_constants(destination);
	if (slice)
	{
		m_transform_constants_heap.mark_modified(slice);
		m_transform_constants_heap.unmap();
		m_vertex_constants_binding = {.resource = slice.buffer, .gpu_address = slice.buffer_gpu_address(),
			.offset = slice.offset, .length = slice.size};
		m_transform_constants_offset = slice.offset;
		m_program->set_buffer(mtl::msl_shader_stage::vertex,
			mtl::vertex_stage_binding_table::constants_buffer, m_vertex_constants_binding);
	}
}

bool MTLGSRender::on_access_violation(u32 address, bool is_writing)
{
	rsx::mm_flush(address);
	mtl::texture_cache::thrashed_set result;
	{
		std::lock_guard lock(m_secondary_command_guard);
		auto* commands = m_secondary_commands.next();
		const rsx::invalidation_cause cause = is_writing
			? rsx::invalidation_cause::deferred_write
			: rsx::invalidation_cause::deferred_read;
		result = m_texture_cache.invalidate_address(*commands, address, cause);
	}
	if (result.invalidate_samplers)
	{
		std::lock_guard lock(m_sampler_mutex);
		m_samplers_dirty.store(true);
	}
	if (!result.violation_handled) return zcull_ctrl->on_access_violation(address);
	if (result.num_flushable <= 0) return true;

	if (g_fxo->get<rsx::dma_manager>().is_current_thread())
	{
		if (m_queue_status & flush_queue_deadlock)
			fmt::throw_exception("Nested Metal DMA memory-fault deadlock");
		m_offloader_fault_range = g_fxo->get<rsx::dma_manager>().get_fault_range(is_writing);
		m_offloader_fault_cause = is_writing ? rsx::invalidation_cause::write :
			rsx::invalidation_cause::read;
		g_fxo->get<rsx::dma_manager>().set_mem_fault_flag();
		m_queue_status |= flush_queue_deadlock;
		m_eng_interrupt_mask |= rsx::backend_interrupt;
		while (m_queue_status & flush_queue_deadlock) utils::pause();
		g_fxo->get<rsx::dma_manager>().clear_mem_fault_flag();
		return true;
	}

	bool owns_flush_request = false;
	std::function<void()> transfer_ready;
	if (!is_current_thread())
	{
		vm::temporary_unlock();
		m_flush_queue_mutex.lock();
		m_flush_requests.post(false);
		m_eng_interrupt_mask |= rsx::backend_interrupt;
		owns_flush_request = true;
		m_flush_requests.producer_wait();
		transfer_ready = [&]
		{
			m_flush_requests.remove_one();
			owns_flush_request = false;
		};
		m_flush_queue_mutex.unlock();
	}
	else
	{
		if (mtl::is_uninterruptible()) rsx_log.error("Metal memory fault in protected renderer code");
		flush_command_queue();
	}

	{
		std::lock_guard lock(m_secondary_command_guard);
		auto* commands = m_secondary_commands.next();
		static_cast<void>(m_texture_cache.flush_all(*commands, result, transfer_ready));
	}
	if (owns_flush_request) m_flush_requests.remove_one();
	return true;
}

void MTLGSRender::on_invalidate_memory_range(const utils::address_range32& range,
	rsx::invalidation_cause cause)
{
	std::lock_guard lock(m_secondary_command_guard);
	auto* commands = m_secondary_commands.next();
	auto result = m_texture_cache.invalidate_range(*commands, range, cause);
	if (!result.empty())
		fmt::throw_exception("Metal range invalidation left deferred texture-cache work");
	if (cause == rsx::invalidation_cause::unmap)
	{
		if (result.violation_handled)
		{
			m_texture_cache.purge_unreleased_sections();
			std::lock_guard sampler_lock(m_sampler_mutex);
			m_samplers_dirty.store(true);
		}
		static_cast<void>(mtl::unmap_dma(range.start, range.length()));
	}
}

void MTLGSRender::on_semaphore_acquire_wait()
{
	if (m_flush_requests.pending() ||
		(async_flip_requested & flip_request::emu_requested) ||
		(m_queue_status & flush_queue_deadlock))
		do_local_task(rsx::FIFO::state::lock_wait);
}

void MTLGSRender::do_local_task(rsx::FIFO::state state)
{
	if (m_queue_status & flush_queue_deadlock)
	{
		on_invalidate_memory_range(m_offloader_fault_range, m_offloader_fault_cause);
		m_queue_status.clear(flush_queue_deadlock);
	}
	if (m_queue_status & flush_queue_active) return;
	if (m_flush_requests.pending())
	{
		if (m_flush_queue_mutex.try_lock())
		{
			flush_command_queue();
			m_flush_requests.clear_pending_flag();
			m_flush_requests.consumer_wait();
			m_flush_queue_mutex.unlock();
		}
	}
	else if (!in_begin_end && state != rsx::FIFO::state::lock_wait &&
		(m_graphics_state & rsx::pipeline_state::framebuffer_reads_dirty))
	{
		m_texture_cache.do_update();
		m_graphics_state.clear(rsx::pipeline_state::framebuffer_reads_dirty);
	}
	rsx::thread::do_local_task(state);
}

bool MTLGSRender::on_vram_exhausted(rsx::problem_severity severity)
{
	if (mtl::is_uninterruptible() || !is_current_thread())
		fmt::throw_exception("Metal memory-pressure recovery requires an interruptible RSX thread");
	bool texture_relieved = false;
	if (severity >= rsx::problem_severity::fatal)
	{
		flush_command_queue(true, true);
		if (m_texture_cache.is_overallocated())
		{
			std::set<u32> exclusions;
			for (const auto& texture : rsx::method_registers.fragment_textures)
				exclusions.insert(rsx::get_address(texture.offset(), texture.location()));
			for (const auto& texture : rsx::method_registers.vertex_textures)
				exclusions.insert(rsx::get_address(texture.offset(), texture.location()));
			std::lock_guard lock(m_secondary_command_guard);
			texture_relieved = m_texture_cache.evict_unused(exclusions);
		}
	}
	texture_relieved |= m_texture_cache.handle_memory_pressure(severity);
	if (severity == rsx::problem_severity::low) return texture_relieved;

	bool surface_relieved = false;
	if (severity >= rsx::problem_severity::fatal && m_render_targets.is_overallocated(*m_device))
	{
		std::vector<std::unique_ptr<mtl::viewable_image>> resolve_cache;
		surface_relieved |= m_render_targets.spill_unused_memory(
			*m_current_command_buffer, resolve_cache);
	}
	if (m_render_targets.handle_memory_pressure(*m_current_command_buffer, severity))
	{
		surface_relieved = true;
		m_render_targets.trim(*m_current_command_buffer, severity);
	}
	const mtl::memory_pressure pressure = severity >= rsx::problem_severity::fatal
		? mtl::memory_pressure::critical : mtl::memory_pressure::warning;
	const u64 completed = m_shared_state->completed_submission();
	m_resources.trim(pressure, completed);
	m_overlay_passes.trim(pressure);
	if (surface_relieved)
	{
		std::lock_guard lock(m_sampler_mutex);
		m_samplers_dirty.store(true);
	}
	if (severity >= rsx::problem_severity::fatal) flush_command_queue(true, true);
	return texture_relieved || surface_relieved;
}

void MTLGSRender::notify_tile_unbound(u32)
{
	std::lock_guard lock(m_sampler_mutex);
	m_samplers_dirty.store(true);
}

bool MTLGSRender::scaled_image_from_memory(const rsx::blit_src_info& source,
	const rsx::blit_dst_info& destination, bool interpolate)
{
	if (m_swapchain_unavailable) return false;
	if (!m_texture_cache.blit(source, destination, interpolate,
		m_render_targets, *m_current_command_buffer)) return false;
	m_samplers_dirty.store(true);
	m_current_command_buffer->set_flag(mtl::command_has_blit_transfer);
	if (m_current_command_buffer->has_flag(mtl::command_has_dma_transfer))
		flush_command_queue();
	return true;
}

void MTLGSRender::on_init_thread()
{
	if (!m_device || !*m_device)
		fmt::throw_exception("No Metal 4 device was created");
	GSRender::on_init_thread();
	zcull_ctrl.reset(static_cast<::rsx::reports::ZCULL_control*>(this));
	if (g_cfg.video.shadermode != shader_mode::interpreter_only)
	{
		if (!m_overlay_manager)
		{
			m_frame->hide();
			m_shader_cache->load(nullptr);
			m_frame->show();
		}
		else
		{
			rsx::shader_loading_dialog_native dialog(this);
			m_shader_cache->load(&dialog);
		}
	}
}

void MTLGSRender::on_exit()
{
	GSRender::on_exit();
	if (m_pipeline_compiler) m_pipeline_compiler.wait_idle();
	zcull_ctrl.release();
}

void MTLGSRender::renderctl(u32 request_code, void* arguments)
{
	rsx::thread::renderctl(request_code, arguments);
}
