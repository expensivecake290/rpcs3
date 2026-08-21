#include "stdafx.h"
#include "MTLGSRender.h"

#include "upscalers/bilinear_pass.hpp"
#include "upscalers/metalfx_pass.h"
#include "upscalers/nearest_pass.hpp"

#include "Emu/Cell/Modules/cellVideoOut.h"
#include "Emu/RSX/Overlays/overlay_debug_overlay.h"
#include "Emu/RSX/Overlays/overlay_manager.h"
#include "util/video_provider.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <thread>

extern atomic_t<bool> g_user_asked_for_screenshot;
extern atomic_t<recording_mode> g_recording_mode;

namespace
{
	[[nodiscard]] u64 display_format_to_metal(u8 format)
	{
		switch (format)
		{
		default:
			rsx_log.error("Unhandled video output format 0x%x", static_cast<s32>(format));
			[[fallthrough]];
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_X8R8G8B8:
			return MTLPixelFormatBGRA8Unorm;
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_X8B8G8R8:
			return MTLPixelFormatRGBA8Unorm;
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_R16G16B16X16_FLOAT:
			return MTLPixelFormatRGBA16Float;
		}
	}

	[[nodiscard]] mtl::presentation_mode presentation_mode_for(vsync_mode mode)
	{
		switch (mode)
		{
		case vsync_mode::off: return mtl::presentation_mode::immediate;
		case vsync_mode::adaptive: return mtl::presentation_mode::adaptive;
		case vsync_mode::full: return mtl::presentation_mode::synchronized;
		}
		fmt::throw_exception("Invalid video synchronization mode");
	}

	[[nodiscard]] rsx::problem_severity surface_pressure(mtl::memory_pressure pressure)
	{
		switch (pressure)
		{
		case mtl::memory_pressure::normal: return rsx::problem_severity::low;
		case mtl::memory_pressure::warning: return rsx::problem_severity::moderate;
		case mtl::memory_pressure::critical: return rsx::problem_severity::severe;
		}
		fmt::throw_exception("Invalid Metal memory-pressure value");
	}

	[[nodiscard]] mtl::image_create_info drawable_image_information(
		const mtl::drawable_frame& drawable, u64 format)
	{
		return {
			.type = mtl::texture_type::texture_2d,
			.formats = mtl::get_view_compatibility(format),
			.width = drawable.size.width,
			.height = drawable.size.height,
			.depth = 1,
			.mip_levels = 1,
			.array_layers = 1,
			.sample_count = 1,
			.usage = mtl::texture_usage_shader_read | mtl::texture_usage_render_target |
				mtl::texture_usage_copy_source | mtl::texture_usage_copy_destination,
			.aspects = mtl::texture_aspect_color,
			.format_class = rsx::RSX_FORMAT_CLASS_COLOR,
			.storage = mtl::storage_mode::private_,
			.hazards = mtl::hazard_tracking::tracked,
			.pool = mtl::allocation_pool::swapchain,
			.label = "RPCS3 Metal drawable",
			.use_placement_heap = false,
		};
	}

	[[nodiscard]] mtl::image_create_info presentation_image_information(
		u64 format, u32 width, u32 height, std::string label)
	{
		return {
			.type = mtl::texture_type::texture_2d,
			.formats = mtl::get_view_compatibility(format),
			.width = width,
			.height = height,
			.depth = 1,
			.mip_levels = 1,
			.array_layers = 1,
			.sample_count = 1,
			.usage = mtl::texture_usage_shader_read | mtl::texture_usage_render_target |
				mtl::texture_usage_copy_source | mtl::texture_usage_copy_destination |
				mtl::texture_usage_pixel_format_view,
			.aspects = mtl::texture_aspect_color,
			.format_class = rsx::RSX_FORMAT_CLASS_COLOR,
			.storage = mtl::storage_mode::private_,
			.hazards = mtl::hazard_tracking::tracked,
			.pool = mtl::allocation_pool::swapchain,
			.label = std::move(label),
		};
	}

	void clear_drawable(mtl::command_buffer& command, mtl::image& target)
	{
		mtl::transition_image(command, target,
			{mtl::queue_kind::graphics, mtl::stage_fragment | mtl::stage_tile,
				mtl::access_color_write, mtl::get_submission_id(), true});
		if (command.active_encoder() != mtl::encoder_kind::none)
			command.end_encoding();

		MTL4RenderPassDescriptor* descriptor = [MTL4RenderPassDescriptor new];
		descriptor.renderTargetWidth = target.width();
		descriptor.renderTargetHeight = target.height();
		descriptor.defaultRasterSampleCount = 1;
		descriptor.colorAttachments[0].texture = target.native_handle();
		descriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
		descriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
		descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
		command.retain_native_object((__bridge void*)target.native_handle(), true);
		static_cast<void>(command.begin_render_encoding((__bridge void*)descriptor));
		command.end_encoding();
	}

	void copy_or_scale(mtl::command_buffer& command, mtl::image& source, mtl::image& destination,
		const mtl::image_rectangle& source_area, const mtl::image_rectangle& destination_area,
		mtl::image_filter filter)
	{
		mtl::image_conversion conversion;
		const bool compatible = mtl::formats_are_bitcast_compatible(source, destination);
		if (!compatible)
			conversion.kind = mtl::image_conversion_kind::color_to_color;
		mtl::copy_scaled_image(command, source, destination, source_area, destination_area,
			1, compatible, filter, conversion);
	}

	[[nodiscard]] bool is_four_byte_capture_format(u64 format)
	{
		return format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatBGRA8Unorm_sRGB ||
			format == MTLPixelFormatRGBA8Unorm || format == MTLPixelFormatRGBA8Unorm_sRGB;
	}
}

bool MTLGSRender::reinitialize_swapchain()
{
	const s32 client_width = m_frame->client_width();
	const s32 client_height = m_frame->client_height();
	if (client_width <= 0 || client_height <= 0)
	{
		m_swapchain_unavailable = true;
		return false;
	}

	invalidate_render_pass();
	if (m_current_command_buffer && m_current_command_buffer->is_recording())
	{
		const mtl::submission synchronization = close_and_submit_command_buffer({}, {}, true);
		if (!synchronization || !synchronization.succeeded())
		{
			m_swapchain_unavailable = true;
			return false;
		}
	}

	while (!m_queued_frames.empty())
		cleanup_frame_context(m_queued_frames.front());
	for (auto& context : m_frame_context_storage)
	{
		if (context.drawable) m_swapchain->discard(context.drawable);
		context.initialize(context.frame_id ? context.frame_id : (&context - m_frame_context_storage.data()) + 1);
	}

	m_primary_commands.wait_all();
	m_secondary_commands.wait_all();
	if (m_async_scheduler) m_async_scheduler.wait_idle();
	if (m_command_stream) m_command_stream.flush(true);
	reclaim_completed_resources();

	auto configuration = m_swapchain->configuration();
	configuration.width = static_cast<u32>(client_width);
	configuration.height = static_cast<u32>(client_height);
	configuration.mode = presentation_mode_for(g_cfg.video.vsync);
	try
	{
		m_swapchain->reconfigure(configuration);
	}
	catch (const std::exception& error)
	{
		rsx_log.warning("Metal swapchain reconfiguration failed: %s", error.what());
		m_swapchain_unavailable = true;
		return false;
	}

	const mtl::drawable_size dimensions = m_swapchain->size();
	if (!dimensions)
	{
		m_swapchain_unavailable = true;
		return false;
	}
	m_swapchain_dimensions = {dimensions.width, dimensions.height};
	m_overlay_recording_image.reset();
	m_present_temporary_images.clear();
	m_present_temporary_buffers.clear();
	m_upscaler.reset();
	m_current_queue_index = 0;
	m_current_frame = &m_frame_context_storage.front();
	m_current_frame->flags |= frame_context_dirty;
	m_current_command_buffer = m_primary_commands.next();
	m_current_command_buffer->begin();
	m_swapchain_unavailable = false;
	m_should_reinitialize_swapchain = false;
	m_vsync_mode = g_cfg.video.vsync;
	return true;
}

void MTLGSRender::present(mtl::frame_context* context)
{
	if (!context || !context->swap_command_buffer || !context->drawable)
		fmt::throw_exception("Invalid Metal presentation frame context");
	context->swap_command_buffer->flush();
	if (m_swapchain_unavailable)
	{
		m_swapchain->discard(context->drawable);
		return;
	}

	const mtl::drawable_present_status status = m_swapchain->present(context->drawable);
	switch (status)
	{
	case mtl::drawable_present_status::success:
		break;
	case mtl::drawable_present_status::dropped:
		rsx_log.warning("Metal presentation engine dropped a drawable");
		break;
	case mtl::drawable_present_status::resized:
		m_should_reinitialize_swapchain = true;
		break;
	case mtl::drawable_present_status::surface_lost:
		m_swapchain_unavailable = true;
		break;
	case mtl::drawable_present_status::failed:
		rsx_log.error("Metal presentation engine rejected a drawable");
		m_swapchain_unavailable = true;
		break;
	}
}

void MTLGSRender::cleanup_frame_context(mtl::frame_context* context)
{
	if (!context || !context->swap_command_buffer)
		fmt::throw_exception("Invalid Metal frame-context cleanup request");
	if (!context->swap_command_buffer->wait(mtl::frame_present_timeout))
		m_swapchain_unavailable = true;

	if (m_overlay_manager && m_overlay_manager->has_dirty())
	{
		m_overlay_manager->lock_shared();
		std::vector<u32> identifiers;
		identifiers.reserve(m_overlay_manager->get_dirty().size());
		for (const auto& view : m_overlay_manager->get_dirty())
		{
			m_overlay_passes.ui().remove_temporary_resources(view->uid);
			identifiers.push_back(view->uid);
		}
		m_overlay_manager->unlock_shared();
		m_overlay_manager->dispose(identifiers);
	}

	reclaim_completed_resources();
	const u64 completed_submission = m_shared_state->completed_submission();
	if (context->last_frame_sync_time > m_last_heap_sync_time)
	{
		m_last_heap_sync_time = context->last_frame_sync_time;
		mtl::get_data_heap_manager().restore_snapshot(context->heap_snapshot, completed_submission);
	}
	context->swap_command_buffer = nullptr;
	context->reset_heap_ptrs();

	while (!m_queued_frames.empty())
	{
		mtl::frame_context* queued = m_queued_frames.front();
		m_queued_frames.pop_front();
		if (queued == context) break;
	}
	mtl::advance_completed_frame_counter();
}

void MTLGSRender::check_present_status()
{
	while (!m_queued_frames.empty())
	{
		mtl::frame_context* context = m_queued_frames.front();
		if (!context->swap_command_buffer || !context->swap_command_buffer->poke()) break;
		cleanup_frame_context(context);
	}
}

void MTLGSRender::advance_queued_frames()
{
	check_present_status();
	reclaim_completed_resources();
	const mtl::memory_pressure pressure = mtl::determine_memory_pressure(m_allocator->usage());
	m_render_targets.trim(*m_current_command_buffer, surface_pressure(pressure));
	m_texture_cache.on_frame_end();
	m_samplers_dirty.store(true);
	if (m_vertex_cache) m_vertex_cache->purge();
	const u64 completed = m_shared_state->completed_submission();
	m_resources.trim(pressure, completed);
	m_overlay_passes.trim(pressure);
	mtl::get_resource_manager().trim(pressure, mtl::last_completed_resource_event());
	m_allocator->trim(pressure);

	if (!m_current_frame->submission_id)
		fmt::throw_exception("Metal frame has no submission identifier");
	m_current_frame->tag_frame_end(m_current_frame->submission_id);
	m_queued_frames.push_back(m_current_frame);
	if (m_queued_frames.size() > m_maximum_async_frames)
		fmt::throw_exception("Metal queued-frame count exceeded drawable capacity");

	m_current_queue_index = (m_current_queue_index + 1) % m_maximum_async_frames;
	m_current_frame = &m_frame_context_storage[m_current_queue_index];
	m_current_frame->flags |= frame_context_dirty;
	mtl::advance_frame_counter();
	static_cast<void>(m_shared_state->begin_frame());
}

void MTLGSRender::queue_swap_request()
{
	if (!m_current_frame || m_current_frame->swap_command_buffer || !m_current_frame->drawable)
		fmt::throw_exception("Invalid Metal swap request");
	m_current_frame->swap_command_buffer = m_current_command_buffer;
	try
	{
		const mtl::submission submitted = close_and_submit_command_buffer();
		if (!submitted) fmt::throw_exception("Metal presentation submission was empty");
		m_current_frame->submission_id = submitted.value();
		present(m_current_frame);
	}
	catch (...)
	{
		if (m_current_frame->drawable) m_swapchain->discard(m_current_frame->drawable);
		m_current_frame->swap_command_buffer = nullptr;
		throw;
	}
	m_current_command_buffer = m_primary_commands.next();
	m_current_command_buffer->begin();
	advance_queued_frames();
}

mtl::viewable_image* MTLGSRender::get_present_source(
	mtl::present_surface_info* information, const rsx::avconf& configuration)
{
	if (!information || !information->address || !information->width || !information->height ||
		!information->pitch)
	{
		return nullptr;
	}

	mtl::viewable_image* source_image = nullptr;
	const u8 bytes_per_pixel = rsx::get_format_block_size_in_bytes(information->format);
	const auto overlaps = m_render_targets.get_merged_texture_memory_region(*m_current_command_buffer,
		information->address, information->width, information->height, information->pitch,
		bytes_per_pixel, rsx::surface_access::transfer_read);
	if (!overlaps.empty())
	{
		const auto& section = overlaps.back();
		mtl::render_target* surface = section.surface;
		bool viable = false;
		if (surface && section.base_address >= information->address)
		{
			const u32 surface_width = surface->get_surface_width<rsx::surface_metrics::samples>();
			const u32 surface_height = surface->get_surface_height<rsx::surface_metrics::samples>();
			if (section.base_address == information->address)
			{
				viable = surface_width >= information->width && surface_height >= information->height;
			}
			else
			{
				const u32 inset_offset = section.base_address - information->address;
				const u32 inset_y = inset_offset / information->pitch;
				const u32 inset_x = (inset_offset % information->pitch) / bytes_per_pixel;
				viable = surface_width + inset_x * 2 == information->width &&
					surface_height + inset_y * 2 == information->height;
			}
			if (viable)
			{
				source_image = surface->get_surface(rsx::surface_access::transfer_read);
				std::tie(information->width, information->height) = rsx::apply_resolution_scale<true>(
					resolution_scaling_config, std::min(surface_width, information->width),
					std::min(surface_height, information->height));
			}
		}
	}
	else if (auto* texture = m_texture_cache.find_texture_from_dimensions<true>(
		information->address, information->format);
		texture && texture->get_width() >= information->width &&
		texture->get_height() >= information->height)
	{
		source_image = texture->get_raw_texture();
	}

	const u64 expected_format = display_format_to_metal(configuration.format);
	if (!source_image)
	{
		const auto range = utils::address_range32::start_length(information->address,
			information->pitch * information->height);
		const u32 lookup_mask = rsx::texture_upload_context::blit_engine_dst |
			rsx::texture_upload_context::framebuffer_storage;
		for (auto* section : m_texture_cache.find_texture_from_range<true>(range, 0, lookup_mask))
		{
			if (!section->is_synchronized()) section->copy_texture(*m_current_command_buffer, true);
		}
		if (m_current_command_buffer->has_flag(mtl::command_has_dma_transfer))
			flush_command_queue();
		static_cast<void>(m_texture_cache.invalidate_range(*m_current_command_buffer,
			range, rsx::invalidation_cause::read));
		std::unique_ptr<mtl::buffer> staging;
		auto uploaded = m_texture_cache.upload_image_simple_owned(*m_current_command_buffer,
			expected_format, information->address, information->width,
			information->height, information->pitch, staging);
		if (uploaded)
		{
			source_image = uploaded.get();
			m_present_temporary_buffers.emplace_back(std::move(staging));
			m_present_temporary_images.emplace_back(std::move(uploaded));
		}
	}
	else if (source_image->format() != expected_format)
	{
		auto converted = m_texture_cache.create_temporary_subresource_storage(
			rsx::RSX_FORMAT_CLASS_COLOR, expected_format,
			static_cast<u16>(information->width), static_cast<u16>(information->height),
			1, 1, 1, mtl::texture_type::texture_2d, 0,
			mtl::texture_usage_render_target | mtl::texture_usage_shader_read |
				mtl::texture_usage_copy_source);
		if (converted)
		{
			const mtl::image_rectangle rectangle{0, 0,
				static_cast<s32>(information->width), static_cast<s32>(information->height)};
			mtl::copy_image_typeless(*m_current_command_buffer, *source_image, *converted,
				rectangle, rectangle, 1);
			source_image = converted.get();
			m_present_temporary_images.emplace_back(std::move(converted));
		}
	}
	return source_image;
}

void MTLGSRender::flip(const rsx::display_flip_info_t& information)
{
	if (m_vsync_mode != g_cfg.video.vsync)
		m_should_reinitialize_swapchain = true;
	if (m_swapchain && *m_swapchain)
	{
		const mtl::drawable_acquire_status resize_status = m_swapchain->resize();
		if (resize_status == mtl::drawable_acquire_status::surface_lost)
			m_swapchain_unavailable = true;
		else if (resize_status == mtl::drawable_acquire_status::unavailable)
			m_swapchain_unavailable = true;
		else
		{
			const auto dimensions = m_swapchain->size();
			m_swapchain_dimensions = {dimensions.width, dimensions.height};
		}
	}

	if (m_swapchain_unavailable || m_should_reinitialize_swapchain)
	{
		for (u32 attempt = 0; attempt < 10; ++attempt)
		{
			if (reinitialize_swapchain()) break;
			if (Emu.IsStopped())
			{
				m_frame->flip(m_context, information.skip_frame);
				rsx::thread::flip(information);
				return;
			}
			if (m_frame->client_width() <= 0 || m_frame->client_height() <= 0) break;
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

	if (!m_current_frame)
		fmt::throw_exception("Metal presentation has no active frame context");
	if (m_current_frame == &m_auxiliary_frame_context)
	{
		m_current_frame = &m_frame_context_storage[m_current_queue_index];
		if (m_current_frame->swap_command_buffer) cleanup_frame_context(m_current_frame);
		m_current_frame->grab_resources(m_auxiliary_frame_context);
	}
	else if (m_current_frame->swap_command_buffer)
	{
		if (information.stats.draw_calls)
			rsx_log.error("Metal frame context was reused before its presentation completed");
		cleanup_frame_context(m_current_frame);
	}

	if (information.skip_frame || m_swapchain_unavailable)
	{
		if (!information.skip_frame && m_current_command_buffer->is_recording())
		{
			invalidate_render_pass();
			flush_command_queue(true, true);
			mtl::advance_frame_counter();
			static_cast<void>(m_shared_state->begin_frame());
		}
		m_frame->flip(m_context, information.skip_frame);
		rsx::thread::flip(information);
		return;
	}

	m_profiler.start();
	invalidate_render_pass();

	u32 buffer_width = information.buffer < display_buffers_count
		? static_cast<u32>(display_buffers[information.buffer].width) : 0u;
	u32 buffer_height = information.buffer < display_buffers_count
		? static_cast<u32>(display_buffers[information.buffer].height) : 0u;
	u32 buffer_pitch = information.buffer < display_buffers_count
		? static_cast<u32>(display_buffers[information.buffer].pitch) : 0u;
	const rsx::avconf& av_configuration = g_fxo->get<rsx::avconf>();
	if (!buffer_width)
	{
		buffer_width = av_configuration.resolution_x;
		buffer_height = av_configuration.resolution_y;
	}

	u32 source_format = CELL_GCM_TEXTURE_A8R8G8B8;
	if (av_configuration.state)
	{
		source_format = av_configuration.get_compatible_gcm_format();
		if (!buffer_pitch) buffer_pitch = buffer_width * av_configuration.get_bpp();
		const size2u frame_size = av_configuration.video_frame_size();
		buffer_width = std::min(buffer_width, frame_size.width);
		buffer_height = std::min(buffer_height, frame_size.height);
	}
	else if (!buffer_pitch)
	{
		buffer_pitch = buffer_width * 4;
	}

	mtl::viewable_image* image_to_flip = nullptr;
	mtl::viewable_image* right_eye_image = nullptr;
	if (information.buffer < display_buffers_count && buffer_width && buffer_height)
	{
		mtl::present_surface_info surface_information{
			.address = rsx::get_address(display_buffers[information.buffer].offset,
				CELL_GCM_LOCATION_LOCAL),
			.format = source_format,
			.width = buffer_width,
			.height = buffer_height,
			.pitch = buffer_pitch,
			.eye = 0,
		};
		image_to_flip = get_present_source(&surface_information, av_configuration);
		if (image_to_flip && av_configuration.stereo_enabled)
		{
			const auto [ignored_width, minimum_height] = rsx::apply_resolution_scale<true>(
				resolution_scaling_config, RSX_SURFACE_DIMENSION_IGNORED, buffer_height + 30);
			static_cast<void>(ignored_width);
			if (image_to_flip->height() < minimum_height)
			{
				const u32 eye_offset = (buffer_height + 30) * buffer_pitch +
					display_buffers[information.buffer].offset;
				surface_information.address = rsx::get_address(eye_offset, CELL_GCM_LOCATION_LOCAL);
				surface_information.width = buffer_width;
				surface_information.height = buffer_height;
				surface_information.eye = 1;
				right_eye_image = get_present_source(&surface_information, av_configuration);
			}
			else
			{
				const auto [ignored_scaled_width, scaled_height] = rsx::apply_resolution_scale<true>(
					resolution_scaling_config, RSX_SURFACE_DIMENSION_IGNORED, buffer_height);
				static_cast<void>(ignored_scaled_width);
				buffer_height = std::min<u32>(image_to_flip->height() - minimum_height, scaled_height);
			}
		}
		buffer_width = surface_information.width;
		buffer_height = surface_information.height;
	}

	if (information.emu_flip) evaluate_cpu_usage_reduction_limits();
	if (m_current_frame->drawable || m_current_frame->swap_command_buffer)
		fmt::throw_exception("Metal frame context was not empty before drawable acquisition");

	auto render_overlays = [&](mtl::image& target, const areau& content_area)
	{
		if (!m_overlay_manager || !m_overlay_manager->has_visible()) return;
		std::lock_guard lock(*m_overlay_manager);
		const areau display_area{0, 0, target.width(), target.height()};
		const mtl::overlay_render_target overlay_target{&target, 0, 0,
			mtl::texture_aspect_color, true};
		for (const auto& view : m_overlay_manager->get_views())
		{
			const areau area = view->use_window_space ? display_area : content_area;
			m_overlay_passes.ui().run(*m_current_command_buffer, area, overlay_target,
				m_texture_upload_heap, *view);
		}
	};

	const bool overlays_visible = m_overlay_manager && m_overlay_manager->has_visible();
	const bool screenshot_requested = g_user_asked_for_screenshot.exchange(false);
	const bool recording = g_recording_mode != recording_mode::stopped &&
		m_frame->can_consume_frame();
	if (image_to_flip && buffer_width && buffer_height && (screenshot_requested || recording))
	{
		mtl::image* capture_source = image_to_flip;
		if (g_cfg.video.record_with_overlays && overlays_visible)
		{
			const bool recreate = !m_overlay_recording_image ||
				m_overlay_recording_image->format() != image_to_flip->format() ||
				m_overlay_recording_image->width() != buffer_width ||
				m_overlay_recording_image->height() != buffer_height;
			if (recreate)
			{
				m_overlay_recording_image = std::make_unique<mtl::viewable_image>(*m_allocator,
					presentation_image_information(image_to_flip->format(), buffer_width,
						buffer_height, "RPCS3 overlay capture image"));
			}
			const mtl::image_rectangle capture_area{0, 0,
				static_cast<s32>(buffer_width), static_cast<s32>(buffer_height)};
			mtl::copy_image_region(*m_current_command_buffer, *image_to_flip,
				*m_overlay_recording_image, capture_area, capture_area, 1);
			render_overlays(*m_overlay_recording_image,
				{0, 0, buffer_width, buffer_height});
			capture_source = m_overlay_recording_image.get();
		}

		std::unique_ptr<mtl::viewable_image> converted_capture;
		if (!is_four_byte_capture_format(capture_source->format()))
		{
			converted_capture = std::make_unique<mtl::viewable_image>(*m_allocator,
				presentation_image_information(MTLPixelFormatBGRA8Unorm, buffer_width,
					buffer_height, "RPCS3 converted capture image"));
			const mtl::image_rectangle capture_area{0, 0,
				static_cast<s32>(buffer_width), static_cast<s32>(buffer_height)};
			copy_or_scale(*m_current_command_buffer, *capture_source, *converted_capture,
				capture_area, capture_area, mtl::image_filter::nearest);
			capture_source = converted_capture.get();
		}

		const u64 packed_row_bytes = static_cast<u64>(buffer_width) * 4;
		const u64 aligned_row_bytes = (packed_row_bytes + 255) & ~255ull;
		const u64 readback_size = aligned_row_bytes * buffer_height;
		mtl::buffer readback(*m_allocator, {
			.size = readback_size,
			.usage = mtl::buffer_usage_copy_destination,
			.storage = mtl::storage_mode::shared,
			.access = mtl::cpu_access::read,
			.pool = mtl::allocation_pool::swapchain,
			.label = "RPCS3 frame capture readback",
		});
		const mtl::buffer_image_copy_region copy_region{
			.buffer_offset = 0,
			.bytes_per_row = aligned_row_bytes,
			.bytes_per_image = readback_size,
			.subresource = {.aspects = mtl::texture_aspect_color},
			.extent = {buffer_width, buffer_height, 1},
		};
		mtl::copy_image_to_buffer(*m_current_command_buffer, *capture_source, readback,
			std::span{&copy_region, 1});
		flush_command_queue(true, true);

		const auto* mapped = static_cast<const u8*>(readback.map(0, readback_size));
		std::vector<u8> frame(static_cast<usz>(packed_row_bytes * buffer_height));
		for (u32 row = 0; row < buffer_height; ++row)
		{
			std::memcpy(frame.data() + row * packed_row_bytes,
				mapped + row * aligned_row_bytes, packed_row_bytes);
		}
		readback.unmap();
		const bool bgra = capture_source->format() == MTLPixelFormatBGRA8Unorm ||
			capture_source->format() == MTLPixelFormatBGRA8Unorm_sRGB;
		if (screenshot_requested)
			m_frame->take_screenshot(std::move(frame), buffer_width, buffer_height, bgra);
		else
			m_frame->present_frame(std::move(frame), static_cast<u32>(packed_row_bytes),
				buffer_width, buffer_height, bgra);
	}

	for (;;)
	{
		m_current_frame->drawable = m_swapchain->acquire_next_drawable(std::chrono::milliseconds(100));
		switch (m_current_frame->drawable.status)
		{
		case mtl::drawable_acquire_status::success:
			break;
		case mtl::drawable_acquire_status::timeout:
			check_present_status();
			continue;
		case mtl::drawable_acquire_status::resized:
			m_should_reinitialize_swapchain = true;
			continue;
		case mtl::drawable_acquire_status::unavailable:
			m_swapchain_unavailable = true;
			break;
		case mtl::drawable_acquire_status::surface_lost:
			m_swapchain_unavailable = true;
			break;
		}
		break;
	}
	if (!m_current_frame->drawable)
	{
		if (m_current_command_buffer->is_recording())
			flush_command_queue(true, true);
		m_texture_cache.on_frame_end();
		m_samplers_dirty.store(true);
		if (m_vertex_cache) m_vertex_cache->purge();
		m_present_temporary_buffers.clear();
		m_present_temporary_images.clear();
		mtl::advance_frame_counter();
		static_cast<void>(m_shared_state->begin_frame());
		m_frame_stats.flip_time = m_profiler.duration();
		m_frame->flip(m_context);
		rsx::thread::flip(information);
		return;
	}

	m_swapchain_dimensions = {m_current_frame->drawable.size.width,
		m_current_frame->drawable.size.height};
	areai aspect_ratio;
	if (!g_cfg.video.stretch_to_display_area)
	{
		aspect_ratio = static_cast<areai>(av_configuration.aspect_convert_region(
			{buffer_width, buffer_height}, m_swapchain_dimensions));
	}
	else
	{
		aspect_ratio = {0, 0, static_cast<s32>(m_swapchain_dimensions.width),
			static_cast<s32>(m_swapchain_dimensions.height)};
	}

	mtl::viewable_image drawable_image;
	const u64 drawable_format = m_swapchain->configuration().pixel_format;
	drawable_image.wrap(m_current_frame->drawable.texture,
		drawable_image_information(m_current_frame->drawable, drawable_format));
	mtl::overlay_render_target drawable_target{&drawable_image, 0, 0,
		mtl::texture_aspect_color, true};

	if (!image_to_flip || aspect_ratio.x1 || aspect_ratio.y1 ||
		aspect_ratio.x2 != static_cast<s32>(m_swapchain_dimensions.width) ||
		aspect_ratio.y2 != static_cast<s32>(m_swapchain_dimensions.height))
	{
		clear_drawable(*m_current_command_buffer, drawable_image);
	}

	const output_scaling_mode requested_scaling = g_cfg.video.output_scaling.get();
	if (!m_upscaler || m_output_scaling != requested_scaling)
	{
		m_output_scaling = requested_scaling;
		switch (m_output_scaling)
		{
		case output_scaling_mode::nearest:
			m_upscaler = std::make_shared<mtl::nearest_upscale_pass>();
			break;
		case output_scaling_mode::fsr:
			if (mtl::metalfx_upscale_pass::supported(*m_device))
				m_upscaler = std::make_shared<mtl::metalfx_upscale_pass>(*m_device, *m_allocator);
			else
			{
				rsx_log.warning("MetalFX spatial scaling is unavailable; using native linear scaling");
				m_upscaler = std::make_shared<mtl::bilinear_upscale_pass>();
			}
			break;
		case output_scaling_mode::bilinear:
			m_upscaler = std::make_shared<mtl::bilinear_upscale_pass>();
			break;
		}
	}
	if (image_to_flip)
	{
		const mtl::image_rectangle source_area{0, 0, static_cast<s32>(buffer_width),
			static_cast<s32>(buffer_height)};
		const mtl::image_rectangle destination_area{aspect_ratio.x1, aspect_ratio.y1,
			aspect_ratio.x2, aspect_ratio.y2};
		const mtl::upscale_request commit_request{source_area, destination_area};
		const bool calibrate = !g_cfg.video.full_rgb_range_output.get() ||
			!rsx::fcmp(av_configuration.gamma, 1.f) || av_configuration.stereo_enabled;
		if (calibrate)
		{
			std::array<mtl::viewable_image*, 2> source_images{image_to_flip, right_eye_image};
			const u32 calibration_source_count = right_eye_image ? 2u : 1u;
			if (m_output_scaling == output_scaling_mode::fsr &&
				mtl::metalfx_upscale_pass::supported(*m_device))
			{
				const mtl::image_rectangle scaled_area{0, 0,
					static_cast<s32>(aspect_ratio.width()),
					static_cast<s32>(aspect_ratio.height())};
				const mtl::upscale_request scale_request{source_area, scaled_area};
				for (u32 index = 0; index < calibration_source_count; ++index)
				{
					const rsx::flags32_t view = index ? mtl::upscale_right_view :
						mtl::upscale_left_view;
					source_images[index] = m_upscaler->scale_output(*m_current_command_buffer,
						*source_images[index], nullptr, scale_request, view);
				}
			}
			else if (m_output_scaling == output_scaling_mode::nearest)
			{
				for (u32 index = 0; index < calibration_source_count; ++index)
				{
					auto scaled = std::make_unique<mtl::viewable_image>(*m_allocator,
						presentation_image_information(source_images[index]->format(),
							aspect_ratio.width(), aspect_ratio.height(),
							"RPCS3 nearest calibration input"));
					const mtl::image_rectangle scaled_area{0, 0,
						static_cast<s32>(aspect_ratio.width()),
						static_cast<s32>(aspect_ratio.height())};
					copy_or_scale(*m_current_command_buffer, *source_images[index], *scaled,
						source_area, scaled_area, mtl::image_filter::nearest);
					source_images[index] = scaled.get();
					m_present_temporary_images.emplace_back(std::move(scaled));
				}
			}
			const std::span<mtl::viewable_image* const> calibration_sources{
				source_images.data(), calibration_source_count};
			m_overlay_passes.video_calibration().run(*m_current_command_buffer,
				static_cast<areau>(aspect_ratio), drawable_target, calibration_sources,
				av_configuration.gamma, !g_cfg.video.full_rgb_range_output.get(),
				av_configuration.stereo_enabled);
		}
		else
		{
			static_cast<void>(m_upscaler->scale_output(*m_current_command_buffer,
				*image_to_flip, &drawable_image, commit_request,
				mtl::upscale_and_commit | mtl::upscale_default_view));
		}
	}

	if (g_cfg.video.debug_overlay)
	{
		const u32 texture_megabytes = static_cast<u32>(
			m_texture_cache.get_texture_memory_in_use() / 0x100000);
		const u32 temporary_megabytes = static_cast<u32>(
			m_texture_cache.get_temporary_memory_in_use() / 0x100000);
		rsx::overlays::set_debug_overlay_text(fmt::format(
			"Internal Resolution: %s\nRSX Load: %3d%%\nDraw calls: %u\nSubmissions: %u\n"
			"Texture cache: %uM\nTemporary textures: %uM\nFlush requests: %u",
			information.stats.framebuffer_stats.to_string(resolution_scaling_config,
				!backend_config.supports_hw_msaa), get_load(), information.stats.draw_calls,
			information.stats.submit_count, texture_megabytes, temporary_megabytes,
			m_texture_cache.get_num_flush_requests()));
	}
	render_overlays(drawable_image, static_cast<areau>(aspect_ratio));

	queue_swap_request();
	for (auto& temporary : m_present_temporary_buffers)
	{
		const u64 bytes = temporary ? temporary->size() : 0;
		mtl::get_resource_manager().retire(temporary, {
			.resource_class = mtl::managed_resource_class::buffer,
			.bytes = bytes,
			.label = "RPCS3 presentation upload staging",
		});
	}
	m_present_temporary_buffers.clear();
	for (auto& temporary : m_present_temporary_images)
		m_texture_cache.dispose_reusable_image(temporary);
	m_present_temporary_images.clear();
	m_frame_stats.flip_time = m_profiler.duration();
	m_frame->flip(m_context);
	rsx::thread::flip(information);

	const rsx::surface_scaling_config_t active_scaling{
		.scale_percent = static_cast<u16>(g_cfg.video.resolution_scale_percent),
		.min_scalable_dimension = static_cast<u16>(g_cfg.video.min_scalable_dimension),
	};
	if (active_scaling != resolution_scaling_config)
	{
		const mtl::memory_pressure pressure = mtl::determine_memory_pressure(m_allocator->usage());
		if (pressure != mtl::memory_pressure::normal &&
			m_render_targets.handle_memory_pressure(*m_current_command_buffer,
				surface_pressure(pressure)))
		{
			flush_command_queue(true, true);
		}
		m_render_targets.sync_scaling_config(*m_current_command_buffer, active_scaling);
		resolution_scaling_config = active_scaling;
		if (pressure != mtl::memory_pressure::normal &&
			m_render_targets.handle_memory_pressure(*m_current_command_buffer,
				surface_pressure(pressure)))
		{
			flush_command_queue(true, true);
		}
	}
}
