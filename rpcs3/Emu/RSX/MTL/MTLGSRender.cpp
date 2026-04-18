#include "stdafx.h"
#include "MTLGSRender.h"

#include "Emu/System.h"
#include "Emu/system_config.h"

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
	rsx_log.todo("MTLGSRender::end()");
	fmt::throw_exception("Metal backend draw submission is not implemented yet");
}

void MTLGSRender::clear_surface(u32 arg)
{
	rsx_log.todo("MTLGSRender::clear_surface(arg=0x%x)", arg);
	fmt::throw_exception("Metal backend RSX surface clear is not implemented yet");
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

	if (!m_device || !m_shader_cache || !m_shader_library_cache || !m_pipeline_cache || !m_render_pipeline_cache)
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

	rsx_log.notice("Metal backend state: device=%s, metal4=%u, frames_in_flight=%u, residency_allocations=%u, residency_size=0x%x",
		caps.name,
		static_cast<u32>(caps.metal4_supported),
		caps.frames_in_flight,
		m_device->residency_allocation_count(),
		m_device->residency_allocated_size());
	rsx_log.notice("Metal backend cache state: source_metadata=%u, pipeline_entries=%u, pipeline_states=%u, dynamic_libraries=%u, pipeline_archives=%u",
		cache_stats.shader_entries,
		cache_stats.pipeline_entries,
		cache_stats.pipeline_state_entries,
		cache_stats.library_entries,
		cache_stats.archive_entries);
	rsx_log.notice("Metal backend pipeline entry cache: available=%u, gated=%u, mesh=%u",
		cache_stats.pipeline_entry_available_entries,
		cache_stats.pipeline_entry_gated_entries,
		cache_stats.mesh_pipeline_entry_entries);
	rsx_log.notice("Metal backend compiler/archive state: compiler_ready=%u, serializer_ready=%u, archive_metadata=%u, archive_metadata_invalid=%u, archive_loaded=%u, archive_load_failed=%u, archive_without_metadata=%u",
		static_cast<u32>(compiler_stats.compiler_ready),
		static_cast<u32>(compiler_stats.pipeline_serializer_ready),
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
	rsx_log.notice("Metal backend pipeline cache workflow: pending=%u, flushed=%u, archived=%u, archive_metadata_invalid=%u, successful_flushes=%u, skipped_flushes=%u, serializer_failures=%u, archive_failures=%u",
		pipeline_stats.pending_pipeline_count,
		pipeline_stats.flushed_pipeline_count,
		pipeline_stats.archived_pipeline_count,
		static_cast<u32>(pipeline_stats.archive_metadata_invalid),
		pipeline_stats.successful_flush_count,
		pipeline_stats.skipped_flush_count,
		pipeline_stats.serializer_missing_failures,
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
