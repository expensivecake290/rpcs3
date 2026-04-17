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
