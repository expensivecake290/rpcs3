#pragma once

#include "Emu/RSX/RSXThread.h"

#include "MTLCommandQueue.h"
#include "MTLDevice.h"
#include "MTLNativeWindow.h"
#include "MTLPipelineCache.h"
#include "MTLPipelineState.h"
#include "MTLPresentation.h"
#include "MTLShaderCache.h"
#include "MTLShaderCompiler.h"
#include "MTLShaderLibrary.h"
#include "MTLShaderRecompiler.h"

#include <memory>

class MTLGSRender : public rsx::thread
{
public:
	u64 get_cycles() final;

	MTLGSRender(utils::serial* ar) noexcept;
	MTLGSRender() noexcept : MTLGSRender(nullptr) {}
	~MTLGSRender() override;

private:
	void on_init_thread() override;
	void on_exit() override;
	void end() override;
	void clear_surface(u32 arg) override;
	void flip(const rsx::display_flip_info_t& info) override;
	f64 get_display_refresh_rate() const override;

	void present_clear_frame();

	std::unique_ptr<rsx::metal::device> m_device;
	std::unique_ptr<rsx::metal::native_window> m_window;
	std::unique_ptr<rsx::metal::command_queue> m_queue;
	std::unique_ptr<rsx::metal::presentation_surface> m_presentation;
	std::unique_ptr<rsx::metal::persistent_shader_cache> m_shader_cache;
	std::unique_ptr<rsx::metal::shader_compiler> m_shader_compiler;
	std::unique_ptr<rsx::metal::shader_library_cache> m_shader_library_cache;
	std::unique_ptr<rsx::metal::shader_recompiler> m_shader_recompiler;
	std::unique_ptr<rsx::metal::pipeline_cache> m_pipeline_cache;
	std::unique_ptr<rsx::metal::render_pipeline_cache> m_render_pipeline_cache;
	u64 m_frame_count = 0;
};
