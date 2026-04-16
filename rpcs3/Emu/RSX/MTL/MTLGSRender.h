#pragma once

#include "Emu/RSX/RSXThread.h"

#include "MTLCommandQueue.h"
#include "MTLDevice.h"
#include "MTLNativeWindow.h"
#include "MTLPresentation.h"

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
	u64 m_frame_count = 0;
};
