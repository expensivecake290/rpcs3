#pragma once

#include "util/types.hpp"

#include <memory>
#include <string_view>

namespace rsx::metal
{
	class native_window
	{
	public:
		native_window(u32 width, u32 height);
		~native_window();

		native_window(const native_window&) = delete;
		native_window& operator=(const native_window&) = delete;

		void show();
		void close();
		void set_title(std::string_view title);
		void update_drawable_size();

		u32 drawable_width() const;
		u32 drawable_height() const;
		f64 refresh_rate() const;
		void* layer_handle() const;

	private:
		struct native_window_impl;
		std::unique_ptr<native_window_impl> m_impl;
	};
}
