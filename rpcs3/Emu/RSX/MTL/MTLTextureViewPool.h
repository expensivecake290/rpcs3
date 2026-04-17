#pragma once

#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
	class device;
	class texture;

	class texture_view_pool
	{
	public:
		texture_view_pool(device& dev, u32 view_count, std::string label);
		~texture_view_pool();

		texture_view_pool(const texture_view_pool&) = delete;
		texture_view_pool& operator=(const texture_view_pool&) = delete;

		u64 set_default_texture_view(u32 index, const texture& tex);

	private:
		struct texture_view_pool_impl;
		std::unique_ptr<texture_view_pool_impl> m_impl;
	};
}
