#pragma once

#include "util/types.hpp"

#include <memory>

namespace rsx::metal
{
	class texture
	{
	public:
		explicit texture(void* texture_handle);
		~texture();

		texture(const texture&) = delete;
		texture& operator=(const texture&) = delete;

		void* handle() const;
		u64 resource_id() const;
		u32 width() const;
		u32 height() const;

	private:
		struct texture_impl;
		std::unique_ptr<texture_impl> m_impl;
	};
}
