#pragma once

#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
	class device;
	struct heap_resource_usage;

	class texture
	{
	public:
		explicit texture(void* texture_handle);
		texture(device& dev, void* texture_descriptor_handle, std::string label);
		~texture();

		texture(const texture&) = delete;
		texture& operator=(const texture&) = delete;

		void* handle() const;
		void* allocation_handle() const;
		heap_resource_usage heap_resource_usage_info() const;
		u64 resource_id() const;
		u32 width() const;
		u32 height() const;

	private:
		struct texture_impl;
		std::unique_ptr<texture_impl> m_impl;
	};
}
