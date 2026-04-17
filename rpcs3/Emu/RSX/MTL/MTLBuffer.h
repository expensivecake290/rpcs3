#pragma once

#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
	class device;

	enum class buffer_storage : u8
	{
		shared,
		private_,
	};

	enum class buffer_cpu_cache_mode : u8
	{
		default_,
		write_combined,
	};

	enum class resource_hazard_tracking : u8
	{
		tracked,
		untracked,
	};

	struct buffer_desc
	{
		u64 size = 0;
		std::string label;
		buffer_storage storage = buffer_storage::private_;
		buffer_cpu_cache_mode cpu_cache = buffer_cpu_cache_mode::default_;
		resource_hazard_tracking hazard_tracking = resource_hazard_tracking::tracked;
	};

	class buffer
	{
	public:
		buffer(device& dev, buffer_desc desc);
		~buffer();

		buffer(const buffer&) = delete;
		buffer& operator=(const buffer&) = delete;

		void* handle() const;
		void* allocation_handle() const;
		u64 length() const;
		u64 allocated_size() const;
		u64 gpu_address() const;
		buffer_storage storage() const;
		resource_hazard_tracking hazard_tracking() const;
		void* map();
		void write(u64 offset, const void* data, u64 size);

	private:
		struct buffer_impl;
		std::unique_ptr<buffer_impl> m_impl;
	};
}
