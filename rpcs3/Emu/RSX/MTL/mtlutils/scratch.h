#pragma once

#include <memory>
#include <string>

#include "barriers.h"
#include "buffer_object.h"
#include "image.h"
#include "sampler.h"

namespace mtl
{
	struct scratch_buffer_request
	{
		u64 minimum_size = 0;
		u64 alignment = 256;
		u32 usage = buffer_usage_copy_source | buffer_usage_copy_destination | buffer_usage_storage;
		storage_mode storage = storage_mode::private_;
		queue_kind queue = queue_kind::graphics;
		u64 destination_stages = stage_all_gpu;
		u64 destination_access = access_shader_read | access_shader_write | access_blit_read | access_blit_write;
		std::string label;
		bool zero_initialize = false;
		bool exact_size = false;
	};

	struct scratch_image_request
	{
		image_create_info image;
		u64 compatibility_class = 0;
		queue_kind queue = queue_kind::graphics;
		bool clear_to_zero = false;
		bool exact_size = false;
	};

	struct scratch_buffer_allocation
	{
		buffer* resource = nullptr;
		u64 token = 0;
		u64 size = 0;
		bool newly_created = false;

		[[nodiscard]] explicit operator bool() const
		{
			return resource && token && size;
		}
	};

	struct scratch_image_allocation
	{
		viewable_image* resource = nullptr;
		u64 token = 0;
		bool newly_created = false;

		[[nodiscard]] explicit operator bool() const
		{
			return resource && token;
		}
	};

	struct scratch_pool_statistics
	{
		u64 buffer_bytes = 0;
		u64 image_bytes = 0;
		u64 peak_bytes = 0;
		u64 buffer_count = 0;
		u64 image_count = 0;
		u64 available_count = 0;
		u64 active_count = 0;
		u64 pending_count = 0;
		u64 reuse_count = 0;
		u64 allocation_count = 0;
		u64 eviction_count = 0;
	};

	class scratch_resource_pool
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		scratch_resource_pool();
		~scratch_resource_pool();
		scratch_resource_pool(const scratch_resource_pool&) = delete;
		scratch_resource_pool& operator=(const scratch_resource_pool&) = delete;
		scratch_resource_pool(scratch_resource_pool&&) = delete;
		scratch_resource_pool& operator=(scratch_resource_pool&&) = delete;

		void create(render_device& device, memory_allocator& allocator);
		void destroy();

		[[nodiscard]] scratch_buffer_allocation acquire_buffer(
			command_buffer& command, const scratch_buffer_request& request);
		[[nodiscard]] scratch_image_allocation acquire_image(
			command_buffer& command, const scratch_image_request& request);
		void retire(scratch_buffer_allocation& allocation, u64 submission_value);
		void retire(scratch_image_allocation& allocation, u64 submission_value);
		void reclaim(u64 completed_submission_value);

		[[nodiscard]] const sampler& null_sampler();
		[[nodiscard]] image_view& null_image_view(command_buffer& command, texture_type type);
		[[nodiscard]] image_view& null_image_view(command_buffer& command, texture_type type,
			u64 pixel_format);
		[[nodiscard]] scratch_image_allocation acquire_typeless_helper(command_buffer& command,
			u64 pixel_format, u64 compatibility_class, u32 width, u32 height,
			u32 usage = texture_usage_copy_source | texture_usage_copy_destination);

		void trim(memory_pressure pressure, u64 completed_submission_value);
		void clear_available();
		[[nodiscard]] scratch_pool_statistics statistics() const;
	};
}
