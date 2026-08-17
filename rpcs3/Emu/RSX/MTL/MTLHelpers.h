#pragma once

#include <atomic>
#include <mutex>
#include <span>

#include "mtlutils/data_heap.h"
#include "mtlutils/descriptors.h"
#include "mtlutils/graphics_pipeline_state.hpp"
#include "mtlutils/image_helpers.h"
#include "mtlutils/scratch.h"
#include "mtlutils/shared.h"

namespace mtl
{
	enum runtime_state : u64
	{
		runtime_state_none = 0,
		runtime_state_uninterruptible = 1 << 0,
		runtime_state_heap_dirty = 1 << 1,
		runtime_state_heap_changed = 1 << 2,
		runtime_state_device_fault = 1 << 3,
		runtime_state_surface_changed = 1 << 4,
	};

	enum image_setup_flag : u32
	{
		image_setup_none = 0,
		image_setup_initialize_state = 1 << 0,
		image_setup_preserve_state = 1 << 1,
		image_setup_source_gpu_resident = 1 << 2,
		image_setup_source_host_pointer = 1 << 3,
		image_setup_byte_swap = 1 << 4,
	};

	struct byte_range
	{
		u64 offset = 0;
		u64 length = 0;

		[[nodiscard]] explicit operator bool() const { return length != 0; }
	};

	struct image_readback_options
	{
		bool swap_bytes = false;
		u32 element_size = 1;
		byte_range synchronize;
	};

	struct image_rectangle
	{
		s32 x0 = 0;
		s32 y0 = 0;
		s32 x1 = 0;
		s32 y1 = 0;

		[[nodiscard]] u32 width() const;
		[[nodiscard]] u32 height() const;
	};

	struct surface_format_mapping
	{
		u64 pixel_format = 0;
		component_mapping components;
		u64 compatibility_class = 0;
		bool requires_conversion = false;

		[[nodiscard]] explicit operator bool() const { return pixel_format != 0; }
	};

	struct global_resource_statistics
	{
		data_heap_statistics upload_heap;
		scratch_pool_statistics scratch;
		argument_table_cache_statistics argument_tables;
		u64 samplers = 0;
		u64 total_frames = 0;
		u64 completed_frames = 0;
		u64 runtime_flags = 0;
	};

	class renderer_resources
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		renderer_resources();
		~renderer_resources();
		renderer_resources(const renderer_resources&) = delete;
		renderer_resources& operator=(const renderer_resources&) = delete;

		void initialize(shared_state& state);
		void reset(u64 completed_submission_value);
		void destroy();
		void trim(memory_pressure pressure, u64 completed_submission_value);

		[[nodiscard]] data_heap& upload_heap();
		[[nodiscard]] scratch_resource_pool& scratch();
		[[nodiscard]] sampler_pool& samplers();
		[[nodiscard]] argument_table_cache& argument_tables();
		[[nodiscard]] global_resource_statistics statistics() const;
		[[nodiscard]] explicit operator bool() const;
	};

	[[nodiscard]] const render_device* get_current_renderer();
	void set_current_renderer(const render_device& device);
	void clear_current_renderer(const render_device& device);

	[[nodiscard]] bool emulate_primitive_restart(primitive_topology topology);
	[[nodiscard]] bool sanitize_floating_point_values();
	[[nodiscard]] bool emulate_conditional_rendering();
	[[nodiscard]] bool use_strict_query_scopes();

	[[nodiscard]] submission submit_serialized(command_buffer& command, const submit_info& info);

	void upload_image(command_buffer& command, const buffer& source, image& destination,
		std::span<const buffer_image_copy_region> regions, u32 setup_flags = image_setup_initialize_state);
	void copy_image_to_buffer(command_buffer& command, image& source, buffer& destination,
		std::span<const buffer_image_copy_region> regions, const image_readback_options& options = {});
	void copy_buffer_to_image(command_buffer& command, const buffer& source, image& destination,
		std::span<const buffer_image_copy_region> regions);
	[[nodiscard]] u64 calculate_working_buffer_size(u64 base_size, u8 aspects, u32 alignment = 256);

	void copy_image_typeless(command_buffer& command, image& source, image& destination,
		const image_rectangle& source_rectangle, const image_rectangle& destination_rectangle,
		u32 mipmaps, u8 source_aspects = 0xff, u8 destination_aspects = 0xff);
	void copy_image_region(command_buffer& command, image& source, image& destination,
		const image_rectangle& source_rectangle, const image_rectangle& destination_rectangle,
		u32 mipmaps, u8 source_aspects = 0xff, u8 destination_aspects = 0xff);
	void copy_scaled_image(command_buffer& command, image& source, image& destination,
		const image_rectangle& source_rectangle, const image_rectangle& destination_rectangle,
		u32 mipmaps, bool compatible_formats, image_filter filter = image_filter::linear,
		const image_conversion& conversion = {});

	[[nodiscard]] surface_format_mapping compatible_surface_format(u32 color_format);

	void raise_status_interrupt(runtime_state status);
	void clear_status_interrupt(runtime_state status);
	[[nodiscard]] bool test_status_interrupt(runtime_state status);
	void enter_uninterruptible();
	void leave_uninterruptible();
	[[nodiscard]] bool is_uninterruptible();

	void advance_completed_frame_counter();
	void advance_frame_counter();
	[[nodiscard]] u64 current_frame_id();
	[[nodiscard]] u64 last_completed_frame_id();

	class blitter
	{
	public:
		void scale_image(command_buffer& command, image& source, image& destination,
			image_rectangle source_area, image_rectangle destination_area, bool interpolate,
			const image_conversion& conversion = {});
	};
}
