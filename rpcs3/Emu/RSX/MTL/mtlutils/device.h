#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../MetalAPI.h"
#include "chip_class.h"

namespace mtl
{
	struct gpu_formats_support
	{
		bool depth24_unorm_stencil8 = false;
		bool depth32_float_stencil8 = false;
		bool bc_texture_compression = false;
		bool bgra8_srgb = false;
		bool bgra10_xr = false;
		bool rgba16_float = false;
		bool rgba32_float = false;
	};

	struct gpu_shader_types_support
	{
		bool float16 = false;
		bool int8 = false;
		bool int16 = false;
		bool int64 = false;
		bool simd_group = false;
		bool quad_group = false;
	};

	struct device_limits
	{
		u64 max_buffer_length = 0;
		u64 max_threadgroup_memory_length = 0;
		u64 recommended_working_set_size = 0;
		u32 max_texture_dimension_1d = 0;
		u32 max_texture_dimension_2d = 0;
		u32 max_texture_dimension_3d = 0;
		u32 max_texture_array_layers = 0;
		u32 max_color_attachments = 0;
		u32 max_threads_per_threadgroup = 0;
		u32 max_argument_buffers_per_stage = 0;
		u32 max_buffers_per_argument_table = 0;
		u32 max_textures_per_argument_table = 0;
		u32 max_samplers_per_argument_table = 0;
		u32 buffer_offset_alignment = 0;
	};

	struct device_features
	{
		bool metal4 = false;
		bool argument_tables = false;
		bool shared_events = false;
		bool residency_sets = false;
		bool placement_heaps = false;
		bool sparse_textures = false;
		bool memoryless_textures = false;
		bool raster_order_groups = false;
		bool framebuffer_fetch = false;
		bool barycentric_coordinates = false;
		bool programmable_sample_positions = false;
		bool counter_sampling = false;
		bool indirect_command_buffers = false;
		bool dynamic_libraries = false;
		bool binary_archives = false;
		bool function_pointers = false;
		bool function_stitching = false;
		bool mesh_shaders = false;
		bool ray_tracing = false;
		bool pull_model_interpolation = false;
		bool float32_filtering = false;
	};

	struct memory_properties
	{
		bool unified = false;
		bool managed_storage = false;
		bool private_storage = true;
		bool memoryless_storage = false;
		u64 recommended_working_set_size = 0;
		u64 current_allocated_size = 0;
	};

	struct device_info
	{
		std::string name;
		u64 registry_id = 0;
		device_identity identity;
		chip_capabilities chip;
		device_features features;
		gpu_formats_support formats;
		gpu_shader_types_support shader_types;
		device_limits limits;
		memory_properties memory;
	};

	class physical_device
	{
		struct impl;
		std::shared_ptr<impl> m_impl;

		explicit physical_device(std::shared_ptr<impl> implementation);
		friend std::vector<physical_device> enumerate_devices();

	public:
		physical_device() = default;

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] device_handle native_handle() const;
		[[nodiscard]] const device_info& info() const;
		[[nodiscard]] const std::string& name() const;
		[[nodiscard]] u64 registry_id() const;
		[[nodiscard]] bool supports_metal4() const;
	};

	class render_device
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		render_device();
		~render_device();
		render_device(const render_device&) = delete;
		render_device& operator=(const render_device&) = delete;
		render_device(render_device&&) noexcept;
		render_device& operator=(render_device&&) noexcept;

		void create(const physical_device& device);
		void destroy();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] const physical_device& gpu() const;
		[[nodiscard]] const device_info& info() const;
		[[nodiscard]] device_handle native_handle() const;
		[[nodiscard]] command_queue_handle graphics_queue() const;
		[[nodiscard]] command_queue_handle transfer_queue() const;
		[[nodiscard]] compiler_handle compiler() const;
	};

	[[nodiscard]] std::vector<physical_device> enumerate_devices();
	[[nodiscard]] physical_device select_device(const std::vector<physical_device>& devices, std::string_view preferred_name);
	[[nodiscard]] const render_device* get_current_renderer();
}
