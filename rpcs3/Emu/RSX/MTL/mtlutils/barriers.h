#pragma once

#include "commands.h"
#include "ex.h"

namespace mtl
{
	enum pipeline_stage : u64
	{
		stage_none = 0,
		stage_vertex = 1ull << 0,
		stage_fragment = 1ull << 1,
		stage_tile = 1ull << 2,
		stage_object = 1ull << 3,
		stage_mesh = 1ull << 4,
		stage_resource_state = 1ull << 26,
		stage_dispatch = 1ull << 27,
		stage_blit = 1ull << 28,
		stage_acceleration_structure = 1ull << 29,
		stage_machine_learning = 1ull << 30,
		stage_host = 1ull << 62,
		stage_all_gpu = stage_vertex | stage_fragment | stage_tile | stage_object | stage_mesh |
			stage_resource_state | stage_dispatch | stage_blit | stage_acceleration_structure | stage_machine_learning,
	};

	enum resource_access_mask : u64
	{
		access_none = 0,
		access_vertex_read = 1ull << 0,
		access_index_read = 1ull << 1,
		access_indirect_read = 1ull << 2,
		access_constant_read = 1ull << 3,
		access_shader_read = 1ull << 4,
		access_shader_write = 1ull << 5,
		access_color_read = 1ull << 6,
		access_color_write = 1ull << 7,
		access_depth_stencil_read = 1ull << 8,
		access_depth_stencil_write = 1ull << 9,
		access_blit_read = 1ull << 10,
		access_blit_write = 1ull << 11,
		access_host_read = 1ull << 12,
		access_host_write = 1ull << 13,
	};

	enum class resource_kind : u8
	{
		global,
		buffer,
		texture,
	};

	enum class barrier_scope : u8
	{
		none,
		within_encoder,
		between_encoders,
		between_queues,
		gpu_to_cpu,
		cpu_to_gpu,
	};

	struct subresource_range
	{
		u32 first_mip = 0;
		u32 mip_count = 1;
		u32 first_slice = 0;
		u32 slice_count = 1;
		bool color = true;
		bool depth = false;
		bool stencil = false;

		[[nodiscard]] bool operator==(const subresource_range&) const = default;
	};

	struct hazard
	{
		resource_kind resource = resource_kind::global;
		resource_identity identity;
		u64 offset = 0;
		u64 length = 0;
		subresource_range subresources;
		u64 producer_stages = stage_all_gpu;
		u64 consumer_stages = stage_all_gpu;
		u64 producer_access = access_shader_write;
		u64 consumer_access = access_shader_read;
		queue_kind producer_queue = queue_kind::graphics;
		queue_kind consumer_queue = queue_kind::graphics;
		bool cpu_visible = false;
		bool preserve_encoder = false;
	};

	struct barrier_plan
	{
		barrier_scope scope = barrier_scope::none;
		u64 after_stages = stage_none;
		u64 before_stages = stage_none;
		bool flush_caches = false;
		bool end_encoder = false;
		bool producer_barrier = false;
		bool require_event = false;
		bool synchronize_managed_resource = false;

		[[nodiscard]] explicit operator bool() const
		{
			return scope != barrier_scope::none;
		}
	};

	[[nodiscard]] bool has_write_access(u64 access);
	[[nodiscard]] bool has_host_access(u64 access);
	[[nodiscard]] barrier_plan classify_hazard(const hazard& value, encoder_kind active_encoder);
	void encode_barrier(native_encoder_handle encoder, const barrier_plan& plan);
}
