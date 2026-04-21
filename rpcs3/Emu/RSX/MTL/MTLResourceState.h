#pragma once

#include "util/types.hpp"

#include <memory>

namespace rsx::metal
{
	enum class resource_access : u8
	{
		read,
		write
	};

	enum class resource_stage : u8
	{
		render,
		mesh,
		compute,
		blit,
		present
	};

	enum class resource_barrier_scope : u8
	{
		none,
		buffers,
		textures,
		render_targets
	};

	struct resource_usage
	{
		u64 resource_id = 0;
		resource_stage stage = resource_stage::render;
		resource_access access = resource_access::read;
		resource_barrier_scope scope = resource_barrier_scope::textures;
	};

	struct resource_barrier
	{
		u64 resource_id = 0;
		resource_stage after_stage = resource_stage::render;
		resource_stage before_stage = resource_stage::render;
		resource_access after_access = resource_access::read;
		resource_access before_access = resource_access::read;
		resource_barrier_scope scope = resource_barrier_scope::none;
		b8 required = false;
	};

	struct resource_state_stats
	{
		u32 tracked_resources = 0;
		u32 usage_count = 0;
		u32 read_usage_count = 0;
		u32 write_usage_count = 0;
		u32 barrier_count = 0;
		u32 read_after_write_barrier_count = 0;
		u32 write_after_read_barrier_count = 0;
		u32 write_after_write_barrier_count = 0;
		u32 cross_stage_barrier_count = 0;
		u32 present_boundary_count = 0;
	};

	class resource_state_tracker
	{
	public:
		explicit resource_state_tracker(u32 frame_index);
		~resource_state_tracker();

		resource_state_tracker(const resource_state_tracker&) = delete;
		resource_state_tracker& operator=(const resource_state_tracker&) = delete;

		void reset();
		resource_barrier record_usage(const resource_usage& usage);
		void record_present_boundary(u64 resource_id);
		resource_state_stats stats() const;

	private:
		u32 tracked_resource_count() const;

		struct resource_state_tracker_impl;
		std::unique_ptr<resource_state_tracker_impl> m_impl;
	};

	const char* describe_resource_access(resource_access access);
	const char* describe_resource_stage(resource_stage stage);
	const char* describe_resource_barrier_scope(resource_barrier_scope scope);
}
