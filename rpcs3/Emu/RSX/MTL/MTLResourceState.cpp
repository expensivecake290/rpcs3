#include "stdafx.h"
#include "MTLResourceState.h"

#include <limits>
#include <unordered_map>

namespace rsx::metal
{
	namespace
	{
		struct resource_state
		{
			resource_stage m_stage = resource_stage::render;
			resource_access m_access = resource_access::read;
			resource_barrier_scope m_scope = resource_barrier_scope::textures;
		};

		b8 requires_barrier(resource_access previous_access, resource_access next_access)
		{
			return previous_access == resource_access::write || next_access == resource_access::write;
		}

		void increment_counter(u32& counter, const char* counter_name)
		{
			if (counter == std::numeric_limits<u32>::max())
			{
				fmt::throw_exception("Metal resource state %s counter overflow", counter_name);
			}

			counter++;
		}

		b8 is_valid_resource_access(resource_access access)
		{
			switch (access)
			{
			case resource_access::read:
			case resource_access::write:
				return true;
			}

			return false;
		}

		b8 is_valid_resource_stage(resource_stage stage)
		{
			switch (stage)
			{
			case resource_stage::render:
			case resource_stage::mesh:
			case resource_stage::compute:
			case resource_stage::blit:
			case resource_stage::present:
				return true;
			}

			return false;
		}

		b8 is_valid_resource_barrier_scope(resource_barrier_scope scope)
		{
			switch (scope)
			{
			case resource_barrier_scope::none:
			case resource_barrier_scope::buffers:
			case resource_barrier_scope::textures:
			case resource_barrier_scope::render_targets:
				return true;
			}

			return false;
		}

		void validate_resource_usage(const resource_usage& usage)
		{
			if (usage.resource_id == 0)
			{
				fmt::throw_exception("Metal resource usage tracking requires a non-zero GPU resource id");
			}

			if (!is_valid_resource_stage(usage.stage))
			{
				fmt::throw_exception("Metal resource usage tracking received an invalid stage: %u",
					static_cast<u32>(usage.stage));
			}

			if (!is_valid_resource_access(usage.access))
			{
				fmt::throw_exception("Metal resource usage tracking received an invalid access mode: %u",
					static_cast<u32>(usage.access));
			}

			if (!is_valid_resource_barrier_scope(usage.scope))
			{
				fmt::throw_exception("Metal resource usage tracking received an invalid barrier scope: %u",
					static_cast<u32>(usage.scope));
			}

			if (usage.stage == resource_stage::present)
			{
				fmt::throw_exception("Metal resource usage tracking cannot record presentation directly; use a present boundary");
			}

			if (usage.scope == resource_barrier_scope::none)
			{
				fmt::throw_exception("Metal resource usage tracking requires a concrete barrier scope");
			}
		}

		resource_barrier_scope merge_barrier_scope(resource_barrier_scope previous_scope, resource_barrier_scope next_scope)
		{
			if (previous_scope == next_scope || previous_scope == resource_barrier_scope::none)
			{
				return next_scope;
			}

			if (next_scope == resource_barrier_scope::none)
			{
				return previous_scope;
			}

			if (previous_scope == resource_barrier_scope::render_targets || next_scope == resource_barrier_scope::render_targets)
			{
				return resource_barrier_scope::render_targets;
			}

			if (previous_scope == resource_barrier_scope::textures || next_scope == resource_barrier_scope::textures)
			{
				return resource_barrier_scope::textures;
			}

			return resource_barrier_scope::buffers;
		}
	}

	struct resource_state_tracker::resource_state_tracker_impl
	{
		std::unordered_map<u64, resource_state> m_resources;
		u32 m_frame_index = 0;
		u32 m_usage_count = 0;
		u32 m_read_usage_count = 0;
		u32 m_write_usage_count = 0;
		u32 m_barrier_count = 0;
		u32 m_read_after_write_barrier_count = 0;
		u32 m_write_after_read_barrier_count = 0;
		u32 m_write_after_write_barrier_count = 0;
		u32 m_cross_stage_barrier_count = 0;
		u32 m_present_boundary_count = 0;
	};

	const char* describe_resource_access(resource_access access)
	{
		switch (access)
		{
		case resource_access::read:
			return "read";
		case resource_access::write:
			return "write";
		}

		return "unknown";
	}

	const char* describe_resource_stage(resource_stage stage)
	{
		switch (stage)
		{
		case resource_stage::render:
			return "render";
		case resource_stage::mesh:
			return "mesh";
		case resource_stage::compute:
			return "compute";
		case resource_stage::blit:
			return "blit";
		case resource_stage::present:
			return "present";
		}

		return "unknown";
	}

	const char* describe_resource_barrier_scope(resource_barrier_scope scope)
	{
		switch (scope)
		{
		case resource_barrier_scope::none:
			return "none";
		case resource_barrier_scope::buffers:
			return "buffers";
		case resource_barrier_scope::textures:
			return "textures";
		case resource_barrier_scope::render_targets:
			return "render_targets";
		}

		return "unknown";
	}

	resource_state_tracker::resource_state_tracker(u32 frame_index)
		: m_impl(std::make_unique<resource_state_tracker_impl>())
	{
		rsx_log.notice("rsx::metal::resource_state_tracker::resource_state_tracker(frame_index=%u)", frame_index);

		m_impl->m_resources.reserve(32);
		m_impl->m_frame_index = frame_index;
	}

	resource_state_tracker::~resource_state_tracker()
	{
		rsx_log.notice("rsx::metal::resource_state_tracker::~resource_state_tracker(frame_index=%u)",
			m_impl ? m_impl->m_frame_index : 0);
	}

	void resource_state_tracker::reset()
	{
		const resource_state_stats previous_stats = stats();
		rsx_log.trace("rsx::metal::resource_state_tracker::reset(frame_index=%u, tracked_resources=%u, usages=%u, reads=%u, writes=%u, barriers=%u, raw=%u, war=%u, waw=%u, cross_stage=%u, present_boundaries=%u)",
			m_impl->m_frame_index,
			previous_stats.tracked_resources,
			previous_stats.usage_count,
			previous_stats.read_usage_count,
			previous_stats.write_usage_count,
			previous_stats.barrier_count,
			previous_stats.read_after_write_barrier_count,
			previous_stats.write_after_read_barrier_count,
			previous_stats.write_after_write_barrier_count,
			previous_stats.cross_stage_barrier_count,
			previous_stats.present_boundary_count);

		m_impl->m_resources.clear();
		m_impl->m_usage_count = 0;
		m_impl->m_read_usage_count = 0;
		m_impl->m_write_usage_count = 0;
		m_impl->m_barrier_count = 0;
		m_impl->m_read_after_write_barrier_count = 0;
		m_impl->m_write_after_read_barrier_count = 0;
		m_impl->m_write_after_write_barrier_count = 0;
		m_impl->m_cross_stage_barrier_count = 0;
		m_impl->m_present_boundary_count = 0;
	}

	resource_barrier resource_state_tracker::record_usage(const resource_usage& usage)
	{
		rsx_log.trace("rsx::metal::resource_state_tracker::record_usage(frame_index=%u, resource_id=0x%x, stage=%s, access=%s, scope=%s)",
			m_impl->m_frame_index,
			usage.resource_id,
			describe_resource_stage(usage.stage),
			describe_resource_access(usage.access),
			describe_resource_barrier_scope(usage.scope));

		validate_resource_usage(usage);

		increment_counter(m_impl->m_usage_count, "usage");
		if (usage.access == resource_access::write)
		{
			increment_counter(m_impl->m_write_usage_count, "write usage");
		}
		else
		{
			increment_counter(m_impl->m_read_usage_count, "read usage");
		}

		const auto found = m_impl->m_resources.find(usage.resource_id);
		if (found == m_impl->m_resources.end())
		{
			if (m_impl->m_resources.size() == std::numeric_limits<u32>::max())
			{
				fmt::throw_exception("Metal resource state tracked resource counter overflow");
			}

			m_impl->m_resources.emplace(usage.resource_id, resource_state
			{
				.m_stage = usage.stage,
				.m_access = usage.access,
				.m_scope = usage.scope
			});

			return {};
		}

		if (found->second.m_stage == resource_stage::present)
		{
			fmt::throw_exception("Metal resource_id=0x%x was reused after a present boundary in the same frame", usage.resource_id);
		}

		resource_barrier barrier
		{
			.resource_id = usage.resource_id,
			.after_stage = found->second.m_stage,
			.before_stage = usage.stage,
			.after_access = found->second.m_access,
			.before_access = usage.access,
			.scope = merge_barrier_scope(found->second.m_scope, usage.scope),
			.required = requires_barrier(found->second.m_access, usage.access)
		};

		found->second = resource_state
		{
			.m_stage = usage.stage,
			.m_access = usage.access,
			.m_scope = usage.scope
		};

		if (barrier.required)
		{
			increment_counter(m_impl->m_barrier_count, "barrier");

			if (barrier.after_access == resource_access::write && barrier.before_access == resource_access::read)
			{
				increment_counter(m_impl->m_read_after_write_barrier_count, "read-after-write barrier");
			}
			else if (barrier.after_access == resource_access::read && barrier.before_access == resource_access::write)
			{
				increment_counter(m_impl->m_write_after_read_barrier_count, "write-after-read barrier");
			}
			else if (barrier.after_access == resource_access::write && barrier.before_access == resource_access::write)
			{
				increment_counter(m_impl->m_write_after_write_barrier_count, "write-after-write barrier");
			}

			if (barrier.after_stage != barrier.before_stage)
			{
				increment_counter(m_impl->m_cross_stage_barrier_count, "cross-stage barrier");
			}

			rsx_log.trace("Metal resource barrier required for resource_id=0x%x after=%s/%s before=%s/%s scope=%s",
				barrier.resource_id,
				describe_resource_stage(barrier.after_stage),
				describe_resource_access(barrier.after_access),
				describe_resource_stage(barrier.before_stage),
				describe_resource_access(barrier.before_access),
				describe_resource_barrier_scope(barrier.scope));
		}

		return barrier;
	}

	void resource_state_tracker::record_present_boundary(u64 resource_id)
	{
		rsx_log.trace("rsx::metal::resource_state_tracker::record_present_boundary(frame_index=%u, resource_id=0x%x)",
			m_impl->m_frame_index,
			resource_id);

		if (resource_id == 0)
		{
			fmt::throw_exception("Metal present boundary tracking requires a non-zero GPU resource id");
		}

		const auto found = m_impl->m_resources.find(resource_id);
		if (found == m_impl->m_resources.end())
		{
			fmt::throw_exception("Metal present boundary for resource_id=0x%x was recorded before render target usage", resource_id);
		}

		if (found->second.m_stage == resource_stage::present)
		{
			rsx_log.trace("Metal present boundary for resource_id=0x%x was already recorded", resource_id);
			return;
		}

		if (found->second.m_access != resource_access::write)
		{
			fmt::throw_exception("Metal present boundary for resource_id=0x%x follows %s/%s usage instead of a GPU write",
				resource_id,
				describe_resource_stage(found->second.m_stage),
				describe_resource_access(found->second.m_access));
		}

		increment_counter(m_impl->m_present_boundary_count, "present boundary");
		rsx_log.trace("Metal present boundary tracked for resource_id=0x%x after=%s/%s before=present/read",
			resource_id,
			describe_resource_stage(found->second.m_stage),
			describe_resource_access(found->second.m_access));

		found->second = resource_state
		{
			.m_stage = resource_stage::present,
			.m_access = resource_access::read,
			.m_scope = resource_barrier_scope::render_targets
		};
	}

	resource_state_stats resource_state_tracker::stats() const
	{
		rsx_log.trace("rsx::metal::resource_state_tracker::stats(frame_index=%u)", m_impl->m_frame_index);

		return resource_state_stats
		{
			.tracked_resources = tracked_resource_count(),
			.usage_count = m_impl->m_usage_count,
			.read_usage_count = m_impl->m_read_usage_count,
			.write_usage_count = m_impl->m_write_usage_count,
			.barrier_count = m_impl->m_barrier_count,
			.read_after_write_barrier_count = m_impl->m_read_after_write_barrier_count,
			.write_after_read_barrier_count = m_impl->m_write_after_read_barrier_count,
			.write_after_write_barrier_count = m_impl->m_write_after_write_barrier_count,
			.cross_stage_barrier_count = m_impl->m_cross_stage_barrier_count,
			.present_boundary_count = m_impl->m_present_boundary_count,
		};
	}

	u32 resource_state_tracker::tracked_resource_count() const
	{
		const auto resource_count = m_impl->m_resources.size();
		if (resource_count > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal resource state tracked resource count exceeds u32 range");
		}

		return static_cast<u32>(resource_count);
	}

}
