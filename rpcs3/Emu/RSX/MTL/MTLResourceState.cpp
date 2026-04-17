#include "stdafx.h"
#include "MTLResourceState.h"

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
		u32 m_barrier_count = 0;
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
		rsx_log.trace("rsx::metal::resource_state_tracker::reset(frame_index=%u, tracked_resources=%u, barriers=%u)",
			m_impl->m_frame_index, tracked_resource_count(), m_impl->m_barrier_count);

		m_impl->m_resources.clear();
		m_impl->m_barrier_count = 0;
	}

	resource_barrier resource_state_tracker::record_usage(const resource_usage& usage)
	{
		rsx_log.trace("rsx::metal::resource_state_tracker::record_usage(frame_index=%u, resource_id=0x%x, stage=%s, access=%s, scope=%s)",
			m_impl->m_frame_index,
			usage.resource_id,
			describe_resource_stage(usage.stage),
			describe_resource_access(usage.access),
			describe_resource_barrier_scope(usage.scope));

		if (usage.resource_id == 0)
		{
			fmt::throw_exception("Metal resource usage tracking requires a non-zero GPU resource id");
		}

		const auto found = m_impl->m_resources.find(usage.resource_id);
		if (found == m_impl->m_resources.end())
		{
			m_impl->m_resources.emplace(usage.resource_id, resource_state
			{
				.m_stage = usage.stage,
				.m_access = usage.access,
				.m_scope = usage.scope
			});

			return {};
		}

		resource_barrier barrier
		{
			.resource_id = usage.resource_id,
			.after_stage = found->second.m_stage,
			.before_stage = usage.stage,
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
			m_impl->m_barrier_count++;
			rsx_log.trace("Metal resource barrier required for resource_id=0x%x after=%s before=%s scope=%s",
				barrier.resource_id,
				describe_resource_stage(barrier.after_stage),
				describe_resource_stage(barrier.before_stage),
				describe_resource_barrier_scope(barrier.scope));
		}

		return barrier;
	}

	u32 resource_state_tracker::tracked_resource_count() const
	{
		return static_cast<u32>(m_impl->m_resources.size());
	}

}
