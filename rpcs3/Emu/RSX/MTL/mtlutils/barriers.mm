#include "stdafx.h"
#include "barriers.h"

#import <Metal/Metal.h>

#include <limits>

namespace mtl
{
	namespace
	{
		constexpr u64 write_access_mask = access_shader_write | access_color_write |
			access_depth_stencil_write | access_blit_write | access_host_write;
		constexpr u64 host_access_mask = access_host_read | access_host_write;

		u64 encoder_stage_mask(encoder_kind encoder)
		{
			switch (encoder)
			{
			case encoder_kind::render:
				return stage_vertex | stage_fragment | stage_tile | stage_object | stage_mesh;
			case encoder_kind::compute:
				return stage_dispatch | stage_blit | stage_acceleration_structure | stage_machine_learning;
			case encoder_kind::none:
				return stage_none;
			}
			fmt::throw_exception("Invalid Metal encoder kind %u", static_cast<u8>(encoder));
		}

		void validate_hazard(const hazard& value)
		{
			if ((value.producer_stages & ~(stage_all_gpu | stage_host)) ||
				(value.consumer_stages & ~(stage_all_gpu | stage_host)))
			{
				fmt::throw_exception("Metal hazard contains unknown pipeline stages");
			}

			if (value.producer_stages == stage_none || value.consumer_stages == stage_none)
			{
				fmt::throw_exception("Metal hazard requires nonempty producer and consumer stages");
			}

			switch (value.resource)
			{
			case resource_kind::global:
				break;
			case resource_kind::buffer:
				if (!value.identity || value.length == 0 || value.offset > std::numeric_limits<u64>::max() - value.length)
				{
					fmt::throw_exception("Invalid Metal buffer hazard range");
				}
				break;
			case resource_kind::texture:
				if (!value.identity || value.subresources.mip_count == 0 || value.subresources.slice_count == 0 ||
					(!value.subresources.color && !value.subresources.depth && !value.subresources.stencil))
				{
					fmt::throw_exception("Invalid Metal texture hazard range");
				}
				break;
			}
		}

		MTLStages native_stages(u64 stages)
		{
			const u64 gpu_stages = stages & stage_all_gpu;
			return static_cast<MTLStages>(gpu_stages ? gpu_stages : stage_all_gpu);
		}
	}

	bool has_write_access(u64 access)
	{
		return (access & write_access_mask) != 0;
	}

	bool has_host_access(u64 access)
	{
		return (access & host_access_mask) != 0;
	}

	barrier_plan classify_hazard(const hazard& value, encoder_kind active_encoder)
	{
		validate_hazard(value);
		barrier_plan result;
		result.after_stages = value.producer_stages & stage_all_gpu;
		result.before_stages = value.consumer_stages & stage_all_gpu;

		const bool producer_host = (value.producer_stages & stage_host) || has_host_access(value.producer_access);
		const bool consumer_host = (value.consumer_stages & stage_host) || has_host_access(value.consumer_access);
		const bool memory_dependency = has_write_access(value.producer_access) || has_write_access(value.consumer_access);
		result.flush_caches = memory_dependency;

		if (producer_host && consumer_host)
		{
			return result;
		}

		if (producer_host)
		{
			result.scope = barrier_scope::cpu_to_gpu;
			result.end_encoder = active_encoder != encoder_kind::none;
			result.synchronize_managed_resource = value.cpu_visible;
			return result;
		}

		if (consumer_host)
		{
			result.scope = barrier_scope::gpu_to_cpu;
			result.end_encoder = active_encoder != encoder_kind::none;
			result.require_event = true;
			result.synchronize_managed_resource = value.cpu_visible;
			return result;
		}

		if (value.producer_queue != value.consumer_queue)
		{
			result.scope = barrier_scope::between_queues;
			result.end_encoder = active_encoder != encoder_kind::none;
			result.require_event = true;
			return result;
		}

		if (!memory_dependency)
		{
			return result;
		}

		const u64 supported_stages = encoder_stage_mask(active_encoder);
		const bool stages_fit_encoder = active_encoder != encoder_kind::none &&
			((result.after_stages | result.before_stages) & ~supported_stages) == 0;
		if (value.preserve_encoder && stages_fit_encoder)
		{
			result.scope = barrier_scope::within_encoder;
			return result;
		}

		result.scope = barrier_scope::between_encoders;
		result.end_encoder = active_encoder != encoder_kind::none;
		result.producer_barrier = result.end_encoder;
		return result;
	}

	void encode_barrier(native_encoder_handle encoder, const barrier_plan& plan)
	{
		if (!plan)
		{
			return;
		}
		if (!encoder)
		{
			fmt::throw_exception("Cannot encode a Metal barrier without an active encoder");
		}

		id<MTL4CommandEncoder> native_encoder = (__bridge id<MTL4CommandEncoder>)encoder;
		const MTL4VisibilityOptions visibility = plan.flush_caches ? MTL4VisibilityOptionDevice : MTL4VisibilityOptionNone;
		switch (plan.scope)
		{
		case barrier_scope::within_encoder:
			[native_encoder barrierAfterEncoderStages:native_stages(plan.after_stages)
				beforeEncoderStages:native_stages(plan.before_stages)
				visibilityOptions:visibility];
			return;
		case barrier_scope::between_encoders:
			if (plan.producer_barrier)
			{
				[native_encoder barrierAfterStages:native_stages(plan.after_stages)
					beforeQueueStages:native_stages(plan.before_stages)
					visibilityOptions:visibility];
			}
			else
			{
				[native_encoder barrierAfterQueueStages:native_stages(plan.after_stages)
					beforeStages:native_stages(plan.before_stages)
					visibilityOptions:visibility];
			}
			return;
		case barrier_scope::none:
			return;
		case barrier_scope::between_queues:
		case barrier_scope::gpu_to_cpu:
		case barrier_scope::cpu_to_gpu:
			fmt::throw_exception("Metal queue/CPU hazard requires its planned event or resource synchronization, not an encoder barrier");
		}

		fmt::throw_exception("Invalid Metal barrier scope %u", static_cast<u8>(plan.scope));
	}
}
