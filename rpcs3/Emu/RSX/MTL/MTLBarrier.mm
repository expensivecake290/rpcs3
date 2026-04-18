#include "stdafx.h"
#include "MTLBarrier.h"

#import <Metal/Metal.h>

namespace rsx::metal
{
	namespace
	{
		MTLStages to_metal_stages(resource_stage stage)
		{
			switch (stage)
			{
			case resource_stage::render:
				return MTLStageVertex | MTLStageFragment | MTLStageTile | MTLStageObject | MTLStageMesh;
			case resource_stage::compute:
				return MTLStageDispatch;
			case resource_stage::blit:
				return MTLStageBlit;
			case resource_stage::present:
				break;
			}

			fmt::throw_exception("Metal presentation stage is not valid for encoder barrier emission");
		}

		MTL4VisibilityOptions to_visibility_options(resource_barrier_scope scope)
		{
			switch (scope)
			{
			case resource_barrier_scope::none:
				return MTL4VisibilityOptionNone;
			case resource_barrier_scope::buffers:
			case resource_barrier_scope::textures:
			case resource_barrier_scope::render_targets:
				return MTL4VisibilityOptionDevice;
			}

			return MTL4VisibilityOptionDevice;
		}
	}

	void encode_consumer_barrier(void* encoder_handle, const resource_barrier& barrier)
	{
		rsx_log.trace("rsx::metal::encode_consumer_barrier(encoder_handle=*0x%x, resource_id=0x%x, required=%d, after=%s/%s, before=%s/%s, scope=%s)",
			encoder_handle,
			barrier.resource_id,
			barrier.required,
			describe_resource_stage(barrier.after_stage),
			describe_resource_access(barrier.after_access),
			describe_resource_stage(barrier.before_stage),
			describe_resource_access(barrier.before_access),
			describe_resource_barrier_scope(barrier.scope));

		if (!barrier.required)
		{
			return;
		}

		if (!encoder_handle)
		{
			fmt::throw_exception("Metal barrier emission requires a valid command encoder");
		}

		if (@available(macOS 26.0, *))
		{
			id<MTL4CommandEncoder> encoder = (__bridge id<MTL4CommandEncoder>)encoder_handle;
			[encoder barrierAfterQueueStages:to_metal_stages(barrier.after_stage)
			                    beforeStages:to_metal_stages(barrier.before_stage)
			               visibilityOptions:to_visibility_options(barrier.scope)];
		}
		else
		{
			fmt::throw_exception("Metal command encoder barriers require macOS 26.0 or newer");
		}
	}
}
