#pragma once

#include "MTLResourceState.h"

namespace rsx::metal
{
	void encode_consumer_barrier(void* encoder_handle, const resource_barrier& barrier);
}
