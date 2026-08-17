#include "stdafx.h"
#include "MetalAPI.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>

namespace mtl
{
	namespace
	{
		std::mutex s_runtime_mutex;
		runtime_info s_runtime_info;
		id<MTLDevice> s_device;

		error make_error(error_code code, NSString* description, NSError* native_error = nil)
		{
			error result;
			result.code = code;
			result.native_code = native_error ? native_error.code : 0;
			result.domain = native_error.domain ? native_error.domain.UTF8String : "Metal";

			NSString* text = native_error.localizedDescription ?: description;
			result.description = text ? text.UTF8String : "Unknown Metal error";
			return result;
		}

		void validate_metal4_device(id<MTLDevice> device)
		{
			NSError* native_error = nil;
			MTL4CommandQueueDescriptor* queue_descriptor = [MTL4CommandQueueDescriptor new];
			queue_descriptor.label = @"RPCS3 Metal 4 validation queue";
			id<MTL4CommandQueue> queue = [device newMTL4CommandQueueWithDescriptor:queue_descriptor error:&native_error];

			if (!queue)
			{
				throw_error(make_error(error_code::metal4_unavailable, @"The selected device cannot create a Metal 4 command queue.", native_error), "Metal 4 validation");
			}

			native_error = nil;
			MTL4CompilerDescriptor* compiler_descriptor = [MTL4CompilerDescriptor new];
			compiler_descriptor.label = @"RPCS3 Metal 4 compiler";
			id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:compiler_descriptor error:&native_error];

			if (!compiler)
			{
				throw_error(make_error(error_code::metal4_unavailable, @"The selected device cannot create a Metal 4 compiler.", native_error), "Metal 4 validation");
			}
		}

		runtime_capabilities query_runtime_capabilities(id<MTLDevice> device)
		{
			runtime_capabilities result;
			result.metal4 = true;
			result.unified_memory = device.hasUnifiedMemory;
			result.low_power = device.lowPower;
			result.removable = device.removable;
			result.argument_tables = true;
			result.compiler = true;
			result.shared_events = [device respondsToSelector:@selector(newSharedEvent)];
			result.residency_sets = [device respondsToSelector:@selector(newResidencySetWithDescriptor:error:)];
			result.placement_heaps = [device respondsToSelector:@selector(newHeapWithDescriptor:)];
			result.sparse_textures = device.sparseTileSizeInBytes != 0;
			result.ray_tracing = device.supportsRaytracing;
			result.dynamic_libraries = device.supportsDynamicLibraries;
			result.counter_sampling = [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
			result.bc_texture_compression = device.supportsBCTextureCompression;
			result.function_pointers = device.supportsFunctionPointers;
			result.recommended_working_set_size = device.recommendedMaxWorkingSetSize;
			result.max_buffer_length = device.maxBufferLength;
			result.max_threads_per_threadgroup = static_cast<u32>(device.maxThreadsPerThreadgroup.width);
			return result;
		}
	}

	bool is_runtime_available()
	{
		if (@available(macOS 26.0, *))
		{
			id<MTLDevice> device = MTLCreateSystemDefaultDevice();
			return device &&
				[device respondsToSelector:@selector(newMTL4CommandQueueWithDescriptor:error:)] &&
				[device respondsToSelector:@selector(newCompilerWithDescriptor:error:)];
		}

		return false;
	}

	runtime_info initialize()
	{
		std::lock_guard lock(s_runtime_mutex);

		if (s_device)
		{
			return s_runtime_info;
		}

		if (@available(macOS 26.0, *))
		{
			s_device = MTLCreateSystemDefaultDevice();

			if (!s_device)
			{
				throw_error(make_error(error_code::device_unavailable, @"Metal did not return a system default device."), "Metal initialization");
			}

			if (![s_device respondsToSelector:@selector(newMTL4CommandQueueWithDescriptor:error:)] ||
				![s_device respondsToSelector:@selector(newCompilerWithDescriptor:error:)])
			{
				s_device = nil;
				throw_error(make_error(error_code::metal4_unavailable, @"The selected device does not expose the required Metal 4 API."), "Metal initialization");
			}

			validate_metal4_device(s_device);
			s_runtime_info.device = s_device;
			s_runtime_info.device_name = s_device.name.UTF8String ?: "Unknown Metal device";
			s_runtime_info.capabilities = query_runtime_capabilities(s_device);
			return s_runtime_info;
		}

		throw_error(make_error(error_code::unsupported_os, @"Metal 4 requires macOS 26.0 or newer."), "Metal initialization");
	}

	const runtime_info& get_runtime_info()
	{
		std::lock_guard lock(s_runtime_mutex);

		if (!s_device)
		{
			fmt::throw_exception("Metal runtime information requested before initialization");
		}

		return s_runtime_info;
	}

	void shutdown()
	{
		std::lock_guard lock(s_runtime_mutex);
		s_runtime_info = {};
		s_device = nil;
	}

	const char* get_error_name(error_code code)
	{
		switch (code)
		{
		case error_code::none: return "none";
		case error_code::unsupported_platform: return "unsupported platform";
		case error_code::unsupported_os: return "unsupported operating system";
		case error_code::framework_unavailable: return "Metal framework unavailable";
		case error_code::device_unavailable: return "Metal device unavailable";
		case error_code::metal4_unavailable: return "Metal 4 unavailable";
		case error_code::resource_creation_failed: return "resource creation failed";
		case error_code::shader_compilation_failed: return "shader compilation failed";
		case error_code::pipeline_creation_failed: return "pipeline creation failed";
		case error_code::command_submission_failed: return "command submission failed";
		case error_code::device_lost: return "device lost";
		case error_code::out_of_memory: return "out of memory";
		case error_code::invalid_operation: return "invalid operation";
		case error_code::internal_error: return "internal error";
		}

		fmt::throw_exception("Unknown Metal error code %u", static_cast<u8>(code));
	}

	std::string format_error(const error& value)
	{
		if (!value)
		{
			return get_error_name(error_code::none);
		}

		if (value.domain.empty())
		{
			return fmt::format("%s: %s", get_error_name(value.code), value.description);
		}

		return fmt::format("%s (%s:%lld): %s", get_error_name(value.code), value.domain, value.native_code, value.description);
	}

	void throw_error(const error& value, std::string_view operation)
	{
		fmt::throw_exception("%s failed: %s", operation, format_error(value));
	}
}
