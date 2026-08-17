#pragma once

#include <string>
#include <string_view>

#include "util/types.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace mtl
{
	inline constexpr u32 minimum_macos_major = 26;
	inline constexpr u32 minimum_macos_minor = 0;

	// Native Metal objects cross the C++ boundary as non-owning opaque pointers.
	// Ownership remains in Objective-C++ resource wrappers.
#ifdef __OBJC__
	using device_handle = id<MTLDevice>;
	using command_queue_handle = id<MTL4CommandQueue>;
	using command_buffer_handle = id<MTL4CommandBuffer>;
	using command_allocator_handle = id<MTL4CommandAllocator>;
	using compiler_handle = id<MTL4Compiler>;
	using argument_table_handle = id<MTL4ArgumentTable>;
	using counter_heap_handle = id<MTL4CounterHeap>;
	using drawable_handle = id<MTLDrawable>;
	using buffer_handle = id<MTLBuffer>;
	using texture_handle = id<MTLTexture>;
	using sampler_handle = id<MTLSamplerState>;
	using library_handle = id<MTLLibrary>;
	using function_handle = id<MTLFunction>;
	using pipeline_archive_handle = id<MTL4Archive>;
	using render_pipeline_handle = id<MTLRenderPipelineState>;
	using compute_pipeline_handle = id<MTLComputePipelineState>;
	using shared_event_handle = id<MTLSharedEvent>;
#else
	using device_handle = void*;
	using command_queue_handle = void*;
	using command_buffer_handle = void*;
	using command_allocator_handle = void*;
	using compiler_handle = void*;
	using argument_table_handle = void*;
	using counter_heap_handle = void*;
	using drawable_handle = void*;
	using buffer_handle = void*;
	using texture_handle = void*;
	using sampler_handle = void*;
	using library_handle = void*;
	using function_handle = void*;
	using pipeline_archive_handle = void*;
	using render_pipeline_handle = void*;
	using compute_pipeline_handle = void*;
	using shared_event_handle = void*;
#endif

	enum class error_code : u8
	{
		none,
		unsupported_platform,
		unsupported_os,
		framework_unavailable,
		device_unavailable,
		metal4_unavailable,
		resource_creation_failed,
		shader_compilation_failed,
		pipeline_creation_failed,
		command_submission_failed,
		device_lost,
		out_of_memory,
		invalid_operation,
		internal_error,
	};

	struct error
	{
		error_code code = error_code::none;
		s64 native_code = 0;
		std::string domain;
		std::string description;

		explicit operator bool() const
		{
			return code != error_code::none;
		}
	};

	struct runtime_capabilities
	{
		bool metal4 = false;
		bool unified_memory = false;
		bool low_power = false;
		bool removable = false;
		bool argument_tables = false;
		bool compiler = false;
		bool shared_events = false;
		bool residency_sets = false;
		bool placement_heaps = false;
		bool sparse_textures = false;
		bool ray_tracing = false;
		bool mesh_shaders = false;
		bool dynamic_libraries = false;
		bool binary_archives = false;
		bool counter_sampling = false;
		bool bc_texture_compression = false;
		bool depth24_stencil8 = false;
		bool framebuffer_fetch = false;
		bool raster_order_groups = false;
		bool barycentric_coordinates = false;
		bool function_pointers = false;
		bool function_stitching = false;
		u64 recommended_working_set_size = 0;
		u64 max_buffer_length = 0;
		u32 max_texture_dimension_2d = 0;
		u32 max_threads_per_threadgroup = 0;
	};

	struct runtime_info
	{
		device_handle device = nullptr;
		std::string device_name;
		runtime_capabilities capabilities;
	};

	[[nodiscard]] bool is_runtime_available();
	[[nodiscard]] runtime_info initialize();
	[[nodiscard]] const runtime_info& get_runtime_info();
	void shutdown();

	[[nodiscard]] const char* get_error_name(error_code code);
	[[nodiscard]] std::string format_error(const error& value);
	[[noreturn]] void throw_error(const error& value, std::string_view operation);
}
