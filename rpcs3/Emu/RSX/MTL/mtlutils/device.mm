#include "stdafx.h"
#include "device.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <mutex>

namespace mtl
{
	struct physical_device::impl
	{
		id<MTLDevice> device;
		device_info information;
	};

	struct render_device::impl
	{
		physical_device gpu;
		id<MTL4CommandQueue> graphics_queue;
		id<MTL4CommandQueue> transfer_queue;
		id<MTL4Compiler> compiler;
	};

	namespace
	{
		std::mutex s_render_device_mutex;
		const render_device* s_render_device = nullptr;

		[[noreturn]] void throw_native_error(error_code code, NSError* native_error, std::string_view operation, std::string_view fallback)
		{
			error value;
			value.code = code;
			value.native_code = native_error ? native_error.code : 0;
			value.domain = native_error.domain ? native_error.domain.UTF8String : "Metal";
			value.description = native_error.localizedDescription ? native_error.localizedDescription.UTF8String : std::string(fallback);
			throw_error(value, operation);
		}

		u32 get_apple_gpu_family(id<MTLDevice> device)
		{
			for (u32 family = 10; family >= 7; --family)
			{
				const auto native_family = static_cast<MTLGPUFamily>(1000 + family);
				if ([device supportsFamily:native_family])
				{
					return family;
				}
			}

			return 0;
		}

		bool supports_renderable_format(id<MTLDevice> device, MTLPixelFormat format)
		{
			MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:4 height:4 mipmapped:NO];
			descriptor.storageMode = MTLStorageModePrivate;
			descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
			return [device newTextureWithDescriptor:descriptor] != nil;
		}

		gpu_formats_support query_format_support(id<MTLDevice> device, bool apple_gpu)
		{
			gpu_formats_support result;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
			result.depth24_unorm_stencil8 = !apple_gpu &&
				supports_renderable_format(device, MTLPixelFormatDepth24Unorm_Stencil8);
#pragma clang diagnostic pop
			result.depth32_float_stencil8 = supports_renderable_format(device, MTLPixelFormatDepth32Float_Stencil8);
			result.bc_texture_compression = device.supportsBCTextureCompression;
			result.bgra8_srgb = supports_renderable_format(device, MTLPixelFormatBGRA8Unorm_sRGB);
			result.bgra10_xr = supports_renderable_format(device, MTLPixelFormatBGRA10_XR);
			result.rgba16_float = supports_renderable_format(device, MTLPixelFormatRGBA16Float);
			result.rgba32_float = supports_renderable_format(device, MTLPixelFormatRGBA32Float);
			return result;
		}

		bool validate_metal4_interfaces(id<MTLDevice> device, device_features& features)
		{
			if (![device respondsToSelector:@selector(newMTL4CommandQueueWithDescriptor:error:)] ||
				![device respondsToSelector:@selector(newCompilerWithDescriptor:error:)] ||
				![device respondsToSelector:@selector(newArgumentTableWithDescriptor:error:)])
			{
				return false;
			}

			NSError* error = nil;
			MTL4CommandQueueDescriptor* queue_descriptor = [MTL4CommandQueueDescriptor new];
			queue_descriptor.label = @"RPCS3 Metal 4 capability probe";
			id<MTL4CommandQueue> queue = [device newMTL4CommandQueueWithDescriptor:queue_descriptor error:&error];
			if (!queue)
			{
				return false;
			}

			MTL4CompilerDescriptor* compiler_descriptor = [MTL4CompilerDescriptor new];
			compiler_descriptor.label = @"RPCS3 Metal 4 capability probe";
			id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:compiler_descriptor error:&error];
			if (!compiler)
			{
				return false;
			}

			MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
			table_descriptor.label = @"RPCS3 Metal 4 capability probe";
			table_descriptor.maxBufferBindCount = 31;
			table_descriptor.maxTextureBindCount = 128;
			table_descriptor.maxSamplerStateBindCount = std::min<NSUInteger>(16, device.maxArgumentBufferSamplerCount);
			id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
			if (!table)
			{
				return false;
			}

			features.argument_tables = true;
			return true;
		}

		device_info query_device_info(id<MTLDevice> device)
		{
			device_info result;
			result.name = device.name.UTF8String ?: "Unknown Metal device";
			result.registry_id = device.registryID;
			result.identity.name = result.name;
			result.identity.apple_gpu_family = get_apple_gpu_family(device);
			result.identity.unified_memory = device.hasUnifiedMemory;
			result.identity.low_power = device.lowPower;
			result.identity.removable = device.removable;
			result.chip = classify_device(result.identity);

			result.features.metal4 = validate_metal4_interfaces(device, result.features);
			result.features.shared_events = [device respondsToSelector:@selector(newSharedEvent)];
			result.features.residency_sets = [device respondsToSelector:@selector(newResidencySetWithDescriptor:error:)];
			result.features.placement_heaps = result.identity.apple_gpu_family >= 6 || result.chip.vendor != driver_vendor::apple;
			result.features.sparse_textures = device.sparseTileSizeInBytes != 0;
			result.features.memoryless_textures = result.chip.tile_based;
			result.features.raster_order_groups = device.areRasterOrderGroupsSupported;
			result.features.framebuffer_fetch = result.chip.tile_based;
			result.features.barycentric_coordinates = device.supportsShaderBarycentricCoordinates;
			result.features.programmable_sample_positions = device.areProgrammableSamplePositionsSupported;
			result.features.counter_sampling = [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
			result.features.indirect_command_buffers = [device respondsToSelector:@selector(newIndirectCommandBufferWithDescriptor:maxCommandCount:options:)];
			result.features.dynamic_libraries = device.supportsDynamicLibraries;
			result.features.binary_archives = [device respondsToSelector:@selector(newBinaryArchiveWithDescriptor:error:)];
			result.features.function_pointers = device.supportsFunctionPointers;
			result.features.function_stitching = device.supportsFunctionPointers;
			result.features.mesh_shaders = result.identity.apple_gpu_family >= 7;
			result.features.ray_tracing = device.supportsRaytracing;
			result.features.pull_model_interpolation = device.supportsPullModelInterpolation;
			result.features.float32_filtering = device.supports32BitFloatFiltering;

			result.formats = query_format_support(device, result.chip.vendor == driver_vendor::apple);
			result.shader_types.float16 = true;
			result.shader_types.int8 = true;
			result.shader_types.int16 = true;
			result.shader_types.int64 = result.chip.vendor != driver_vendor::apple;
			result.shader_types.simd_group = result.identity.apple_gpu_family >= 6 || result.chip.vendor != driver_vendor::apple;
			result.shader_types.quad_group = result.shader_types.simd_group;

			result.limits.max_buffer_length = device.maxBufferLength;
			result.limits.max_threadgroup_memory_length = device.maxThreadgroupMemoryLength;
			result.limits.recommended_working_set_size = device.recommendedMaxWorkingSetSize;
			result.limits.max_texture_dimension_1d = 16384;
			result.limits.max_texture_dimension_2d = 16384;
			result.limits.max_texture_dimension_3d = 2048;
			result.limits.max_texture_array_layers = 2048;
			result.limits.max_color_attachments = 8;
			result.limits.max_threads_per_threadgroup = static_cast<u32>(device.maxThreadsPerThreadgroup.width);
			result.limits.max_argument_buffers_per_stage = 8;
			result.limits.max_buffers_per_argument_table = 31;
			result.limits.max_textures_per_argument_table = 128;
			result.limits.max_samplers_per_argument_table = static_cast<u32>(device.maxArgumentBufferSamplerCount);
			result.limits.buffer_offset_alignment = 256;

			result.memory.unified = device.hasUnifiedMemory;
			result.memory.managed_storage = !device.hasUnifiedMemory;
			result.memory.private_storage = true;
			result.memory.memoryless_storage = result.features.memoryless_textures;
			result.memory.recommended_working_set_size = device.recommendedMaxWorkingSetSize;
			result.memory.current_allocated_size = device.currentAllocatedSize;
			return result;
		}
	}

	physical_device::physical_device(std::shared_ptr<impl> implementation)
		: m_impl(std::move(implementation))
	{
	}

	physical_device::operator bool() const
	{
		return m_impl && m_impl->device;
	}

	device_handle physical_device::native_handle() const
	{
		return m_impl ? m_impl->device : nil;
	}

	const device_info& physical_device::info() const
	{
		if (!m_impl)
		{
			fmt::throw_exception("Metal physical device information requested from an empty device");
		}

		return m_impl->information;
	}

	const std::string& physical_device::name() const
	{
		return info().name;
	}

	u64 physical_device::registry_id() const
	{
		return info().registry_id;
	}

	bool physical_device::supports_metal4() const
	{
		return m_impl && m_impl->information.features.metal4;
	}

	render_device::render_device()
		: m_impl(std::make_unique<impl>())
	{
	}

	render_device::~render_device()
	{
		destroy();
	}

	render_device::render_device(render_device&& other) noexcept
		: m_impl(std::move(other.m_impl))
	{
		std::lock_guard lock(s_render_device_mutex);
		if (s_render_device == &other)
		{
			s_render_device = this;
		}
	}

	render_device& render_device::operator=(render_device&& other) noexcept
	{
		if (this != &other)
		{
			destroy();
			m_impl = std::move(other.m_impl);

			std::lock_guard lock(s_render_device_mutex);
			if (s_render_device == &other)
			{
				s_render_device = this;
			}
		}

		return *this;
	}

	void render_device::create(const physical_device& device)
	{
		if (!device)
		{
			fmt::throw_exception("Cannot create a Metal render device from an empty physical device");
		}

		if (!device.supports_metal4())
		{
			fmt::throw_exception("Metal device '%s' does not support the required Metal 4 interfaces", device.name());
		}

		destroy();
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		id<MTLDevice> native_device = device.native_handle();
		NSError* native_error = nil;
		MTL4CommandQueueDescriptor* graphics_descriptor = [MTL4CommandQueueDescriptor new];
		graphics_descriptor.label = @"RPCS3 Metal graphics queue";
		m_impl->graphics_queue = [native_device newMTL4CommandQueueWithDescriptor:graphics_descriptor error:&native_error];
		if (!m_impl->graphics_queue)
		{
			throw_native_error(error_code::resource_creation_failed, native_error, "Metal graphics queue creation", "Metal returned no graphics queue");
		}

		native_error = nil;
		MTL4CommandQueueDescriptor* transfer_descriptor = [MTL4CommandQueueDescriptor new];
		transfer_descriptor.label = @"RPCS3 Metal transfer queue";
		m_impl->transfer_queue = [native_device newMTL4CommandQueueWithDescriptor:transfer_descriptor error:&native_error];
		if (!m_impl->transfer_queue)
		{
			throw_native_error(error_code::resource_creation_failed, native_error, "Metal transfer queue creation", "Metal returned no transfer queue");
		}

		native_error = nil;
		MTL4CompilerDescriptor* compiler_descriptor = [MTL4CompilerDescriptor new];
		compiler_descriptor.label = @"RPCS3 Metal shader compiler";
		m_impl->compiler = [native_device newCompilerWithDescriptor:compiler_descriptor error:&native_error];
		if (!m_impl->compiler)
		{
			throw_native_error(error_code::resource_creation_failed, native_error, "Metal compiler creation", "Metal returned no compiler");
		}

		m_impl->gpu = device;
		{
			std::lock_guard lock(s_render_device_mutex);
			if (s_render_device && s_render_device != this)
			{
				fmt::throw_exception("A different Metal render device is already active");
			}
			s_render_device = this;
		}

		rsx_log.notice("Using Metal 4 device '%s' (%s, %s)", device.name(), get_driver_vendor_name(device.info().chip.vendor), get_chip_class_name(device.info().chip.chip));
	}

	void render_device::destroy()
	{
		{
			std::lock_guard lock(s_render_device_mutex);
			if (s_render_device == this)
			{
				s_render_device = nullptr;
			}
		}

		if (m_impl)
		{
			m_impl->compiler = nil;
			m_impl->transfer_queue = nil;
			m_impl->graphics_queue = nil;
			m_impl->gpu = {};
		}
	}

	render_device::operator bool() const
	{
		return m_impl && m_impl->gpu && m_impl->graphics_queue && m_impl->transfer_queue && m_impl->compiler;
	}

	const physical_device& render_device::gpu() const
	{
		if (!m_impl || !m_impl->gpu)
		{
			fmt::throw_exception("Metal GPU requested before render-device creation");
		}

		return m_impl->gpu;
	}

	const device_info& render_device::info() const
	{
		return gpu().info();
	}

	device_handle render_device::native_handle() const
	{
		return gpu().native_handle();
	}

	command_queue_handle render_device::graphics_queue() const
	{
		return m_impl ? m_impl->graphics_queue : nil;
	}

	command_queue_handle render_device::transfer_queue() const
	{
		return m_impl ? m_impl->transfer_queue : nil;
	}

	compiler_handle render_device::compiler() const
	{
		return m_impl ? m_impl->compiler : nil;
	}

	std::vector<physical_device> enumerate_devices()
	{
		std::vector<physical_device> result;
		if (@available(macOS 26.0, *))
		{
			NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
			for (id<MTLDevice> device in devices)
			{
				auto implementation = std::make_shared<physical_device::impl>();
				implementation->device = device;
				implementation->information = query_device_info(device);
				result.emplace_back(physical_device(std::move(implementation)));
			}

			if (result.empty())
			{
				if (id<MTLDevice> device = MTLCreateSystemDefaultDevice())
				{
					auto implementation = std::make_shared<physical_device::impl>();
					implementation->device = device;
					implementation->information = query_device_info(device);
					result.emplace_back(physical_device(std::move(implementation)));
				}
			}
		}

		std::stable_sort(result.begin(), result.end(), [](const physical_device& lhs, const physical_device& rhs)
		{
			if (lhs.supports_metal4() != rhs.supports_metal4())
			{
				return lhs.supports_metal4();
			}

			if (lhs.info().identity.low_power != rhs.info().identity.low_power)
			{
				return !lhs.info().identity.low_power;
			}

			return !lhs.info().identity.removable && rhs.info().identity.removable;
		});
		return result;
	}

	physical_device select_device(const std::vector<physical_device>& devices, std::string_view preferred_name)
	{
		if (!preferred_name.empty())
		{
			const auto found = std::find_if(devices.begin(), devices.end(), [&](const physical_device& device)
			{
				return device.supports_metal4() && device.name() == preferred_name;
			});

			if (found != devices.end())
			{
				return *found;
			}

			rsx_log.warning("Configured Metal device '%s' is unavailable; selecting another Metal 4 device", preferred_name);
		}

		const auto found = std::find_if(devices.begin(), devices.end(), [](const physical_device& device)
		{
			return device.supports_metal4();
		});

		if (found == devices.end())
		{
			fmt::throw_exception("No Metal 4 capable device is available");
		}

		return *found;
	}

	const render_device* get_current_renderer()
	{
		std::lock_guard lock(s_render_device_mutex);
		return s_render_device;
	}
}
