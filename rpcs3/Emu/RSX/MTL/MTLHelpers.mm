#include "stdafx.h"
#include "MTLHelpers.h"
#include "MTLFormats.h"

#include "Emu/RSX/gcm_enums.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace mtl
{
	namespace
	{
		std::atomic<const render_device*> g_helper_renderer = nullptr;
		std::atomic<u64> g_runtime_state = runtime_state_none;
		std::atomic<u64> g_total_frames = 0;
		std::atomic<u64> g_completed_frames = 0;
		std::mutex g_submit_mutex;

		constexpr u64 known_runtime_states = runtime_state_uninterruptible | runtime_state_heap_dirty |
			runtime_state_heap_changed | runtime_state_device_fault | runtime_state_surface_changed;
		constexpr u32 known_image_setup_flags = image_setup_initialize_state | image_setup_preserve_state |
			image_setup_source_gpu_resident | image_setup_source_host_pointer | image_setup_byte_swap;

		[[noreturn]] void throw_native_error(NSError* error, const char* operation)
		{
			fmt::throw_exception("%s failed: %s", operation,
				error.localizedDescription.UTF8String ?: "Metal returned no diagnostic");
		}

		u32 absolute_difference(s32 first, s32 second)
		{
			const s64 difference = static_cast<s64>(second) - first;
			const u64 result = difference < 0 ? static_cast<u64>(-difference) : static_cast<u64>(difference);
			if (!result || result > std::numeric_limits<u32>::max())
			{
				fmt::throw_exception("Metal image rectangle has an invalid extent");
			}
			return static_cast<u32>(result);
		}

		u32 mip_dimension(u32 value, u32 level)
		{
			return std::max(1u, value >> std::min(level, 31u));
		}

		s32 mip_coordinate(s32 value, u32 level)
		{
			if (!level)
			{
				return value;
			}
			const s64 divisor = s64{1} << std::min(level, 31u);
			const s64 wide = value;
			const s64 result = wide >= 0 ? wide / divisor : -((-wide + divisor - 1) / divisor);
			return static_cast<s32>(result);
		}

		void validate_rectangle(const image& resource, const image_rectangle& rectangle, u32 mip)
		{
			const s32 x0 = mip_coordinate(rectangle.x0, mip);
			const s32 x1 = mip_coordinate(rectangle.x1, mip);
			const s32 y0 = mip_coordinate(rectangle.y0, mip);
			const s32 y1 = mip_coordinate(rectangle.y1, mip);
			const s32 minimum_x = std::min(x0, x1);
			const s32 minimum_y = std::min(y0, y1);
			if (minimum_x < 0 || minimum_y < 0 ||
				static_cast<u64>(std::max(x0, x1)) > mip_dimension(resource.width(), mip) ||
				static_cast<u64>(std::max(y0, y1)) > mip_dimension(resource.height(), mip) || x0 == x1 || y0 == y1)
			{
				fmt::throw_exception("Metal image rectangle exceeds its mip dimensions");
			}
		}

		u8 select_aspects(const image& resource, u8 requested)
		{
			const u8 selected = requested == 0xff ? resource.aspects() : static_cast<u8>(requested & resource.aspects());
			if (!selected || (selected & (selected - 1)) != 0)
			{
				fmt::throw_exception("Metal image helper requires exactly one available texture aspect");
			}
			return selected;
		}

		image_scale_region make_scale_region(const image& source, const image& destination,
			const image_rectangle& source_rectangle, const image_rectangle& destination_rectangle,
			u32 mip, u8 source_aspects, u8 destination_aspects)
		{
			validate_rectangle(source, source_rectangle, mip);
			validate_rectangle(destination, destination_rectangle, mip);
			return
			{
				.source = {mip, 0, source_aspects},
				.destination = {mip, 0, destination_aspects},
				.source_box = {mip_coordinate(source_rectangle.x0, mip), mip_coordinate(source_rectangle.y0, mip), 0,
					mip_coordinate(source_rectangle.x1, mip), mip_coordinate(source_rectangle.y1, mip), 1},
				.destination_box = {mip_coordinate(destination_rectangle.x0, mip), mip_coordinate(destination_rectangle.y0, mip), 0,
					mip_coordinate(destination_rectangle.x1, mip), mip_coordinate(destination_rectangle.y1, mip), 1},
				.layer_count = std::min(source.layers(), destination.layers()),
			};
		}

		struct alignas(16) byte_swap_parameters
		{
			u64 element_count = 0;
			u32 element_size = 0;
			u32 padding = 0;
		};

		struct swap_range
		{
			u64 offset = 0;
			u64 elements = 0;
		};

		u64 checked_multiply(u64 left, u64 right)
		{
			if (right && left > std::numeric_limits<u64>::max() / right)
			{
				fmt::throw_exception("Metal image helper size overflows");
			}
			return left * right;
		}

		u64 checked_add(u64 left, u64 right)
		{
			if (left > std::numeric_limits<u64>::max() - right)
			{
				fmt::throw_exception("Metal image helper size overflows");
			}
			return left + right;
		}

		std::vector<swap_range> make_swap_ranges(const buffer& destination,
			std::span<const buffer_image_copy_region> regions, const image_readback_options& options)
		{
			if (options.element_size != 2 && options.element_size != 4 && options.element_size != 8)
			{
				fmt::throw_exception("Metal readback byte swapping supports 2, 4, or 8-byte elements");
			}
			if (options.synchronize)
			{
				if (options.synchronize.length % options.element_size ||
					!destination.in_range(options.synchronize.offset, options.synchronize.length))
				{
					fmt::throw_exception("Metal readback synchronization range is invalid");
				}
				return {{options.synchronize.offset, options.synchronize.length / options.element_size}};
			}

			std::vector<swap_range> result;
			for (const auto& region : regions)
			{
				const u64 active_row_bytes = checked_multiply(region.extent.width, options.element_size);
				const u64 row_pitch = region.bytes_per_row ? region.bytes_per_row : active_row_bytes;
				const u64 image_pitch = region.bytes_per_image ? region.bytes_per_image :
					checked_multiply(row_pitch, region.extent.height);
				if (row_pitch < active_row_bytes || row_pitch % options.element_size ||
					image_pitch < checked_multiply(row_pitch, region.extent.height))
				{
					fmt::throw_exception("Metal readback byte-swap pitches are invalid");
				}
				for (u32 layer = 0; layer < region.layer_count; ++layer)
				{
					for (u32 z = 0; z < region.extent.depth; ++z)
					{
						const u64 plane = checked_add(checked_multiply(layer, checked_multiply(image_pitch, region.extent.depth)),
							checked_multiply(z, image_pitch));
						for (u32 y = 0; y < region.extent.height; ++y)
						{
							const u64 offset = checked_add(region.buffer_offset,
								checked_add(plane, checked_multiply(y, row_pitch)));
							if (!destination.in_range(offset, active_row_bytes))
							{
								fmt::throw_exception("Metal readback byte-swap range exceeds its destination buffer");
							}
							result.push_back({offset, region.extent.width});
						}
					}
				}
			}
			return result;
		}

		void encode_byte_swap(command_buffer& command, buffer& destination,
			std::span<const buffer_image_copy_region> regions, const image_readback_options& options)
		{
			const std::vector<swap_range> ranges = make_swap_ranges(destination, regions, options);
			if (ranges.empty())
			{
				return;
			}

			static constexpr const char* source = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct byte_swap_parameters { ulong element_count; uint element_size; uint padding; };
kernel void rsx_byte_swap(device uchar* values [[buffer(0)]],
	constant byte_swap_parameters& parameters [[buffer(1)]], uint index [[thread_position_in_grid]])
{
	if (index >= parameters.element_count) return;
	device uchar* element = values + ulong(index) * parameters.element_size;
	for (uint first = 0, last = parameters.element_size - 1; first < last; ++first, --last)
	{
		const uchar temporary = element[first];
		element[first] = element[last];
		element[last] = temporary;
	}
}
)MSL";

			render_device& render = command.allocator().owner();
			id<MTLDevice> device = render.native_handle();
			id<MTL4Compiler> compiler = render.compiler();
			MTLCompileOptions* compile_options = [MTLCompileOptions new];
			compile_options.languageVersion = MTLLanguageVersion4_0;
			MTL4LibraryDescriptor* library_descriptor = [MTL4LibraryDescriptor new];
			library_descriptor.name = @"RPCS3 readback byte swap";
			library_descriptor.source = [NSString stringWithUTF8String:source];
			library_descriptor.options = compile_options;
			NSError* error = nil;
			id<MTLLibrary> library = [compiler newLibraryWithDescriptor:library_descriptor error:&error];
			if (!library)
			{
				throw_native_error(error, "Metal readback byte-swap library compilation");
			}
			MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
			function.library = library;
			function.name = @"rsx_byte_swap";
			MTL4ComputePipelineDescriptor* pipeline_descriptor = [MTL4ComputePipelineDescriptor new];
			pipeline_descriptor.label = @"RPCS3 readback byte swap";
			pipeline_descriptor.computeFunctionDescriptor = function;
			pipeline_descriptor.maxTotalThreadsPerThreadgroup = 256;
			id<MTLComputePipelineState> pipeline = [compiler newComputePipelineStateWithDescriptor:pipeline_descriptor
				compilerTaskOptions:nil error:&error];
			if (!pipeline)
			{
				throw_native_error(error, "Metal readback byte-swap pipeline compilation");
			}

			const u64 parameter_stride = 256;
			const u64 parameter_bytes = checked_multiply(parameter_stride, ranges.size());
			id<MTLBuffer> parameter_buffer = [device newBufferWithLength:parameter_bytes
				options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
			if (!parameter_buffer)
			{
				fmt::throw_exception("Metal failed to allocate readback byte-swap parameters");
			}
			parameter_buffer.label = @"RPCS3 readback byte-swap parameters";
			for (usz index = 0; index < ranges.size(); ++index)
			{
				const byte_swap_parameters parameters{ranges[index].elements, options.element_size, 0};
				std::memcpy(static_cast<u8*>(parameter_buffer.contents) + parameter_stride * index,
					&parameters, sizeof(parameters));
			}

			MTL4ArgumentTableDescriptor* table_descriptor = [MTL4ArgumentTableDescriptor new];
			table_descriptor.maxBufferBindCount = 2;
			table_descriptor.initializeBindings = YES;
			table_descriptor.label = @"RPCS3 readback byte-swap arguments";
			id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:table_descriptor error:&error];
			if (!table)
			{
				throw_native_error(error, "Metal readback byte-swap argument-table creation");
			}

			id<MTL4ComputeCommandEncoder> encoder = (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder();
			if (!encoder || command.active_encoder() != encoder_kind::compute)
			{
				encoder = (__bridge id<MTL4ComputeCommandEncoder>)command.begin_compute_encoding();
			}
			[encoder barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
			[encoder setComputePipelineState:pipeline];
			[encoder setArgumentTable:table];
			for (usz index = 0; index < ranges.size(); ++index)
			{
				[table setAddress:destination.gpu_address() + ranges[index].offset atIndex:0];
				[table setAddress:parameter_buffer.gpuAddress + parameter_stride * index atIndex:1];
				[encoder dispatchThreads:MTLSizeMake(ranges[index].elements, 1, 1)
					threadsPerThreadgroup:MTLSizeMake(std::min<u64>(pipeline.maxTotalThreadsPerThreadgroup, 256), 1, 1)];
			}
			command.retain_native_object((__bridge void*)destination.native_handle(), true);
			command.retain_native_object((__bridge void*)parameter_buffer, true);
			command.retain_native_object((__bridge void*)pipeline, false);
			command.retain_native_object((__bridge void*)table, false);
			command.set_flag(command_has_dma_transfer);
		}

		void validate_runtime_state(runtime_state state)
		{
			const u64 bits = static_cast<u64>(state);
			if (!bits || (bits & ~known_runtime_states) != 0)
			{
				fmt::throw_exception("Invalid Metal renderer runtime-state mask");
			}
		}

		void encode_raw_upload(command_buffer& command, const buffer& source, image& destination,
			std::span<const buffer_image_copy_region> regions)
		{
			using upload_function = void (*)(command_buffer&, const buffer&, image&,
				std::span<const buffer_image_copy_region>);
			static_cast<upload_function>(&mtl::upload_image)(command, source, destination, regions);
		}
	}

	struct renderer_resources::impl
	{
		mutable std::mutex mutex;
		shared_state* shared = nullptr;
		data_heap upload;
		scratch_resource_pool scratch;
		sampler_pool samplers;
		argument_table_cache argument_tables;
		bool initialized = false;
	};

	renderer_resources::renderer_resources()
		: m_impl(std::make_unique<impl>())
	{
	}

	renderer_resources::~renderer_resources()
	{
		destroy();
	}

	void renderer_resources::initialize(shared_state& state)
	{
		if (!state)
		{
			fmt::throw_exception("Metal renderer resources require initialized shared state");
		}
		destroy();
		std::lock_guard lock(m_impl->mutex);
		data_heap_create_info heap_info;
		heap_info.initial_size = 64ull * 1024 * 1024;
		heap_info.maximum_size = 1024ull * 1024 * 1024;
		heap_info.growth_quantum = 64ull * 1024 * 1024;
		heap_info.guard_size = 1024ull * 1024;
		heap_info.usage = buffer_usage_copy_source | buffer_usage_constant | buffer_usage_storage;
		heap_info.flags = data_heap_persistent_mapping;
		heap_info.pool = allocation_pool::system;
		heap_info.label = "RPCS3 renderer upload heap";
		heap_info.growth_callback = [](u64, u64)
		{
			raise_status_interrupt(runtime_state_heap_changed);
		};
		try
		{
			m_impl->upload.create(state.allocator(), heap_info);
			m_impl->scratch.create(state.device(), state.allocator());
			m_impl->samplers.create(state.device());
			m_impl->argument_tables.create(state.device());
			m_impl->shared = &state;
			m_impl->initialized = true;
		}
		catch (...)
		{
			m_impl->argument_tables.destroy();
			m_impl->samplers.destroy();
			m_impl->scratch.destroy();
			m_impl->upload.destroy();
			m_impl->shared = nullptr;
			throw;
		}
	}

	void renderer_resources::reset(u64 completed_submission_value)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized)
		{
			fmt::throw_exception("Metal renderer resources are not initialized");
		}
		m_impl->upload.reclaim(completed_submission_value);
		m_impl->scratch.reclaim(completed_submission_value);
		m_impl->argument_tables.reclaim(completed_submission_value);
		clear_status_interrupt(runtime_state_heap_dirty);
	}

	void renderer_resources::destroy()
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized)
		{
			return;
		}
		m_impl->argument_tables.destroy();
		m_impl->samplers.destroy();
		m_impl->scratch.destroy();
		m_impl->upload.destroy();
		m_impl->shared = nullptr;
		m_impl->initialized = false;
	}

	void renderer_resources::trim(memory_pressure pressure, u64 completed_submission_value)
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized)
		{
			return;
		}
		m_impl->upload.reclaim(completed_submission_value);
		m_impl->upload.trim(pressure);
		m_impl->scratch.trim(pressure, completed_submission_value);
		m_impl->argument_tables.reclaim(completed_submission_value);
		m_impl->argument_tables.trim(pressure == memory_pressure::normal ? 64 :
			(pressure == memory_pressure::warning ? 16 : 0));
		if (pressure == memory_pressure::critical)
		{
			m_impl->samplers.clear();
		}
	}

	data_heap& renderer_resources::upload_heap()
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized) fmt::throw_exception("Metal renderer resources are not initialized");
		return m_impl->upload;
	}

	scratch_resource_pool& renderer_resources::scratch()
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized) fmt::throw_exception("Metal renderer resources are not initialized");
		return m_impl->scratch;
	}

	sampler_pool& renderer_resources::samplers()
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized) fmt::throw_exception("Metal renderer resources are not initialized");
		return m_impl->samplers;
	}

	argument_table_cache& renderer_resources::argument_tables()
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->initialized) fmt::throw_exception("Metal renderer resources are not initialized");
		return m_impl->argument_tables;
	}

	global_resource_statistics renderer_resources::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		global_resource_statistics result;
		if (m_impl->initialized)
		{
			result.upload_heap = m_impl->upload.statistics();
			result.scratch = m_impl->scratch.statistics();
			result.argument_tables = m_impl->argument_tables.statistics();
			result.samplers = m_impl->samplers.size();
		}
		result.total_frames = current_frame_id();
		result.completed_frames = last_completed_frame_id();
		result.runtime_flags = g_runtime_state.load(std::memory_order_acquire);
		return result;
	}

	renderer_resources::operator bool() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->initialized;
	}

	void set_current_renderer(const render_device& device)
	{
		if (!device)
		{
			fmt::throw_exception("Cannot select an uninitialized Metal renderer");
		}
		if (mtl::get_current_renderer() != &device)
		{
			fmt::throw_exception("Metal helper renderer must match the active render device");
		}
		const render_device* expected = nullptr;
		if (!g_helper_renderer.compare_exchange_strong(expected, &device, std::memory_order_acq_rel))
		{
			if (expected == &device)
			{
				return;
			}
			fmt::throw_exception("A different Metal renderer is already active");
		}
		g_runtime_state.store(runtime_state_none, std::memory_order_release);
		g_total_frames.store(0, std::memory_order_release);
		g_completed_frames.store(0, std::memory_order_release);
	}

	void clear_current_renderer(const render_device& device)
	{
		const render_device* expected = &device;
		if (!g_helper_renderer.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel))
		{
			fmt::throw_exception("Attempted to clear a Metal renderer that is not current");
		}
		g_runtime_state.store(runtime_state_none, std::memory_order_release);
	}

	bool emulate_primitive_restart(primitive_topology topology)
	{
		return topology == primitive_topology::line_strip || topology == primitive_topology::triangle_strip;
	}

	bool sanitize_floating_point_values()
	{
		const render_device* renderer = get_current_renderer();
		return renderer && !is_apple(renderer->info().chip.chip);
	}

	bool emulate_conditional_rendering()
	{
		return true;
	}

	bool use_strict_query_scopes()
	{
		return true;
	}

	submission submit_serialized(command_buffer& command, const submit_info& info)
	{
		std::lock_guard lock(g_submit_mutex);
		return command.submit(info);
	}

	u32 image_rectangle::width() const
	{
		return absolute_difference(x0, x1);
	}

	u32 image_rectangle::height() const
	{
		return absolute_difference(y0, y1);
	}

	void upload_image(command_buffer& command, const buffer& source, image& destination,
		std::span<const buffer_image_copy_region> regions, u32 setup_flags)
	{
		if (setup_flags & ~known_image_setup_flags)
		{
			fmt::throw_exception("Unknown Metal image setup flags");
		}
		if ((setup_flags & image_setup_source_host_pointer) && !source.is_cpu_visible())
		{
			fmt::throw_exception("Metal host-pointer image upload requires CPU-visible source storage");
		}
		const image_state previous = destination.state();
		if (setup_flags & image_setup_byte_swap)
		{
			shared_state& shared = get_shared_state();
			if (!shared || &shared.device() != &command.allocator().owner())
			{
				fmt::throw_exception("Metal byte-swapped upload requires the active shared renderer");
			}
			buffer swapped(shared.allocator(),
				{
					.size = source.size(),
					.usage = buffer_usage_copy_source | buffer_usage_copy_destination | buffer_usage_storage,
					.storage = storage_mode::private_,
					.access = cpu_access::none,
					.pool = allocation_pool::scratch,
					.label = "RPCS3 byte-swapped image upload",
				});
			if (!command.is_recording())
			{
				fmt::throw_exception("Metal image upload requires active command recording");
			}
			if (command.active_encoder() == encoder_kind::render)
			{
				command.end_encoding();
			}
			id<MTL4ComputeCommandEncoder> encoder = command.active_encoder() == encoder_kind::compute ?
				(__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder() :
				(__bridge id<MTL4ComputeCommandEncoder>)command.begin_compute_encoding();
			[encoder copyFromBuffer:source.native_handle() sourceOffset:0
				toBuffer:swapped.native_handle() destinationOffset:0 size:source.size()];
			command.retain_native_object((__bridge void*)source.native_handle(), true);
			command.retain_native_object((__bridge void*)swapped.native_handle(), true);
			image_readback_options swap_options{.swap_bytes = true, .element_size = 4};
			encode_byte_swap(command, swapped, regions, swap_options);
			encoder = (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder();
			[encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageBlit
				visibilityOptions:MTL4VisibilityOptionDevice];
			encode_raw_upload(command, swapped, destination, regions);
		}
		else
		{
			encode_raw_upload(command, source, destination, regions);
		}
		if ((setup_flags & image_setup_preserve_state) && previous.initialized)
		{
			image_state restored = previous;
			restored.submission = get_submission_id();
			transition_image(command, destination, restored);
		}
		else if ((setup_flags & image_setup_initialize_state) && !destination.state().initialized)
		{
			fmt::throw_exception("Metal image upload failed to initialize destination state");
		}
	}

	void copy_image_to_buffer(command_buffer& command, image& source, buffer& destination,
		std::span<const buffer_image_copy_region> regions, const image_readback_options& options)
	{
		mtl::download_image(command, source, destination, regions);
		if (options.swap_bytes)
		{
			encode_byte_swap(command, destination, regions, options);
		}
	}

	void copy_buffer_to_image(command_buffer& command, const buffer& source, image& destination,
		std::span<const buffer_image_copy_region> regions)
	{
		encode_raw_upload(command, source, destination, regions);
	}

	u64 calculate_working_buffer_size(u64 base_size, u8 aspects, u32 alignment)
	{
		if (!base_size || !alignment || (alignment & (alignment - 1)) != 0 ||
			!(aspects & (texture_aspect_color | texture_aspect_depth | texture_aspect_stencil)))
		{
			fmt::throw_exception("Invalid Metal working-buffer size request");
		}
		const u64 plane_count = ((aspects & texture_aspect_depth) && (aspects & texture_aspect_stencil)) ? 2 : 1;
		const u64 size = checked_multiply(base_size, plane_count);
		if (size > std::numeric_limits<u64>::max() - (alignment - 1))
		{
			fmt::throw_exception("Metal working-buffer aligned size overflows");
		}
		return (size + alignment - 1) & ~(static_cast<u64>(alignment) - 1);
	}

	void copy_scaled_image(command_buffer& command, image& source, image& destination,
		const image_rectangle& source_rectangle, const image_rectangle& destination_rectangle,
		u32 mipmaps, bool compatible_formats, image_filter filter, const image_conversion& conversion)
	{
		if (!mipmaps || mipmaps > source.mipmaps() || mipmaps > destination.mipmaps())
		{
			fmt::throw_exception("Invalid Metal scaled-copy mip count");
		}
		const u8 source_aspects = select_aspects(source, 0xff);
		const u8 destination_aspects = select_aspects(destination, 0xff);
		if (!compatible_formats && conversion.kind == image_conversion_kind::none)
		{
			fmt::throw_exception("Incompatible Metal scaled-copy formats require an explicit conversion");
		}
		for (u32 mip = 0; mip < mipmaps; ++mip)
		{
			mtl::scale_image(command, source, destination,
				make_scale_region(source, destination, source_rectangle, destination_rectangle,
					mip, source_aspects, destination_aspects), filter, conversion);
		}
	}

	void copy_image_region(command_buffer& command, image& source, image& destination,
		const image_rectangle& source_rectangle, const image_rectangle& destination_rectangle,
		u32 mipmaps, u8 source_aspects, u8 destination_aspects)
	{
		if (!mipmaps || mipmaps > source.mipmaps() || mipmaps > destination.mipmaps())
		{
			fmt::throw_exception("Invalid Metal image-copy mip count");
		}
		const u8 selected_source = select_aspects(source, source_aspects);
		const u8 selected_destination = select_aspects(destination, destination_aspects);
		for (u32 mip = 0; mip < mipmaps; ++mip)
		{
			validate_rectangle(source, source_rectangle, mip);
			validate_rectangle(destination, destination_rectangle, mip);
			const u32 source_width = absolute_difference(mip_coordinate(source_rectangle.x0, mip), mip_coordinate(source_rectangle.x1, mip));
			const u32 source_height = absolute_difference(mip_coordinate(source_rectangle.y0, mip), mip_coordinate(source_rectangle.y1, mip));
			const u32 destination_width = absolute_difference(mip_coordinate(destination_rectangle.x0, mip), mip_coordinate(destination_rectangle.x1, mip));
			const u32 destination_height = absolute_difference(mip_coordinate(destination_rectangle.y0, mip), mip_coordinate(destination_rectangle.y1, mip));
			const bool direct = source.format() == destination.format() && source.samples() == destination.samples() &&
				source_width == destination_width && source_height == destination_height &&
				source_rectangle.x1 >= source_rectangle.x0 && source_rectangle.y1 >= source_rectangle.y0 &&
				destination_rectangle.x1 >= destination_rectangle.x0 && destination_rectangle.y1 >= destination_rectangle.y0;
			if (direct)
			{
				const image_copy_region region
				{
					.source = {mip, 0, selected_source},
					.destination = {mip, 0, selected_destination},
					.source_origin = {static_cast<u32>(mip_coordinate(source_rectangle.x0, mip)),
						static_cast<u32>(mip_coordinate(source_rectangle.y0, mip)), 0},
					.destination_origin = {static_cast<u32>(mip_coordinate(destination_rectangle.x0, mip)),
						static_cast<u32>(mip_coordinate(destination_rectangle.y0, mip)), 0},
					.extent = {source_width, source_height, 1},
					.layer_count = std::min(source.layers(), destination.layers()),
				};
				mtl::copy_image(command, source, destination, std::span{&region, 1});
			}
			else
			{
				image_conversion conversion;
				conversion.kind = source.format() == destination.format() ? image_conversion_kind::none :
					((selected_source & texture_aspect_depth) ? image_conversion_kind::depth_to_color :
					 ((selected_destination & texture_aspect_depth) ? image_conversion_kind::color_to_depth :
					  image_conversion_kind::color_to_color));
				mtl::scale_image(command, source, destination,
					make_scale_region(source, destination, source_rectangle, destination_rectangle,
						mip, selected_source, selected_destination), image_filter::nearest, conversion);
			}
		}
	}

	void copy_image_typeless(command_buffer& command, image& source, image& destination,
		const image_rectangle& source_rectangle, const image_rectangle& destination_rectangle,
		u32 mipmaps, u8 source_aspects, u8 destination_aspects)
	{
		copy_image_region(command, source, destination, source_rectangle, destination_rectangle,
			mipmaps, source_aspects, destination_aspects);
	}

	surface_format_mapping compatible_surface_format(u32 color_format)
	{
		const render_device* renderer = get_current_renderer();
		if (!renderer)
		{
			fmt::throw_exception("Metal surface format selection requires an active renderer");
		}
		const native_format_description format = get_color_surface_format(renderer->info(),
			static_cast<rsx::surface_color_format>(color_format));
		return {format.pixel_format, format.components,
			static_cast<u64>(format.compatibility_class), format.requires_conversion()};
	}

	void raise_status_interrupt(runtime_state status)
	{
		validate_runtime_state(status);
		g_runtime_state.fetch_or(static_cast<u64>(status), std::memory_order_acq_rel);
	}

	void clear_status_interrupt(runtime_state status)
	{
		validate_runtime_state(status);
		g_runtime_state.fetch_and(~static_cast<u64>(status), std::memory_order_acq_rel);
	}

	bool test_status_interrupt(runtime_state status)
	{
		validate_runtime_state(status);
		return (g_runtime_state.load(std::memory_order_acquire) & static_cast<u64>(status)) != 0;
	}

	void enter_uninterruptible()
	{
		raise_status_interrupt(runtime_state_uninterruptible);
	}

	void leave_uninterruptible()
	{
		clear_status_interrupt(runtime_state_uninterruptible);
	}

	bool is_uninterruptible()
	{
		return test_status_interrupt(runtime_state_uninterruptible);
	}

	void advance_completed_frame_counter()
	{
		u64 completed = g_completed_frames.load(std::memory_order_relaxed);
		for (;;)
		{
			const u64 total = g_total_frames.load(std::memory_order_acquire);
			if (completed >= total)
			{
				fmt::throw_exception("Metal completed-frame counter cannot overtake submitted frames");
			}
			if (g_completed_frames.compare_exchange_weak(completed, completed + 1, std::memory_order_acq_rel))
			{
				return;
			}
		}
	}

	void advance_frame_counter()
	{
		u64 current = g_total_frames.load(std::memory_order_relaxed);
		for (;;)
		{
			if (current == std::numeric_limits<u64>::max())
			{
				fmt::throw_exception("Metal frame counter overflowed");
			}
			if (g_total_frames.compare_exchange_weak(current, current + 1, std::memory_order_acq_rel))
			{
				return;
			}
		}
	}

	u64 current_frame_id()
	{
		return g_total_frames.load(std::memory_order_acquire);
	}

	u64 last_completed_frame_id()
	{
		return g_completed_frames.load(std::memory_order_acquire);
	}

	void blitter::scale_image(command_buffer& command, image& source, image& destination,
		image_rectangle source_area, image_rectangle destination_area, bool interpolate,
		const image_conversion& conversion)
	{
		copy_scaled_image(command, source, destination, source_area, destination_area, 1,
			source.format() == destination.format(), interpolate ? image_filter::linear : image_filter::nearest,
			conversion);
	}
}
