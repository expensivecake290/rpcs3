#include "stdafx.h"
#include "scratch.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mtl
{
	namespace
	{
		enum class lease_state : u8
		{
			available,
			active,
			pending,
		};

		struct buffer_entry
		{
			std::unique_ptr<buffer> resource;
			lease_state state = lease_state::available;
			queue_kind queue = queue_kind::graphics;
			u64 token = 0;
			u64 submission = 0;
			u64 last_use = 0;
		};

		struct image_entry
		{
			std::unique_ptr<viewable_image> resource;
			lease_state state = lease_state::available;
			queue_kind queue = queue_kind::graphics;
			u64 compatibility_class = 0;
			u64 token = 0;
			u64 submission = 0;
			u64 last_use = 0;
			u64 allocated_size = 0;
		};

		struct null_image_entry
		{
			std::unique_ptr<viewable_image> resource;
			image_view* view = nullptr;
		};

		bool is_power_of_two(u64 value)
		{
			return value && (value & (value - 1)) == 0;
		}

		u64 align_up(u64 value, u64 alignment)
		{
			if (!is_power_of_two(alignment) || value > std::numeric_limits<u64>::max() - (alignment - 1))
			{
				fmt::throw_exception("Metal scratch-resource size or alignment overflows");
			}
			return (value + alignment - 1) & ~(alignment - 1);
		}

		id<MTL4ComputeCommandEncoder> compute_encoder(command_buffer& command)
		{
			if (!command.is_recording())
			{
				fmt::throw_exception("Metal scratch operation requires active command recording");
			}
			if (command.active_encoder() == encoder_kind::render) command.end_encoding();
			if (command.active_encoder() == encoder_kind::none)
			{
				return (__bridge id<MTL4ComputeCommandEncoder>)command.begin_compute_encoding();
			}
			return (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder();
		}

		void zero_buffer(command_buffer& command, buffer& resource, u64 destination_stages)
		{
			id<MTL4ComputeCommandEncoder> encoder = compute_encoder(command);
			[encoder fillBuffer:resource.native_handle() range:NSMakeRange(0, resource.size()) value:0];
			barrier_plan visibility;
			visibility.scope = barrier_scope::between_encoders;
			visibility.after_stages = stage_blit;
			visibility.before_stages = destination_stages ? destination_stages : stage_all_gpu;
			visibility.flush_caches = true;
			visibility.end_encoder = true;
			visibility.producer_barrier = true;
			encode_barrier(command.active_native_encoder(), visibility);
			command.end_encoding();
			command.set_flag(command_has_blit_transfer);
		}

		void clear_image(command_buffer& command, const image& resource)
		{
			const image_create_info& info = resource.info();
			if (!(info.usage & (texture_usage_render_target | texture_usage_depth_stencil)))
			{
				fmt::throw_exception("Metal scratch image must be renderable when zero initialization is requested");
			}
			if (command.active_encoder() != encoder_kind::none) command.end_encoding();
			id<MTLTexture> texture = resource.native_handle();
			for (u32 mip = 0; mip < info.mip_levels; ++mip)
			{
				const u32 width = std::max(1u, info.width >> mip);
				const u32 height = std::max(1u, info.height >> mip);
				const u32 planes = info.type == texture_type::texture_3d
					? std::max(1u, info.depth >> mip) : info.array_layers;
				for (u32 plane = 0; plane < planes; ++plane)
				{
					MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
					pass.renderTargetWidth = width;
					pass.renderTargetHeight = height;
					pass.defaultRasterSampleCount = info.sample_count;
					if (info.aspects & texture_aspect_color)
					{
						MTLRenderPassColorAttachmentDescriptor* attachment = pass.colorAttachments[0];
						attachment.texture = texture;
						attachment.level = mip;
						if (info.type == texture_type::texture_3d) attachment.depthPlane = plane;
						else attachment.slice = plane;
						attachment.loadAction = MTLLoadActionClear;
						attachment.storeAction = MTLStoreActionStore;
						attachment.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0);
					}
					if (info.aspects & texture_aspect_depth)
					{
						pass.depthAttachment.texture = texture;
						pass.depthAttachment.level = mip;
						if (info.type == texture_type::texture_3d) pass.depthAttachment.depthPlane = plane;
						else pass.depthAttachment.slice = plane;
						pass.depthAttachment.loadAction = MTLLoadActionClear;
						pass.depthAttachment.storeAction = MTLStoreActionStore;
						pass.depthAttachment.clearDepth = 0.0;
					}
					if (info.aspects & texture_aspect_stencil)
					{
						pass.stencilAttachment.texture = texture;
						pass.stencilAttachment.level = mip;
						if (info.type == texture_type::texture_3d) pass.stencilAttachment.depthPlane = plane;
						else pass.stencilAttachment.slice = plane;
						pass.stencilAttachment.loadAction = MTLLoadActionClear;
						pass.stencilAttachment.storeAction = MTLStoreActionStore;
						pass.stencilAttachment.clearStencil = 0;
					}
					static_cast<void>(command.begin_render_encoding((__bridge void*)pass));
					command.end_encoding();
				}
			}
			command.set_flag(command_reload_dynamic_state);
		}

		bool formats_match(const format_compatibility& left, const format_compatibility& right)
		{
			return left.base_format == right.base_format && left.view_formats == right.view_formats;
		}

		bool image_matches(const image_entry& entry, const scratch_image_request& request)
		{
			const image_create_info& existing = entry.resource->info();
			const image_create_info& desired = request.image;
			if (entry.state != lease_state::available || entry.queue != request.queue ||
				entry.compatibility_class != request.compatibility_class || existing.type != desired.type ||
				!formats_match(existing.formats, desired.formats) || existing.sample_count != desired.sample_count ||
				existing.storage != desired.storage || existing.hazards != desired.hazards ||
				existing.aspects != desired.aspects || (existing.usage & desired.usage) != desired.usage)
			{
				return false;
			}
			if (request.exact_size)
			{
				return existing.width == desired.width && existing.height == desired.height &&
					existing.depth == desired.depth && existing.mip_levels == desired.mip_levels &&
					existing.array_layers == desired.array_layers;
			}
			return existing.width >= desired.width && existing.height >= desired.height &&
				existing.depth >= desired.depth && existing.mip_levels >= desired.mip_levels &&
				existing.array_layers >= desired.array_layers;
		}

		u64 image_fit_score(const image_entry& entry)
		{
			return entry.allocated_size;
		}

		bool valid_image_request(const scratch_image_request& request)
		{
			const image_create_info& info = request.image;
			return info.formats && info.width && info.height && info.depth && info.mip_levels &&
				info.array_layers && info.sample_count && info.usage != texture_usage_none &&
				info.aspects != texture_aspect_none && !info.label.empty() &&
				info.storage != storage_mode::automatic && info.storage != storage_mode::memoryless;
		}

		bool valid_destination_hazard(u64 stages, u64 access)
		{
			constexpr u64 known_access = (access_host_write << 1) - 1;
			return (stages & ~stage_all_gpu) == 0 && (access & ~known_access) == 0 && !has_host_access(access);
		}

		std::pair<image_create_info, u32> null_image_info(texture_type type, u64 pixel_format)
		{
			image_create_info info;
			info.formats.base_format = pixel_format;
			info.type = type;
			info.width = 4;
			info.height = 4;
			info.depth = 1;
			info.array_layers = 1;
			info.mip_levels = 1;
			info.sample_count = 1;
			info.usage = texture_usage_shader_read | texture_usage_render_target;
			info.aspects = texture_aspect_color;
			info.storage = storage_mode::private_;
			info.hazards = hazard_tracking::tracked;
			info.pool = allocation_pool::scratch;
			info.use_placement_heap = true;
			u32 view_slices = 1;
			switch (type)
			{
			case texture_type::texture_1d:
				info.width = 1;
				info.height = 1;
				break;
			case texture_type::texture_1d_array:
				info.width = 1;
				info.height = 1;
				info.array_layers = view_slices = 2;
				break;
			case texture_type::texture_2d:
				break;
			case texture_type::texture_2d_array:
				info.array_layers = view_slices = 2;
				break;
			case texture_type::texture_2d_multisample:
				info.sample_count = 4;
				break;
			case texture_type::texture_2d_multisample_array:
				info.sample_count = 4;
				info.array_layers = view_slices = 2;
				break;
			case texture_type::texture_3d:
				info.depth = 4;
				break;
			case texture_type::texture_cube:
				info.array_layers = view_slices = 6;
				break;
			case texture_type::texture_cube_array:
				info.array_layers = view_slices = 12;
				break;
			}
			info.label = fmt::format("Metal null texture type %u format 0x%llx",
				static_cast<u8>(type), pixel_format);
			return {std::move(info), view_slices};
		}
	}

	struct scratch_resource_pool::impl
	{
		render_device* device = nullptr;
		memory_allocator* allocator = nullptr;
		std::vector<std::unique_ptr<buffer_entry>> buffers;
		std::vector<std::unique_ptr<image_entry>> images;
		std::unordered_map<u64, null_image_entry> null_images;
		std::unique_ptr<sampler> fallback_sampler;
		mutable std::mutex mutex;
		u64 next_token = 1;
		u64 use_serial = 1;
		u64 peak_bytes = 0;
		u64 reuse_count = 0;
		u64 allocation_count = 0;
		u64 eviction_count = 0;

		u64 current_bytes() const
		{
			u64 result = 0;
			for (const auto& entry : buffers) result += entry->resource->size();
			for (const auto& entry : images) result += entry->allocated_size;
			return result;
		}

		void update_peak()
		{
			peak_bytes = std::max(peak_bytes, current_bytes());
		}

		buffer_entry* find_buffer(u64 token)
		{
			for (auto& entry : buffers)
			{
				if (entry->token == token) return entry.get();
			}
			return nullptr;
		}

		image_entry* find_image(u64 token)
		{
			for (auto& entry : images)
			{
				if (entry->token == token) return entry.get();
			}
			return nullptr;
		}
	};

	scratch_resource_pool::scratch_resource_pool()
		: m_impl(std::make_unique<impl>())
	{
	}

	scratch_resource_pool::~scratch_resource_pool()
	{
		destroy();
	}

	void scratch_resource_pool::create(render_device& device, memory_allocator& allocator)
	{
		destroy();
		if (!device || allocator.device().native_handle() != device.native_handle())
		{
			fmt::throw_exception("Invalid Metal scratch-resource pool device or allocator");
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->device = &device;
		m_impl->allocator = &allocator;
		m_impl->next_token = 1;
		m_impl->use_serial = 1;
		m_impl->peak_bytes = 0;
		m_impl->reuse_count = 0;
		m_impl->allocation_count = 0;
		m_impl->eviction_count = 0;
	}

	void scratch_resource_pool::destroy()
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->fallback_sampler.reset();
		m_impl->null_images.clear();
		m_impl->buffers.clear();
		m_impl->images.clear();
		m_impl->device = nullptr;
		m_impl->allocator = nullptr;
		m_impl->next_token = 1;
		m_impl->use_serial = 1;
		m_impl->peak_bytes = 0;
		m_impl->reuse_count = 0;
		m_impl->allocation_count = 0;
		m_impl->eviction_count = 0;
	}

	scratch_buffer_allocation scratch_resource_pool::acquire_buffer(
		command_buffer& command, const scratch_buffer_request& request)
	{
		if (!m_impl || !m_impl->allocator || !command.is_recording() || request.minimum_size == 0 ||
			!is_power_of_two(request.alignment) || request.usage == buffer_usage_none || request.label.empty() ||
			request.storage == storage_mode::automatic || request.storage == storage_mode::memoryless ||
			!valid_destination_hazard(request.destination_stages, request.destination_access) ||
			command.allocator().owner().native_handle() != m_impl->device->native_handle())
		{
			fmt::throw_exception("Invalid Metal scratch-buffer request");
		}
		const u64 allocation_alignment = request.exact_size ? request.alignment : std::max<u64>(request.alignment, 1024 * 1024);
		const u64 allocation_size = align_up(request.minimum_size, allocation_alignment);
		buffer_entry* selected = nullptr;
		bool newly_created = false;
		{
			std::lock_guard lock(m_impl->mutex);
			for (auto& entry : m_impl->buffers)
			{
				if (entry->state != lease_state::available || entry->queue != request.queue ||
					entry->resource->storage() != request.storage ||
					(entry->resource->usage() & request.usage) != request.usage ||
					entry->resource->size() < request.minimum_size ||
					(request.exact_size && entry->resource->size() != allocation_size))
				{
					continue;
				}
				if (!selected || entry->resource->size() < selected->resource->size()) selected = entry.get();
			}
			if (!selected)
			{
				buffer_create_info info;
				info.size = allocation_size;
				info.usage = request.usage;
				info.storage = request.storage;
				info.cache = cpu_cache_mode::default_cache;
				info.access = request.storage == storage_mode::shared || request.storage == storage_mode::managed
					? cpu_access::read_write : cpu_access::none;
				info.pool = allocation_pool::scratch;
				info.label = request.label;
				info.use_placement_heap = true;
				auto entry = std::make_unique<buffer_entry>();
				entry->resource = std::make_unique<buffer>(*m_impl->allocator, info);
				entry->queue = request.queue;
				selected = entry.get();
				m_impl->buffers.push_back(std::move(entry));
				m_impl->allocation_count++;
				m_impl->update_peak();
				newly_created = true;
			}
			else
			{
				m_impl->reuse_count++;
			}
			selected->state = lease_state::active;
			selected->token = m_impl->next_token++;
			selected->submission = 0;
			selected->last_use = m_impl->use_serial++;
		}
		try
		{
			command.retain_native_object((__bridge void*)selected->resource->native_handle(), true);
			if (request.zero_initialize) zero_buffer(command, *selected->resource, request.destination_stages);
		}
		catch (...)
		{
			std::lock_guard lock(m_impl->mutex);
			selected->state = lease_state::available;
			selected->token = 0;
			throw;
		}
		return {selected->resource.get(), selected->token, selected->resource->size(), newly_created};
	}

	scratch_image_allocation scratch_resource_pool::acquire_image(
		command_buffer& command, const scratch_image_request& request)
	{
		if (!m_impl || !m_impl->allocator || !command.is_recording() || !valid_image_request(request) ||
			command.allocator().owner().native_handle() != m_impl->device->native_handle())
		{
			fmt::throw_exception("Invalid Metal scratch-image request");
		}
		image_entry* selected = nullptr;
		bool newly_created = false;
		{
			std::lock_guard lock(m_impl->mutex);
			for (auto& entry : m_impl->images)
			{
				if (!image_matches(*entry, request)) continue;
				if (!selected || image_fit_score(*entry) < image_fit_score(*selected)) selected = entry.get();
			}
			if (!selected)
			{
				image_create_info info = request.image;
				info.pool = allocation_pool::scratch;
				info.use_placement_heap = true;
				auto entry = std::make_unique<image_entry>();
				entry->resource = std::make_unique<viewable_image>(*m_impl->allocator, info);
				entry->queue = request.queue;
				entry->compatibility_class = request.compatibility_class;
				entry->allocated_size = entry->resource->allocation()
					? entry->resource->allocation().size() : entry->resource->native_handle().allocatedSize;
				selected = entry.get();
				m_impl->images.push_back(std::move(entry));
				m_impl->allocation_count++;
				m_impl->update_peak();
				newly_created = true;
			}
			else
			{
				m_impl->reuse_count++;
			}
			selected->state = lease_state::active;
			selected->token = m_impl->next_token++;
			selected->submission = 0;
			selected->last_use = m_impl->use_serial++;
		}
		try
		{
			command.retain_native_object((__bridge void*)selected->resource->native_handle(), true);
			if (request.clear_to_zero)
			{
				clear_image(command, *selected->resource);
				image_state state;
				state.queue = request.queue;
				state.stages = stage_fragment;
				state.access = selected->resource->aspects() & texture_aspect_color
					? access_color_write : access_depth_stencil_write;
				state.initialized = true;
				selected->resource->set_state(state);
			}
		}
		catch (...)
		{
			std::lock_guard lock(m_impl->mutex);
			selected->state = lease_state::available;
			selected->token = 0;
			throw;
		}
		return {selected->resource.get(), selected->token, newly_created};
	}

	void scratch_resource_pool::retire(scratch_buffer_allocation& allocation, u64 submission_value)
	{
		if (!allocation || submission_value == 0) fmt::throw_exception("Invalid Metal scratch-buffer retirement");
		std::lock_guard lock(m_impl->mutex);
		buffer_entry* entry = m_impl->find_buffer(allocation.token);
		if (!entry || entry->state != lease_state::active || entry->resource.get() != allocation.resource)
		{
			fmt::throw_exception("Metal scratch-buffer lease is stale or already retired");
		}
		entry->state = lease_state::pending;
		entry->submission = submission_value;
		allocation = {};
	}

	void scratch_resource_pool::retire(scratch_image_allocation& allocation, u64 submission_value)
	{
		if (!allocation || submission_value == 0) fmt::throw_exception("Invalid Metal scratch-image retirement");
		std::lock_guard lock(m_impl->mutex);
		image_entry* entry = m_impl->find_image(allocation.token);
		if (!entry || entry->state != lease_state::active || entry->resource.get() != allocation.resource)
		{
			fmt::throw_exception("Metal scratch-image lease is stale or already retired");
		}
		entry->state = lease_state::pending;
		entry->submission = submission_value;
		allocation = {};
	}

	void scratch_resource_pool::reclaim(u64 completed_submission_value)
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		for (auto& entry : m_impl->buffers)
		{
			if (entry->state == lease_state::pending && entry->submission <= completed_submission_value)
			{
				entry->state = lease_state::available;
				entry->submission = 0;
				entry->token = 0;
			}
		}
		for (auto& entry : m_impl->images)
		{
			if (entry->state == lease_state::pending && entry->submission <= completed_submission_value)
			{
				entry->state = lease_state::available;
				entry->submission = 0;
				entry->token = 0;
			}
		}
	}

	const sampler& scratch_resource_pool::null_sampler()
	{
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->device) fmt::throw_exception("Metal scratch-resource pool is not initialized");
		if (!m_impl->fallback_sampler)
		{
			sampler_description description;
			description.min_filter = sampler_filter::nearest;
			description.mag_filter = sampler_filter::nearest;
			description.mip_filter = sampler_mip_filter::nearest;
			description.address_s = sampler_address_mode::wrap;
			description.address_t = sampler_address_mode::wrap;
			description.address_r = sampler_address_mode::wrap;
			description.compare_enabled = false;
			description.border = border_color::opaque_white();
			m_impl->fallback_sampler = std::make_unique<sampler>(*m_impl->device, description, "Metal null sampler");
		}
		return *m_impl->fallback_sampler;
	}

	image_view& scratch_resource_pool::null_image_view(command_buffer& command, texture_type type)
	{
		return null_image_view(command, type, MTLPixelFormatRGBA8Unorm);
	}

	image_view& scratch_resource_pool::null_image_view(command_buffer& command, texture_type type,
		u64 pixel_format)
	{
		if (!command.is_recording()) fmt::throw_exception("Metal null texture creation requires active command recording");
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->allocator) fmt::throw_exception("Metal scratch-resource pool is not initialized");
		const u64 key = pixel_format ^ (static_cast<u64>(static_cast<u8>(type)) << 56);
		auto& entry = m_impl->null_images[key];
		if (!entry.resource)
		{
			auto [info, slices] = null_image_info(type, pixel_format);
			auto resource = std::make_unique<viewable_image>(*m_impl->allocator, info);
			clear_image(command, *resource);
			subresource_range range;
			range.first_mip = 0;
			range.mip_count = 1;
			range.first_slice = 0;
			range.slice_count = slices;
			range.color = true;
			image_view* view = resource->get_view(info.formats.base_format, type, {}, range);
			image_state state;
			state.queue = queue_kind::graphics;
			state.stages = stage_vertex | stage_fragment;
			state.access = access_shader_read;
			state.initialized = true;
			resource->set_state(state);
			entry.view = view;
			entry.resource = std::move(resource);
		}
		command.retain_native_object((__bridge void*)entry.resource->native_handle(), true);
		return *entry.view;
	}

	scratch_image_allocation scratch_resource_pool::acquire_typeless_helper(command_buffer& command,
		u64 pixel_format, u64 compatibility_class, u32 width, u32 height, u32 usage)
	{
		if (!pixel_format || !width || !height || usage == texture_usage_none)
		{
			fmt::throw_exception("Invalid Metal typeless scratch-image request");
		}
		scratch_image_request request;
		request.image.type = texture_type::texture_2d;
		request.image.formats.base_format = pixel_format;
		request.image.width = static_cast<u32>(align_up(width, 256));
		request.image.height = static_cast<u32>(align_up(height, 256));
		request.image.depth = 1;
		request.image.mip_levels = 1;
		request.image.array_layers = 1;
		request.image.sample_count = 1;
		request.image.usage = usage;
		request.image.aspects = texture_aspect_color;
		request.image.storage = storage_mode::private_;
		request.image.hazards = hazard_tracking::tracked;
		request.image.pool = allocation_pool::scratch;
		request.image.label = fmt::format("Metal typeless scratch image 0x%llx", pixel_format);
		request.compatibility_class = compatibility_class;
		return acquire_image(command, request);
	}

	void scratch_resource_pool::trim(memory_pressure pressure, u64 completed_submission_value)
	{
		if (!m_impl || !m_impl->allocator) return;
		reclaim(completed_submission_value);
		{
			std::lock_guard lock(m_impl->mutex);
			const auto erase_available = [&](auto& resources, bool all)
			{
				std::vector<typename std::remove_reference_t<decltype(resources)>::value_type> retained;
				retained.reserve(resources.size());
				std::vector<typename std::remove_reference_t<decltype(resources)>::value_type> available;
				for (auto& entry : resources)
				{
					if (entry->state == lease_state::available) available.push_back(std::move(entry));
					else retained.push_back(std::move(entry));
				}
				std::sort(available.begin(), available.end(), [](const auto& left, const auto& right)
				{
					return left->last_use > right->last_use;
				});
				const usz keep = all ? 0 : (available.size() + 1) / 2;
				for (usz index = 0; index < keep; ++index) retained.push_back(std::move(available[index]));
				m_impl->eviction_count += available.size() - keep;
				resources = std::move(retained);
			};
			if (pressure == memory_pressure::warning)
			{
				erase_available(m_impl->buffers, false);
				erase_available(m_impl->images, false);
			}
			else if (pressure == memory_pressure::critical)
			{
				erase_available(m_impl->buffers, true);
				erase_available(m_impl->images, true);
				m_impl->null_images.clear();
				m_impl->fallback_sampler.reset();
			}
		}
		m_impl->allocator->trim(pressure);
	}

	void scratch_resource_pool::clear_available()
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		auto clear = [&](auto& resources)
		{
			const usz previous = resources.size();
			std::erase_if(resources, [](const auto& entry)
			{
				return entry->state == lease_state::available;
			});
			m_impl->eviction_count += previous - resources.size();
		};
		clear(m_impl->buffers);
		clear(m_impl->images);
	}

	scratch_pool_statistics scratch_resource_pool::statistics() const
	{
		scratch_pool_statistics result;
		if (!m_impl) return result;
		std::lock_guard lock(m_impl->mutex);
		result.peak_bytes = m_impl->peak_bytes;
		result.reuse_count = m_impl->reuse_count;
		result.allocation_count = m_impl->allocation_count;
		result.eviction_count = m_impl->eviction_count;
		result.buffer_count = m_impl->buffers.size();
		result.image_count = m_impl->images.size();
		for (const auto& entry : m_impl->buffers)
		{
			result.buffer_bytes += entry->resource->size();
			if (entry->state == lease_state::available) result.available_count++;
			else if (entry->state == lease_state::active) result.active_count++;
			else result.pending_count++;
		}
		for (const auto& entry : m_impl->images)
		{
			result.image_bytes += entry->allocated_size;
			if (entry->state == lease_state::available) result.available_count++;
			else if (entry->state == lease_state::active) result.active_count++;
			else result.pending_count++;
		}
		return result;
	}
}
