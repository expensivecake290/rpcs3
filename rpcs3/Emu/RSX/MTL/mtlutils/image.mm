#include "stdafx.h"
#include "image.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <limits>
#include <mutex>

namespace mtl
{
	struct image::impl
	{
		memory_allocation memory;
		id<MTLTexture> texture;
		image_create_info creation;
		std::string name;
		mutable std::mutex state_mutex;
		image_state logical_state;
		bool standalone_tracked = false;
		u64 standalone_size = 0;
	};

	struct image_view::impl
	{
		mtl::image* source_resource = nullptr;
		id<MTLTexture> source;
		id<MTLTexture> view;
		u64 source_uid = 0;
		u64 native_format = 0;
		texture_type view_type = texture_type::texture_2d;
		component_mapping component_map;
		subresource_range subresources;
	};

	namespace
	{
		MTLTextureType to_native_type(texture_type type)
		{
			switch (type)
			{
			case texture_type::texture_1d: return MTLTextureType1D;
			case texture_type::texture_1d_array: return MTLTextureType1DArray;
			case texture_type::texture_2d: return MTLTextureType2D;
			case texture_type::texture_2d_array: return MTLTextureType2DArray;
			case texture_type::texture_2d_multisample: return MTLTextureType2DMultisample;
			case texture_type::texture_2d_multisample_array: return MTLTextureType2DMultisampleArray;
			case texture_type::texture_3d: return MTLTextureType3D;
			case texture_type::texture_cube: return MTLTextureTypeCube;
			case texture_type::texture_cube_array: return MTLTextureTypeCubeArray;
			}
			fmt::throw_exception("Invalid Metal texture type %u", static_cast<u8>(type));
		}

		MTLStorageMode to_native_storage(storage_mode mode)
		{
			switch (mode)
			{
			case storage_mode::shared: return MTLStorageModeShared;
			case storage_mode::managed: return MTLStorageModeManaged;
			case storage_mode::private_: return MTLStorageModePrivate;
			case storage_mode::memoryless: return MTLStorageModeMemoryless;
			case storage_mode::automatic: break;
			}
			fmt::throw_exception("Metal image storage mode must be explicit");
		}

		MTLHazardTrackingMode to_native_hazards(hazard_tracking mode)
		{
			return mode == hazard_tracking::untracked ? MTLHazardTrackingModeUntracked : MTLHazardTrackingModeTracked;
		}

		MTLTextureUsage to_native_usage(u32 usage)
		{
			MTLTextureUsage result = MTLTextureUsageUnknown;
			if (usage & texture_usage_shader_read) result |= MTLTextureUsageShaderRead;
			if (usage & texture_usage_shader_write) result |= MTLTextureUsageShaderWrite;
			if (usage & (texture_usage_render_target | texture_usage_depth_stencil)) result |= MTLTextureUsageRenderTarget;
			if (usage & texture_usage_pixel_format_view) result |= MTLTextureUsagePixelFormatView;
			return result;
		}

		MTLTextureSwizzle to_native_swizzle(component_swizzle swizzle)
		{
			switch (swizzle)
			{
			case component_swizzle::zero: return MTLTextureSwizzleZero;
			case component_swizzle::one: return MTLTextureSwizzleOne;
			case component_swizzle::red: return MTLTextureSwizzleRed;
			case component_swizzle::green: return MTLTextureSwizzleGreen;
			case component_swizzle::blue: return MTLTextureSwizzleBlue;
			case component_swizzle::alpha: return MTLTextureSwizzleAlpha;
			}
			fmt::throw_exception("Invalid Metal component swizzle %u", static_cast<u8>(swizzle));
		}

		MTLTextureSwizzleChannels to_native_swizzle(component_mapping mapping)
		{
			return MTLTextureSwizzleChannelsMake(
				to_native_swizzle(mapping.red),
				to_native_swizzle(mapping.green),
				to_native_swizzle(mapping.blue),
				to_native_swizzle(mapping.alpha));
		}

		void validate_create_info(const render_device& device, const image_create_info& info)
		{
			if (!info.formats || info.width == 0 || info.height == 0 || info.depth == 0 || info.mip_levels == 0 ||
				info.array_layers == 0 || info.sample_count == 0 || info.usage == texture_usage_none || info.aspects == texture_aspect_none)
			{
				fmt::throw_exception("Invalid Metal image creation information");
			}

			const auto& limits = device.info().limits;
			const u32 dimension_limit = info.type == texture_type::texture_3d
				? limits.max_texture_dimension_3d
				: (info.type == texture_type::texture_1d || info.type == texture_type::texture_1d_array
					? limits.max_texture_dimension_1d : limits.max_texture_dimension_2d);
			if (std::max({info.width, info.height, info.depth}) > dimension_limit || info.array_layers > limits.max_texture_array_layers)
			{
				fmt::throw_exception("Metal image dimensions exceed device limits");
			}

			const bool array_type = info.type == texture_type::texture_1d_array || info.type == texture_type::texture_2d_array ||
				info.type == texture_type::texture_2d_multisample_array || info.type == texture_type::texture_cube_array;
			const bool multisampled = info.type == texture_type::texture_2d_multisample || info.type == texture_type::texture_2d_multisample_array;
			if (multisampled != (info.sample_count > 1) || (multisampled && info.mip_levels != 1))
			{
				fmt::throw_exception("Invalid Metal multisample image configuration");
			}
			if (![device.native_handle() supportsTextureSampleCount:info.sample_count])
			{
				fmt::throw_exception("Metal device does not support image sample count %u", info.sample_count);
			}

			if ((info.type == texture_type::texture_cube || info.type == texture_type::texture_cube_array) &&
				(info.width != info.height || (info.array_layers % 6) != 0))
			{
				fmt::throw_exception("Metal cube images must be square and contain complete six-face cubes");
			}
			if (info.type == texture_type::texture_cube && info.array_layers != 6)
			{
				fmt::throw_exception("A non-array Metal cube image must contain exactly six faces");
			}

			const bool requires_height_one = info.type == texture_type::texture_1d || info.type == texture_type::texture_1d_array;
			const bool requires_depth_one = info.type != texture_type::texture_3d;
			if ((requires_height_one && info.height != 1) || (requires_depth_one && info.depth != 1) ||
				(info.type == texture_type::texture_3d && info.array_layers != 1) || (!array_type &&
				 info.type != texture_type::texture_cube && info.array_layers != 1))
			{
				fmt::throw_exception("Metal image dimensions do not match the requested texture type");
			}

			if (info.storage == storage_mode::automatic)
			{
				fmt::throw_exception("Metal image storage mode must be explicit");
			}
			if (info.storage == storage_mode::memoryless &&
				(!(info.usage & (texture_usage_render_target | texture_usage_depth_stencil)) ||
				 (info.usage & (texture_usage_shader_read | texture_usage_shader_write |
					 texture_usage_copy_source | texture_usage_copy_destination)) ||
				 info.mip_levels != 1))
			{
				fmt::throw_exception("Memoryless Metal images may only be transient attachments");
			}
			if (info.storage == storage_mode::memoryless && !device.info().features.memoryless_textures)
			{
				fmt::throw_exception("Metal device does not support memoryless images");
			}
			if (info.storage == storage_mode::managed && !device.info().memory.managed_storage)
			{
				fmt::throw_exception("Metal device does not support managed image storage");
			}
			if (info.use_placement_heap && info.storage != storage_mode::memoryless && !device.info().features.placement_heaps)
			{
				fmt::throw_exception("Metal device does not support placement-heap images");
			}
			if (info.formats.is_mutable() && !(info.usage & texture_usage_pixel_format_view))
			{
				fmt::throw_exception("Mutable Metal images require pixel-format-view usage");
			}
		}

		MTLTextureDescriptor* make_descriptor(const image_create_info& info)
		{
			MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
			descriptor.textureType = to_native_type(info.type);
			descriptor.pixelFormat = static_cast<MTLPixelFormat>(info.formats.base_format);
			descriptor.width = info.width;
			descriptor.height = info.height;
			descriptor.depth = info.depth;
			descriptor.mipmapLevelCount = info.mip_levels;
			switch (info.type)
			{
			case texture_type::texture_1d_array:
			case texture_type::texture_2d_array:
			case texture_type::texture_2d_multisample_array:
				descriptor.arrayLength = info.array_layers;
				break;
			case texture_type::texture_cube_array:
				descriptor.arrayLength = info.array_layers / 6;
				break;
			default:
				descriptor.arrayLength = 1;
				break;
			}
			descriptor.sampleCount = info.sample_count;
			descriptor.storageMode = to_native_storage(info.storage);
			descriptor.hazardTrackingMode = to_native_hazards(info.hazards);
			descriptor.usage = to_native_usage(info.usage);
			return descriptor;
		}

		u32 native_array_length(const image_create_info& info)
		{
			switch (info.type)
			{
			case texture_type::texture_1d_array:
			case texture_type::texture_2d_array:
			case texture_type::texture_2d_multisample_array:
				return info.array_layers;
			case texture_type::texture_cube_array:
				return info.array_layers / 6;
			default:
				return 1;
			}
		}

		bool is_identity(component_mapping mapping)
		{
			return mapping == component_mapping{};
		}

		bool range_fits(u32 first, u32 count, u32 total)
		{
			return count != 0 && first < total && count <= total - first;
		}

		u8 dimension_family(texture_type type)
		{
			switch (type)
			{
			case texture_type::texture_1d:
			case texture_type::texture_1d_array:
				return 1;
			case texture_type::texture_2d:
			case texture_type::texture_2d_array:
			case texture_type::texture_cube:
			case texture_type::texture_cube_array:
				return 2;
			case texture_type::texture_2d_multisample:
			case texture_type::texture_2d_multisample_array:
				return 3;
			case texture_type::texture_3d:
				return 4;
			}
			fmt::throw_exception("Invalid Metal texture type %u", static_cast<u8>(type));
		}

		void validate_view_info(const image& resource, u64 format, texture_type type,
			component_mapping mapping, const subresource_range& range)
		{
			if (!resource || !resource.info().formats.allows(format) ||
				!range_fits(range.first_mip, range.mip_count, resource.mipmaps()) ||
				!range_fits(range.first_slice, range.slice_count, resource.layers()))
			{
				fmt::throw_exception("Invalid Metal image view range or format");
			}

			const u8 requested_aspects = (range.color ? texture_aspect_color : 0) |
				(range.depth ? texture_aspect_depth : 0) | (range.stencil ? texture_aspect_stencil : 0);
			if (requested_aspects == texture_aspect_none || (requested_aspects & ~resource.aspects()) != 0)
			{
				fmt::throw_exception("Metal image view requests unavailable texture aspects");
			}
			if (dimension_family(resource.type()) != dimension_family(type))
			{
				fmt::throw_exception("Metal image view changes to an incompatible texture dimension");
			}

			const bool cube = type == texture_type::texture_cube || type == texture_type::texture_cube_array;
			const bool array = type == texture_type::texture_1d_array || type == texture_type::texture_2d_array ||
				type == texture_type::texture_2d_multisample_array || type == texture_type::texture_cube_array;
			if (cube && ((range.first_slice % 6) != 0 || (range.slice_count % 6) != 0 ||
				(type == texture_type::texture_cube && range.slice_count != 6)))
			{
				fmt::throw_exception("Metal cube views require aligned groups of six faces");
			}
			if (!array && !cube && range.slice_count != 1)
			{
				fmt::throw_exception("A non-array Metal image view may expose only one slice");
			}
			if ((format != resource.format() || !is_identity(mapping)) &&
				!(resource.info().usage & texture_usage_pixel_format_view))
			{
				fmt::throw_exception("Metal image was not created for format or swizzle views");
			}
		}

		u64 hash_combine(u64 seed, u64 value)
		{
			return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
		}
	}

	u32 component_mapping::encode() const
	{
		return static_cast<u32>(red) |
			(static_cast<u32>(green) << 3) |
			(static_cast<u32>(blue) << 6) |
			(static_cast<u32>(alpha) << 9);
	}

	image::image()
		: m_impl(std::make_unique<impl>())
	{
	}

	image::image(memory_allocator& allocator, const image_create_info& info)
		: image()
	{
		create(allocator, info);
	}

	image::~image()
	{
		destroy();
	}

	void image::create(memory_allocator& allocator, const image_create_info& info)
	{
		destroy();
		validate_create_info(allocator.device(), info);
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		id<MTLDevice> device = allocator.device().native_handle();
		MTLTextureDescriptor* descriptor = make_descriptor(info);
		const MTLSizeAndAlign requirements = [device heapTextureSizeAndAlignWithDescriptor:descriptor];
		if (requirements.size == 0 && info.storage != storage_mode::memoryless)
		{
			fmt::throw_exception("Metal reported no allocation requirements for image '%s'", info.label);
		}

		if (info.storage == storage_mode::memoryless)
		{
			m_impl->texture = [device newTextureWithDescriptor:descriptor];
		}
		else if (info.use_placement_heap)
		{
			memory_allocation_request request;
			request.size = requirements.size;
			request.alignment = requirements.align;
			request.storage = info.storage;
			request.access = info.storage == storage_mode::shared || info.storage == storage_mode::managed ? cpu_access::read_write : cpu_access::none;
			request.hazards = info.hazards;
			request.pool = info.pool;
			request.label = info.label;
			request.throw_on_failure = !info.allow_failure;
			request.recover_on_failure = info.recover_on_failure;
			m_impl->memory = allocator.allocate_placement(request);
			if (m_impl->memory)
			{
				id<MTLHeap> heap = (__bridge id<MTLHeap>)m_impl->memory.heap();
				m_impl->texture = [heap newTextureWithDescriptor:descriptor offset:m_impl->memory.offset()];
			}
		}
		else
		{
			m_impl->texture = [device newTextureWithDescriptor:descriptor];
			if (!m_impl->texture && info.recover_on_failure)
			{
				allocator.trim(memory_pressure::critical);
				m_impl->texture = [device newTextureWithDescriptor:descriptor];
			}
			if (m_impl->texture)
			{
				notify_memory_allocated(m_impl.get(), requirements.size, info.pool);
				m_impl->standalone_tracked = true;
				m_impl->standalone_size = requirements.size;
			}
		}

		if (!m_impl->texture)
		{
			m_impl->memory = {};
			if (info.allow_failure)
			{
				return;
			}
			fmt::throw_exception("Failed to create Metal image '%s'", info.label);
		}

		m_impl->creation = info;
		set_debug_name(info.label);
	}

	void image::wrap(texture_handle texture, const image_create_info& info)
	{
		destroy();
		if (!texture)
		{
			fmt::throw_exception("Cannot wrap a null Metal texture");
		}
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		id<MTLTexture> native = texture;
		if (!info.formats || native.pixelFormat != info.formats.base_format || native.width != info.width || native.height != info.height ||
			native.depth != info.depth || native.mipmapLevelCount != info.mip_levels || native.arrayLength != native_array_length(info) ||
			native.sampleCount != info.sample_count || native.textureType != to_native_type(info.type))
		{
			fmt::throw_exception("Wrapped Metal texture does not match its declared image information");
		}

		m_impl->texture = native;
		m_impl->creation = info;
		set_debug_name(info.label);
	}

	void image::destroy()
	{
		if (!m_impl)
		{
			return;
		}
		if (m_impl->standalone_tracked)
		{
			notify_memory_freed(m_impl.get());
		}
		m_impl->texture = nil;
		m_impl->memory = {};
		m_impl->creation = {};
		m_impl->name.clear();
		m_impl->logical_state = {};
		m_impl->standalone_tracked = false;
		m_impl->standalone_size = 0;
	}

	image::operator bool() const
	{
		return m_impl && m_impl->texture;
	}

	texture_handle image::native_handle() const
	{
		return m_impl ? m_impl->texture : nil;
	}

	const image_create_info& image::info() const
	{
		if (!*this) fmt::throw_exception("Information requested from an empty Metal image");
		return m_impl->creation;
	}

	u32 image::width() const { return info().width; }
	u32 image::height() const { return info().height; }
	u32 image::depth() const { return info().depth; }
	u32 image::mipmaps() const { return info().mip_levels; }
	u32 image::layers() const { return info().array_layers; }
	u32 image::samples() const { return info().sample_count; }
	u64 image::format() const { return info().formats.base_format; }
	texture_type image::type() const { return info().type; }
	u8 image::aspects() const { return info().aspects; }
	storage_mode image::storage() const { return info().storage; }
	bool image::is_memoryless() const { return *this && info().storage == storage_mode::memoryless; }
	bool image::is_shareable() const { return *this && info().shareable; }

	const memory_allocation& image::allocation() const
	{
		if (!m_impl || !m_impl->memory)
		{
			fmt::throw_exception("Allocator-backed memory requested from a wrapped, memoryless, standalone, or empty Metal image");
		}
		return m_impl->memory;
	}

	void image::set_debug_name(std::string_view name)
	{
		if (!*this)
		{
			fmt::throw_exception("Cannot name an empty Metal image");
		}
		m_impl->name = name;
		m_impl->texture.label = [NSString stringWithUTF8String:m_impl->name.c_str()];
	}

	const std::string& image::debug_name() const
	{
		if (!m_impl) fmt::throw_exception("Debug name requested from an empty Metal image");
		return m_impl->name;
	}

	image_state image::state() const
	{
		if (!m_impl) fmt::throw_exception("State requested from an empty Metal image");
		std::lock_guard lock(m_impl->state_mutex);
		return m_impl->logical_state;
	}

	void image::set_state(const image_state& state)
	{
		if (!*this || (state.initialized && (state.stages == stage_none || state.access == access_none)))
		{
			fmt::throw_exception("Invalid Metal image state");
		}
		std::lock_guard lock(m_impl->state_mutex);
		m_impl->logical_state = state;
	}

	hazard image::transition_hazard(const image_state& next, const subresource_range& range, bool preserve_encoder) const
	{
		if (!*this || !next.initialized || next.stages == stage_none || next.access == access_none ||
			!range_fits(range.first_mip, range.mip_count, mipmaps()) ||
			!range_fits(range.first_slice, range.slice_count, layers()))
		{
			fmt::throw_exception("Invalid Metal image transition");
		}

		const image_state previous = state();
		hazard result;
		result.resource = resource_kind::texture;
		result.identity = {uid(), uid()};
		result.subresources = range;
		result.producer_stages = previous.initialized ? previous.stages : next.stages;
		result.consumer_stages = next.stages;
		result.producer_access = previous.initialized ? previous.access : access_none;
		result.consumer_access = previous.initialized ? next.access : access_none;
		result.producer_queue = previous.initialized ? previous.queue : next.queue;
		result.consumer_queue = next.queue;
		result.cpu_visible = storage() == storage_mode::shared || storage() == storage_mode::managed;
		result.preserve_encoder = preserve_encoder;
		return result;
	}

	image_view::image_view()
		: m_impl(std::make_unique<impl>())
	{
	}

	image_view::image_view(const mtl::image& resource, u64 format, texture_type type, component_mapping mapping, subresource_range range)
		: image_view()
	{
		create(resource, format, type, mapping, range);
	}

	image_view::~image_view()
	{
		destroy();
	}

	void image_view::create(const mtl::image& resource, u64 format, texture_type type, component_mapping mapping, subresource_range range)
	{
		destroy();
		validate_view_info(resource, format, type, mapping, range);
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		id<MTLTexture> source = resource.native_handle();
		m_impl->view = [source newTextureViewWithPixelFormat:static_cast<MTLPixelFormat>(format)
			textureType:to_native_type(type)
			levels:NSMakeRange(range.first_mip, range.mip_count)
			slices:NSMakeRange(range.first_slice, range.slice_count)
			swizzle:to_native_swizzle(mapping)];
		if (!m_impl->view)
		{
			fmt::throw_exception("Metal rejected image view for '%s'", resource.debug_name());
		}

		m_impl->source = source;
		m_impl->source_resource = const_cast<mtl::image*>(&resource);
		m_impl->source_uid = resource.uid();
		m_impl->native_format = format;
		m_impl->view_type = type;
		m_impl->component_map = mapping;
		m_impl->subresources = range;
		m_impl->view.label = [NSString stringWithFormat:@"%s view %llu", resource.debug_name().c_str(), uid()];
	}

	void image_view::destroy()
	{
		if (m_impl)
		{
			m_impl->view = nil;
			m_impl->source = nil;
			m_impl->source_resource = nullptr;
			m_impl->source_uid = 0;
			m_impl->native_format = 0;
			m_impl->subresources = {};
		}
	}

	image_view::operator bool() const { return m_impl && m_impl->view; }
	texture_handle image_view::native_handle() const { return m_impl ? m_impl->view : nil; }
	mtl::image* image_view::image() const { return m_impl ? m_impl->source_resource : nullptr; }
	u64 image_view::image_uid() const { return m_impl ? m_impl->source_uid : 0; }
	u64 image_view::format() const { return m_impl ? m_impl->native_format : 0; }
	texture_type image_view::type() const { return m_impl ? m_impl->view_type : texture_type::texture_2d; }
	component_mapping image_view::mapping() const { return m_impl ? m_impl->component_map : component_mapping{}; }
	u32 image_view::encoded_component_map() const { return mapping().encode(); }
	subresource_range image_view::range() const { return m_impl ? m_impl->subresources : subresource_range{}; }

	image_view* viewable_image::get_view(u64 format, texture_type type, component_mapping mapping, subresource_range range)
	{
		u64 key = format;
		key = hash_combine(key, static_cast<u8>(type));
		key = hash_combine(key, mapping.encode());
		key = hash_combine(key, range.first_mip);
		key = hash_combine(key, range.mip_count);
		key = hash_combine(key, range.first_slice);
		key = hash_combine(key, range.slice_count);
		key = hash_combine(key, range.color | (range.depth << 1) | (range.stencil << 2));

		for (;;)
		{
			const auto found = m_views.find(key);
			if (found == m_views.end())
			{
				break;
			}
			if (found->second->format() == format && found->second->type() == type &&
				found->second->mapping() == mapping && found->second->range() == range)
			{
				return found->second.get();
			}
			key = hash_combine(key, 0xd6e8feb86659fd93ull);
		}

		auto view = std::make_unique<image_view>(*this, format, type, mapping, range);
		image_view* result = view.get();
		m_views.emplace(key, std::move(view));
		return result;
	}

	void viewable_image::create(memory_allocator& allocator, const image_create_info& info)
	{
		clear_views();
		image::create(allocator, info);
	}

	void viewable_image::wrap(texture_handle texture, const image_create_info& info)
	{
		clear_views();
		image::wrap(texture, info);
	}

	void viewable_image::destroy()
	{
		clear_views();
		image::destroy();
	}

	void viewable_image::clear_views()
	{
		m_views.clear();
	}
}
