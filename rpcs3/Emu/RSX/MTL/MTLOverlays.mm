#include "stdafx.h"
#include "MTLOverlays.h"

#include "MTLFormats.h"
#include "MTLPipelineCompiler.h"
#include "MTLRenderTargets.h"
#include "MTLResourceManager.h"
#include "mtlutils/barriers.h"
#include "mtlutils/image_helpers.h"
#include "mtlutils/shared.h"

#include "Emu/RSX/Overlays/overlays.h"
#include "Emu/RSX/Program/RSXOverlay.h"
#include "Utilities/stereo_config.h"
#include "Emu/Cell/timers.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

namespace mtl
{
	namespace
	{
		[[nodiscard]] u32 mip_dimension(u32 value, u32 level)
		{
			return std::max(value >> level, 1u);
		}

		[[nodiscard]] NSString* native_string(std::string_view value)
		{
			return [NSString stringWithUTF8String:std::string(value).c_str()];
		}

		[[noreturn]] void throw_native_error(NSError* error, std::string_view operation)
		{
			const char* description = error ? error.localizedDescription.UTF8String : nullptr;
			fmt::throw_exception("Metal %s failed: %s", operation,
				description ? description : "unknown error");
		}

		[[nodiscard]] id<MTLLibrary> compile_library(id<MTL4Compiler> compiler,
			std::string_view source, std::string_view label)
		{
			MTLCompileOptions* options = [MTLCompileOptions new];
			options.languageVersion = MTLLanguageVersion4_0;
			options.mathMode = MTLMathModeFast;
			MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
			descriptor.name = native_string(label);
			descriptor.source = native_string(source);
			descriptor.options = options;
			NSError* error = nil;
			id<MTLLibrary> library = [compiler newLibraryWithDescriptor:descriptor error:&error];
			if (!library) throw_native_error(error, fmt::format("%s library compilation", label));
			return library;
		}

		[[nodiscard]] MTLPrimitiveTopologyClass topology_class(primitive_topology topology)
		{
			switch (topology)
			{
			case primitive_topology::point: return MTLPrimitiveTopologyClassPoint;
			case primitive_topology::line:
			case primitive_topology::line_strip: return MTLPrimitiveTopologyClassLine;
			case primitive_topology::triangle:
			case primitive_topology::triangle_strip: return MTLPrimitiveTopologyClassTriangle;
			}
			fmt::throw_exception("Invalid Metal overlay primitive topology");
		}

		[[nodiscard]] MTLPrimitiveType primitive_type(primitive_topology topology)
		{
			switch (topology)
			{
			case primitive_topology::point: return MTLPrimitiveTypePoint;
			case primitive_topology::line: return MTLPrimitiveTypeLine;
			case primitive_topology::line_strip: return MTLPrimitiveTypeLineStrip;
			case primitive_topology::triangle: return MTLPrimitiveTypeTriangle;
			case primitive_topology::triangle_strip: return MTLPrimitiveTypeTriangleStrip;
			}
			fmt::throw_exception("Invalid Metal overlay primitive topology");
		}

		[[nodiscard]] MTLColorWriteMask native_color_mask(u8 mask)
		{
			MTLColorWriteMask result = MTLColorWriteMaskNone;
			if (mask & color_write_red) result |= MTLColorWriteMaskRed;
			if (mask & color_write_green) result |= MTLColorWriteMaskGreen;
			if (mask & color_write_blue) result |= MTLColorWriteMaskBlue;
			if (mask & color_write_alpha) result |= MTLColorWriteMaskAlpha;
			return result;
		}

		[[nodiscard]] subresource_range target_range(const overlay_render_target& target)
		{
			return {
				.first_mip = target.mip_level,
				.mip_count = 1,
				.first_slice = target.array_slice,
				.slice_count = 1,
				.color = bool(target.write_aspects & texture_aspect_color),
				.depth = bool(target.write_aspects & texture_aspect_depth),
				.stencil = bool(target.write_aspects & texture_aspect_stencil),
			};
		}

		[[nodiscard]] image_state operation_state(const image& resource, u64 stages, u64 access)
		{
			const image_state previous = resource.state();
			return {
				.queue = previous.initialized ? previous.queue : queue_kind::graphics,
				.stages = stages,
				.access = access,
				.submission = get_submission_id(),
				.initialized = true,
			};
		}

		void end_active_encoder(command_buffer& command)
		{
			if (command.active_encoder() != encoder_kind::none) command.end_encoding();
		}

		[[nodiscard]] id<MTLBuffer> make_shared_buffer(id<MTLDevice> device,
			std::span<const std::byte> bytes, std::string_view label)
		{
			if (bytes.empty()) return nil;
			const NSUInteger length = std::max<NSUInteger>(bytes.size(), 256);
			id<MTLBuffer> result = [device newBufferWithLength:length
				options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
			if (!result) fmt::throw_exception("Metal overlay buffer allocation failed");
			result.label = native_string(label);
			std::memcpy(result.contents, bytes.data(), bytes.size());
			return result;
		}

		[[nodiscard]] id<MTL4ArgumentTable> make_argument_table(id<MTLDevice> device,
			u32 buffers, u32 textures, u32 samplers, std::string_view label)
		{
			MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
			descriptor.maxBufferBindCount = buffers;
			descriptor.maxTextureBindCount = textures;
			descriptor.maxSamplerStateBindCount = samplers;
			descriptor.initializeBindings = YES;
			descriptor.label = native_string(label);
			NSError* error = nil;
			id<MTL4ArgumentTable> result = [device newArgumentTableWithDescriptor:descriptor error:&error];
			if (!result) throw_native_error(error, fmt::format("%s argument-table creation", label));
			return result;
		}

		void retire_texture(std::unique_ptr<image_view>& view, std::unique_ptr<viewable_image>& texture,
			std::string_view label)
		{
			auto& manager = get_resource_manager();
			if (manager)
			{
				manager.retire(view, {.resource_class = managed_resource_class::texture_view,
					.label = fmt::format("%s view", label)});
				manager.retire(texture, {.resource_class = managed_resource_class::texture,
					.label = std::string(label)});
			}
			else
			{
				view.reset();
				texture.reset();
			}
		}
	}

	void overlay_render_target::validate() const
	{
		if (!resource || !*resource || !write_aspects ||
			(write_aspects & ~resource->aspects()) || mip_level >= resource->mipmaps() ||
			array_slice >= resource->layers() ||
			(write_aspects & texture_aspect_color && write_aspects != texture_aspect_color))
		{
			fmt::throw_exception("Invalid Metal overlay render target");
		}
		if ((write_aspects & texture_aspect_color) && !(resource->info().usage & texture_usage_render_target))
			fmt::throw_exception("Metal overlay color target lacks render-target usage");
		if ((write_aspects & (texture_aspect_depth | texture_aspect_stencil)) &&
			!(resource->info().usage & texture_usage_depth_stencil))
			fmt::throw_exception("Metal overlay depth/stencil target lacks attachment usage");
	}

	overlay_render_target::operator bool() const
	{
		return resource && static_cast<bool>(*resource);
	}

	u32 overlay_render_target::width() const
	{
		validate();
		return mip_dimension(resource->width(), mip_level);
	}

	u32 overlay_render_target::height() const
	{
		validate();
		return mip_dimension(resource->height(), mip_level);
	}

	u32 overlay_render_target::samples() const
	{
		validate();
		return resource->samples();
	}

	u64 overlay_pipeline_key::hash() const
	{
		u64 result = 0x9e3779b97f4a7c15ull;
		auto mix = [&](u64 value)
		{
			result ^= value + 0x9e3779b97f4a7c15ull + (result << 6) + (result >> 2);
		};
		mix(color_format);
		mix(depth_stencil_format);
		mix(sample_count);
		mix(static_cast<u8>(topology));
		mix(color_write_mask);
		mix(blending);
		mix(depth_write);
		mix(stencil_write);
		return result;
	}

	usz overlay_pipeline_key_hash::operator()(const overlay_pipeline_key& key) const noexcept
	{
		return static_cast<usz>(key.hash());
	}

	struct overlay_pass::impl
	{
		render_device* device = nullptr;
		memory_allocator* allocator = nullptr;
		MTLPipelineCompiler* compiler = nullptr;
		std::string label;
		std::string vertex_source;
		std::string fragment_source;
		std::string vertex_function;
		std::string fragment_function;
		std::vector<std::byte> vertex_bytes;
		std::vector<std::byte> constant_bytes;
		std::unordered_map<overlay_pipeline_key, id<MTLRenderPipelineState>,
			overlay_pipeline_key_hash> pipelines;
		std::unique_ptr<sampler> texture_sampler;
		overlay_pass_statistics stats;
		sampler_filter filter = sampler_filter::linear;
		primitive_topology topology = primitive_topology::triangle_strip;
		u32 source_texture_count = 0;
		u32 vertex_stride = 0;
		u8 color_mask = color_write_all;
		bool blending = false;
		bool depth_write = false;
		bool stencil_write = false;
		mutable std::mutex mutex;

		[[nodiscard]] id<MTLRenderPipelineState> pipeline(const overlay_pipeline_key& key)
		{
			std::lock_guard lock(mutex);
			if (const auto found = pipelines.find(key); found != pipelines.end())
			{
				++stats.pipeline_cache_hits;
				return found->second;
			}
			if (vertex_source.empty() || fragment_source.empty() ||
				vertex_function.empty() || fragment_function.empty())
			{
				fmt::throw_exception("Metal overlay pass '%s' has no shader program", label);
			}
			const std::string combined = vertex_source + "\n" + fragment_source;
			id<MTLLibrary> library = compile_library(compiler->native_compiler(), combined, label);
			MTL4LibraryFunctionDescriptor* vertex = [MTL4LibraryFunctionDescriptor new];
			vertex.library = library;
			vertex.name = native_string(vertex_function);
			MTL4LibraryFunctionDescriptor* fragment = [MTL4LibraryFunctionDescriptor new];
			fragment.library = library;
			fragment.name = native_string(fragment_function);
			MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
			descriptor.label = native_string(label);
			descriptor.vertexFunctionDescriptor = vertex;
			descriptor.fragmentFunctionDescriptor = fragment;
			descriptor.inputPrimitiveTopology = topology_class(key.topology);
			descriptor.rasterSampleCount = key.sample_count;
			if (key.color_format)
			{
				auto* attachment = descriptor.colorAttachments[0];
				attachment.pixelFormat = static_cast<MTLPixelFormat>(key.color_format);
				attachment.writeMask = native_color_mask(key.color_write_mask);
				attachment.blendingEnabled = key.blending;
				if (key.blending)
				{
					attachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
					attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
					attachment.rgbBlendOperation = MTLBlendOperationAdd;
					attachment.sourceAlphaBlendFactor = MTLBlendFactorZero;
					attachment.destinationAlphaBlendFactor = MTLBlendFactorOne;
					attachment.alphaBlendOperation = MTLBlendOperationAdd;
				}
			}
			NSError* error = nil;
			id<MTLRenderPipelineState> result = [compiler->native_compiler()
				newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
			if (!result) throw_native_error(error, fmt::format("%s pipeline compilation", label));
			pipelines.emplace(key, result);
			++stats.pipeline_builds;
			return result;
		}
	};

	overlay_pass::overlay_pass(std::string label)
		: m_impl(std::make_unique<impl>())
	{
		m_impl->label = std::move(label);
	}

	overlay_pass::~overlay_pass()
	{
		destroy();
	}

	void overlay_pass::set_shader_sources(std::string vertex_source, std::string fragment_source,
		std::string vertex_function, std::string fragment_function)
	{
		if (m_impl->device) fmt::throw_exception("Cannot replace an initialized Metal overlay shader");
		m_impl->vertex_source = std::move(vertex_source);
		m_impl->fragment_source = std::move(fragment_source);
		m_impl->vertex_function = std::move(vertex_function);
		m_impl->fragment_function = std::move(fragment_function);
	}

	void overlay_pass::set_sampler_filter(sampler_filter filter) { m_impl->filter = filter; }
	void overlay_pass::set_source_texture_count(u32 count) { m_impl->source_texture_count = count; }
	void overlay_pass::set_primitive_topology(primitive_topology topology) { m_impl->topology = topology; }
	void overlay_pass::set_color_write_mask(u8 mask) { m_impl->color_mask = mask & color_write_all; }
	void overlay_pass::set_blending(bool enabled) { m_impl->blending = enabled; }
	void overlay_pass::set_depth_write(bool enabled) { m_impl->depth_write = enabled; }
	void overlay_pass::set_stencil_write(bool enabled) { m_impl->stencil_write = enabled; }

	void overlay_pass::upload_vertex_bytes(std::span<const std::byte> bytes, u32 stride)
	{
		if (bytes.empty() || !stride || bytes.size() % stride)
			fmt::throw_exception("Invalid Metal overlay vertex payload");
		m_impl->vertex_bytes.assign(bytes.begin(), bytes.end());
		m_impl->vertex_stride = stride;
	}

	void overlay_pass::upload_constant_bytes(std::span<const std::byte> bytes)
	{
		if (bytes.empty()) fmt::throw_exception("Invalid Metal overlay constant payload");
		m_impl->constant_bytes.assign(bytes.begin(), bytes.end());
	}

	void overlay_pass::initialize(render_device& device, memory_allocator& allocator,
		MTLPipelineCompiler& compiler)
	{
		if (!device || !compiler || &compiler.owner() != &device || &allocator.device() != &device)
			fmt::throw_exception("Metal overlay pass requires matching device resources");
		if (m_impl->device && (m_impl->device != &device || m_impl->allocator != &allocator ||
			m_impl->compiler != &compiler))
		{
			fmt::throw_exception("Metal overlay pass cannot change devices while initialized");
		}
		m_impl->device = &device;
		m_impl->allocator = &allocator;
		m_impl->compiler = &compiler;
		if (m_impl->source_texture_count && !m_impl->texture_sampler)
		{
			sampler_description description;
			description.address_s = sampler_address_mode::clamp_to_edge;
			description.address_t = sampler_address_mode::clamp_to_edge;
			description.address_r = sampler_address_mode::clamp_to_edge;
			description.min_filter = m_impl->filter;
			description.mag_filter = m_impl->filter;
			description.mip_filter = sampler_mip_filter::nearest;
			description.border = border_color::opaque_black();
			m_impl->texture_sampler = std::make_unique<sampler>(device, description,
				fmt::format("%s sampler", m_impl->label));
		}
	}

	void overlay_pass::destroy()
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->pipelines.clear();
		m_impl->texture_sampler.reset();
		m_impl->vertex_bytes.clear();
		m_impl->constant_bytes.clear();
		m_impl->device = nullptr;
		m_impl->allocator = nullptr;
		m_impl->compiler = nullptr;
		m_impl->stats = {};
	}

	void overlay_pass::reclaim(u64 completed_submission)
	{
		if (!completed_submission) fmt::throw_exception("Invalid Metal overlay completion value");
		get_resource_manager().complete(completed_submission);
	}

	void overlay_pass::trim(memory_pressure pressure)
	{
		if (pressure == memory_pressure::none) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->pipelines.clear();
		if (pressure == memory_pressure::critical)
		{
			m_impl->vertex_bytes.clear();
			m_impl->constant_bytes.clear();
		}
	}

	overlay_pass::operator bool() const
	{
		return m_impl && m_impl->device && m_impl->allocator && m_impl->compiler;
	}

	overlay_pass_statistics overlay_pass::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->stats;
	}

	void overlay_pass::draw(command_buffer& command, const areau& viewport,
		const overlay_render_target& target, std::span<image_view* const> sources,
		u32 vertex_count, u32 first_vertex, u32 instance_count, u32 base_instance,
		const areau* scissor, u32 stencil_reference, u32 stencil_write_mask)
	{
		target.validate();
		if (!*this || !command.is_recording() || !vertex_count || !instance_count ||
			viewport.width() == 0 || viewport.height() == 0 ||
			viewport.x2 > target.width() || viewport.y2 > target.height() ||
			sources.size() > m_impl->source_texture_count)
		{
			fmt::throw_exception("Invalid Metal overlay draw");
		}

		for (image_view* source : sources)
		{
			if (!source) continue;
			image* resource = source->image();
			if (!resource || !*resource || !(resource->info().usage & texture_usage_shader_read))
				fmt::throw_exception("Invalid Metal overlay source texture");
			transition_image(command, *resource,
				operation_state(*resource, stage_fragment, access_shader_read), source->range(), false);
		}
		const u64 target_access = target.write_aspects & texture_aspect_color
			? access_color_write : access_depth_stencil_write;
		transition_image(command, *target.resource,
			operation_state(*target.resource, stage_fragment | stage_tile, target_access),
			target_range(target), false);
		end_active_encoder(command);

		const overlay_pipeline_key key{
			.color_format = target.write_aspects & texture_aspect_color ? target.resource->format() : 0,
			.depth_stencil_format = target.write_aspects & (texture_aspect_depth | texture_aspect_stencil)
				? target.resource->format() : 0,
			.sample_count = target.samples(),
			.topology = m_impl->topology,
			.color_write_mask = m_impl->color_mask,
			.blending = m_impl->blending,
			.depth_write = m_impl->depth_write,
			.stencil_write = m_impl->stencil_write,
		};
		id<MTLRenderPipelineState> pipeline = m_impl->pipeline(key);
		id<MTLDevice> device = m_impl->device->native_handle();
		id<MTLBuffer> vertex_buffer = make_shared_buffer(device, m_impl->vertex_bytes,
			fmt::format("%s vertices", m_impl->label));
		id<MTLBuffer> constant_buffer = make_shared_buffer(device, m_impl->constant_bytes,
			fmt::format("%s constants", m_impl->label));
		id<MTL4ArgumentTable> vertex_table = make_argument_table(device, 2, 0, 0,
			fmt::format("%s vertex arguments", m_impl->label));
		id<MTL4ArgumentTable> fragment_table = make_argument_table(device, 1,
			m_impl->source_texture_count, m_impl->source_texture_count ? 1 : 0,
			fmt::format("%s fragment arguments", m_impl->label));
		if (vertex_buffer) [vertex_table setAddress:vertex_buffer.gpuAddress atIndex:0];
		if (constant_buffer)
		{
			[vertex_table setAddress:constant_buffer.gpuAddress atIndex:1];
			[fragment_table setAddress:constant_buffer.gpuAddress atIndex:0];
		}
		for (u32 index = 0; index < sources.size(); ++index)
		{
			if (sources[index])
				[fragment_table setTexture:sources[index]->native_handle().gpuResourceID atIndex:index];
		}
		if (m_impl->source_texture_count)
			[fragment_table setSamplerState:m_impl->texture_sampler->native_handle().gpuResourceID atIndex:0];

		command.retain_native_object((__bridge void*)target.resource->native_handle(), true);
		command.retain_native_object((__bridge void*)pipeline, false);
		command.retain_native_object((__bridge void*)vertex_table, false);
		command.retain_native_object((__bridge void*)fragment_table, false);
		if (vertex_buffer) command.retain_native_object((__bridge void*)vertex_buffer, true);
		if (constant_buffer) command.retain_native_object((__bridge void*)constant_buffer, true);
		if (m_impl->texture_sampler)
			command.retain_native_object((__bridge void*)m_impl->texture_sampler->native_handle(), false);
		for (image_view* source : sources)
		{
			if (!source) continue;
			command.retain_native_object((__bridge void*)source->native_handle(), true);
			command.retain_native_object((__bridge void*)source->image()->native_handle(), true);
		}

		MTL4RenderPassDescriptor* pass = [MTL4RenderPassDescriptor new];
		pass.renderTargetWidth = target.width();
		pass.renderTargetHeight = target.height();
		pass.defaultRasterSampleCount = target.samples();
		const MTLLoadAction load = target.preserve_contents ? MTLLoadActionLoad : MTLLoadActionDontCare;
		if (target.write_aspects & texture_aspect_color)
		{
			auto* attachment = pass.colorAttachments[0];
			attachment.texture = target.resource->native_handle();
			attachment.level = target.mip_level;
			attachment.slice = target.array_slice;
			attachment.loadAction = load;
			attachment.storeAction = MTLStoreActionStore;
		}
		if (target.write_aspects & texture_aspect_depth)
		{
			pass.depthAttachment.texture = target.resource->native_handle();
			pass.depthAttachment.level = target.mip_level;
			pass.depthAttachment.slice = target.array_slice;
			pass.depthAttachment.loadAction = load;
			pass.depthAttachment.storeAction = MTLStoreActionStore;
		}
		if (target.write_aspects & texture_aspect_stencil)
		{
			pass.stencilAttachment.texture = target.resource->native_handle();
			pass.stencilAttachment.level = target.mip_level;
			pass.stencilAttachment.slice = target.array_slice;
			pass.stencilAttachment.loadAction = load;
			pass.stencilAttachment.storeAction = MTLStoreActionStore;
			if ((target.resource->aspects() & texture_aspect_depth) &&
				!(target.write_aspects & texture_aspect_depth))
			{
				pass.depthAttachment.texture = target.resource->native_handle();
				pass.depthAttachment.level = target.mip_level;
				pass.depthAttachment.slice = target.array_slice;
				pass.depthAttachment.loadAction = MTLLoadActionLoad;
				pass.depthAttachment.storeAction = MTLStoreActionStore;
			}
		}

		id<MTLDepthStencilState> depth_stencil_state = nil;
		if (m_impl->depth_write || m_impl->stencil_write)
		{
			MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
			descriptor.depthCompareFunction = MTLCompareFunctionAlways;
			descriptor.depthWriteEnabled = m_impl->depth_write;
			if (m_impl->stencil_write)
			{
				MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
				stencil.stencilCompareFunction = MTLCompareFunctionAlways;
				stencil.stencilFailureOperation = MTLStencilOperationReplace;
				stencil.depthFailureOperation = MTLStencilOperationReplace;
				stencil.depthStencilPassOperation = MTLStencilOperationReplace;
				stencil.readMask = 0xff;
				stencil.writeMask = stencil_write_mask;
				descriptor.frontFaceStencil = stencil;
				descriptor.backFaceStencil = stencil;
			}
			depth_stencil_state = [device newDepthStencilStateWithDescriptor:descriptor];
			if (!depth_stencil_state) fmt::throw_exception("Metal overlay depth/stencil state creation failed");
			command.retain_native_object((__bridge void*)depth_stencil_state, false);
		}

		id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)
			command.begin_render_encoding((__bridge void*)pass);
		[encoder setRenderPipelineState:pipeline];
		[encoder setArgumentTable:vertex_table atStages:MTLRenderStageVertex];
		[encoder setArgumentTable:fragment_table atStages:MTLRenderStageFragment];
		if (depth_stencil_state) [encoder setDepthStencilState:depth_stencil_state];
		if (m_impl->stencil_write) [encoder setStencilReferenceValue:stencil_reference];
		[encoder setViewport:MTLViewport{static_cast<f64>(viewport.x1), static_cast<f64>(viewport.y1),
			static_cast<f64>(viewport.width()), static_cast<f64>(viewport.height()), 0.0, 1.0}];
		const areau& clipping = scissor ? *scissor : viewport;
		[encoder setScissorRect:MTLScissorRect{clipping.x1, clipping.y1,
			clipping.width(), clipping.height()}];
		[encoder drawPrimitives:primitive_type(m_impl->topology) vertexStart:first_vertex
			vertexCount:vertex_count instanceCount:instance_count baseInstance:base_instance];
		barrier_plan visibility;
		visibility.scope = barrier_scope::between_encoders;
		visibility.after_stages = stage_fragment | stage_tile;
		visibility.before_stages = stage_all_gpu;
		visibility.flush_caches = true;
		visibility.end_encoder = true;
		visibility.producer_barrier = true;
		encode_barrier(command.active_native_encoder(), visibility);
		command.end_encoding();
		command.set_flag(command_reload_dynamic_state);

		std::lock_guard lock(m_impl->mutex);
		++m_impl->stats.draw_calls;
		m_impl->stats.vertices += static_cast<u64>(vertex_count) * instance_count;
	}
}
