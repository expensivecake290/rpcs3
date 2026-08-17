#include "stdafx.h"
#include "MTLProgramPipeline.h"
#include "MTLCommonPipelineLayout.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <utility>

namespace mtl
{
	namespace
	{
		constexpr std::string_view vertex_entry_point = "rsx_vertex_main";
		constexpr std::string_view fragment_entry_point = "rsx_fragment_main";

		NSString* native_string(std::string_view value)
		{
			return [NSString stringWithUTF8String:std::string(value).c_str()];
		}

		[[noreturn]] void throw_native_error(NSError* error, std::string_view operation)
		{
			const std::string description = error.localizedDescription.UTF8String ?: "unknown error";
			fmt::throw_exception("%s failed: %s", operation, description);
		}

		MTLPrimitiveTopologyClass native_topology(primitive_topology value)
		{
			switch (value)
			{
			case primitive_topology::point: return MTLPrimitiveTopologyClassPoint;
			case primitive_topology::line:
			case primitive_topology::line_strip: return MTLPrimitiveTopologyClassLine;
			case primitive_topology::triangle:
			case primitive_topology::triangle_strip: return MTLPrimitiveTopologyClassTriangle;
			}
			fmt::throw_exception("Invalid Metal primitive topology");
		}

		MTLBlendFactor native_blend_factor(blend_factor value)
		{
			switch (value)
			{
			case blend_factor::zero: return MTLBlendFactorZero;
			case blend_factor::one: return MTLBlendFactorOne;
			case blend_factor::source_color: return MTLBlendFactorSourceColor;
			case blend_factor::one_minus_source_color: return MTLBlendFactorOneMinusSourceColor;
			case blend_factor::source_alpha: return MTLBlendFactorSourceAlpha;
			case blend_factor::one_minus_source_alpha: return MTLBlendFactorOneMinusSourceAlpha;
			case blend_factor::destination_color: return MTLBlendFactorDestinationColor;
			case blend_factor::one_minus_destination_color: return MTLBlendFactorOneMinusDestinationColor;
			case blend_factor::destination_alpha: return MTLBlendFactorDestinationAlpha;
			case blend_factor::one_minus_destination_alpha: return MTLBlendFactorOneMinusDestinationAlpha;
			case blend_factor::source_alpha_saturated: return MTLBlendFactorSourceAlphaSaturated;
			case blend_factor::blend_color: return MTLBlendFactorBlendColor;
			case blend_factor::one_minus_blend_color: return MTLBlendFactorOneMinusBlendColor;
			case blend_factor::blend_alpha: return MTLBlendFactorBlendAlpha;
			case blend_factor::one_minus_blend_alpha: return MTLBlendFactorOneMinusBlendAlpha;
			case blend_factor::source1_color: return MTLBlendFactorSource1Color;
			case blend_factor::one_minus_source1_color: return MTLBlendFactorOneMinusSource1Color;
			case blend_factor::source1_alpha: return MTLBlendFactorSource1Alpha;
			case blend_factor::one_minus_source1_alpha: return MTLBlendFactorOneMinusSource1Alpha;
			}
			fmt::throw_exception("Invalid Metal blend factor");
		}

		MTLBlendOperation native_blend_operation(blend_operation value)
		{
			switch (value)
			{
			case blend_operation::add: return MTLBlendOperationAdd;
			case blend_operation::subtract: return MTLBlendOperationSubtract;
			case blend_operation::reverse_subtract: return MTLBlendOperationReverseSubtract;
			case blend_operation::minimum: return MTLBlendOperationMin;
			case blend_operation::maximum: return MTLBlendOperationMax;
			}
			fmt::throw_exception("Invalid Metal blend operation");
		}

		MTLColorWriteMask native_color_write_mask(u8 value)
		{
			MTLColorWriteMask result = MTLColorWriteMaskNone;
			if (value & color_write_red) result |= MTLColorWriteMaskRed;
			if (value & color_write_green) result |= MTLColorWriteMaskGreen;
			if (value & color_write_blue) result |= MTLColorWriteMaskBlue;
			if (value & color_write_alpha) result |= MTLColorWriteMaskAlpha;
			if (value & ~color_write_all) fmt::throw_exception("Invalid Metal color write mask 0x%x", value);
			return result;
		}

		MTLCompareFunction native_compare_function(compare_function value)
		{
			switch (value)
			{
			case compare_function::never: return MTLCompareFunctionNever;
			case compare_function::less: return MTLCompareFunctionLess;
			case compare_function::equal: return MTLCompareFunctionEqual;
			case compare_function::less_equal: return MTLCompareFunctionLessEqual;
			case compare_function::greater: return MTLCompareFunctionGreater;
			case compare_function::not_equal: return MTLCompareFunctionNotEqual;
			case compare_function::greater_equal: return MTLCompareFunctionGreaterEqual;
			case compare_function::always: return MTLCompareFunctionAlways;
			}
			fmt::throw_exception("Invalid Metal comparison function");
		}

		MTLStencilOperation native_stencil_operation(stencil_operation value)
		{
			switch (value)
			{
			case stencil_operation::keep: return MTLStencilOperationKeep;
			case stencil_operation::zero: return MTLStencilOperationZero;
			case stencil_operation::replace: return MTLStencilOperationReplace;
			case stencil_operation::increment_clamp: return MTLStencilOperationIncrementClamp;
			case stencil_operation::decrement_clamp: return MTLStencilOperationDecrementClamp;
			case stencil_operation::invert: return MTLStencilOperationInvert;
			case stencil_operation::increment_wrap: return MTLStencilOperationIncrementWrap;
			case stencil_operation::decrement_wrap: return MTLStencilOperationDecrementWrap;
			}
			fmt::throw_exception("Invalid Metal stencil operation");
		}

		MTLCullMode native_cull_mode(cull_mode value)
		{
			switch (value)
			{
			case cull_mode::none: return MTLCullModeNone;
			case cull_mode::front: return MTLCullModeFront;
			case cull_mode::back: return MTLCullModeBack;
			}
			fmt::throw_exception("Invalid Metal cull mode");
		}

		MTLWinding native_winding(front_face value)
		{
			return value == front_face::clockwise ? MTLWindingClockwise : MTLWindingCounterClockwise;
		}

		MTLTriangleFillMode native_fill_mode(triangle_fill_mode value)
		{
			return value == triangle_fill_mode::fill ? MTLTriangleFillModeFill : MTLTriangleFillModeLines;
		}

		MTLDepthClipMode native_depth_clip_mode(depth_clip_mode value)
		{
			return value == depth_clip_mode::clip ? MTLDepthClipModeClip : MTLDepthClipModeClamp;
		}

		MTLStencilDescriptor* native_stencil_descriptor(const stencil_face_pipeline_state& state)
		{
			MTLStencilDescriptor* descriptor = [MTLStencilDescriptor new];
			descriptor.stencilCompareFunction = native_compare_function(state.compare);
			descriptor.stencilFailureOperation = native_stencil_operation(state.stencil_fail);
			descriptor.depthFailureOperation = native_stencil_operation(state.depth_fail);
			descriptor.depthStencilPassOperation = native_stencil_operation(state.depth_pass);
			descriptor.readMask = state.read_mask;
			descriptor.writeMask = state.write_mask;
			return descriptor;
		}

		bool equal_binding(const argument_buffer_binding& left, const argument_buffer_binding& right)
		{
			return left.resource == right.resource && left.gpu_address == right.gpu_address &&
				left.offset == right.offset && left.length == right.length &&
				left.attribute_stride == right.attribute_stride && left.access == right.access;
		}

		bool equal_binding(const argument_texture_binding& left, const argument_texture_binding& right)
		{
			return left.resource == right.resource && left.access == right.access;
		}

		bool equal_binding(const argument_sampler_binding& left, const argument_sampler_binding& right)
		{
			return left.resource == right.resource;
		}

		void validate_reference(const program_binding_reference& binding, const argument_table_layout& layout,
			msl_shader_stage expected_stage)
		{
			if (binding.stage != expected_stage || binding.index == umax || binding.name.empty())
			{
				fmt::throw_exception("Invalid Metal required program binding");
			}
			const u32 count = binding.resource == argument_binding_class::buffer ? layout.buffer_count :
				binding.resource == argument_binding_class::texture ? layout.texture_count : layout.sampler_count;
			if (binding.index >= count)
			{
				fmt::throw_exception("Metal required binding '%s' index %u exceeds its argument-table layout",
					binding.name, binding.index);
			}
		}

		template <typename Input>
		program_binding_reference make_reference(msl_shader_stage stage, const Input& input)
		{
			if (!input.binding)
			{
				fmt::throw_exception("Metal program input '%s' has no binding location", input.name);
			}
			const u8 expected_stage = stage == msl_shader_stage::vertex ? argument_stage_vertex : argument_stage_fragment;
			if ((input.binding.stages & expected_stage) == 0)
			{
				fmt::throw_exception("Metal program input '%s' has incompatible stage visibility", input.name);
			}
			return {stage, input.binding.type, input.binding.index, input.texture_unit, input.name};
		}
	}

	void graphics_pipeline_configuration::validate() const
	{
		graphics_pipeline_state validated = state;
		validated.render.vertex_function = {1, 1, 0, 0};
		if (validated.render.rasterization_enabled)
		{
			validated.render.fragment_function = {1, 1, 0, 0};
		}
		const u32 samples = validated.render.multisample.sample_count;
		if (samples != 1 && samples != 2 && samples != 4 && samples != 8)
		{
			fmt::throw_exception("Metal graphics pipeline sample count %u is invalid", samples);
		}
		validated.validate();
	}

	u64 graphics_pipeline_configuration::signature() const
	{
		validate();
		u64 result = state.pipeline_cache_hash();
		detail::hash_combine(result, state.dynamic_state_hash());
		return result;
	}

	void compute_pipeline_configuration::validate() const
	{
		if (!library || function_name.empty())
		{
			fmt::throw_exception("Metal compute pipeline requires a library and function name");
		}
		layout.validate();
		for (const program_binding_reference& binding : required_bindings)
		{
			validate_reference(binding, layout, msl_shader_stage::compute);
		}
	}

	void native_graphics_stage_configuration::validate(msl_shader_stage expected_stage) const
	{
		if ((expected_stage != msl_shader_stage::vertex && expected_stage != msl_shader_stage::fragment) ||
			!library || function_name.empty() || !guest_program_hash || !source_hash)
		{
			fmt::throw_exception("Invalid native Metal graphics shader stage");
		}
		layout.validate();
		for (const program_binding_reference& binding : required_bindings)
		{
			validate_reference(binding, layout, expected_stage);
		}
	}

	bool resource_dirty_state::any() const
	{
		return buffers || textures[0] || textures[1] || samplers || dynamic_offsets;
	}

	void resource_dirty_state::clear()
	{
		*this = {};
	}

	struct MTLProgramPipeline::impl
	{
		struct stage_resources
		{
			std::unique_ptr<argument_table> table;
			std::vector<argument_buffer_binding> buffers;
			std::vector<argument_texture_binding> textures;
			std::vector<argument_sampler_binding> samplers;
			std::vector<u64> dynamic_offsets;
			resource_dirty_state dirty;

			void adopt(std::unique_ptr<argument_table> source)
			{
				if (!source || !*source) fmt::throw_exception("Cannot adopt an empty Metal argument table");
				const argument_table_layout layout = source->layout();
				table = std::move(source);
				buffers.resize(layout.buffer_count);
				textures.resize(layout.texture_count);
				samplers.resize(layout.sampler_count);
				dynamic_offsets.resize(layout.buffer_count);
			}

			void create(const render_device& device, const argument_table_layout& layout,
				u8 stages, std::string_view label)
			{
				auto result = std::make_unique<argument_table>();
				result->create(device, layout, stages, label);
				adopt(std::move(result));
			}
		};

		program_pipeline_kind pipeline_kind = program_pipeline_kind::graphics;
		const render_device* owner = nullptr;
		id<MTLRenderPipelineState> render_state = nil;
		id<MTLComputePipelineState> compute_state = nil;
		id<MTLDepthStencilState> depth_stencil_state = nil;
		graphics_pipeline_state graphics_state;
		stage_resources vertex;
		stage_resources fragment;
		stage_resources compute;
		std::vector<program_binding_reference> required;
		program_pipeline_statistics counters;
		bool is_linked = false;

		stage_resources& resources(msl_shader_stage stage)
		{
			if (stage == msl_shader_stage::vertex && pipeline_kind == program_pipeline_kind::graphics && vertex.table)
				return vertex;
			if (stage == msl_shader_stage::fragment && pipeline_kind == program_pipeline_kind::graphics && fragment.table)
				return fragment;
			if (stage == msl_shader_stage::compute && pipeline_kind == program_pipeline_kind::compute && compute.table)
				return compute;
			fmt::throw_exception("Metal shader stage is incompatible with this program pipeline");
		}

		const stage_resources& resources(msl_shader_stage stage) const
		{
			return const_cast<impl*>(this)->resources(stage);
		}

		bool is_bound(const program_binding_reference& binding) const
		{
			const stage_resources& stage = resources(binding.stage);
			switch (binding.resource)
			{
			case argument_binding_class::buffer: return binding.index < stage.buffers.size() && bool(stage.buffers[binding.index]);
			case argument_binding_class::texture: return binding.index < stage.textures.size() && bool(stage.textures[binding.index]);
			case argument_binding_class::sampler: return binding.index < stage.samplers.size() && bool(stage.samplers[binding.index]);
			}
			return false;
		}
	};

	MTLProgramPipeline::MTLProgramPipeline()
		: m_impl(std::make_unique<impl>())
	{
	}

	MTLProgramPipeline::~MTLProgramPipeline() = default;
	MTLProgramPipeline::MTLProgramPipeline(MTLProgramPipeline&&) noexcept = default;
	MTLProgramPipeline& MTLProgramPipeline::operator=(MTLProgramPipeline&&) noexcept = default;

	void MTLProgramPipeline::create_graphics(render_device& device, const MTLVertexProgram& vertex,
		const MTLFragmentProgram& fragment, const graphics_pipeline_configuration& configuration,
		compiler_handle compiler, pipeline_archive_handle lookup_archive)
	{
		if (!vertex || !fragment || !vertex.library() || !fragment.library())
		{
			fmt::throw_exception("Metal graphics pipeline requires compiled vertex and fragment programs");
		}
		const u64 vertex_hash = vertex.metadata().source_hash ? vertex.metadata().source_hash : 1;
		const u64 fragment_hash = fragment.metadata().source_hash ? fragment.metadata().source_hash : 1;
		native_graphics_stage_configuration vertex_stage;
		vertex_stage.library = vertex.library();
		vertex_stage.function_name = vertex_entry_point;
		vertex_stage.layout = MTLCommonPipelineLayout::vertex_definition().layout;
		vertex_stage.guest_program_hash = vertex_hash;
		vertex_stage.source_hash = vertex_hash;
		for (const vertex_program_input& input : vertex.inputs())
		{
			MTLCommonPipelineLayout::validate_binding(msl_shader_stage::vertex, input.binding);
			vertex_stage.required_bindings.push_back(make_reference(msl_shader_stage::vertex, input));
		}
		native_graphics_stage_configuration fragment_stage;
		fragment_stage.library = fragment.library();
		fragment_stage.function_name = fragment_entry_point;
		fragment_stage.layout = MTLCommonPipelineLayout::fragment_definition().layout;
		fragment_stage.guest_program_hash = static_cast<u64>(fragment.id()) + 1;
		fragment_stage.source_hash = fragment_hash;
		for (const fragment_program_input& input : fragment.inputs())
		{
			MTLCommonPipelineLayout::validate_binding(msl_shader_stage::fragment, input.binding);
			fragment_stage.required_bindings.push_back(make_reference(msl_shader_stage::fragment, input));
		}
		create_graphics_native(device, vertex_stage, fragment_stage, configuration, compiler, lookup_archive);
	}

	void MTLProgramPipeline::create_graphics_native(render_device& device,
		const native_graphics_stage_configuration& vertex,
		const native_graphics_stage_configuration& fragment,
		const graphics_pipeline_configuration& configuration,
		compiler_handle compiler, pipeline_archive_handle lookup_archive)
	{
		configuration.validate();
		vertex.validate(msl_shader_stage::vertex);
		fragment.validate(msl_shader_stage::fragment);
		compiler = compiler ? compiler : device.compiler();
		if (!device || !compiler)
			fmt::throw_exception("Metal native graphics pipeline requires an initialized device and compiler");

		auto next = std::make_unique<impl>();
		next->pipeline_kind = program_pipeline_kind::graphics;
		next->owner = &device;
		next->graphics_state = configuration.state;
		next->graphics_state.render.vertex_function = {vertex.guest_program_hash, vertex.source_hash, 0, 0};
		next->graphics_state.render.fragment_function = {fragment.guest_program_hash, fragment.source_hash, 0, 0};
		u64 binding_signature = vertex.layout.signature();
		detail::hash_combine(binding_signature, fragment.layout.signature());
		next->graphics_state.render.binding_layout_signature = binding_signature;
		next->graphics_state.validate();

		MTL4LibraryFunctionDescriptor* vertex_function = [MTL4LibraryFunctionDescriptor new];
		vertex_function.library = vertex.library;
		vertex_function.name = native_string(vertex.function_name);
		MTL4LibraryFunctionDescriptor* fragment_function = [MTL4LibraryFunctionDescriptor new];
		fragment_function.library = fragment.library;
		fragment_function.name = native_string(fragment.function_name);
		MTL4RenderPipelineDescriptor* descriptor = [MTL4RenderPipelineDescriptor new];
		descriptor.label = native_string(configuration.label.empty() ? "RPCS3 graphics pipeline" : configuration.label);
		descriptor.vertexFunctionDescriptor = vertex_function;
		descriptor.fragmentFunctionDescriptor = fragment_function;
		descriptor.inputPrimitiveTopology = native_topology(next->graphics_state.render.topology);
		descriptor.rasterSampleCount = next->graphics_state.render.multisample.sample_count;
		descriptor.alphaToCoverageState = next->graphics_state.render.multisample.alpha_to_coverage ?
			MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
		descriptor.alphaToOneState = next->graphics_state.render.multisample.alpha_to_one ?
			MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
		descriptor.rasterizationEnabled = next->graphics_state.render.rasterization_enabled;
		descriptor.supportIndirectCommandBuffers = next->graphics_state.render.support_indirect_commands ?
			MTL4IndirectCommandBufferSupportStateEnabled : MTL4IndirectCommandBufferSupportStateDisabled;
		for (u32 index = 0; index < next->graphics_state.render.color_attachment_count; ++index)
		{
			const color_attachment_pipeline_state& source = next->graphics_state.render.color_attachments[index];
			MTL4RenderPipelineColorAttachmentDescriptor* destination = descriptor.colorAttachments[index];
			destination.pixelFormat = static_cast<MTLPixelFormat>(source.pixel_format);
			destination.blendingState = source.blend_enabled ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
			destination.sourceRGBBlendFactor = native_blend_factor(source.source_rgb);
			destination.destinationRGBBlendFactor = native_blend_factor(source.destination_rgb);
			destination.rgbBlendOperation = native_blend_operation(source.rgb_operation);
			destination.sourceAlphaBlendFactor = native_blend_factor(source.source_alpha);
			destination.destinationAlphaBlendFactor = native_blend_factor(source.destination_alpha);
			destination.alphaBlendOperation = native_blend_operation(source.alpha_operation);
			destination.writeMask = native_color_write_mask(source.write_mask);
		}

		NSError* native_error = nil;
		MTL4CompilerTaskOptions* task_options = nil;
		if (lookup_archive)
		{
			task_options = [MTL4CompilerTaskOptions new];
			task_options.lookupArchives = @[lookup_archive];
		}
		next->render_state = [compiler newRenderPipelineStateWithDescriptor:descriptor
			compilerTaskOptions:task_options error:&native_error];
		if (!next->render_state) throw_native_error(native_error, "Metal graphics pipeline compilation");

		MTLDepthStencilDescriptor* depth_descriptor = [MTLDepthStencilDescriptor new];
		depth_descriptor.label = descriptor.label;
		depth_descriptor.depthCompareFunction = next->graphics_state.depth_stencil.depth_test_enabled ?
			native_compare_function(next->graphics_state.depth_stencil.depth_compare) : MTLCompareFunctionAlways;
		depth_descriptor.depthWriteEnabled = next->graphics_state.depth_stencil.depth_write_enabled;
		if (next->graphics_state.depth_stencil.stencil_test_enabled)
		{
			depth_descriptor.frontFaceStencil = native_stencil_descriptor(next->graphics_state.depth_stencil.front);
			depth_descriptor.backFaceStencil = native_stencil_descriptor(next->graphics_state.depth_stencil.back);
		}
		next->depth_stencil_state = [device.native_handle() newDepthStencilStateWithDescriptor:depth_descriptor];
		if (!next->depth_stencil_state)
		{
			fmt::throw_exception("Metal depth/stencil state creation failed");
		}

		const std::string label = configuration.label.empty() ? "RPCS3 graphics pipeline" : configuration.label;
		next->vertex.create(device, vertex.layout, argument_stage_vertex, label + " vertex arguments");
		next->fragment.create(device, fragment.layout, argument_stage_fragment, label + " fragment arguments");
		next->required = vertex.required_bindings;
		next->required.insert(next->required.end(), fragment.required_bindings.begin(), fragment.required_bindings.end());
		next->is_linked = true;
		m_impl = std::move(next);
	}

	void MTLProgramPipeline::create_compute(render_device& device, const compute_pipeline_configuration& configuration,
		compiler_handle compiler, pipeline_archive_handle lookup_archive)
	{
		configuration.validate();
		compiler = compiler ? compiler : device.compiler();
		if (!device || !compiler) fmt::throw_exception("Metal compute pipeline requires an initialized device");
		if (configuration.maximum_threads_per_threadgroup > device.info().limits.max_threads_per_threadgroup)
		{
			fmt::throw_exception("Metal compute pipeline threadgroup limit exceeds the device limit");
		}

		auto next = std::make_unique<impl>();
		next->pipeline_kind = program_pipeline_kind::compute;
		next->owner = &device;
		MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
		function.library = configuration.library;
		function.name = native_string(configuration.function_name);
		MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
		descriptor.label = native_string(configuration.label.empty() ? "RPCS3 compute pipeline" : configuration.label);
		descriptor.computeFunctionDescriptor = function;
		if (configuration.maximum_threads_per_threadgroup)
		{
			descriptor.maxTotalThreadsPerThreadgroup = configuration.maximum_threads_per_threadgroup;
		}
		descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth =
			configuration.threadgroup_size_is_multiple_of_execution_width;
		NSError* native_error = nil;
		MTL4CompilerTaskOptions* task_options = nil;
		if (lookup_archive)
		{
			task_options = [MTL4CompilerTaskOptions new];
			task_options.lookupArchives = @[lookup_archive];
		}
		next->compute_state = [compiler newComputePipelineStateWithDescriptor:descriptor
			compilerTaskOptions:task_options error:&native_error];
		if (!next->compute_state) throw_native_error(native_error, "Metal compute pipeline compilation");

		const std::string label = configuration.label.empty() ? "RPCS3 compute pipeline" : configuration.label;
		MTLCommonPipelineLayout common_layout;
		common_layout.create(device);
		next->compute.adopt(common_layout.create_compute_table(configuration.layout, label + " arguments"));
		next->required = configuration.required_bindings;
		next->is_linked = true;
		m_impl = std::move(next);
	}

	std::unique_ptr<MTLProgramPipeline> MTLProgramPipeline::create_binding_instance() const
	{
		if (!m_impl || !m_impl->is_linked || !m_impl->owner)
			fmt::throw_exception("Cannot instantiate bindings from an empty Metal program pipeline");
		auto result = std::make_unique<MTLProgramPipeline>();
		auto next = std::make_unique<impl>();
		next->pipeline_kind = m_impl->pipeline_kind;
		next->owner = m_impl->owner;
		next->render_state = m_impl->render_state;
		next->compute_state = m_impl->compute_state;
		next->depth_stencil_state = m_impl->depth_stencil_state;
		next->graphics_state = m_impl->graphics_state;
		next->required = m_impl->required;
		if (next->pipeline_kind == program_pipeline_kind::graphics)
		{
			next->vertex.create(*next->owner, m_impl->vertex.table->layout(), argument_stage_vertex,
				"RPCS3 cached graphics vertex arguments");
			next->fragment.create(*next->owner, m_impl->fragment.table->layout(), argument_stage_fragment,
				"RPCS3 cached graphics fragment arguments");
		}
		else
		{
			next->compute.create(*next->owner, m_impl->compute.table->layout(), argument_stage_compute,
				"RPCS3 cached compute pipeline arguments");
		}
		next->is_linked = true;
		result->m_impl = std::move(next);
		return result;
	}

	void MTLProgramPipeline::destroy()
	{
		m_impl = std::make_unique<impl>();
	}

	void MTLProgramPipeline::set_buffer(msl_shader_stage stage, u32 index, const argument_buffer_binding& binding)
	{
		auto& resources = m_impl->resources(stage);
		if (index >= resources.buffers.size()) fmt::throw_exception("Metal pipeline buffer index %u is out of range", index);
		if (equal_binding(resources.buffers[index], binding)) return;
		resources.table->set_buffer(index, binding);
		resources.buffers[index] = binding;
		if (!binding && resources.dynamic_offsets[index])
		{
			resources.dynamic_offsets[index] = 0;
			resources.dirty.dynamic_offsets = true;
		}
		resources.dirty.buffers |= 1u << index;
		m_impl->counters.binding_mutations++;
	}

	void MTLProgramPipeline::set_buffer(msl_shader_stage stage, u32 index, const buffer& resource,
		u64 offset, u64 length, u32 attribute_stride, argument_access access)
	{
		if (!resource || !length || !resource.in_range(offset, length))
			fmt::throw_exception("Invalid Metal pipeline buffer range");
		set_buffer(stage, index, {resource.native_handle(), resource.gpu_address(), offset, length, attribute_stride, access});
	}

	void MTLProgramPipeline::clear_buffer(msl_shader_stage stage, u32 index)
	{
		set_buffer(stage, index, {});
	}

	void MTLProgramPipeline::set_texture(msl_shader_stage stage, u32 index, const argument_texture_binding& binding)
	{
		auto& resources = m_impl->resources(stage);
		if (index >= resources.textures.size()) fmt::throw_exception("Metal pipeline texture index %u is out of range", index);
		if (equal_binding(resources.textures[index], binding)) return;
		resources.table->set_texture(index, binding);
		resources.textures[index] = binding;
		resources.dirty.textures[index / 64] |= 1ull << (index % 64);
		m_impl->counters.binding_mutations++;
	}

	void MTLProgramPipeline::set_texture(msl_shader_stage stage, u32 index, const image_view& resource,
		argument_access access)
	{
		if (!resource) fmt::throw_exception("Cannot bind an empty Metal image view");
		set_texture(stage, index, {resource.native_handle(), access});
	}

	void MTLProgramPipeline::set_texture(msl_shader_stage stage, u32 index, const buffer_view& resource,
		argument_access access)
	{
		if (!resource) fmt::throw_exception("Cannot bind an empty Metal buffer view");
		set_texture(stage, index, {resource.native_handle(), access});
	}

	void MTLProgramPipeline::clear_texture(msl_shader_stage stage, u32 index)
	{
		set_texture(stage, index, argument_texture_binding{});
	}

	void MTLProgramPipeline::set_sampler(msl_shader_stage stage, u32 index, const argument_sampler_binding& binding)
	{
		auto& resources = m_impl->resources(stage);
		if (index >= resources.samplers.size()) fmt::throw_exception("Metal pipeline sampler index %u is out of range", index);
		if (equal_binding(resources.samplers[index], binding)) return;
		resources.table->set_sampler(index, binding);
		resources.samplers[index] = binding;
		resources.dirty.samplers |= static_cast<u16>(1u << index);
		m_impl->counters.binding_mutations++;
	}

	void MTLProgramPipeline::set_sampler(msl_shader_stage stage, u32 index, const sampler& resource)
	{
		if (!resource) fmt::throw_exception("Cannot bind an empty Metal sampler");
		set_sampler(stage, index, {resource.native_handle()});
	}

	void MTLProgramPipeline::clear_sampler(msl_shader_stage stage, u32 index)
	{
		set_sampler(stage, index, argument_sampler_binding{});
	}

	void MTLProgramPipeline::set_dynamic_offset(msl_shader_stage stage, u32 buffer_index, u64 offset)
	{
		auto& resources = m_impl->resources(stage);
		if (buffer_index >= resources.dynamic_offsets.size())
			fmt::throw_exception("Metal pipeline dynamic buffer index %u is out of range", buffer_index);
		if (resources.dynamic_offsets[buffer_index] == offset) return;
		resources.table->set_dynamic_offset(buffer_index, offset);
		resources.dynamic_offsets[buffer_index] = offset;
		resources.dirty.buffers |= 1u << buffer_index;
		resources.dirty.dynamic_offsets = true;
		m_impl->counters.binding_mutations++;
	}

	void MTLProgramPipeline::clear_dynamic_offsets(msl_shader_stage stage)
	{
		auto& resources = m_impl->resources(stage);
		if (std::none_of(resources.dynamic_offsets.begin(), resources.dynamic_offsets.end(), [](u64 value) { return value != 0; }))
			return;
		resources.table->clear_dynamic_offsets();
		for (u32 index = 0; index < resources.dynamic_offsets.size(); ++index)
		{
			if (resources.dynamic_offsets[index]) resources.dirty.buffers |= 1u << index;
		}
		std::fill(resources.dynamic_offsets.begin(), resources.dynamic_offsets.end(), 0);
		resources.dirty.dynamic_offsets = true;
		m_impl->counters.binding_mutations++;
	}

	void MTLProgramPipeline::apply_bindings()
	{
		if (!m_impl || !m_impl->is_linked) fmt::throw_exception("Cannot apply bindings for an empty Metal program pipeline");
		auto apply = [&](impl::stage_resources& resources)
		{
			if (!resources.table || !resources.dirty.any()) return;
			resources.table->apply();
			resources.dirty.clear();
			m_impl->counters.binding_applies++;
		};
		apply(m_impl->vertex);
		apply(m_impl->fragment);
		apply(m_impl->compute);
	}

	void MTLProgramPipeline::bind(command_buffer& command)
	{
		if (!m_impl || !m_impl->is_linked || !command.is_recording())
			fmt::throw_exception("Metal program pipeline binding requires a linked pipeline and active recording");
		validate_required_bindings();
		if (m_impl->pipeline_kind == program_pipeline_kind::graphics)
		{
			if (command.active_encoder() != encoder_kind::render)
				fmt::throw_exception("Metal graphics pipeline requires an active render encoder");
			id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)command.active_native_encoder();
			[encoder setRenderPipelineState:m_impl->render_state];
			[encoder setDepthStencilState:m_impl->depth_stencil_state];
			const dynamic_pipeline_state& dynamic = m_impl->graphics_state.dynamic;
			[encoder setCullMode:native_cull_mode(dynamic.cull)];
			[encoder setFrontFacingWinding:native_winding(dynamic.winding)];
			[encoder setTriangleFillMode:native_fill_mode(dynamic.fill)];
			[encoder setDepthClipMode:native_depth_clip_mode(dynamic.depth_clip)];
			[encoder setDepthBias:dynamic.depth_bias_enabled ? dynamic.depth_bias : 0.f
				slopeScale:dynamic.depth_bias_enabled ? dynamic.depth_bias_slope : 0.f
				clamp:dynamic.depth_bias_enabled ? dynamic.depth_bias_clamp : 0.f];
			[encoder setBlendColorRed:dynamic.blend_color[0] green:dynamic.blend_color[1]
				blue:dynamic.blend_color[2] alpha:dynamic.blend_color[3]];
			[encoder setStencilFrontReferenceValue:dynamic.stencil_front_reference
				backReferenceValue:dynamic.stencil_back_reference];
			[encoder setDepthTestMinBound:m_impl->graphics_state.depth_stencil.depth_bounds_enabled ?
				dynamic.minimum_depth_bounds : 0.f maxBound:m_impl->graphics_state.depth_stencil.depth_bounds_enabled ?
				dynamic.maximum_depth_bounds : 1.f];
			const bool vertex_dirty = m_impl->vertex.dirty.any();
			const bool fragment_dirty = m_impl->fragment.dirty.any();
			m_impl->vertex.table->bind(command);
			m_impl->fragment.table->bind(command);
			if (vertex_dirty) m_impl->counters.binding_applies++;
			if (fragment_dirty) m_impl->counters.binding_applies++;
			m_impl->vertex.dirty.clear();
			m_impl->fragment.dirty.clear();
			command.retain_native_object((__bridge void*)m_impl->render_state, false);
			command.retain_native_object((__bridge void*)m_impl->depth_stencil_state, false);
		}
		else
		{
			if (command.active_encoder() != encoder_kind::compute)
				fmt::throw_exception("Metal compute pipeline requires an active compute encoder");
			id<MTL4ComputeCommandEncoder> encoder = (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder();
			[encoder setComputePipelineState:m_impl->compute_state];
			const bool compute_dirty = m_impl->compute.dirty.any();
			m_impl->compute.table->bind(command);
			if (compute_dirty) m_impl->counters.binding_applies++;
			m_impl->compute.dirty.clear();
			command.retain_native_object((__bridge void*)m_impl->compute_state, false);
		}
		m_impl->counters.binds++;
	}

	std::vector<program_binding_reference> MTLProgramPipeline::missing_required_bindings() const
	{
		std::vector<program_binding_reference> result;
		if (!m_impl || !m_impl->is_linked) return result;
		for (const program_binding_reference& binding : m_impl->required)
		{
			if (!m_impl->is_bound(binding)) result.push_back(binding);
		}
		return result;
	}

	void MTLProgramPipeline::validate_required_bindings()
	{
		const auto missing = missing_required_bindings();
		if (!missing.empty())
		{
			m_impl->counters.required_binding_failures++;
			fmt::throw_exception("Metal program pipeline is missing required %s binding '%s' at index %u",
				missing.front().resource == argument_binding_class::buffer ? "buffer" :
				missing.front().resource == argument_binding_class::texture ? "texture" : "sampler",
				missing.front().name, missing.front().index);
		}
	}

	MTLProgramPipeline::operator bool() const
	{
		return m_impl && m_impl->is_linked;
	}

	program_pipeline_kind MTLProgramPipeline::kind() const
	{
		if (!m_impl || !m_impl->is_linked) fmt::throw_exception("Program pipeline kind requested from an empty Metal pipeline");
		return m_impl->pipeline_kind;
	}

	bool MTLProgramPipeline::linked() const
	{
		return m_impl && m_impl->is_linked;
	}

	render_pipeline_handle MTLProgramPipeline::render_pipeline() const
	{
		return m_impl ? m_impl->render_state : nil;
	}

	compute_pipeline_handle MTLProgramPipeline::compute_pipeline() const
	{
		return m_impl ? m_impl->compute_state : nil;
	}

	argument_table* MTLProgramPipeline::vertex_arguments()
	{
		return m_impl && m_impl->vertex.table ? m_impl->vertex.table.get() : nullptr;
	}

	argument_table* MTLProgramPipeline::fragment_arguments()
	{
		return m_impl && m_impl->fragment.table ? m_impl->fragment.table.get() : nullptr;
	}

	argument_table* MTLProgramPipeline::compute_arguments()
	{
		return m_impl && m_impl->compute.table ? m_impl->compute.table.get() : nullptr;
	}

	std::span<const program_binding_reference> MTLProgramPipeline::required_bindings() const
	{
		return m_impl ? std::span<const program_binding_reference>(m_impl->required) : std::span<const program_binding_reference>{};
	}

	std::optional<program_binding_reference> MTLProgramPipeline::find_binding(
		msl_shader_stage stage, std::string_view name) const
	{
		if (!m_impl) return std::nullopt;
		const auto found = std::find_if(m_impl->required.begin(), m_impl->required.end(), [&](const auto& binding)
		{
			return binding.stage == stage && binding.name == name;
		});
		return found == m_impl->required.end() ? std::nullopt : std::optional<program_binding_reference>(*found);
	}

	const resource_dirty_state& MTLProgramPipeline::vertex_dirty_state() const
	{
		static const resource_dirty_state empty;
		return m_impl ? m_impl->vertex.dirty : empty;
	}

	const resource_dirty_state& MTLProgramPipeline::fragment_dirty_state() const
	{
		static const resource_dirty_state empty;
		return m_impl ? m_impl->fragment.dirty : empty;
	}

	const resource_dirty_state& MTLProgramPipeline::compute_dirty_state() const
	{
		static const resource_dirty_state empty;
		return m_impl ? m_impl->compute.dirty : empty;
	}

	program_pipeline_statistics MTLProgramPipeline::statistics() const
	{
		return m_impl ? m_impl->counters : program_pipeline_statistics{};
	}
}
