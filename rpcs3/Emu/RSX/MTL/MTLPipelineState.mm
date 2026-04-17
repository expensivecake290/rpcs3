#include "stdafx.h"
#include "MTLPipelineState.h"

#include "MTLPipelineCache.h"
#include "MTLShaderCache.h"
#include "MTLShaderCompiler.h"

#include "Utilities/File.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace
{
	std::string get_ns_string(NSString* value)
	{
		if (!value)
		{
			return {};
		}

		const char* text = [value UTF8String];
		return text ? std::string(text) : std::string();
	}

	NSString* make_ns_string(const std::string& value)
	{
		return [NSString stringWithUTF8String:value.c_str()];
	}

	NSString* make_label(const rsx::metal::render_pipeline_desc& desc)
	{
		if (!desc.label.empty())
		{
			return make_ns_string(desc.label);
		}

		return make_ns_string(fmt::format("RPCS3 Metal render pipeline 0x%llx", desc.pipeline_hash));
	}

	NSString* make_label(const rsx::metal::mesh_pipeline_desc& desc)
	{
		if (!desc.label.empty())
		{
			return make_ns_string(desc.label);
		}

		return make_ns_string(fmt::format("RPCS3 Metal mesh pipeline 0x%llx", desc.pipeline_hash));
	}

	void validate_shader_source(const char* stage, const rsx::metal::render_pipeline_shader& shader)
	{
		if (!shader.source_hash)
		{
			fmt::throw_exception("Metal %s pipeline shader requires a non-zero source hash", stage);
		}

		if (shader.source.empty())
		{
			fmt::throw_exception("Metal %s pipeline shader requires MSL source", stage);
		}

		if (shader.entry_point.empty())
		{
			fmt::throw_exception("Metal %s pipeline shader requires an entry point", stage);
		}
	}

	b8 has_shader_source(const rsx::metal::render_pipeline_shader& shader)
	{
		return shader.source_hash || !shader.source.empty() || !shader.entry_point.empty();
	}

	MTLSize make_mtl_size(const rsx::metal::mesh_threadgroup_size& size)
	{
		return MTLSizeMake(size.width, size.height, size.depth);
	}

	void persist_pipeline_source(const rsx::metal::persistent_shader_cache& cache, const char* stage, const rsx::metal::render_pipeline_shader& shader)
	{
		const std::string path = cache.msl_path() + fmt::format("pipeline_%s_%llX.msl", stage, shader.source_hash);

		if (fs::stat_t source_stat{}; fs::get_stat(path, source_stat) && !source_stat.is_directory && source_stat.size)
		{
			return;
		}

		if (!fs::write_file(path, fs::rewrite, shader.source))
		{
			fmt::throw_exception("Metal pipeline source cache write failed for '%s' (%s)", path.c_str(), fs::g_tls_error);
		}
	}

	NSArray<id<MTLDynamicLibrary>>* make_linked_libraries(const std::vector<rsx::metal::shader_library_record>& libraries)
	{
		NSMutableArray<id<MTLDynamicLibrary>>* linked_libraries = [NSMutableArray arrayWithCapacity:libraries.size()];

		for (const rsx::metal::shader_library_record& library : libraries)
		{
			id<MTLDynamicLibrary> dynamic_library = (__bridge id<MTLDynamicLibrary>)library.dynamic_library_handle;
			if (!dynamic_library)
			{
				fmt::throw_exception("Metal pipeline dependency '%s' is missing a dynamic library handle", library.dynamic_library_path.c_str());
			}

			[linked_libraries addObject:dynamic_library];
		}

		return linked_libraries.count ? linked_libraries : nil;
	}

	id<MTLLibrary> compile_executable_library(id<MTL4Compiler> compiler, const char* stage, const rsx::metal::render_pipeline_shader& shader, NSArray<id<MTLDynamicLibrary>>* linked_libraries)
	{
		MTLCompileOptions* options = [MTLCompileOptions new];
		options.languageVersion = MTLLanguageVersion4_0;
		options.libraryType = MTLLibraryTypeExecutable;
		options.libraries = linked_libraries;
		options.mathMode = MTLMathModeSafe;
		options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
		options.optimizationLevel = MTLLibraryOptimizationLevelDefault;
		options.preserveInvariance = YES;

		MTL4LibraryDescriptor* library_desc = [MTL4LibraryDescriptor new];
		library_desc.name = make_ns_string(fmt::format("RPCS3 Metal %s pipeline entry 0x%llx", stage, shader.source_hash));
		library_desc.source = make_ns_string(shader.source);
		library_desc.options = options;

		NSError* error = nil;
		id<MTLLibrary> library = [compiler newLibraryWithDescriptor:library_desc error:&error];
		if (!library)
		{
			const std::string message = error ? get_ns_string([error localizedDescription]) : "unknown error";
			fmt::throw_exception("Metal executable %s library compilation failed for '%s': %s", stage, shader.entry_point.c_str(), message.c_str());
		}

		return library;
	}

	MTL4LibraryFunctionDescriptor* make_function_descriptor(id<MTLLibrary> library, const std::string& entry_point)
	{
		MTL4LibraryFunctionDescriptor* function_desc = [MTL4LibraryFunctionDescriptor new];
		function_desc.name = make_ns_string(entry_point);
		function_desc.library = library;
		return function_desc;
	}
}

namespace rsx::metal
{
	struct render_pipeline_cache::render_pipeline_cache_impl
	{
		shader_compiler& m_compiler;
		persistent_shader_cache& m_cache;
		pipeline_cache& m_pipeline_cache;
		NSMutableDictionary<NSNumber*, id<MTLRenderPipelineState>>* m_render_pipelines = nil;
		NSMutableDictionary<NSNumber*, id<MTLRenderPipelineState>>* m_mesh_pipelines = nil;
		u32 m_compiled_pipelines = 0;
		u32 m_compiled_mesh_pipelines = 0;
		u32 m_cache_hits = 0;
		u32 m_mesh_cache_hits = 0;

		render_pipeline_cache_impl(shader_compiler& compiler, persistent_shader_cache& cache, pipeline_cache& pipelines)
			: m_compiler(compiler)
			, m_cache(cache)
			, m_pipeline_cache(pipelines)
		{
		}
	};

	render_pipeline_cache::render_pipeline_cache(shader_compiler& compiler, persistent_shader_cache& cache, pipeline_cache& pipelines)
		: m_impl(std::make_unique<render_pipeline_cache_impl>(compiler, cache, pipelines))
	{
		rsx_log.notice("rsx::metal::render_pipeline_cache::render_pipeline_cache()");

		if (@available(macOS 26.0, *))
		{
			m_impl->m_render_pipelines = [NSMutableDictionary dictionary];
			m_impl->m_mesh_pipelines = [NSMutableDictionary dictionary];
		}
		else
		{
			fmt::throw_exception("Metal render pipeline cache requires macOS 26.0 or newer");
		}
	}

	render_pipeline_cache::~render_pipeline_cache()
	{
		rsx_log.notice("rsx::metal::render_pipeline_cache::~render_pipeline_cache()");
	}

	render_pipeline_record render_pipeline_cache::get_or_compile_render_pipeline(const render_pipeline_desc& desc)
	{
		rsx_log.notice("rsx::metal::render_pipeline_cache::get_or_compile_render_pipeline(pipeline_hash=0x%llx, color_pixel_format=%u, sample_count=%u)",
			desc.pipeline_hash, desc.color_pixel_format, desc.raster_sample_count);

		if (!desc.pipeline_hash)
		{
			fmt::throw_exception("Metal render pipeline requires a non-zero pipeline hash");
		}

		if (!desc.color_pixel_format)
		{
			fmt::throw_exception("Metal render pipeline requires a color pixel format");
		}

		if (!desc.raster_sample_count)
		{
			fmt::throw_exception("Metal render pipeline requires a non-zero sample count");
		}

		validate_shader_source("vertex", desc.vertex);

		if (desc.rasterization_enabled)
		{
			validate_shader_source("fragment", desc.fragment);
		}
		else if (!desc.fragment.source.empty() || !desc.fragment.entry_point.empty() || desc.fragment.source_hash)
		{
			fmt::throw_exception("Metal rasterization-disabled pipeline must not provide a fragment shader");
		}

		NSNumber* key = [NSNumber numberWithUnsignedLongLong:desc.pipeline_hash];
		id<MTLRenderPipelineState> cached_pipeline = [m_impl->m_render_pipelines objectForKey:key];
		if (cached_pipeline)
		{
			m_impl->m_cache_hits++;

			return
			{
				.pipeline_hash = desc.pipeline_hash,
				.pipeline_handle = (__bridge void*)cached_pipeline,
				.cached = true,
			};
		}

		if (@available(macOS 26.0, *))
		{
			persist_pipeline_source(m_impl->m_cache, "vp", desc.vertex);
			if (desc.rasterization_enabled)
			{
				persist_pipeline_source(m_impl->m_cache, "fp", desc.fragment);
			}

			id<MTL4Compiler> compiler = (__bridge id<MTL4Compiler>)m_impl->m_compiler.compiler_handle();
			MTL4CompilerTaskOptions* task_options = (__bridge MTL4CompilerTaskOptions*)m_impl->m_compiler.task_options_handle();
			NSArray<id<MTLDynamicLibrary>>* linked_libraries = make_linked_libraries(desc.linked_libraries);

			id<MTLLibrary> vertex_library = compile_executable_library(compiler, "vertex", desc.vertex, linked_libraries);
			id<MTLLibrary> fragment_library = nil;
			if (desc.rasterization_enabled)
			{
				fragment_library = compile_executable_library(compiler, "fragment", desc.fragment, linked_libraries);
			}

			MTL4RenderPipelineDescriptor* pipeline_desc = [MTL4RenderPipelineDescriptor new];
			pipeline_desc.label = make_label(desc);
			pipeline_desc.vertexFunctionDescriptor = make_function_descriptor(vertex_library, desc.vertex.entry_point);
			pipeline_desc.fragmentFunctionDescriptor = desc.rasterization_enabled ? make_function_descriptor(fragment_library, desc.fragment.entry_point) : nil;
			pipeline_desc.rasterSampleCount = desc.raster_sample_count;
			pipeline_desc.rasterizationEnabled = desc.rasterization_enabled;
			pipeline_desc.inputPrimitiveTopology = static_cast<MTLPrimitiveTopologyClass>(desc.input_primitive_topology);
			pipeline_desc.alphaToCoverageState = desc.alpha_to_coverage ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
			pipeline_desc.alphaToOneState = desc.alpha_to_one ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
			pipeline_desc.supportIndirectCommandBuffers = desc.support_indirect_command_buffers ?
				MTL4IndirectCommandBufferSupportStateEnabled :
				MTL4IndirectCommandBufferSupportStateDisabled;
			pipeline_desc.colorAttachments[0].pixelFormat = static_cast<MTLPixelFormat>(desc.color_pixel_format);

			MTL4RenderPipelineDynamicLinkingDescriptor* dynamic_linking_desc = nil;
			if (linked_libraries.count)
			{
				dynamic_linking_desc = [MTL4RenderPipelineDynamicLinkingDescriptor new];
				dynamic_linking_desc.vertexLinkingDescriptor.preloadedLibraries = linked_libraries;
				if (desc.rasterization_enabled)
				{
					dynamic_linking_desc.fragmentLinkingDescriptor.preloadedLibraries = linked_libraries;
				}
			}

			NSError* error = nil;
			id<MTLRenderPipelineState> pipeline = [compiler newRenderPipelineStateWithDescriptor:pipeline_desc
				dynamicLinkingDescriptor:dynamic_linking_desc
				compilerTaskOptions:task_options
				error:&error];

			if (!pipeline)
			{
				const std::string message = error ? get_ns_string([error localizedDescription]) : "unknown error";
				const std::string label = get_ns_string(pipeline_desc.label);
				fmt::throw_exception("Metal render pipeline compilation failed for '%s': %s", label.c_str(), message.c_str());
			}

			[m_impl->m_render_pipelines setObject:pipeline forKey:key];
			m_impl->m_compiled_pipelines++;
			m_impl->m_pipeline_cache.record_pipeline_compilation();

			return
			{
				.pipeline_hash = desc.pipeline_hash,
				.pipeline_handle = (__bridge void*)pipeline,
				.cached = false,
			};
		}

		fmt::throw_exception("Metal render pipeline compilation requires macOS 26.0 or newer");
	}

	render_pipeline_record render_pipeline_cache::get_or_compile_mesh_pipeline(const mesh_pipeline_desc& desc)
	{
		rsx_log.notice("rsx::metal::render_pipeline_cache::get_or_compile_mesh_pipeline(pipeline_hash=0x%llx, color_pixel_format=%u, sample_count=%u)",
			desc.pipeline_hash, desc.color_pixel_format, desc.raster_sample_count);

		if (!desc.pipeline_hash)
		{
			fmt::throw_exception("Metal mesh pipeline requires a non-zero pipeline hash");
		}

		if (!desc.color_pixel_format)
		{
			fmt::throw_exception("Metal mesh pipeline requires a color pixel format");
		}

		if (!desc.raster_sample_count)
		{
			fmt::throw_exception("Metal mesh pipeline requires a non-zero sample count");
		}

		validate_shader_source("mesh", desc.mesh);

		const b8 has_object_stage = has_shader_source(desc.object);
		if (has_object_stage)
		{
			validate_shader_source("object", desc.object);
		}

		if (desc.rasterization_enabled)
		{
			validate_shader_source("fragment", desc.fragment);
		}
		else if (has_shader_source(desc.fragment))
		{
			fmt::throw_exception("Metal rasterization-disabled mesh pipeline must not provide a fragment shader");
		}

		NSNumber* key = [NSNumber numberWithUnsignedLongLong:desc.pipeline_hash];
		id<MTLRenderPipelineState> cached_pipeline = [m_impl->m_mesh_pipelines objectForKey:key];
		if (cached_pipeline)
		{
			m_impl->m_mesh_cache_hits++;

			return
			{
				.pipeline_hash = desc.pipeline_hash,
				.pipeline_handle = (__bridge void*)cached_pipeline,
				.cached = true,
			};
		}

		if (@available(macOS 26.0, *))
		{
			if (has_object_stage)
			{
				persist_pipeline_source(m_impl->m_cache, "obj", desc.object);
			}

			persist_pipeline_source(m_impl->m_cache, "mesh", desc.mesh);
			if (desc.rasterization_enabled)
			{
				persist_pipeline_source(m_impl->m_cache, "fp", desc.fragment);
			}

			id<MTL4Compiler> compiler = (__bridge id<MTL4Compiler>)m_impl->m_compiler.compiler_handle();
			MTL4CompilerTaskOptions* task_options = (__bridge MTL4CompilerTaskOptions*)m_impl->m_compiler.task_options_handle();
			NSArray<id<MTLDynamicLibrary>>* linked_libraries = make_linked_libraries(desc.linked_libraries);

			id<MTLLibrary> object_library = nil;
			if (has_object_stage)
			{
				object_library = compile_executable_library(compiler, "object", desc.object, linked_libraries);
			}

			id<MTLLibrary> mesh_library = compile_executable_library(compiler, "mesh", desc.mesh, linked_libraries);
			id<MTLLibrary> fragment_library = nil;
			if (desc.rasterization_enabled)
			{
				fragment_library = compile_executable_library(compiler, "fragment", desc.fragment, linked_libraries);
			}

			MTL4MeshRenderPipelineDescriptor* pipeline_desc = [MTL4MeshRenderPipelineDescriptor new];
			pipeline_desc.label = make_label(desc);
			pipeline_desc.objectFunctionDescriptor = has_object_stage ? make_function_descriptor(object_library, desc.object.entry_point) : nil;
			pipeline_desc.meshFunctionDescriptor = make_function_descriptor(mesh_library, desc.mesh.entry_point);
			pipeline_desc.fragmentFunctionDescriptor = desc.rasterization_enabled ? make_function_descriptor(fragment_library, desc.fragment.entry_point) : nil;
			pipeline_desc.maxTotalThreadsPerObjectThreadgroup = desc.max_total_threads_per_object_threadgroup;
			pipeline_desc.maxTotalThreadsPerMeshThreadgroup = desc.max_total_threads_per_mesh_threadgroup;
			pipeline_desc.requiredThreadsPerObjectThreadgroup = make_mtl_size(desc.required_threads_per_object_threadgroup);
			pipeline_desc.requiredThreadsPerMeshThreadgroup = make_mtl_size(desc.required_threads_per_mesh_threadgroup);
			pipeline_desc.objectThreadgroupSizeIsMultipleOfThreadExecutionWidth = desc.object_threadgroup_size_is_multiple_of_thread_execution_width;
			pipeline_desc.meshThreadgroupSizeIsMultipleOfThreadExecutionWidth = desc.mesh_threadgroup_size_is_multiple_of_thread_execution_width;
			pipeline_desc.payloadMemoryLength = desc.payload_memory_length;
			pipeline_desc.maxTotalThreadgroupsPerMeshGrid = desc.max_total_threadgroups_per_mesh_grid;
			pipeline_desc.rasterSampleCount = desc.raster_sample_count;
			pipeline_desc.rasterizationEnabled = desc.rasterization_enabled;
			pipeline_desc.alphaToCoverageState = desc.alpha_to_coverage ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
			pipeline_desc.alphaToOneState = desc.alpha_to_one ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
			pipeline_desc.supportIndirectCommandBuffers = desc.support_indirect_command_buffers ?
				MTL4IndirectCommandBufferSupportStateEnabled :
				MTL4IndirectCommandBufferSupportStateDisabled;
			pipeline_desc.colorAttachments[0].pixelFormat = static_cast<MTLPixelFormat>(desc.color_pixel_format);

			MTL4RenderPipelineDynamicLinkingDescriptor* dynamic_linking_desc = nil;
			if (linked_libraries.count)
			{
				dynamic_linking_desc = [MTL4RenderPipelineDynamicLinkingDescriptor new];
				if (has_object_stage)
				{
					dynamic_linking_desc.objectLinkingDescriptor.preloadedLibraries = linked_libraries;
				}

				dynamic_linking_desc.meshLinkingDescriptor.preloadedLibraries = linked_libraries;
				if (desc.rasterization_enabled)
				{
					dynamic_linking_desc.fragmentLinkingDescriptor.preloadedLibraries = linked_libraries;
				}
			}

			NSError* error = nil;
			id<MTLRenderPipelineState> pipeline = [compiler newRenderPipelineStateWithDescriptor:pipeline_desc
				dynamicLinkingDescriptor:dynamic_linking_desc
				compilerTaskOptions:task_options
				error:&error];

			if (!pipeline)
			{
				const std::string message = error ? get_ns_string([error localizedDescription]) : "unknown error";
				const std::string label = get_ns_string(pipeline_desc.label);
				fmt::throw_exception("Metal mesh pipeline compilation failed for '%s': %s", label.c_str(), message.c_str());
			}

			[m_impl->m_mesh_pipelines setObject:pipeline forKey:key];
			m_impl->m_compiled_mesh_pipelines++;
			m_impl->m_pipeline_cache.record_pipeline_compilation();

			return
			{
				.pipeline_hash = desc.pipeline_hash,
				.pipeline_handle = (__bridge void*)pipeline,
				.cached = false,
			};
		}

		fmt::throw_exception("Metal mesh pipeline compilation requires macOS 26.0 or newer");
	}

	void render_pipeline_cache::report() const
	{
		rsx_log.notice("rsx::metal::render_pipeline_cache::report()");
		rsx_log.notice("Metal render pipelines: compiled=%u, cache_hits=%u, retained=%u",
			m_impl->m_compiled_pipelines,
			m_impl->m_cache_hits,
			static_cast<u32>(m_impl->m_render_pipelines.count));
		rsx_log.notice("Metal mesh pipelines: compiled=%u, cache_hits=%u, retained=%u",
			m_impl->m_compiled_mesh_pipelines,
			m_impl->m_mesh_cache_hits,
			static_cast<u32>(m_impl->m_mesh_pipelines.count));
	}
}
