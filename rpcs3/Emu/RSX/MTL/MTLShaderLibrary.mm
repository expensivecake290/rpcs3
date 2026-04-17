#include "stdafx.h"
#include "MTLShaderLibrary.h"

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

	NSURL* make_file_url(const std::string& path)
	{
		return [NSURL fileURLWithPath:make_ns_string(path)];
	}

	const char* stage_suffix(rsx::metal::shader_stage stage)
	{
		switch (stage)
		{
		case rsx::metal::shader_stage::vertex: return "vp";
		case rsx::metal::shader_stage::fragment: return "fp";
		case rsx::metal::shader_stage::mesh: return "mesh";
		default:
			fmt::throw_exception("Unknown Metal shader stage: %u", static_cast<u32>(stage));
		}
	}

	std::string dynamic_library_path(const rsx::metal::persistent_shader_cache& cache, const rsx::metal::translated_shader& shader)
	{
		rsx_log.trace("dynamic_library_path(stage=%u, id=%u, source_hash=0x%x)",
			static_cast<u32>(shader.stage), shader.id, shader.source_hash);

		return cache.library_path() + fmt::format("%llX.%s.metallib", shader.source_hash, stage_suffix(shader.stage));
	}
}

namespace rsx::metal
{
	struct shader_library_cache::shader_library_cache_impl
	{
		shader_compiler& m_compiler;
		persistent_shader_cache& m_cache;
		NSMutableDictionary<NSString*, id<MTLDynamicLibrary>>* m_dynamic_libraries = nil;
		NSMutableSet<NSString*>* m_disk_loaded_libraries = nil;
		u32 m_loaded_libraries = 0;
		u32 m_compiled_libraries = 0;

		shader_library_cache_impl(shader_compiler& compiler, persistent_shader_cache& cache)
			: m_compiler(compiler)
			, m_cache(cache)
		{
		}
	};

	shader_library_cache::shader_library_cache(shader_compiler& compiler, persistent_shader_cache& cache)
		: m_impl(std::make_unique<shader_library_cache_impl>(compiler, cache))
	{
		rsx_log.notice("rsx::metal::shader_library_cache::shader_library_cache()");

		if (@available(macOS 26.0, *))
		{
			m_impl->m_dynamic_libraries = [NSMutableDictionary dictionary];
			m_impl->m_disk_loaded_libraries = [NSMutableSet set];
		}
		else
		{
			fmt::throw_exception("Metal shader library cache requires macOS 26.0 or newer");
		}
	}

	shader_library_cache::~shader_library_cache()
	{
		rsx_log.notice("rsx::metal::shader_library_cache::~shader_library_cache()");
	}

	shader_library_record shader_library_cache::get_or_compile_dynamic_library(const translated_shader& shader)
	{
		rsx_log.notice("rsx::metal::shader_library_cache::get_or_compile_dynamic_library(stage=%u, id=%u, source_hash=0x%x)",
			static_cast<u32>(shader.stage), shader.id, shader.source_hash);

		if (shader.source.empty())
		{
			fmt::throw_exception("Metal dynamic library compilation requires non-empty MSL source");
		}

		if (shader.entry_point.empty())
		{
			fmt::throw_exception("Metal dynamic library compilation requires a shader entry point name");
		}

		const std::string library_path = dynamic_library_path(m_impl->m_cache, shader);
		NSString* key = make_ns_string(library_path);

		if (id<MTLDynamicLibrary> cached_library = [m_impl->m_dynamic_libraries objectForKey:key])
		{
			return
			{
				.stage = shader.stage,
				.id = shader.id,
				.source_hash = shader.source_hash,
				.entry_point = shader.entry_point,
				.dynamic_library_path = library_path,
				.dynamic_library_handle = (__bridge void*)cached_library,
				.loaded_from_disk = [m_impl->m_disk_loaded_libraries containsObject:key],
			};
		}

		if (@available(macOS 26.0, *))
		{
			id<MTL4Compiler> compiler = (__bridge id<MTL4Compiler>)m_impl->m_compiler.compiler_handle();
			id<MTLDynamicLibrary> dynamic_library = nil;

			if (fs::stat_t library_stat{}; fs::get_stat(library_path, library_stat) && !library_stat.is_directory && library_stat.size)
			{
				NSError* load_error = nil;
				dynamic_library = [compiler newDynamicLibraryWithURL:make_file_url(library_path) error:&load_error];
				if (!dynamic_library)
				{
					const std::string error = load_error ? get_ns_string([load_error localizedDescription]) : "unknown error";
					rsx_log.warning("Metal dynamic library cache miss for '%s': %s", library_path, error);
				}
				else
				{
					[m_impl->m_dynamic_libraries setObject:dynamic_library forKey:key];
					[m_impl->m_disk_loaded_libraries addObject:key];
					m_impl->m_loaded_libraries++;

					return
					{
						.stage = shader.stage,
						.id = shader.id,
						.source_hash = shader.source_hash,
						.entry_point = shader.entry_point,
						.dynamic_library_path = library_path,
						.dynamic_library_handle = (__bridge void*)dynamic_library,
						.loaded_from_disk = true,
					};
				}
			}

			MTLCompileOptions* options = [MTLCompileOptions new];
			options.languageVersion = MTLLanguageVersion4_0;
			options.libraryType = MTLLibraryTypeDynamic;
			options.installName = make_ns_string(library_path);
			options.mathMode = MTLMathModeSafe;
			options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
			options.optimizationLevel = MTLLibraryOptimizationLevelDefault;
			options.preserveInvariance = YES;

			MTL4LibraryDescriptor* library_desc = [MTL4LibraryDescriptor new];
			library_desc.name = make_ns_string(shader.entry_point);
			library_desc.source = make_ns_string(shader.source);
			library_desc.options = options;

			NSError* library_error = nil;
			id<MTLLibrary> library = [compiler newLibraryWithDescriptor:library_desc error:&library_error];
			if (!library)
			{
				const std::string error = library_error ? get_ns_string([library_error localizedDescription]) : "unknown error";
				fmt::throw_exception("Metal dynamic library source compilation failed for '%s': %s", shader.entry_point, error);
			}

			NSError* dynamic_library_error = nil;
			dynamic_library = [compiler newDynamicLibrary:library error:&dynamic_library_error];
			if (!dynamic_library)
			{
				const std::string error = dynamic_library_error ? get_ns_string([dynamic_library_error localizedDescription]) : "unknown error";
				fmt::throw_exception("Metal dynamic library creation failed for '%s': %s", shader.entry_point, error);
			}

			NSError* serialize_error = nil;
			if (![dynamic_library serializeToURL:make_file_url(library_path) error:&serialize_error])
			{
				const std::string error = serialize_error ? get_ns_string([serialize_error localizedDescription]) : "unknown error";
				fmt::throw_exception("Metal dynamic library serialization failed for '%s': %s", library_path, error);
			}

			[m_impl->m_dynamic_libraries setObject:dynamic_library forKey:key];
			m_impl->m_compiled_libraries++;

			return
			{
				.stage = shader.stage,
				.id = shader.id,
				.source_hash = shader.source_hash,
				.entry_point = shader.entry_point,
				.dynamic_library_path = library_path,
				.dynamic_library_handle = (__bridge void*)dynamic_library,
				.loaded_from_disk = false,
			};
		}

		fmt::throw_exception("Metal shader library compilation requires macOS 26.0 or newer");
	}

	void shader_library_cache::report() const
	{
		rsx_log.notice("rsx::metal::shader_library_cache::report()");
		rsx_log.notice("Metal dynamic shader libraries: loaded=%u, compiled=%u, retained=%u",
			m_impl->m_loaded_libraries,
			m_impl->m_compiled_libraries,
			static_cast<u32>(m_impl->m_dynamic_libraries.count));
	}
}
