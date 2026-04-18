#include "stdafx.h"
#include "MTLShaderLibrary.h"

#include "MTLShaderCache.h"
#include "MTLShaderCompiler.h"
#include "MTLShaderInterface.h"

#include "Utilities/File.h"
#include "util/fnv_hash.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string_view>

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
		rsx_log.trace("dynamic_library_path(stage=%u, id=%u, source_hash=0x%llx)",
			static_cast<u32>(shader.stage), shader.id, shader.source_hash);

		return cache.library_path() + fmt::format("%llX.%s.metallib", shader.source_hash, stage_suffix(shader.stage));
	}

	b8 validate_shader_source_metadata(
		rsx::metal::persistent_shader_cache& cache,
		const char* stage,
		const rsx::metal::translated_shader& shader,
		u64 source_text_hash,
		std::string& error)
	{
		rsx_log.trace("validate_shader_source_metadata(stage=%s, id=%u, source_hash=0x%llx, source_text_hash=0x%llx)",
			stage ? stage : "<null>", shader.id, shader.source_hash, source_text_hash);

		rsx::metal::shader_source_metadata metadata;
		return cache.try_find_shader_source_metadata(
			stage,
			shader.source_hash,
			source_text_hash,
			shader.entry_point,
			shader.cache_path,
			metadata,
			error);
	}

	b8 validate_shader_completion_metadata(
		rsx::metal::persistent_shader_cache& cache,
		const char* stage,
		const rsx::metal::translated_shader& shader,
		u64 source_text_hash,
		std::string& error)
	{
		rsx_log.trace("validate_shader_completion_metadata(stage=%s, id=%u, source_hash=0x%llx, source_text_hash=0x%llx)",
			stage ? stage : "<null>", shader.id, shader.source_hash, source_text_hash);

		rsx::metal::shader_completion_metadata metadata;
		return cache.try_find_shader_completion_metadata(
			stage,
			shader.source_hash,
			source_text_hash,
			shader.entry_point,
			shader.cache_path,
			shader.pipeline_source_hash,
			shader.pipeline_entry_point,
			shader.pipeline_cache_path,
			shader.pipeline_entry_error,
			shader.pipeline_requirement_mask,
			shader.pipeline_entry_available,
			metadata,
			error);
	}

	u64 shader_source_text_hash(std::string_view source)
	{
		rsx_log.trace("shader_source_text_hash(size=0x%x)", source.size());

		usz hash = rpcs3::fnv_seed;
		for (const char c : source)
		{
			hash = rpcs3::hash64(hash, static_cast<u8>(c));
		}

		return static_cast<u64>(hash);
	}

	rsx::metal::shader_library_record make_shader_library_record(
		const rsx::metal::translated_shader& shader,
		u64 source_text_hash,
		const std::string& library_path,
		id<MTLDynamicLibrary> dynamic_library,
		b8 loaded_from_disk)
	{
		rsx_log.trace("make_shader_library_record(stage=%u, id=%u, source_hash=0x%llx, source_text_hash=0x%llx, library_path=%s, loaded_from_disk=%u)",
			static_cast<u32>(shader.stage),
			shader.id,
			shader.source_hash,
			source_text_hash,
			library_path.c_str(),
			static_cast<u32>(loaded_from_disk));

		rsx::metal::shader_library_record record =
		{
			.stage = shader.stage,
			.id = shader.id,
			.source_hash = shader.source_hash,
			.source_text_hash = source_text_hash,
			.entry_point = shader.entry_point,
			.dynamic_library_path = library_path,
			.dynamic_library_handle = (__bridge void*)dynamic_library,
			.pipeline_requirement_mask = shader.pipeline_requirement_mask,
			.pipeline_entry_available = shader.pipeline_entry_available,
			.loaded_from_disk = loaded_from_disk,
		};

		rsx::metal::validate_shader_library_record(record, true);
		return record;
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
		u32 m_memory_hits = 0;
		u32 m_disk_probes = 0;
		u32 m_disk_file_misses = 0;
		u32 m_loaded_libraries = 0;
		u32 m_compiled_libraries = 0;
		u32 m_source_metadata_misses = 0;
		u32 m_source_metadata_invalid = 0;
		u32 m_completion_metadata_misses = 0;
		u32 m_completion_metadata_invalid = 0;
		u32 m_library_metadata_misses = 0;
		u32 m_library_metadata_invalid = 0;
		u32 m_disk_load_failures = 0;
		u32 m_source_compile_failures = 0;
		u32 m_dynamic_library_failures = 0;
		u32 m_serialization_failures = 0;

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

	void validate_shader_library_record(const shader_library_record& record, b8 require_handle)
	{
		rsx_log.trace("rsx::metal::validate_shader_library_record(stage=%u, id=%u, source_hash=0x%llx, source_text_hash=0x%llx, require_handle=%u)",
			static_cast<u32>(record.stage),
			record.id,
			record.source_hash,
			record.source_text_hash,
			static_cast<u32>(require_handle));

		stage_suffix(record.stage);

		if (!record.source_hash)
		{
			fmt::throw_exception("Metal shader library record requires a non-zero source hash");
		}

		if (!record.source_text_hash)
		{
			fmt::throw_exception("Metal shader library record requires a non-zero source text hash");
		}

		if (record.entry_point.empty())
		{
			fmt::throw_exception("Metal shader library record requires an entry point");
		}

		if (record.dynamic_library_path.empty())
		{
			fmt::throw_exception("Metal shader library record requires a dynamic library path");
		}

		if (require_handle && !record.dynamic_library_handle)
		{
			fmt::throw_exception("Metal shader library record '%s' requires a dynamic library handle", record.dynamic_library_path.c_str());
		}

		validate_pipeline_entry_requirement_mask(record.pipeline_requirement_mask);

		if (record.pipeline_entry_available && record.pipeline_requirement_mask)
		{
			fmt::throw_exception("Metal shader library record '%s' is available with unsatisfied pipeline requirements: 0x%x",
				record.dynamic_library_path.c_str(),
				record.pipeline_requirement_mask);
		}

		if (!record.pipeline_entry_available && !record.pipeline_requirement_mask)
		{
			fmt::throw_exception("Metal shader library record '%s' is gated without a requirement mask", record.dynamic_library_path.c_str());
		}
	}

	std::string describe_shader_library_record(const shader_library_record& record)
	{
		rsx_log.trace("rsx::metal::describe_shader_library_record(stage=%u, id=%u, source_hash=0x%llx)",
			static_cast<u32>(record.stage), record.id, record.source_hash);

		validate_shader_library_record(record, false);

		return fmt::format("stage=%s, id=%u, source_hash=0x%llx, source_text_hash=0x%llx, entry=%s, path=%s, requirements=%s, available=%u, disk=%u",
			stage_suffix(record.stage),
			record.id,
			record.source_hash,
			record.source_text_hash,
			record.entry_point.c_str(),
			record.dynamic_library_path.c_str(),
			describe_pipeline_entry_requirements(record.pipeline_requirement_mask).c_str(),
			static_cast<u32>(record.pipeline_entry_available),
			static_cast<u32>(record.loaded_from_disk));
	}

	shader_library_record shader_library_cache::get_or_compile_dynamic_library(const translated_shader& shader)
	{
		rsx_log.notice("rsx::metal::shader_library_cache::get_or_compile_dynamic_library(stage=%u, id=%u, source_hash=0x%llx)",
			static_cast<u32>(shader.stage), shader.id, shader.source_hash);

		if (shader.source.empty())
		{
			fmt::throw_exception("Metal dynamic library compilation requires non-empty MSL source");
		}

		if (shader.entry_point.empty())
		{
			fmt::throw_exception("Metal dynamic library compilation requires a shader entry point name");
		}

		if (shader.pipeline_entry_available && shader.pipeline_requirement_mask)
		{
			fmt::throw_exception("Metal dynamic library cache received an available pipeline entry with unsatisfied requirements");
		}

		if (!shader.pipeline_entry_available)
		{
			if (!shader.pipeline_requirement_mask || shader.pipeline_entry_error.empty())
			{
				fmt::throw_exception("Metal dynamic library cache requires recorded pipeline-entry gating state");
			}

			rsx_log.warning("Metal dynamic library cache is compiling helper shader source for a gated pipeline entry: requirements=0x%x, %s",
				shader.pipeline_requirement_mask,
				shader.pipeline_entry_error.c_str());
		}

		const char* stage = stage_suffix(shader.stage);
		const u64 source_text_hash = shader_source_text_hash(shader.source);
		const std::string library_path = dynamic_library_path(m_impl->m_cache, shader);
		NSString* key = make_ns_string(library_path);

		if (id<MTLDynamicLibrary> cached_library = [m_impl->m_dynamic_libraries objectForKey:key])
		{
			m_impl->m_memory_hits++;

			return make_shader_library_record(
				shader,
				source_text_hash,
				library_path,
				cached_library,
				[m_impl->m_disk_loaded_libraries containsObject:key]);
		}

		if (@available(macOS 26.0, *))
		{
			id<MTL4Compiler> compiler = (__bridge id<MTL4Compiler>)m_impl->m_compiler.compiler_handle();
			id<MTLDynamicLibrary> dynamic_library = nil;
			std::string source_metadata_error;
			const b8 source_metadata_valid = validate_shader_source_metadata(m_impl->m_cache, stage, shader, source_text_hash, source_metadata_error);
			if (!source_metadata_valid)
			{
				if (source_metadata_error.empty())
				{
					m_impl->m_source_metadata_misses++;
					rsx_log.warning("Metal dynamic library source metadata miss for stage=%s, source_hash=0x%llx, entry=%s",
						stage, shader.source_hash, shader.entry_point.c_str());
				}
				else
				{
					m_impl->m_source_metadata_invalid++;
					rsx_log.warning("Metal dynamic library source metadata for stage=%s, source_hash=0x%llx is unusable: %s",
						stage,
						shader.source_hash,
						source_metadata_error.c_str());
				}
			}

			std::string completion_metadata_error;
			const b8 completion_metadata_valid = validate_shader_completion_metadata(m_impl->m_cache, stage, shader, source_text_hash, completion_metadata_error);
			if (!completion_metadata_valid)
			{
				if (completion_metadata_error.empty())
				{
					m_impl->m_completion_metadata_misses++;
					rsx_log.warning("Metal dynamic library completion metadata miss for stage=%s, source_hash=0x%llx, entry=%s",
						stage, shader.source_hash, shader.entry_point.c_str());
				}
				else
				{
					m_impl->m_completion_metadata_invalid++;
					rsx_log.warning("Metal dynamic library completion metadata for stage=%s, source_hash=0x%llx is unusable: %s",
						stage,
						shader.source_hash,
						completion_metadata_error.c_str());
				}
			}

			if (fs::stat_t library_stat{}; fs::get_stat(library_path, library_stat) && !library_stat.is_directory && library_stat.size)
			{
				m_impl->m_disk_probes++;

				shader_library_metadata metadata;
				const b8 metadata_ready = source_metadata_valid && completion_metadata_valid;
				std::string metadata_error;
				if (metadata_ready &&
					m_impl->m_cache.try_find_shader_library_metadata(
						stage,
						shader.source_hash,
						source_text_hash,
						shader.entry_point,
						library_path,
						shader.pipeline_requirement_mask,
						shader.pipeline_entry_available,
						metadata,
						metadata_error))
				{
					NSError* load_error = nil;
					dynamic_library = [compiler newDynamicLibraryWithURL:make_file_url(library_path) error:&load_error];
					if (!dynamic_library)
					{
						const std::string error = load_error ? get_ns_string([load_error localizedDescription]) : "unknown error";
						m_impl->m_disk_load_failures++;
						rsx_log.warning("Metal dynamic library cache miss for '%s': %s", library_path, error);
					}
					else
					{
						[m_impl->m_dynamic_libraries setObject:dynamic_library forKey:key];
						[m_impl->m_disk_loaded_libraries addObject:key];
						m_impl->m_loaded_libraries++;

						return make_shader_library_record(
							shader,
							source_text_hash,
							library_path,
							dynamic_library,
							true);
					}
				}
				else
				{
					if (metadata_ready)
					{
						if (metadata_error.empty())
						{
							m_impl->m_library_metadata_misses++;
							rsx_log.warning("Metal dynamic library cache metadata miss for '%s'", library_path);
						}
						else
						{
							m_impl->m_library_metadata_invalid++;
							rsx_log.warning("Metal dynamic library cache metadata for '%s' is unusable: %s",
								library_path,
								metadata_error.c_str());
						}
					}
					else
					{
						rsx_log.warning("Metal dynamic library disk cache skipped for '%s' because source or completion metadata is incomplete", library_path);
					}
				}
			}
			else
			{
				m_impl->m_disk_file_misses++;
				rsx_log.trace("Metal dynamic library disk cache has no file for '%s'", library_path);
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
				m_impl->m_source_compile_failures++;
				fmt::throw_exception("Metal dynamic library source compilation failed for '%s': %s", shader.entry_point, error);
			}

			NSError* dynamic_library_error = nil;
			dynamic_library = [compiler newDynamicLibrary:library error:&dynamic_library_error];
			if (!dynamic_library)
			{
				const std::string error = dynamic_library_error ? get_ns_string([dynamic_library_error localizedDescription]) : "unknown error";
				m_impl->m_dynamic_library_failures++;
				fmt::throw_exception("Metal dynamic library creation failed for '%s': %s", shader.entry_point, error);
			}

			NSError* serialize_error = nil;
			if (![dynamic_library serializeToURL:make_file_url(library_path) error:&serialize_error])
			{
				const std::string error = serialize_error ? get_ns_string([serialize_error localizedDescription]) : "unknown error";
				m_impl->m_serialization_failures++;
				fmt::throw_exception("Metal dynamic library serialization failed for '%s': %s", library_path, error);
			}

			m_impl->m_cache.store_shader_library_metadata(
				stage,
				shader.id,
				shader.source_hash,
				source_text_hash,
				shader.entry_point,
				library_path,
				shader.pipeline_requirement_mask,
				shader.pipeline_entry_available);

			[m_impl->m_dynamic_libraries setObject:dynamic_library forKey:key];
			m_impl->m_compiled_libraries++;

			return make_shader_library_record(
				shader,
				source_text_hash,
				library_path,
				dynamic_library,
				false);
		}

		fmt::throw_exception("Metal shader library compilation requires macOS 26.0 or newer");
	}

	shader_library_cache_stats shader_library_cache::stats() const
	{
		rsx_log.trace("rsx::metal::shader_library_cache::stats()");

		return
		{
			.memory_hits = m_impl->m_memory_hits,
			.disk_probes = m_impl->m_disk_probes,
			.disk_file_misses = m_impl->m_disk_file_misses,
			.loaded_libraries = m_impl->m_loaded_libraries,
			.compiled_libraries = m_impl->m_compiled_libraries,
			.source_metadata_misses = m_impl->m_source_metadata_misses,
			.source_metadata_invalid = m_impl->m_source_metadata_invalid,
			.completion_metadata_misses = m_impl->m_completion_metadata_misses,
			.completion_metadata_invalid = m_impl->m_completion_metadata_invalid,
			.library_metadata_misses = m_impl->m_library_metadata_misses,
			.library_metadata_invalid = m_impl->m_library_metadata_invalid,
			.disk_load_failures = m_impl->m_disk_load_failures,
			.source_compile_failures = m_impl->m_source_compile_failures,
			.dynamic_library_failures = m_impl->m_dynamic_library_failures,
			.serialization_failures = m_impl->m_serialization_failures,
			.retained_libraries = static_cast<u32>(m_impl->m_dynamic_libraries.count),
		};
	}

	void shader_library_cache::report() const
	{
		rsx_log.notice("rsx::metal::shader_library_cache::report()");
		const shader_library_cache_stats cache_stats = stats();
		rsx_log.notice("Metal dynamic shader libraries: memory_hits=%u, disk_probes=%u, disk_file_misses=%u, loaded=%u, compiled=%u, retained=%u",
			cache_stats.memory_hits,
			cache_stats.disk_probes,
			cache_stats.disk_file_misses,
			cache_stats.loaded_libraries,
			cache_stats.compiled_libraries,
			cache_stats.retained_libraries);
		rsx_log.notice("Metal dynamic shader library cache misses: source_metadata=%u, source_metadata_invalid=%u, completion_metadata=%u, completion_metadata_invalid=%u, library_metadata=%u, library_metadata_invalid=%u, disk_load_failures=%u",
			cache_stats.source_metadata_misses,
			cache_stats.source_metadata_invalid,
			cache_stats.completion_metadata_misses,
			cache_stats.completion_metadata_invalid,
			cache_stats.library_metadata_misses,
			cache_stats.library_metadata_invalid,
			cache_stats.disk_load_failures);
		rsx_log.notice("Metal dynamic shader library failures: source_compile=%u, dynamic_library=%u, serialization=%u",
			cache_stats.source_compile_failures,
			cache_stats.dynamic_library_failures,
			cache_stats.serialization_failures);
	}
}
