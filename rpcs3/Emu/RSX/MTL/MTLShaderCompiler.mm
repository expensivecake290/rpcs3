#include "stdafx.h"
#include "MTLShaderCompiler.h"

#include "MTLDevice.h"
#include "MTLShaderCache.h"

#include "Utilities/File.h"

#include <utility>

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

	NSURL* make_file_url(const std::string& path)
	{
		NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
		return [NSURL fileURLWithPath:ns_path];
	}

	void* require_shader_compiler_handle(void* handle, const char* handle_name)
	{
		if (!handle)
		{
			fmt::throw_exception("Metal shader compiler %s handle is not ready", handle_name);
		}

		return handle;
	}
}

namespace rsx::metal
{
	struct shader_compiler::shader_compiler_impl
	{
		id<MTL4Compiler> m_compiler = nil;
		id<MTL4PipelineDataSetSerializer> m_pipeline_serializer = nil;
		MTL4CompilerTaskOptions* m_task_options = nil;
		id<MTL4Archive> m_archive = nil;
		std::string m_archive_path;
		std::string m_archive_metadata_error;
		std::string m_archive_load_error;
		b8 m_archive_metadata_found = false;
		b8 m_archive_metadata_invalid = false;
		b8 m_archive_loaded = false;
		b8 m_archive_without_metadata = false;
		b8 m_archive_load_failed = false;
	};

	void validate_shader_compiler_state(
		id<MTL4Compiler> compiler,
		id<MTL4PipelineDataSetSerializer> pipeline_serializer,
		MTL4CompilerTaskOptions* task_options,
		id<MTL4Archive> archive,
		const std::string& archive_path,
		const std::string& archive_metadata_error,
		const std::string& archive_load_error,
		b8 archive_metadata_found,
		b8 archive_metadata_invalid,
		b8 archive_loaded,
		b8 archive_without_metadata,
		b8 archive_load_failed)
	{
		rsx_log.trace("validate_shader_compiler_state(compiler=*0x%x, serializer=*0x%x, task_options=*0x%x, archive=*0x%x)",
			compiler,
			pipeline_serializer,
			task_options,
			archive);

		if (!compiler)
		{
			fmt::throw_exception("Metal shader compiler state requires a compiler handle");
		}

		if (!pipeline_serializer)
		{
			fmt::throw_exception("Metal shader compiler state requires a pipeline data serializer");
		}

		if (!task_options)
		{
			fmt::throw_exception("Metal shader compiler state requires compiler task options");
		}

		if (archive_path.empty())
		{
			fmt::throw_exception("Metal shader compiler state requires an archive path");
		}

		if (archive_loaded && (!archive_metadata_found || !archive))
		{
			fmt::throw_exception("Metal shader compiler loaded archive state requires validated metadata and an archive handle");
		}

		if (archive_loaded && (archive_metadata_invalid || archive_load_failed || archive_without_metadata))
		{
			fmt::throw_exception("Metal shader compiler archive state is both loaded and invalid or missing metadata");
		}

		if (archive_metadata_invalid && archive_metadata_error.empty())
		{
			fmt::throw_exception("Metal shader compiler archive metadata is invalid without an error");
		}

		if (archive_load_failed && (!archive_metadata_found || archive_load_error.empty() || archive))
		{
			fmt::throw_exception("Metal shader compiler archive load failure state is incomplete");
		}

		if (archive_without_metadata && (archive_metadata_found || archive_metadata_invalid || archive || archive_loaded))
		{
			fmt::throw_exception("Metal shader compiler archive-without-metadata state conflicts with loaded archive state");
		}

		if (archive)
		{
			if (!task_options.lookupArchives || task_options.lookupArchives.count != 1)
			{
				fmt::throw_exception("Metal shader compiler lookup archive was not attached to task options");
			}
		}
		else if (task_options.lookupArchives && task_options.lookupArchives.count)
		{
			fmt::throw_exception("Metal shader compiler task options reference lookup archives without an archive handle");
		}
	}

	shader_compiler::shader_compiler(device& metal_device, const persistent_shader_cache& cache)
		: m_impl(std::make_unique<shader_compiler_impl>())
	{
		rsx_log.notice("rsx::metal::shader_compiler::shader_compiler()");

		if (@available(macOS 26.0, *))
		{
			id<MTLDevice> device = (__bridge id<MTLDevice>)metal_device.handle();

			MTL4PipelineDataSetSerializerDescriptor* serializer_desc = [MTL4PipelineDataSetSerializerDescriptor new];
			serializer_desc.configuration =
				MTL4PipelineDataSetSerializerConfigurationCaptureDescriptors |
				MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;

			m_impl->m_pipeline_serializer = [device newPipelineDataSetSerializerWithDescriptor:serializer_desc];
			if (!m_impl->m_pipeline_serializer)
			{
				fmt::throw_exception("Metal shader compiler failed to create pipeline data set serializer");
			}

			MTL4CompilerDescriptor* compiler_desc = [MTL4CompilerDescriptor new];
			compiler_desc.label = @"RPCS3 Metal shader compiler";
			compiler_desc.pipelineDataSetSerializer = m_impl->m_pipeline_serializer;

			NSError* compiler_error = nil;
			m_impl->m_compiler = [device newCompilerWithDescriptor:compiler_desc error:&compiler_error];
			if (!m_impl->m_compiler)
			{
				const std::string error = compiler_error ? get_ns_string([compiler_error localizedDescription]) : "unknown error";
				fmt::throw_exception("Metal shader compiler creation failed: %s", error);
			}

			m_impl->m_archive_path = cache.pipeline_archive_file_path();
			pipeline_archive_metadata archive_metadata;
			std::string archive_metadata_error;
			if (cache.try_find_pipeline_archive_metadata(archive_metadata, archive_metadata_error))
			{
				m_impl->m_archive_metadata_found = true;
				NSError* archive_error = nil;
				m_impl->m_archive = [device newArchiveWithURL:make_file_url(archive_metadata.archive_path) error:&archive_error];

				if (!m_impl->m_archive)
				{
					const std::string error = archive_error ? get_ns_string([archive_error localizedDescription]) : "unknown error";
					m_impl->m_archive_load_failed = true;
					m_impl->m_archive_load_error = error;
					rsx_log.warning("Metal pipeline archive load failed for '%s': %s", archive_metadata.archive_path, error);
				}
				else
				{
					m_impl->m_archive.label = @"RPCS3 Metal pipeline archive";
					m_impl->m_archive_loaded = true;
					m_impl->m_archive_path = archive_metadata.archive_path;
				}
			}
			else if (!archive_metadata_error.empty())
			{
				m_impl->m_archive_metadata_invalid = true;
				m_impl->m_archive_metadata_error = std::move(archive_metadata_error);
				rsx_log.warning("Metal pipeline archive metadata is invalid and will not be used: %s", m_impl->m_archive_metadata_error);
			}
			else if (fs::is_file(m_impl->m_archive_path))
			{
				m_impl->m_archive_without_metadata = true;
				rsx_log.warning("Metal pipeline archive exists without valid metadata and will not be used: %s", m_impl->m_archive_path);
			}

			m_impl->m_task_options = [MTL4CompilerTaskOptions new];
			if (m_impl->m_archive)
			{
				m_impl->m_task_options.lookupArchives = @[m_impl->m_archive];
			}

			validate_shader_compiler_state(
				m_impl->m_compiler,
				m_impl->m_pipeline_serializer,
				m_impl->m_task_options,
				m_impl->m_archive,
				m_impl->m_archive_path,
				m_impl->m_archive_metadata_error,
				m_impl->m_archive_load_error,
				m_impl->m_archive_metadata_found,
				m_impl->m_archive_metadata_invalid,
				m_impl->m_archive_loaded,
				m_impl->m_archive_without_metadata,
				m_impl->m_archive_load_failed);
		}
		else
		{
			fmt::throw_exception("Metal shader compiler requires macOS 26.0 or newer");
		}
	}

	shader_compiler::~shader_compiler()
	{
		rsx_log.notice("rsx::metal::shader_compiler::~shader_compiler()");
	}

	void shader_compiler::report() const
	{
		rsx_log.notice("rsx::metal::shader_compiler::report()");
		const shader_compiler_stats compiler_stats = stats();
		rsx_log.notice("Metal 4 compiler: %s", compiler_stats.compiler_ready ? "ready" : "missing");
		rsx_log.notice("Metal 4 pipeline data serializer: %s", compiler_stats.pipeline_serializer_ready ? "ready" : "missing");
		rsx_log.notice("Metal 4 compiler task options: %s, lookup_archive=%u",
			compiler_stats.task_options_ready ? "ready" : "missing",
			static_cast<u32>(compiler_stats.lookup_archive_configured));
		rsx_log.notice("Metal 4 lookup archive: metadata=%u, metadata_invalid=%u, loaded=%u, load_failed=%u, archive_without_metadata=%u, path=%s",
			static_cast<u32>(compiler_stats.archive_metadata_found),
			static_cast<u32>(compiler_stats.archive_metadata_invalid),
			static_cast<u32>(compiler_stats.archive_loaded),
			static_cast<u32>(compiler_stats.archive_load_failed),
			static_cast<u32>(compiler_stats.archive_without_metadata),
			compiler_stats.archive_path);
		if (compiler_stats.archive_metadata_invalid)
		{
			rsx_log.warning("Metal 4 lookup archive metadata error: %s", compiler_stats.archive_metadata_error);
		}
		if (compiler_stats.archive_load_failed)
		{
			rsx_log.warning("Metal 4 lookup archive load error: %s", compiler_stats.archive_load_error);
		}
		if (compiler_stats.lookup_archive_configured && !compiler_stats.archive_loaded)
		{
			rsx_log.warning("Metal 4 compiler task options have a lookup archive configured without a loaded archive");
		}
		if (compiler_stats.archive_loaded && !compiler_stats.archive_metadata_found)
		{
			rsx_log.warning("Metal 4 lookup archive is loaded without validated archive metadata");
		}
		if (compiler_stats.archive_metadata_invalid && compiler_stats.archive_metadata_error.empty())
		{
			rsx_log.warning("Metal 4 lookup archive metadata is invalid without a recorded error");
		}
		if (compiler_stats.archive_load_failed && compiler_stats.archive_load_error.empty())
		{
			rsx_log.warning("Metal 4 lookup archive load failed without a recorded error");
		}
		if (compiler_stats.archive_loaded && compiler_stats.archive_load_failed)
		{
			rsx_log.warning("Metal 4 lookup archive is marked both loaded and failed");
		}
		if (compiler_stats.archive_without_metadata && compiler_stats.archive_metadata_found)
		{
			rsx_log.warning("Metal 4 lookup archive is marked as both metadata-backed and missing metadata");
		}
	}

	shader_compiler_stats shader_compiler::stats() const
	{
		rsx_log.trace("rsx::metal::shader_compiler::stats()");
		validate_shader_compiler_state(
			m_impl->m_compiler,
			m_impl->m_pipeline_serializer,
			m_impl->m_task_options,
			m_impl->m_archive,
			m_impl->m_archive_path,
			m_impl->m_archive_metadata_error,
			m_impl->m_archive_load_error,
			m_impl->m_archive_metadata_found,
			m_impl->m_archive_metadata_invalid,
			m_impl->m_archive_loaded,
			m_impl->m_archive_without_metadata,
			m_impl->m_archive_load_failed);

		return
		{
			.compiler_ready = m_impl->m_compiler != nil,
			.pipeline_serializer_ready = m_impl->m_pipeline_serializer != nil,
			.task_options_ready = m_impl->m_task_options != nil,
			.lookup_archive_configured = m_impl->m_task_options != nil && m_impl->m_archive != nil,
			.archive_metadata_found = m_impl->m_archive_metadata_found,
			.archive_metadata_invalid = m_impl->m_archive_metadata_invalid,
			.archive_loaded = m_impl->m_archive_loaded,
			.archive_without_metadata = m_impl->m_archive_without_metadata,
			.archive_load_failed = m_impl->m_archive_load_failed,
			.archive_path = m_impl->m_archive_path,
			.archive_metadata_error = m_impl->m_archive_metadata_error,
			.archive_load_error = m_impl->m_archive_load_error,
		};
	}

	void* shader_compiler::compiler_handle() const
	{
		rsx_log.trace("rsx::metal::shader_compiler::compiler_handle()");
		return require_shader_compiler_handle((__bridge void*)m_impl->m_compiler, "compiler");
	}

	void* shader_compiler::pipeline_serializer_handle() const
	{
		rsx_log.trace("rsx::metal::shader_compiler::pipeline_serializer_handle()");
		return require_shader_compiler_handle((__bridge void*)m_impl->m_pipeline_serializer, "pipeline serializer");
	}

	void* shader_compiler::task_options_handle() const
	{
		rsx_log.trace("rsx::metal::shader_compiler::task_options_handle()");
		return require_shader_compiler_handle((__bridge void*)m_impl->m_task_options, "task options");
	}

	void* shader_compiler::archive_handle() const
	{
		rsx_log.trace("rsx::metal::shader_compiler::archive_handle()");
		return (__bridge void*)m_impl->m_archive;
	}
}
