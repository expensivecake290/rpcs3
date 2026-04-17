#include "stdafx.h"
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

	NSURL* make_file_url(const std::string& path)
	{
		NSString* ns_path = [NSString stringWithUTF8String:path.c_str()];
		return [NSURL fileURLWithPath:ns_path];
	}
}

namespace rsx::metal
{
	struct pipeline_cache::pipeline_cache_impl
	{
		shader_compiler& m_compiler;
		persistent_shader_cache& m_cache;
		u32 m_dirty_pipeline_count = 0;
		u32 m_flushed_pipeline_count = 0;

		pipeline_cache_impl(shader_compiler& compiler, persistent_shader_cache& cache)
			: m_compiler(compiler)
			, m_cache(cache)
		{
		}
	};

	pipeline_cache::pipeline_cache(shader_compiler& compiler, persistent_shader_cache& cache)
		: m_impl(std::make_unique<pipeline_cache_impl>(compiler, cache))
	{
		rsx_log.notice("rsx::metal::pipeline_cache::pipeline_cache()");
	}

	pipeline_cache::~pipeline_cache()
	{
		rsx_log.notice("rsx::metal::pipeline_cache::~pipeline_cache()");
	}

	void pipeline_cache::record_pipeline_compilation()
	{
		rsx_log.trace("rsx::metal::pipeline_cache::record_pipeline_compilation()");
		m_impl->m_dirty_pipeline_count++;
	}

	void pipeline_cache::flush()
	{
		rsx_log.notice("rsx::metal::pipeline_cache::flush()");

		if (!m_impl->m_dirty_pipeline_count)
		{
			rsx_log.trace("Metal pipeline cache flush skipped because no Metal pipeline states were compiled");
			return;
		}

		if (@available(macOS 26.0, *))
		{
			id<MTL4PipelineDataSetSerializer> serializer =
				(__bridge id<MTL4PipelineDataSetSerializer>)m_impl->m_compiler.pipeline_serializer_handle();

			if (!serializer)
			{
				fmt::throw_exception("Metal pipeline cache flush requires a pipeline data set serializer");
			}

			NSError* script_error = nil;
			NSData* pipeline_script = [serializer serializeAsPipelinesScriptWithError:&script_error];
			if (!pipeline_script)
			{
				const std::string error = script_error ? get_ns_string([script_error localizedDescription]) : "unknown error";
				fmt::throw_exception("Metal pipeline cache failed to serialize pipeline script: %s", error);
			}

			const std::string script_path = m_impl->m_cache.pipeline_script_file_path();
			if (!fs::write_file(script_path, fs::rewrite, [pipeline_script bytes], static_cast<usz>([pipeline_script length])))
			{
				fmt::throw_exception("Metal pipeline cache failed to write pipeline script '%s' (%s)", script_path, fs::g_tls_error);
			}

			NSError* archive_error = nil;
			const std::string archive_path = m_impl->m_cache.pipeline_archive_file_path();
			if (![serializer serializeAsArchiveAndFlushToURL:make_file_url(archive_path) error:&archive_error])
			{
				const std::string error = archive_error ? get_ns_string([archive_error localizedDescription]) : "unknown error";
				fmt::throw_exception("Metal pipeline cache failed to serialize pipeline archive '%s': %s", archive_path, error);
			}

			m_impl->m_cache.store_pipeline_archive_metadata(
				script_path,
				archive_path,
				m_impl->m_flushed_pipeline_count + m_impl->m_dirty_pipeline_count);
			m_impl->m_flushed_pipeline_count += m_impl->m_dirty_pipeline_count;
			m_impl->m_dirty_pipeline_count = 0;
		}
		else
		{
			fmt::throw_exception("Metal pipeline cache serialization requires macOS 26.0 or newer");
		}
	}

	void pipeline_cache::report() const
	{
		rsx_log.notice("rsx::metal::pipeline_cache::report()");
			rsx_log.notice("Metal pipeline cache: pending=%u, flushed=%u, archive=%s",
				m_impl->m_dirty_pipeline_count,
				m_impl->m_flushed_pipeline_count,
				m_impl->m_cache.pipeline_archive_file_path().c_str());
		}
	}
