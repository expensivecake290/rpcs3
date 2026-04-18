#include "stdafx.h"
#include "MTLPipelineCache.h"

#include "MTLShaderCache.h"
#include "MTLShaderCompiler.h"

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
}

namespace rsx::metal
{
	struct pipeline_cache::pipeline_cache_impl
	{
		shader_compiler& m_compiler;
		persistent_shader_cache& m_cache;
		u32 m_dirty_pipeline_count = 0;
		u32 m_flushed_pipeline_count = 0;
		u32 m_skipped_flush_count = 0;
		u32 m_successful_flush_count = 0;
		u32 m_serializer_missing_failures = 0;
		u32 m_script_serialization_failures = 0;
		u32 m_archive_serialization_failures = 0;
		u32 m_archived_pipeline_count = 0;
		u64 m_archive_script_size = 0;
		u64 m_archive_size = 0;
		b8 m_archive_metadata_found = false;
		b8 m_archive_metadata_invalid = false;
		std::string m_archive_metadata_error;

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

		pipeline_archive_metadata archive_metadata;
		std::string archive_metadata_error;
		if (m_impl->m_cache.try_find_pipeline_archive_metadata(archive_metadata, archive_metadata_error))
		{
			m_impl->m_archive_metadata_found = true;
			m_impl->m_archived_pipeline_count = archive_metadata.flushed_pipeline_count;
			m_impl->m_flushed_pipeline_count = archive_metadata.flushed_pipeline_count;
			m_impl->m_archive_script_size = archive_metadata.script_size;
			m_impl->m_archive_size = archive_metadata.archive_size;

			rsx_log.notice("Metal pipeline archive metadata restored: pipelines=%u, script_size=0x%llx, archive_size=0x%llx",
				archive_metadata.flushed_pipeline_count,
				archive_metadata.script_size,
				archive_metadata.archive_size);
		}
		else if (!archive_metadata_error.empty())
		{
			m_impl->m_archive_metadata_invalid = true;
			m_impl->m_archive_metadata_error = std::move(archive_metadata_error);
			rsx_log.warning("Metal pipeline archive metadata is invalid and will not seed the pipeline cache: %s", m_impl->m_archive_metadata_error);
		}
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
			m_impl->m_skipped_flush_count++;
			rsx_log.trace("Metal pipeline cache flush skipped because no Metal pipeline states were compiled");
			return;
		}

		if (@available(macOS 26.0, *))
		{
			id<MTL4PipelineDataSetSerializer> serializer =
				(__bridge id<MTL4PipelineDataSetSerializer>)m_impl->m_compiler.pipeline_serializer_handle();

			if (!serializer)
			{
				m_impl->m_serializer_missing_failures++;
				fmt::throw_exception("Metal pipeline cache flush requires a pipeline data set serializer");
			}

			NSError* script_error = nil;
			NSData* pipeline_script = [serializer serializeAsPipelinesScriptWithError:&script_error];
			if (!pipeline_script)
			{
				const std::string error = script_error ? get_ns_string([script_error localizedDescription]) : "unknown error";
				m_impl->m_script_serialization_failures++;
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
				m_impl->m_archive_serialization_failures++;
				fmt::throw_exception("Metal pipeline cache failed to serialize pipeline archive '%s': %s", archive_path, error);
			}

			m_impl->m_cache.store_pipeline_archive_metadata(
				script_path,
				archive_path,
				m_impl->m_flushed_pipeline_count + m_impl->m_dirty_pipeline_count);

			pipeline_archive_metadata archive_metadata;
			std::string archive_metadata_error;
			if (!m_impl->m_cache.try_find_pipeline_archive_metadata(archive_metadata, archive_metadata_error))
			{
				if (archive_metadata_error.empty())
				{
					archive_metadata_error = "metadata not found";
				}

				fmt::throw_exception("Metal pipeline archive metadata lookup failed after flush: %s", archive_metadata_error);
			}

			m_impl->m_flushed_pipeline_count += m_impl->m_dirty_pipeline_count;
			m_impl->m_dirty_pipeline_count = 0;
			m_impl->m_archived_pipeline_count = archive_metadata.flushed_pipeline_count;
			m_impl->m_archive_script_size = archive_metadata.script_size;
			m_impl->m_archive_size = archive_metadata.archive_size;
			m_impl->m_archive_metadata_found = true;
			m_impl->m_archive_metadata_invalid = false;
			m_impl->m_archive_metadata_error.clear();
			m_impl->m_successful_flush_count++;
		}
		else
		{
			fmt::throw_exception("Metal pipeline cache serialization requires macOS 26.0 or newer");
		}
	}

	pipeline_cache_stats pipeline_cache::stats() const
	{
		rsx_log.trace("rsx::metal::pipeline_cache::stats()");

		return
		{
			.pending_pipeline_count = m_impl->m_dirty_pipeline_count,
			.flushed_pipeline_count = m_impl->m_flushed_pipeline_count,
			.skipped_flush_count = m_impl->m_skipped_flush_count,
			.successful_flush_count = m_impl->m_successful_flush_count,
			.serializer_missing_failures = m_impl->m_serializer_missing_failures,
			.script_serialization_failures = m_impl->m_script_serialization_failures,
			.archive_serialization_failures = m_impl->m_archive_serialization_failures,
			.archived_pipeline_count = m_impl->m_archived_pipeline_count,
			.archive_script_size = m_impl->m_archive_script_size,
			.archive_size = m_impl->m_archive_size,
			.archive_metadata_found = m_impl->m_archive_metadata_found,
			.archive_metadata_invalid = m_impl->m_archive_metadata_invalid,
			.script_path = m_impl->m_cache.pipeline_script_file_path(),
			.archive_path = m_impl->m_cache.pipeline_archive_file_path(),
			.archive_metadata_error = m_impl->m_archive_metadata_error,
		};
	}

	void pipeline_cache::report() const
	{
		rsx_log.notice("rsx::metal::pipeline_cache::report()");
		const pipeline_cache_stats cache_stats = stats();
		rsx_log.notice("Metal pipeline cache: pending=%u, flushed=%u, successful_flushes=%u, skipped_flushes=%u",
			cache_stats.pending_pipeline_count,
			cache_stats.flushed_pipeline_count,
			cache_stats.successful_flush_count,
			cache_stats.skipped_flush_count);
		rsx_log.notice("Metal pipeline cache paths: script=%s, archive=%s",
			cache_stats.script_path,
			cache_stats.archive_path);
		rsx_log.notice("Metal pipeline archive metadata: found=%u, invalid=%u, archived_pipelines=%u, script_size=0x%llx, archive_size=0x%llx",
			static_cast<u32>(cache_stats.archive_metadata_found),
			static_cast<u32>(cache_stats.archive_metadata_invalid),
			cache_stats.archived_pipeline_count,
			cache_stats.archive_script_size,
			cache_stats.archive_size);
		if (cache_stats.archive_metadata_invalid)
		{
			rsx_log.warning("Metal pipeline archive metadata error: %s", cache_stats.archive_metadata_error);
		}
		rsx_log.notice("Metal pipeline cache failures: serializer_missing=%u, script_serialization=%u, archive_serialization=%u",
			cache_stats.serializer_missing_failures,
			cache_stats.script_serialization_failures,
			cache_stats.archive_serialization_failures);
	}
}
