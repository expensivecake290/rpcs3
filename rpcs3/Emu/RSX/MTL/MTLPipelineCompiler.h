#pragma once

#include "MTLProgramPipeline.h"

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mtl
{
	enum class pipeline_compile_priority : u8
	{
		background,
		normal,
		urgent,
	};

	enum class pipeline_compile_state : u8
	{
		queued,
		compiling,
		completed,
		failed,
		cancelled,
	};

	struct pipeline_cache_key
	{
		program_pipeline_kind kind = program_pipeline_kind::graphics;
		u64 first_shader = 0;
		u64 second_shader = 0;
		u64 pipeline_state = 0;
		u64 binding_layout = 0;
		u64 specialization = 0;

		[[nodiscard]] bool operator==(const pipeline_cache_key&) const = default;
		[[nodiscard]] u64 hash() const;
	};

	struct pipeline_cache_key_hash
	{
		[[nodiscard]] usz operator()(const pipeline_cache_key& key) const noexcept;
	};

	struct graphics_pipeline_compile_request
	{
		std::shared_ptr<MTLVertexProgram> vertex;
		std::shared_ptr<MTLFragmentProgram> fragment;
		graphics_pipeline_configuration configuration;

		void validate() const;
		[[nodiscard]] pipeline_cache_key cache_key() const;
	};

	struct compute_pipeline_compile_request
	{
		std::string source;
		std::string function_name;
		argument_table_layout layout;
		std::vector<program_binding_reference> required_bindings;
		u32 maximum_threads_per_threadgroup = 0;
		bool threadgroup_size_is_multiple_of_execution_width = false;
		bool fast_math = true;
		std::string label;

		void validate() const;
		[[nodiscard]] pipeline_cache_key cache_key() const;
	};

	struct native_graphics_pipeline_compile_request
	{
		std::string vertex_source;
		std::string fragment_source;
		std::string vertex_function_name;
		std::string fragment_function_name;
		argument_table_layout vertex_layout;
		argument_table_layout fragment_layout;
		std::vector<program_binding_reference> vertex_required_bindings;
		std::vector<program_binding_reference> fragment_required_bindings;
		graphics_pipeline_configuration configuration;
		bool fast_math = true;

		void validate() const;
		[[nodiscard]] pipeline_cache_key cache_key() const;
	};

	class pipeline_compile_job;
	using pipeline_compile_callback = std::function<void(const pipeline_compile_job&)>;

	class pipeline_compile_job final
	{
		struct shared_state;
		std::shared_ptr<shared_state> m_state;

		explicit pipeline_compile_job(std::shared_ptr<shared_state> state);
		friend class MTLPipelineCompiler;

	public:
		pipeline_compile_job() = default;

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] u64 id() const;
		[[nodiscard]] pipeline_cache_key key() const;
		[[nodiscard]] pipeline_compile_state state() const;
		[[nodiscard]] bool ready() const;
		[[nodiscard]] bool succeeded() const;
		[[nodiscard]] bool cancel();
		void wait() const;
		[[nodiscard]] bool wait_for(std::chrono::nanoseconds timeout) const;
		[[nodiscard]] std::shared_ptr<MTLProgramPipeline> result() const;
		[[nodiscard]] std::string diagnostic() const;
	};

	struct pipeline_compiler_configuration
	{
		u32 worker_count = 0;
		usz maximum_cached_pipelines = 4096;
		std::filesystem::path archive_path;
		std::filesystem::path pipeline_script_path;
		bool capture_pipeline_descriptors = true;
		bool capture_pipeline_binaries = false;

		void validate() const;
	};

	struct pipeline_compiler_statistics
	{
		u64 submitted = 0;
		u64 compiled = 0;
		u64 failed = 0;
		u64 cancelled = 0;
		u64 cache_hits = 0;
		u64 coalesced = 0;
		u64 evicted = 0;
		u64 archive_loads = 0;
		u64 archive_flushes = 0;
		u64 pipeline_script_flushes = 0;
		u64 queued = 0;
		u64 active = 0;
		u64 cached = 0;
	};

	class MTLPipelineCompiler final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		MTLPipelineCompiler();
		~MTLPipelineCompiler();
		MTLPipelineCompiler(const MTLPipelineCompiler&) = delete;
		MTLPipelineCompiler& operator=(const MTLPipelineCompiler&) = delete;
		MTLPipelineCompiler(MTLPipelineCompiler&&) = delete;
		MTLPipelineCompiler& operator=(MTLPipelineCompiler&&) = delete;

		void initialize(render_device& device, const pipeline_compiler_configuration& configuration = {});
		void shutdown(bool drain = true);

		[[nodiscard]] pipeline_compile_job submit_graphics(graphics_pipeline_compile_request request,
			pipeline_compile_priority priority = pipeline_compile_priority::normal,
			pipeline_compile_callback callback = {});
		[[nodiscard]] pipeline_compile_job submit_compute(compute_pipeline_compile_request request,
			pipeline_compile_priority priority = pipeline_compile_priority::normal,
			pipeline_compile_callback callback = {});
		[[nodiscard]] pipeline_compile_job submit_native_graphics(
			native_graphics_pipeline_compile_request request,
			pipeline_compile_priority priority = pipeline_compile_priority::normal,
			pipeline_compile_callback callback = {});
		[[nodiscard]] std::shared_ptr<MTLProgramPipeline> compile_graphics_inline(
			const graphics_pipeline_compile_request& request);
		[[nodiscard]] std::shared_ptr<MTLProgramPipeline> compile_compute_inline(
			const compute_pipeline_compile_request& request);
		[[nodiscard]] std::shared_ptr<MTLProgramPipeline> compile_native_graphics_inline(
			const native_graphics_pipeline_compile_request& request);

		[[nodiscard]] std::shared_ptr<MTLProgramPipeline> find_cached(const pipeline_cache_key& key) const;
		void wait_idle();
		void clear_cache();
		void trim_cache(usz maximum_entries);
		void flush_archive();
		void flush_archive(const std::filesystem::path& path);
		void flush_pipeline_script();
		void flush_pipeline_script(const std::filesystem::path& path);

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] render_device& owner() const;
		[[nodiscard]] compiler_handle native_compiler() const;
		[[nodiscard]] pipeline_archive_handle lookup_archive() const;
		[[nodiscard]] pipeline_compiler_statistics statistics() const;
	};
}
