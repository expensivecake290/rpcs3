#include "stdafx.h"
#include "MTLPipelineCompiler.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <variant>

namespace mtl
{
	namespace
	{
		constexpr u64 hash_seed = 0xcbf29ce484222325ull;
		constexpr u64 hash_prime = 0x100000001b3ull;

		void hash_bytes(u64& hash, const void* data, usz size)
		{
			const auto* bytes = static_cast<const u8*>(data);
			for (usz index = 0; index < size; ++index)
			{
				hash ^= bytes[index];
				hash *= hash_prime;
			}
		}

		template <typename T>
		void hash_value(u64& hash, const T& value)
		{
			hash_bytes(hash, &value, sizeof(value));
		}

		void hash_string(u64& hash, std::string_view value)
		{
			hash_bytes(hash, value.data(), value.size());
			const u8 terminator = 0xff;
			hash_value(hash, terminator);
		}

		u64 source_hash(std::string_view source)
		{
			u64 result = hash_seed;
			hash_string(result, source);
			return result;
		}

		std::string exception_diagnostic()
		{
			try
			{
				throw;
			}
			catch (const std::exception& exception)
			{
				return exception.what();
			}
			catch (...)
			{
				return "Unknown Metal pipeline compilation failure";
			}
		}

		u32 default_worker_count()
		{
			const u32 hardware = std::max(1u, std::thread::hardware_concurrency());
			return std::clamp(hardware / 2, 1u, 12u);
		}

		NSString* native_string(std::string_view value)
		{
			return [NSString stringWithUTF8String:std::string(value).c_str()];
		}

		NSURL* file_url(const std::filesystem::path& path)
		{
			return [NSURL fileURLWithPath:native_string(path.string())];
		}

		void invoke_callbacks(const std::vector<pipeline_compile_callback>& callbacks,
			const pipeline_compile_job& job)
		{
			for (const auto& callback : callbacks)
			{
				if (!callback) continue;
				try
				{
					callback(job);
				}
				catch (const std::exception& exception)
				{
					rsx_log.error("Metal pipeline callback failed: %s", exception.what());
				}
				catch (...)
				{
					rsx_log.error("Metal pipeline callback failed with an unknown exception");
				}
			}
		}
	}

	u64 pipeline_cache_key::hash() const
	{
		u64 result = hash_seed;
		hash_value(result, kind);
		hash_value(result, first_shader);
		hash_value(result, second_shader);
		hash_value(result, pipeline_state);
		hash_value(result, binding_layout);
		hash_value(result, specialization);
		return result;
	}

	usz pipeline_cache_key_hash::operator()(const pipeline_cache_key& key) const noexcept
	{
		return static_cast<usz>(key.hash());
	}

	void graphics_pipeline_compile_request::validate() const
	{
		if (!vertex || !fragment || vertex->source().empty() || fragment->source().empty() ||
			(vertex->state() != vertex_compile_state::decompiled && vertex->state() != vertex_compile_state::compiled) ||
			(fragment->state() != fragment_compile_state::decompiled && fragment->state() != fragment_compile_state::compiled))
		{
			fmt::throw_exception("Invalid Metal graphics pipeline compile request");
		}
		configuration.validate();
	}

	pipeline_cache_key graphics_pipeline_compile_request::cache_key() const
	{
		validate();
		u64 bindings = hash_seed;
		const u64 vertex_bindings = vertex->bindings().signature();
		const u64 fragment_bindings = fragment->bindings().signature();
		hash_value(bindings, vertex_bindings);
		hash_value(bindings, fragment_bindings);
		return {program_pipeline_kind::graphics, vertex->metadata().source_hash,
			fragment->metadata().source_hash, configuration.signature(), bindings, 0};
	}

	void compute_pipeline_compile_request::validate() const
	{
		if (source.empty() || function_name.empty())
		{
			fmt::throw_exception("Metal compute pipeline compile request requires source and an entry point");
		}
		layout.validate();
		for (const program_binding_reference& binding : required_bindings)
		{
			if (binding.stage != msl_shader_stage::compute || binding.index == umax || binding.name.empty())
			{
				fmt::throw_exception("Invalid required binding in Metal compute pipeline request");
			}
			const u32 count = binding.resource == argument_binding_class::buffer ? layout.buffer_count :
				binding.resource == argument_binding_class::texture ? layout.texture_count : layout.sampler_count;
			if (binding.index >= count)
			{
				fmt::throw_exception("Metal compute binding '%s' exceeds its argument-table layout", binding.name);
			}
		}
	}

	pipeline_cache_key compute_pipeline_compile_request::cache_key() const
	{
		validate();
		u64 specialization = hash_seed;
		hash_string(specialization, function_name);
		hash_value(specialization, maximum_threads_per_threadgroup);
		hash_value(specialization, threadgroup_size_is_multiple_of_execution_width);
		hash_value(specialization, fast_math);
		for (const program_binding_reference& binding : required_bindings)
		{
			hash_value(specialization, binding.stage);
			hash_value(specialization, binding.resource);
			hash_value(specialization, binding.index);
			hash_value(specialization, binding.texture_unit);
			hash_string(specialization, binding.name);
		}
		return {program_pipeline_kind::compute, source_hash(source), 0, specialization,
			layout.signature(), specialization};
	}

	void native_graphics_pipeline_compile_request::validate() const
	{
		if (vertex_source.empty() || fragment_source.empty() ||
			vertex_function_name.empty() || fragment_function_name.empty())
		{
			fmt::throw_exception("Metal native graphics compile request requires both shader stages");
		}
		vertex_layout.validate();
		fragment_layout.validate();
		configuration.validate();
		auto validate_bindings = [](std::span<const program_binding_reference> bindings,
			msl_shader_stage stage, const argument_table_layout& layout)
		{
			for (const program_binding_reference& binding : bindings)
			{
				if (binding.stage != stage || binding.index == umax || binding.name.empty())
					fmt::throw_exception("Invalid required binding in native Metal graphics request");
				const u32 count = binding.resource == argument_binding_class::buffer ? layout.buffer_count :
					binding.resource == argument_binding_class::texture ? layout.texture_count : layout.sampler_count;
				if (binding.index >= count)
					fmt::throw_exception("Native Metal graphics binding '%s' exceeds its layout", binding.name);
			}
		};
		validate_bindings(vertex_required_bindings, msl_shader_stage::vertex, vertex_layout);
		validate_bindings(fragment_required_bindings, msl_shader_stage::fragment, fragment_layout);
	}

	pipeline_cache_key native_graphics_pipeline_compile_request::cache_key() const
	{
		validate();
		u64 bindings = hash_seed;
		const u64 vertex_signature = vertex_layout.signature();
		const u64 fragment_signature = fragment_layout.signature();
		hash_value(bindings, vertex_signature);
		hash_value(bindings, fragment_signature);
		auto hash_bindings = [&](std::span<const program_binding_reference> values)
		{
			for (const auto& binding : values)
			{
				hash_value(bindings, binding.stage);
				hash_value(bindings, binding.resource);
				hash_value(bindings, binding.index);
				hash_value(bindings, binding.texture_unit);
				hash_string(bindings, binding.name);
			}
		};
		hash_bindings(vertex_required_bindings);
		hash_bindings(fragment_required_bindings);
		u64 specialization = hash_seed;
		hash_string(specialization, vertex_function_name);
		hash_string(specialization, fragment_function_name);
		hash_value(specialization, fast_math);
		return {program_pipeline_kind::graphics, source_hash(vertex_source), source_hash(fragment_source),
			configuration.signature(), bindings, specialization};
	}

	struct pipeline_compile_job::shared_state
	{
		mutable std::mutex mutex;
		std::condition_variable condition;
		pipeline_cache_key cache_key;
		pipeline_compile_state compile_state = pipeline_compile_state::queued;
		std::shared_ptr<MTLProgramPipeline> pipeline;
		std::string error;
		std::vector<pipeline_compile_callback> callbacks;
		u64 job_id = 0;
	};

	pipeline_compile_job::pipeline_compile_job(std::shared_ptr<shared_state> state)
		: m_state(std::move(state))
	{
	}

	pipeline_compile_job::operator bool() const
	{
		return m_state != nullptr;
	}

	u64 pipeline_compile_job::id() const
	{
		return m_state ? m_state->job_id : 0;
	}

	pipeline_cache_key pipeline_compile_job::key() const
	{
		return m_state ? m_state->cache_key : pipeline_cache_key{};
	}

	pipeline_compile_state pipeline_compile_job::state() const
	{
		if (!m_state) return pipeline_compile_state::cancelled;
		std::lock_guard lock(m_state->mutex);
		return m_state->compile_state;
	}

	bool pipeline_compile_job::ready() const
	{
		const auto current = state();
		return current == pipeline_compile_state::completed || current == pipeline_compile_state::failed ||
			current == pipeline_compile_state::cancelled;
	}

	bool pipeline_compile_job::succeeded() const
	{
		return state() == pipeline_compile_state::completed;
	}

	bool pipeline_compile_job::cancel()
	{
		if (!m_state) return false;
		std::vector<pipeline_compile_callback> callbacks;
		{
			std::lock_guard lock(m_state->mutex);
			if (m_state->compile_state != pipeline_compile_state::queued) return false;
			m_state->compile_state = pipeline_compile_state::cancelled;
			m_state->error = "Metal pipeline compilation was cancelled";
			callbacks.swap(m_state->callbacks);
		}
		m_state->condition.notify_all();
		invoke_callbacks(callbacks, *this);
		return true;
	}

	void pipeline_compile_job::wait() const
	{
		if (!m_state) return;
		std::unique_lock lock(m_state->mutex);
		m_state->condition.wait(lock, [&]
		{
			return m_state->compile_state == pipeline_compile_state::completed ||
				m_state->compile_state == pipeline_compile_state::failed ||
				m_state->compile_state == pipeline_compile_state::cancelled;
		});
	}

	bool pipeline_compile_job::wait_for(std::chrono::nanoseconds timeout) const
	{
		if (!m_state) return true;
		std::unique_lock lock(m_state->mutex);
		return m_state->condition.wait_for(lock, timeout, [&]
		{
			return m_state->compile_state == pipeline_compile_state::completed ||
				m_state->compile_state == pipeline_compile_state::failed ||
				m_state->compile_state == pipeline_compile_state::cancelled;
		});
	}

	std::shared_ptr<MTLProgramPipeline> pipeline_compile_job::result() const
	{
		wait();
		if (!m_state) return {};
		std::lock_guard lock(m_state->mutex);
		return m_state->pipeline;
	}

	std::string pipeline_compile_job::diagnostic() const
	{
		if (!m_state) return "Empty Metal pipeline compilation job";
		std::lock_guard lock(m_state->mutex);
		return m_state->error;
	}

	void pipeline_compiler_configuration::validate() const
	{
		if (worker_count > 64 || maximum_cached_pipelines == 0)
		{
			fmt::throw_exception("Invalid Metal pipeline compiler configuration");
		}
		if (!capture_pipeline_descriptors && !pipeline_script_path.empty())
		{
			fmt::throw_exception("Metal pipeline script path requires descriptor capture");
		}
	}

	struct MTLPipelineCompiler::impl
	{
		using request_variant = std::variant<graphics_pipeline_compile_request, compute_pipeline_compile_request,
			native_graphics_pipeline_compile_request>;

		struct queued_job
		{
			request_variant request;
			std::shared_ptr<pipeline_compile_job::shared_state> state;
			pipeline_compile_priority priority = pipeline_compile_priority::normal;
			u64 sequence = 0;
		};

		struct job_compare
		{
			bool operator()(const queued_job& left, const queued_job& right) const
			{
				if (left.priority != right.priority)
					return static_cast<u8>(left.priority) < static_cast<u8>(right.priority);
				return left.sequence > right.sequence;
			}
		};

		struct cache_entry
		{
			std::shared_ptr<MTLProgramPipeline> pipeline;
			u64 last_use = 0;
		};

		render_device* device = nullptr;
		id<MTL4Compiler> compiler = nil;
		id<MTL4PipelineDataSetSerializer> serializer = nil;
		id<MTL4Archive> archive = nil;
		pipeline_compiler_configuration configuration;
		std::priority_queue<queued_job, std::vector<queued_job>, job_compare> queue;
		std::unordered_map<pipeline_cache_key, cache_entry, pipeline_cache_key_hash> cache;
		std::unordered_map<pipeline_cache_key, std::weak_ptr<pipeline_compile_job::shared_state>, pipeline_cache_key_hash> in_flight;
		std::vector<std::thread> workers;
		pipeline_compiler_statistics counters;
		mutable std::mutex mutex;
		std::mutex shader_compile_mutex;
		std::mutex archive_mutex;
		std::condition_variable work_available;
		std::condition_variable idle;
		u64 next_job_id = 1;
		u64 next_sequence = 1;
		u64 cache_clock = 1;
		u64 active_jobs = 0;
		bool accepting = false;
		bool stopping = false;

		void require_initialized() const
		{
			if (!device || !compiler || !accepting)
				fmt::throw_exception("Metal pipeline compiler is not initialized");
		}

		void trim_cache_locked(usz maximum_entries)
		{
			while (cache.size() > maximum_entries)
			{
				auto oldest = std::min_element(cache.begin(), cache.end(), [](const auto& left, const auto& right)
				{
					return left.second.last_use < right.second.last_use;
				});
				if (oldest == cache.end()) break;
				cache.erase(oldest);
				counters.evicted++;
			}
		}

		static std::shared_ptr<MTLProgramPipeline> create_binding_instance(
			const std::shared_ptr<MTLProgramPipeline>& pipeline)
		{
			return std::shared_ptr<MTLProgramPipeline>(pipeline->create_binding_instance().release());
		}

		std::shared_ptr<MTLProgramPipeline> compile_graphics(const graphics_pipeline_compile_request& request)
		{
			request.validate();
			{
				std::lock_guard compile_lock(shader_compile_mutex);
				if (request.vertex->state() == vertex_compile_state::decompiled) request.vertex->Compile(compiler);
				if (request.fragment->state() == fragment_compile_state::decompiled) request.fragment->Compile(compiler);
				if (!*request.vertex || !*request.fragment)
					fmt::throw_exception("Metal graphics shader compilation did not produce valid programs");
			}
			auto pipeline = std::make_shared<MTLProgramPipeline>();
			pipeline->create_graphics(*device, *request.vertex, *request.fragment,
				request.configuration, compiler, archive);
			return pipeline;
		}

		std::shared_ptr<MTLProgramPipeline> compile_compute(const compute_pipeline_compile_request& request)
		{
			request.validate();
			MTLCompileOptions* options = [MTLCompileOptions new];
			options.languageVersion = MTLLanguageVersion4_0;
			options.mathMode = request.fast_math ? MTLMathModeFast : MTLMathModeSafe;
			MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
			descriptor.name = native_string(request.label.empty() ? "RPCS3 compute library" : request.label);
			descriptor.source = native_string(request.source);
			descriptor.options = options;
			NSError* native_error = nil;
			id<MTLLibrary> library = [compiler newLibraryWithDescriptor:descriptor error:&native_error];
			if (!library)
			{
				const std::string diagnostic = native_error.localizedDescription.UTF8String ?: "unknown error";
				fmt::throw_exception("Metal compute library compilation failed: %s", diagnostic);
			}
			compute_pipeline_configuration pipeline_configuration;
			pipeline_configuration.library = library;
			pipeline_configuration.function_name = request.function_name;
			pipeline_configuration.layout = request.layout;
			pipeline_configuration.required_bindings = request.required_bindings;
			pipeline_configuration.maximum_threads_per_threadgroup = request.maximum_threads_per_threadgroup;
			pipeline_configuration.threadgroup_size_is_multiple_of_execution_width =
				request.threadgroup_size_is_multiple_of_execution_width;
			pipeline_configuration.label = request.label;
			auto pipeline = std::make_shared<MTLProgramPipeline>();
			pipeline->create_compute(*device, pipeline_configuration, compiler, archive);
			return pipeline;
		}

		id<MTLLibrary> compile_library(std::string_view source, std::string_view label, bool fast_math)
		{
			MTLCompileOptions* options = [MTLCompileOptions new];
			options.languageVersion = MTLLanguageVersion4_0;
			options.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
			MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
			descriptor.name = native_string(label);
			descriptor.source = native_string(source);
			descriptor.options = options;
			NSError* native_error = nil;
			id<MTLLibrary> library = [compiler newLibraryWithDescriptor:descriptor error:&native_error];
			if (!library)
			{
				const std::string diagnostic = native_error.localizedDescription.UTF8String ?: "unknown error";
				fmt::throw_exception("Metal library '%s' compilation failed: %s", label, diagnostic);
			}
			return library;
		}

		std::shared_ptr<MTLProgramPipeline> compile_native_graphics(
			const native_graphics_pipeline_compile_request& request)
		{
			request.validate();
			id<MTLLibrary> vertex_library = compile_library(request.vertex_source,
				"RPCS3 native vertex library", request.fast_math);
			id<MTLLibrary> fragment_library = compile_library(request.fragment_source,
				"RPCS3 native fragment library", request.fast_math);
			native_graphics_stage_configuration vertex;
			vertex.library = vertex_library;
			vertex.function_name = request.vertex_function_name;
			vertex.layout = request.vertex_layout;
			vertex.required_bindings = request.vertex_required_bindings;
			vertex.guest_program_hash = source_hash(request.vertex_source);
			vertex.source_hash = vertex.guest_program_hash;
			native_graphics_stage_configuration fragment;
			fragment.library = fragment_library;
			fragment.function_name = request.fragment_function_name;
			fragment.layout = request.fragment_layout;
			fragment.required_bindings = request.fragment_required_bindings;
			fragment.guest_program_hash = source_hash(request.fragment_source);
			fragment.source_hash = fragment.guest_program_hash;
			auto pipeline = std::make_shared<MTLProgramPipeline>();
			pipeline->create_graphics_native(*device, vertex, fragment, request.configuration, compiler, archive);
			return pipeline;
		}

		std::shared_ptr<MTLProgramPipeline> compile_request(const request_variant& request)
		{
			return std::visit([&](const auto& value) -> std::shared_ptr<MTLProgramPipeline>
			{
				using request_type = std::decay_t<decltype(value)>;
				if constexpr (std::is_same_v<request_type, graphics_pipeline_compile_request>)
					return compile_graphics(value);
				else if constexpr (std::is_same_v<request_type, compute_pipeline_compile_request>)
					return compile_compute(value);
				else
					return compile_native_graphics(value);
			}, request);
		}

		template <typename Request>
		pipeline_compile_job submit(Request request, pipeline_compile_priority priority,
			pipeline_compile_callback callback)
		{
			request.validate();
			const pipeline_cache_key key = request.cache_key();
			std::shared_ptr<pipeline_compile_job::shared_state> state;
			bool immediate = false;
			{
				std::lock_guard lock(mutex);
				require_initialized();
				if (auto found = cache.find(key); found != cache.end())
				{
					found->second.last_use = cache_clock++;
					state = std::make_shared<pipeline_compile_job::shared_state>();
					state->cache_key = key;
					state->compile_state = pipeline_compile_state::completed;
					state->pipeline = create_binding_instance(found->second.pipeline);
					state->job_id = next_job_id++;
					counters.cache_hits++;
					immediate = true;
				}
				else if (auto found = in_flight.find(key); found != in_flight.end())
				{
					state = found->second.lock();
					if (state)
					{
						std::lock_guard state_lock(state->mutex);
						if (state->compile_state == pipeline_compile_state::queued ||
							state->compile_state == pipeline_compile_state::compiling)
						{
							if (callback) state->callbacks.push_back(std::move(callback));
							counters.coalesced++;
							return pipeline_compile_job(state);
						}
					}
					state.reset();
					in_flight.erase(found);
				}
				if (!state)
				{
					state = std::make_shared<pipeline_compile_job::shared_state>();
					state->cache_key = key;
					state->job_id = next_job_id++;
					if (callback) state->callbacks.push_back(std::move(callback));
					queue.push({std::move(request), state, priority, next_sequence++});
					in_flight[key] = state;
					counters.submitted++;
				}
			}
			pipeline_compile_job result(state);
			if (immediate && callback) invoke_callbacks({callback}, result);
			if (!immediate) work_available.notify_one();
			return result;
		}

		void worker_loop()
		{
			for (;;)
			{
				queued_job job;
				{
					std::unique_lock lock(mutex);
					work_available.wait(lock, [&] { return stopping || !queue.empty(); });
					if (stopping && queue.empty()) return;
					job = queue.top();
					queue.pop();
					active_jobs++;
				}

				bool cancelled = false;
				{
					std::lock_guard state_lock(job.state->mutex);
					cancelled = job.state->compile_state == pipeline_compile_state::cancelled;
					if (!cancelled) job.state->compile_state = pipeline_compile_state::compiling;
				}

				std::shared_ptr<MTLProgramPipeline> pipeline;
				std::shared_ptr<MTLProgramPipeline> cache_prototype;
				std::string diagnostic;
				if (!cancelled)
				{
					try
					{
						pipeline = compile_request(job.request);
						cache_prototype = create_binding_instance(pipeline);
					}
					catch (...)
					{
						diagnostic = exception_diagnostic();
					}
				}

				std::vector<pipeline_compile_callback> callbacks;
				{
					std::lock_guard state_lock(job.state->mutex);
					if (!cancelled)
					{
						job.state->pipeline = pipeline;
						job.state->error = std::move(diagnostic);
						job.state->compile_state = pipeline ? pipeline_compile_state::completed : pipeline_compile_state::failed;
					}
					callbacks.swap(job.state->callbacks);
				}
				job.state->condition.notify_all();

				{
					std::lock_guard lock(mutex);
					in_flight.erase(job.state->cache_key);
					if (cancelled)
					{
						counters.cancelled++;
					}
					else if (pipeline)
					{
						cache[job.state->cache_key] = {std::move(cache_prototype), cache_clock++};
						trim_cache_locked(configuration.maximum_cached_pipelines);
						counters.compiled++;
					}
					else
					{
						counters.failed++;
					}
				}
				invoke_callbacks(callbacks, pipeline_compile_job(job.state));
				{
					std::lock_guard lock(mutex);
					active_jobs--;
					if (queue.empty() && active_jobs == 0) idle.notify_all();
				}
			}
		}
	};

	MTLPipelineCompiler::MTLPipelineCompiler()
		: m_impl(std::make_unique<impl>())
	{
	}

	MTLPipelineCompiler::~MTLPipelineCompiler()
	{
		shutdown(true);
	}

	void MTLPipelineCompiler::initialize(render_device& device,
		const pipeline_compiler_configuration& configuration)
	{
		shutdown(true);
		configuration.validate();
		if (!device || !device.compiler() || !device.info().features.argument_tables)
		{
			fmt::throw_exception("Metal pipeline compiler requires compiler and argument-table support");
		}
		m_impl->device = &device;
		m_impl->configuration = configuration;
		if (configuration.capture_pipeline_descriptors || configuration.capture_pipeline_binaries)
		{
			MTL4PipelineDataSetSerializerDescriptor* serializer_descriptor =
				[MTL4PipelineDataSetSerializerDescriptor new];
			serializer_descriptor.configuration = 0;
			if (configuration.capture_pipeline_descriptors)
				serializer_descriptor.configuration |= MTL4PipelineDataSetSerializerConfigurationCaptureDescriptors;
			if (configuration.capture_pipeline_binaries)
				serializer_descriptor.configuration |= MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
			m_impl->serializer = [device.native_handle() newPipelineDataSetSerializerWithDescriptor:serializer_descriptor];
			if (!m_impl->serializer) fmt::throw_exception("Metal pipeline data serializer creation failed");
		}

		if (!configuration.archive_path.empty())
		{
			std::error_code filesystem_error;
			if (std::filesystem::exists(configuration.archive_path, filesystem_error))
			{
				NSError* native_error = nil;
				m_impl->archive = [device.native_handle() newArchiveWithURL:file_url(configuration.archive_path)
					error:&native_error];
				if (!m_impl->archive)
				{
					const std::string diagnostic = native_error.localizedDescription.UTF8String ?: "unknown error";
					shutdown(false);
					fmt::throw_exception("Metal pipeline archive loading failed: %s", diagnostic);
				}
				m_impl->counters.archive_loads++;
			}
			else if (filesystem_error)
			{
				shutdown(false);
				fmt::throw_exception("Metal pipeline archive path could not be inspected: %s", filesystem_error.message());
			}
		}

		MTL4CompilerDescriptor* compiler_descriptor = [MTL4CompilerDescriptor new];
		compiler_descriptor.label = @"RPCS3 asynchronous pipeline compiler";
		compiler_descriptor.pipelineDataSetSerializer = m_impl->serializer;
		NSError* native_error = nil;
		m_impl->compiler = [device.native_handle() newCompilerWithDescriptor:compiler_descriptor error:&native_error];
		if (!m_impl->compiler)
		{
			const std::string diagnostic = native_error.localizedDescription.UTF8String ?: "unknown error";
			shutdown(false);
			fmt::throw_exception("Metal asynchronous compiler creation failed: %s", diagnostic);
		}

		const u32 worker_count = configuration.worker_count ? configuration.worker_count : default_worker_count();
		m_impl->accepting = true;
		m_impl->stopping = false;
		m_impl->workers.reserve(worker_count);
		for (u32 index = 0; index < worker_count; ++index)
		{
			m_impl->workers.emplace_back([implementation = m_impl.get()] { implementation->worker_loop(); });
		}
	}

	void MTLPipelineCompiler::shutdown(bool drain)
	{
		if (!m_impl || (!m_impl->device && m_impl->workers.empty())) return;
		if (drain) wait_idle();

		std::vector<std::shared_ptr<pipeline_compile_job::shared_state>> cancelled_states;
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->accepting = false;
			if (!drain)
			{
				while (!m_impl->queue.empty())
				{
					auto state = m_impl->queue.top().state;
					m_impl->queue.pop();
					m_impl->in_flight.erase(state->cache_key);
					cancelled_states.push_back(std::move(state));
					m_impl->counters.cancelled++;
				}
			}
			m_impl->stopping = true;
		}
		for (const auto& state : cancelled_states)
		{
			std::vector<pipeline_compile_callback> callbacks;
			{
				std::lock_guard state_lock(state->mutex);
				state->compile_state = pipeline_compile_state::cancelled;
				state->error = "Metal pipeline compiler shut down before this job ran";
				callbacks.swap(state->callbacks);
			}
			state->condition.notify_all();
			invoke_callbacks(callbacks, pipeline_compile_job(state));
		}
		m_impl->work_available.notify_all();
		for (std::thread& worker : m_impl->workers)
		{
			if (worker.joinable()) worker.join();
		}
		m_impl->workers.clear();
		m_impl->queue = {};
		m_impl->in_flight.clear();
		m_impl->cache.clear();
		m_impl->archive = nil;
		m_impl->compiler = nil;
		m_impl->serializer = nil;
		m_impl->device = nullptr;
		m_impl->active_jobs = 0;
		m_impl->stopping = false;
		m_impl->counters = {};
	}

	pipeline_compile_job MTLPipelineCompiler::submit_graphics(graphics_pipeline_compile_request request,
		pipeline_compile_priority priority, pipeline_compile_callback callback)
	{
		return m_impl->submit(std::move(request), priority, std::move(callback));
	}

	pipeline_compile_job MTLPipelineCompiler::submit_compute(compute_pipeline_compile_request request,
		pipeline_compile_priority priority, pipeline_compile_callback callback)
	{
		return m_impl->submit(std::move(request), priority, std::move(callback));
	}

	pipeline_compile_job MTLPipelineCompiler::submit_native_graphics(
		native_graphics_pipeline_compile_request request, pipeline_compile_priority priority,
		pipeline_compile_callback callback)
	{
		return m_impl->submit(std::move(request), priority, std::move(callback));
	}

	std::shared_ptr<MTLProgramPipeline> MTLPipelineCompiler::compile_graphics_inline(
		const graphics_pipeline_compile_request& request)
	{
		request.validate();
		const pipeline_cache_key key = request.cache_key();
		if (auto cached = find_cached(key)) return cached;
		std::shared_ptr<MTLProgramPipeline> pipeline;
		std::shared_ptr<MTLProgramPipeline> cache_prototype;
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->require_initialized();
			m_impl->counters.submitted++;
		}
		try
		{
			pipeline = m_impl->compile_graphics(request);
			cache_prototype = impl::create_binding_instance(pipeline);
		}
		catch (...)
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->counters.failed++;
			throw;
		}
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->cache[key] = {std::move(cache_prototype), m_impl->cache_clock++};
			m_impl->trim_cache_locked(m_impl->configuration.maximum_cached_pipelines);
			m_impl->counters.compiled++;
		}
		return pipeline;
	}

	std::shared_ptr<MTLProgramPipeline> MTLPipelineCompiler::compile_compute_inline(
		const compute_pipeline_compile_request& request)
	{
		request.validate();
		const pipeline_cache_key key = request.cache_key();
		if (auto cached = find_cached(key)) return cached;
		std::shared_ptr<MTLProgramPipeline> pipeline;
		std::shared_ptr<MTLProgramPipeline> cache_prototype;
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->require_initialized();
			m_impl->counters.submitted++;
		}
		try
		{
			pipeline = m_impl->compile_compute(request);
			cache_prototype = impl::create_binding_instance(pipeline);
		}
		catch (...)
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->counters.failed++;
			throw;
		}
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->cache[key] = {std::move(cache_prototype), m_impl->cache_clock++};
			m_impl->trim_cache_locked(m_impl->configuration.maximum_cached_pipelines);
			m_impl->counters.compiled++;
		}
		return pipeline;
	}

	std::shared_ptr<MTLProgramPipeline> MTLPipelineCompiler::compile_native_graphics_inline(
		const native_graphics_pipeline_compile_request& request)
	{
		request.validate();
		const pipeline_cache_key key = request.cache_key();
		if (auto cached = find_cached(key)) return cached;
		std::shared_ptr<MTLProgramPipeline> pipeline;
		std::shared_ptr<MTLProgramPipeline> cache_prototype;
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->require_initialized();
			m_impl->counters.submitted++;
		}
		try
		{
			pipeline = m_impl->compile_native_graphics(request);
			cache_prototype = impl::create_binding_instance(pipeline);
		}
		catch (...)
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->counters.failed++;
			throw;
		}
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->cache[key] = {std::move(cache_prototype), m_impl->cache_clock++};
			m_impl->trim_cache_locked(m_impl->configuration.maximum_cached_pipelines);
			m_impl->counters.compiled++;
		}
		return pipeline;
	}

	std::shared_ptr<MTLProgramPipeline> MTLPipelineCompiler::find_cached(const pipeline_cache_key& key) const
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_initialized();
		const auto found = m_impl->cache.find(key);
		if (found == m_impl->cache.end()) return {};
		found->second.last_use = m_impl->cache_clock++;
		m_impl->counters.cache_hits++;
		return impl::create_binding_instance(found->second.pipeline);
	}

	void MTLPipelineCompiler::wait_idle()
	{
		if (!m_impl || !m_impl->device) return;
		std::unique_lock lock(m_impl->mutex);
		m_impl->idle.wait(lock, [&] { return m_impl->queue.empty() && m_impl->active_jobs == 0; });
	}

	void MTLPipelineCompiler::clear_cache()
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_initialized();
		m_impl->cache.clear();
	}

	void MTLPipelineCompiler::trim_cache(usz maximum_entries)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_initialized();
		m_impl->trim_cache_locked(maximum_entries);
	}

	void MTLPipelineCompiler::flush_archive()
	{
		if (!m_impl || m_impl->configuration.archive_path.empty())
			fmt::throw_exception("Metal pipeline compiler has no configured archive path");
		flush_archive(m_impl->configuration.archive_path);
	}

	void MTLPipelineCompiler::flush_archive(const std::filesystem::path& path)
	{
		if (path.empty()) fmt::throw_exception("Metal pipeline archive path is empty");
		if (!m_impl->configuration.capture_pipeline_binaries)
			fmt::throw_exception("Metal pipeline binary capture is disabled");
		std::lock_guard archive_lock(m_impl->archive_mutex);
		wait_idle();
		id<MTL4PipelineDataSetSerializer> serializer = nil;
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->require_initialized();
			serializer = m_impl->serializer;
		}
		if (!serializer) fmt::throw_exception("Metal pipeline archive capture is disabled");

		std::error_code filesystem_error;
		if (!path.parent_path().empty())
		{
			std::filesystem::create_directories(path.parent_path(), filesystem_error);
			if (filesystem_error)
				fmt::throw_exception("Metal pipeline archive directory creation failed: %s", filesystem_error.message());
		}
		const std::filesystem::path temporary = path.string() + ".pending";
		std::filesystem::remove(temporary, filesystem_error);
		filesystem_error.clear();
		NSError* native_error = nil;
		if (![serializer serializeAsArchiveAndFlushToURL:file_url(temporary) error:&native_error])
		{
			const std::string diagnostic = native_error.localizedDescription.UTF8String ?: "unknown error";
			fmt::throw_exception("Metal pipeline archive serialization failed: %s", diagnostic);
		}
		std::filesystem::rename(temporary, path, filesystem_error);
		if (filesystem_error)
		{
			std::error_code cleanup_error;
			std::filesystem::remove(temporary, cleanup_error);
			fmt::throw_exception("Metal pipeline archive installation failed: %s", filesystem_error.message());
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->counters.archive_flushes++;
	}

	void MTLPipelineCompiler::flush_pipeline_script()
	{
		if (!m_impl || m_impl->configuration.pipeline_script_path.empty())
			fmt::throw_exception("Metal pipeline compiler has no configured pipeline script path");
		flush_pipeline_script(m_impl->configuration.pipeline_script_path);
	}

	void MTLPipelineCompiler::flush_pipeline_script(const std::filesystem::path& path)
	{
		if (path.empty()) fmt::throw_exception("Metal pipeline script path is empty");
		if (!m_impl->configuration.capture_pipeline_descriptors)
			fmt::throw_exception("Metal pipeline descriptor capture is disabled");
		std::lock_guard archive_lock(m_impl->archive_mutex);
		wait_idle();
		id<MTL4PipelineDataSetSerializer> serializer = nil;
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->require_initialized();
			serializer = m_impl->serializer;
		}
		if (!serializer) fmt::throw_exception("Metal pipeline data capture is unavailable");

		std::error_code filesystem_error;
		if (!path.parent_path().empty())
		{
			std::filesystem::create_directories(path.parent_path(), filesystem_error);
			if (filesystem_error)
				fmt::throw_exception("Metal pipeline script directory creation failed: %s", filesystem_error.message());
		}
		NSError* native_error = nil;
		NSData* script = [serializer serializeAsPipelinesScriptWithError:&native_error];
		if (!script)
		{
			const std::string diagnostic = native_error.localizedDescription.UTF8String ?: "unknown error";
			fmt::throw_exception("Metal pipeline script serialization failed: %s", diagnostic);
		}
		if (![script writeToURL:file_url(path) options:NSDataWritingAtomic error:&native_error])
		{
			const std::string diagnostic = native_error.localizedDescription.UTF8String ?: "unknown error";
			fmt::throw_exception("Metal pipeline script write failed: %s", diagnostic);
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->counters.pipeline_script_flushes++;
	}

	MTLPipelineCompiler::operator bool() const
	{
		if (!m_impl) return false;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->device && m_impl->compiler && m_impl->accepting;
	}

	render_device& MTLPipelineCompiler::owner() const
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_initialized();
		return *m_impl->device;
	}

	compiler_handle MTLPipelineCompiler::native_compiler() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->compiler;
	}

	pipeline_archive_handle MTLPipelineCompiler::lookup_archive() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->archive;
	}

	pipeline_compiler_statistics MTLPipelineCompiler::statistics() const
	{
		if (!m_impl) return {};
		std::lock_guard lock(m_impl->mutex);
		auto result = m_impl->counters;
		result.queued = m_impl->queue.size();
		result.active = m_impl->active_jobs;
		result.cached = m_impl->cache.size();
		return result;
	}
}
