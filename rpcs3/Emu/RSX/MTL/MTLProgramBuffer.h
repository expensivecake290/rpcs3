#pragma once

#include "MTLHelpers.h"
#include "MTLPipelineCompiler.h"
#include "Emu/system_config.h"
#include "Emu/RSX/Program/ProgramStateCache.h"

namespace mtl
{
	struct cached_vertex_program
	{
		u32 id = 0;
		std::shared_ptr<MTLVertexProgram> program;
	};

	struct cached_fragment_program
	{
		u32 id = 0;
		std::shared_ptr<MTLFragmentProgram> program;
		std::vector<u32> constant_offsets;
	};

	struct MTLProgramTraits
	{
		using vertex_program_type = cached_vertex_program;
		using fragment_program_type = cached_fragment_program;
		using pipeline_type = MTLProgramPipeline;
		using pipeline_storage_type = std::shared_ptr<MTLProgramPipeline>;
		using pipeline_properties = graphics_pipeline_configuration;

		static void recompile_fragment_program(const RSXFragmentProgram& source,
			fragment_program_type& destination, usz id)
		{
			destination.id = static_cast<u32>(id);
			destination.program = std::make_shared<MTLFragmentProgram>();
			const render_device* renderer = get_current_renderer();
			destination.program->Decompile(source, destination.id, {
				// The RSX ISA permits mixed half/full register expressions. MSL requires explicit
				// conversions for those expressions, so use the lossless full-precision path.
				.use_native_half = false,
				.framebuffer_fetch = renderer && renderer->info().features.framebuffer_fetch,
				.log_source = g_cfg.video.log_programs.get(),
			});
			destination.constant_offsets.assign(destination.program->constant_offsets().begin(),
				destination.program->constant_offsets().end());
		}

		static void recompile_vertex_program(const RSXVertexProgram& source,
			vertex_program_type& destination, usz id)
		{
			destination.id = static_cast<u32>(id);
			destination.program = std::make_shared<MTLVertexProgram>();
			destination.program->id = destination.id;
			destination.program->Decompile(source, {
				.emulate_conditional_rendering = emulate_conditional_rendering(),
				.log_source = g_cfg.video.log_programs.get(),
			});
		}

		static void validate_pipeline_properties(const vertex_program_type& vertex,
			const fragment_program_type& fragment, pipeline_properties& properties)
		{
			if (!vertex.program || !fragment.program)
				fmt::throw_exception("Metal program cache contains an empty shader program");
			auto& render = properties.state.render;
			const auto& masks = fragment.program->metadata().output_color_masks;
			for (u32 index = 0; index < render.color_attachment_count; ++index)
			{
				render.color_attachments[index].write_mask &= static_cast<u8>(masks[index]);
			}
		}

		static pipeline_type* build_pipeline(const vertex_program_type& vertex,
			const fragment_program_type& fragment, const pipeline_properties& properties,
			bool compile_async, std::function<pipeline_type*(pipeline_storage_type&)> callback,
			MTLPipelineCompiler& compiler)
		{
			if (!vertex.program || !fragment.program || !callback)
				fmt::throw_exception("Invalid Metal cached pipeline build request");
			pipeline_properties normalized = properties;
			validate_pipeline_properties(vertex, fragment, normalized);
			graphics_pipeline_compile_request request{vertex.program, fragment.program, std::move(normalized)};
			if (!compile_async)
			{
				auto pipeline = compiler.compile_graphics_inline(request);
				return callback(pipeline);
			}
			(void)compiler.submit_graphics(std::move(request), pipeline_compile_priority::normal,
				[callback = std::move(callback)](const pipeline_compile_job& job) mutable
				{
					auto pipeline = job.result();
					if (!pipeline)
					{
						rsx_log.error("Metal cached pipeline compilation failed: %s", job.diagnostic());
					}
					callback(pipeline);
				});
			return nullptr;
		}
	};

	class MTLProgramBuffer final : public program_state_cache<MTLProgramTraits>
	{
		using base = program_state_cache<MTLProgramTraits>;
		MTLPipelineCompiler* m_compiler = nullptr;

	public:
		MTLProgramBuffer() = default;
		explicit MTLProgramBuffer(MTLPipelineCompiler& compiler, decompiler_callback_t callback = {})
		{
			initialize(compiler, std::move(callback));
		}

		~MTLProgramBuffer()
		{
			if (m_compiler && *m_compiler) m_compiler->wait_idle();
		}

		void initialize(MTLPipelineCompiler& compiler, decompiler_callback_t callback = {})
		{
			if (!compiler) fmt::throw_exception("Metal program buffer requires an initialized pipeline compiler");
			if (m_compiler && m_compiler != &compiler) clear();
			m_compiler = &compiler;
			notify_pipeline_compiled = std::move(callback);
		}

		template <typename... Args>
		auto get_graphics_pipeline(rsx::program_cache_hint_t* cache_hint,
			const RSXVertexProgram& vertex, const RSXFragmentProgram& fragment,
			graphics_pipeline_configuration& properties, bool compile_async,
			bool allow_notification, Args&&... args)
		{
			if (!m_compiler || !*m_compiler)
				fmt::throw_exception("Metal program buffer is not initialized");
			return base::get_graphics_pipeline(cache_hint, vertex, fragment, properties,
				compile_async, allow_notification, std::forward<Args>(args)..., *m_compiler);
		}

		void clear()
		{
			if (m_compiler && *m_compiler) m_compiler->wait_idle();
			base::clear();
		}

		[[nodiscard]] u64 get_hash(const graphics_pipeline_configuration& properties) const
		{
			return properties.signature();
		}

		[[nodiscard]] u64 get_hash(const RSXVertexProgram& program) const
		{
			return program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(program);
		}

		[[nodiscard]] u64 get_hash(const RSXFragmentProgram& program) const
		{
			return program_hash_util::fragment_program_utils::get_fragment_program_ucode_hash(program);
		}

		template <typename... Args>
		void add_pipeline_entry(const RSXVertexProgram& vertex, const RSXFragmentProgram& fragment,
			graphics_pipeline_configuration& properties, Args&&... args)
		{
			get_graphics_pipeline(nullptr, vertex, fragment, properties, false, false,
				std::forward<Args>(args)...);
		}

		void preload_programs(rsx::program_cache_hint_t* cache_hint,
			const RSXVertexProgram& vertex, const RSXFragmentProgram& fragment)
		{
			search_vertex_program(cache_hint, vertex);
			search_fragment_program(cache_hint, fragment);
		}

		[[nodiscard]] bool check_cache_missed() const
		{
			return m_cache_miss_flag;
		}
	};
}

namespace rpcs3
{
	template <>
	inline usz hash_struct<mtl::graphics_pipeline_configuration>(
		const mtl::graphics_pipeline_configuration& properties)
	{
		return static_cast<usz>(properties.signature());
	}
}
