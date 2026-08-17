#pragma once

#include "MTLPipelineCompiler.h"
#include "Emu/RSX/Program/ShaderInterpreter.h"

#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace mtl
{
	enum class interpreter_alpha_test : u8
	{
		disabled,
		never,
		greater_equal,
		greater,
		less_equal,
		less,
		equal,
		not_equal,
	};

	struct interpreter_fragment_metadata
	{
		u16 referenced_textures_mask = 0;
		bool has_pack_instructions = false;
		bool has_branch_instructions = false;
	};

	struct interpreter_vertex_metadata
	{
		u32 referenced_textures_mask = 0;
	};

	struct interpreter_program_state
	{
		interpreter_fragment_metadata fragment_metadata;
		interpreter_vertex_metadata vertex_metadata;
		u32 vertex_control = 0;
		u32 fragment_control = 0;
		interpreter_alpha_test alpha_test = interpreter_alpha_test::disabled;
		bool polygon_stipple = false;

		[[nodiscard]] u32 compiler_options() const;
	};

	struct interpreter_pipeline_key
	{
		u32 compiler_options = 0;
		graphics_pipeline_configuration configuration;

		[[nodiscard]] bool operator==(const interpreter_pipeline_key& other) const
		{
			return compiler_options == other.compiler_options && configuration == other.configuration;
		}
		[[nodiscard]] u64 hash() const;
	};

	struct interpreter_pipeline_key_hash
	{
		[[nodiscard]] usz operator()(const interpreter_pipeline_key& key) const noexcept;
	};

	struct interpreter_pipeline_info
	{
		u32 vertex_instruction_buffer = 9;
		u32 fragment_instruction_buffer = 5;
		u32 fragment_texture_first = 0;
		u32 fragment_texture_count = 64;
		u32 fragment_sampler_first = 0;
		u32 fragment_sampler_count = 16;
	};

	struct interpreter_shader_sources
	{
		std::string vertex;
		std::string fragment;
		argument_table_layout vertex_layout;
		argument_table_layout fragment_layout;
		std::vector<program_binding_reference> vertex_required_bindings;
		std::vector<program_binding_reference> fragment_required_bindings;

		void validate() const;
	};

	struct interpreter_statistics
	{
		u64 source_variants = 0;
		u64 pipelines = 0;
		u64 cache_hits = 0;
		u64 synchronous_compiles = 0;
		u64 asynchronous_compiles = 0;
		u64 failed_compiles = 0;
	};

	using interpreter_progress_callback = std::function<void(u32 completed, u32 total)>;
	using interpreter_completion_callback = std::function<void(
		const interpreter_pipeline_key&, std::shared_ptr<MTLProgramPipeline>, std::string_view diagnostic)>;

	class MTLShaderInterpreter final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		MTLShaderInterpreter();
		~MTLShaderInterpreter();
		MTLShaderInterpreter(const MTLShaderInterpreter&) = delete;
		MTLShaderInterpreter& operator=(const MTLShaderInterpreter&) = delete;
		MTLShaderInterpreter(MTLShaderInterpreter&&) = delete;
		MTLShaderInterpreter& operator=(MTLShaderInterpreter&&) = delete;

		void initialize(render_device& device, MTLPipelineCompiler& compiler);
		void destroy();
		void preload(std::span<const graphics_pipeline_configuration> configurations,
			interpreter_progress_callback progress = {});

		[[nodiscard]] std::shared_ptr<MTLProgramPipeline> get(
			const graphics_pipeline_configuration& configuration,
			const interpreter_program_state& program_state,
			bool asynchronous = false,
			interpreter_completion_callback completion = {});
		[[nodiscard]] std::shared_ptr<MTLProgramPipeline> current_pipeline() const;
		[[nodiscard]] bool is_interpreter(const MTLProgramPipeline* pipeline) const;
		[[nodiscard]] std::pair<interpreter_shader_sources, interpreter_pipeline_info>
			variant(u32 compiler_options) const;

		void set_vertex_instruction_buffer(const buffer& resource, u64 offset, u64 length);
		void set_fragment_instruction_buffer(const buffer& resource, u64 offset, u64 length);
		void update_fragment_textures(u32 first_index,
			std::span<const argument_texture_binding> textures);
		void update_fragment_samplers(u32 first_index,
			std::span<const argument_sampler_binding> samplers);
		void bind(command_buffer& command);

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] interpreter_pipeline_info current_pipeline_info() const;
		[[nodiscard]] interpreter_statistics statistics() const;
	};
}
