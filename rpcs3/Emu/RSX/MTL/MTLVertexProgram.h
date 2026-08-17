#pragma once

#include "MetalAPI.h"
#include "MTLCommonDecompiler.h"
#include "Emu/RSX/Program/VertexProgramDecompiler.h"

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mtl
{
	class MTLVertexProgram;

	enum class vertex_program_resource : u8
	{
		buffer,
		texture,
		sampler,
	};

	struct vertex_program_input
	{
		vertex_program_resource resource = vertex_program_resource::buffer;
		shader_binding_location binding;
		std::string name;
		u32 texture_unit = umax;
		argument_access access = argument_access::read;

		[[nodiscard]] bool operator==(const vertex_program_input&) const = default;
	};

	struct vertex_program_bindings
	{
		shader_binding_location persistent_vertex_buffer =
			vertex_stage_binding_table::buffer(vertex_stage_binding_table::persistent_vertex_buffer);
		shader_binding_location volatile_vertex_buffer =
			vertex_stage_binding_table::buffer(vertex_stage_binding_table::volatile_vertex_buffer);
		shader_binding_location draw_parameters_buffer =
			vertex_stage_binding_table::buffer(vertex_stage_binding_table::draw_parameters_buffer);
		shader_binding_location context_buffer =
			vertex_stage_binding_table::buffer(vertex_stage_binding_table::context_buffer);
		shader_binding_location conditional_render_predicate_buffer;
		shader_binding_location constants_buffer;
		shader_binding_location instancing_lookup_buffer;
		shader_binding_location instancing_constants_buffer;
		std::array<shader_binding_location, vertex_texture_unit_count> textures{};
		std::array<shader_binding_location, vertex_texture_unit_count> samplers{};
		u8 texture_mask = 0;
		bool uses_conditional_rendering = false;
		bool uses_instanced_constants = false;

		void validate() const;
		[[nodiscard]] u64 signature() const;
	};

	struct vertex_compile_options
	{
		bool emulate_conditional_rendering = false;
		bool use_native_half = true;
		bool low_precision_comparisons = false;
		bool log_source = false;
		std::string label;
	};

	struct vertex_program_metadata
	{
		u32 control = 0;
		u32 output_mask = 0;
		u16 referenced_inputs_mask = 0;
		u16 varying_mask = 0;
		u8 referenced_textures_mask = 0;
		u32 instruction_count = 0;
		u32 constant_count = 0;
		bool has_lit_op = false;
		bool has_indexed_constants = false;
		bool uses_instanced_constants = false;
		bool uses_conditional_rendering = false;
		u64 source_hash = 0;
	};

	enum class vertex_compile_state : u8
	{
		empty,
		decompiled,
		compiled,
		failed,
	};

	class MTLVertexDecompilerThread final : public VertexProgramDecompiler
	{
		std::string& m_shader;
		MTLVertexProgram& m_destination;
		const RSXVertexProgram& m_rsx_program;
		vertex_compile_options m_options;
		std::vector<vertex_program_input> m_inputs;

		std::string getFloatTypeName(usz element_count) override;
		std::string getIntTypeName(usz element_count) override;
		std::string getFunction(FUNCTION function) override;
		std::string compareFunction(COMPARE comparison, std::string_view left,
			std::string_view right, bool scalar) override;

		void insertHeader(std::stringstream& output) override;
		void insertInputs(std::stringstream& output, const std::vector<ParamType>& inputs) override;
		void insertConstants(std::stringstream& output, const std::vector<ParamType>& constants) override;
		void insertOutputs(std::stringstream& output, const std::vector<ParamType>& outputs) override;
		void insertMainStart(std::stringstream& output) override;
		void insertMainEnd(std::stringstream& output) override;
		void prepare_binding_table();

	public:
		MTLVertexDecompilerThread(const RSXVertexProgram& program, std::string& shader,
			MTLVertexProgram& destination, const vertex_compile_options& options);

		void Task();
		[[nodiscard]] const std::vector<vertex_program_input>& inputs() const;
	};

	class MTLVertexProgram final : public rsx::VertexProgramBase
	{
		struct impl;
		std::unique_ptr<impl> m_impl;
		RSXVertexProgram m_rsx_program;
		ParamArray m_parameters;
		std::string m_source;
		std::vector<vertex_program_input> m_inputs;
		vertex_program_bindings m_bindings;
		vertex_program_metadata m_metadata;
		vertex_compile_options m_options;
		vertex_compile_state m_state = vertex_compile_state::empty;
		std::string m_diagnostic;

		friend class MTLVertexDecompilerThread;

	public:
		MTLVertexProgram();
		~MTLVertexProgram();
		MTLVertexProgram(const MTLVertexProgram&) = delete;
		MTLVertexProgram& operator=(const MTLVertexProgram&) = delete;
		MTLVertexProgram(MTLVertexProgram&&) noexcept;
		MTLVertexProgram& operator=(MTLVertexProgram&&) noexcept;

		void Decompile(const RSXVertexProgram& program,
			const vertex_compile_options& options = {});
		void Compile(compiler_handle compiler);
		void reset();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] vertex_compile_state state() const;
		[[nodiscard]] const RSXVertexProgram& rsx_program() const;
		[[nodiscard]] const ParamArray& parameters() const;
		[[nodiscard]] const std::string& source() const;
		[[nodiscard]] const std::string& diagnostic() const;
		[[nodiscard]] const vertex_program_bindings& bindings() const;
		[[nodiscard]] const vertex_program_metadata& metadata() const;
		[[nodiscard]] std::span<const vertex_program_input> inputs() const;
		[[nodiscard]] library_handle library() const;
		[[nodiscard]] function_handle function() const;
	};
}
