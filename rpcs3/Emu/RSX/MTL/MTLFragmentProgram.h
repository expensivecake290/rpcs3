#pragma once

#include "MetalAPI.h"
#include "MTLCommonDecompiler.h"
#include "Emu/RSX/Program/FragmentProgramDecompiler.h"

#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mtl
{
	class MTLFragmentProgram;

	enum class fragment_program_resource : u8
	{
		buffer,
		texture,
		sampler,
	};

	struct fragment_program_input
	{
		fragment_program_resource resource = fragment_program_resource::buffer;
		shader_binding_location binding;
		std::string name;
		u32 texture_unit = umax;
		argument_access access = argument_access::read;

		[[nodiscard]] bool operator==(const fragment_program_input&) const = default;
	};

	struct fragment_program_bindings
	{
		shader_binding_location state_buffer =
			fragment_stage_binding_table::buffer(fragment_stage_binding_table::state_buffer);
		shader_binding_location constants_buffer;
		shader_binding_location texture_parameters_buffer =
			fragment_stage_binding_table::buffer(fragment_stage_binding_table::texture_parameters_buffer);
		shader_binding_location rasterizer_environment_buffer =
			fragment_stage_binding_table::buffer(fragment_stage_binding_table::rasterizer_environment_buffer);
		shader_binding_location sampler_state_buffer =
			fragment_stage_binding_table::buffer(fragment_stage_binding_table::sampler_state_buffer);
		std::array<shader_binding_location, fragment_texture_unit_count> textures{};
		std::array<shader_binding_location, fragment_texture_unit_count> stencil_textures{};
		std::array<shader_binding_location, fragment_texture_unit_count> samplers{};
		shader_binding_location depth_input_texture;
		u16 texture_mask = 0;
		u16 stencil_texture_mask = 0;
		u16 sampler_mask = 0;
		bool uses_depth_input = false;

		void validate() const;
		[[nodiscard]] u64 signature() const;
	};

	struct fragment_compile_options
	{
		bool use_native_half = true;
		bool low_precision_comparisons = false;
		bool emulate_shadow_compare = false;
		bool round_8bit_outputs = true;
		bool framebuffer_fetch = false;
		bool log_source = false;
		std::string label;
	};

	struct fragment_program_metadata
	{
		u32 control = 0;
		u16 input_mask = 0;
		u16 common_texture_mask = 0;
		u16 shadow_texture_mask = 0;
		u16 redirected_texture_mask = 0;
		u16 multisampled_texture_mask = 0;
		std::array<u32, 4> output_color_masks{};
		u32 instruction_bytes = 0;
		u32 constant_count = 0;
		bool writes_depth = false;
		bool uses_discard = false;
		bool uses_half = false;
		bool has_no_output = false;
		u64 source_hash = 0;
	};

	enum class fragment_compile_state : u8
	{
		empty,
		decompiled,
		compiled,
		failed,
	};

	class MTLFragmentDecompilerThread final : public FragmentProgramDecompiler
	{
		std::string& m_shader;
		MTLFragmentProgram& m_destination;
		fragment_compile_options m_options;
		std::vector<fragment_program_input> m_inputs;
		glsl::shader_properties m_shader_properties{};

		std::string getFloatTypeName(usz element_count) override;
		std::string getHalfTypeName(usz element_count) override;
		std::string getFunction(FUNCTION function) override;
		std::string compareFunction(COMPARE comparison, std::string_view left,
			std::string_view right) override;

		void insertHeader(std::stringstream& output) override;
		void insertInputs(std::stringstream& output) override;
		void insertOutputs(std::stringstream& output) override;
		void insertConstants(std::stringstream& output) override;
		void insertGlobalFunctions(std::stringstream& output) override;
		void insertMainStart(std::stringstream& output) override;
		void insertMainEnd(std::stringstream& output) override;
		void prepare_binding_table();

	public:
		MTLFragmentDecompilerThread(std::string& shader, const RSXFragmentProgram& program,
			u32& size, MTLFragmentProgram& destination, const fragment_compile_options& options);

		void Task();
		[[nodiscard]] const std::vector<fragment_program_input>& inputs() const;
		[[nodiscard]] const glsl::shader_properties& shader_properties() const;
	};

	class MTLFragmentProgram final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;
		RSXFragmentProgram m_rsx_program;
		ParamArray m_parameters;
		std::string m_source;
		std::vector<fragment_program_input> m_inputs;
		std::vector<u32> m_constant_offsets;
		fragment_program_bindings m_bindings;
		fragment_program_metadata m_metadata;
		fragment_compile_options m_options;
		fragment_compile_state m_state = fragment_compile_state::empty;
		std::string m_diagnostic;
		u32 m_id = 0;

		friend class MTLFragmentDecompilerThread;

	public:
		MTLFragmentProgram();
		~MTLFragmentProgram();
		MTLFragmentProgram(const MTLFragmentProgram&) = delete;
		MTLFragmentProgram& operator=(const MTLFragmentProgram&) = delete;
		MTLFragmentProgram(MTLFragmentProgram&&) noexcept;
		MTLFragmentProgram& operator=(MTLFragmentProgram&&) noexcept;

		void Decompile(const RSXFragmentProgram& program, u32 id,
			const fragment_compile_options& options = {});
		void Compile(compiler_handle compiler);
		void reset();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] u32 id() const;
		[[nodiscard]] fragment_compile_state state() const;
		[[nodiscard]] const RSXFragmentProgram& rsx_program() const;
		[[nodiscard]] const ParamArray& parameters() const;
		[[nodiscard]] const std::string& source() const;
		[[nodiscard]] const std::string& diagnostic() const;
		[[nodiscard]] const fragment_program_bindings& bindings() const;
		[[nodiscard]] const fragment_program_metadata& metadata() const;
		[[nodiscard]] std::span<const fragment_program_input> inputs() const;
		[[nodiscard]] std::span<const u32> constant_offsets() const;
		[[nodiscard]] library_handle library() const;
		[[nodiscard]] function_handle function() const;
		[[nodiscard]] bool uses_framebuffer_fetch() const;
	};
}
