#pragma once

#include <span>
#include <string>
#include <string_view>

#include "Utilities/StrFmt.h"
#include "Emu/RSX/Program/GLSLTypes.h"
#include "Emu/RSX/Program/ShaderParam.h"
#include "mtlutils/pipeline_binding_table.h"

namespace mtl
{
	enum class msl_shader_stage : u8
	{
		vertex,
		fragment,
		compute,
	};

	enum class msl_texture_dimension : u8
	{
		texture_1d,
		texture_2d,
		texture_3d,
		texture_cube,
	};

	struct msl_varying_location
	{
		std::string_view canonical_name;
		u32 location = 0;
		u32 components = 4;
		bool flat = false;

		[[nodiscard]] bool operator==(const msl_varying_location&) const = default;
	};

	struct msl_texture_type
	{
		msl_texture_dimension dimension = msl_texture_dimension::texture_2d;
		std::string_view sample_type = "float";
		bool array = false;
		bool multisampled = false;
		bool depth = false;
		bool writable = false;
	};

	struct msl_helper_requirements
	{
		glsl::program_domain domain = glsl::glsl_invalid_program;
		bool lit = false;
		bool comparison = false;
		bool low_precision_comparisons = false;
		bool fog = false;
		bool srgb_to_linear = false;
		bool linear_to_srgb = false;
		bool depth_conversion = false;
		bool shadow_compare = false;
		bool texture_expand = false;
		bool output_rounding = false;
		bool alpha_test = false;
		bool polygon_stipple = false;
		bool fp16 = false;
	};

	class msl_source_builder final
	{
		std::string m_source;
		u32 m_indentation = 0;

	public:
		void append(std::string_view text);
		void line(std::string_view text = {});
		void open_scope(std::string_view declaration);
		void close_scope(std::string_view suffix = {});
		[[nodiscard]] u32 indentation() const;
		[[nodiscard]] const std::string& str() const;
		[[nodiscard]] std::string take();
	};

	[[nodiscard]] msl_varying_location get_varying_register(std::string_view name);
	[[nodiscard]] int get_varying_register_location(std::string_view name);
	[[nodiscard]] int get_texture_index(std::string_view name, u32 maximum = fragment_texture_unit_count);
	[[nodiscard]] std::string sanitize_msl_identifier(std::string_view name);

	[[nodiscard]] std::string get_msl_float_type(usz components);
	[[nodiscard]] std::string get_msl_half_type(usz components);
	[[nodiscard]] std::string get_msl_int_type(usz components);
	[[nodiscard]] std::string get_msl_uint_type(usz components);
	[[nodiscard]] std::string get_msl_texture_type(const msl_texture_type& type);
	[[nodiscard]] std::string get_msl_function(FUNCTION function);
	[[nodiscard]] std::string get_msl_comparison(COMPARE comparison,
		std::string_view left, std::string_view right, bool scalar = false);

	[[nodiscard]] u32 vertex_texture_binding(u32 texture_index);
	[[nodiscard]] u32 fragment_texture_binding(u32 texture_index, bool stencil_view = false);
	[[nodiscard]] u32 vertex_sampler_binding(u32 texture_index);
	[[nodiscard]] u32 fragment_sampler_binding(u32 texture_index);

	[[nodiscard]] msl_helper_requirements get_msl_helper_requirements(
		const glsl::shader_properties& properties);
	[[nodiscard]] std::string generate_msl_prelude(msl_shader_stage stage,
		const msl_helper_requirements& requirements);
	void append_msl_helpers(msl_source_builder& builder,
		const msl_helper_requirements& requirements);
	[[nodiscard]] std::string join_msl_arguments(std::span<const std::string> arguments);
}
