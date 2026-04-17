#include "stdafx.h"
#include "MTLShaderRecompiler.h"

#include "MTLMSL.h"
#include "MTLShaderCache.h"

#include "Emu/RSX/Program/FragmentProgramDecompiler.h"
#include "Emu/RSX/Program/ProgramStateCache.h"
#include "Emu/RSX/Program/RSXFragmentProgram.h"
#include "Emu/RSX/Program/RSXVertexProgram.h"
#include "Emu/RSX/Program/VertexProgramDecompiler.h"
#include "Utilities/File.h"

#include <array>

namespace
{
	constexpr std::array<std::string_view, 16> s_vertex_inputs =
	{
		"in_pos",
		"in_weight",
		"in_normal",
		"in_diff_color",
		"in_spec_color",
		"in_fog",
		"in_point_size",
		"in_7",
		"in_tc0",
		"in_tc1",
		"in_tc2",
		"in_tc3",
		"in_tc4",
		"in_tc5",
		"in_tc6",
		"in_tc7",
	};

	int fragment_input_index(std::string_view name)
	{
		rsx_log.trace("fragment_input_index(name=%s)", name);

		static constexpr std::array<std::string_view, 15> s_fragment_inputs =
		{
			"wpos",
			"diff_color",
			"spec_color",
			"fogc",
			"tc0",
			"tc1",
			"tc2",
			"tc3",
			"tc4",
			"tc5",
			"tc6",
			"tc7",
			"tc8",
			"tc9",
			"ssa",
		};

		for (u32 index = 0; index < s_fragment_inputs.size(); index++)
		{
			if (s_fragment_inputs[index] == name)
			{
				return index;
			}
		}

		fmt::throw_exception("Unknown Metal fragment input '%s'", name);
	}

	b8 has_sampler_uniforms(const ParamArray& params)
	{
		rsx_log.trace("has_sampler_uniforms()");

		for (const ParamType& param_type : params.params[PF_PARAM_UNIFORM])
		{
			if (param_type.type.starts_with("sampler"))
			{
				return true;
			}
		}

		return false;
	}

	std::string shader_cache_path(const rsx::metal::persistent_shader_cache& cache, const char* suffix, u64 hash)
	{
		rsx_log.trace("shader_cache_path(suffix=%s, hash=0x%x)", suffix, hash);
		return cache.msl_path() + fmt::format("%llX.%s.msl", hash, suffix);
	}

	void store_shader_source(const std::string& path, const std::string& source)
	{
		rsx_log.trace("store_shader_source(path=%s, size=0x%x)", path, source.size());

		if (!fs::write_file(path, fs::rewrite, source))
		{
			fmt::throw_exception("Failed to write Metal shader source '%s' (%s)", path, fs::g_tls_error);
		}
	}

	class metal_vertex_decompiler final : public VertexProgramDecompiler
	{
	public:
		metal_vertex_decompiler(const RSXVertexProgram& program, u32 id)
			: VertexProgramDecompiler(program)
			, m_entry_point(fmt::format("rpcs3_mtl_vp_%u", id))
		{
			rsx_log.trace("metal_vertex_decompiler::metal_vertex_decompiler(id=%u)", id);
		}

		std::string translate()
		{
			rsx_log.trace("metal_vertex_decompiler::translate()");

			std::string source = Decompile();

			if (properties.has_lit_op)
			{
				fmt::throw_exception("Metal vertex shader translation does not yet support RSX LIT emulation");
			}

			if (has_sampler_uniforms(m_parr))
			{
				fmt::throw_exception("Metal vertex shader translation does not yet support vertex texture fetch");
			}

			return source;
		}

		const std::string& entry_point() const
		{
			rsx_log.trace("metal_vertex_decompiler::entry_point()");
			return m_entry_point;
		}

	private:
		std::string getFloatTypeName(usz element_count) override
		{
			return rsx::metal::msl::get_float_type_name(element_count);
		}

		std::string getIntTypeName(usz element_count) override
		{
			return rsx::metal::msl::get_int_type_name(element_count);
		}

		std::string getFunction(FUNCTION function) override
		{
			return rsx::metal::msl::get_function(function);
		}

		std::string compareFunction(COMPARE function, const std::string& op0, const std::string& op1, bool scalar) override
		{
			return rsx::metal::msl::compare_function(function, op0, op1, scalar);
		}

		void insertHeader(std::stringstream& stream) override
		{
			rsx_log.trace("metal_vertex_decompiler::insertHeader()");

			rsx::metal::msl::insert_header(stream);
			rsx::metal::msl::insert_legacy_helpers(stream);
			stream <<
				"struct rpcs3_mtl_vertex_context\n"
				"{\n"
				"	float4 input[16];\n"
				"	float4 output[16];\n"
				"	constant float4* constants;\n"
				"};\n\n";
		}

		void insertInputs(std::stringstream&, const std::vector<ParamType>&) override
		{
			rsx_log.trace("metal_vertex_decompiler::insertInputs()");
		}

		void insertConstants(std::stringstream&, const std::vector<ParamType>&) override
		{
			rsx_log.trace("metal_vertex_decompiler::insertConstants()");
		}

		void insertOutputs(std::stringstream&, const std::vector<ParamType>&) override
		{
			rsx_log.trace("metal_vertex_decompiler::insertOutputs()");
		}

		void insertMainStart(std::stringstream& stream) override
		{
			rsx_log.trace("metal_vertex_decompiler::insertMainStart()");

			stream << "void " << m_entry_point << "(thread rpcs3_mtl_vertex_context& ctx)\n";
			stream << "{\n";
			stream << "#define _fetch_constant(index) ctx.constants[index]\n";

			for (const ParamType& param_type : m_parr.params[PF_PARAM_OUT])
			{
				for (const ParamItem& item : param_type.items)
				{
					stream << "\t" << param_type.type << " " << item.name;
					if (!item.value.empty())
					{
						stream << " = " << item.value;
					}
					stream << ";\n";
				}
			}

			for (const ParamType& param_type : m_parr.params[PF_PARAM_NONE])
			{
				for (const ParamItem& item : param_type.items)
				{
					if (item.name.starts_with("dst_reg"))
					{
						continue;
					}

					stream << "\t" << param_type.type << " " << item.name;
					if (!item.value.empty())
					{
						stream << " = " << item.value;
					}
					stream << ";\n";
				}
			}

			for (const ParamType& param_type : m_parr.params[PF_PARAM_IN])
			{
				for (const ParamItem& item : param_type.items)
				{
					ensure(item.location >= 0 && item.location < static_cast<int>(s_vertex_inputs.size()));
					stream << "\t" << param_type.type << " " << item.name << " = ctx.input[" << item.location << "];\n";
				}
			}

			stream << "\n";
		}

		void insertMainEnd(std::stringstream& stream) override
		{
			rsx_log.trace("metal_vertex_decompiler::insertMainEnd()");

			for (u32 index = 0; index < 16; index++)
			{
				const std::string output_name = fmt::format("dst_reg%u", index);
				if (m_parr.HasParam(PF_PARAM_OUT, "float4", output_name))
				{
					stream << "\tctx.output[" << index << "] = " << output_name << ";\n";
				}
			}

			stream << "#undef _fetch_constant\n";
			stream << "}\n";
		}

		std::string m_entry_point;
	};

	class metal_fragment_decompiler final : public FragmentProgramDecompiler
	{
	public:
		metal_fragment_decompiler(const RSXFragmentProgram& program, u32& size, u32 id)
			: FragmentProgramDecompiler(program, size)
			, m_entry_point(fmt::format("rpcs3_mtl_fp_%u", id))
		{
			rsx_log.trace("metal_fragment_decompiler::metal_fragment_decompiler(id=%u)", id);
			device_props.has_native_half_support = false;
		}

		std::string translate()
		{
			rsx_log.trace("metal_fragment_decompiler::translate()");

			std::string source = Decompile();

			if (properties.has_discard_op)
			{
				fmt::throw_exception("Metal fragment shader translation does not yet support discard");
			}

			if (properties.has_tex_op || has_sampler_uniforms(m_parr))
			{
				fmt::throw_exception("Metal fragment shader translation does not yet support texture sampling");
			}

			if (properties.has_lit_op || properties.has_divsq || properties.has_dynamic_register_load)
			{
				fmt::throw_exception("Metal fragment shader translation encountered an RSX feature that is not mapped to MSL yet");
			}

			return source;
		}

		const std::string& entry_point() const
		{
			rsx_log.trace("metal_fragment_decompiler::entry_point()");
			return m_entry_point;
		}

	private:
		std::string getFloatTypeName(usz element_count) override
		{
			return rsx::metal::msl::get_float_type_name(element_count);
		}

		std::string getHalfTypeName(usz element_count) override
		{
			return rsx::metal::msl::get_half_type_name(element_count);
		}

		std::string getFunction(FUNCTION function) override
		{
			return rsx::metal::msl::get_function(function);
		}

		std::string compareFunction(COMPARE function, const std::string& op0, const std::string& op1) override
		{
			return rsx::metal::msl::compare_function(function, op0, op1, false);
		}

		void insertHeader(std::stringstream& stream) override
		{
			rsx_log.trace("metal_fragment_decompiler::insertHeader()");

			rsx::metal::msl::insert_header(stream);
			rsx::metal::msl::insert_legacy_helpers(stream);
			stream <<
				"struct rpcs3_mtl_fragment_context\n"
				"{\n"
				"	float4 input[15];\n"
				"	float4 output[4];\n"
				"	float depth;\n"
				"	constant float4* constants;\n"
				"	bool discarded;\n"
				"};\n\n";
		}

		void insertInputs(std::stringstream&) override
		{
			rsx_log.trace("metal_fragment_decompiler::insertInputs()");
		}

		void insertOutputs(std::stringstream&) override
		{
			rsx_log.trace("metal_fragment_decompiler::insertOutputs()");
		}

		void insertConstants(std::stringstream&) override
		{
			rsx_log.trace("metal_fragment_decompiler::insertConstants()");
		}

		void insertGlobalFunctions(std::stringstream&) override
		{
			rsx_log.trace("metal_fragment_decompiler::insertGlobalFunctions()");
		}

		void insertMainStart(std::stringstream& stream) override
		{
			rsx_log.trace("metal_fragment_decompiler::insertMainStart()");

			stream << "void " << m_entry_point << "(thread rpcs3_mtl_fragment_context& ctx)\n";
			stream << "{\n";
			stream << "#define _fetch_constant(index) ctx.constants[index]\n";
			stream << "#define _kill() do { ctx.discarded = true; return; } while (false)\n";
			stream << "\tfloat4 gl_FragCoord = ctx.input[0];\n";

			for (const ParamType& param_type : m_parr.params[PF_PARAM_NONE])
			{
				for (const ParamItem& item : param_type.items)
				{
					stream << "\t" << param_type.type << " " << item.name;
					if (!item.value.empty())
					{
						stream << " = " << item.value;
					}
					stream << ";\n";
				}
			}

			for (const ParamType& param_type : m_parr.params[PF_PARAM_IN])
			{
				for (const ParamItem& item : param_type.items)
				{
					stream << "\t" << param_type.type << " " << item.name << " = ctx.input[" << fragment_input_index(item.name) << "];\n";
				}
			}

			stream << "\n";
		}

		void insertMainEnd(std::stringstream& stream) override
		{
			rsx_log.trace("metal_fragment_decompiler::insertMainEnd()");

			static constexpr std::array<std::pair<std::string_view, u32>, 4> s_fp32_outputs =
			{ {
				{ "r0", 0 },
				{ "r2", 1 },
				{ "r3", 2 },
				{ "r4", 3 },
			} };

			static constexpr std::array<std::pair<std::string_view, u32>, 4> s_fp16_outputs =
			{ {
				{ "h0", 0 },
				{ "h4", 1 },
				{ "h6", 2 },
				{ "h8", 3 },
			} };

			const auto& outputs = (m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS) ? s_fp32_outputs : s_fp16_outputs;
			for (const auto& [name, index] : outputs)
			{
				if (index >= m_prog.mrt_buffers_count)
				{
					continue;
				}

				if (m_parr.HasParam(PF_PARAM_NONE, "float4", std::string(name)))
				{
					stream << "\tctx.output[" << index << "] = " << name << ";\n";
				}
			}

			if (m_prog.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT)
			{
				stream << "\tctx.depth = r1.z;\n";
			}

			stream << "#undef _kill\n";
			stream << "#undef _fetch_constant\n";
			stream << "}\n";
		}

		std::string m_entry_point;
	};
}

namespace rsx::metal
{
	shader_recompiler::shader_recompiler(persistent_shader_cache& cache)
		: m_cache(cache)
	{
		rsx_log.notice("rsx::metal::shader_recompiler::shader_recompiler()");
	}

	shader_recompiler::~shader_recompiler()
	{
		rsx_log.notice("rsx::metal::shader_recompiler::~shader_recompiler()");
	}

	translated_shader shader_recompiler::translate_vertex_program(const RSXVertexProgram& program, u32 id)
	{
		rsx_log.notice("rsx::metal::shader_recompiler::translate_vertex_program(id=%u)", id);

		if (program.data.empty())
		{
			fmt::throw_exception("Metal vertex shader translation requires non-empty RSX bytecode");
		}

		metal_vertex_decompiler decompiler(program, id);
		const std::string source = decompiler.translate();
		const u64 hash = static_cast<u64>(program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(program));
		const std::string path = shader_cache_path(m_cache, "vp", hash);
		store_shader_source(path, source);

		return
		{
			.stage = shader_stage::vertex,
			.id = id,
			.source_hash = hash,
			.entry_point = decompiler.entry_point(),
			.source = source,
			.cache_path = path,
		};
	}

	translated_shader shader_recompiler::translate_fragment_program(const RSXFragmentProgram& program, u32 id)
	{
		rsx_log.notice("rsx::metal::shader_recompiler::translate_fragment_program(id=%u)", id);

		if (!program.ucode_length)
		{
			fmt::throw_exception("Metal fragment shader translation requires non-empty RSX bytecode");
		}

		u32 size = 0;
		metal_fragment_decompiler decompiler(program, size, id);
		const std::string source = decompiler.translate();
		const u64 hash = static_cast<u64>(program_hash_util::fragment_program_utils::get_fragment_program_ucode_hash(program));
		const std::string path = shader_cache_path(m_cache, "fp", hash);
		store_shader_source(path, source);

		return
		{
			.stage = shader_stage::fragment,
			.id = id,
			.source_hash = hash,
			.entry_point = decompiler.entry_point(),
			.source = source,
			.cache_path = path,
		};
	}

	void shader_recompiler::report() const
	{
		rsx_log.notice("rsx::metal::shader_recompiler::report()");
		rsx_log.notice("Metal shader recompiler: MSL helper-function translation enabled for non-textured RSX vertex/fragment programs");
		rsx_log.warning("Metal shader recompiler: texture sampling, discard, advanced fragment control flow, and mesh wrappers are gated until MSL/resource bindings are implemented");
	}
}
