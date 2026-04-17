#include "stdafx.h"
#include "MTLShaderRecompiler.h"

#include "MTLMSL.h"
#include "MTLPipelineState.h"
#include "MTLShaderCache.h"
#include "MTLShaderEntrypoint.h"
#include "MTLShaderInterface.h"

#include "Emu/RSX/Program/FragmentProgramDecompiler.h"
#include "Emu/RSX/Program/ProgramStateCache.h"
#include "Emu/RSX/Program/RSXFragmentProgram.h"
#include "Emu/RSX/Program/RSXVertexProgram.h"
#include "Emu/RSX/Program/VertexProgramDecompiler.h"
#include "Utilities/File.h"
#include "util/fnv_hash.hpp"

#include <array>

namespace
{
	const rsx::metal::shader_vertex_input_slot& vertex_input_slot_at_location(int location)
	{
		rsx_log.trace("vertex_input_slot_at_location(location=%d)", location);

		if (location < 0)
		{
			fmt::throw_exception("Metal vertex shader input location is negative: %d", location);
		}

		const u32 attribute_index = static_cast<u32>(location);
		const auto& slots = rsx::metal::vertex_input_slots();
		if (attribute_index >= slots.size())
		{
			fmt::throw_exception("Metal vertex shader input location is out of range: %u", attribute_index);
		}

		const rsx::metal::shader_vertex_input_slot& slot = slots[attribute_index];
		if (slot.attribute_index != attribute_index)
		{
			fmt::throw_exception("Metal vertex shader input slot table is not ordered at index %u", attribute_index);
		}

		return slot;
	}

	u32 fragment_input_index(std::string_view name)
	{
		rsx_log.trace("fragment_input_index(name=%s)", name);

		for (const rsx::metal::shader_named_slot& slot : rsx::metal::fragment_input_slots())
		{
			if (slot.name == name)
			{
				return slot.index;
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
		rsx_log.trace("shader_cache_path(suffix=%s, hash=0x%llx)", suffix, hash);
		return cache.msl_path() + fmt::format("%llX.%s.msl", hash, suffix);
	}

	u64 shader_source_text_hash(std::string_view source)
	{
		rsx_log.trace("shader_source_text_hash(size=0x%x)", source.size());

		usz hash = rpcs3::fnv_seed;
		for (const char c : source)
		{
			hash = rpcs3::hash64(hash, static_cast<u8>(c));
		}

		return static_cast<u64>(hash);
	}

	void validate_translated_shader_source(const char* stage, const std::string& entry_point, const std::string& source)
	{
		rsx_log.trace("validate_translated_shader_source(stage=%s, entry_point=%s, size=0x%x)",
			stage ? stage : "<null>", entry_point.c_str(), static_cast<u32>(source.size()));

		if (!stage || !stage[0])
		{
			fmt::throw_exception("Metal shader source validation requires a stage");
		}

		if (entry_point.empty())
		{
			fmt::throw_exception("Metal %s shader source validation requires an entry point", stage);
		}

		if (source.empty())
		{
			fmt::throw_exception("Metal %s shader source validation requires non-empty MSL", stage);
		}

		if (source.find("#include <metal_stdlib>") == std::string::npos)
		{
			fmt::throw_exception("Metal %s shader source is missing the MSL standard library include", stage);
		}

		const std::string function_decl = fmt::format("void %s(", entry_point);
		if (source.find(function_decl) == std::string::npos)
		{
			fmt::throw_exception("Metal %s shader source does not contain helper entry '%s'", stage, entry_point);
		}
	}

	const char* pipeline_entry_stage_suffix(rsx::metal::shader_stage stage)
	{
		rsx_log.trace("pipeline_entry_stage_suffix(stage=%u)", static_cast<u32>(stage));

		switch (stage)
		{
		case rsx::metal::shader_stage::vertex:
			return "vp";
		case rsx::metal::shader_stage::fragment:
			return "fp";
		case rsx::metal::shader_stage::mesh:
			return "mesh";
		}

		fmt::throw_exception("Unknown Metal shader stage %u", static_cast<u32>(stage));
	}

	void store_pipeline_entry_metadata(rsx::metal::persistent_shader_cache& cache, const rsx::metal::translated_shader& shader)
	{
		rsx_log.trace("store_pipeline_entry_metadata(stage=%u, id=%u, source_hash=0x%llx)",
			static_cast<u32>(shader.stage), shader.id, shader.source_hash);

		const char* stage = pipeline_entry_stage_suffix(shader.stage);
		cache.store_pipeline_entry_metadata(
			stage,
			shader.id,
			shader.source_hash,
			shader.pipeline_source_hash,
			shader.pipeline_entry_point,
			shader.pipeline_cache_path,
			shader.pipeline_entry_error,
			shader.pipeline_requirement_mask,
			shader.pipeline_entry_available);

		rsx::metal::pipeline_entry_metadata metadata;
		if (!cache.find_pipeline_entry_metadata(stage, shader.source_hash, metadata))
		{
			fmt::throw_exception("Metal pipeline entry metadata lookup failed after storing stage=%s, source_hash=0x%llx", stage, shader.source_hash);
		}

		const rsx::metal::render_pipeline_shader pipeline_shader = rsx::metal::make_render_pipeline_shader(metadata);
		if (pipeline_shader.source_hash != shader.pipeline_source_hash ||
			pipeline_shader.entry_point != shader.pipeline_entry_point ||
			pipeline_shader.entry_error != shader.pipeline_entry_error ||
			pipeline_shader.requirement_mask != shader.pipeline_requirement_mask ||
			pipeline_shader.entry_available != shader.pipeline_entry_available ||
			metadata.requirement_description != rsx::metal::describe_pipeline_entry_requirements(shader.pipeline_requirement_mask))
		{
			fmt::throw_exception("Metal pipeline entry metadata round-trip mismatch for stage=%s, source_hash=0x%llx", stage, shader.source_hash);
		}
	}

	u64 store_shader_source(const char* stage, const std::string& path, const std::string& entry_point, const std::string& source)
	{
		rsx_log.trace("store_shader_source(stage=%s, path=%s, entry_point=%s, size=0x%x)",
			stage ? stage : "<null>", path, entry_point, static_cast<u32>(source.size()));

		validate_translated_shader_source(stage, entry_point, source);
		const u64 source_text_hash = shader_source_text_hash(source);

		if (fs::is_file(path))
		{
			fs::file file{path, fs::read};
			if (!file)
			{
				fmt::throw_exception("Metal %s shader cache entry '%s' exists but is not readable", stage, path);
			}

			const std::string cached_source = file.to_string();
			if (cached_source == source)
			{
				rsx_log.trace("Metal %s shader source cache hit: %s", stage, path);
				return source_text_hash;
			}

			rsx_log.warning("Metal %s shader source cache mismatch for '%s'; rewriting cached helper MSL", stage, path);
		}

		if (!fs::write_file(path, fs::rewrite, source))
		{
			fmt::throw_exception("Failed to write Metal %s shader source '%s' (%s)", stage, path, fs::g_tls_error);
		}

		fs::file written_file{path, fs::read};
		if (!written_file || written_file.to_string() != source)
		{
			fmt::throw_exception("Metal %s shader source cache verification failed for '%s'", stage, path);
		}

		return source_text_hash;
	}

	void store_shader_source_metadata(
		rsx::metal::persistent_shader_cache& cache,
		const rsx::metal::translated_shader& shader,
		u64 source_text_hash)
	{
		rsx_log.trace("store_shader_source_metadata(stage=%u, id=%u, source_hash=0x%llx, source_text_hash=0x%llx)",
			static_cast<u32>(shader.stage), shader.id, shader.source_hash, source_text_hash);

		const char* stage = pipeline_entry_stage_suffix(shader.stage);
		cache.store_shader_source_metadata(
			stage,
			shader.id,
			shader.source_hash,
			source_text_hash,
			shader.entry_point,
			shader.cache_path);

		rsx::metal::shader_source_metadata metadata;
		if (!cache.find_shader_source_metadata(stage, shader.source_hash, source_text_hash, shader.entry_point, shader.cache_path, metadata))
		{
			fmt::throw_exception("Metal shader source metadata lookup failed after storing stage=%s, source_hash=0x%llx", stage, shader.source_hash);
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
				"	float4 input[" << rsx::metal::shader_vertex_input_count << "];\n"
				"	float4 output[" << rsx::metal::shader_vertex_output_count << "];\n"
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
					const rsx::metal::shader_vertex_input_slot& slot = vertex_input_slot_at_location(item.location);
					if (slot.name != item.name)
					{
						fmt::throw_exception("Metal vertex input location %u is named '%s' but RSX decompiler emitted '%s'",
							slot.attribute_index, std::string(slot.name), item.name);
					}

					stream << "\t" << param_type.type << " " << item.name << " = ctx.input[" << slot.attribute_index << "];\n";
				}
			}

			stream << "\n";
		}

		void insertMainEnd(std::stringstream& stream) override
		{
			rsx_log.trace("metal_vertex_decompiler::insertMainEnd()");

			for (const rsx::metal::shader_named_slot& slot : rsx::metal::vertex_output_slots())
			{
				const std::string output_name = std::string(slot.name);
				if (m_parr.HasParam(PF_PARAM_OUT, "float4", output_name))
				{
					stream << "\tctx.output[" << slot.index << "] = " << output_name << ";\n";
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
				"	float4 input[" << rsx::metal::shader_fragment_input_count << "];\n"
				"	float4 output[" << rsx::metal::shader_fragment_color_output_count << "];\n"
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
				if (index >= rsx::metal::fragment_color_output_slots().size())
				{
					fmt::throw_exception("Metal fragment output index %u exceeds color output slots", index);
				}

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
		const u64 source_text_hash = store_shader_source("vertex", path, decompiler.entry_point(), source);

		translated_shader shader =
		{
			.stage = shader_stage::vertex,
			.id = id,
			.source_hash = hash,
			.entry_point = decompiler.entry_point(),
			.source = source,
			.cache_path = path,
		};

		store_shader_source_metadata(m_cache, shader, source_text_hash);
		mark_vertex_pipeline_entry_status(shader);
		store_pipeline_entry_metadata(m_cache, shader);
		report_shader_pipeline_entry_status(shader);
		return shader;
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
		const u64 source_text_hash = store_shader_source("fragment", path, decompiler.entry_point(), source);

		translated_shader shader =
		{
			.stage = shader_stage::fragment,
			.id = id,
			.source_hash = hash,
			.entry_point = decompiler.entry_point(),
			.source = source,
			.cache_path = path,
		};

		store_shader_source_metadata(m_cache, shader, source_text_hash);
		mark_fragment_pipeline_entry_status(shader, program);
		store_pipeline_entry_metadata(m_cache, shader);
		report_shader_pipeline_entry_status(shader);
		return shader;
	}

	void shader_recompiler::report() const
	{
		rsx_log.notice("rsx::metal::shader_recompiler::report()");
		rsx_log.notice("Metal shader recompiler: MSL helper-function translation enabled for non-textured RSX vertex/fragment programs");
		rsx_log.warning("Metal shader recompiler: pipeline entry points are gated until argument-table shader binding, vertex input fetch, and transform constants are implemented");
		rsx_log.warning("Metal shader recompiler: texture sampling, discard, advanced fragment control flow, and mesh wrappers are gated until MSL/resource bindings are implemented");
	}
}
