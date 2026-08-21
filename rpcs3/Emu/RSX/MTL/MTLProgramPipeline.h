#pragma once

#include "MTLFragmentProgram.h"
#include "MTLVertexProgram.h"
#include "mtlutils/descriptors.h"
#include "mtlutils/graphics_pipeline_state.hpp"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mtl
{
	enum class program_pipeline_kind : u8
	{
		graphics,
		compute,
	};

	struct graphics_pipeline_configuration
	{
		graphics_pipeline_state state;
		std::string label;

		void validate() const;
		[[nodiscard]] u64 signature() const;
		[[nodiscard]] bool operator==(const graphics_pipeline_configuration& other) const
		{
			return state == other.state;
		}
	};

	struct program_binding_reference
	{
		msl_shader_stage stage = msl_shader_stage::vertex;
		argument_binding_class resource = argument_binding_class::buffer;
		u32 index = umax;
		u32 texture_unit = umax;
		std::string name;

		[[nodiscard]] bool operator==(const program_binding_reference&) const = default;
	};

	struct compute_pipeline_configuration
	{
		library_handle library = nullptr;
		std::string function_name;
		argument_table_layout layout;
		std::vector<program_binding_reference> required_bindings;
		u32 maximum_threads_per_threadgroup = 0;
		bool threadgroup_size_is_multiple_of_execution_width = false;
		std::string label;

		void validate() const;
	};

	struct native_graphics_stage_configuration
	{
		library_handle library = nullptr;
		std::string function_name;
		argument_table_layout layout;
		std::vector<program_binding_reference> required_bindings;
		u64 guest_program_hash = 0;
		u64 source_hash = 0;

		void validate(msl_shader_stage expected_stage) const;
	};

	struct resource_dirty_state
	{
		u32 buffers = 0;
		std::array<u64, 2> textures{};
		u16 samplers = 0;
		bool dynamic_offsets = false;

		[[nodiscard]] bool any() const;
		void clear();
	};

	struct program_pipeline_statistics
	{
		u64 binding_mutations = 0;
		u64 binding_applies = 0;
		u64 binds = 0;
		u64 required_binding_failures = 0;
	};

	class MTLProgramPipeline final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		MTLProgramPipeline();
		~MTLProgramPipeline();
		MTLProgramPipeline(const MTLProgramPipeline&) = delete;
		MTLProgramPipeline& operator=(const MTLProgramPipeline&) = delete;
		MTLProgramPipeline(MTLProgramPipeline&&) noexcept;
		MTLProgramPipeline& operator=(MTLProgramPipeline&&) noexcept;

		void create_graphics(render_device& device, const MTLVertexProgram& vertex,
			const MTLFragmentProgram& fragment, const graphics_pipeline_configuration& configuration,
			compiler_handle compiler = nullptr, pipeline_archive_handle lookup_archive = nullptr);
		void create_graphics_native(render_device& device,
			const native_graphics_stage_configuration& vertex,
			const native_graphics_stage_configuration& fragment,
			const graphics_pipeline_configuration& configuration,
			compiler_handle compiler = nullptr, pipeline_archive_handle lookup_archive = nullptr);
		void create_compute(render_device& device, const compute_pipeline_configuration& configuration,
			compiler_handle compiler = nullptr, pipeline_archive_handle lookup_archive = nullptr);
		[[nodiscard]] std::unique_ptr<MTLProgramPipeline> create_binding_instance(
			bool inherit_bindings = false) const;
		void destroy();

		void set_buffer(msl_shader_stage stage, u32 index, const argument_buffer_binding& binding);
		void set_buffer(msl_shader_stage stage, u32 index, const buffer& resource,
			u64 offset, u64 length, u32 attribute_stride = 0,
			argument_access access = argument_access::read);
		void clear_buffer(msl_shader_stage stage, u32 index);
		void set_texture(msl_shader_stage stage, u32 index, const argument_texture_binding& binding);
		void set_texture(msl_shader_stage stage, u32 index, const image_view& resource,
			argument_access access = argument_access::read);
		void set_texture(msl_shader_stage stage, u32 index, const buffer_view& resource,
			argument_access access = argument_access::read);
		void clear_texture(msl_shader_stage stage, u32 index);
		void set_sampler(msl_shader_stage stage, u32 index, const argument_sampler_binding& binding);
		void set_sampler(msl_shader_stage stage, u32 index, const sampler& resource);
		void clear_sampler(msl_shader_stage stage, u32 index);
		void set_dynamic_offset(msl_shader_stage stage, u32 buffer_index, u64 offset);
		void clear_dynamic_offsets(msl_shader_stage stage);

		void apply_bindings();
		void bind(command_buffer& command);
		[[nodiscard]] std::vector<program_binding_reference> missing_required_bindings() const;
		void validate_required_bindings();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] program_pipeline_kind kind() const;
		[[nodiscard]] bool linked() const;
		[[nodiscard]] render_pipeline_handle render_pipeline() const;
		[[nodiscard]] compute_pipeline_handle compute_pipeline() const;
		[[nodiscard]] argument_table* vertex_arguments();
		[[nodiscard]] argument_table* fragment_arguments();
		[[nodiscard]] argument_table* compute_arguments();
		[[nodiscard]] std::span<const program_binding_reference> required_bindings() const;
		[[nodiscard]] std::optional<program_binding_reference> find_binding(
			msl_shader_stage stage, std::string_view name) const;
		[[nodiscard]] const resource_dirty_state& vertex_dirty_state() const;
		[[nodiscard]] const resource_dirty_state& fragment_dirty_state() const;
		[[nodiscard]] const resource_dirty_state& compute_dirty_state() const;
		[[nodiscard]] program_pipeline_statistics statistics() const;
	};
}
