#pragma once

#include "MTLCommonDecompiler.h"
#include "mtlutils/device.h"

#include <memory>
#include <string_view>

namespace mtl
{
	enum class common_argument_table_index : u8
	{
		vertex = 0,
		fragment = 1,
		compute = 0,
		maximum = 2,
	};

	struct common_argument_table_definition
	{
		msl_shader_stage stage = msl_shader_stage::vertex;
		argument_table_layout layout;
		u8 visibility = argument_stage_none;
		common_argument_table_index table_index = common_argument_table_index::vertex;
		std::string_view canonical_label;

		void validate() const;
		[[nodiscard]] u64 signature() const;
	};

	struct common_graphics_argument_tables
	{
		std::unique_ptr<argument_table> vertex;
		std::unique_ptr<argument_table> fragment;

		void reset_bindings();
		void apply();
		void bind(command_buffer& command);
		[[nodiscard]] explicit operator bool() const;
	};

	struct common_pipeline_layout_statistics
	{
		u64 graphics_table_pairs_created = 0;
		u64 vertex_tables_created = 0;
		u64 fragment_tables_created = 0;
		u64 compute_tables_created = 0;
	};

	class MTLCommonPipelineLayout final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		MTLCommonPipelineLayout();
		~MTLCommonPipelineLayout();
		MTLCommonPipelineLayout(const MTLCommonPipelineLayout&) = delete;
		MTLCommonPipelineLayout& operator=(const MTLCommonPipelineLayout&) = delete;
		MTLCommonPipelineLayout(MTLCommonPipelineLayout&&) = delete;
		MTLCommonPipelineLayout& operator=(MTLCommonPipelineLayout&&) = delete;

		void create(const render_device& device);
		void destroy();

		[[nodiscard]] static common_argument_table_definition vertex_definition();
		[[nodiscard]] static common_argument_table_definition fragment_definition();
		[[nodiscard]] static common_argument_table_definition compute_definition(
			const argument_table_layout& layout);
		[[nodiscard]] static u64 graphics_signature();
		static void validate_binding(msl_shader_stage stage, const shader_binding_location& binding,
			const argument_table_layout& compute_layout = {});

		[[nodiscard]] std::unique_ptr<argument_table> create_vertex_table(std::string_view label) const;
		[[nodiscard]] std::unique_ptr<argument_table> create_fragment_table(std::string_view label) const;
		[[nodiscard]] common_graphics_argument_tables create_graphics_tables(std::string_view label) const;
		[[nodiscard]] std::unique_ptr<argument_table> create_compute_table(
			const argument_table_layout& layout, std::string_view label) const;

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] const render_device& owner() const;
		[[nodiscard]] common_pipeline_layout_statistics statistics() const;
	};
}
