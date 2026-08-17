#include "stdafx.h"
#include "MTLCommonPipelineLayout.h"

#include <mutex>
#include <string>

namespace mtl
{
	namespace
	{
		u8 stage_visibility(msl_shader_stage stage)
		{
			switch (stage)
			{
			case msl_shader_stage::vertex: return argument_stage_vertex;
			case msl_shader_stage::fragment: return argument_stage_fragment;
			case msl_shader_stage::compute: return argument_stage_compute;
			}
			fmt::throw_exception("Invalid Metal shader stage");
		}

		common_argument_table_index stage_table_index(msl_shader_stage stage)
		{
			return stage == msl_shader_stage::fragment ? common_argument_table_index::fragment :
				stage == msl_shader_stage::compute ? common_argument_table_index::compute :
				common_argument_table_index::vertex;
		}

		argument_table_layout stage_layout(msl_shader_stage stage, const argument_table_layout& compute_layout)
		{
			switch (stage)
			{
			case msl_shader_stage::vertex: return vertex_stage_binding_table::layout();
			case msl_shader_stage::fragment: return fragment_stage_binding_table::layout();
			case msl_shader_stage::compute:
				compute_layout.validate();
				return compute_layout;
			}
			fmt::throw_exception("Invalid Metal shader stage");
		}

		u32 binding_count(const argument_table_layout& layout, argument_binding_class resource)
		{
			switch (resource)
			{
			case argument_binding_class::buffer: return layout.buffer_count;
			case argument_binding_class::texture: return layout.texture_count;
			case argument_binding_class::sampler: return layout.sampler_count;
			}
			fmt::throw_exception("Invalid Metal argument binding class");
		}
	}

	void common_argument_table_definition::validate() const
	{
		layout.validate();
		if (visibility != stage_visibility(stage) || table_index != stage_table_index(stage) || canonical_label.empty())
		{
			fmt::throw_exception("Invalid Metal common argument-table definition");
		}
	}

	u64 common_argument_table_definition::signature() const
	{
		validate();
		u64 result = layout.signature();
		result |= static_cast<u64>(visibility) << 32;
		result |= static_cast<u64>(stage) << 40;
		result |= static_cast<u64>(table_index) << 48;
		return result;
	}

	void common_graphics_argument_tables::reset_bindings()
	{
		if (!*this) fmt::throw_exception("Cannot reset incomplete Metal graphics argument tables");
		vertex->reset_bindings();
		fragment->reset_bindings();
	}

	void common_graphics_argument_tables::apply()
	{
		if (!*this) fmt::throw_exception("Cannot apply incomplete Metal graphics argument tables");
		vertex->apply();
		fragment->apply();
	}

	void common_graphics_argument_tables::bind(command_buffer& command)
	{
		if (!*this || !command.is_recording() || command.active_encoder() != encoder_kind::render)
		{
			fmt::throw_exception("Metal graphics argument tables require an active render encoder");
		}
		vertex->bind(command);
		fragment->bind(command);
	}

	common_graphics_argument_tables::operator bool() const
	{
		return vertex && fragment && bool(*vertex) && bool(*fragment) &&
			vertex->layout() == vertex_stage_binding_table::layout() &&
			fragment->layout() == fragment_stage_binding_table::layout() &&
			vertex->stages() == argument_stage_vertex && fragment->stages() == argument_stage_fragment;
	}

	struct MTLCommonPipelineLayout::impl
	{
		const render_device* device = nullptr;
		mutable common_pipeline_layout_statistics counters;
		mutable std::mutex mutex;

		const render_device& owner() const
		{
			if (!device) fmt::throw_exception("Metal common pipeline layout is not initialized");
			return *device;
		}

		void record(u64 common_pipeline_layout_statistics::*counter) const
		{
			std::lock_guard lock(mutex);
			counters.*counter += 1;
		}
	};

	MTLCommonPipelineLayout::MTLCommonPipelineLayout()
		: m_impl(std::make_unique<impl>())
	{
	}

	MTLCommonPipelineLayout::~MTLCommonPipelineLayout()
	{
		destroy();
	}

	void MTLCommonPipelineLayout::create(const render_device& device)
	{
		destroy();
		if (!device || !device.info().features.argument_tables)
		{
			fmt::throw_exception("Metal common pipeline layout requires argument-table support");
		}
		const auto vertex = vertex_definition();
		const auto fragment = fragment_definition();
		vertex.validate();
		fragment.validate();
		if (graphics_signature() != pipeline_binding_table::signature())
		{
			fmt::throw_exception("Metal common pipeline binding signature is inconsistent");
		}
		const auto& limits = device.info().limits;
		auto fits_device = [&](const argument_table_layout& layout)
		{
			return layout.buffer_count <= limits.max_buffers_per_argument_table &&
				layout.texture_count <= limits.max_textures_per_argument_table &&
				layout.sampler_count <= limits.max_samplers_per_argument_table;
		};
		if (!fits_device(vertex.layout) || !fits_device(fragment.layout))
		{
			fmt::throw_exception("Metal common pipeline layout exceeds device argument-table limits");
		}
		m_impl->device = &device;
	}

	void MTLCommonPipelineLayout::destroy()
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->device = nullptr;
		m_impl->counters = {};
	}

	common_argument_table_definition MTLCommonPipelineLayout::vertex_definition()
	{
		return {msl_shader_stage::vertex, vertex_stage_binding_table::layout(), argument_stage_vertex,
			common_argument_table_index::vertex, "vertex"};
	}

	common_argument_table_definition MTLCommonPipelineLayout::fragment_definition()
	{
		return {msl_shader_stage::fragment, fragment_stage_binding_table::layout(), argument_stage_fragment,
			common_argument_table_index::fragment, "fragment"};
	}

	common_argument_table_definition MTLCommonPipelineLayout::compute_definition(
		const argument_table_layout& layout)
	{
		common_argument_table_definition result{msl_shader_stage::compute, layout, argument_stage_compute,
			common_argument_table_index::compute, "compute"};
		result.validate();
		return result;
	}

	u64 MTLCommonPipelineLayout::graphics_signature()
	{
		return pipeline_binding_table::signature();
	}

	void MTLCommonPipelineLayout::validate_binding(msl_shader_stage stage,
		const shader_binding_location& binding, const argument_table_layout& compute_layout)
	{
		if (!binding || (binding.stages & stage_visibility(stage)) == 0)
		{
			fmt::throw_exception("Invalid Metal common pipeline binding location");
		}
		const argument_table_layout layout = stage_layout(stage, compute_layout);
		if (binding.index >= binding_count(layout, binding.type))
		{
			fmt::throw_exception("Metal common pipeline binding index %u exceeds its stage layout", binding.index);
		}
	}

	std::unique_ptr<argument_table> MTLCommonPipelineLayout::create_vertex_table(std::string_view label) const
	{
		const render_device& device = m_impl->owner();
		const auto definition = vertex_definition();
		auto table = std::make_unique<argument_table>();
		const std::string table_label = label.empty() ? "RPCS3 vertex arguments" : std::string(label);
		table->create(device, definition.layout, definition.visibility, table_label);
		m_impl->record(&common_pipeline_layout_statistics::vertex_tables_created);
		return table;
	}

	std::unique_ptr<argument_table> MTLCommonPipelineLayout::create_fragment_table(std::string_view label) const
	{
		const render_device& device = m_impl->owner();
		const auto definition = fragment_definition();
		auto table = std::make_unique<argument_table>();
		const std::string table_label = label.empty() ? "RPCS3 fragment arguments" : std::string(label);
		table->create(device, definition.layout, definition.visibility, table_label);
		m_impl->record(&common_pipeline_layout_statistics::fragment_tables_created);
		return table;
	}

	common_graphics_argument_tables MTLCommonPipelineLayout::create_graphics_tables(std::string_view label) const
	{
		const std::string prefix = label.empty() ? "RPCS3 graphics pipeline" : std::string(label);
		common_graphics_argument_tables result;
		result.vertex = create_vertex_table(prefix + " vertex arguments");
		result.fragment = create_fragment_table(prefix + " fragment arguments");
		m_impl->record(&common_pipeline_layout_statistics::graphics_table_pairs_created);
		return result;
	}

	std::unique_ptr<argument_table> MTLCommonPipelineLayout::create_compute_table(
		const argument_table_layout& layout, std::string_view label) const
	{
		const render_device& device = m_impl->owner();
		const auto definition = compute_definition(layout);
		auto table = std::make_unique<argument_table>();
		const std::string table_label = label.empty() ? "RPCS3 compute arguments" : std::string(label);
		table->create(device, definition.layout, definition.visibility, table_label);
		m_impl->record(&common_pipeline_layout_statistics::compute_tables_created);
		return table;
	}

	MTLCommonPipelineLayout::operator bool() const
	{
		if (!m_impl) return false;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->device != nullptr;
	}

	const render_device& MTLCommonPipelineLayout::owner() const
	{
		return m_impl->owner();
	}

	common_pipeline_layout_statistics MTLCommonPipelineLayout::statistics() const
	{
		if (!m_impl) return {};
		std::lock_guard lock(m_impl->mutex);
		return m_impl->counters;
	}
}
