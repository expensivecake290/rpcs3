#include "stdafx.h"
#include "MTLGSRender.h"

#include "Emu/RSX/Common/BufferUtils.h"
#include "Emu/RSX/rsx_methods.h"

#include <span>

namespace
{
	[[nodiscard]] bool is_line_primitive(rsx::primitive_type primitive)
	{
		return primitive == rsx::primitive_type::lines ||
			primitive == rsx::primitive_type::line_loop ||
			primitive == rsx::primitive_type::line_strip;
	}

	struct emulated_index_data
	{
		u32 count = 0;
		std::tuple<u64, mtl::index_element_type> binding;
		std::vector<u32> values;
	};

	[[nodiscard]] std::vector<u32> copy_indices(const void* data,
		mtl::index_element_type type, u32 count)
	{
		std::vector<u32> result(count);
		if (type == mtl::index_element_type::u16)
		{
			const auto* source = static_cast<const u16*>(data);
			std::copy_n(source, count, result.begin());
		}
		else
		{
			const auto* source = static_cast<const u32*>(data);
			std::copy_n(source, count, result.begin());
		}
		return result;
	}

	emulated_index_data generate_emulated_indices(
		const rsx::draw_clause& clause, u32 vertex_count, mtl::data_heap& heap)
	{
		const u32 index_count = get_index_count(clause.primitive, vertex_count);
		const mtl::data_heap_slice slice = heap.allocate(
			static_cast<u64>(index_count) * sizeof(u16), 256);
		void* data = heap.map(slice);
		g_fxo->get<rsx::dma_manager>().emulate_as_indexed(data, clause.primitive, vertex_count);
		std::vector<u32> values;
		if (is_line_primitive(clause.primitive))
			values = copy_indices(data, mtl::index_element_type::u16, index_count);
		heap.mark_modified(slice);
		heap.unmap();
		return {index_count, {slice.offset, mtl::index_element_type::u16}, std::move(values)};
	}

	struct uploaded_vertex_input
	{
		mtl::primitive_topology primitive = mtl::primitive_topology::triangle;
		bool index_rebase = false;
		bool emulated_indices = false;
		bool primitive_restart = false;
		u32 minimum_index = 0;
		u32 maximum_index = 0;
		u32 draw_count = 0;
		u32 index_offset = 0;
		std::optional<std::tuple<u64, mtl::index_element_type>> index_info;
		std::vector<u32> source_indices;
	};

	struct draw_command_visitor
	{
		mtl::data_heap& index_heap;
		rsx::vertex_input_layout& layout;

		uploaded_vertex_input operator()(const rsx::draw_array_command&) const
		{
			const auto& clause = rsx::method_registers.current_draw_clause;
			const mtl::primitive_mapping mapping = mtl::get_primitive_mapping(clause.primitive);
			const u32 vertex_count = clause.get_elements_count();
			const u32 minimum = clause.min_index();
			if (!vertex_count) return {.primitive = mapping.topology};
			if (mapping.requires_index_emulation)
			{
				const auto generated = generate_emulated_indices(clause, vertex_count, index_heap);
				return {mapping.topology, false, true, false, minimum,
					minimum + vertex_count - 1, generated.count, 0, generated.binding,
					generated.values};
			}
			std::vector<u32> source_indices;
			if (is_line_primitive(clause.primitive))
			{
				source_indices.resize(vertex_count);
				std::iota(source_indices.begin(), source_indices.end(), 0u);
			}
			return {mapping.topology, false, false, false, minimum,
				minimum + vertex_count - 1, vertex_count, 0, {}, std::move(source_indices)};
		}

		uploaded_vertex_input operator()(const rsx::draw_indexed_array_command& command) const
		{
			const auto& clause = rsx::method_registers.current_draw_clause;
			const mtl::primitive_mapping mapping = mtl::get_primitive_mapping(clause.primitive);
			const bool emulate_restart = rsx::method_registers.restart_index_enabled() &&
				mtl::emulate_primitive_restart(mapping.topology);
			const rsx::index_array_type type = clause.is_immediate_draw
				? rsx::index_array_type::u32 : rsx::method_registers.index_type();
			const u32 element_size = get_index_type_size(type);
			u32 index_count = clause.get_elements_count();
			if (mapping.requires_index_emulation)
				index_count = get_index_count(clause.primitive, index_count);
			if (!index_count) return {.primitive = mapping.topology};
			const u64 base_size = static_cast<u64>(index_count) * element_size;
			const u64 upload_size = emulate_restart ? base_size * 2 : base_size;
			const mtl::data_heap_slice slice = index_heap.allocate(upload_size, 64);
			void* output = index_heap.map(slice);

			std::span<std::byte> destination;
			stx::single_ptr<std::byte[]> temporary;
			if (emulate_restart)
			{
				temporary = stx::make_single<std::byte[], false, 64>(upload_size);
				destination = {temporary.get(), static_cast<usz>(upload_size)};
			}
			else
			{
				destination = {static_cast<std::byte*>(output), static_cast<usz>(upload_size)};
			}

			u32 minimum;
			u32 maximum;
			std::tie(minimum, maximum, index_count) = write_index_array_data_to_buffer(
				destination, command.raw_index_buffer, type, clause.primitive,
				rsx::method_registers.restart_index_enabled(),
				rsx::method_registers.restart_index(),
				[](rsx::primitive_type primitive)
				{
					return mtl::get_primitive_mapping(primitive).requires_index_emulation;
				});
			if (minimum >= maximum)
			{
				index_heap.unmap();
				return {.primitive = mapping.topology};
			}
			std::vector<u32> source_indices;
			if (is_line_primitive(clause.primitive))
			{
				const void* source = emulate_restart ? static_cast<const void*>(temporary.get()) : output;
				source_indices = copy_indices(source, mtl::get_index_type(type), index_count);
			}
			if (emulate_restart)
			{
				if (type == rsx::index_array_type::u16)
					index_count = rsx::remove_restart_index(static_cast<u16*>(output),
						reinterpret_cast<u16*>(temporary.get()), index_count, u16{umax});
				else
					index_count = rsx::remove_restart_index(static_cast<u32*>(output),
						reinterpret_cast<u32*>(temporary.get()), index_count, u32{umax});
			}
			index_heap.mark_modified(slice, 0, static_cast<u64>(index_count) * element_size);
			index_heap.unmap();
			return {mapping.topology, true, mapping.requires_index_emulation,
				!emulate_restart && rsx::method_registers.restart_index_enabled(),
				minimum, maximum, index_count,
				rsx::method_registers.vertex_data_base_index(),
				std::make_tuple(slice.offset, mtl::get_index_type(type)), std::move(source_indices)};
		}

		uploaded_vertex_input operator()(const rsx::draw_inlined_array&) const
		{
			const auto& clause = rsx::method_registers.current_draw_clause;
			const mtl::primitive_mapping mapping = mtl::get_primitive_mapping(clause.primitive);
			if (layout.interleaved_blocks.empty() || !layout.interleaved_blocks[0]->attribute_stride)
				fmt::throw_exception("Metal inline draw has no interleaved vertex stride");
			const usz stream_bytes = clause.inline_vertex_array.size() * sizeof(u32);
			const u32 vertex_count = static_cast<u32>(
				stream_bytes / layout.interleaved_blocks[0]->attribute_stride);
			if (!vertex_count) return {.primitive = mapping.topology};
			if (!mapping.requires_index_emulation)
			{
				std::vector<u32> source_indices;
				if (is_line_primitive(clause.primitive))
				{
					source_indices.resize(vertex_count);
					std::iota(source_indices.begin(), source_indices.end(), 0u);
				}
				return {mapping.topology, false, false, false, 0, vertex_count - 1,
					vertex_count, 0, {}, std::move(source_indices)};
			}
			const auto generated = generate_emulated_indices(clause, vertex_count, index_heap);
			return {mapping.topology, false, true, false, 0, vertex_count - 1,
				generated.count, 0, generated.binding, generated.values};
		}
	};

	struct line_mapping
	{
		u32 vertex_id;
		u32 other_vertex_id;
		f32 side;
		u32 reserved = 0;
	};

	static_assert(sizeof(line_mapping) == 16);

	void append_line_segment(std::vector<line_mapping>& output, u32 first, u32 second)
	{
		output.insert(output.end(), {
			{first, second, 1.f}, {second, first, -1.f}, {first, second, -1.f},
			{first, second, -1.f}, {second, first, -1.f}, {second, first, 1.f},
		});
	}

	void append_line_sequence(std::vector<line_mapping>& output, std::span<const u32> indices,
		rsx::primitive_type primitive)
	{
		if (primitive == rsx::primitive_type::lines)
		{
			for (usz index = 1; index < indices.size(); index += 2)
				append_line_segment(output, indices[index - 1], indices[index]);
		}
		else
		{
			for (usz index = 1; index < indices.size(); ++index)
				append_line_segment(output, indices[index - 1], indices[index]);
			if (primitive == rsx::primitive_type::line_loop && indices.size() > 2 &&
				indices.front() != indices.back())
				append_line_segment(output, indices.back(), indices.front());
		}
	}

	[[nodiscard]] std::vector<line_mapping> build_line_mappings(
		const uploaded_vertex_input& input, const rsx::draw_clause& draw)
	{
		std::vector<line_mapping> result;
		if (input.source_indices.empty()) return result;
		result.reserve(input.source_indices.size() * 6);

		const bool indexed_restart = draw.command == rsx::draw_command::indexed &&
			rsx::method_registers.restart_index_enabled();
		if (indexed_restart)
		{
			const mtl::index_element_type type = input.index_info
				? std::get<1>(*input.index_info) : mtl::index_element_type::u16;
			const u32 restart = type == mtl::index_element_type::u16 ? u16{umax} : u32{umax};
			usz begin = 0;
			for (usz index = 0; index <= input.source_indices.size(); ++index)
			{
				if (index != input.source_indices.size() && input.source_indices[index] != restart)
					continue;
				append_line_sequence(result,
					std::span<const u32>(input.source_indices).subspan(begin, index - begin), draw.primitive);
				begin = index + 1;
			}
			return result;
		}

		if (!draw.is_single_draw())
		{
			usz begin = 0;
			for (const auto& range : draw.get_subranges())
			{
				const usz count = std::min<usz>(range.count, input.source_indices.size() - begin);
				append_line_sequence(result,
					std::span<const u32>(input.source_indices).subspan(begin, count), draw.primitive);
				begin += count;
				if (begin == input.source_indices.size()) break;
			}
			if (begin < input.source_indices.size())
				append_line_sequence(result,
					std::span<const u32>(input.source_indices).subspan(begin), draw.primitive);
			return result;
		}

		append_line_sequence(result, input.source_indices, draw.primitive);
		return result;
	}

#pragma pack(push, 1)
	struct draw_parameters
	{
		u32 vertex_base_index;
		u32 vertex_index_offset;
		u32 draw_id;
		u32 transform_constants_offset;
		u32 vertex_context_offset;
		u32 fragment_constants_offset;
		u32 fragment_context_offset;
		u32 fragment_texture_base_index;
		u32 stipple_pattern_offset;
		u32 reserved;
		s32 attribute_data[32];
	};
#pragma pack(pop)

	static_assert(sizeof(draw_parameters) == 168);
}

mtl::vertex_upload_info MTLGSRender::upload_vertex_data()
{
	const auto& draw = rsx::method_registers.current_draw_clause;
	draw_command_visitor visitor{m_index_heap, m_vertex_layout};
	const uploaded_vertex_input input = std::visit(visitor,
		m_draw_processor.get_draw_command(rsx::method_registers));
	if (!input.draw_count)
		return {.primitive = input.primitive};

	const u32 vertex_count = input.maximum_index - input.minimum_index + 1;
	u32 vertex_base = input.minimum_index;
	u32 index_base = 0;
	if (input.index_rebase)
	{
		vertex_base = rsx::get_index_from_base(vertex_base,
			rsx::method_registers.vertex_data_base_index());
		index_base = input.minimum_index;
	}
	const auto required = calculate_memory_requirements(m_vertex_layout, vertex_base, vertex_count);
	u32 persistent_offset = 0;
	u32 volatile_offset = 0;
	mtl::data_heap_slice persistent_slice;
	mtl::data_heap_slice volatile_slice;

	if (required.first)
	{
		bool cached = false;
		bool store = false;
		u32 storage_address = 0;
		m_frame_stats.vertex_cache_request_count++;
		if (m_vertex_layout.interleaved_blocks.size() == 1 &&
			rsx::method_registers.current_draw_clause.command != rsx::draw_command::inlined_array)
		{
			const auto* block = m_vertex_layout.interleaved_blocks[0];
			storage_address = block->real_offset_address + vertex_base * block->attribute_stride;
			if (const auto* entry = m_vertex_cache->find_vertex_range(
				storage_address, static_cast<u32>(required.first)))
			{
				if (entry->local_address != storage_address)
					fmt::throw_exception("Metal vertex-cache address mismatch");
				persistent_offset = entry->offset_in_heap;
				cached = true;
			}
			else
			{
				store = true;
			}
		}
		if (!cached)
		{
			m_frame_stats.vertex_cache_miss_count++;
			persistent_slice = m_attribute_heap.allocate(required.first, 256);
			persistent_offset = static_cast<u32>(persistent_slice.offset);
			if (store)
				m_vertex_cache->store_range(storage_address, static_cast<u32>(required.first),
					persistent_offset);
			void* destination = m_attribute_heap.map(persistent_slice);
			m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base,
				vertex_count, destination, nullptr);
			m_attribute_heap.mark_modified(persistent_slice);
			m_attribute_heap.unmap();
		}
	}
	if (required.second)
	{
		volatile_slice = m_attribute_heap.allocate(required.second, 256);
		volatile_offset = static_cast<u32>(volatile_slice.offset);
		void* destination = m_attribute_heap.map(volatile_slice);
		m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base,
			vertex_count, nullptr, destination);
		m_attribute_heap.mark_modified(volatile_slice);
		m_attribute_heap.unmap();
	}

	mtl::vertex_upload_info result{
		.primitive = input.primitive,
		.vertex_draw_count = input.draw_count,
		.allocated_vertex_count = vertex_count,
		.first_vertex = vertex_base,
		.vertex_index_base = index_base,
		.vertex_index_offset = input.index_offset,
		.persistent_window_offset = persistent_offset,
		.volatile_window_offset = volatile_offset,
		.index_info = input.index_info,
		.emulated_indices = input.emulated_indices,
		.primitive_restart = input.primitive_restart,
	};
	if (is_line_primitive(draw.primitive))
	{
		const std::vector<line_mapping> mappings = build_line_mappings(input, draw);
		if (mappings.empty()) return {.primitive = mtl::primitive_topology::triangle};
		const mtl::data_heap_slice slice = m_vertex_layout_heap.allocate(
			mappings.size() * sizeof(line_mapping), 256);
		std::memcpy(m_vertex_layout_heap.map(slice), mappings.data(), slice.size);
		m_vertex_layout_heap.mark_modified(slice);
		m_vertex_layout_heap.unmap();
		result.primitive = mtl::primitive_topology::triangle;
		result.vertex_draw_count = ::size32(mappings);
		result.index_info.reset();
		result.emulated_indices = true;
		result.primitive_restart = false;
		result.line_expansion = true;
		result.line_mapping_binding = {
			.resource = slice.buffer,
			.gpu_address = slice.buffer_gpu_address(),
			.offset = slice.offset,
			.length = slice.size,
		};
	}
	return result;
}

void MTLGSRender::update_vertex_environment(u32 id,
	const mtl::vertex_upload_info& vertex_information)
{
	if (!m_program) fmt::throw_exception("Metal vertex environment requires a pipeline");
	const mtl::data_heap_slice slice = m_vertex_layout_heap.allocate(sizeof(draw_parameters), 256);
	auto* parameters = static_cast<draw_parameters*>(m_vertex_layout_heap.map(slice));
	*parameters = {
		.vertex_base_index = vertex_information.vertex_index_base,
		.vertex_index_offset = vertex_information.vertex_index_offset,
		.draw_id = id,
		.reserved = vertex_information.line_expansion ? 1u : 0u,
	};
	m_draw_processor.fill_vertex_layout_state(m_vertex_layout, current_vp_metadata,
		vertex_information.first_vertex, vertex_information.allocated_vertex_count,
		parameters->attribute_data, vertex_information.persistent_window_offset,
		vertex_information.volatile_window_offset);
	m_vertex_layout_heap.mark_modified(slice);
	m_vertex_layout_heap.unmap();
	m_vertex_layout_binding = {.resource = slice.buffer, .gpu_address = slice.buffer_gpu_address(),
		.offset = slice.offset, .length = slice.size};
	m_vertex_layout_offset = slice.offset;

	const mtl::buffer& attributes = m_attribute_heap.target_buffer();
	m_program->set_buffer(mtl::msl_shader_stage::vertex,
		mtl::vertex_stage_binding_table::persistent_vertex_buffer,
		attributes, 0, attributes.size());
	m_program->set_buffer(mtl::msl_shader_stage::vertex,
		mtl::vertex_stage_binding_table::volatile_vertex_buffer,
		attributes, 0, attributes.size());
	m_program->set_buffer(mtl::msl_shader_stage::vertex,
		mtl::vertex_stage_binding_table::draw_parameters_buffer, m_vertex_layout_binding);
	if (vertex_information.line_expansion)
	{
		m_program->set_buffer(mtl::msl_shader_stage::vertex,
			mtl::vertex_stage_binding_table::line_mapping_buffer,
			vertex_information.line_mapping_binding);
	}
	else
	{
		m_program->set_buffer(mtl::msl_shader_stage::vertex,
			mtl::vertex_stage_binding_table::line_mapping_buffer,
			*m_null_buffer, 0, m_null_buffer->size());
	}
}
