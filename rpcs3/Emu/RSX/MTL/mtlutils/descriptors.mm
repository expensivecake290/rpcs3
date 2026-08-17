#include "stdafx.h"
#include "descriptors.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <mutex>
#include <vector>

namespace mtl
{
	namespace
	{
		constexpr u8 valid_stage_mask = argument_stage_vertex | argument_stage_fragment |
			argument_stage_tile | argument_stage_compute;
		constexpr u8 graphics_stage_mask = argument_stage_vertex | argument_stage_fragment | argument_stage_tile;

		bool add_overflows(u64 left, u64 right)
		{
			return left > std::numeric_limits<u64>::max() - right;
		}

		bool equal_binding(const argument_buffer_binding& left, const argument_buffer_binding& right)
		{
			return left.resource == right.resource && left.gpu_address == right.gpu_address &&
				left.offset == right.offset && left.length == right.length &&
				left.attribute_stride == right.attribute_stride && left.access == right.access;
		}

		bool equal_binding(const argument_texture_binding& left, const argument_texture_binding& right)
		{
			return left.resource == right.resource && left.access == right.access;
		}

		bool equal_binding(const argument_sampler_binding& left, const argument_sampler_binding& right)
		{
			return left.resource == right.resource;
		}

		template <typename T>
		void validate_index(u32 index, const std::vector<T>& bindings, std::string_view type)
		{
			if (index >= bindings.size())
			{
				fmt::throw_exception("Metal argument-table %s index %u exceeds count %u",
					type, index, bindings.size());
			}
		}

		template <typename T>
		void validate_span(u32 first, usz count, const std::vector<T>& bindings, std::string_view type)
		{
			if (first > bindings.size() || count > bindings.size() - first)
			{
				fmt::throw_exception("Metal argument-table %s range exceeds its layout", type);
			}
		}

		void validate_buffer_binding(const argument_table_layout& layout, const argument_buffer_binding& binding)
		{
			if (!binding)
			{
				if (binding.resource || binding.gpu_address || binding.offset || binding.length || binding.attribute_stride)
				{
					fmt::throw_exception("Metal argument-table buffer binding is only partially initialized");
				}
				return;
			}
			if (add_overflows(binding.gpu_address, binding.offset))
			{
				fmt::throw_exception("Metal argument-table buffer address overflows");
			}
			if (binding.attribute_stride && !layout.support_attribute_strides)
			{
				fmt::throw_exception("Metal argument-table layout does not support attribute strides");
			}
		}

		void validate_stages(u8 stages)
		{
			if (!stages || (stages & ~valid_stage_mask) ||
				((stages & argument_stage_compute) && (stages & graphics_stage_mask)))
			{
				fmt::throw_exception("Invalid Metal argument-table stage visibility 0x%x", stages);
			}
		}

		MTLRenderStages native_render_stages(u8 stages)
		{
			MTLRenderStages result = 0;
			if (stages & argument_stage_vertex) result |= MTLRenderStageVertex;
			if (stages & argument_stage_fragment) result |= MTLRenderStageFragment;
			if (stages & argument_stage_tile) result |= MTLRenderStageTile;
			return result;
		}

		template <typename T>
		u32 count_bound(const std::vector<T>& bindings)
		{
			return static_cast<u32>(std::count_if(bindings.begin(), bindings.end(), [](const T& binding)
			{
				return static_cast<bool>(binding);
			}));
		}

		u32 count_dirty(const std::vector<u8>& dirty)
		{
			return static_cast<u32>(std::count(dirty.begin(), dirty.end(), u8{1}));
		}
	}

	argument_table_layout::operator bool() const
	{
		return buffer_count || texture_count || sampler_count;
	}

	void argument_table_layout::validate() const
	{
		if (!*this || buffer_count > maximum_argument_buffers ||
			texture_count > maximum_argument_textures || sampler_count > maximum_argument_samplers)
		{
			fmt::throw_exception("Invalid Metal argument-table layout (%u buffers, %u textures, %u samplers)",
				buffer_count, texture_count, sampler_count);
		}
	}

	u64 argument_table_layout::signature() const
	{
		validate();
		return static_cast<u64>(buffer_count) |
			(static_cast<u64>(texture_count) << 8) |
			(static_cast<u64>(sampler_count) << 16) |
			(static_cast<u64>(support_attribute_strides) << 24);
	}

	struct argument_table::impl
	{
		const render_device* owner = nullptr;
		id<MTL4ArgumentTable> table = nil;
		argument_table_layout table_layout;
		std::string label;
		u8 stage_mask = argument_stage_none;
		std::vector<argument_buffer_binding> buffers;
		std::vector<argument_texture_binding> textures;
		std::vector<argument_sampler_binding> samplers;
		std::vector<u64> dynamic_offsets;
		std::vector<u8> dirty_buffers;
		std::vector<u8> dirty_textures;
		std::vector<u8> dirty_samplers;
		u64 mutation = 0;
		u64 applied = 0;
		u64 binds = 0;
		mutable std::mutex mutex;

		void require_valid() const
		{
			if (!table || !owner)
			{
				fmt::throw_exception("Metal argument table is not initialized");
			}
		}

		void touch_buffer(u32 index)
		{
			dirty_buffers[index] = 1;
			mutation++;
		}

		void apply_locked()
		{
			require_valid();
			for (u32 index = 0; index < buffers.size(); ++index)
			{
				if (!dirty_buffers[index]) continue;
				const argument_buffer_binding& binding = buffers[index];
				u64 address = 0;
				if (binding)
				{
					if (dynamic_offsets[index] >= binding.length || add_overflows(binding.offset, dynamic_offsets[index]) ||
						add_overflows(binding.gpu_address, binding.offset + dynamic_offsets[index]))
					{
						fmt::throw_exception("Metal argument-table dynamic buffer offset exceeds binding %u", index);
					}
					address = binding.gpu_address + binding.offset + dynamic_offsets[index];
				}
				if (binding.attribute_stride)
				{
					[table setAddress:address attributeStride:binding.attribute_stride atIndex:index];
				}
				else
				{
					[table setAddress:address atIndex:index];
				}
				dirty_buffers[index] = 0;
			}
			for (u32 index = 0; index < textures.size(); ++index)
			{
				if (!dirty_textures[index]) continue;
				const id<MTLTexture> texture = textures[index].resource;
				[table setTexture:texture ? texture.gpuResourceID : MTLResourceID{0} atIndex:index];
				dirty_textures[index] = 0;
			}
			for (u32 index = 0; index < samplers.size(); ++index)
			{
				if (!dirty_samplers[index]) continue;
				const id<MTLSamplerState> sampler_state = samplers[index].resource;
				[table setSamplerState:sampler_state ? sampler_state.gpuResourceID : MTLResourceID{0} atIndex:index];
				dirty_samplers[index] = 0;
			}
			applied = mutation;
		}
	};

	argument_table::argument_table()
		: m_impl(std::make_unique<impl>())
	{
	}

	argument_table::~argument_table()
	{
		destroy();
	}

	argument_table::argument_table(argument_table&&) noexcept = default;
	argument_table& argument_table::operator=(argument_table&&) noexcept = default;

	void argument_table::create(const render_device& device, const argument_table_layout& layout,
		u8 stages, std::string_view label)
	{
		destroy();
		layout.validate();
		validate_stages(stages);
		if (label.empty() || !device || !device.info().features.argument_tables)
		{
			fmt::throw_exception("Invalid Metal argument-table creation request");
		}
		const auto& limits = device.info().limits;
		if (layout.buffer_count > limits.max_buffers_per_argument_table ||
			layout.texture_count > limits.max_textures_per_argument_table ||
			layout.sampler_count > limits.max_samplers_per_argument_table)
		{
			fmt::throw_exception("Metal argument-table layout exceeds device limits");
		}
		if (!m_impl) m_impl = std::make_unique<impl>();

		MTL4ArgumentTableDescriptor* descriptor = [MTL4ArgumentTableDescriptor new];
		descriptor.maxBufferBindCount = layout.buffer_count;
		descriptor.maxTextureBindCount = layout.texture_count;
		descriptor.maxSamplerStateBindCount = layout.sampler_count;
		descriptor.initializeBindings = YES;
		descriptor.supportAttributeStrides = layout.support_attribute_strides;
		descriptor.label = [NSString stringWithUTF8String:std::string(label).c_str()];
		NSError* error = nil;
		id<MTL4ArgumentTable> native = [device.native_handle() newArgumentTableWithDescriptor:descriptor error:&error];
		if (!native)
		{
			const std::string description = error.localizedDescription.UTF8String ?: "unknown error";
			fmt::throw_exception("Metal failed to create argument table '%s': %s", label, description);
		}

		std::lock_guard lock(m_impl->mutex);
		m_impl->owner = &device;
		m_impl->table = native;
		m_impl->table_layout = layout;
		m_impl->label = label;
		m_impl->stage_mask = stages;
		m_impl->buffers.resize(layout.buffer_count);
		m_impl->textures.resize(layout.texture_count);
		m_impl->samplers.resize(layout.sampler_count);
		m_impl->dynamic_offsets.resize(layout.buffer_count);
		m_impl->dirty_buffers.resize(layout.buffer_count);
		m_impl->dirty_textures.resize(layout.texture_count);
		m_impl->dirty_samplers.resize(layout.sampler_count);
		m_impl->mutation = 0;
		m_impl->applied = 0;
		m_impl->binds = 0;
	}

	void argument_table::destroy()
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->owner = nullptr;
		m_impl->table = nil;
		m_impl->table_layout = {};
		m_impl->label.clear();
		m_impl->stage_mask = argument_stage_none;
		m_impl->buffers.clear();
		m_impl->textures.clear();
		m_impl->samplers.clear();
		m_impl->dynamic_offsets.clear();
		m_impl->dirty_buffers.clear();
		m_impl->dirty_textures.clear();
		m_impl->dirty_samplers.clear();
		m_impl->mutation = 0;
		m_impl->applied = 0;
		m_impl->binds = 0;
	}

	void argument_table::reset_bindings()
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		bool changed = false;
		for (u32 index = 0; index < m_impl->buffers.size(); ++index)
		{
			if (m_impl->buffers[index] || m_impl->dynamic_offsets[index])
			{
				m_impl->buffers[index] = {};
				m_impl->dynamic_offsets[index] = 0;
				m_impl->dirty_buffers[index] = 1;
				changed = true;
			}
		}
		for (u32 index = 0; index < m_impl->textures.size(); ++index)
		{
			if (m_impl->textures[index])
			{
				m_impl->textures[index] = {};
				m_impl->dirty_textures[index] = 1;
				changed = true;
			}
		}
		for (u32 index = 0; index < m_impl->samplers.size(); ++index)
		{
			if (m_impl->samplers[index])
			{
				m_impl->samplers[index] = {};
				m_impl->dirty_samplers[index] = 1;
				changed = true;
			}
		}
		if (changed) m_impl->mutation++;
	}

	void argument_table::set_buffer(u32 index, const argument_buffer_binding& binding)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		validate_index(index, m_impl->buffers, "buffer");
		validate_buffer_binding(m_impl->table_layout, binding);
		if (!equal_binding(m_impl->buffers[index], binding))
		{
			m_impl->buffers[index] = binding;
			if (!binding) m_impl->dynamic_offsets[index] = 0;
			m_impl->touch_buffer(index);
		}
	}

	void argument_table::set_buffer(u32 index, const buffer& resource, u64 offset, u64 length,
		u32 attribute_stride, argument_access access)
	{
		if (!resource || length == 0 || !resource.in_range(offset, length))
		{
			fmt::throw_exception("Invalid Metal argument-table buffer range");
		}
		set_buffer(index, {resource.native_handle(), resource.gpu_address(), offset, length, attribute_stride, access});
	}

	void argument_table::set_buffers(u32 first_index, std::span<const argument_buffer_binding> bindings)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		validate_span(first_index, bindings.size(), m_impl->buffers, "buffer");
		bool changed = false;
		for (u32 relative = 0; relative < bindings.size(); ++relative)
		{
			const u32 index = first_index + relative;
			validate_buffer_binding(m_impl->table_layout, bindings[relative]);
			if (equal_binding(m_impl->buffers[index], bindings[relative])) continue;
			m_impl->buffers[index] = bindings[relative];
			if (!bindings[relative]) m_impl->dynamic_offsets[index] = 0;
			m_impl->dirty_buffers[index] = 1;
			changed = true;
		}
		if (changed) m_impl->mutation++;
	}

	void argument_table::clear_buffer(u32 index)
	{
		set_buffer(index, {});
	}

	void argument_table::set_texture(u32 index, const argument_texture_binding& binding)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		validate_index(index, m_impl->textures, "texture");
		if (!equal_binding(m_impl->textures[index], binding))
		{
			m_impl->textures[index] = binding;
			m_impl->dirty_textures[index] = 1;
			m_impl->mutation++;
		}
	}

	void argument_table::set_texture(u32 index, const image_view& resource, argument_access access)
	{
		if (!resource) fmt::throw_exception("Cannot bind an empty Metal image view");
		set_texture(index, {resource.native_handle(), access});
	}

	void argument_table::set_texture(u32 index, const buffer_view& resource, argument_access access)
	{
		if (!resource) fmt::throw_exception("Cannot bind an empty Metal buffer view");
		set_texture(index, {resource.native_handle(), access});
	}

	void argument_table::set_textures(u32 first_index, std::span<const argument_texture_binding> bindings)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		validate_span(first_index, bindings.size(), m_impl->textures, "texture");
		bool changed = false;
		for (u32 relative = 0; relative < bindings.size(); ++relative)
		{
			const u32 index = first_index + relative;
			if (equal_binding(m_impl->textures[index], bindings[relative])) continue;
			m_impl->textures[index] = bindings[relative];
			m_impl->dirty_textures[index] = 1;
			changed = true;
		}
		if (changed) m_impl->mutation++;
	}

	void argument_table::clear_texture(u32 index)
	{
		set_texture(index, argument_texture_binding{});
	}

	void argument_table::set_sampler(u32 index, const argument_sampler_binding& binding)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		validate_index(index, m_impl->samplers, "sampler");
		if (!equal_binding(m_impl->samplers[index], binding))
		{
			m_impl->samplers[index] = binding;
			m_impl->dirty_samplers[index] = 1;
			m_impl->mutation++;
		}
	}

	void argument_table::set_sampler(u32 index, const sampler& resource)
	{
		if (!resource) fmt::throw_exception("Cannot bind an empty Metal sampler");
		set_sampler(index, {resource.native_handle()});
	}

	void argument_table::set_samplers(u32 first_index, std::span<const argument_sampler_binding> bindings)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		validate_span(first_index, bindings.size(), m_impl->samplers, "sampler");
		bool changed = false;
		for (u32 relative = 0; relative < bindings.size(); ++relative)
		{
			const u32 index = first_index + relative;
			if (equal_binding(m_impl->samplers[index], bindings[relative])) continue;
			m_impl->samplers[index] = bindings[relative];
			m_impl->dirty_samplers[index] = 1;
			changed = true;
		}
		if (changed) m_impl->mutation++;
	}

	void argument_table::clear_sampler(u32 index)
	{
		set_sampler(index, argument_sampler_binding{});
	}

	void argument_table::set_dynamic_offset(u32 buffer_index, u64 offset)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		validate_index(buffer_index, m_impl->buffers, "dynamic buffer");
		const argument_buffer_binding& binding = m_impl->buffers[buffer_index];
		if (!binding || offset >= binding.length)
		{
			fmt::throw_exception("Metal argument-table dynamic offset exceeds binding %u", buffer_index);
		}
		if (m_impl->dynamic_offsets[buffer_index] != offset)
		{
			m_impl->dynamic_offsets[buffer_index] = offset;
			m_impl->touch_buffer(buffer_index);
		}
	}

	void argument_table::set_dynamic_offsets(std::span<const argument_dynamic_offset> offsets)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		bool changed = false;
		for (const argument_dynamic_offset& offset : offsets)
		{
			validate_index(offset.index, m_impl->buffers, "dynamic buffer");
			const argument_buffer_binding& binding = m_impl->buffers[offset.index];
			if (!binding || offset.value >= binding.length)
			{
				fmt::throw_exception("Metal argument-table dynamic offset exceeds binding %u", offset.index);
			}
			if (m_impl->dynamic_offsets[offset.index] == offset.value) continue;
			m_impl->dynamic_offsets[offset.index] = offset.value;
			m_impl->dirty_buffers[offset.index] = 1;
			changed = true;
		}
		if (changed) m_impl->mutation++;
	}

	void argument_table::clear_dynamic_offsets()
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		bool changed = false;
		for (u32 index = 0; index < m_impl->dynamic_offsets.size(); ++index)
		{
			if (!m_impl->dynamic_offsets[index]) continue;
			m_impl->dynamic_offsets[index] = 0;
			m_impl->dirty_buffers[index] = 1;
			changed = true;
		}
		if (changed) m_impl->mutation++;
	}

	void argument_table::copy_bindings_from(const argument_table& source,
		std::span<const argument_binding_range> buffer_ranges,
		std::span<const argument_binding_range> texture_ranges,
		std::span<const argument_binding_range> sampler_ranges)
	{
		if (!m_impl || !source.m_impl)
		{
			fmt::throw_exception("Cannot copy bindings using an empty Metal argument table");
		}
		std::vector<argument_buffer_binding> source_buffers;
		std::vector<argument_texture_binding> source_textures;
		std::vector<argument_sampler_binding> source_samplers;
		std::vector<u64> source_offsets;
		{
			std::lock_guard source_lock(source.m_impl->mutex);
			source.m_impl->require_valid();
			source_buffers = source.m_impl->buffers;
			source_textures = source.m_impl->textures;
			source_samplers = source.m_impl->samplers;
			source_offsets = source.m_impl->dynamic_offsets;
		}

		std::lock_guard destination_lock(m_impl->mutex);
		m_impl->require_valid();
		bool changed = false;
		auto copy_ranges = [&](const auto& input, auto& output, auto& dirty, std::span<const argument_binding_range> ranges,
			std::string_view type, const std::vector<u64>* input_offsets, std::vector<u64>* output_offsets)
		{
			for (const argument_binding_range& range : ranges)
			{
				validate_span(range.source_index, range.count, input, type);
				validate_span(range.destination_index, range.count, output, type);
				for (u32 relative = 0; relative < range.count; ++relative)
				{
					const u32 source_index = range.source_index + relative;
					const u32 destination_index = range.destination_index + relative;
					const bool offset_changed = input_offsets &&
						(*input_offsets)[source_index] != (*output_offsets)[destination_index];
					if (equal_binding(input[source_index], output[destination_index]) && !offset_changed) continue;
					output[destination_index] = input[source_index];
					if (input_offsets) (*output_offsets)[destination_index] = (*input_offsets)[source_index];
					dirty[destination_index] = 1;
					changed = true;
				}
			}
		};
		copy_ranges(source_buffers, m_impl->buffers, m_impl->dirty_buffers, buffer_ranges,
			"buffer", &source_offsets, &m_impl->dynamic_offsets);
		copy_ranges(source_textures, m_impl->textures, m_impl->dirty_textures, texture_ranges,
			"texture", nullptr, nullptr);
		copy_ranges(source_samplers, m_impl->samplers, m_impl->dirty_samplers, sampler_ranges,
			"sampler", nullptr, nullptr);
		if (changed) m_impl->mutation++;
	}

	void argument_table::apply()
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->apply_locked();
	}

	void argument_table::bind(command_buffer& command)
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->require_valid();
		if (!command.is_recording())
		{
			fmt::throw_exception("Metal argument-table binding requires active command recording");
		}
		m_impl->apply_locked();
		if (m_impl->stage_mask & argument_stage_compute)
		{
			if (command.active_encoder() != encoder_kind::compute)
			{
				fmt::throw_exception("Compute argument table requires an active Metal compute encoder");
			}
			id<MTL4ComputeCommandEncoder> encoder = (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder();
			[encoder setArgumentTable:m_impl->table];
		}
		else
		{
			if (command.active_encoder() != encoder_kind::render)
			{
				fmt::throw_exception("Graphics argument table requires an active Metal render encoder");
			}
			id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)command.active_native_encoder();
			[encoder setArgumentTable:m_impl->table atStages:native_render_stages(m_impl->stage_mask)];
		}

		command.retain_native_object((__bridge void*)m_impl->table, false);
		for (const argument_buffer_binding& binding : m_impl->buffers)
		{
			if (binding) command.retain_native_object((__bridge void*)binding.resource, true);
		}
		for (const argument_texture_binding& binding : m_impl->textures)
		{
			if (binding) command.retain_native_object((__bridge void*)binding.resource, true);
		}
		for (const argument_sampler_binding& binding : m_impl->samplers)
		{
			if (binding) command.retain_native_object((__bridge void*)binding.resource, false);
		}
		m_impl->binds++;
	}

	argument_table::operator bool() const
	{
		if (!m_impl) return false;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->table != nil;
	}

	argument_table_handle argument_table::native_handle() const
	{
		if (!m_impl) return nil;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->table;
	}

	const argument_table_layout& argument_table::layout() const
	{
		if (!m_impl || !m_impl->table) fmt::throw_exception("Layout requested from an empty Metal argument table");
		return m_impl->table_layout;
	}

	u8 argument_table::stages() const
	{
		if (!m_impl) return argument_stage_none;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->stage_mask;
	}

	bool argument_table::dirty() const
	{
		if (!m_impl) return false;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->mutation != m_impl->applied;
	}

	u64 argument_table::mutation_serial() const
	{
		if (!m_impl) return 0;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->mutation;
	}

	u64 argument_table::applied_serial() const
	{
		if (!m_impl) return 0;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->applied;
	}

	argument_table_statistics argument_table::statistics() const
	{
		argument_table_statistics result;
		if (!m_impl) return result;
		std::lock_guard lock(m_impl->mutex);
		result.mutation_serial = m_impl->mutation;
		result.applied_serial = m_impl->applied;
		result.bind_count = m_impl->binds;
		result.bound_buffers = count_bound(m_impl->buffers);
		result.bound_textures = count_bound(m_impl->textures);
		result.bound_samplers = count_bound(m_impl->samplers);
		result.dirty_buffers = count_dirty(m_impl->dirty_buffers);
		result.dirty_textures = count_dirty(m_impl->dirty_textures);
		result.dirty_samplers = count_dirty(m_impl->dirty_samplers);
		return result;
	}

	struct argument_table_cache::impl
	{
		struct pending_table
		{
			u64 submission = 0;
			std::unique_ptr<argument_table> table;
		};

		const render_device* owner = nullptr;
		std::vector<std::unique_ptr<argument_table>> available;
		std::vector<pending_table> pending;
		argument_table_cache_statistics counters;
		mutable std::mutex mutex;
	};

	argument_table_cache::argument_table_cache()
		: m_impl(std::make_unique<impl>())
	{
	}

	argument_table_cache::~argument_table_cache()
	{
		destroy();
	}

	void argument_table_cache::create(const render_device& device)
	{
		destroy();
		if (!device || !device.info().features.argument_tables)
		{
			fmt::throw_exception("Cannot create argument-table cache on an unsupported Metal device");
		}
		std::lock_guard lock(m_impl->mutex);
		m_impl->owner = &device;
		m_impl->counters = {};
	}

	void argument_table_cache::destroy()
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->available.clear();
		m_impl->pending.clear();
		m_impl->owner = nullptr;
		m_impl->counters = {};
	}

	std::unique_ptr<argument_table> argument_table_cache::acquire(
		const argument_table_layout& layout, u8 stages, std::string_view label)
	{
		layout.validate();
		validate_stages(stages);
		if (label.empty()) fmt::throw_exception("Metal argument table requires a debug label");
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->owner) fmt::throw_exception("Metal argument-table cache is not initialized");
		for (auto iterator = m_impl->available.begin(); iterator != m_impl->available.end(); ++iterator)
		{
			if ((*iterator)->layout() == layout && (*iterator)->stages() == stages)
			{
				auto result = std::move(*iterator);
				m_impl->available.erase(iterator);
				m_impl->counters.reused++;
				return result;
			}
		}
		auto result = std::make_unique<argument_table>();
		result->create(*m_impl->owner, layout, stages, label);
		m_impl->counters.created++;
		return result;
	}

	void argument_table_cache::retire(std::unique_ptr<argument_table> table, u64 submission_value)
	{
		if (!table || !*table || submission_value == 0)
		{
			fmt::throw_exception("Invalid Metal argument-table retirement request");
		}
		std::lock_guard lock(m_impl->mutex);
		if (!m_impl->owner) fmt::throw_exception("Metal argument-table cache is not initialized");
		m_impl->pending.push_back({submission_value, std::move(table)});
		m_impl->counters.retired++;
	}

	void argument_table_cache::reclaim(u64 completed_submission_value)
	{
		std::lock_guard lock(m_impl->mutex);
		for (auto iterator = m_impl->pending.begin(); iterator != m_impl->pending.end();)
		{
			if (iterator->submission > completed_submission_value)
			{
				++iterator;
				continue;
			}
			iterator->table->reset_bindings();
			iterator->table->apply();
			m_impl->available.push_back(std::move(iterator->table));
			iterator = m_impl->pending.erase(iterator);
			m_impl->counters.reclaimed++;
		}
	}

	void argument_table_cache::trim(usz maximum_available)
	{
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->available.size() <= maximum_available) return;
		const usz discard_count = m_impl->available.size() - maximum_available;
		m_impl->available.erase(m_impl->available.begin(), m_impl->available.begin() + discard_count);
		m_impl->counters.discarded += discard_count;
	}

	argument_table_cache_statistics argument_table_cache::statistics() const
	{
		argument_table_cache_statistics result;
		if (!m_impl) return result;
		std::lock_guard lock(m_impl->mutex);
		result = m_impl->counters;
		result.available = m_impl->available.size();
		result.pending = m_impl->pending.size();
		return result;
	}
}
