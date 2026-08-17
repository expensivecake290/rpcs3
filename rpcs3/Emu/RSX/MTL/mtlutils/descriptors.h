#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "buffer_object.h"
#include "commands.h"
#include "image.h"
#include "sampler.h"

namespace mtl
{
	inline constexpr u32 maximum_argument_buffers = 31;
	inline constexpr u32 maximum_argument_textures = 128;
	inline constexpr u32 maximum_argument_samplers = 16;

	enum argument_stage : u8
	{
		argument_stage_none = 0,
		argument_stage_vertex = 1 << 0,
		argument_stage_fragment = 1 << 1,
		argument_stage_tile = 1 << 2,
		argument_stage_compute = 1 << 3,
	};

	enum class argument_access : u8
	{
		read,
		write,
		read_write,
	};

	struct argument_table_layout
	{
		u32 buffer_count = 0;
		u32 texture_count = 0;
		u32 sampler_count = 0;
		bool support_attribute_strides = false;

		[[nodiscard]] bool operator==(const argument_table_layout&) const = default;
		[[nodiscard]] explicit operator bool() const;
		void validate() const;
		[[nodiscard]] u64 signature() const;
	};

	struct argument_buffer_binding
	{
		buffer_handle resource = nullptr;
		u64 gpu_address = 0;
		u64 offset = 0;
		u64 length = 0;
		u32 attribute_stride = 0;
		argument_access access = argument_access::read;

		[[nodiscard]] explicit operator bool() const
		{
			return resource && gpu_address != 0 && length != 0;
		}
	};

	struct argument_texture_binding
	{
		texture_handle resource = nullptr;
		argument_access access = argument_access::read;

		[[nodiscard]] explicit operator bool() const
		{
			return resource != nullptr;
		}
	};

	struct argument_sampler_binding
	{
		sampler_handle resource = nullptr;

		[[nodiscard]] explicit operator bool() const
		{
			return resource != nullptr;
		}
	};

	struct argument_dynamic_offset
	{
		u32 index = 0;
		u64 value = 0;
	};

	struct argument_binding_range
	{
		u32 source_index = 0;
		u32 destination_index = 0;
		u32 count = 0;
	};

	struct argument_table_statistics
	{
		u64 mutation_serial = 0;
		u64 applied_serial = 0;
		u64 bind_count = 0;
		u32 bound_buffers = 0;
		u32 bound_textures = 0;
		u32 bound_samplers = 0;
		u32 dirty_buffers = 0;
		u32 dirty_textures = 0;
		u32 dirty_samplers = 0;
	};

	class argument_table
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		argument_table();
		~argument_table();
		argument_table(const argument_table&) = delete;
		argument_table& operator=(const argument_table&) = delete;
		argument_table(argument_table&&) noexcept;
		argument_table& operator=(argument_table&&) noexcept;

		void create(const render_device& device, const argument_table_layout& layout,
			u8 stages, std::string_view label);
		void destroy();
		void reset_bindings();

		void set_buffer(u32 index, const argument_buffer_binding& binding);
		void set_buffer(u32 index, const buffer& resource, u64 offset, u64 length,
			u32 attribute_stride = 0, argument_access access = argument_access::read);
		void set_buffers(u32 first_index, std::span<const argument_buffer_binding> bindings);
		void clear_buffer(u32 index);

		void set_texture(u32 index, const argument_texture_binding& binding);
		void set_texture(u32 index, const image_view& resource,
			argument_access access = argument_access::read);
		void set_texture(u32 index, const buffer_view& resource,
			argument_access access = argument_access::read);
		void set_textures(u32 first_index, std::span<const argument_texture_binding> bindings);
		void clear_texture(u32 index);

		void set_sampler(u32 index, const argument_sampler_binding& binding);
		void set_sampler(u32 index, const sampler& resource);
		void set_samplers(u32 first_index, std::span<const argument_sampler_binding> bindings);
		void clear_sampler(u32 index);

		void set_dynamic_offset(u32 buffer_index, u64 offset);
		void set_dynamic_offsets(std::span<const argument_dynamic_offset> offsets);
		void clear_dynamic_offsets();

		void copy_bindings_from(const argument_table& source,
			std::span<const argument_binding_range> buffer_ranges,
			std::span<const argument_binding_range> texture_ranges,
			std::span<const argument_binding_range> sampler_ranges);

		void apply();
		void bind(command_buffer& command);

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] argument_table_handle native_handle() const;
		[[nodiscard]] const argument_table_layout& layout() const;
		[[nodiscard]] u8 stages() const;
		[[nodiscard]] bool dirty() const;
		[[nodiscard]] u64 mutation_serial() const;
		[[nodiscard]] u64 applied_serial() const;
		[[nodiscard]] argument_table_statistics statistics() const;
	};

	struct argument_table_cache_statistics
	{
		u64 created = 0;
		u64 reused = 0;
		u64 retired = 0;
		u64 reclaimed = 0;
		u64 discarded = 0;
		u64 available = 0;
		u64 pending = 0;
	};

	class argument_table_cache
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		argument_table_cache();
		~argument_table_cache();
		argument_table_cache(const argument_table_cache&) = delete;
		argument_table_cache& operator=(const argument_table_cache&) = delete;
		argument_table_cache(argument_table_cache&&) = delete;
		argument_table_cache& operator=(argument_table_cache&&) = delete;

		void create(const render_device& device);
		void destroy();
		[[nodiscard]] std::unique_ptr<argument_table> acquire(
			const argument_table_layout& layout, u8 stages, std::string_view label);
		void retire(std::unique_ptr<argument_table> table, u64 submission_value);
		void reclaim(u64 completed_submission_value);
		void trim(usz maximum_available);
		[[nodiscard]] argument_table_cache_statistics statistics() const;
	};
}
