#pragma once

#include <variant>
#include <vector>

#include "../MetalAPI.h"

namespace mtl
{
	enum class resource_access : u8
	{
		read,
		write,
		read_write,
	};

	enum class shader_stage : u8
	{
		vertex,
		fragment,
		compute,
	};

	struct resource_identity
	{
		u64 resource_uid = 0;
		u64 view_uid = 0;

		[[nodiscard]] explicit operator bool() const
		{
			return resource_uid != 0;
		}

		[[nodiscard]] bool operator==(const resource_identity&) const = default;
	};

	struct buffer_binding
	{
		resource_identity identity;
		buffer_handle buffer = nullptr;
		u64 offset = 0;
		u64 length = 0;
		u64 gpu_address = 0;
		resource_access access = resource_access::read;

		[[nodiscard]] bool valid() const;
	};

	struct texture_binding
	{
		resource_identity identity;
		texture_handle texture = nullptr;
		sampler_handle sampler = nullptr;
		resource_access access = resource_access::read;

		[[nodiscard]] bool valid() const;
	};

	struct sampler_binding
	{
		u64 resource_uid = 0;
		sampler_handle sampler = nullptr;

		[[nodiscard]] bool valid() const;
	};

	using binding_value = std::variant<std::monostate, buffer_binding, texture_binding, sampler_binding>;

	struct argument_binding
	{
		shader_stage stage = shader_stage::vertex;
		u32 index = 0;
		binding_value value;

		[[nodiscard]] bool valid() const;
		[[nodiscard]] resource_identity identity() const;
	};

	struct format_compatibility
	{
		u64 base_format = 0;
		std::vector<u64> view_formats;

		[[nodiscard]] explicit operator bool() const
		{
			return base_format != 0;
		}

		[[nodiscard]] bool is_mutable() const
		{
			return !view_formats.empty();
		}

		[[nodiscard]] bool allows(u64 format) const;
	};

	[[nodiscard]] const char* get_resource_access_name(resource_access access);
	[[nodiscard]] const char* get_shader_stage_name(shader_stage stage);
	[[nodiscard]] bool has_read_access(resource_access access);
	[[nodiscard]] bool has_write_access(resource_access access);
}
