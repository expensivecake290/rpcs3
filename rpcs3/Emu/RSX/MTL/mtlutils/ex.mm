#include "stdafx.h"
#include "ex.h"

#include <algorithm>
#include <limits>

namespace mtl
{
	bool buffer_binding::valid() const
	{
		if (!identity || !buffer || length == 0)
		{
			return false;
		}

		if (offset > std::numeric_limits<u64>::max() - length)
		{
			return false;
		}

		return gpu_address == 0 || gpu_address <= std::numeric_limits<u64>::max() - length;
	}

	bool texture_binding::valid() const
	{
		if (!identity || !texture)
		{
			return false;
		}

		if (access != resource_access::read && sampler)
		{
			return false;
		}

		return true;
	}

	bool sampler_binding::valid() const
	{
		return resource_uid != 0 && sampler;
	}

	bool argument_binding::valid() const
	{
		return std::visit([](const auto& binding)
		{
			using type = std::decay_t<decltype(binding)>;
			if constexpr (std::is_same_v<type, std::monostate>)
			{
				return false;
			}
			else
			{
				return binding.valid();
			}
		}, value);
	}

	resource_identity argument_binding::identity() const
	{
		return std::visit([](const auto& binding) -> resource_identity
		{
			using type = std::decay_t<decltype(binding)>;
			if constexpr (std::is_same_v<type, buffer_binding> || std::is_same_v<type, texture_binding>)
			{
				return binding.identity;
			}
			else if constexpr (std::is_same_v<type, sampler_binding>)
			{
				return {binding.resource_uid, binding.resource_uid};
			}
			else
			{
				return {};
			}
		}, value);
	}

	bool format_compatibility::allows(u64 format) const
	{
		if (format == 0)
		{
			return false;
		}

		return format == base_format || std::find(view_formats.begin(), view_formats.end(), format) != view_formats.end();
	}

	const char* get_resource_access_name(resource_access access)
	{
		switch (access)
		{
		case resource_access::read: return "read";
		case resource_access::write: return "write";
		case resource_access::read_write: return "read-write";
		}

		fmt::throw_exception("Invalid Metal resource access %u", static_cast<u8>(access));
	}

	const char* get_shader_stage_name(shader_stage stage)
	{
		switch (stage)
		{
		case shader_stage::vertex: return "vertex";
		case shader_stage::fragment: return "fragment";
		case shader_stage::compute: return "compute";
		}

		fmt::throw_exception("Invalid Metal shader stage %u", static_cast<u8>(stage));
	}

	bool has_read_access(resource_access access)
	{
		return access == resource_access::read || access == resource_access::read_write;
	}

	bool has_write_access(resource_access access)
	{
		return access == resource_access::write || access == resource_access::read_write;
	}
}
