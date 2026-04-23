#pragma once

#include "Emu/RSX/RSXTexture.h"
#include "util/types.hpp"

#include <memory>

namespace rsx::metal
{
	class command_frame;
	class device;
	class texture;

	struct sampled_texture_cache_stats
	{
		u32 retained_texture_count = 0;
		u32 created_texture_count = 0;
		u32 reused_texture_count = 0;
		u32 uploaded_texture_count = 0;
		u64 uploaded_byte_count = 0;
	};

	class sampled_texture_cache
	{
	public:
		explicit sampled_texture_cache(device& dev);
		~sampled_texture_cache();

		sampled_texture_cache(const sampled_texture_cache&) = delete;
		sampled_texture_cache& operator=(const sampled_texture_cache&) = delete;

		texture& prepare_fragment_texture(command_frame& frame, const rsx::fragment_texture& rsx_texture, u32 texture_index);
		texture& get_fragment_texture(const rsx::fragment_texture& rsx_texture, u32 texture_index);
		sampled_texture_cache_stats stats() const;
		void report() const;

	private:
		struct sampled_texture_cache_impl;
		std::unique_ptr<sampled_texture_cache_impl> m_impl;
	};
}
