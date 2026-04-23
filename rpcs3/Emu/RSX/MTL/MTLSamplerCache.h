#pragma once

#include "MTLSampler.h"

#include "Emu/RSX/RSXTexture.h"
#include "util/types.hpp"

#include <memory>
#include <vector>

namespace rsx::metal
{
	class device;

	struct sampler_cache_stats
	{
		u32 retained_sampler_count = 0;
		u32 created_sampler_count = 0;
		u32 sampler_cache_hit_count = 0;
	};

	class sampler_cache
	{
	public:
		explicit sampler_cache(device& dev);
		~sampler_cache();

		sampler_cache(const sampler_cache&) = delete;
		sampler_cache& operator=(const sampler_cache&) = delete;

		sampler& get_fragment_sampler(const rsx::fragment_texture& texture, b8 allow_mipmaps);
		sampler_cache_stats stats() const;
		void report() const;

	private:
		struct sampler_record
		{
			sampler_desc desc;
			std::unique_ptr<sampler> state;
		};

		device& m_device;
		std::vector<sampler_record> m_samplers;
		u32 m_created_sampler_count = 0;
		u32 m_sampler_cache_hit_count = 0;
	};
}
