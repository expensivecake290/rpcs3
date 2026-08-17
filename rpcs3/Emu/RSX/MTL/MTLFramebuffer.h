#pragma once

#include <array>
#include <memory>

#include "mtlutils/framebuffer_object.hpp"

namespace mtl
{
	struct attachment_view_descriptor
	{
		const image* resource = nullptr;
		u64 pixel_format = 0;
		texture_type type = texture_type::texture_2d;
		bool use_resource_type = true;
		component_mapping mapping;
		subresource_range subresources;
		u32 mip_level = 0;
		u32 array_slice = 0;
		u32 depth_plane = 0;

		[[nodiscard]] explicit operator bool() const
		{
			return resource != nullptr;
		}
	};

	struct framebuffer_attachment_request
	{
		attachment_view_descriptor render;
		attachment_view_descriptor resolve;

		[[nodiscard]] explicit operator bool() const
		{
			return static_cast<bool>(render);
		}

		[[nodiscard]] bool has_resolve() const
		{
			return static_cast<bool>(resolve);
		}
	};

	struct framebuffer_request
	{
		u32 width = 0;
		u32 height = 0;
		u32 layers = 1;
		u32 samples = 1;
		std::array<framebuffer_attachment_request, maximum_color_attachments> colors{};
		framebuffer_attachment_request depth;
		framebuffer_attachment_request stencil;
		bool reads_attachments = false;

		[[nodiscard]] explicit operator bool() const
		{
			return width != 0 && height != 0 && layers != 0 && samples != 0;
		}
	};

	struct framebuffer_cache_statistics
	{
		u64 hits = 0;
		u64 misses = 0;
		u64 created = 0;
		u64 invalidated = 0;
		u64 retired = 0;
		usz entries = 0;
		usz buckets = 0;
	};

	class framebuffer final : public unique_resource
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

		explicit framebuffer(const framebuffer_request& request);
		friend class framebuffer_cache;

		void touch(u64 serial);
		[[nodiscard]] u32 note_idle_check();

	public:
		~framebuffer();
		framebuffer(const framebuffer&) = delete;
		framebuffer& operator=(const framebuffer&) = delete;
		framebuffer(framebuffer&&) = delete;
		framebuffer& operator=(framebuffer&&) = delete;

		[[nodiscard]] const framebuffer_attachment_set& attachments() const;
		[[nodiscard]] u32 width() const;
		[[nodiscard]] u32 height() const;
		[[nodiscard]] u32 layers() const;
		[[nodiscard]] u32 samples() const;
		[[nodiscard]] u64 signature() const;
		[[nodiscard]] u64 last_use_serial() const;
		[[nodiscard]] usz view_count() const;
		[[nodiscard]] bool references_resource(u64 resource_uid) const;
	};

	class framebuffer_cache final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		framebuffer_cache();
		~framebuffer_cache();
		framebuffer_cache(const framebuffer_cache&) = delete;
		framebuffer_cache& operator=(const framebuffer_cache&) = delete;
		framebuffer_cache(framebuffer_cache&&) = delete;
		framebuffer_cache& operator=(framebuffer_cache&&) = delete;

		[[nodiscard]] std::shared_ptr<framebuffer> acquire(const framebuffer_request& request);
		[[nodiscard]] usz invalidate_resource(u64 resource_uid);
		[[nodiscard]] usz remove_unused(u32 required_idle_checks = 2);
		void clear();

		[[nodiscard]] usz size() const;
		[[nodiscard]] framebuffer_cache_statistics statistics() const;
	};

	[[nodiscard]] std::shared_ptr<framebuffer> get_framebuffer(const framebuffer_request& request);
	[[nodiscard]] usz invalidate_framebuffers_for_resource(u64 resource_uid);
	[[nodiscard]] usz remove_unused_framebuffers(u32 required_idle_checks = 2);
	void clear_framebuffer_cache();
	[[nodiscard]] framebuffer_cache_statistics get_framebuffer_cache_statistics();
}
