#include "stdafx.h"
#include "MTLFramebuffer.h"

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mtl
{
	namespace
	{
		enum class attachment_role : u8
		{
			color,
			depth,
			stencil,
		};

		struct normalized_view_descriptor
		{
			const image* resource = nullptr;
			u64 pixel_format = 0;
			texture_type type = texture_type::texture_2d;
			component_mapping mapping;
			subresource_range subresources;
			u32 mip_level = 0;
			u32 array_slice = 0;
			u32 depth_plane = 0;
		};

		normalized_view_descriptor normalize_view(
			const attachment_view_descriptor& descriptor, attachment_role role)
		{
			if (!descriptor.resource || !*descriptor.resource)
			{
				fmt::throw_exception("Metal framebuffer attachment references an empty image");
			}

			const image& resource = *descriptor.resource;
			const u32 required_usage = role == attachment_role::color
				? texture_usage_render_target : texture_usage_depth_stencil;
			if (!(resource.info().usage & required_usage))
			{
				fmt::throw_exception("Metal image '%s' was not created for the requested attachment role",
					resource.debug_name());
			}

			normalized_view_descriptor result;
			result.resource = descriptor.resource;
			result.pixel_format = descriptor.pixel_format ? descriptor.pixel_format : resource.format();
			result.type = descriptor.use_resource_type ? resource.type() : descriptor.type;
			result.mapping = descriptor.mapping;
			result.subresources = descriptor.subresources;
			result.subresources.color = role == attachment_role::color;
			result.subresources.depth = role == attachment_role::depth;
			result.subresources.stencil = role == attachment_role::stencil;
			result.mip_level = descriptor.mip_level;
			result.array_slice = descriptor.array_slice;
			result.depth_plane = descriptor.depth_plane;
			return result;
		}

		struct attachment_view_key
		{
			u64 resource_uid = 0;
			u64 pixel_format = 0;
			texture_type type = texture_type::texture_2d;
			u32 component_mapping = 0;
			subresource_range subresources;
			u32 mip_level = 0;
			u32 array_slice = 0;
			u32 depth_plane = 0;

			[[nodiscard]] bool operator==(const attachment_view_key&) const = default;
		};

		struct attachment_request_key
		{
			attachment_view_key render;
			attachment_view_key resolve;

			[[nodiscard]] bool operator==(const attachment_request_key&) const = default;
		};

		struct framebuffer_request_key
		{
			u32 width = 0;
			u32 height = 0;
			u32 layers = 1;
			u32 samples = 1;
			std::array<attachment_request_key, maximum_color_attachments> colors{};
			attachment_request_key depth;
			attachment_request_key stencil;
			bool reads_attachments = false;

			[[nodiscard]] bool operator==(const framebuffer_request_key&) const = default;
		};

		void hash_mix(u64& hash, u64 value)
		{
			hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
		}

		void hash_view(u64& hash, const attachment_view_key& view)
		{
			hash_mix(hash, view.resource_uid);
			hash_mix(hash, view.pixel_format);
			hash_mix(hash, static_cast<u8>(view.type));
			hash_mix(hash, view.component_mapping);
			hash_mix(hash, view.subresources.first_mip);
			hash_mix(hash, view.subresources.mip_count);
			hash_mix(hash, view.subresources.first_slice);
			hash_mix(hash, view.subresources.slice_count);
			hash_mix(hash, view.subresources.color);
			hash_mix(hash, view.subresources.depth);
			hash_mix(hash, view.subresources.stencil);
			hash_mix(hash, view.mip_level);
			hash_mix(hash, view.array_slice);
			hash_mix(hash, view.depth_plane);
		}

		struct framebuffer_request_key_hash
		{
			[[nodiscard]] usz operator()(const framebuffer_request_key& key) const noexcept
			{
				u64 hash = 0x6a09e667f3bcc909ull;
				hash_mix(hash, key.width);
				hash_mix(hash, key.height);
				hash_mix(hash, key.layers);
				hash_mix(hash, key.samples);
				hash_mix(hash, key.reads_attachments);
				for (const auto& attachment : key.colors)
				{
					hash_view(hash, attachment.render);
					hash_view(hash, attachment.resolve);
				}
				hash_view(hash, key.depth.render);
				hash_view(hash, key.depth.resolve);
				hash_view(hash, key.stencil.render);
				hash_view(hash, key.stencil.resolve);
				return static_cast<usz>(hash);
			}
		};

		attachment_view_key make_view_key(
			const attachment_view_descriptor& descriptor, attachment_role role)
		{
			if (!descriptor)
			{
				return {};
			}
			const auto normalized = normalize_view(descriptor, role);
			return {
				normalized.resource->uid(),
				normalized.pixel_format,
				normalized.type,
				normalized.mapping.encode(),
				normalized.subresources,
				normalized.mip_level,
				normalized.array_slice,
				normalized.depth_plane,
			};
		}

		attachment_request_key make_attachment_key(
			const framebuffer_attachment_request& request, attachment_role role)
		{
			if (!request && request.has_resolve())
			{
				fmt::throw_exception("Metal resolve attachment has no multisample source");
			}
			return {make_view_key(request.render, role), make_view_key(request.resolve, role)};
		}

		framebuffer_request_key make_request_key(const framebuffer_request& request)
		{
			if (!request || (request.samples != 1 && request.samples != 2 &&
				request.samples != 4 && request.samples != 8))
			{
				fmt::throw_exception("Invalid Metal framebuffer request extent or sample count");
			}

			framebuffer_request_key result;
			result.width = request.width;
			result.height = request.height;
			result.layers = request.layers;
			result.samples = request.samples;
			result.reads_attachments = request.reads_attachments;
			for (u32 index = 0; index < maximum_color_attachments; ++index)
			{
				result.colors[index] = make_attachment_key(request.colors[index], attachment_role::color);
			}
			result.depth = make_attachment_key(request.depth, attachment_role::depth);
			result.stencil = make_attachment_key(request.stencil, attachment_role::stencil);
			return result;
		}

		attachment_reference create_attachment(
			std::vector<std::unique_ptr<image_view>>& views,
			const framebuffer_attachment_request& request, attachment_role role)
		{
			const auto render = normalize_view(request.render, role);
			auto render_view = std::make_unique<image_view>(*render.resource, render.pixel_format,
				render.type, render.mapping, render.subresources);
			auto attachment = attachment_reference::from_view(*render.resource, *render_view,
				render.mip_level, render.array_slice, render.depth_plane);

			std::unique_ptr<image_view> resolve_view;
			if (request.has_resolve())
			{
				const auto resolve = normalize_view(request.resolve, role);
				resolve_view = std::make_unique<image_view>(*resolve.resource, resolve.pixel_format,
					resolve.type, resolve.mapping, resolve.subresources);
				attachment.set_resolve(*resolve.resource, *resolve_view,
					resolve.mip_level, resolve.array_slice, resolve.depth_plane);
			}

			views.push_back(std::move(render_view));
			if (resolve_view)
			{
				views.push_back(std::move(resolve_view));
			}
			return attachment;
		}

		bool attachment_references(const attachment_reference& attachment, u64 resource_uid)
		{
			return attachment.resource_uid == resource_uid || attachment.resolve_resource_uid == resource_uid;
		}
	}

	struct framebuffer::impl
	{
		framebuffer_attachment_set attachment_set;
		std::vector<std::unique_ptr<image_view>> views;
		std::atomic<u64> last_use{0};
		std::atomic<u32> idle_checks{0};
	};

	framebuffer::framebuffer(const framebuffer_request& request)
		: m_impl(std::make_unique<impl>())
	{
		m_impl->attachment_set.configure(request.width, request.height, request.layers,
			request.samples, request.reads_attachments);
		m_impl->views.reserve((maximum_color_attachments + 2) * 2);

		for (u32 index = 0; index < maximum_color_attachments; ++index)
		{
			if (!request.colors[index])
			{
				if (request.colors[index].has_resolve())
				{
					fmt::throw_exception("Metal color resolve attachment %u has no multisample source", index);
				}
				continue;
			}
			m_impl->attachment_set.set_color(index,
				create_attachment(m_impl->views, request.colors[index], attachment_role::color));
		}

		if (request.depth)
		{
			m_impl->attachment_set.set_depth(
				create_attachment(m_impl->views, request.depth, attachment_role::depth));
		}
		else if (request.depth.has_resolve())
		{
			fmt::throw_exception("Metal depth resolve attachment has no multisample source");
		}

		if (request.stencil)
		{
			m_impl->attachment_set.set_stencil(
				create_attachment(m_impl->views, request.stencil, attachment_role::stencil));
		}
		else if (request.stencil.has_resolve())
		{
			fmt::throw_exception("Metal stencil resolve attachment has no multisample source");
		}

		m_impl->attachment_set.validate();
	}

	framebuffer::~framebuffer() = default;

	void framebuffer::touch(u64 serial)
	{
		m_impl->last_use.store(serial, std::memory_order_release);
		m_impl->idle_checks.store(0, std::memory_order_release);
	}

	u32 framebuffer::note_idle_check()
	{
		return m_impl->idle_checks.fetch_add(1, std::memory_order_acq_rel) + 1;
	}

	const framebuffer_attachment_set& framebuffer::attachments() const
	{
		return m_impl->attachment_set;
	}

	u32 framebuffer::width() const { return m_impl->attachment_set.width(); }
	u32 framebuffer::height() const { return m_impl->attachment_set.height(); }
	u32 framebuffer::layers() const { return m_impl->attachment_set.layers(); }
	u32 framebuffer::samples() const { return m_impl->attachment_set.samples(); }
	u64 framebuffer::signature() const { return m_impl->attachment_set.signature(); }
	u64 framebuffer::last_use_serial() const { return m_impl->last_use.load(std::memory_order_acquire); }
	usz framebuffer::view_count() const { return m_impl->views.size(); }

	bool framebuffer::references_resource(u64 resource_uid) const
	{
		if (!resource_uid)
		{
			return false;
		}
		for (u32 index = 0; index < m_impl->attachment_set.color_count(); ++index)
		{
			if (attachment_references(m_impl->attachment_set.color(index), resource_uid))
			{
				return true;
			}
		}
		return attachment_references(m_impl->attachment_set.depth(), resource_uid) ||
			attachment_references(m_impl->attachment_set.stencil(), resource_uid);
	}

	struct framebuffer_cache::impl
	{
		mutable std::mutex mutex;
		std::unordered_map<framebuffer_request_key, std::shared_ptr<framebuffer>,
			framebuffer_request_key_hash> entries;
		framebuffer_cache_statistics stats;
		u64 use_serial = 0;
	};

	framebuffer_cache::framebuffer_cache()
		: m_impl(std::make_unique<impl>())
	{
	}

	framebuffer_cache::~framebuffer_cache() = default;

	std::shared_ptr<framebuffer> framebuffer_cache::acquire(const framebuffer_request& request)
	{
		const auto key = make_request_key(request);
		std::lock_guard lock(m_impl->mutex);
		const u64 serial = ++m_impl->use_serial;
		if (const auto found = m_impl->entries.find(key); found != m_impl->entries.end())
		{
			++m_impl->stats.hits;
			found->second->touch(serial);
			return found->second;
		}

		++m_impl->stats.misses;
		auto result = std::shared_ptr<framebuffer>(new framebuffer(request));
		result->touch(serial);
		m_impl->entries.emplace(key, result);
		++m_impl->stats.created;
		return result;
	}

	usz framebuffer_cache::invalidate_resource(u64 resource_uid)
	{
		if (!resource_uid)
		{
			fmt::throw_exception("Cannot invalidate Metal framebuffers for resource UID zero");
		}
		std::lock_guard lock(m_impl->mutex);
		usz removed = 0;
		for (auto current = m_impl->entries.begin(); current != m_impl->entries.end();)
		{
			if (current->second->references_resource(resource_uid))
			{
				current = m_impl->entries.erase(current);
				++removed;
			}
			else
			{
				++current;
			}
		}
		m_impl->stats.invalidated += removed;
		return removed;
	}

	usz framebuffer_cache::remove_unused(u32 required_idle_checks)
	{
		if (!required_idle_checks)
		{
			fmt::throw_exception("Metal framebuffer idle-check threshold must be nonzero");
		}
		std::lock_guard lock(m_impl->mutex);
		usz removed = 0;
		for (auto current = m_impl->entries.begin(); current != m_impl->entries.end();)
		{
			auto& entry = current->second;
			if (entry.use_count() != 1)
			{
				entry->touch(entry->last_use_serial());
				++current;
				continue;
			}
			if (entry->note_idle_check() < required_idle_checks)
			{
				++current;
				continue;
			}
			current = m_impl->entries.erase(current);
			++removed;
		}
		m_impl->stats.retired += removed;
		return removed;
	}

	void framebuffer_cache::clear()
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->stats.retired += m_impl->entries.size();
		m_impl->entries.clear();
	}

	usz framebuffer_cache::size() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->entries.size();
	}

	framebuffer_cache_statistics framebuffer_cache::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		auto result = m_impl->stats;
		result.entries = m_impl->entries.size();
		result.buckets = m_impl->entries.bucket_count();
		return result;
	}

	namespace
	{
		framebuffer_cache& global_framebuffer_cache()
		{
			static framebuffer_cache cache;
			return cache;
		}
	}

	std::shared_ptr<framebuffer> get_framebuffer(const framebuffer_request& request)
	{
		return global_framebuffer_cache().acquire(request);
	}

	usz invalidate_framebuffers_for_resource(u64 resource_uid)
	{
		return global_framebuffer_cache().invalidate_resource(resource_uid);
	}

	usz remove_unused_framebuffers(u32 required_idle_checks)
	{
		return global_framebuffer_cache().remove_unused(required_idle_checks);
	}

	void clear_framebuffer_cache()
	{
		global_framebuffer_cache().clear();
	}

	framebuffer_cache_statistics get_framebuffer_cache_statistics()
	{
		return global_framebuffer_cache().statistics();
	}
}
