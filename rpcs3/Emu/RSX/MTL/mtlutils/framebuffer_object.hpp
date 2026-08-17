#pragma once

#include <algorithm>
#include <array>

#include "Utilities/StrFmt.h"
#include "graphics_pipeline_state.hpp"
#include "image.h"

namespace mtl
{
	struct attachment_reference
	{
		texture_handle texture = nullptr;
		texture_handle resolve_texture = nullptr;
		u64 resource_uid = 0;
		u64 resolve_resource_uid = 0;
		u64 pixel_format = 0;
		u32 width = 0;
		u32 height = 0;
		u32 depth = 1;
		u32 array_length = 1;
		u32 mip_level = 0;
		u32 array_slice = 0;
		u32 depth_plane = 0;
		u32 resolve_mip_level = 0;
		u32 resolve_array_slice = 0;
		u32 resolve_depth_plane = 0;
		u32 sample_count = 1;
		u8 aspects = texture_aspect_none;
		bool memoryless = false;
		bool layered = false;

		[[nodiscard]] explicit operator bool() const
		{
			return texture && resource_uid && pixel_format && width && height && depth && sample_count;
		}

		[[nodiscard]] bool has_resolve() const
		{
			return resolve_texture && resolve_resource_uid;
		}

		[[nodiscard]] bool operator==(const attachment_reference&) const = default;

		[[nodiscard]] static attachment_reference from_view(const image& resource,
			const image_view& view, u32 mip_level = 0, u32 array_slice = 0, u32 depth_plane = 0)
		{
			if (!resource || !view || view.image_uid() != resource.uid() ||
				mip_level >= view.range().mip_count || array_slice >= view.range().slice_count)
			{
				fmt::throw_exception("Invalid Metal render attachment view");
			}
			const u32 resource_mip = view.range().first_mip + mip_level;
			const u32 mip_depth = std::max(1u, resource.depth() >> resource_mip);
			if (resource_mip >= resource.mipmaps() || depth_plane >= mip_depth)
			{
				fmt::throw_exception("Metal render attachment subresource is out of range");
			}
			attachment_reference result;
			result.texture = view.native_handle();
			result.resource_uid = resource.uid();
			result.pixel_format = view.format();
			result.width = std::max(1u, resource.width() >> resource_mip);
			result.height = std::max(1u, resource.height() >> resource_mip);
			result.depth = mip_depth;
			result.array_length = view.range().slice_count;
			result.mip_level = mip_level;
			result.array_slice = array_slice;
			result.depth_plane = depth_plane;
			result.sample_count = resource.samples();
			result.aspects = resource.aspects();
			result.memoryless = resource.is_memoryless();
			result.layered = result.array_length > 1;
			return result;
		}

		void set_resolve(const image& resource, const image_view& view,
			u32 mip_level = 0, u32 array_slice = 0, u32 depth_plane = 0)
		{
			const attachment_reference resolve = from_view(resource, view, mip_level, array_slice, depth_plane);
			if (sample_count == 1 || resolve.sample_count != 1 || resolve.memoryless ||
				resolve.pixel_format != pixel_format || resolve.width < width ||
				resolve.height < height || resolve.depth < depth)
			{
				fmt::throw_exception("Incompatible Metal resolve attachment");
			}
			resolve_texture = resolve.texture;
			resolve_resource_uid = resolve.resource_uid;
			resolve_mip_level = mip_level;
			resolve_array_slice = array_slice;
			resolve_depth_plane = depth_plane;
		}
	};

	class framebuffer_attachment_set
	{
		std::array<attachment_reference, maximum_color_attachments> m_colors{};
		attachment_reference m_depth;
		attachment_reference m_stencil;
		u32 m_width = 0;
		u32 m_height = 0;
		u32 m_layers = 1;
		u32 m_samples = 1;
		u32 m_color_count = 0;
		bool m_reads_attachments = false;

		void validate_attachment(const attachment_reference& attachment, u8 required_aspects) const
		{
			if (!attachment || !(attachment.aspects & required_aspects) ||
				attachment.width < m_width || attachment.height < m_height ||
				attachment.sample_count != m_samples || attachment.depth_plane >= attachment.depth ||
				attachment.array_slice >= attachment.array_length ||
				m_layers > attachment.array_length - attachment.array_slice)
			{
				fmt::throw_exception("Metal render attachment is incompatible with its attachment set");
			}
			if (attachment.has_resolve() && m_samples == 1)
			{
				fmt::throw_exception("Single-sampled Metal attachment cannot have a resolve target");
			}
		}

	public:
		framebuffer_attachment_set() = default;

		framebuffer_attachment_set(u32 width, u32 height, u32 layers, u32 samples,
			bool reads_attachments = false)
		{
			configure(width, height, layers, samples, reads_attachments);
		}

		void configure(u32 width, u32 height, u32 layers, u32 samples, bool reads_attachments = false)
		{
			if (!width || !height || !layers ||
				(samples != 1 && samples != 2 && samples != 4 && samples != 8))
			{
				fmt::throw_exception("Invalid Metal framebuffer-equivalent extent or sample count");
			}
			clear();
			m_width = width;
			m_height = height;
			m_layers = layers;
			m_samples = samples;
			m_reads_attachments = reads_attachments;
		}

		void clear()
		{
			m_colors = {};
			m_depth = {};
			m_stencil = {};
			m_color_count = 0;
		}

		void set_color(u32 index, const attachment_reference& attachment)
		{
			if (index >= maximum_color_attachments)
			{
				fmt::throw_exception("Metal color attachment index %u is out of range", index);
			}
			validate_attachment(attachment, texture_aspect_color);
			m_colors[index] = attachment;
			m_color_count = std::max(m_color_count, index + 1);
		}

		void clear_color(u32 index)
		{
			if (index >= maximum_color_attachments)
			{
				fmt::throw_exception("Metal color attachment index %u is out of range", index);
			}
			m_colors[index] = {};
			while (m_color_count && !m_colors[m_color_count - 1]) --m_color_count;
		}

		void set_depth(const attachment_reference& attachment)
		{
			validate_attachment(attachment, texture_aspect_depth);
			m_depth = attachment;
		}

		void set_stencil(const attachment_reference& attachment)
		{
			validate_attachment(attachment, texture_aspect_stencil);
			m_stencil = attachment;
		}

		void clear_depth()
		{
			m_depth = {};
		}

		void clear_stencil()
		{
			m_stencil = {};
		}

		void validate() const
		{
			if (!m_width || !m_height || !m_layers || !m_samples)
			{
				fmt::throw_exception("Metal framebuffer-equivalent attachment set is not configured");
			}
			bool has_attachment = false;
			for (u32 index = 0; index < m_color_count; ++index)
			{
				if (!m_colors[index]) continue;
				validate_attachment(m_colors[index], texture_aspect_color);
				has_attachment = true;
			}
			if (m_depth)
			{
				validate_attachment(m_depth, texture_aspect_depth);
				has_attachment = true;
			}
			if (m_stencil)
			{
				validate_attachment(m_stencil, texture_aspect_stencil);
				has_attachment = true;
			}
			if (!has_attachment)
			{
				fmt::throw_exception("Metal framebuffer-equivalent attachment set contains no attachments");
			}
			if (m_depth && m_stencil && m_depth.resource_uid == m_stencil.resource_uid &&
				(m_depth.mip_level != m_stencil.mip_level || m_depth.array_slice != m_stencil.array_slice ||
					m_depth.depth_plane != m_stencil.depth_plane))
			{
				fmt::throw_exception("Packed Metal depth and stencil attachments reference different subresources");
			}
		}

		[[nodiscard]] bool matches(const framebuffer_attachment_set& other) const
		{
			return *this == other;
		}

		[[nodiscard]] const attachment_reference& color(u32 index) const
		{
			if (index >= maximum_color_attachments)
			{
				fmt::throw_exception("Metal color attachment index %u is out of range", index);
			}
			return m_colors[index];
		}

		[[nodiscard]] const attachment_reference& depth() const { return m_depth; }
		[[nodiscard]] const attachment_reference& stencil() const { return m_stencil; }
		[[nodiscard]] u32 width() const { return m_width; }
		[[nodiscard]] u32 height() const { return m_height; }
		[[nodiscard]] u32 layers() const { return m_layers; }
		[[nodiscard]] u32 samples() const { return m_samples; }
		[[nodiscard]] u32 color_count() const { return m_color_count; }
		[[nodiscard]] bool reads_attachments() const { return m_reads_attachments; }

		[[nodiscard]] u64 color_format(u32 index = 0) const
		{
			return color(index).pixel_format;
		}

		[[nodiscard]] u64 depth_format() const
		{
			return m_depth.pixel_format;
		}

		[[nodiscard]] bool operator==(const framebuffer_attachment_set&) const = default;

		[[nodiscard]] u64 signature() const
		{
			u64 hash = 0x243f6a8885a308d3ull;
			auto mix = [&](u64 value)
			{
				hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
			};
			auto mix_attachment = [&](const attachment_reference& attachment)
			{
				mix(attachment.resource_uid);
				mix(attachment.resolve_resource_uid);
				mix(attachment.pixel_format);
				mix(attachment.mip_level);
				mix(attachment.array_slice);
				mix(attachment.depth_plane);
				mix(attachment.array_length);
				mix(attachment.resolve_mip_level);
				mix(attachment.resolve_array_slice);
				mix(attachment.resolve_depth_plane);
				mix(attachment.sample_count);
				mix(attachment.aspects);
				mix(attachment.memoryless);
			};
			mix(m_width);
			mix(m_height);
			mix(m_layers);
			mix(m_samples);
			mix(m_color_count);
			mix(m_reads_attachments);
			for (const auto& attachment : m_colors) mix_attachment(attachment);
			mix_attachment(m_depth);
			mix_attachment(m_stencil);
			return hash;
		}
	};

	struct framebuffer_attachment_set_hash
	{
		[[nodiscard]] usz operator()(const framebuffer_attachment_set& attachments) const noexcept
		{
			return static_cast<usz>(attachments.signature());
		}
	};
}
