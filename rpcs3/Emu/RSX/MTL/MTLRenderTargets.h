#pragma once

#include "MTLFormats.h"
#include "mtlutils/image_helpers.h"
#include "Emu/RSX/Common/surface_store.h"

#include <memory>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace mtl
{
	class surface_dma_buffer;

	namespace surface_cache_utils
	{
		void dispose(surface_dma_buffer* resource);
	}

	class image_reference_sync_barrier
	{
		u32 m_texture_barrier_count = 0;
		u32 m_draw_barrier_count = 0;
		bool m_allow_skip_barrier = true;

	public:
		void on_insert_texture_barrier();
		void on_insert_draw_barrier();
		void allow_skip();
		void reset();
		[[nodiscard]] bool can_skip() const;
		[[nodiscard]] bool is_enabled() const;
		[[nodiscard]] bool requires_post_loop_barrier() const;
	};

	class surface_dma_buffer final : public buffer
	{
		void* m_host_memory = nullptr;

	public:
		u32 base_address = 0;

		surface_dma_buffer(render_device& device, u64 size, u32 address,
			std::string_view label = "RPCS3 surface DMA buffer");
		~surface_dma_buffer();
		surface_dma_buffer(const surface_dma_buffer&) = delete;
		surface_dma_buffer& operator=(const surface_dma_buffer&) = delete;
	};

	class render_target final : public viewable_image,
		public rsx::render_target_descriptor<viewable_image*>
	{
		memory_allocator* m_allocator = nullptr;
		image_create_info m_create_info;
		native_format_description m_format;
		component_mapping m_native_components;
		image_reference_sync_barrier m_cyclic_reference;
		std::unique_ptr<buffer> m_spilled_memory;
		u64 m_spilled_row_bytes = 0;
		bool m_depth_surface = false;
		bool m_memory_initialized = false;

		[[nodiscard]] viewable_image* get_resolve_target_safe(command_buffer& command);
		void resolve(command_buffer& command);
		void unresolve(command_buffer& command);
		void clear_memory(command_buffer& command, image& surface);
		void load_memory(command_buffer& command);
		void initialize_memory(command_buffer& command, rsx::surface_access access);
		void unspill(command_buffer& command);
		[[nodiscard]] std::vector<buffer_image_copy_region> build_spill_transfer_descriptors(image& target) const;

	public:
		u64 frame_tag = 0;
		u64 last_rw_access_tag = 0;
		u64 spill_request_tag = 0;
		bool is_bound = false;

		render_target(memory_allocator& allocator, const image_create_info& info,
			const native_format_description& format, bool depth_surface);
		~render_target() override;

		viewable_image* get_surface(rsx::surface_access access_type) override;
		[[nodiscard]] bool is_depth_surface() const override;
		[[nodiscard]] bool matches_dimensions(u16 width, u16 height) const;
		void reset_surface_counters();
		[[nodiscard]] image_view* get_view(const rsx::texture_channel_remap_t& remap,
			u8 aspect_mask = texture_aspect_color | texture_aspect_depth | texture_aspect_stencil);
		[[nodiscard]] image_view* get_view(u8 aspect_mask =
			texture_aspect_color | texture_aspect_depth | texture_aspect_stencil);

		[[nodiscard]] bool spill(command_buffer& command,
			std::vector<std::unique_ptr<viewable_image>>& resolve_cache);
		void texture_barrier(command_buffer& command);
		void post_texture_barrier(command_buffer& command);
		void memory_barrier(command_buffer& command, rsx::surface_access access);
		void read_barrier(command_buffer& command);
		void write_barrier(command_buffer& command);

		void set_native_component_layout(component_mapping mapping);
		[[nodiscard]] component_mapping native_component_layout() const;
		[[nodiscard]] const native_format_description& native_format() const;
		[[nodiscard]] memory_allocator& allocator() const;
		[[nodiscard]] bool memory_initialized() const;
		[[nodiscard]] bool spilled() const;
	};

	[[nodiscard]] render_target* as_render_target(image* resource);
	[[nodiscard]] const render_target* as_render_target(const image* resource);

	void resolve_image(command_buffer& command, viewable_image& destination, viewable_image& source);
	void unresolve_image(command_buffer& command, viewable_image& destination, viewable_image& source);

	struct surface_cache_traits
	{
		using surface_storage_type = std::unique_ptr<render_target>;
		using surface_type = render_target*;
		using buffer_object_storage_type = std::unique_ptr<surface_dma_buffer>;
		using buffer_object_type = surface_dma_buffer*;
		using command_list_type = command_buffer&;
		using download_buffer_object = void*;
		using barrier_descriptor_t = rsx::deferred_clipped_region<render_target*>;

		static std::unique_ptr<render_target> create_new_surface(u32 address,
			rsx::surface_color_format format, usz width, usz height, usz pitch,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& resolution_scaling,
			memory_allocator& allocator, render_device& device, command_buffer& command);
		static std::unique_ptr<render_target> create_new_surface(u32 address,
			rsx::surface_depth_format2 format, usz width, usz height, usz pitch,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& resolution_scaling,
			memory_allocator& allocator, render_device& device, command_buffer& command);

		static void clone_surface(command_buffer& command, std::unique_ptr<render_target>& destination,
			render_target* source, u32 address, barrier_descriptor_t& previous,
			const rsx::surface_scaling_config_t& scaling);
		static std::unique_ptr<render_target> convert_pitch(command_buffer& command,
			std::unique_ptr<render_target>& source, usz output_pitch);
		[[nodiscard]] static bool is_compatible_surface(const render_target* surface,
			const render_target* reference, u16 width, u16 height, u8 sample_count);
		static void prepare_surface_for_drawing(command_buffer& command, render_target* surface);
		static void prepare_surface_for_sampling(command_buffer& command, render_target* surface);
		[[nodiscard]] static bool surface_is_pitch_compatible(
			const std::unique_ptr<render_target>& surface, usz pitch);

		static void invalidate_surface_contents(command_buffer& command, render_target* surface,
			rsx::surface_color_format format, u32 address, usz pitch);
		static void invalidate_surface_contents(command_buffer& command, render_target* surface,
			rsx::surface_depth_format2 format, u32 address, usz pitch);
		static void notify_surface_invalidated(const std::unique_ptr<render_target>& surface);
		static void notify_surface_persist(const std::unique_ptr<render_target>& surface);
		static void notify_surface_reused(const std::unique_ptr<render_target>& surface);

		[[nodiscard]] static bool surface_matches_properties(
			const std::unique_ptr<render_target>& surface, rsx::surface_color_format format,
			usz width, usz height, rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& scaling, bool check_references = false);
		[[nodiscard]] static bool surface_matches_properties(
			const std::unique_ptr<render_target>& surface, rsx::surface_depth_format2 format,
			usz width, usz height, rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& scaling, bool check_references = false);

		static void spill_buffer(std::unique_ptr<surface_dma_buffer>& resource);
		static void unspill_buffer(std::unique_ptr<surface_dma_buffer>& resource);
		static void write_render_target_to_memory(command_buffer& command, surface_dma_buffer* destination,
			render_target* surface, u64 destination_offset, u64 source_offset, u64 maximum_copy_length);

		template <int BlockSize>
		static surface_dma_buffer* merge_bo_list(command_buffer& command,
			std::vector<surface_dma_buffer*>& resources)
		{
			u64 required_size = 0;
			u64 prefix_size = 0;
			u32 base_address = 0;
			bool found_base = false;
			for (const auto* resource : resources)
			{
				if (!found_base && resource)
				{
					if (resource->base_address < prefix_size)
						fmt::throw_exception("Invalid Metal surface DMA block ordering");
					base_address = resource->base_address - static_cast<u32>(prefix_size);
					found_base = true;
				}
				const u64 size = resource ? resource->size() : BlockSize;
				required_size += size;
				prefix_size += size;
			}
			if (!required_size || required_size > std::numeric_limits<u32>::max())
				fmt::throw_exception("Invalid Metal surface DMA merge size");
			auto* destination = new surface_dma_buffer(command.allocator().owner(), required_size,
				base_address,
				"RPCS3 merged surface DMA buffer");
			u64 offset = 0;
			for (auto*& resource : resources)
			{
				if (!resource)
				{
					offset += BlockSize;
					continue;
				}
				std::memcpy(static_cast<u8*>(destination->map(offset, resource->size())),
					resource->map(0, resource->size()), resource->size());
				destination->unmap();
				resource->unmap();
				offset += resource->size();
				surface_cache_utils::dispose(resource);
				resource = nullptr;
			}
			return destination;
		}

		template <typename T>
		[[nodiscard]] static T* get(const std::unique_ptr<T>& resource)
		{
			return resource.get();
		}
	};

	class surface_cache final : public rsx::surface_store<surface_cache_traits>
	{
		[[nodiscard]] u64 get_surface_cache_memory_quota(u64 total_device_memory) const;

	public:
		void destroy();
		[[nodiscard]] bool spill_unused_memory(command_buffer& command,
			std::vector<std::unique_ptr<viewable_image>>& resolve_cache);
		[[nodiscard]] bool is_overallocated(const render_device& device) const;
		[[nodiscard]] bool can_collapse_surface(const std::unique_ptr<render_target>& surface,
			rsx::problem_severity severity) override;
		[[nodiscard]] bool handle_memory_pressure(command_buffer& command,
			rsx::problem_severity severity) override;
		void trim(command_buffer& command, rsx::problem_severity memory_pressure);
	};
}
