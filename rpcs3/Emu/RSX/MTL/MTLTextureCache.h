#pragma once

#include "MTLCompute.h"
#include "MTLDMA.h"
#include "MTLRenderTargets.h"
#include "MTLResourceManager.h"
#include "mtlutils/sync.h"

#include "Emu/RSX/Common/texture_cache.h"
#include "Emu/RSX/Common/tiled_dma_copy.hpp"

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mtl
{
	class cached_texture_section;
	class texture_cache;

	struct texture_cache_traits
	{
		using commandbuffer_type = command_buffer;
		using section_storage_type = cached_texture_section;
		using texture_cache_type = texture_cache;
		using texture_cache_base_type = rsx::texture_cache<texture_cache_type, texture_cache_traits>;
		using image_resource_type = image*;
		using image_view_type = image_view*;
		using image_storage_type = image;
		using texture_format = u64;
		using viewable_image_type = viewable_image*;
	};

	class cached_texture_section final :
		public rsx::cached_texture_section<cached_texture_section, texture_cache_traits>
	{
		using baseclass = rsx::cached_texture_section<cached_texture_section, texture_cache_traits>;
		friend baseclass;

		struct dma_completion_state;

		std::unique_ptr<viewable_image> m_managed_texture;
		std::unique_ptr<viewable_image> m_scaled_texture;
		std::unique_ptr<buffer> m_dma_buffer;
		std::shared_ptr<dma_completion_state> m_dma_completion;
		render_device* m_device = nullptr;
		memory_allocator* m_allocator = nullptr;
		viewable_image* m_vram_texture = nullptr;
		u64 m_native_format = 0;
		u64 m_dma_offset = 0;
		u64 m_dma_length = 0;
		bool m_dma_mapped = false;

		void release_dma_resources();
		void prepare_dma_completion(command_buffer& command);

	public:
		using baseclass::cached_texture_section;

		void create(u16 width, u16 height, u16 depth, u16 mipmaps, image* resource,
			u32 pitch, bool managed, u32 format, bool swap_bytes = false);
		void set_dimensions(u16 width, u16 height, u16 depth, u32 pitch);
		void set_unpack_swap_bytes(bool swap_bytes);
		void set_rsx_pitch(u32 pitch);

		void dma_transfer(command_buffer& command, image* source, const areai& source_area,
			const utils::address_range32& valid_range, u32 pitch);
		void copy_texture(command_buffer& command, bool miss);
		void imp_flush() override;
		void dma_abort() override;
		void* map_synchronized(u32 offset, u32 size);
		void finish_flush();

		void destroy();
		[[nodiscard]] bool exists() const;
		[[nodiscard]] bool is_managed() const;
		[[nodiscard]] bool is_flushed() const;
		[[nodiscard]] bool is_depth_texture() const;
		[[nodiscard]] bool has_compatible_format(image* resource) const;
		[[nodiscard]] u64 get_format() const;
		[[nodiscard]] image_view* get_view(const rsx::texture_channel_remap_t& remap);
		[[nodiscard]] image_view* get_raw_view();
		[[nodiscard]] viewable_image* get_raw_texture() const;
		[[nodiscard]] std::unique_ptr<viewable_image>& get_texture();
		[[nodiscard]] render_target* get_render_target() const;
		void sync_surface_memory(const rsx::simple_array<cached_texture_section*>& surfaces);
	};

	class texture_cache final : public rsx::texture_cache<texture_cache, texture_cache_traits>
	{
		using baseclass = rsx::texture_cache<texture_cache, texture_cache_traits>;
		friend baseclass;

		struct cached_image_reference
		{
			std::unique_ptr<viewable_image> data;
			texture_cache* parent = nullptr;

			cached_image_reference(texture_cache& cache, std::unique_ptr<viewable_image>& previous);
			~cached_image_reference();
		};

		struct cached_image
		{
			u64 key = 0;
			std::unique_ptr<viewable_image> data;

			cached_image() = default;
			cached_image(u64 key, std::unique_ptr<viewable_image>& resource);
		};

		struct active_temporary_image
		{
			std::unique_ptr<viewable_image> data;
			u32 references = 1;
		};

		render_device* m_device = nullptr;
		memory_allocator* m_allocator = nullptr;
		surface_cache* m_surface_cache = nullptr;
		std::deque<cached_image> m_cached_images;
		std::unordered_map<image*, active_temporary_image> m_temporary_images;
		atomic_t<u64> m_cached_memory_size = 0;
		shared_mutex m_cached_pool_lock;
		atomic_t<bool> m_cache_is_exiting = false;

		static constexpr u32 max_cached_image_pool_size = 256;

		void clear();
		[[nodiscard]] component_mapping apply_component_mapping_flags(u32 format,
			rsx::component_order flags, const rsx::texture_channel_remap_t& remap) const;
		void copy_transfer_regions_impl(command_buffer& command, image* destination,
			const rsx::simple_array<copy_region_descriptor>& sections) const;
		[[nodiscard]] image* get_template_from_collection_impl(
			const rsx::simple_array<copy_region_descriptor>& sections) const;
		[[nodiscard]] std::unique_ptr<viewable_image> find_cached_image(
			const image_create_info& info);

	protected:
		[[nodiscard]] image_view* create_temporary_subresource_view_impl(command_buffer& command,
			image* source, texture_type image_type, texture_type view_type, u32 format,
			u16 x, u16 y, u16 width, u16 height, u16 depth, u8 mipmaps,
			const rsx::texture_channel_remap_t& remap, bool copy);
		[[nodiscard]] image_view* create_temporary_subresource_view(command_buffer& command,
			const deferred_subresource& description) override;
		[[nodiscard]] image_view* generate_cubemap_from_images(command_buffer& command,
			const deferred_subresource& description) override;
		[[nodiscard]] image_view* generate_3d_from_2d_images(command_buffer& command,
			const deferred_subresource& description) override;
		[[nodiscard]] image_view* generate_atlas_from_images(command_buffer& command,
			const deferred_subresource& description) override;
		[[nodiscard]] image_view* generate_2d_mipmaps_from_images(command_buffer& command,
			const deferred_subresource& description) override;
		void release_temporary_subresource(image_view* view) override;

		void initialize_subresource_from_memory(command_buffer& command, image* destination,
			const deferred_subresource& description, rsx::texture_dimension_extended type) const;
		void update_image_contents(command_buffer& command, image_view* destination,
			const deferred_subresource& description) override;

		cached_texture_section* create_new_texture(command_buffer& command,
			const utils::address_range32& range, u16 width, u16 height, u16 depth,
			u16 mipmaps, u32 pitch, u32 format, rsx::texture_upload_context context,
			rsx::texture_dimension_extended type, bool swizzled,
			rsx::component_order component_order, rsx::flags32_t flags) override;
		cached_texture_section* create_nul_section(command_buffer& command,
			const utils::address_range32& range, const rsx::image_section_attributes_t& attributes,
			const rsx::GCM_tile_reference& tile, bool memory_load) override;
		cached_texture_section* upload_image_from_cpu(command_buffer& command,
			const utils::address_range32& range, u16 width, u16 height, u16 depth,
			u16 mipmaps, u32 pitch, u32 format, rsx::texture_upload_context context,
			const std::vector<rsx::subresource_layout>& layouts,
			rsx::texture_dimension_extended type, bool swizzled) override;
		void set_component_order(cached_texture_section& section, u32 format,
			rsx::component_order expected) override;
		void insert_texture_barrier(command_buffer& command, image* resource,
			bool strong_ordering) override;
		[[nodiscard]] bool render_target_format_is_compatible(image* resource,
			u32 format) override;
		void prepare_for_dma_transfers(command_buffer& command) override;
		void cleanup_after_dma_transfers(command_buffer& command) override;

	public:
		enum texture_create_flag : u32
		{
			initialize_image_contents = 1 << 0,
			do_not_reuse = 1 << 1,
			shareable = 1 << 2,
			mutable_format = 1 << 3,
		};

		using baseclass::texture_cache;

		void initialize(render_device& device, memory_allocator& allocator,
			surface_cache& surfaces);
		void destroy() override;
		void on_section_destroyed(cached_texture_section& texture) override;

		[[nodiscard]] std::unique_ptr<viewable_image> create_temporary_subresource_storage(
			rsx::format_class format_class, u64 format, u16 width, u16 height, u16 depth,
			u16 layers, u8 mipmaps, texture_type type, u32 image_flags, u32 usage_flags);
		void dispose_reusable_image(std::unique_ptr<viewable_image>& resource);

		[[nodiscard]] bool is_depth_texture(u32 address, u32 size) override;
		void on_frame_end() override;
		[[nodiscard]] std::unique_ptr<viewable_image> upload_image_simple_owned(
			command_buffer& command, u64 format, u32 address, u32 width, u32 height,
			u32 pitch, std::unique_ptr<buffer>& staging_lifetime);
		[[nodiscard]] viewable_image* upload_image_simple(command_buffer& command,
			u64 format, u32 address, u32 width, u32 height, u32 pitch);
		[[nodiscard]] bool blit(const rsx::blit_src_info& source,
			const rsx::blit_dst_info& destination, bool interpolate,
			surface_cache& surfaces, command_buffer& command);
		[[nodiscard]] u32 get_unreleased_textures_count() const override;
		[[nodiscard]] bool handle_memory_pressure(rsx::problem_severity severity) override;
		[[nodiscard]] u64 get_temporary_memory_in_use() const;
		[[nodiscard]] bool is_overallocated() const;
		[[nodiscard]] memory_allocator& allocator() const;
	};

	[[nodiscard]] u64 hash_image_properties(const image_create_info& info);

	[[nodiscard]] texture_type get_texture_type(rsx::texture_dimension_extended type,
		u16 depth, u16 layers = 1);
	void upload_texture(command_buffer& command, memory_allocator& allocator,
		image& destination, u32 format,
		bool swizzled, const std::vector<rsx::subresource_layout>& layouts,
		const rsx::GCM_tile_reference* tile = nullptr);
}
