#include "stdafx.h"
#include "MTLTextureCache.h"

#include "Emu/RSX/Common/TextureUtils.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace mtl
{
	namespace
	{
		[[nodiscard]] u32 checked_u32(u64 value, std::string_view operation)
		{
			if (value > std::numeric_limits<u32>::max())
				fmt::throw_exception("Metal texture %s exceeds the 32-bit compute range", operation);
			return static_cast<u32>(value);
		}

		[[nodiscard]] u64 checked_multiply(u64 left, u64 right, std::string_view operation)
		{
			if (right && left > std::numeric_limits<u64>::max() / right)
				fmt::throw_exception("Metal texture %s size overflows", operation);
			return left * right;
		}

		[[nodiscard]] u64 checked_add(u64 left, u64 right, std::string_view operation)
		{
			if (left > std::numeric_limits<u64>::max() - right)
				fmt::throw_exception("Metal texture %s size overflows", operation);
			return left + right;
		}

		[[nodiscard]] u32 source_address(const void* pointer)
		{
			const auto [address, valid] = vm::try_get_addr(pointer);
			if (!valid)
				fmt::throw_exception("Metal tiled texture source is outside guest memory");
			return address;
		}

		template <usz Bytes>
		void detile_blocks(std::span<std::byte> destination, const void* source,
			const rsx::GCM_tile_reference& tile, u32 address, u16 width, u16 height)
		{
			using block = std::array<u8, Bytes>;
			rsx::tile_texel_data<block, true>(destination.data(), source,
				tile.base_address, address - tile.base_address, tile.tile->size,
				tile.tile->bank, ::narrow<u16>(tile.tile->pitch), width, height);
		}

		[[nodiscard]] std::vector<std::byte> detile_layout(const rsx::subresource_layout& layout,
			u32 format, const rsx::GCM_tile_reference& tile)
		{
			if (!tile || layout.data.empty())
				fmt::throw_exception("Invalid Metal tiled texture layout");
			const u32 block_size = rsx::get_format_block_size_in_bytes(format);
			if (!block_size || tile.tile->pitch % block_size)
				fmt::throw_exception("Metal tiled texture pitch is incompatible with its block size");
			const u64 size = checked_multiply(tile.tile->pitch,
				checked_multiply(layout.height_in_block, layout.depth, "detile depth"),
				"detile storage");
			std::vector<std::byte> result(size);
			const u32 address = source_address(layout.data.data());
			switch (block_size)
			{
			case 1: detile_blocks<1>(result, layout.data.data(), tile, address,
				layout.width_in_block, layout.height_in_block); break;
			case 2: detile_blocks<2>(result, layout.data.data(), tile, address,
				layout.width_in_block, layout.height_in_block); break;
			case 4: detile_blocks<4>(result, layout.data.data(), tile, address,
				layout.width_in_block, layout.height_in_block); break;
			case 8: detile_blocks<8>(result, layout.data.data(), tile, address,
				layout.width_in_block, layout.height_in_block); break;
			case 16: detile_blocks<16>(result, layout.data.data(), tile, address,
				layout.width_in_block, layout.height_in_block); break;
			default:
				fmt::throw_exception("Unsupported Metal tiled texture block size %u", block_size);
			}
			return result;
		}

		struct upload_footprint
		{
			u64 decoded_row = 0;
			u64 decoded_image = 0;
			u64 decoded_length = 0;
			u64 depth_offset = 0;
			u64 stencil_offset = 0;
			u64 allocation_length = 0;
		};

		[[nodiscard]] upload_footprint get_upload_footprint(
			const rsx::subresource_layout& layout, u32 format,
			const native_format_description& native)
		{
			const bool depth16_float = format == CELL_GCM_TEXTURE_DEPTH16_FLOAT;
			const bool depth_stencil = bool(native.aspects & texture_aspect_stencil);
			const u64 rows = (native.capabilities & format_capability_compressed)
				? layout.height_in_block : layout.height_in_texel;
			const u64 blocks = (native.capabilities & format_capability_compressed)
				? layout.width_in_block : layout.width_in_texel;
			// Both RSX depth/stencil encodings arrive as one packed 32-bit word per
			// texel. Metal stores D32S8 as independent 32-bit depth and 8-bit stencil
			// planes, so size the decoded source from the packed representation rather
			// than the native Metal block size.
			const u64 decoded_element_size = depth16_float ? 2 : (depth_stencil ? 4 : native.bytes_per_block);
			upload_footprint result;
			result.decoded_row = utils::align(checked_multiply(blocks, decoded_element_size,
				"decoded row"), 4ull);
			result.decoded_image = checked_multiply(result.decoded_row, rows, "decoded image");
			result.decoded_length = checked_multiply(result.decoded_image, layout.depth,
				"decoded volume");
			result.allocation_length = utils::align(result.decoded_length, 256ull);
			if (depth16_float)
			{
				result.depth_offset = result.allocation_length;
				const u64 converted = checked_multiply(result.decoded_length, 2, "expanded depth16");
				result.allocation_length = utils::align(
					checked_add(result.depth_offset, converted, "expanded depth16"), 256ull);
			}
			else if (depth_stencil)
			{
				result.depth_offset = result.allocation_length;
				result.stencil_offset = utils::align(checked_add(result.depth_offset,
					result.decoded_length, "depth plane"), 256ull);
				const u64 stencil_length = result.decoded_length / 4;
				result.allocation_length = utils::align(checked_add(result.stencil_offset,
					stencil_length, "stencil plane"), 256ull);
			}
			return result;
		}

		void validate_layout(const image& destination, const rsx::subresource_layout& layout)
		{
			if (layout.level >= destination.mipmaps() ||
				(destination.type() != texture_type::texture_3d && layout.layer >= destination.layers()) ||
				!layout.width_in_texel || !layout.height_in_texel || !layout.depth ||
				layout.width_in_texel > std::max(destination.width() >> layout.level, 1u) ||
				layout.height_in_texel > std::max(destination.height() >> layout.level, 1u) ||
				layout.depth > std::max(destination.depth() >> layout.level, 1u))
			{
				fmt::throw_exception("Metal texture upload subresource exceeds the destination image");
			}
		}

		void make_compute_write_visible_to_transfer(command_buffer& command)
		{
			if (command.active_encoder() != encoder_kind::compute)
				fmt::throw_exception("Metal texture conversion requires an active compute encoder");
			barrier_plan visibility;
			visibility.scope = barrier_scope::within_encoder;
			visibility.after_stages = stage_dispatch;
			visibility.before_stages = stage_blit;
			visibility.flush_caches = true;
			encode_barrier(command.active_native_encoder(), visibility);
		}
	}

	void upload_texture(command_buffer& command, memory_allocator& allocator,
		image& destination, u32 format,
		bool swizzled, const std::vector<rsx::subresource_layout>& layouts,
		const rsx::GCM_tile_reference* tile)
	{
		if (!command.is_recording() || !destination || layouts.empty())
			fmt::throw_exception("Invalid Metal texture upload request");
		const auto native = get_sampler_format(command.allocator().owner().info(), format);
		if (!native || destination.format() != native.pixel_format)
			fmt::throw_exception("Metal texture upload format does not match its destination");

		rsx::texture_uploader_capabilities capabilities{
			.supports_byteswap = false,
			.supports_vtc_decoding = false,
			.supports_hw_deswizzle = false,
			.supports_zero_copy = false,
			.supports_dxt = command.allocator().owner().info().formats.bc_texture_compression,
			.alignment = 4,
		};
		std::vector<std::unique_ptr<buffer>> staging_resources;
		staging_resources.reserve(layouts.size());
		for (const rsx::subresource_layout& original_layout : layouts)
		{
			validate_layout(destination, original_layout);
			rsx::subresource_layout layout = original_layout;
			std::vector<std::byte> detiled;
			if (tile && *tile)
			{
				if (layout.level || layout.layer || layout.depth != 1)
					fmt::throw_exception("Metal tiled upload accepts one two-dimensional base subresource");
				detiled = detile_layout(layout, format, *tile);
				layout.data = rsx::io_buffer(detiled.data(), detiled.size());
				layout.pitch_in_block = (*tile).tile->pitch /
					rsx::get_format_block_size_in_bytes(format);
			}

			const upload_footprint footprint = get_upload_footprint(layout, format, native);
				auto staging = std::make_unique<buffer>(allocator, buffer_create_info{
				.size = footprint.allocation_length,
				.usage = buffer_usage_storage | buffer_usage_copy_source |
					buffer_usage_copy_destination,
				.storage = storage_mode::shared,
				.cache = cpu_cache_mode::write_combined,
				.access = cpu_access::read_write,
				.hazards = hazard_tracking::tracked,
				.pool = allocation_pool::texture_cache,
				.label = "RPCS3 texture upload staging",
			});
			auto* mapped = static_cast<std::byte*>(staging->map(0, footprint.allocation_length));
			std::memset(mapped, 0, footprint.allocation_length);
			rsx::io_buffer output(mapped, footprint.decoded_length);
			capabilities.alignment = footprint.decoded_row;
			const rsx::texture_memory_info operation =
				rsx::upload_texture_subresource(output, layout, format, swizzled, capabilities);
			if (operation.require_swap || operation.require_deswizzle || operation.require_upload ||
				!operation.deferred_cmds.empty())
			{
				fmt::throw_exception("Metal CPU texture decode returned deferred work unexpectedly");
			}
			staging->did_modify(0, footprint.allocation_length);
			staging->unmap();

			const texture_subresource subresource{
				.mip_level = layout.level,
				.array_slice = destination.type() == texture_type::texture_3d ? 0u : layout.layer,
				.aspects = native.aspects,
			};
			const texture_extent extent{
				.width = layout.width_in_texel,
				.height = layout.height_in_texel,
				.depth = layout.depth,
			};
			if (format == CELL_GCM_TEXTURE_DEPTH16_FLOAT)
			{
				get_compute_task<cs_fconvert_task<u16, u32>>()->run(command, *staging, 0,
					checked_u32(footprint.decoded_length, "depth16 source"),
					checked_u32(footprint.depth_offset, "depth16 destination offset"));
				make_compute_write_visible_to_transfer(command);
				const buffer_image_copy_region region{
					.buffer_offset = footprint.depth_offset,
					.bytes_per_row = footprint.decoded_row * 2,
					.bytes_per_image = footprint.decoded_image * 2,
					.subresource = {.mip_level = subresource.mip_level,
						.array_slice = subresource.array_slice, .aspects = texture_aspect_depth},
					.extent = extent,
				};
				upload_image(command, *staging, destination, std::span{&region, 1});
			}
			else if (native.aspects & texture_aspect_stencil)
			{
				const u32 decoded_length = checked_u32(footprint.decoded_length,
					"depth/stencil source");
				if (native.bytes_per_block == 4)
				{
					get_compute_task<cs_scatter_d24x8>()->run(command, *staging, 0,
						decoded_length, checked_u32(footprint.depth_offset, "depth offset"),
						checked_u32(footprint.stencil_offset, "stencil offset"));
				}
				else if (format == CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT)
				{
					get_compute_task<cs_scatter_d32x8<true>>()->run(command, *staging, 0,
						decoded_length, checked_u32(footprint.depth_offset, "depth offset"),
						checked_u32(footprint.stencil_offset, "stencil offset"));
				}
				else
				{
					get_compute_task<cs_scatter_d32x8<false>>()->run(command, *staging, 0,
						decoded_length, checked_u32(footprint.depth_offset, "depth offset"),
						checked_u32(footprint.stencil_offset, "stencil offset"));
				}
				make_compute_write_visible_to_transfer(command);
				const std::array regions = {
					buffer_image_copy_region{
						.buffer_offset = footprint.depth_offset,
						.bytes_per_row = footprint.decoded_row,
						.bytes_per_image = footprint.decoded_image,
						.subresource = {.mip_level = subresource.mip_level,
							.array_slice = subresource.array_slice, .aspects = texture_aspect_depth},
						.extent = extent,
					},
					buffer_image_copy_region{
						.buffer_offset = footprint.stencil_offset,
						.bytes_per_row = layout.width_in_texel,
						.bytes_per_image = static_cast<u64>(layout.width_in_texel) *
							layout.height_in_texel,
						.subresource = {.mip_level = subresource.mip_level,
							.array_slice = subresource.array_slice, .aspects = texture_aspect_stencil},
						.extent = extent,
					},
				};
				upload_image(command, *staging, destination, regions);
			}
			else
			{
				const buffer_image_copy_region region{
					.buffer_offset = 0,
					.bytes_per_row = footprint.decoded_row,
					.bytes_per_image = footprint.decoded_image,
					.subresource = subresource,
					.extent = extent,
				};
				upload_image(command, *staging, destination, std::span{&region, 1});
			}
			staging_resources.emplace_back(std::move(staging));
		}

		for (auto& staging : staging_resources)
		{
			const u64 size = staging->size();
			get_resource_manager().retire(staging, {
				.resource_class = managed_resource_class::buffer,
				.bytes = size,
				.label = "RPCS3 texture upload staging",
			});
		}
	}
}
