#include "stdafx.h"
#include "MTLCompute.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mtl
{
	namespace
	{
		constexpr u32 parameter_alignment = 256;

		[[nodiscard]] u32 divide_up(u32 value, u32 divisor)
		{
			if (!divisor) fmt::throw_exception("Metal compute division by zero");
			return value / divisor + (value % divisor != 0);
		}

		[[nodiscard]] u32 checked_multiply(u32 left, u32 right, std::string_view operation)
		{
			if (left && right > std::numeric_limits<u32>::max() / left)
				fmt::throw_exception("Metal compute %s size overflows", operation);
			return left * right;
		}

		[[nodiscard]] u32 checked_add(u32 left, u32 right, std::string_view operation)
		{
			if (right > std::numeric_limits<u32>::max() - left)
				fmt::throw_exception("Metal compute %s size overflows", operation);
			return left + right;
		}

		[[nodiscard]] std::string bool_literal(bool value)
		{
			return value ? "true" : "false";
		}

		[[nodiscard]] std::string common_source()
		{
			return R"MSL(
#include <metal_stdlib>
using namespace metal;

inline uint rsx_bswap_u16(uint value)
{
	return ((value & 0x00ff00ffu) << 8) | ((value & 0xff00ff00u) >> 8);
}
inline uint rsx_bswap_u32(uint value)
{
	return ((value & 0x000000ffu) << 24) | ((value & 0x0000ff00u) << 8) |
		((value & 0x00ff0000u) >> 8) | ((value & 0xff000000u) >> 24);
}
inline uint rsx_bswap_u16_u32(uint value) { return (value << 16) | (value >> 16); }
inline uint rsx_depth24_to_float(uint value)
{
	return as_type<uint>(float(value) / 16777215.0f);
}
inline uint rsx_float_to_depth24(uint value)
{
	return uint(as_type<float>(value) * 16777215.0f);
}
inline uint rsx_depth24_float_to_float(uint value) { return value << 7; }
inline uint rsx_float_to_depth24_float(uint value) { return value >> 7; }
inline uint rsx_depth24_swap(uint value)
{
	return (value & 0x0000ff00u) | ((value & 0x00ff0000u) >> 16) |
		((value & 0x000000ffu) << 16);
}
)MSL";
		}

		[[nodiscard]] std::string shuffle_source(const compute_kernel_specification& specification)
		{
			std::string operation;
			switch (specification.shuffle)
			{
			case compute_shuffle_operation::byte_swap_u16: operation = "rsx_bswap_u16(value)"; break;
			case compute_shuffle_operation::byte_swap_u32: operation = "rsx_bswap_u32(value)"; break;
			case compute_shuffle_operation::byte_swap_u16_u32: operation = "rsx_bswap_u16_u32(value)"; break;
			case compute_shuffle_operation::depth24_to_float32: operation = "rsx_depth24_to_float(value >> 8)"; break;
			case compute_shuffle_operation::float32_to_depth24_swapped:
				operation = "rsx_depth24_swap(rsx_float_to_depth24(value))"; break;
			case compute_shuffle_operation::depth24_byte_swap: operation = "rsx_depth24_swap(value)"; break;
			}

			return common_source() + R"MSL(
struct parameters { uint word_count; uint3 reserved; };
kernel void rsx_compute(device uint* data [[buffer(0)]],
	constant parameters& params [[buffer(1)]], uint3 position [[thread_position_in_grid]],
	uint3 grid [[threads_per_grid]])
{
	uint index = position.x + grid.x * (position.y + grid.y * position.z);
	if (index >= params.word_count) return;
	uint value = data[index];
	data[index] = )MSL" + operation + ";\n}\n";
		}

		[[nodiscard]] std::string interleave_source(const compute_kernel_specification& specification)
		{
			const bool gather = specification.depth_stencil == depth_stencil_operation::gather_depth24 ||
				specification.depth_stencil == depth_stencil_operation::gather_depth32;
			const bool depth32 = specification.depth_stencil == depth_stencil_operation::gather_depth32 ||
				specification.depth_stencil == depth_stencil_operation::scatter_depth32;
			std::string source = common_source() + R"MSL(
struct parameters
{
	uint word_count;
	uint depth_offset;
	uint stencil_offset;
	uint reserved;
};
kernel void rsx_compute(device uint* data [[buffer(0)]],
	constant parameters& params [[buffer(1)]], uint3 position [[thread_position_in_grid]],
	uint3 grid [[threads_per_grid]])
{
	uint index = position.x + grid.x * (position.y + grid.y * position.z);
	if (index >= params.word_count) return;
	uint stencil_word = params.stencil_offset + index / 4u;
	uint stencil_shift = (index & 3u) * 8u;
)MSL";
			if (gather)
			{
				source += "\tuint depth = data[index + params.depth_offset];\n";
				if (depth32)
					source += specification.depth_float ?
						"\tdepth = rsx_float_to_depth24_float(depth);\n" :
						"\tdepth = rsx_float_to_depth24(depth);\n";
				source += R"MSL(
	uint stencil = (data[stencil_word] >> stencil_shift) & 0xffu;
	uint value = (depth << 8) | stencil;
)MSL";
				if (specification.swap_destination) source += "\tvalue = rsx_bswap_u32(value);\n";
				source += "\tdata[index] = value;\n";
			}
			else
			{
				source += R"MSL(
	uint value = data[index];
	uint depth = value >> 8;
)MSL";
				if (depth32)
					source += specification.depth_float ?
						"\tdepth = rsx_depth24_float_to_float(depth);\n" :
						"\tdepth = rsx_depth24_to_float(depth);\n";
				source += R"MSL(
	data[index + params.depth_offset] = depth;
	uint stencil = (value & 0xffu) << stencil_shift;
	device atomic_uint* atomic_data = reinterpret_cast<device atomic_uint*>(data);
	atomic_fetch_or_explicit(&atomic_data[stencil_word], stencil, memory_order_relaxed);
)MSL";
			}
			return source + "}\n";
		}

		[[nodiscard]] std::string format_convert_source(const compute_kernel_specification& specification)
		{
			const bool contract = specification.source_element_size == 4;
			std::string source = common_source() + R"MSL(
struct parameters
{
	uint invocation_count;
	uint source_offset;
	uint destination_offset;
	uint reserved;
};
inline uint2 rsx_unpack_e4m12(uint value)
{
	uint2 result = uint2(value & 0xffffu, value >> 16);
	return (result << 11) + (120u << 23);
}
inline uint rsx_pack_e4m12(uint2 value)
{
	uint2 result = (value - (120u << 23)) >> 11;
	return (result.x & 0xffffu) | (result.y << 16);
}
kernel void rsx_compute(device uint* data [[buffer(0)]],
	constant parameters& params [[buffer(1)]], uint3 position [[thread_position_in_grid]],
	uint3 grid [[threads_per_grid]])
{
	uint index = position.x + grid.x * (position.y + grid.y * position.z);
	if (index >= params.invocation_count) return;
)MSL";
			if (contract)
			{
				source += R"MSL(
	uint2 value = uint2(data[params.source_offset + index * 2u],
		data[params.source_offset + index * 2u + 1u]);
)MSL";
				if (specification.swap_source) source += "\tvalue = uint2(rsx_bswap_u32(value.x), rsx_bswap_u32(value.y));\n";
				source += "\tuint packed = rsx_pack_e4m12(value);\n";
				if (specification.swap_destination) source += "\tpacked = rsx_bswap_u16(packed);\n";
				source += "\tdata[params.destination_offset + index] = packed;\n";
			}
			else
			{
				source += "\tuint packed = data[params.source_offset + index];\n";
				if (specification.swap_source) source += "\tpacked = rsx_bswap_u16(packed);\n";
				source += "\tuint2 value = rsx_unpack_e4m12(packed);\n";
				if (specification.swap_destination) source += "\tvalue = uint2(rsx_bswap_u32(value.x), rsx_bswap_u32(value.y));\n";
				source += R"MSL(
	data[params.destination_offset + index * 2u] = value.x;
	data[params.destination_offset + index * 2u + 1u] = value.y;
)MSL";
			}
			return source + "}\n";
		}

		[[nodiscard]] std::string aggregate_source()
		{
			return common_source() + R"MSL(
struct parameters { uint word_count; uint3 reserved; };
kernel void rsx_compute(device const uint* source [[buffer(0)]],
	device atomic_uint* destination [[buffer(1)]], constant parameters& params [[buffer(2)]],
	uint3 position [[thread_position_in_grid]], uint3 grid [[threads_per_grid]])
{
	uint index = position.x + grid.x * (position.y + grid.y * position.z);
	if (index < params.word_count)
		atomic_fetch_add_explicit(destination, source[index], memory_order_relaxed);
}
)MSL";
		}

		[[nodiscard]] std::string deswizzle_source(const compute_kernel_specification& specification)
		{
			std::ostringstream source;
			source << common_source() << R"MSL(
struct parameters
{
	uint width; uint height; uint depth; uint log_width;
	uint log_height; uint log_depth; uint mipmap_count; uint reserved;
};
struct invocation_properties
{
	uint data_offset;
	uint3 size;
	uint3 size_log2;
};
inline bool rsx_init_properties(uint offset, constant parameters& params,
	thread invocation_properties& invocation)
{
	invocation.data_offset = 0;
	invocation.size = uint3(params.width, params.height, params.depth);
	invocation.size_log2 = uint3(params.log_width, params.log_height, params.log_depth);
	uint level_end = params.width * params.height * params.depth;
	uint level = 1;
	while (offset >= level_end && level < params.mipmap_count)
	{
		invocation.data_offset = level_end;
		invocation.size.xy = max(invocation.size.xy / 2u, uint2(1));
		invocation.size_log2.xy = max(invocation.size_log2.xy, uint2(1)) - 1u;
		level_end += invocation.size.x * invocation.size.y * params.depth;
		++level;
	}
	return offset < level_end;
}
inline uint rsx_z_index(uint x, uint y, uint z, thread const invocation_properties& invocation)
{
	uint offset = 0, shift = 0;
	uint log_width = invocation.size_log2.x;
	uint log_height = invocation.size_log2.y;
	uint log_depth = invocation.size_log2.z;
	do
	{
		if (log_width > 0) { offset |= (x & 1u) << shift++; x >>= 1; --log_width; }
		if (log_height > 0) { offset |= (y & 1u) << shift++; y >>= 1; --log_height; }
		if (log_depth > 0) { offset |= (z & 1u) << shift++; z >>= 1; --log_depth; }
	}
	while (x > 0 || y > 0 || z > 0);
	return offset;
}
)MSL";
			source << "inline uint rsx_transform(uint value) { return ";
			if (specification.swap_destination && specification.destination_element_size == 4)
				source << "rsx_bswap_u32(value)";
			else if (specification.swap_destination && specification.destination_element_size == 2)
				source << "rsx_bswap_u16(value)";
			else
				source << "value";
			source << "; }\n";
			source << "constant uint rsx_block_size = " << specification.block_size << "u;\n";
			source << R"MSL(
kernel void rsx_compute(device const uint* source [[buffer(0)]], device uint* destination [[buffer(1)]],
	constant parameters& params [[buffer(2)]], uint3 position [[thread_position_in_grid]],
	uint3 grid [[threads_per_grid]])
{
	uint linear_id = position.x + grid.x * (position.y + grid.y * position.z);
	uint texel_id = linear_id;
	if (rsx_block_size == 1u) texel_id *= 4u;
	else if (rsx_block_size == 2u) texel_id *= 2u;
	invocation_properties invocation;
	if (!rsx_init_properties(texel_id, params, invocation)) return;
	uint row_length = invocation.size.x;
	uint slice_length = invocation.size.y * row_length;
	uint level_offset = texel_id - invocation.data_offset;
	uint slice_offset = level_offset % slice_length;
	uint z = level_offset / slice_length;
	uint y = slice_offset / row_length;
	uint x = slice_offset % row_length;
	if (rsx_block_size == 1u)
	{
		uint accumulator = 0;
		uint count = min(4u, row_length - x);
		for (uint subword = 0; subword < count; ++subword)
		{
			uint source_texel = rsx_z_index(x + subword, y, z, invocation) + invocation.data_offset;
			uint value = (source[source_texel / 4u] >> ((source_texel & 3u) * 8u)) & 0xffu;
			accumulator |= value << (subword * 8u);
		}
		destination[texel_id / 4u] = accumulator;
	}
	else if (rsx_block_size == 2u)
	{
		uint accumulator = 0;
		uint count = min(2u, row_length - x);
		for (uint subword = 0; subword < count; ++subword)
		{
			uint source_texel = rsx_z_index(x + subword, y, z, invocation) + invocation.data_offset;
			uint value = (source[source_texel / 2u] >> ((source_texel & 1u) * 16u)) & 0xffffu;
			accumulator |= value << (subword * 16u);
		}
		destination[texel_id / 2u] = rsx_transform(accumulator);
	}
	else
	{
		uint source_texel = rsx_z_index(x, y, z, invocation) + invocation.data_offset;
		uint word_count = rsx_block_size / 4u;
		for (uint word = 0; word < word_count; ++word)
			destination[texel_id * word_count + word] =
				rsx_transform(source[source_texel * word_count + word]);
	}
}
)MSL";
			return source.str();
		}

		[[nodiscard]] std::string tile_copy_source(const compute_kernel_specification& specification)
		{
			std::ostringstream source;
			source << common_source() << R"MSL(
struct parameters
{
	uint prime; uint factor; uint tiles_per_row; uint tile_base_address;
	uint tile_size; uint tile_address_offset; uint tile_rw_offset; uint tile_pitch;
	uint tile_bank; uint image_width; uint image_height; uint image_pitch;
	uint image_bytes_per_pixel; uint3 reserved;
};
constant uint rsx_bank_distribution[16] =
	{0,1,2,3, 2,3,0,1, 1,2,3,0, 3,0,1,2};

inline uint4 rsx_read(device const uint* data, uint offset, uint bytes_per_pixel)
{
	switch (bytes_per_pixel)
	{
	case 16: return uint4(data[offset*4u], data[offset*4u+1u], data[offset*4u+2u], data[offset*4u+3u]);
	case 8: return uint4(data[offset*2u], data[offset*2u+1u], 0u, 0u);
	case 4: return uint4(data[offset], 0u, 0u, 0u);
	case 2: return uint4((data[offset/2u] >> ((offset&1u)*16u)) & 0xffffu, 0u, 0u, 0u);
	case 1: return uint4((data[offset/4u] >> ((offset&3u)*8u)) & 0xffu, 0u, 0u, 0u);
	default: return 0u;
	}
}
inline void rsx_write(device uint* data, uint offset, uint bytes_per_pixel, uint4 value)
{
	switch (bytes_per_pixel)
	{
	case 16: data[offset*4u]=value.x; data[offset*4u+1u]=value.y; data[offset*4u+2u]=value.z; data[offset*4u+3u]=value.w; break;
	case 8: data[offset*2u]=value.x; data[offset*2u+1u]=value.y; break;
	case 4: data[offset]=value.x; break;
	case 2:
	{
		uint word_offset=offset/2u, shift=(offset&1u)*16u, mask=0xffffu<<shift;
		data[word_offset]=(data[word_offset]&~mask)|((value.x&0xffffu)<<shift); break;
	}
	case 1:
	{
		uint word_offset=offset/4u, shift=(offset&3u)*8u, mask=0xffu<<shift;
		data[word_offset]=(data[word_offset]&~mask)|((value.x&0xffu)<<shift); break;
	}
	default: break;
	}
}
inline uint rsx_tiled_offset(uint row, uint column, constant parameters& params)
{
	uint row_offset=row*params.tile_pitch+params.tile_base_address+params.tile_address_offset;
	uint address=row_offset+column*params.image_bytes_per_pixel;
	uint texel_offset=(address-params.tile_base_address)/256u;
	uint tile_x=texel_offset%params.tiles_per_row;
	uint tile_y=(texel_offset/params.tiles_per_row)/64u;
	uint tile_id=tile_y*params.tiles_per_row+tile_x;
	uint tile_selector=(tile_id+(params.tile_base_address>>14))&0x3ffffu;
	uint row_address=(tile_selector>>2)&0xffffu;
	uint bank_selector=0;
	if(params.factor==1u) bank_selector=tile_selector&3u;
	else if(params.factor==2u)
	{
		uint index=((tile_selector+((tile_y&1u)<<1))&3u)*4u+(tile_y&3u);
		bank_selector=rsx_bank_distribution[index];
	}
	else if(params.factor>=4u)
		bank_selector=rsx_bank_distribution[(tile_selector&3u)*4u+(tile_y&3u)];
	bank_selector=(bank_selector+params.tile_bank)&3u;
	uint line=(texel_offset/params.tiles_per_row)%64u;
	uint column_selector=((line>>3)&7u)<<7;
	column_selector|=((address>>5)&7u)<<4;
	column_selector|=(line&3u)<<2;
	uint partition=(((line>>2)&1u)+((address>>6)&1u))&1u;
	uint tiled=(row_address&0xffffu)<<16;
	tiled|=(bank_selector&3u)<<14;
	tiled|=((column_selector>>4)&0x3fu)<<8;
	tiled|=partition<<7;
	tiled|=((column_selector>>2)&3u)<<5;
	tiled|=address&0x1fu;
	tiled^=(((tiled>>12)^((bank_selector^tile_selector)&1u)^(tiled>>14))&1u)<<9;
	tiled^=((tiled>>11)&1u)<<10;
	return tiled-params.tile_base_address-params.tile_rw_offset;
}
)MSL";
			source << "constant bool rsx_encode = " << bool_literal(specification.detiler == rsx_detiler_operation::encode) << ";\n";
			source << R"MSL(
kernel void rsx_compute(device uint* tiled [[buffer(0)]], device uint* linear [[buffer(1)]],
	constant parameters& params [[buffer(2)]], uint2 position [[thread_position_in_grid]])
{
	uint iterations=params.image_bytes_per_pixel<4u?4u/params.image_bytes_per_pixel:1u;
	uint row=position.y, first_column=position.x*iterations;
	if(row>=params.image_height||first_column>=params.image_width)return;
	for(uint column=first_column;column<min(first_column+iterations,params.image_width);++column)
	{
		uint absolute_address=row*params.tile_pitch+params.tile_base_address+
			params.tile_address_offset+column*params.image_bytes_per_pixel;
		uint base_offset=absolute_address-params.tile_base_address;
		if(base_offset>=params.tile_size)continue;
		uint tiled_offset=rsx_tiled_offset(row,column,params)/params.image_bytes_per_pixel;
		uint linear_offset=(row*params.image_pitch)/params.image_bytes_per_pixel+column;
		if(rsx_encode)rsx_write(tiled,tiled_offset,params.image_bytes_per_pixel,
			rsx_read(linear,linear_offset,params.image_bytes_per_pixel));
		else rsx_write(linear,linear_offset,params.image_bytes_per_pixel,
			rsx_read(tiled,tiled_offset,params.image_bytes_per_pixel));
	}
}
)MSL";
			return source.str();
		}

		[[nodiscard]] std::string kernel_source(const compute_kernel_specification& specification)
		{
			switch (specification.kind)
			{
			case compute_kernel_kind::shuffle: return shuffle_source(specification);
			case compute_kernel_kind::gather_depth_stencil:
			case compute_kernel_kind::scatter_depth_stencil: return interleave_source(specification);
			case compute_kernel_kind::format_convert: return format_convert_source(specification);
			case compute_kernel_kind::deswizzle_3d: return deswizzle_source(specification);
			case compute_kernel_kind::aggregate: return aggregate_source();
			case compute_kernel_kind::tile_copy: return tile_copy_source(specification);
			}
			throw std::logic_error("Unknown Metal compute kernel kind");
		}
	}

	void compute_kernel_specification::validate() const
	{
		if ((source_element_size != 1 && source_element_size != 2 && source_element_size != 4 &&
			source_element_size != 8 && source_element_size != 16) ||
			(destination_element_size != 1 && destination_element_size != 2 &&
			destination_element_size != 4 && destination_element_size != 8 &&
			destination_element_size != 16) || !block_size)
		{
			fmt::throw_exception("Invalid Metal compute kernel element sizes");
		}
		if (kind == compute_kernel_kind::format_convert &&
			!((source_element_size == 4 && destination_element_size == 2) ||
			(source_element_size == 2 && destination_element_size == 4)))
		{
			fmt::throw_exception("Metal E4M12 conversion requires 16-bit and 32-bit element sizes");
		}
		if (kind == compute_kernel_kind::deswizzle_3d &&
			(block_size < destination_element_size || block_size % destination_element_size))
		{
			fmt::throw_exception("Invalid Metal 3D deswizzle block layout");
		}
	}

	u32 compute_kernel_specification::data_buffer_count() const
	{
		return kind == compute_kernel_kind::deswizzle_3d || kind == compute_kernel_kind::aggregate ||
			kind == compute_kernel_kind::tile_copy ? 2 : 1;
	}

	std::string compute_kernel_specification::label() const
	{
		switch (kind)
		{
		case compute_kernel_kind::shuffle:
			switch (shuffle)
			{
			case compute_shuffle_operation::byte_swap_u16: return "RPCS3 byte-swap 16";
			case compute_shuffle_operation::byte_swap_u32: return "RPCS3 byte-swap 32";
			case compute_shuffle_operation::byte_swap_u16_u32: return "RPCS3 half-word swap";
			case compute_shuffle_operation::depth24_to_float32: return "RPCS3 depth24 to float32";
			case compute_shuffle_operation::float32_to_depth24_swapped: return "RPCS3 float32 to depth24 swapped";
			case compute_shuffle_operation::depth24_byte_swap: return "RPCS3 depth24 byte swap";
			}
			break;
		case compute_kernel_kind::gather_depth_stencil: return "RPCS3 depth-stencil gather";
		case compute_kernel_kind::scatter_depth_stencil: return "RPCS3 depth-stencil scatter";
		case compute_kernel_kind::format_convert: return "RPCS3 E4M12 format conversion";
		case compute_kernel_kind::deswizzle_3d: return "RPCS3 3D deswizzle";
		case compute_kernel_kind::aggregate: return "RPCS3 word aggregation";
		case compute_kernel_kind::tile_copy: return detiler == rsx_detiler_operation::decode ?
			"RPCS3 tiled memory decode" : "RPCS3 tiled memory encode";
		}
		fmt::throw_exception("Unknown Metal compute kernel label");
	}

	void compute_buffer_range::validate() const
	{
		if (!resource || !*resource || !length || !resource->in_range(offset, length) ||
			(offset & 3) || (length & 3))
		{
			fmt::throw_exception("Invalid Metal compute buffer range");
		}
	}

	void compute_dispatch_size::validate() const
	{
		if (!x || !y || !z)
			fmt::throw_exception("Metal compute dispatch dimensions must be nonzero");
	}

	struct compute_task::impl
	{
		compute_kernel_specification specification;
		render_device* device = nullptr;
		MTLPipelineCompiler* compiler = nullptr;
		std::shared_ptr<MTLProgramPipeline> pipeline;
		compute_task_statistics counters;
		mutable std::mutex mutex;

		explicit impl(compute_kernel_specification value)
			: specification(std::move(value))
		{
			specification.validate();
		}
	};

	compute_task::compute_task(compute_kernel_specification specification)
		: m_impl(std::make_unique<impl>(std::move(specification)))
	{
	}

	compute_task::~compute_task()
	{
		destroy();
	}

	void compute_task::create(render_device& device, MTLPipelineCompiler& compiler)
	{
		if (!device || !compiler || &compiler.owner() != &device)
			fmt::throw_exception("Metal compute task requires a compiler for its render device");
		std::lock_guard lock(m_impl->mutex);
		if (m_impl->pipeline)
		{
			if (m_impl->device == &device && m_impl->compiler == &compiler) return;
			fmt::throw_exception("Metal compute task is already initialized for another device");
		}
		compute_pipeline_compile_request request;
		request.source = kernel_source(m_impl->specification);
		request.function_name = "rsx_compute";
		request.layout = {m_impl->specification.data_buffer_count() + 1, 0, 0, false};
		request.maximum_threads_per_threadgroup = std::min(device.info().limits.max_threads_per_threadgroup, 256u);
		request.label = m_impl->specification.label();
		for (u32 index = 0; index < m_impl->specification.data_buffer_count(); ++index)
			request.required_bindings.push_back({msl_shader_stage::compute, argument_binding_class::buffer,
				index, umax, "data buffer " + std::to_string(index)});
		request.required_bindings.push_back({msl_shader_stage::compute, argument_binding_class::buffer,
			m_impl->specification.data_buffer_count(), umax, "kernel parameters"});
		m_impl->pipeline = compiler.compile_compute_inline(request);
		m_impl->device = &device;
		m_impl->compiler = &compiler;
	}

	void compute_task::destroy()
	{
		if (!m_impl) return;
		std::lock_guard lock(m_impl->mutex);
		m_impl->pipeline.reset();
		m_impl->device = nullptr;
		m_impl->compiler = nullptr;
		m_impl->counters = {};
	}

	void compute_task::dispatch(command_buffer& command, std::span<const compute_buffer_range> buffers,
		const void* parameters, usz parameter_size, compute_dispatch_size invocations)
	{
		invocations.validate();
		if (!parameters || !parameter_size || parameter_size > parameter_alignment)
			fmt::throw_exception("Invalid Metal compute parameter payload");
		for (const auto& range : buffers) range.validate();

		std::shared_ptr<MTLProgramPipeline> pipeline;
		render_device* device = nullptr;
		{
			std::lock_guard lock(m_impl->mutex);
			if (!m_impl->pipeline || !m_impl->device)
				fmt::throw_exception("Metal compute task is not initialized");
			if (buffers.size() != m_impl->specification.data_buffer_count())
				fmt::throw_exception("Metal compute task received an incorrect data-buffer count");
			pipeline = m_impl->pipeline;
			device = m_impl->device;
		}
		if (!command.is_recording() || &command.allocator().owner() != device)
			fmt::throw_exception("Metal compute task requires a recording command buffer for its device");
		if (command.active_encoder() == encoder_kind::render) command.end_encoding();
		if (command.active_encoder() == encoder_kind::none) (void)command.begin_compute_encoding();

		id<MTLBuffer> parameter_buffer = [device->native_handle() newBufferWithLength:parameter_alignment
			options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
		if (!parameter_buffer) fmt::throw_exception("Metal compute parameter-buffer allocation failed");
		parameter_buffer.label = [NSString stringWithUTF8String:(m_impl->specification.label() + " parameters").c_str()];
		std::memcpy(parameter_buffer.contents, parameters, parameter_size);

		auto binding_instance = pipeline->create_binding_instance();
		for (u32 index = 0; index < buffers.size(); ++index)
		{
			const auto& range = buffers[index];
			binding_instance->set_buffer(msl_shader_stage::compute, index, *range.resource,
				range.offset, range.length, 0, range.access);
		}
		argument_buffer_binding parameter_binding;
		parameter_binding.resource = parameter_buffer;
		parameter_binding.gpu_address = parameter_buffer.gpuAddress;
		parameter_binding.length = parameter_alignment;
		parameter_binding.access = argument_access::read;
		binding_instance->set_buffer(msl_shader_stage::compute, static_cast<u32>(buffers.size()), parameter_binding);
		binding_instance->bind(command);

		id<MTL4ComputeCommandEncoder> encoder = (__bridge id<MTL4ComputeCommandEncoder>)command.active_native_encoder();
		const id<MTLComputePipelineState> native_pipeline = binding_instance->compute_pipeline();
		const NSUInteger group_x = std::min<NSUInteger>(native_pipeline.maxTotalThreadsPerThreadgroup, 256);
		[encoder dispatchThreads:MTLSizeMake(invocations.x, invocations.y, invocations.z)
			threadsPerThreadgroup:MTLSizeMake(group_x, 1, 1)];
		command.set_flag(command_has_dma_transfer);
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->counters.dispatches++;
			m_impl->counters.invocations += static_cast<u64>(invocations.x) * invocations.y * invocations.z;
			m_impl->counters.parameter_bytes += parameter_size;
		}
	}

	compute_task::operator bool() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->pipeline != nullptr;
	}

	const compute_kernel_specification& compute_task::specification() const
	{
		return m_impl->specification;
	}

	compute_task_statistics compute_task::statistics() const
	{
		std::lock_guard lock(m_impl->mutex);
		return m_impl->counters;
	}

	namespace
	{
		compute_kernel_specification shuffle_specification(compute_shuffle_operation operation)
		{
			compute_kernel_specification result;
			result.kind = compute_kernel_kind::shuffle;
			result.shuffle = operation;
			return result;
		}

		compute_kernel_specification interleave_specification(depth_stencil_operation operation,
			bool swap_bytes, bool depth_float)
		{
			compute_kernel_specification result;
			result.kind = operation == depth_stencil_operation::gather_depth24 ||
				operation == depth_stencil_operation::gather_depth32 ?
				compute_kernel_kind::gather_depth_stencil : compute_kernel_kind::scatter_depth_stencil;
			result.depth_stencil = operation;
			result.swap_destination = swap_bytes;
			result.depth_float = depth_float;
			return result;
		}

		compute_kernel_specification format_specification(u32 source_size, u32 destination_size,
			bool swap_source, bool swap_destination)
		{
			compute_kernel_specification result;
			result.kind = compute_kernel_kind::format_convert;
			result.source_element_size = source_size;
			result.destination_element_size = destination_size;
			result.swap_source = swap_source;
			result.swap_destination = swap_destination;
			return result;
		}

		compute_kernel_specification deswizzle_specification(u32 block_size, u32 base_size,
			bool swap_bytes)
		{
			compute_kernel_specification result;
			result.kind = compute_kernel_kind::deswizzle_3d;
			result.block_size = block_size;
			result.source_element_size = block_size;
			result.destination_element_size = base_size;
			result.swap_destination = swap_bytes;
			return result;
		}

		compute_kernel_specification simple_specification(compute_kernel_kind kind)
		{
			compute_kernel_specification result;
			result.kind = kind;
			return result;
		}

		compute_kernel_specification tile_specification(rsx_detiler_operation operation)
		{
			auto result = simple_specification(compute_kernel_kind::tile_copy);
			result.detiler = operation;
			return result;
		}

		[[nodiscard]] u32 aligned_to_word(u32 value)
		{
			if (value > std::numeric_limits<u32>::max() - 3)
				fmt::throw_exception("Metal compute buffer size overflows alignment");
			return (value + 3) & ~3u;
		}
	}

	cs_shuffle_base::cs_shuffle_base(compute_shuffle_operation operation)
		: compute_task(shuffle_specification(operation))
	{
	}

	void cs_shuffle_base::run(command_buffer& command, const buffer& data, u32 data_length, u32 data_offset)
	{
		if (!data_length) return;
		if ((data_length & 3) || (data_offset & 3))
			fmt::throw_exception("Metal shuffle ranges must be word aligned");
		struct alignas(16) parameters { u32 word_count; std::array<u32, 3> reserved{}; };
		const parameters params{data_length / 4};
		const compute_buffer_range range{&data, data_offset, data_length, argument_access::read_write};
		dispatch(command, std::span(&range, 1), &params, sizeof(params), {params.word_count, 1, 1});
	}

	cs_shuffle_16::cs_shuffle_16()
		: cs_shuffle_base(compute_shuffle_operation::byte_swap_u16)
	{
	}

	cs_shuffle_32::cs_shuffle_32()
		: cs_shuffle_base(compute_shuffle_operation::byte_swap_u32)
	{
	}

	cs_shuffle_32_16::cs_shuffle_32_16()
		: cs_shuffle_base(compute_shuffle_operation::byte_swap_u16_u32)
	{
	}

	cs_shuffle_d24x8_f32::cs_shuffle_d24x8_f32()
		: cs_shuffle_base(compute_shuffle_operation::depth24_to_float32)
	{
	}

	cs_shuffle_se_f32_d24x8::cs_shuffle_se_f32_d24x8()
		: cs_shuffle_base(compute_shuffle_operation::float32_to_depth24_swapped)
	{
	}

	cs_shuffle_se_d24x8::cs_shuffle_se_d24x8()
		: cs_shuffle_base(compute_shuffle_operation::depth24_byte_swap)
	{
	}

	cs_interleave_task::cs_interleave_task(depth_stencil_operation operation, bool swap_bytes, bool depth_float)
		: compute_task(interleave_specification(operation, swap_bytes, depth_float))
	{
	}

	void cs_interleave_task::run(command_buffer& command, const buffer& data, u32 data_offset,
		u32 data_length, u32 depth_offset, u32 stencil_offset)
	{
		if (!data_length) return;
		if ((data_offset | data_length | depth_offset | stencil_offset) & 3)
			fmt::throw_exception("Metal depth-stencil ranges must be word aligned");
		const u32 base = std::min({data_offset, depth_offset, stencil_offset});
		const u32 end = std::max({checked_add(data_offset, data_length, "depth-stencil"),
			checked_add(depth_offset, data_length, "depth-stencil"),
			checked_add(stencil_offset, data_length / 4, "depth-stencil")});
		struct alignas(16) parameters
		{
			u32 word_count;
			u32 depth_offset;
			u32 stencil_offset;
			u32 reserved = 0;
		};
		const parameters params{data_length / 4, (depth_offset - base) / 4, (stencil_offset - base) / 4};
		const compute_buffer_range range{&data, base, end - base, argument_access::read_write};
		dispatch(command, std::span(&range, 1), &params, sizeof(params), {params.word_count, 1, 1});
	}

	cs_scatter_d24x8::cs_scatter_d24x8()
		: cs_interleave_task(depth_stencil_operation::scatter_depth24, false, false)
	{
	}

	cs_fconvert_base::cs_fconvert_base(u32 source_element_size, u32 destination_element_size,
		bool swap_source, bool swap_destination)
		: compute_task(format_specification(source_element_size, destination_element_size,
			swap_source, swap_destination))
	{
	}

	void cs_fconvert_base::run(command_buffer& command, const buffer& data, u32 source_offset,
		u32 source_length, u32 destination_offset)
	{
		if (!source_length) return;
		if ((source_offset | source_length | destination_offset) & 3)
			fmt::throw_exception("Metal format-conversion ranges must be word aligned");
		const auto& spec = specification();
		if (spec.source_element_size == 4 && (source_length & 7))
			fmt::throw_exception("Metal 32-to-16 conversion requires pairs of source elements");
		const u32 destination_length = checked_multiply(source_length / spec.source_element_size,
			spec.destination_element_size, "format conversion");
		const u32 base = std::min(source_offset, destination_offset);
		const u32 end = std::max(checked_add(source_offset, source_length, "format conversion"),
			checked_add(destination_offset, destination_length, "format conversion"));
		struct alignas(16) parameters
		{
			u32 invocation_count;
			u32 source_offset;
			u32 destination_offset;
			u32 reserved = 0;
		};
		const parameters params{spec.source_element_size == 4 ? source_length / 8 : source_length / 4,
			(source_offset - base) / 4, (destination_offset - base) / 4};
		const compute_buffer_range range{&data, base, end - base, argument_access::read_write};
		dispatch(command, std::span(&range, 1), &params, sizeof(params), {params.invocation_count, 1, 1});
	}

	cs_deswizzle_base::cs_deswizzle_base(u32 block_size, u32 base_element_size, bool swap_bytes)
		: compute_task(deswizzle_specification(block_size, base_element_size, swap_bytes))
	{
	}

	void cs_deswizzle_base::run(command_buffer& command, const buffer& destination, u32 destination_offset,
		const buffer& source, u32 source_offset, u32 data_length, u32 width, u32 height,
		u32 depth, u32 mipmaps)
	{
		if (!data_length) return;
		if (!width || !height || !depth || !mipmaps || (source_offset | destination_offset | data_length) & 3)
			fmt::throw_exception("Invalid Metal 3D deswizzle dimensions or ranges");
		struct alignas(16) parameters
		{
			u32 width, height, depth, log_width;
			u32 log_height, log_depth, mipmap_count, reserved = 0;
		};
		const parameters params{width, height, depth, static_cast<u32>(std::bit_width(width - 1)),
			static_cast<u32>(std::bit_width(height - 1)),
			static_cast<u32>(std::bit_width(depth - 1)), mipmaps};
		const std::array ranges{
			compute_buffer_range{&source, source_offset, data_length, argument_access::read},
			compute_buffer_range{&destination, destination_offset, data_length, argument_access::write}};
		const u32 block_size = specification().block_size;
		const u32 invocations = block_size < 4 ? divide_up(data_length, 4) : divide_up(data_length, block_size);
		dispatch(command, ranges, &params, sizeof(params), {invocations, 1, 1});
	}

	cs_aggregator::cs_aggregator()
		: compute_task(simple_specification(compute_kernel_kind::aggregate))
	{
	}

	void cs_aggregator::run(command_buffer& command, const buffer& destination, const buffer& source,
		u32 num_words, u32 destination_offset, u32 source_offset)
	{
		if (!num_words) return;
		if ((destination_offset | source_offset) & 3)
			fmt::throw_exception("Metal aggregation ranges must be word aligned");
		struct alignas(16) parameters { u32 word_count; std::array<u32, 3> reserved{}; };
		const parameters params{num_words};
		const std::array ranges{
			compute_buffer_range{&source, source_offset, checked_multiply(num_words, 4, "aggregation"), argument_access::read},
			compute_buffer_range{&destination, destination_offset, 4, argument_access::read_write}};
		dispatch(command, ranges, &params, sizeof(params), {num_words, 1, 1});
	}

	void rsx_detiler_config::validate() const
	{
		if (!destination || !source || !*destination || !*source || !tile_size || !tile_pitch ||
			!image_width || !image_height || !image_pitch ||
			(image_bytes_per_pixel != 1 && image_bytes_per_pixel != 2 && image_bytes_per_pixel != 4 &&
			image_bytes_per_pixel != 8 && image_bytes_per_pixel != 16) ||
			tile_base_offset >= tile_size || tile_rw_offset > tile_base_offset ||
			(tile_pitch & 0xff) || (source_offset & 3) || (destination_offset & 3))
		{
			fmt::throw_exception("Invalid Metal tiled-memory copy configuration");
		}
	}

	cs_tile_memcpy_base::cs_tile_memcpy_base(rsx_detiler_operation operation)
		: compute_task(tile_specification(operation))
	{
	}

	void cs_tile_memcpy_base::run(command_buffer& command, const rsx_detiler_config& configuration)
	{
		configuration.validate();
		const u32 available_tile_bytes = configuration.tile_size - configuration.tile_base_offset;
		const u32 tile_height = std::min<u32>((configuration.image_height + 63u) & ~63u,
			available_tile_bytes / configuration.tile_pitch);
		if (!tile_height) fmt::throw_exception("Metal tiled-memory copy has no addressable rows");

		auto prime_factor = [](u32 pitch) -> std::pair<u32, u32>
		{
			const u32 base = pitch >> 8;
			if (!base) fmt::throw_exception("Metal tiled-memory pitch is smaller than one tile");
			if ((pitch & (pitch - 1)) == 0) return {1, base};
			for (const u32 prime : {3u, 5u, 7u, 11u, 13u})
				if (base % prime == 0) return {prime, base / prime};
			fmt::throw_exception("Metal tiled-memory pitch 0x%x has no supported prime factor", pitch);
		};

		const auto [prime, factor] = prime_factor(configuration.tile_pitch);
		struct alignas(16) parameters
		{
			u32 prime, factor, tiles_per_row, tile_base_address;
			u32 tile_size, tile_address_offset, tile_rw_offset, tile_pitch;
			u32 tile_bank, image_width, image_height, image_pitch;
			u32 image_bytes_per_pixel;
			std::array<u32, 3> reserved{};
		};
		const parameters params{prime, factor, prime * factor, configuration.tile_base_address,
			configuration.tile_size, configuration.tile_base_offset, configuration.tile_rw_offset,
			configuration.tile_pitch, configuration.bank, configuration.image_width,
			specification().detiler == rsx_detiler_operation::decode ? tile_height : configuration.image_height,
			configuration.image_pitch, configuration.image_bytes_per_pixel};

		const u32 tiled_length = aligned_to_word(checked_multiply(tile_height,
			configuration.tile_pitch, "tiled-memory range"));
		const u32 linear_length = aligned_to_word(checked_multiply(configuration.image_height,
			configuration.image_pitch, "linear-image range"));
		const bool decode = specification().detiler == rsx_detiler_operation::decode;
		const std::array ranges{
			compute_buffer_range{decode ? configuration.source : configuration.destination,
				decode ? configuration.source_offset : configuration.destination_offset,
				tiled_length, decode ? argument_access::read : argument_access::write},
			compute_buffer_range{decode ? configuration.destination : configuration.source,
				decode ? configuration.destination_offset : configuration.source_offset,
				linear_length, decode ? argument_access::write : argument_access::read}};
		const u32 pixels_per_invocation = configuration.image_bytes_per_pixel < 4 ?
			4 / configuration.image_bytes_per_pixel : 1;
		dispatch(command, ranges, &params, sizeof(params),
			{divide_up(configuration.image_width, pixels_per_invocation), params.image_height, 1});
	}

	compute_task_manager::~compute_task_manager()
	{
		reset();
	}

	void compute_task_manager::initialize(render_device& device, MTLPipelineCompiler& compiler)
	{
		if (!device || !compiler || &compiler.owner() != &device)
			fmt::throw_exception("Metal compute-task manager requires a compiler for its render device");
		std::lock_guard lock(m_mutex);
		if (!m_tasks.empty() && (m_device != &device || m_compiler != &compiler))
			fmt::throw_exception("Metal compute-task manager cannot change devices while tasks are live");
		m_device = &device;
		m_compiler = &compiler;
	}

	void compute_task_manager::reset()
	{
		std::lock_guard lock(m_mutex);
		m_tasks.clear();
		m_device = nullptr;
		m_compiler = nullptr;
	}

	compute_task_manager::operator bool() const
	{
		std::lock_guard lock(m_mutex);
		return m_device && m_compiler && *m_compiler;
	}

	usz compute_task_manager::size() const
	{
		std::lock_guard lock(m_mutex);
		return m_tasks.size();
	}

	compute_task_manager& compute_tasks()
	{
		static compute_task_manager manager;
		return manager;
	}

	void initialize_compute_tasks(render_device& device, MTLPipelineCompiler& compiler)
	{
		compute_tasks().initialize(device, compiler);
	}

	void reset_compute_tasks()
	{
		compute_tasks().reset();
	}
}
