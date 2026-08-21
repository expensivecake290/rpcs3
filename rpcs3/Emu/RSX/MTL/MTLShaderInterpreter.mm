#include "stdafx.h"
#include "MTLShaderInterpreter.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

namespace mtl
{
	using namespace program_common::interpreter;

	namespace
	{
		constexpr u64 key_seed = 0x9e3779b97f4a7c15ull;
		constexpr u32 compiler_opt_point_size = 1u << 31;
		constexpr u32 compiler_opt_framebuffer_fetch_shift = 16;
		constexpr u32 compiler_opt_framebuffer_fetch_mask = 0xfu << compiler_opt_framebuffer_fetch_shift;

		u32 framebuffer_fetch_options(const graphics_pipeline_configuration& configuration,
			bool framebuffer_fetch_supported)
		{
			if (!framebuffer_fetch_supported) return 0;
			u32 result = 0;
			for (u32 index = 0; index < configuration.state.render.color_attachment_count; ++index)
			{
				if (configuration.state.render.color_attachments[index].pixel_format)
					result |= 1u << (compiler_opt_framebuffer_fetch_shift + index);
			}
			return result;
		}

		void combine(u64& hash, u64 value)
		{
			hash ^= value + key_seed + (hash << 6) + (hash >> 2);
		}

		std::string vertex_interpreter_source(u32 options)
		{
			std::string source = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint VEC_NOP=0, VEC_MOV=1, VEC_MUL=2, VEC_ADD=3, VEC_MAD=4, VEC_DP3=5,
 VEC_DPH=6, VEC_DP4=7, VEC_DST=8, VEC_MIN=9, VEC_MAX=10, VEC_SLT=11,
 VEC_SGE=12, VEC_ARL=13, VEC_FRC=14, VEC_FLR=15, VEC_SEQ=16, VEC_SFL=17,
 VEC_SGT=18, VEC_SLE=19, VEC_SNE=20, VEC_STR=21, VEC_SSG=22, VEC_TXL=25;
constant uint SCA_NOP=0, SCA_MOV=1, SCA_RCP=2, SCA_RCC=3, SCA_RSQ=4, SCA_EXP=5,
 SCA_LOG=6, SCA_LIT=7, SCA_BRA=8, SCA_BRI=9, SCA_CAL=10, SCA_CLI=11,
 SCA_RET=12, SCA_LG2=13, SCA_EX2=14, SCA_SIN=15, SCA_COS=16,
 SCA_BRB=17, SCA_CLB=18;
constant uint REG_TEMP=1, REG_INPUT=2, REG_CONSTANT=3;

inline uint rsx_bits(uint value, uint offset, uint count)
{
	return (value >> offset) & ((1u << count) - 1u);
}
inline bool rsx_bit(uint value, uint offset) { return ((value >> offset) & 1u) != 0; }
inline float4 rsx_swizzle(float4 value, uint code)
{
	return float4(value[rsx_bits(code,6,2)], value[rsx_bits(code,4,2)],
		value[rsx_bits(code,2,2)], value[rsx_bits(code,0,2)]);
}
inline bool4 rsx_condition(float4 value, uint mode)
{
	switch (mode)
	{
	case 7: return bool4(true);
	case 6: return value >= 0.0;
	case 3: return value <= 0.0;
	case 5: return value != 0.0;
	case 4: return value > 0.0;
	case 1: return value < 0.0;
	case 2: return value == 0.0;
	default: return bool4(false);
	}
}
inline float4 rsx_lit(float4 value)
{
	return float4(1.0, max(value.x, 0.0), value.x > 0.0 ?
		pow(max(value.y, 0.0), clamp(value.w, -128.0, 128.0)) : 0.0, 1.0);
}

struct vertex_context
{
	float4x4 scale_offset_matrix;
	uint user_clip_configuration_bits;
	uint transform_branch_bits;
	float point_size;
	float z_near;
	float z_far;
	float line_width;
	float viewport_width;
	float viewport_height;
};
struct line_mapping
{
	uint vertex_id;
	uint other_vertex_id;
	float side;
	uint reserved;
};
struct draw_parameters
{
	uint vertex_base_index;
	uint vertex_index_offset;
	uint draw_id;
	uint transform_constants_offset;
	uint vertex_context_offset;
	uint fragment_constants_offset;
	uint fragment_context_offset;
	uint fragment_texture_base_index;
	uint stipple_pattern_offset;
	uint reserved;
	uint2 attribute_data[16];
};
struct attribute_description
{
	uint type;
	uint component_count;
	uint starting_offset;
	uint stride;
	uint frequency;
	bool swap_bytes;
	bool is_volatile;
	bool modulo;
};
struct vertex_result
{
	float4 position [[position]];
	float point_size [[point_size]];
	float4 user0 [[user(locn0)]]; float4 user1 [[user(locn1)]];
	float4 user2 [[user(locn2)]]; float4 user3 [[user(locn3)]];
	float4 user4 [[user(locn4)]]; float4 user5 [[user(locn5)]];
	float4 user6 [[user(locn6)]]; float4 user7 [[user(locn7)]];
	float4 user8 [[user(locn8)]]; float4 user9 [[user(locn9)]];
	float4 user10 [[user(locn10)]]; float4 user11 [[user(locn11)]];
	float4 user12 [[user(locn12)]]; float4 user13 [[user(locn13)]];
	float4 user14 [[user(locn14)]]; float4 user15 [[user(locn15)]];
};
struct vertex_machine
{
	float4 temp[32];
	float4 dest[16];
	int4 address[2];
	float4 condition[2];
};
struct sampler_state
{
	float4 border_color;float lod_bias;uint emulation_flags;uint address_modes;uint border_metadata;
};
inline float vertex_sampler_coord(float c,constant sampler_state& state,uint size,uint axis)
{if(state.emulation_flags&(1u<<5))c/=float(max(size,1u));uint mode=(state.address_modes>>(axis*4u))&15u;
if(mode==6u||mode==7u)c=abs(c);if(mode==4u||mode==7u){float h=0.5/float(max(size,1u));c=clamp(c,-h,1.0+h);}return c;}
inline float4 sample_vertex_texture(texture2d<float> t,sampler s,float2 c,float lod,constant sampler_state& state)
{uint2 size=uint2(t.get_width(),t.get_height());c=float2(vertex_sampler_coord(c.x,state,size.x,0),vertex_sampler_coord(c.y,state,size.y,1));
if((state.emulation_flags&1u)&&any(c<0.0||c>1.0))return state.border_color;return t.sample(s,c,level(lod));}
inline float4 apply_zclip(float4 position,float near_plane,float far_plane)
{if(position.w==0.0)return position;float real_near=min(far_plane,near_plane),real_far=max(far_plane,near_plane);
float depth_range=real_far-real_near;float inverse_range=depth_range>0.000001?1.0/(depth_range*position.w):0.0;
float actual_depth=(position.z-real_near*position.w)*inverse_range;float nearest_depth=floor(actual_depth+0.5);
float epsilon=(inverse_range*position.w)/16777215.0;float depth=abs(actual_depth-nearest_depth)<epsilon?nearest_depth:actual_depth;
return float4(position.xy,depth*position.w,position.w);}

inline attribute_description fetch_attribute_description(
	constant draw_parameters& draw, uint location)
{
	const uint2 packed = draw.attribute_data[location];
	attribute_description result;
	result.stride = rsx_bits(packed.x, 0u, 8u);
	result.frequency = rsx_bits(packed.x, 8u, 16u);
	result.type = rsx_bits(packed.x, 24u, 3u);
	result.component_count = rsx_bits(packed.x, 27u, 3u);
	result.starting_offset = rsx_bits(packed.y, 0u, 29u);
	result.swap_bytes = rsx_bit(packed.y, 29u);
	result.is_volatile = rsx_bit(packed.y, 30u);
	result.modulo = rsx_bit(packed.y, 31u);
	return result;
}

inline uint load_vertex_component(device const uchar* stream, uint offset,
	uint size, bool swap_bytes)
{
	const uint x = stream[offset];
	if (size == 1u) return x;
	const uint y = stream[offset + 1u];
	if (size == 2u) return swap_bytes ? (y | (x << 8u)) : (x | (y << 8u));
	const uint z = stream[offset + 2u];
	const uint w = stream[offset + 3u];
	return swap_bytes ? (w | (z << 8u) | (y << 16u) | (x << 24u)) :
		(x | (y << 8u) | (z << 16u) | (w << 24u));
}

inline float sign_extend_16(uint value)
{
	return value < 0x8000u ? float(value) : float(int(value) - 65536);
}

inline float4 fetch_attribute(attribute_description description, int vertex_index,
	device const uchar* stream)
{
	uint element_size = 1u;
	if (description.type == 1u || description.type == 3u || description.type == 5u)
		element_size = 2u;
	else if (description.type == 2u || description.type == 6u)
		element_size = 4u;
	uint4 bits = uint4(0u);
	uint address = uint(vertex_index * int(description.stride) + int(description.starting_offset));
	for (uint component = 0u; component < min(description.component_count, 4u); ++component)
	{
		bits[component] = load_vertex_component(stream, address, element_size,
			description.swap_bytes);
		address += element_size;
	}
	float scale = 1.0;
	float4 result = float4(0.0);
	if (description.type == 1u || description.type == 5u)
	{
		result = float4(sign_extend_16(bits.x), sign_extend_16(bits.y),
			sign_extend_16(bits.z), sign_extend_16(bits.w));
		if (description.type == 1u)
		{
			result = fma(float4(0.5), float4(1.0), result);
			scale = 32767.5;
		}
	}
	else if (description.type == 2u)
		result = as_type<float4>(bits);
	else if (description.type == 3u)
	{
		result.xy = float2(as_type<half2>(bits.x | (bits.y << 16u)));
		result.zw = float2(as_type<half2>(bits.z | (bits.w << 16u)));
	}
	else if (description.type == 4u || description.type == 7u)
	{
		result = float4(bits);
		scale = description.type == 4u ? 255.0 : 1.0;
	}
	else if (description.type == 6u)
	{
		const uint4 packed = uint4(rsx_bits(bits.x, 0u, 11u),
			rsx_bits(bits.x, 11u, 11u), rsx_bits(bits.x, 22u, 10u), 32767u);
		result = float4(sign_extend_16(packed.x << 5u), sign_extend_16(packed.y << 5u),
			sign_extend_16(packed.z << 6u), float(packed.w));
		scale = 32767.0;
	}
	if (description.component_count < 4u) result.w = scale;
	return result / scale;
}

inline float4 read_vertex_input(uint location, uint vertex_id,
	constant draw_parameters& draw, device const uchar* persistent_stream,
	device const uchar* volatile_stream)
{
	const attribute_description description = fetch_attribute_description(draw, location);
	int index = 0;
	if (description.frequency != 0u)
	{
		index = description.modulo ?
			int((vertex_id + draw.vertex_index_offset) % description.frequency) :
			(int(vertex_id) - int(draw.vertex_base_index)) / int(description.frequency);
	}
	return description.is_volatile ? fetch_attribute(description, index, volatile_stream) :
		fetch_attribute(description, index, persistent_stream);
}

inline float4 read_vertex_source(thread vertex_machine& machine, uint index, uint4 instruction,
	uint input_source, uint constant_source, bool indexed_constant, uint address_select,
	uint address_swizzle, uint vertex_index, constant draw_parameters& draw,
	device const uchar* persistent_stream, device const uchar* volatile_stream,
	constant float4* constants)
{
	uint encoded = index == 0 ? (rsx_bits(instruction.y,0,8) << 9) | rsx_bits(instruction.z,23,9) :
		index == 1 ? rsx_bits(instruction.z,6,17) :
		(rsx_bits(instruction.z,0,6) << 11) | rsx_bits(instruction.w,21,11);
	float4 value = 0.0;
	switch (rsx_bits(encoded,0,2))
	{
	case REG_TEMP: value = machine.temp[rsx_bits(encoded,2,6) & 31u]; break;
	case REG_INPUT: value = read_vertex_input(input_source & 15u, vertex_index, draw,
		persistent_stream, volatile_stream); break;
	case REG_CONSTANT:
	{
		int location = int(constant_source);
		if (indexed_constant) location += machine.address[address_select][address_swizzle];
		value = constants[draw.transform_constants_offset + uint(clamp(location, 0, 467))];
		break;
	}
	}
	if (rsx_bits(encoded,8,8) != 0x1b) value = rsx_swizzle(value, rsx_bits(encoded,8,8));
	if (rsx_bit(instruction.x,21u + index)) value = abs(value);
	if (rsx_bit(encoded,16)) value = -value;
	return value;
}

inline void masked_write(thread float4& destination, float4 value, bool4 mask)
{
	destination = select(destination, value, mask);
}

inline vertex_machine run_vertex_program(
	uint vertex_id,
	device const uchar* persistent_stream,
	device const uchar* volatile_stream,
	constant draw_parameters* draw_parameters_buffer,
	constant vertex_context* vertex_contexts,
	constant float4* constants,
	constant sampler_state* sampler_states,
	device const uint4* program,
	array<texture2d<float>, 4> vertex_textures,
	array<sampler, 4> vertex_samplers)
{
	constant draw_parameters& draw = draw_parameters_buffer[0];
	constant vertex_context& context = vertex_contexts[draw.vertex_context_offset];
	vertex_machine machine;
	for (uint i=0; i<32; ++i) machine.temp[i]=0.0;
	for (uint i=0; i<16; ++i) machine.dest[i]=float4(0.0,0.0,0.0,1.0);
	machine.address[0]=0; machine.address[1]=0; machine.condition[0]=0.0; machine.condition[1]=0.0;
	uint4 header = program[0];
	int ip = int(header.y) - int(header.x);
	int callstack[8]; int stack_pointer=0;
	for (uint watchdog=0; watchdog<4096 && ip>=0 && ip<544; ++watchdog)
	{
		uint4 instruction = program[uint(ip)+1u]; ++ip;
		uint address_swizzle=rsx_bits(instruction.x,0,2), condition_swizzle=rsx_bits(instruction.x,2,8);
		uint condition_mode=rsx_bits(instruction.x,10,3), destination_temp=rsx_bits(instruction.x,15,6);
		uint address_select=rsx_bits(instruction.x,24,1), condition_select=rsx_bits(instruction.x,25,1);
		bool condition_test=rsx_bit(instruction.x,13), saturate=rsx_bit(instruction.x,26);
		bool vector_result=rsx_bit(instruction.x,30), indexed_constant=rsx_bit(instruction.w,1);
		uint input_source=rsx_bits(instruction.y,8,4), constant_source=rsx_bits(instruction.y,12,10);
		uint vector_opcode=rsx_bits(instruction.y,22,5), scalar_opcode=rsx_bits(instruction.y,27,5);
		uint output_index=rsx_bits(instruction.w,2,5), scalar_temp=rsx_bits(instruction.w,7,6);
		bool4 vector_mask=bool4(rsx_bit(instruction.w,16),rsx_bit(instruction.w,15),
			rsx_bit(instruction.w,14),rsx_bit(instruction.w,13));
		bool4 scalar_mask=bool4(rsx_bit(instruction.w,20),rsx_bit(instruction.w,19),
			rsx_bit(instruction.w,18),rsx_bit(instruction.w,17));
		float4 condition = rsx_swizzle(machine.condition[condition_select], condition_swizzle);
		if (condition_test && condition_mode==0) { vector_opcode=VEC_NOP; scalar_opcode=SCA_NOP; }
		auto source = [&](uint index) { return read_vertex_source(machine,index,instruction,input_source,
			constant_source,indexed_constant,address_select,address_swizzle,vertex_id,draw,
			persistent_stream,volatile_stream,constants); };
		if (vector_opcode==VEC_ARL) machine.address[destination_temp&1u]=int4(floor(source(0)));
		else if (vector_opcode!=VEC_NOP)
		{
			float4 value=source(0), second;
			switch (vector_opcode)
			{
			case VEC_MOV: break; case VEC_MUL: value*=source(1); break;
			case VEC_ADD: value+=source(2); break; case VEC_MAD: value=fma(value,source(1),source(2)); break;
			case VEC_DP3: value=dot(value.xyz,source(1).xyz); break;
			case VEC_DPH: value=dot(float4(value.xyz,1.0),source(1)); break;
			case VEC_DP4: value=dot(value,source(1)); break;
			case VEC_DST: second=source(1); value=float4(1.0,value.y*second.y,value.z,second.w); break;
			case VEC_MIN: value=min(value,source(1)); break; case VEC_MAX: value=max(value,source(1)); break;
			case VEC_SLT: value=float4(value<source(1)); break; case VEC_SGE: value=float4(value>=source(1)); break;
			case VEC_FRC: value=fract(value); break; case VEC_FLR: value=floor(value); break;
			case VEC_SEQ: value=float4(value==source(1)); break; case VEC_SFL: value=0.0; break;
			case VEC_SGT: value=float4(value>source(1)); break; case VEC_SLE: value=float4(value<=source(1)); break;
			case VEC_SNE: value=float4(value!=source(1)); break; case VEC_STR: value=1.0; break;
			case VEC_SSG: value=sign(value); break;
			case VEC_TXL:
			{ uint unit=rsx_bits(instruction.z,8,2); value=sample_vertex_texture(vertex_textures[unit],vertex_samplers[unit],value.xy,value.w,sampler_states[unit]); break; }
			default: value=0.0; break;
			}
			if (saturate) value=clamp(value,0.0,1.0);
			bool4 mask=vector_mask;
			if (condition_test) mask=mask & rsx_condition(condition,condition_mode);
			if (destination_temp!=0x3f) masked_write(machine.temp[destination_temp&31u],value,mask);
			if (vector_result && output_index<16) masked_write(machine.dest[output_index],value,mask);
			if (destination_temp==0x3f && !vector_result) masked_write(machine.condition[condition_select],value,mask);
		}
		if (scalar_opcode!=SCA_NOP)
		{
			float4 scalar_value=source(2); float value=scalar_value.x;
			bool branch=any(rsx_condition(condition,condition_mode));
			uint branch_address=(rsx_bits(instruction.x,23,1)<<9)|(rsx_bits(instruction.z,0,6)<<3)|rsx_bits(instruction.w,29,3);
			switch (scalar_opcode)
			{
			case SCA_MOV: break; case SCA_RCP: value=1.0/value; break;
			case SCA_RCC: value=clamp(1.0/value,5.42101e-20f,1.884467e19f); break;
			case SCA_RSQ: value=rsqrt(abs(value)); break; case SCA_EXP: value=exp(value); break;
			case SCA_LOG: value=log(abs(value)); break; case SCA_LG2: value=log2(abs(value)); break;
			case SCA_EX2: value=exp2(value); break; case SCA_SIN: value=sin(value); break; case SCA_COS: value=cos(value); break;
			case SCA_LIT: scalar_value=rsx_lit(source(2)); break;
			case SCA_BRA: if(branch) ip=machine.address[address_select].x-int(header.x); continue;
			case SCA_BRI: if(branch) ip=int(branch_address); continue;
			case SCA_CAL: if(branch&&stack_pointer<8){callstack[stack_pointer++]=ip;ip=int(branch_address);} continue;
			case SCA_CLI: continue;
			case SCA_RET: if(branch){if(stack_pointer==0) watchdog=4096;else ip=callstack[--stack_pointer];} continue;
			case SCA_BRB: case SCA_CLB:
			{ uint bit=1u<<rsx_bits(instruction.w,23,5); bool expected=rsx_bit(instruction.w,28);
			  if((((context.transform_branch_bits&bit)!=0)==expected)){if(scalar_opcode==SCA_CLB&&stack_pointer<8)callstack[stack_pointer++]=ip;ip=int(branch_address);} continue; }
			default: scalar_value=0.0; break;
			}
			if (scalar_opcode!=SCA_LIT) scalar_value=value;
			if (saturate) scalar_value=clamp(scalar_value,0.0,1.0);
			if (scalar_temp!=0x3f) masked_write(machine.temp[scalar_temp&31u],scalar_value,scalar_mask);
			else if (!vector_result && output_index<16) masked_write(machine.dest[output_index],scalar_value,scalar_mask);
			else masked_write(machine.condition[condition_select],scalar_value,scalar_mask);
		}
		if (rsx_bit(instruction.w,0)) break;
	}
	return machine;
}

vertex vertex_result rsx_interpreter_vertex(
	uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
	device const uchar* persistent_stream [[buffer(0)]],
	device const uchar* volatile_stream [[buffer(1)]],
	constant draw_parameters* draw_parameters_buffer [[buffer(2)]],
	constant vertex_context* vertex_contexts [[buffer(3)]],
	device const uint* conditional_render_predicate [[buffer(4)]],
	constant float4* constants [[buffer(5)]],
	constant sampler_state* sampler_states [[buffer(8)]],
	device const uint4* program [[buffer(9)]],
	device const line_mapping* line_mappings [[buffer(10)]],
	array<texture2d<float>, 4> vertex_textures [[texture(0)]],
	array<sampler, 4> vertex_samplers [[sampler(0)]])
{
	constant draw_parameters& draw=draw_parameters_buffer[0];
	constant vertex_context& context=vertex_contexts[draw.vertex_context_offset];
	if(conditional_render_predicate[0]==0u)
	{vertex_result discarded={};discarded.position=float4(0.0,0.0,0.0,-1.0);return discarded;}
	line_mapping mapping={vertex_id,vertex_id,0.0,0u};
	if(draw.reserved&1u)mapping=line_mappings[vertex_id];
	vertex_machine machine=run_vertex_program(mapping.vertex_id,persistent_stream,volatile_stream,
		draw_parameters_buffer,vertex_contexts,constants,sampler_states,program,vertex_textures,vertex_samplers);
	vertex_machine other_machine=machine;
	if(draw.reserved&1u)other_machine=run_vertex_program(mapping.other_vertex_id,persistent_stream,
		volatile_stream,draw_parameters_buffer,vertex_contexts,constants,sampler_states,program,
		vertex_textures,vertex_samplers);
	vertex_result output;
	output.position=apply_zclip(machine.dest[0]*context.scale_offset_matrix,context.z_near,context.z_far);
	if(draw.reserved&1u)
	{
		float4 other_position=apply_zclip(other_machine.dest[0]*context.scale_offset_matrix,context.z_near,context.z_far);
		float2 viewport_size=max(float2(context.viewport_width,context.viewport_height),1.0);
		float first_w=abs(output.position.w)>0.000001?output.position.w:copysign(0.000001,output.position.w);
		float other_w=abs(other_position.w)>0.000001?other_position.w:copysign(0.000001,other_position.w);
		float2 delta=(other_position.xy/other_w-output.position.xy/first_w)*viewport_size;
		float delta_length=length(delta);float2 normal=delta_length>0.000001?float2(-delta.y,delta.x)/delta_length:float2(0.0,1.0);
		output.position.xy+=normal*mapping.side*context.line_width*output.position.w/viewport_size;
	}
	output.point_size=(program[0].z&(1u<<5))?machine.dest[6].x:context.point_size;
	output.user0=machine.dest[0]; output.user1=machine.dest[1]; output.user2=machine.dest[2]; output.user3=machine.dest[3];
	output.user4=machine.dest[4]; output.user5=machine.dest[5]; output.user6=machine.dest[6]; output.user7=machine.dest[7];
	output.user8=machine.dest[8]; output.user9=machine.dest[9]; output.user10=machine.dest[10]; output.user11=machine.dest[11];
	output.user12=machine.dest[12]; output.user13=machine.dest[13]; output.user14=machine.dest[14]; output.user15=machine.dest[15];
	(void)instance_id;
	return output;
}
)MSL";
			if (!(options & compiler_opt_point_size))
			{
				source.erase(source.find("\tfloat point_size [[point_size]];\n"),
					std::string_view("\tfloat point_size [[point_size]];\n").size());
				source.erase(source.find("\toutput.point_size=(program[0].z&(1u<<5))?machine.dest[6].x:context.point_size;\n"),
					std::string_view("\toutput.point_size=(program[0].z&(1u<<5))?machine.dest[6].x:context.point_size;\n").size());
			}
			if (!(options & COMPILER_OPT_ENABLE_VTX_TEXTURES))
			{
				source.replace(source.find("case VEC_TXL:"), source.find("default: value=0.0;", source.find("case VEC_TXL:")) - source.find("case VEC_TXL:"), "");
			}
			return source;
		}

		std::string fragment_interpreter_source(u32 options)
		{
			std::string source = R"MSL(
#include <metal_stdlib>
using namespace metal;

inline uint rsx_bits(uint value,uint offset,uint count){return(value>>offset)&((1u<<count)-1u);}
inline bool rsx_bit(uint value,uint offset){return((value>>offset)&1u)!=0;}
inline uint rsx_swap16(uint value){return((value<<8)&0xff00ff00u)|((value>>8)&0x00ff00ffu);}
inline uint4 rsx_swap16(uint4 value){return((value<<8)&0xff00ff00u)|((value>>8)&0x00ff00ffu);}
inline float4 rsx_swizzle(float4 value,uint code)
{return float4(value[rsx_bits(code,0,2)],value[rsx_bits(code,2,2)],value[rsx_bits(code,4,2)],value[rsx_bits(code,6,2)]);}
inline bool4 rsx_cond(float4 value,uint mode)
{switch(mode){case 7:return true;case 6:return value>=0.0;case 3:return value<=0.0;case 5:return value!=0.0;
case 4:return value>0.0;case 1:return value<0.0;case 2:return value==0.0;default:return false;}}
inline float4 rsx_lit(float4 v){return float4(1.0,max(v.x,0.0),v.x>0.0?pow(max(v.y,0.0),clamp(v.w,-128.0,128.0)):0.0,1.0);}
inline float4 rsx_lif(float4 v){return float4(1.0,v.x,max(v.y,0.0),v.x>0.0?pow(max(v.y,0.0),clamp(v.w,-128.0,128.0)):0.0);}

struct fragment_input
{
	float4 position [[position]]; bool front_facing [[front_facing]];
	float4 user0 [[user(locn0)]]; float4 user1 [[user(locn1)]];
	float4 user2 [[user(locn2)]]; float4 user3 [[user(locn3)]];
	float4 user4 [[user(locn4)]]; float4 user5 [[user(locn5)]];
	float4 user6 [[user(locn6)]]; float4 user7 [[user(locn7)]];
	float4 user8 [[user(locn8)]]; float4 user9 [[user(locn9)]];
	float4 user10 [[user(locn10)]]; float4 user11 [[user(locn11)]];
	float4 user12 [[user(locn12)]]; float4 user13 [[user(locn13)]];
	float4 user14 [[user(locn14)]]; float4 user15 [[user(locn15)]];
};
struct fragment_result
{
	float4 color0 [[color(0)]]; float4 color1 [[color(1)]];
	float4 color2 [[color(2)]]; float4 color3 [[color(3)]];
	uint sample_mask [[sample_mask]];
)MSL";
			if (options & COMPILER_OPT_ENABLE_DEPTH_EXPORT)
				source += "\tfloat depth [[depth(any)]];\n";
			source += R"MSL(};
struct fragment_state
{
	float fog_param0;float fog_param1;uint rop_control;float alpha_ref;
	uint fog_mode;float wpos_scale;float2 wpos_bias;
	uint logic_operation;uint rop_emulation;uint logic_types[4];uint2 logic_padding;uint4 logic_scales[4];
	float4 blend_constants;uint blend_equations;uint blend_factors_alpha;uint blend_factors_rgb;uint programmable_blend_mask;
	uint sample_mask;uint3 sample_padding;
	uint polygon_modes;float polygon_line_width;float polygon_point_size;uint polygon_padding;
};
struct raster_context{float fog_param0;float fog_param1;float wpos_scale;float wpos_bias;};
struct texture_parameter
{float scale_x,scale_y,scale_z,bias_x,bias_y,bias_z,clamp_min_x,clamp_min_y,clamp_max_x,clamp_max_y;uint remap;uint flags;};
struct sampler_state{float4 border_color;float lod_bias;uint emulation_flags;uint address_modes;uint border_metadata;};
inline bool polygon_fragment_covered(float3 barycentric,bool front_facing,constant fragment_state& state)
{if(!(state.polygon_modes&(1u<<31u)))return true;uint mode=front_facing?(state.polygon_modes&3u):((state.polygon_modes>>2u)&3u);
if(mode==0u)return true;float3 derivatives=max(fwidth(barycentric),float3(0.000001));
if(mode==1u)return min(barycentric.x/derivatives.x,min(barycentric.y/derivatives.y,barycentric.z/derivatives.z))<=max(state.polygon_line_width*0.5,0.5);
float3 normalized=barycentric/derivatives;float vertex_distance=min(max(normalized.y,normalized.z),min(max(normalized.x,normalized.z),max(normalized.x,normalized.y)));
return vertex_distance<=max(state.polygon_point_size*0.5,0.5);}
struct fragment_machine
{
	float4 regs16[48]; float4 regs32[48]; float4 cc[2];
	uint4 words; uint opcode; uint instruction_length; int ip;
};
inline float4 rsx_input(uint index,thread const float4* input,float4 position,bool front,bool two_sided)
{
	switch(index){case 0:return position;case 1:return two_sided&&!front?input[3]:input[1];case 2:return two_sided&&!front?input[4]:input[2];
	case 3:return input[5];case 13:return input[6];case 14:return front?float4(1.0):float4(-1.0);default:return input[min(index+3u,15u)];}
}
inline float4 read_fragment_source(thread fragment_machine& m,uint index,thread const float4* input,
	float4 position,bool front,device const uint4* program)
{
	uint source_word=index==0u?m.words.y:index==1u?m.words.z:m.words.w;
	uint type=rsx_bits(source_word,0,2); float4 value=0.0;
	if(type==0){uint reg=rsx_bits(source_word,2,6);
		value=rsx_bit(source_word,8)?m.regs16[reg%48u]:m.regs32[reg%48u];}
	else if(type==1){value=rsx_input(rsx_bits(m.words.x,13,4),input,position,front,program[0].w!=0u);}
	else if(type==2){m.instruction_length=2;value=as_type<float4>(rsx_swap16(program[uint(m.ip)+2u]));}
	value=rsx_swizzle(value,rsx_bits(source_word,9,8));
	if(index==0u?rsx_bit(source_word,29):rsx_bit(source_word,18))value=abs(value);
	if(rsx_bit(source_word,17))value=-value;
	return value;
}
inline float4 read_fragment_condition(thread fragment_machine& m)
{return rsx_swizzle(m.cc[rsx_bits(m.words.y,31,1)],rsx_bits(m.words.y,21,8));}
inline bool fragment_condition(thread fragment_machine& m)
{uint mode=rsx_bits(m.words.y,18,3);return mode==7||any(rsx_cond(read_fragment_condition(m),mode));}
inline void fragment_write(thread fragment_machine& m,float4 value)
{
	bool4 mask=bool4(rsx_bit(m.words.x,9),rsx_bit(m.words.x,10),rsx_bit(m.words.x,11),rsx_bit(m.words.x,12));
	float scales[8]={1.0,2.0,4.0,8.0,1.0,0.5,0.25,0.125};
	value*=scales[rsx_bits(m.words.z,28,3)]; if(rsx_bit(m.words.x,31))value=clamp(value,0.0,1.0);
	uint mode=rsx_bits(m.words.y,18,3);if(mode!=7)mask&=rsx_cond(read_fragment_condition(m),mode);
	if(rsx_bit(m.words.x,8)){uint cc=rsx_bits(m.words.y,30,1);m.cc[cc]=select(m.cc[cc],value,mask);}
	if(rsx_bit(m.words.x,30))return;uint reg=rsx_bits(m.words.x,1,6)%48u;
	if(rsx_bit(m.words.x,7))m.regs16[reg]=select(m.regs16[reg],value,mask);else m.regs32[reg]=select(m.regs32[reg],value,mask);
}
inline uint pack_unorm4(float4 value)
{uint4 v=uint4(round(clamp(value,0.0,1.0)*255.0));return v.x|(v.y<<8)|(v.z<<16)|(v.w<<24);}
inline uint pack_snorm4(float4 value)
{int4 v=int4(round(clamp(value,-1.0,1.0)*127.0));return(uint(v.x)&255u)|((uint(v.y)&255u)<<8)|((uint(v.z)&255u)<<16)|((uint(v.w)&255u)<<24);}
inline float4 unpack_unorm4(uint value)
{return float4(value&255u,(value>>8)&255u,(value>>16)&255u,(value>>24)&255u)/255.0;}
inline float4 unpack_snorm4(uint value)
{char4 v=as_type<char4>(value);return clamp(float4(v)/127.0,-1.0,1.0);}
inline uint4 rsx_logic_bits(uint4 s,uint4 d,uint op)
{switch(op){case 0:return 0u;case 1:return s&d;case 2:return s&~d;case 3:return s;case 4:return ~s&d;
case 5:return d;case 6:return s^d;case 7:return s|d;case 8:return ~(s|d);case 9:return ~(s^d);
case 10:return ~d;case 11:return s|~d;case 12:return ~s;case 13:return ~s|d;case 14:return ~(s&d);default:return ~0u;}}
inline float4 rsx_logic_apply(float4 s,float4 d,uint op,uint type,uint4 scale)
{if(type==1){ushort4 bits=ushort4(rsx_logic_bits(uint4(as_type<ushort4>(half4(s))),uint4(as_type<ushort4>(half4(d))),op));return float4(as_type<half4>(bits));}
if(type==2)return as_type<float4>(rsx_logic_bits(as_type<uint4>(s),as_type<uint4>(d),op));
float4 maximum=float4(max(scale,uint4(1u)));uint4 sb=uint4(round(clamp(s,0.0,1.0)*maximum));
uint4 db=uint4(round(clamp(d,0.0,1.0)*maximum));return float4(rsx_logic_bits(sb,db,op)&scale)/maximum;}
inline float blend_factor_a(uint f,float4 s,float4 d,float4 c)
{switch(f){case 0:return 0.0;case 1:return 1.0;case 0x300:case 0x302:return s.w;case 0x301:case 0x303:return 1.0-s.w;
case 0x304:case 0x306:return d.w;case 0x305:case 0x307:return 1.0-d.w;case 0x308:return 1.0;
case 0x8001:case 0x8003:return c.w;case 0x8002:case 0x8004:return 1.0-c.w;default:return 0.0;}}
inline float3 blend_factor_rgb(uint f,float4 s,float4 d,float4 c)
{switch(f){case 0:return 0.0;case 1:return 1.0;case 0x300:return s.xyz;case 0x301:return 1.0-s.xyz;
case 0x302:return s.www;case 0x303:return 1.0-s.www;case 0x304:return d.www;case 0x305:return 1.0-d.www;
case 0x306:return d.xyz;case 0x307:return 1.0-d.xyz;case 0x308:return min(s.www,1.0-d.www);
case 0x8001:return c.xyz;case 0x8002:return 1.0-c.xyz;case 0x8003:return c.www;case 0x8004:return 1.0-c.www;default:return 0.0;}}
inline float blend_eq_a(float s,float d,uint e)
{switch(e){case 0x8006:return saturate(s)+d;case 0x8007:return min(saturate(s),d);case 0x8008:return max(saturate(s),d);
case 0x800a:return saturate(s)-d;case 0x800b:return d-saturate(s);case 0xf005:return d-s;case 0xf006:case 0xf007:return s+d;default:return 0.0;}}
inline float3 blend_eq_rgb(float3 s,float3 d,uint e)
{switch(e){case 0x8006:return saturate(s)+d;case 0x8007:return min(saturate(s),d);case 0x8008:return max(saturate(s),d);
case 0x800a:return saturate(s)-d;case 0x800b:return d-saturate(s);case 0xf005:return d-s;case 0xf006:case 0xf007:return s+d;default:return 0.0;}}
inline float4 programmable_blend(float4 s,float4 d,constant fragment_state& state)
{float4 r;r.xyz=blend_eq_rgb(s.xyz*blend_factor_rgb(state.blend_factors_rgb&0xffff,s,d,state.blend_constants),
d.xyz*blend_factor_rgb(state.blend_factors_rgb>>16,s,d,state.blend_constants),state.blend_equations&0xffff);
r.w=blend_eq_a(s.w*blend_factor_a(state.blend_factors_alpha&0xffff,s,d,state.blend_constants),
d.w*blend_factor_a(state.blend_factors_alpha>>16,s,d,state.blend_constants),state.blend_equations>>16);
float4 t=as_type<float4>(as_type<uint4>(r)&0xfffff000u);return max(floor(fma(t,255.0,0.5)),0.0)/255.0;}
inline float tx_xform(float c,constant texture_parameter& p)
{float r=fma(c,p.scale_x,p.bias_x);return(p.flags&(1u<<21))?clamp(r,p.clamp_min_x,p.clamp_max_x):r;}
inline float2 tx_xform(float2 c,constant texture_parameter& p)
{float2 r=fma(c,float2(p.scale_x,p.scale_y),float2(p.bias_x,p.bias_y));return(p.flags&(1u<<21))?clamp(r,float2(p.clamp_min_x,p.clamp_min_y),float2(p.clamp_max_x,p.clamp_max_y)):r;}
inline float3 tx_xform(float3 c,constant texture_parameter& p)
{return fma(c,float3(p.scale_x,p.scale_y,p.scale_z),float3(p.bias_x,p.bias_y,p.bias_z));}
inline float sampler_coord(float c,constant sampler_state& state,uint size,uint axis)
{if(state.emulation_flags&(1u<<5))c/=float(max(size,1u));uint mode=(state.address_modes>>(axis*4u))&15u;
if(mode==6u||mode==7u)c=abs(c);if(mode==4u||mode==7u){float h=0.5/float(max(size,1u));c=clamp(c,-h,1.0+h);}return c;}
inline float border_coverage(float c,uint size,bool linear)
{float x=c*float(max(size,1u))-0.5;if(!linear)return(x>=-0.5&&x<float(size)-0.5)?1.0:0.0;
float lo=floor(x),f=x-lo,r=(lo>=0.0&&lo<float(size))?1.0-f:0.0;r+=(lo+1.0>=0.0&&lo+1.0<float(size))?f:0.0;return r;}
inline float sampler_bias(constant sampler_state& s)
{if(s.emulation_flags&(1u<<4))return s.lod_bias;int e=int((s.border_metadata>>20)&0x7ffu);if(e&0x400)e-=0x800;return float(e)/64.0;}
inline bool linear_filter(constant sampler_state& s,float lod){return!!(s.border_metadata&(1u<<(lod<=0.0?17u:16u)));}
inline float sampler_lod(float c,uint size){return log2(max(max(abs(dfdx(c)),abs(dfdy(c)))*float(size),0.000001));}
inline float sampler_lod(float2 c,uint2 size){float2 d=float2(size);return log2(max(max(length(dfdx(c)*d),length(dfdy(c)*d)),0.000001));}
inline float sampler_lod(float3 c,uint3 size){float3 d=float3(size);return log2(max(max(length(dfdx(c)*d),length(dfdy(c)*d)),0.000001));}
inline float4 custom_border(float4 v,float c,constant sampler_state& s,uint size,float lod)
{if(!(s.emulation_flags&1u))return v;float coverage=border_coverage(c,size,linear_filter(s,lod));return v+s.border_color*(1.0-coverage);}
inline float4 custom_border(float4 v,float2 c,constant sampler_state& s,uint2 size,float lod)
{if(!(s.emulation_flags&1u))return v;bool linear=linear_filter(s,lod);float coverage=border_coverage(c.x,size.x,linear)*border_coverage(c.y,size.y,linear);return v+s.border_color*(1.0-coverage);}
inline float4 custom_border(float4 v,float3 c,constant sampler_state& s,uint3 size,float lod)
{if(!(s.emulation_flags&1u))return v;bool linear=linear_filter(s,lod);float coverage=border_coverage(c.x,size.x,linear)*border_coverage(c.y,size.y,linear)*border_coverage(c.z,size.z,linear);return v+s.border_color*(1.0-coverage);}
inline float4 sample_1d(texture2d<float> t,sampler s,float c,constant sampler_state& state)
{uint size=t.get_width();c=sampler_coord(c,state,size,0);return custom_border(t.sample(s,float2(c,0.5),bias(state.lod_bias)),c,state,size,sampler_lod(c,size)+sampler_bias(state));}
inline float4 sample_2d(texture2d<float> t,sampler s,float2 c,constant sampler_state& state)
{uint2 size=uint2(t.get_width(),t.get_height());c=float2(sampler_coord(c.x,state,size.x,0),sampler_coord(c.y,state,size.y,1));return custom_border(t.sample(s,c,bias(state.lod_bias)),c,state,size,sampler_lod(c,size)+sampler_bias(state));}
inline float4 sample_3d(texture3d<float> t,sampler s,float3 c,constant sampler_state& state)
{uint3 size=uint3(t.get_width(),t.get_height(),t.get_depth());c=float3(sampler_coord(c.x,state,size.x,0),sampler_coord(c.y,state,size.y,1),sampler_coord(c.z,state,size.z,2));return custom_border(t.sample(s,c,bias(state.lod_bias)),c,state,size,sampler_lod(c,size)+sampler_bias(state));}
inline float4 sample_cube(texturecube<float> t,sampler s,float3 c,constant sampler_state& state)
{return t.sample(s,c,bias(state.lod_bias));}
inline float4 sample_1d_lod(texture2d<float> t,sampler s,float c,float lod,constant sampler_state& state)
{uint level_index=min(uint(max(lod,0.0)),t.get_num_mip_levels()-1u),size=max(t.get_width()>>level_index,1u);c=sampler_coord(c,state,size,0);return custom_border(t.sample(s,float2(c,0.5),level(lod)),c,state,size,lod);}
inline float4 sample_2d_lod(texture2d<float> t,sampler s,float2 c,float lod,constant sampler_state& state)
{uint li=min(uint(max(lod,0.0)),t.get_num_mip_levels()-1u);uint2 size=uint2(max(t.get_width()>>li,1u),max(t.get_height()>>li,1u));c=float2(sampler_coord(c.x,state,size.x,0),sampler_coord(c.y,state,size.y,1));return custom_border(t.sample(s,c,level(lod)),c,state,size,lod);}
inline float4 sample_3d_lod(texture3d<float> t,sampler s,float3 c,float lod,constant sampler_state& state)
{uint li=min(uint(max(lod,0.0)),t.get_num_mip_levels()-1u);uint3 size=uint3(max(t.get_width()>>li,1u),max(t.get_height()>>li,1u),max(t.get_depth()>>li,1u));c=float3(sampler_coord(c.x,state,size.x,0),sampler_coord(c.y,state,size.y,1),sampler_coord(c.z,state,size.z,2));return custom_border(t.sample(s,c,level(lod)),c,state,size,lod);}
inline float4 sample_ms(texture2d_ms<float,access::read> t,float2 c,constant sampler_state& state)
{uint2 size=uint2(t.get_width(),t.get_height());uint2 p=(state.emulation_flags&(1u<<5))?uint2(max(c,float2(0.0))):uint2(max(c,float2(0.0))*float2(size));p=min(p,size-1u);
float4 result=0.0;uint count=t.get_num_samples();for(uint sample=0;sample<count;++sample)result+=t.read(p,sample);return result/float(max(count,1u));}
inline float4 decode_depth(float depth,uint stencil,constant texture_parameter& p)
{uint value=(p.flags&(1u<<14))?((as_type<uint>(depth)>>7)&0x00ffffffu):uint(depth*16777215.0);
float4 result=float4(float((value>>8)&255u),float(value&255u),float(stencil&255u),float((value>>16)&255u))/255.0;
if(p.remap==0x0000aae4u)return result;uint4 channels=((uint4(p.remap)>>uint4(2,4,6,0))&3u)+3u;channels%=4u;
result=float4(result[channels.x],result[channels.y],result[channels.z],result[channels.w]);uint4 controls=(uint4(p.remap)>>uint4(10,12,14,8))&3u;return select(float4(controls),result,controls<2u);}
inline uint sample_stencil(texture2d<uint> t,float2 c)
{if(any(c<0.0||c>=1.0))return 0u;uint2 size=uint2(t.get_width(),t.get_height());return t.read(min(uint2(c*float2(size)),size-1u)).x;}
inline float4 process_texel(float4 v,uint flags)
{if((flags&(1u<<4))&&v.w<0.000001)discard_fragment();if(flags&(1u<<5))v=floor(v*255.0)/255.0;
if(flags&0x3c00){int4 bits=int4(round(v*((flags&(1u<<28))?65535.0:255.0)));float4 converted=(flags&(1u<<28))?float4((bits<<16)>>16)/32767.0:float4((bits<<24)>>24)/127.0;v=select(v,converted,bool4(flags&(1u<<11),flags&(1u<<12),flags&(1u<<13),flags&(1u<<10)));}
if(flags&0x000f){float3 linear=select(v.xyz/12.92,pow((v.xyz+0.055)/1.055,2.4),v.xyz>0.04045);v=select(v,float4(linear,v.w),bool4(flags&2u,flags&4u,flags&8u,flags&1u));}
if(flags&0x03c0){float4 converted=(flags&(1u<<28))?(floor(v*65535.0+0.5)-32768.0)/32767.0:(floor(v*255.0+0.5)-128.0)/127.0;v=select(v,converted,bool4(flags&(1u<<7),flags&(1u<<8),flags&(1u<<9),flags&(1u<<6)));}return v;}
inline bool shadow_compare(float sampled,float reference,uint op)
{switch(op){case 1:return sampled<reference;case 2:return sampled==reference;case 3:return sampled<=reference;case 4:return sampled>reference;case 5:return sampled!=reference;case 6:return sampled>=reference;case 7:return true;default:return false;}}
inline float4 finish_texel(float4 v,float4 coordinate,uint type,constant texture_parameter& p,bool expand)
{uint comparison=(p.flags>>15)&7u;if(comparison){float reference=type==0u?coordinate.y:type==1u?coordinate.z:coordinate.w;v=float4(float(shadow_compare(v.x,reference,comparison)));}
v=process_texel(v,p.flags);return expand?v*2.0-1.0:v;}

fragment fragment_result rsx_interpreter_fragment(fragment_input stage [[stage_in]],
	constant fragment_state& state [[buffer(0)]],
	constant texture_parameter* texture_parameters [[buffer(2)]],
	constant raster_context& raster [[buffer(3)]],
	constant uint* stipple_pattern [[buffer(4)]],
	device const uint4* program [[buffer(5)]],
	constant sampler_state* sampler_states [[buffer(6)]],
	array<texture2d<float>,16> textures1d [[texture(0)]],
	array<texture2d<float>,16> textures2d [[texture(16)]],
	array<texturecube<float>,16> texturescube [[texture(32)]],
	array<texture3d<float>,16> textures3d [[texture(48)]],
	array<texture2d_ms<float,access::read>,16> texturesms [[texture(64)]],
	array<texture2d<uint>,16> stencil_textures [[texture(80)]],
	array<sampler,16> texture_samplers [[sampler(0)]],
	uint sample_id [[sample_id]],float3 barycentric [[barycentric_coord]]
)MSL";
			const u32 framebuffer_fetch_mask =
				(options & compiler_opt_framebuffer_fetch_mask) >> compiler_opt_framebuffer_fetch_shift;
			for (u32 index = 0; index < 4; ++index)
			{
				if (framebuffer_fetch_mask & (1u << index))
					source += fmt::format(",float4 framebuffer_color%u [[color(%u)]]", index, index);
			}
			source += R"MSL()
{
	if(!polygon_fragment_covered(barycentric,stage.front_facing,state))discard_fragment();
	bool guest_front_facing=(state.polygon_modes&(1u<<30u))?true:stage.front_facing;
	fragment_machine m;for(uint i=0;i<48;++i){m.regs16[i]=0.0;m.regs32[i]=0.0;}m.cc[0]=0.0;m.cc[1]=0.0;m.ip=-1;m.instruction_length=1;
	float4 inputs[16]={stage.user0,stage.user1,stage.user2,stage.user3,stage.user4,stage.user5,stage.user6,stage.user7,
		stage.user8,stage.user9,stage.user10,stage.user11,stage.user12,stage.user13,stage.user14,stage.user15};
)MSL";
			if (options & COMPILER_OPT_ENABLE_STIPPLING)
			{
				source += R"MSL(
	uint2 pixel=uint2(stage.position.xy)&31u;uint address=pixel.y*32u+pixel.x;
	if((stipple_pattern[address>>5]&(1u<<(address&31u)))==0u)discard_fragment();
)MSL";
			}
			source += R"MSL(
	int test_address=-1,jump_address=-1,loop_start=-1,loop_end=-1,counter=0;
	for(uint watchdog=0;watchdog<8192;++watchdog)
	{
		m.ip+=int(m.instruction_length);m.instruction_length=1;
		if(test_address==m.ip){m.ip=jump_address;test_address=-1;jump_address=-1;}
		else if(loop_end==m.ip){if(counter-->0)m.ip=loop_start;else{loop_start=-1;loop_end=-1;}}
		uint instruction_count=program[0].z;if(m.ip<0||uint(m.ip)>=instruction_count)break;
		m.words=rsx_swap16(program[uint(m.ip)+1u]);m.opcode=rsx_bits(m.words.x,24,6);bool end=rsx_bit(m.words.x,0);
		if(rsx_bit(m.words.z,31))
		{
			uint flow=m.opcode|64u;
			if(flow==0x45){if(fragment_condition(m))break;continue;}
			if(flow==0x42){uint alternate=rsx_bits(m.words.z,0,31);if(fragment_condition(m)){if(alternate<m.words.w){test_address=int(alternate>>2);jump_address=int(m.words.w>>2);}}
				else{m.ip=int(alternate>>2);m.instruction_length=0;}continue;}
			if(flow==0x43||flow==0x44){if(fragment_condition(m)){uint first=rsx_bits(m.words.z,10,8),last=rsx_bits(m.words.z,2,8),step=max(rsx_bits(m.words.z,19,8),1u);
				if(last>first){counter=int((last-first-1u)/step);loop_start=m.ip+1;loop_end=int(m.words.w>>2);continue;}}
				m.ip=int(m.words.w>>2);m.instruction_length=0;continue;}
			if(flow==0x40){if(fragment_condition(m)&&loop_end>0){m.ip=loop_end;m.instruction_length=0;counter=0;}continue;}
			continue;
		}
		if(m.opcode==0||m.opcode==0x3d||m.opcode==0x3e){if(end)break;continue;}
)MSL";
			if (options & COMPILER_OPT_ENABLE_KIL)
				source += "\t\tif(m.opcode==0x12){if(fragment_condition(m))discard_fragment();if(end)break;continue;}\n";
			source += R"MSL(
		float4 s0=read_fragment_source(m,0,inputs,stage.position,guest_front_facing,program),result=0.0;
		bool handled=true;
		switch(m.opcode)
		{
		case 0x01:result=s0;break;case 0x10:result=fract(s0);break;case 0x11:result=floor(s0);break;
		case 0x15:result=dfdx(s0);break;case 0x16:result=dfdy(s0);break;case 0x1a:result=1.0/s0.x;break;
		case 0x1b:result=rsqrt(abs(s0.x));break;case 0x1c:result=exp2(s0.x);break;case 0x1d:result=log2(abs(s0.x));break;
		case 0x20:result=1.0;break;case 0x21:result=0.0;break;case 0x22:result=cos(s0.x);break;case 0x23:result=sin(s0.x);break;
		case 0x39:result=float4(normalize(s0.xyz),normalize(s0.xyz).z);break;case 0x1e:result=rsx_lit(s0);break;case 0x3c:result=rsx_lif(s0);break;
)MSL";
			if (options & COMPILER_OPT_ENABLE_PACKING)
			{
				source += R"MSL(
		case 0x13:result=as_type<float>(pack_snorm4(s0));break;case 0x14:result=unpack_snorm4(as_type<uint>(s0.x));break;
		case 0x27:case 0x2c:result=as_type<float>(pack_unorm4(s0));break;case 0x28:case 0x2d:result=unpack_unorm4(as_type<uint>(s0.x));break;
		case 0x24:{half2 h=half2(s0.xy);result=as_type<float>(h);break;}case 0x25:{half2 h=as_type<half2>(s0.x);result=float4(float2(h).xyxy);break;}
)MSL";
			}
			if (options & COMPILER_OPT_ENABLE_TEXTURES)
			{
				source += R"MSL(
		case 0x17:case 0x18:
		{uint unit=rsx_bits(m.words.x,17,4),type=rsx_bits(program[0].y,unit*2u,2);float4 coord=m.opcode==0x18?float4(s0.xyz/s0.w,s0.w):s0;
		 constant texture_parameter& p=texture_parameters[unit];
		 bool multisampled=!!(sampler_states[unit].border_metadata&(1u<<9));
		 if(multisampled){coord.xy=tx_xform(coord.xy,p);result=sample_ms(texturesms[unit],coord.xy,sampler_states[unit]);}
		 else switch(type){case 0:coord.x=tx_xform(coord.x,p);result=sample_1d(textures1d[unit],texture_samplers[unit],coord.x,sampler_states[unit]);break;
		 case 1:coord.xy=tx_xform(coord.xy,p);result=sample_2d(textures2d[unit],texture_samplers[unit],coord.xy,sampler_states[unit]);break;
		 case 2:coord.xyz=tx_xform(coord.xyz,p);result=sample_cube(texturescube[unit],texture_samplers[unit],coord.xyz,sampler_states[unit]);break;
		 default:coord.xyz=tx_xform(coord.xyz,p);result=sample_3d(textures3d[unit],texture_samplers[unit],coord.xyz,sampler_states[unit]);break;}
		 if(sampler_states[unit].border_metadata&(1u<<10)){uint stencil=multisampled?0u:sample_stencil(stencil_textures[unit],type==0u?float2(coord.x,0.5):coord.xy);result=decode_depth(result.x,stencil,p);}
		 result=finish_texel(result,coord,type,p,rsx_bit(m.words.x,21));break;}
)MSL";
			}
			source += R"MSL(
		default:handled=false;break;}
		float4 s1=0.0;
		if(!handled){s1=read_fragment_source(m,1,inputs,stage.position,guest_front_facing,program);handled=true;
		 switch(m.opcode){case 0x02:result=s0*s1;break;case 0x03:result=s0+s1;break;case 0x38:result=dot(s0.xy,s1.xy);break;
		 case 0x05:result=dot(s0.xyz,s1.xyz);break;case 0x06:result=dot(s0,s1);break;case 0x07:result=float4(1.0,s0.y*s1.y,s0.z,s1.w);break;
		 case 0x08:result=min(s0,s1);break;case 0x09:result=max(s0,s1);break;case 0x0a:result=float4(s0<s1);break;
		 case 0x0b:result=float4(s0>=s1);break;case 0x0c:result=float4(s0<=s1);break;case 0x0d:result=float4(s0>s1);break;
		 case 0x0e:result=float4(s0!=s1);break;case 0x0f:result=float4(s0==s1);break;case 0x26:result=pow(s0.x,s1.x);break;
		 case 0x3a:result=s0/s1.x;break;case 0x3b:result=select(s0,s0*rsqrt(abs(s1.x)),s0!=0.0);break;case 0x36:result=reflect(s0,s1);break;
)MSL";
			if (options & COMPILER_OPT_ENABLE_TEXTURES)
			{
				source += R"MSL(
		 case 0x2f:case 0x31:{uint unit=rsx_bits(m.words.x,17,4),type=rsx_bits(program[0].y,unit*2u,2);float4 coord=s0;
		  constant texture_parameter& p=texture_parameters[unit];bool multisampled=!!(sampler_states[unit].border_metadata&(1u<<9));
		  if(multisampled){coord.xy=tx_xform(coord.xy,p);result=sample_ms(texturesms[unit],coord.xy,sampler_states[unit]);}
		  else if(type==0){coord.x=tx_xform(coord.x,p);result=sample_1d_lod(textures1d[unit],texture_samplers[unit],coord.x,s1.x,sampler_states[unit]);}
		  else if(type==1){coord.xy=tx_xform(coord.xy,p);result=sample_2d_lod(textures2d[unit],texture_samplers[unit],coord.xy,s1.x,sampler_states[unit]);}
		  else if(type==2){coord.xyz=tx_xform(coord.xyz,p);result=texturescube[unit].sample(texture_samplers[unit],coord.xyz,level(s1.x));}
		  else{coord.xyz=tx_xform(coord.xyz,p);result=sample_3d_lod(textures3d[unit],texture_samplers[unit],coord.xyz,s1.x,sampler_states[unit]);}
		  if(sampler_states[unit].border_metadata&(1u<<10)){uint stencil=multisampled?0u:sample_stencil(stencil_textures[unit],type==0u?float2(coord.x,0.5):coord.xy);result=decode_depth(result.x,stencil,p);}
		  result=finish_texel(result,coord,type,p,false);break;}
)MSL";
			}
			source += R"MSL(
		 default:handled=false;break;}}
		if(!handled){float4 s2=read_fragment_source(m,2,inputs,stage.position,guest_front_facing,program);
		 switch(m.opcode){case 0x04:result=fma(s0,s1,s2);break;case 0x1f:result=mix(s2,s1,s0);break;
		 case 0x2e:result=dot(s0.xy,s1.xy)+s2.x;break;default:result=0.0;break;}}
		fragment_write(m,result);if(end)break;
	}
	fragment_result output;
)MSL";
			if (options & COMPILER_OPT_ENABLE_F32_EXPORT)
				source += "\toutput.color0=m.regs32[0];output.color1=m.regs32[2];output.color2=m.regs32[3];output.color3=m.regs32[4];\n";
			else
				source += "\toutput.color0=m.regs16[0];output.color1=m.regs16[4];output.color2=m.regs16[6];output.color3=m.regs16[8];\n";
			source += "\toutput.sample_mask=state.sample_mask;(void)sample_id;\n";
			if (options & COMPILER_OPT_ENABLE_DEPTH_EXPORT)
				source += "\toutput.depth=m.regs32[1].z;\n";
			switch (options & COMPILER_OPT_ALPHA_TEST_MASK)
			{
			case COMPILER_OPT_ENABLE_ALPHA_TEST_GE: source += "\tif(output.color0.a<state.alpha_ref)discard_fragment();\n"; break;
			case COMPILER_OPT_ENABLE_ALPHA_TEST_G: source += "\tif(output.color0.a<=state.alpha_ref)discard_fragment();\n"; break;
			case COMPILER_OPT_ENABLE_ALPHA_TEST_LE: source += "\tif(output.color0.a>state.alpha_ref)discard_fragment();\n"; break;
			case COMPILER_OPT_ENABLE_ALPHA_TEST_L: source += "\tif(output.color0.a>=state.alpha_ref)discard_fragment();\n"; break;
			case COMPILER_OPT_ENABLE_ALPHA_TEST_EQ: source += "\tif(output.color0.a!=state.alpha_ref)discard_fragment();\n"; break;
			case COMPILER_OPT_ENABLE_ALPHA_TEST_NE: source += "\tif(output.color0.a==state.alpha_ref)discard_fragment();\n"; break;
			default: break;
			}
			if (framebuffer_fetch_mask)
			{
				for (u32 index = 0; index < 4; ++index)
				{
					if (!(framebuffer_fetch_mask & (1u << index))) continue;
					source += fmt::format("\tif(state.rop_emulation&1u)output.color%u=rsx_logic_apply(output.color%u,framebuffer_color%u,state.logic_operation,state.logic_types[%u],state.logic_scales[%u]);\n",
						index, index, index, index, index);
					source += fmt::format("\tif((state.rop_emulation&2u)&&(state.programmable_blend_mask&%uu))output.color%u=programmable_blend(output.color%u,framebuffer_color%u,state);\n",
						1u << index, index, index, index);
				}
			}
			source += "\t(void)raster;return output;\n}\n";
			return source;
		}

		interpreter_shader_sources make_sources(u32 options)
		{
			interpreter_shader_sources result;
			result.vertex = vertex_interpreter_source(options);
			result.fragment = fragment_interpreter_source(options);
			result.vertex_layout = {11, 4, 4, false};
			result.fragment_layout = {7, 96, 16, false};
			auto add_buffer = [](auto& list, msl_shader_stage stage, u32 index, std::string_view name)
			{
				list.push_back({stage, argument_binding_class::buffer, index, umax, std::string(name)});
			};
			add_buffer(result.vertex_required_bindings, msl_shader_stage::vertex, 0, "persistent vertex stream");
			add_buffer(result.vertex_required_bindings, msl_shader_stage::vertex, 1, "volatile vertex stream");
			add_buffer(result.vertex_required_bindings, msl_shader_stage::vertex, 2, "draw parameters");
			add_buffer(result.vertex_required_bindings, msl_shader_stage::vertex, 3, "vertex interpreter context");
			add_buffer(result.vertex_required_bindings, msl_shader_stage::vertex, 4, "conditional-render predicate");
			add_buffer(result.vertex_required_bindings, msl_shader_stage::vertex, 5, "vertex constants");
			add_buffer(result.vertex_required_bindings, msl_shader_stage::vertex, 8, "vertex sampler state");
			add_buffer(result.vertex_required_bindings, msl_shader_stage::vertex, 9, "vertex instructions");
			add_buffer(result.vertex_required_bindings, msl_shader_stage::vertex, 10, "line mapping");
			add_buffer(result.fragment_required_bindings, msl_shader_stage::fragment, 0, "fragment interpreter state");
			add_buffer(result.fragment_required_bindings, msl_shader_stage::fragment, 2, "fragment texture parameters");
			add_buffer(result.fragment_required_bindings, msl_shader_stage::fragment, 3, "fragment raster context");
			add_buffer(result.fragment_required_bindings, msl_shader_stage::fragment, 5, "fragment instructions");
			add_buffer(result.fragment_required_bindings, msl_shader_stage::fragment, 6, "fragment sampler state");
			if (options & COMPILER_OPT_ENABLE_STIPPLING)
				add_buffer(result.fragment_required_bindings, msl_shader_stage::fragment, 4, "polygon stipple pattern");
			result.validate();
			return result;
		}

		native_graphics_pipeline_compile_request make_request(const interpreter_shader_sources& sources,
			const graphics_pipeline_configuration& configuration)
		{
			native_graphics_pipeline_compile_request request;
			request.vertex_source = sources.vertex;
			request.fragment_source = sources.fragment;
			request.vertex_function_name = "rsx_interpreter_vertex";
			request.fragment_function_name = "rsx_interpreter_fragment";
			request.vertex_layout = sources.vertex_layout;
			request.fragment_layout = sources.fragment_layout;
			request.vertex_required_bindings = sources.vertex_required_bindings;
			request.fragment_required_bindings = sources.fragment_required_bindings;
			request.configuration = configuration;
			request.configuration.label = "RPCS3 shader interpreter";
			request.fast_math = false;
			return request;
		}
	}

	u32 interpreter_program_state::compiler_options() const
	{
		u32 result = 0;
		switch (alpha_test)
		{
		case interpreter_alpha_test::disabled: break;
		case interpreter_alpha_test::never: break;
		case interpreter_alpha_test::greater_equal: result |= COMPILER_OPT_ENABLE_ALPHA_TEST_GE; break;
		case interpreter_alpha_test::greater: result |= COMPILER_OPT_ENABLE_ALPHA_TEST_G; break;
		case interpreter_alpha_test::less_equal: result |= COMPILER_OPT_ENABLE_ALPHA_TEST_LE; break;
		case interpreter_alpha_test::less: result |= COMPILER_OPT_ENABLE_ALPHA_TEST_L; break;
		case interpreter_alpha_test::equal: result |= COMPILER_OPT_ENABLE_ALPHA_TEST_EQ; break;
		case interpreter_alpha_test::not_equal: result |= COMPILER_OPT_ENABLE_ALPHA_TEST_NE; break;
		}
		if (fragment_control & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT) result |= COMPILER_OPT_ENABLE_DEPTH_EXPORT;
		if (fragment_control & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS) result |= COMPILER_OPT_ENABLE_F32_EXPORT;
		if (fragment_control & RSX_SHADER_CONTROL_USES_KIL) result |= COMPILER_OPT_ENABLE_KIL;
		if (fragment_metadata.referenced_textures_mask) result |= COMPILER_OPT_ENABLE_TEXTURES;
		if (fragment_metadata.has_branch_instructions) result |= COMPILER_OPT_ENABLE_FLOW_CTRL;
		if (fragment_metadata.has_pack_instructions) result |= COMPILER_OPT_ENABLE_PACKING;
		if (polygon_stipple) result |= COMPILER_OPT_ENABLE_STIPPLING;
		if (vertex_control & RSX_SHADER_CONTROL_INSTANCED_CONSTANTS) result |= COMPILER_OPT_ENABLE_INSTANCING;
		if (vertex_metadata.referenced_textures_mask) result |= COMPILER_OPT_ENABLE_VTX_TEXTURES;
		return result;
	}

	u64 interpreter_pipeline_key::hash() const
	{
		u64 result = key_seed;
		combine(result, compiler_options);
		combine(result, configuration.signature());
		return result;
	}

	usz interpreter_pipeline_key_hash::operator()(const interpreter_pipeline_key& key) const noexcept
	{
		try
		{
			return static_cast<usz>(key.hash());
		}
		catch (...)
		{
			return 0;
		}
	}

	void interpreter_shader_sources::validate() const
	{
		if (vertex.empty() || fragment.empty() || vertex_layout.buffer_count < 11 ||
		fragment_layout.buffer_count < 7 || fragment_layout.texture_count < 96 ||
		fragment_layout.sampler_count < 16)
		{
			fmt::throw_exception("Invalid Metal shader interpreter source variant");
		}
		vertex_layout.validate();
		fragment_layout.validate();
	}

	struct MTLShaderInterpreter::impl
	{
		render_device* device = nullptr;
		MTLPipelineCompiler* compiler = nullptr;
		std::unordered_map<u32, interpreter_shader_sources> sources;
		std::unordered_map<interpreter_pipeline_key, std::shared_ptr<MTLProgramPipeline>,
			interpreter_pipeline_key_hash> pipelines;
		interpreter_pipeline_key current_key;
		std::shared_ptr<MTLProgramPipeline> current;
		interpreter_pipeline_info info;
		interpreter_statistics counters;
		bool has_current_key = false;
		mutable std::mutex mutex;

		void require_initialized() const
		{
			if (!device || !compiler || !*compiler)
				fmt::throw_exception("Metal shader interpreter is not initialized");
		}

		interpreter_shader_sources source_variant(u32 options)
		{
			std::lock_guard lock(mutex);
			if (const auto found = sources.find(options); found != sources.end()) return found->second;
			auto value = make_sources(options);
			sources.emplace(options, value);
			counters.source_variants++;
			return value;
		}
	};

	MTLShaderInterpreter::MTLShaderInterpreter()
		: m_impl(std::make_unique<impl>())
	{
	}

	MTLShaderInterpreter::~MTLShaderInterpreter()
	{
		destroy();
	}

	void MTLShaderInterpreter::initialize(render_device& device, MTLPipelineCompiler& compiler)
	{
		destroy();
		if (!device || !compiler || &compiler.owner() != &device)
			fmt::throw_exception("Metal shader interpreter requires a compiler for its render device");
		m_impl->device = &device;
		m_impl->compiler = &compiler;
	}

	void MTLShaderInterpreter::destroy()
	{
		if (!m_impl) return;
		MTLPipelineCompiler* compiler = nullptr;
		{
			std::lock_guard lock(m_impl->mutex);
			compiler = m_impl->compiler;
		}
		if (compiler && *compiler) compiler->wait_idle();
		std::lock_guard lock(m_impl->mutex);
		m_impl->current.reset();
		m_impl->pipelines.clear();
		m_impl->sources.clear();
		m_impl->device = nullptr;
		m_impl->compiler = nullptr;
		m_impl->has_current_key = false;
		m_impl->counters = {};
	}

	std::shared_ptr<MTLProgramPipeline> MTLShaderInterpreter::get(
		const graphics_pipeline_configuration& configuration,
		const interpreter_program_state& program_state, bool asynchronous,
		interpreter_completion_callback completion)
	{
		m_impl->require_initialized();
		if (program_state.alpha_test == interpreter_alpha_test::never) return {};
		const u32 options = program_state.compiler_options() |
			framebuffer_fetch_options(configuration, m_impl->device->info().features.framebuffer_fetch) |
			(configuration.state.render.topology == primitive_topology::point ? compiler_opt_point_size : 0);
		interpreter_pipeline_key key{options, configuration};
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->current_key = key;
			m_impl->has_current_key = true;
			if (const auto found = m_impl->pipelines.find(key); found != m_impl->pipelines.end())
			{
				m_impl->current = found->second;
				m_impl->counters.cache_hits++;
				return found->second;
			}
		}
		const auto sources = m_impl->source_variant(options);
		auto request = make_request(sources, configuration);
		if (!asynchronous)
		{
			try
			{
				auto pipeline = m_impl->compiler->compile_native_graphics_inline(request);
				{
					std::lock_guard lock(m_impl->mutex);
					m_impl->pipelines[key] = pipeline;
					m_impl->current = pipeline;
					m_impl->counters.pipelines = m_impl->pipelines.size();
					m_impl->counters.synchronous_compiles++;
				}
				if (completion) completion(key, pipeline, {});
				return pipeline;
			}
			catch (const std::exception& exception)
			{
				{
					std::lock_guard lock(m_impl->mutex);
					m_impl->counters.failed_compiles++;
				}
				if (completion) completion(key, {}, exception.what());
				throw;
			}
		}

		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->counters.asynchronous_compiles++;
		}
		(void)m_impl->compiler->submit_native_graphics(std::move(request), pipeline_compile_priority::normal,
			[this, key, completion = std::move(completion)](const pipeline_compile_job& job)
			{
				auto pipeline = job.result();
				const std::string diagnostic = job.diagnostic();
				{
					std::lock_guard lock(m_impl->mutex);
					if (pipeline)
					{
						m_impl->pipelines[key] = pipeline;
						if (m_impl->has_current_key && m_impl->current_key == key) m_impl->current = pipeline;
						m_impl->counters.pipelines = m_impl->pipelines.size();
					}
					else
					{
						m_impl->counters.failed_compiles++;
					}
				}
				if (completion) completion(key, pipeline, diagnostic);
			});
		return {};
	}

	void MTLShaderInterpreter::preload(std::span<const graphics_pipeline_configuration> configurations,
		interpreter_progress_callback progress)
	{
		m_impl->require_initialized();
		const auto variants = get_interpreter_variants();
		const u32 total = static_cast<u32>(configurations.size() * variants.base_pipelines.size());
		u32 completed = 0;
		if (progress) progress(0, total);
		for (const auto& configuration : configurations)
		{
			for (const auto& [vertex_options, fragment_options] : variants.base_pipelines)
			{
				const u32 options = vertex_options | fragment_options |
					framebuffer_fetch_options(configuration, m_impl->device->info().features.framebuffer_fetch) |
					(configuration.state.render.topology == primitive_topology::point ? compiler_opt_point_size : 0);
				interpreter_pipeline_key key{options, configuration};
				bool cached = false;
				{
					std::lock_guard lock(m_impl->mutex);
					cached = m_impl->pipelines.contains(key);
				}
				if (!cached)
				{
					const auto sources = m_impl->source_variant(options);
					try
					{
						auto pipeline = m_impl->compiler->compile_native_graphics_inline(
							make_request(sources, configuration));
						std::lock_guard lock(m_impl->mutex);
						m_impl->pipelines[key] = std::move(pipeline);
						m_impl->counters.synchronous_compiles++;
						m_impl->counters.pipelines = m_impl->pipelines.size();
					}
					catch (...)
					{
						std::lock_guard lock(m_impl->mutex);
						m_impl->counters.failed_compiles++;
						throw;
					}
				}
				++completed;
				if (progress) progress(completed, total);
			}
		}
	}

	std::shared_ptr<MTLProgramPipeline> MTLShaderInterpreter::current_pipeline() const
	{
		if (!m_impl) return {};
		std::lock_guard lock(m_impl->mutex);
		return m_impl->current;
	}

	bool MTLShaderInterpreter::is_interpreter(const MTLProgramPipeline* pipeline) const
	{
		if (!m_impl || !pipeline) return false;
		std::lock_guard lock(m_impl->mutex);
		for (const auto& [key, value] : m_impl->pipelines)
		{
			(void)key;
			if (value.get() == pipeline) return true;
		}
		return false;
	}

	std::pair<interpreter_shader_sources, interpreter_pipeline_info>
	MTLShaderInterpreter::variant(u32 compiler_options) const
	{
		return {make_sources(compiler_options), interpreter_pipeline_info{}};
	}

	void MTLShaderInterpreter::set_vertex_instruction_buffer(const buffer& resource, u64 offset, u64 length)
	{
		auto pipeline = current_pipeline();
		if (!pipeline) fmt::throw_exception("Metal shader interpreter has no current pipeline");
		pipeline->set_buffer(msl_shader_stage::vertex, m_impl->info.vertex_instruction_buffer,
			resource, offset, length);
	}

	void MTLShaderInterpreter::set_fragment_instruction_buffer(const buffer& resource, u64 offset, u64 length)
	{
		auto pipeline = current_pipeline();
		if (!pipeline) fmt::throw_exception("Metal shader interpreter has no current pipeline");
		pipeline->set_buffer(msl_shader_stage::fragment, m_impl->info.fragment_instruction_buffer,
			resource, offset, length);
	}

	void MTLShaderInterpreter::update_fragment_textures(u32 first_index,
		std::span<const argument_texture_binding> textures)
	{
		if (first_index > m_impl->info.fragment_texture_count ||
			textures.size() > m_impl->info.fragment_texture_count - first_index)
		{
			fmt::throw_exception("Metal shader interpreter texture range exceeds its layout");
		}
		auto pipeline = current_pipeline();
		if (!pipeline) fmt::throw_exception("Metal shader interpreter has no current pipeline");
		for (u32 index = 0; index < textures.size(); ++index)
		{
			pipeline->set_texture(msl_shader_stage::fragment,
				m_impl->info.fragment_texture_first + first_index + index, textures[index]);
		}
	}

	void MTLShaderInterpreter::update_fragment_samplers(u32 first_index,
		std::span<const argument_sampler_binding> samplers)
	{
		if (first_index > m_impl->info.fragment_sampler_count ||
			samplers.size() > m_impl->info.fragment_sampler_count - first_index)
		{
			fmt::throw_exception("Metal shader interpreter sampler range exceeds its layout");
		}
		auto pipeline = current_pipeline();
		if (!pipeline) fmt::throw_exception("Metal shader interpreter has no current pipeline");
		for (u32 index = 0; index < samplers.size(); ++index)
		{
			pipeline->set_sampler(msl_shader_stage::fragment,
				m_impl->info.fragment_sampler_first + first_index + index, samplers[index]);
		}
	}

	void MTLShaderInterpreter::bind(command_buffer& command)
	{
		auto pipeline = current_pipeline();
		if (!pipeline) fmt::throw_exception("Metal shader interpreter has no current pipeline");
		pipeline->bind(command);
	}

	MTLShaderInterpreter::operator bool() const
	{
		if (!m_impl) return false;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->device && m_impl->compiler && bool(*m_impl->compiler);
	}

	interpreter_pipeline_info MTLShaderInterpreter::current_pipeline_info() const
	{
		return m_impl ? m_impl->info : interpreter_pipeline_info{};
	}

	interpreter_statistics MTLShaderInterpreter::statistics() const
	{
		if (!m_impl) return {};
		std::lock_guard lock(m_impl->mutex);
		return m_impl->counters;
	}
}
