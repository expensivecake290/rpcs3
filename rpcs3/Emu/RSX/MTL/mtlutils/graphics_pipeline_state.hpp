#pragma once

#include <array>
#include <bit>
#include <functional>

#include "Utilities/StrFmt.h"
#include "pipeline_binding_table.h"

namespace mtl
{
	inline constexpr u32 maximum_color_attachments = 4;

	enum class primitive_topology : u8
	{
		point,
		line,
		line_strip,
		triangle,
		triangle_strip,
	};

	enum class triangle_fill_mode : u8
	{
		fill,
		lines,
	};

	enum class cull_mode : u8
	{
		none,
		front,
		back,
	};

	enum class front_face : u8
	{
		clockwise,
		counter_clockwise,
	};

	enum class depth_clip_mode : u8
	{
		clip,
		clamp,
	};

	enum class compare_function : u8
	{
		never,
		less,
		equal,
		less_equal,
		greater,
		not_equal,
		greater_equal,
		always,
	};

	enum class stencil_operation : u8
	{
		keep,
		zero,
		replace,
		increment_clamp,
		decrement_clamp,
		invert,
		increment_wrap,
		decrement_wrap,
	};

	enum class blend_factor : u8
	{
		zero,
		one,
		source_color,
		one_minus_source_color,
		source_alpha,
		one_minus_source_alpha,
		destination_color,
		one_minus_destination_color,
		destination_alpha,
		one_minus_destination_alpha,
		source_alpha_saturated,
		blend_color,
		one_minus_blend_color,
		blend_alpha,
		one_minus_blend_alpha,
		source1_color,
		one_minus_source1_color,
		source1_alpha,
		one_minus_source1_alpha,
	};

	enum class blend_operation : u8
	{
		add,
		subtract,
		reverse_subtract,
		minimum,
		maximum,
	};

	enum class logic_operation : u8
	{
		clear,
		and_,
		and_reverse,
		copy,
		and_inverted,
		no_op,
		xor_,
		or_,
		nor,
		equivalent,
		invert,
		or_reverse,
		copy_inverted,
		or_inverted,
		nand,
		set,
	};

	enum color_write_mask : u8
	{
		color_write_none = 0,
		color_write_red = 1 << 0,
		color_write_green = 1 << 1,
		color_write_blue = 1 << 2,
		color_write_alpha = 1 << 3,
		color_write_all = color_write_red | color_write_green | color_write_blue | color_write_alpha,
	};

	enum pipeline_emulation_flag : u32
	{
		pipeline_emulation_none = 0,
		pipeline_emulation_logic_operation = 1 << 0,
		pipeline_emulation_sample_mask = 1 << 1,
		pipeline_emulation_sample_shading = 1 << 2,
		pipeline_emulation_depth_bounds = 1 << 3,
		pipeline_emulation_primitive_restart = 1 << 4,
		pipeline_emulation_signed_blend = 1 << 5,
		pipeline_emulation_reverse_signed_blend = 1 << 6,
		pipeline_emulation_wide_lines = 1 << 7,
		pipeline_emulation_polygon_mode = 1 << 8,
	};

	struct shader_function_identity
	{
		u64 guest_program_hash = 0;
		u64 translated_source_hash = 0;
		u64 specialization_hash = 0;
		u64 compiler_generation = 0;

		[[nodiscard]] bool operator==(const shader_function_identity&) const = default;
		[[nodiscard]] explicit operator bool() const
		{
			return guest_program_hash != 0 && translated_source_hash != 0;
		}
	};

	struct color_attachment_pipeline_state
	{
		u64 pixel_format = 0;
		u8 write_mask = color_write_all;
		bool blend_enabled = false;
		blend_factor source_rgb = blend_factor::one;
		blend_factor destination_rgb = blend_factor::zero;
		blend_operation rgb_operation = blend_operation::add;
		blend_factor source_alpha = blend_factor::one;
		blend_factor destination_alpha = blend_factor::zero;
		blend_operation alpha_operation = blend_operation::add;

		[[nodiscard]] bool operator==(const color_attachment_pipeline_state&) const = default;
	};

	struct stencil_face_pipeline_state
	{
		compare_function compare = compare_function::always;
		stencil_operation stencil_fail = stencil_operation::keep;
		stencil_operation depth_fail = stencil_operation::keep;
		stencil_operation depth_pass = stencil_operation::keep;
		u32 read_mask = 0xff;
		u32 write_mask = 0xff;

		[[nodiscard]] bool operator==(const stencil_face_pipeline_state&) const = default;
	};

	struct depth_stencil_pipeline_state
	{
		u64 depth_pixel_format = 0;
		u64 stencil_pixel_format = 0;
		compare_function depth_compare = compare_function::always;
		bool depth_test_enabled = false;
		bool depth_write_enabled = false;
		bool stencil_test_enabled = false;
		bool depth_bounds_enabled = false;
		stencil_face_pipeline_state front;
		stencil_face_pipeline_state back;

		[[nodiscard]] bool operator==(const depth_stencil_pipeline_state&) const = default;
	};

	struct multisample_pipeline_state
	{
		u32 sample_count = 1;
		u32 sample_mask = 0xffffffff;
		f32 minimum_sample_shading = 0.f;
		bool multisampling_enabled = false;
		bool alpha_to_coverage = false;
		bool alpha_to_one = false;
		bool sample_shading_enabled = false;

		[[nodiscard]] bool operator==(const multisample_pipeline_state&) const = default;
	};

	struct render_pipeline_state
	{
		shader_function_identity vertex_function;
		shader_function_identity fragment_function;
		u64 vertex_layout_hash = 0;
		u64 binding_layout_signature = pipeline_binding_table::signature();
		std::array<color_attachment_pipeline_state, maximum_color_attachments> color_attachments{};
		u32 color_attachment_count = 0;
		primitive_topology topology = primitive_topology::triangle;
		multisample_pipeline_state multisample;
		bool rasterization_enabled = true;
		bool support_indirect_commands = false;
		bool logic_operation_enabled = false;
		logic_operation logic = logic_operation::copy;
		u32 emulation_flags = pipeline_emulation_none;

		[[nodiscard]] bool operator==(const render_pipeline_state&) const = default;
	};

	struct dynamic_pipeline_state
	{
		cull_mode cull = cull_mode::none;
		front_face winding = front_face::counter_clockwise;
		triangle_fill_mode fill = triangle_fill_mode::fill;
		depth_clip_mode depth_clip = depth_clip_mode::clip;
		f32 depth_bias = 0.f;
		f32 depth_bias_slope = 0.f;
		f32 depth_bias_clamp = 0.f;
		std::array<f32, 4> blend_color{};
		u32 stencil_front_reference = 0;
		u32 stencil_back_reference = 0;
		f32 minimum_depth_bounds = 0.f;
		f32 maximum_depth_bounds = 1.f;
		bool depth_bias_enabled = false;

		[[nodiscard]] bool operator==(const dynamic_pipeline_state&) const = default;
	};

	namespace detail
	{
		constexpr u64 hash_seed = 0x9e3779b97f4a7c15ull;

		inline void hash_combine(u64& hash, u64 value)
		{
			value += hash_seed + (hash << 6) + (hash >> 2);
			hash ^= value;
		}

		template <typename T>
		void hash_value(u64& hash, T value)
		{
			if constexpr (std::is_enum_v<T>)
			{
				hash_combine(hash, static_cast<u64>(value));
			}
			else if constexpr (std::is_same_v<T, f32>)
			{
				hash_combine(hash, std::bit_cast<u32>(value));
			}
			else
			{
				hash_combine(hash, static_cast<u64>(value));
			}
		}

		inline void hash_shader(u64& hash, const shader_function_identity& shader)
		{
			hash_value(hash, shader.guest_program_hash);
			hash_value(hash, shader.translated_source_hash);
			hash_value(hash, shader.specialization_hash);
			hash_value(hash, shader.compiler_generation);
		}

		inline void hash_attachment(u64& hash, const color_attachment_pipeline_state& attachment)
		{
			hash_value(hash, attachment.pixel_format);
			hash_value(hash, attachment.write_mask);
			hash_value(hash, attachment.blend_enabled);
			hash_value(hash, attachment.source_rgb);
			hash_value(hash, attachment.destination_rgb);
			hash_value(hash, attachment.rgb_operation);
			hash_value(hash, attachment.source_alpha);
			hash_value(hash, attachment.destination_alpha);
			hash_value(hash, attachment.alpha_operation);
		}

		inline void hash_stencil_face(u64& hash, const stencil_face_pipeline_state& face)
		{
			hash_value(hash, face.compare);
			hash_value(hash, face.stencil_fail);
			hash_value(hash, face.depth_fail);
			hash_value(hash, face.depth_pass);
			hash_value(hash, face.read_mask);
			hash_value(hash, face.write_mask);
		}
	}

	class graphics_pipeline_state
	{
	public:
		render_pipeline_state render;
		depth_stencil_pipeline_state depth_stencil;
		dynamic_pipeline_state dynamic;

		void set_primitive_type(primitive_topology type)
		{
			render.topology = type;
		}

		void enable_primitive_restart(bool enable = true)
		{
			if (enable) render.emulation_flags |= pipeline_emulation_primitive_restart;
			else render.emulation_flags &= ~pipeline_emulation_primitive_restart;
		}

		void set_color_mask(u32 index, bool red, bool green, bool blue, bool alpha)
		{
			if (index >= maximum_color_attachments)
			{
				fmt::throw_exception("Metal color attachment index %u is out of range", index);
			}
			u8 mask = color_write_none;
			if (red) mask |= color_write_red;
			if (green) mask |= color_write_green;
			if (blue) mask |= color_write_blue;
			if (alpha) mask |= color_write_alpha;
			render.color_attachments[index].write_mask = mask;
		}

		void set_depth_mask(bool enable)
		{
			depth_stencil.depth_write_enabled = enable;
		}

		void set_stencil_mask(u32 mask)
		{
			depth_stencil.front.write_mask = mask;
			depth_stencil.back.write_mask = mask;
		}

		void set_stencil_mask_separate(bool back_face, u32 mask)
		{
			(back_face ? depth_stencil.back : depth_stencil.front).write_mask = mask;
		}

		void enable_depth_test(compare_function operation)
		{
			depth_stencil.depth_test_enabled = true;
			depth_stencil.depth_compare = operation;
		}

		void disable_depth_test()
		{
			depth_stencil.depth_test_enabled = false;
			depth_stencil.depth_compare = compare_function::always;
		}

		void enable_depth_clamp(bool enable = true)
		{
			dynamic.depth_clip = enable ? depth_clip_mode::clamp : depth_clip_mode::clip;
		}

		void enable_depth_bias(bool enable = true)
		{
			dynamic.depth_bias_enabled = enable;
		}

		void enable_depth_bounds_test(bool enable = true)
		{
			depth_stencil.depth_bounds_enabled = enable;
			if (enable) render.emulation_flags |= pipeline_emulation_depth_bounds;
			else render.emulation_flags &= ~pipeline_emulation_depth_bounds;
		}

		void enable_blend(u32 attachment,
			blend_factor source_rgb, blend_factor source_alpha,
			blend_factor destination_rgb, blend_factor destination_alpha,
			blend_operation rgb_operation, blend_operation alpha_operation)
		{
			if (attachment >= maximum_color_attachments)
			{
				fmt::throw_exception("Metal blend attachment index %u is out of range", attachment);
			}
			auto& state = render.color_attachments[attachment];
			state.source_rgb = source_rgb;
			state.source_alpha = source_alpha;
			state.destination_rgb = destination_rgb;
			state.destination_alpha = destination_alpha;
			state.rgb_operation = rgb_operation;
			state.alpha_operation = alpha_operation;
			state.blend_enabled = true;
		}

		void disable_blend(u32 attachment)
		{
			if (attachment >= maximum_color_attachments)
			{
				fmt::throw_exception("Metal blend attachment index %u is out of range", attachment);
			}
			render.color_attachments[attachment].blend_enabled = false;
		}

		void enable_stencil_test(stencil_operation fail, stencil_operation depth_fail,
			stencil_operation pass, compare_function function, u32 function_mask, u32 reference)
		{
			stencil_face_pipeline_state state;
			state.stencil_fail = fail;
			state.depth_fail = depth_fail;
			state.depth_pass = pass;
			state.compare = function;
			state.read_mask = function_mask;
			state.write_mask = depth_stencil.front.write_mask;
			depth_stencil.front = state;
			depth_stencil.back = state;
			dynamic.stencil_front_reference = reference;
			dynamic.stencil_back_reference = reference;
			depth_stencil.stencil_test_enabled = true;
		}

		void enable_stencil_test_separate(bool back_face, stencil_operation fail,
			stencil_operation depth_fail, stencil_operation pass, compare_function function,
			u32 function_mask, u32 reference)
		{
			auto& state = back_face ? depth_stencil.back : depth_stencil.front;
			state.stencil_fail = fail;
			state.depth_fail = depth_fail;
			state.depth_pass = pass;
			state.compare = function;
			state.read_mask = function_mask;
			(back_face ? dynamic.stencil_back_reference : dynamic.stencil_front_reference) = reference;
			depth_stencil.stencil_test_enabled = true;
		}

		void enable_logic_op(logic_operation operation)
		{
			render.logic_operation_enabled = true;
			render.logic = operation;
			render.emulation_flags |= pipeline_emulation_logic_operation;
		}

		void disable_logic_op()
		{
			render.logic_operation_enabled = false;
			render.logic = logic_operation::copy;
			render.emulation_flags &= ~pipeline_emulation_logic_operation;
		}

		void enable_cull_face(cull_mode mode)
		{
			dynamic.cull = mode;
		}

		void set_front_face(front_face face)
		{
			dynamic.winding = face;
		}

		void set_attachment_count(u32 count)
		{
			if (count > maximum_color_attachments)
			{
				fmt::throw_exception("Metal color attachment count %u is out of range", count);
			}
			render.color_attachment_count = count;
		}

		void set_multisample_state(u32 sample_count, u32 sample_mask, bool enabled,
			bool alpha_to_coverage, bool alpha_to_one)
		{
			if (sample_count != 1 && sample_count != 2 && sample_count != 4 && sample_count != 8)
			{
				fmt::throw_exception("Metal sample count %u is invalid", sample_count);
			}
			render.multisample.sample_count = sample_count;
			render.multisample.sample_mask = sample_mask;
			render.multisample.multisampling_enabled = enabled;
			render.multisample.alpha_to_coverage = alpha_to_coverage;
			render.multisample.alpha_to_one = alpha_to_one;
			if (sample_mask != 0xffffffff) render.emulation_flags |= pipeline_emulation_sample_mask;
			else render.emulation_flags &= ~pipeline_emulation_sample_mask;
		}

		void set_multisample_shading_rate(f32 shading_rate)
		{
			if (!(shading_rate >= 0.f && shading_rate <= 1.f))
			{
				fmt::throw_exception("Metal sample shading rate must be between zero and one");
			}
			render.multisample.sample_shading_enabled = true;
			render.multisample.minimum_sample_shading = shading_rate;
			render.emulation_flags |= pipeline_emulation_sample_shading;
		}

		void validate() const
		{
			if (!render.vertex_function || render.color_attachment_count > maximum_color_attachments)
			{
				fmt::throw_exception("Metal graphics pipeline is missing a valid vertex function or attachment count");
			}
			if (render.rasterization_enabled && !render.fragment_function && render.color_attachment_count)
			{
				fmt::throw_exception("Metal graphics pipeline with color output requires a fragment function");
			}
			for (u32 index = 0; index < render.color_attachment_count; ++index)
			{
				if (!render.color_attachments[index].pixel_format)
				{
					fmt::throw_exception("Metal graphics pipeline attachment %u has no pixel format", index);
				}
			}
			if ((depth_stencil.depth_test_enabled || depth_stencil.depth_write_enabled) && !depth_stencil.depth_pixel_format)
			{
				fmt::throw_exception("Metal depth state requires a depth attachment format");
			}
			if (depth_stencil.stencil_test_enabled && !depth_stencil.stencil_pixel_format)
			{
				fmt::throw_exception("Metal stencil state requires a stencil attachment format");
			}
		}

		[[nodiscard]] u64 render_pipeline_hash() const
		{
			u64 hash = detail::hash_seed;
			detail::hash_shader(hash, render.vertex_function);
			detail::hash_shader(hash, render.fragment_function);
			detail::hash_value(hash, render.vertex_layout_hash);
			detail::hash_value(hash, render.binding_layout_signature);
			detail::hash_value(hash, render.color_attachment_count);
			for (u32 index = 0; index < render.color_attachment_count; ++index)
			{
				detail::hash_attachment(hash, render.color_attachments[index]);
			}
			detail::hash_value(hash, render.topology);
			detail::hash_value(hash, render.multisample.sample_count);
			detail::hash_value(hash, render.multisample.sample_mask);
			detail::hash_value(hash, render.multisample.minimum_sample_shading);
			detail::hash_value(hash, render.multisample.multisampling_enabled);
			detail::hash_value(hash, render.multisample.alpha_to_coverage);
			detail::hash_value(hash, render.multisample.alpha_to_one);
			detail::hash_value(hash, render.multisample.sample_shading_enabled);
			detail::hash_value(hash, render.rasterization_enabled);
			detail::hash_value(hash, render.support_indirect_commands);
			detail::hash_value(hash, render.logic_operation_enabled);
			detail::hash_value(hash, render.logic);
			detail::hash_value(hash, render.emulation_flags);
			detail::hash_value(hash, depth_stencil.depth_pixel_format);
			detail::hash_value(hash, depth_stencil.stencil_pixel_format);
			return hash;
		}

		[[nodiscard]] u64 depth_stencil_hash() const
		{
			u64 hash = detail::hash_seed;
			detail::hash_value(hash, depth_stencil.depth_compare);
			detail::hash_value(hash, depth_stencil.depth_test_enabled);
			detail::hash_value(hash, depth_stencil.depth_write_enabled);
			detail::hash_value(hash, depth_stencil.stencil_test_enabled);
			detail::hash_value(hash, depth_stencil.depth_bounds_enabled);
			detail::hash_stencil_face(hash, depth_stencil.front);
			detail::hash_stencil_face(hash, depth_stencil.back);
			return hash;
		}

		[[nodiscard]] u64 dynamic_state_hash() const
		{
			u64 hash = detail::hash_seed;
			detail::hash_value(hash, dynamic.cull);
			detail::hash_value(hash, dynamic.winding);
			detail::hash_value(hash, dynamic.fill);
			detail::hash_value(hash, dynamic.depth_clip);
			detail::hash_value(hash, dynamic.depth_bias);
			detail::hash_value(hash, dynamic.depth_bias_slope);
			detail::hash_value(hash, dynamic.depth_bias_clamp);
			for (f32 value : dynamic.blend_color) detail::hash_value(hash, value);
			detail::hash_value(hash, dynamic.stencil_front_reference);
			detail::hash_value(hash, dynamic.stencil_back_reference);
			detail::hash_value(hash, dynamic.minimum_depth_bounds);
			detail::hash_value(hash, dynamic.maximum_depth_bounds);
			detail::hash_value(hash, dynamic.depth_bias_enabled);
			return hash;
		}

		[[nodiscard]] u64 pipeline_cache_hash() const
		{
			u64 hash = render_pipeline_hash();
			detail::hash_combine(hash, depth_stencil_hash());
			return hash;
		}

		[[nodiscard]] bool pipeline_cache_equal(const graphics_pipeline_state& other) const
		{
			if (render.vertex_function != other.render.vertex_function ||
				render.fragment_function != other.render.fragment_function ||
				render.vertex_layout_hash != other.render.vertex_layout_hash ||
				render.binding_layout_signature != other.render.binding_layout_signature ||
				render.color_attachment_count != other.render.color_attachment_count ||
				render.topology != other.render.topology ||
				render.multisample != other.render.multisample ||
				render.rasterization_enabled != other.render.rasterization_enabled ||
				render.support_indirect_commands != other.render.support_indirect_commands ||
				render.logic_operation_enabled != other.render.logic_operation_enabled ||
				render.logic != other.render.logic ||
				render.emulation_flags != other.render.emulation_flags ||
				depth_stencil != other.depth_stencil)
			{
				return false;
			}
			for (u32 index = 0; index < render.color_attachment_count; ++index)
			{
				if (render.color_attachments[index] != other.render.color_attachments[index]) return false;
			}
			return true;
		}

		[[nodiscard]] bool operator==(const graphics_pipeline_state&) const = default;
	};

	struct graphics_pipeline_state_hash
	{
		[[nodiscard]] usz operator()(const graphics_pipeline_state& state) const noexcept
		{
			return static_cast<usz>(state.pipeline_cache_hash());
		}
	};

	struct graphics_pipeline_cache_equal
	{
		[[nodiscard]] bool operator()(const graphics_pipeline_state& left,
			const graphics_pipeline_state& right) const noexcept
		{
			return left.pipeline_cache_equal(right);
		}
	};
}
