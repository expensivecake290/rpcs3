#include "stdafx.h"
#include "MTLRenderState.h"

#include "MTLCommandBuffer.h"
#include "MTLDevice.h"

#import <Metal/Metal.h>

#include <limits>
#include <mutex>
#include <vector>

namespace
{
	NSString* make_ns_string(const std::string& value)
	{
		return [NSString stringWithUTF8String:value.c_str()];
	}

	b8 is_line_primitive(rsx::primitive_type primitive)
	{
		rsx_log.trace("is_line_primitive(primitive=0x%x)", static_cast<u32>(primitive));

		switch (primitive)
		{
		case rsx::primitive_type::lines:
		case rsx::primitive_type::line_strip:
			return true;
		default:
			return false;
		}
	}

	b8 is_triangle_primitive(rsx::primitive_type primitive)
	{
		rsx_log.trace("is_triangle_primitive(primitive=0x%x)", static_cast<u32>(primitive));

		switch (primitive)
		{
		case rsx::primitive_type::triangles:
		case rsx::primitive_type::triangle_strip:
			return true;
		default:
			return false;
		}
	}

	MTLWinding make_front_face(rsx::front_face front_face)
	{
		rsx_log.trace("make_front_face(front_face=%u)", static_cast<u32>(front_face));

		switch (front_face)
		{
		case rsx::front_face::cw:
			return MTLWindingClockwise;
		case rsx::front_face::ccw:
			return MTLWindingCounterClockwise;
		}

		fmt::throw_exception("Unsupported Metal front face winding: %u", static_cast<u32>(front_face));
	}

	MTLCullMode make_cull_mode(const rsx::metal::dynamic_render_state_desc& desc)
	{
		rsx_log.trace("make_cull_mode(enabled=%u, cull_face=%u)",
			static_cast<u32>(desc.cull_face_enabled),
			static_cast<u32>(desc.cull_face));

		if (!desc.cull_face_enabled)
		{
			return MTLCullModeNone;
		}

		switch (desc.cull_face)
		{
		case rsx::cull_face::front:
			return MTLCullModeFront;
		case rsx::cull_face::back:
			return MTLCullModeBack;
		case rsx::cull_face::front_and_back:
			rsx_log.todo("Metal front-and-back culling requires a draw suppression path");
			break;
		}

		fmt::throw_exception("Unsupported Metal cull face mode: %u", static_cast<u32>(desc.cull_face));
	}

	MTLTriangleFillMode make_triangle_fill_mode(const rsx::metal::dynamic_render_state_desc& desc)
	{
		rsx_log.trace("make_triangle_fill_mode(front=%u, back=%u)",
			static_cast<u32>(desc.front_polygon_mode),
			static_cast<u32>(desc.back_polygon_mode));

		if (desc.front_polygon_mode != desc.back_polygon_mode)
		{
			rsx_log.todo("Metal split front/back polygon modes are not implemented");
			fmt::throw_exception("Metal backend split front/back polygon modes are not implemented yet");
		}

		switch (desc.front_polygon_mode)
		{
		case rsx::polygon_mode::fill:
			return MTLTriangleFillModeFill;
		case rsx::polygon_mode::line:
			return MTLTriangleFillModeLines;
		case rsx::polygon_mode::point:
			rsx_log.todo("Metal polygon point mode requires a GPU-only expansion path");
			break;
		}

		fmt::throw_exception("Unsupported Metal polygon mode: %u", static_cast<u32>(desc.front_polygon_mode));
	}

	MTLDepthClipMode make_depth_clip_mode(const rsx::metal::dynamic_render_state_desc& desc)
	{
		rsx_log.trace("make_depth_clip_mode(depth_clip=%u, depth_clamp=%u)",
			static_cast<u32>(desc.depth_clip_enabled),
			static_cast<u32>(desc.depth_clamp_enabled));

		return (desc.depth_clamp_enabled || !desc.depth_clip_enabled) ? MTLDepthClipModeClamp : MTLDepthClipModeClip;
	}

	MTLCompareFunction make_compare_function(rsx::comparison_function func)
	{
		rsx_log.trace("make_compare_function(func=%u)", static_cast<u32>(func));

		switch (func)
		{
		case rsx::comparison_function::never:
			return MTLCompareFunctionNever;
		case rsx::comparison_function::less:
			return MTLCompareFunctionLess;
		case rsx::comparison_function::equal:
			return MTLCompareFunctionEqual;
		case rsx::comparison_function::less_or_equal:
			return MTLCompareFunctionLessEqual;
		case rsx::comparison_function::greater:
			return MTLCompareFunctionGreater;
		case rsx::comparison_function::not_equal:
			return MTLCompareFunctionNotEqual;
		case rsx::comparison_function::greater_or_equal:
			return MTLCompareFunctionGreaterEqual;
		case rsx::comparison_function::always:
			return MTLCompareFunctionAlways;
		}

		fmt::throw_exception("Unsupported Metal comparison function: %u", static_cast<u32>(func));
	}

	MTLStencilOperation make_stencil_operation(rsx::stencil_op op)
	{
		rsx_log.trace("make_stencil_operation(op=%u)", static_cast<u32>(op));

		switch (op)
		{
		case rsx::stencil_op::keep:
			return MTLStencilOperationKeep;
		case rsx::stencil_op::zero:
			return MTLStencilOperationZero;
		case rsx::stencil_op::replace:
			return MTLStencilOperationReplace;
		case rsx::stencil_op::incr:
			return MTLStencilOperationIncrementClamp;
		case rsx::stencil_op::decr:
			return MTLStencilOperationDecrementClamp;
		case rsx::stencil_op::invert:
			return MTLStencilOperationInvert;
		case rsx::stencil_op::incr_wrap:
			return MTLStencilOperationIncrementWrap;
		case rsx::stencil_op::decr_wrap:
			return MTLStencilOperationDecrementWrap;
		}

		fmt::throw_exception("Unsupported Metal stencil operation: %u", static_cast<u32>(op));
	}

	struct depth_stencil_state_key
	{
		u32 depth_func = 0;
		u32 stencil_func = 0;
		u32 back_stencil_func = 0;
		u32 stencil_fail = 0;
		u32 stencil_zfail = 0;
		u32 stencil_zpass = 0;
		u32 back_stencil_fail = 0;
		u32 back_stencil_zfail = 0;
		u32 back_stencil_zpass = 0;
		u32 stencil_read_mask = 0;
		u32 back_stencil_read_mask = 0;
		u32 stencil_write_mask = 0;
		u32 back_stencil_write_mask = 0;
		b8 depth_test_enabled = false;
		b8 depth_write_enabled = false;
		b8 stencil_test_enabled = false;
		b8 two_sided_stencil_test_enabled = false;
	};

	bool operator==(const depth_stencil_state_key& lhs, const depth_stencil_state_key& rhs)
	{
		return lhs.depth_func == rhs.depth_func &&
			lhs.stencil_func == rhs.stencil_func &&
			lhs.back_stencil_func == rhs.back_stencil_func &&
			lhs.stencil_fail == rhs.stencil_fail &&
			lhs.stencil_zfail == rhs.stencil_zfail &&
			lhs.stencil_zpass == rhs.stencil_zpass &&
			lhs.back_stencil_fail == rhs.back_stencil_fail &&
			lhs.back_stencil_zfail == rhs.back_stencil_zfail &&
			lhs.back_stencil_zpass == rhs.back_stencil_zpass &&
			lhs.stencil_read_mask == rhs.stencil_read_mask &&
			lhs.back_stencil_read_mask == rhs.back_stencil_read_mask &&
			lhs.stencil_write_mask == rhs.stencil_write_mask &&
			lhs.back_stencil_write_mask == rhs.back_stencil_write_mask &&
			lhs.depth_test_enabled == rhs.depth_test_enabled &&
			lhs.depth_write_enabled == rhs.depth_write_enabled &&
			lhs.stencil_test_enabled == rhs.stencil_test_enabled &&
			lhs.two_sided_stencil_test_enabled == rhs.two_sided_stencil_test_enabled;
	}

	depth_stencil_state_key make_depth_stencil_key(const rsx::metal::dynamic_render_state_desc& desc)
	{
		rsx_log.trace("make_depth_stencil_key(depth_test=%u, depth_write=%u, stencil_test=%u, two_sided=%u)",
			static_cast<u32>(desc.depth_test_enabled),
			static_cast<u32>(desc.depth_write_enabled),
			static_cast<u32>(desc.stencil_test_enabled),
			static_cast<u32>(desc.two_sided_stencil_test_enabled));

		return
		{
			.depth_func = static_cast<u32>(desc.depth_func),
			.stencil_func = static_cast<u32>(desc.stencil_func),
			.back_stencil_func = static_cast<u32>(desc.back_stencil_func),
			.stencil_fail = static_cast<u32>(desc.stencil_fail),
			.stencil_zfail = static_cast<u32>(desc.stencil_zfail),
			.stencil_zpass = static_cast<u32>(desc.stencil_zpass),
			.back_stencil_fail = static_cast<u32>(desc.back_stencil_fail),
			.back_stencil_zfail = static_cast<u32>(desc.back_stencil_zfail),
			.back_stencil_zpass = static_cast<u32>(desc.back_stencil_zpass),
			.stencil_read_mask = desc.stencil_read_mask,
			.back_stencil_read_mask = desc.back_stencil_read_mask,
			.stencil_write_mask = desc.stencil_write_mask,
			.back_stencil_write_mask = desc.back_stencil_write_mask,
			.depth_test_enabled = desc.depth_test_enabled,
			.depth_write_enabled = desc.depth_write_enabled,
			.stencil_test_enabled = desc.stencil_test_enabled,
			.two_sided_stencil_test_enabled = desc.two_sided_stencil_test_enabled,
		};
	}

	void configure_stencil_descriptor(
		MTLStencilDescriptor* stencil,
		rsx::comparison_function compare,
		rsx::stencil_op fail,
		rsx::stencil_op zfail,
		rsx::stencil_op zpass,
		u32 read_mask,
		u32 write_mask)
	{
		rsx_log.trace("configure_stencil_descriptor(compare=%u, fail=%u, zfail=%u, zpass=%u, read_mask=0x%x, write_mask=0x%x)",
			static_cast<u32>(compare),
			static_cast<u32>(fail),
			static_cast<u32>(zfail),
			static_cast<u32>(zpass),
			read_mask,
			write_mask);

		stencil.stencilCompareFunction = make_compare_function(compare);
		stencil.stencilFailureOperation = make_stencil_operation(fail);
		stencil.depthFailureOperation = make_stencil_operation(zfail);
		stencil.depthStencilPassOperation = make_stencil_operation(zpass);
		stencil.readMask = read_mask;
		stencil.writeMask = write_mask;
	}

	void validate_dynamic_render_state(const rsx::metal::dynamic_render_state_desc& desc)
	{
		rsx_log.trace("validate_dynamic_render_state(render=%ux%u, viewport=%fx%f, scissor=%u,%u %ux%u)",
			desc.render_width,
			desc.render_height,
			desc.viewport.width,
			desc.viewport.height,
			desc.scissor.x,
			desc.scissor.y,
			desc.scissor.width,
			desc.scissor.height);

		if (!desc.render_width || !desc.render_height)
		{
			fmt::throw_exception("Metal render state requires a non-zero render target size");
		}

		if (desc.viewport.width <= 0. || desc.viewport.height <= 0.)
		{
			fmt::throw_exception("Metal render state requires a non-zero viewport");
		}

		if (!desc.scissor.width || !desc.scissor.height)
		{
			fmt::throw_exception("Metal render state requires a non-zero scissor");
		}

		const u64 scissor_right = static_cast<u64>(desc.scissor.x) + desc.scissor.width;
		const u64 scissor_bottom = static_cast<u64>(desc.scissor.y) + desc.scissor.height;
		if (scissor_right > desc.render_width || scissor_bottom > desc.render_height)
		{
			fmt::throw_exception("Metal scissor is outside the render target: scissor=%u,%u %ux%u target=%ux%u",
				desc.scissor.x,
				desc.scissor.y,
				desc.scissor.width,
				desc.scissor.height,
				desc.render_width,
				desc.render_height);
		}

		if (desc.logic_op_enabled)
		{
			rsx_log.todo("Metal logic operation pipeline variants are not implemented");
			fmt::throw_exception("Metal backend logic operation pipeline variants are not implemented yet");
		}

		if (desc.alpha_test_enabled)
		{
			rsx_log.todo("Metal alpha test shader lowering is not implemented");
			fmt::throw_exception("Metal backend alpha test shader lowering is not implemented yet");
		}

		if (desc.dither_enabled)
		{
			rsx_log.todo("Metal dither state is not implemented");
			fmt::throw_exception("Metal backend dither state is not implemented yet");
		}

		if (desc.line_smooth_enabled || desc.poly_smooth_enabled)
		{
			rsx_log.todo("Metal line/polygon smoothing state is not implemented");
			fmt::throw_exception("Metal backend line/polygon smoothing state is not implemented yet");
		}

		if (desc.polygon_stipple_enabled)
		{
			rsx_log.todo("Metal polygon stipple shader path is not implemented");
			fmt::throw_exception("Metal backend polygon stipple shader path is not implemented yet");
		}

		if (desc.depth_test_enabled && !desc.has_depth_stencil_target)
		{
			fmt::throw_exception("Metal depth test requires a depth target");
		}

		if (desc.depth_write_enabled && !desc.has_depth_stencil_target)
		{
			fmt::throw_exception("Metal depth write requires a depth target");
		}

		if (desc.stencil_test_enabled && !desc.has_stencil_attachment)
		{
			fmt::throw_exception("Metal stencil test requires a stencil attachment");
		}

		if (desc.depth_bounds_test_enabled)
		{
			if (desc.depth_bounds_min < 0.f || desc.depth_bounds_min > 1.f ||
				desc.depth_bounds_max < 0.f || desc.depth_bounds_max > 1.f ||
				desc.depth_bounds_min > desc.depth_bounds_max)
			{
				fmt::throw_exception("Metal depth bounds must be inside [0, 1] and ordered: min=%f, max=%f",
					desc.depth_bounds_min,
					desc.depth_bounds_max);
			}
		}

		if (desc.cull_face_enabled && desc.cull_face == rsx::cull_face::front_and_back)
		{
			rsx_log.todo("Metal front-and-back culling requires a draw suppression path");
			fmt::throw_exception("Metal backend front-and-back culling is not implemented yet");
		}

		if (desc.front_polygon_mode != desc.back_polygon_mode)
		{
			rsx_log.todo("Metal split polygon modes are not implemented");
			fmt::throw_exception("Metal backend split polygon modes are not implemented yet");
		}

		if (desc.front_polygon_mode == rsx::polygon_mode::point)
		{
			rsx_log.todo("Metal polygon point mode requires GPU-only expansion");
			fmt::throw_exception("Metal backend polygon point mode is not implemented yet");
		}

		if (is_line_primitive(desc.primitive) && desc.line_width != 1.f)
		{
			rsx_log.todo("Metal wide line rasterization is not implemented");
			fmt::throw_exception("Metal backend wide line rasterization is not implemented yet");
		}

		if (desc.polygon_offset_point_enabled && desc.primitive == rsx::primitive_type::points)
		{
			rsx_log.todo("Metal polygon offset for points is not implemented");
			fmt::throw_exception("Metal backend polygon offset for points is not implemented yet");
		}

		if (desc.polygon_offset_line_enabled && is_line_primitive(desc.primitive))
		{
			rsx_log.todo("Metal polygon offset for lines is not implemented");
			fmt::throw_exception("Metal backend polygon offset for lines is not implemented yet");
		}
	}
}

namespace rsx::metal
{
	struct depth_stencil_state_record
	{
		depth_stencil_state_key key{};
		id<MTLDepthStencilState> state = nil;
	};

	struct render_state_cache::render_state_cache_impl
	{
		device& m_device;
		std::vector<depth_stencil_state_record> m_depth_stencil_states;
		mutable std::mutex m_mutex;

		explicit render_state_cache_impl(device& dev)
			: m_device(dev)
		{
		}
	};

	render_state_cache::render_state_cache(device& dev)
		: m_impl(std::make_unique<render_state_cache_impl>(dev))
	{
		rsx_log.notice("rsx::metal::render_state_cache::render_state_cache(device=*0x%x)", dev.handle());
	}

	render_state_cache::~render_state_cache()
	{
		rsx_log.notice("rsx::metal::render_state_cache::~render_state_cache(retained_depth_stencil=%u)",
			retained_depth_stencil_state_count());
	}

	void* create_depth_stencil_state(device& dev, const dynamic_render_state_desc& desc)
	{
		rsx_log.trace("create_depth_stencil_state(device=*0x%x, depth_test=%u, depth_write=%u, stencil_test=%u)",
			dev.handle(),
			static_cast<u32>(desc.depth_test_enabled),
			static_cast<u32>(desc.depth_write_enabled),
			static_cast<u32>(desc.stencil_test_enabled));

		if (@available(macOS 26.0, *))
		{
			MTLDepthStencilDescriptor* state_desc = [MTLDepthStencilDescriptor new];
			state_desc.label = make_ns_string("RPCS3 Metal depth/stencil state");
			state_desc.depthCompareFunction = desc.depth_test_enabled ? make_compare_function(desc.depth_func) : MTLCompareFunctionAlways;
			state_desc.depthWriteEnabled = desc.depth_write_enabled;

			if (desc.stencil_test_enabled)
			{
				MTLStencilDescriptor* front = [MTLStencilDescriptor new];
				configure_stencil_descriptor(front,
					desc.stencil_func,
					desc.stencil_fail,
					desc.stencil_zfail,
					desc.stencil_zpass,
					desc.stencil_read_mask,
					desc.stencil_write_mask);
				state_desc.frontFaceStencil = front;

				MTLStencilDescriptor* back = [MTLStencilDescriptor new];
				if (desc.two_sided_stencil_test_enabled)
				{
					configure_stencil_descriptor(back,
						desc.back_stencil_func,
						desc.back_stencil_fail,
						desc.back_stencil_zfail,
						desc.back_stencil_zpass,
						desc.back_stencil_read_mask,
						desc.back_stencil_write_mask);
				}
				else
				{
					configure_stencil_descriptor(back,
						desc.stencil_func,
						desc.stencil_fail,
						desc.stencil_zfail,
						desc.stencil_zpass,
						desc.stencil_read_mask,
						desc.stencil_write_mask);
				}
				state_desc.backFaceStencil = back;
			}

			id<MTLDevice> metal_device = (__bridge id<MTLDevice>)dev.handle();
			id<MTLDepthStencilState> state = [metal_device newDepthStencilStateWithDescriptor:state_desc];
			if (!state)
			{
				fmt::throw_exception("Metal depth/stencil state creation failed");
			}

			return (__bridge_retained void*)state;
		}

		fmt::throw_exception("Metal depth/stencil state creation requires macOS 26.0 or newer");
	}

	void render_state_cache::bind_dynamic_render_state(command_frame& frame, void* render_encoder_handle, const dynamic_render_state_desc& desc)
	{
		rsx_log.trace("rsx::metal::render_state_cache::bind_dynamic_render_state(frame_index=%u, encoder=*0x%x, render=%ux%u, depth=%u, stencil=%u)",
			frame.frame_index(),
			render_encoder_handle,
			desc.render_width,
			desc.render_height,
			static_cast<u32>(desc.has_depth_stencil_target),
			static_cast<u32>(desc.has_stencil_attachment));

		if (!render_encoder_handle)
		{
			fmt::throw_exception("Metal render state binding requires a valid render encoder");
		}

		validate_dynamic_render_state(desc);

		if (@available(macOS 26.0, *))
		{
			id<MTL4RenderCommandEncoder> encoder = (__bridge id<MTL4RenderCommandEncoder>)render_encoder_handle;
			const MTLViewport viewport =
			{
				.originX = desc.viewport.x,
				.originY = desc.viewport.y,
				.width = desc.viewport.width,
				.height = desc.viewport.height,
				.znear = desc.viewport.z_near,
				.zfar = desc.viewport.z_far,
			};
			const MTLScissorRect scissor =
			{
				.x = desc.scissor.x,
				.y = desc.scissor.y,
				.width = desc.scissor.width,
				.height = desc.scissor.height,
			};

			[encoder setViewport:viewport];
			[encoder setScissorRect:scissor];
			[encoder setFrontFacingWinding:make_front_face(desc.front_face)];
			[encoder setCullMode:make_cull_mode(desc)];
			[encoder setDepthClipMode:make_depth_clip_mode(desc)];
			[encoder setTriangleFillMode:make_triangle_fill_mode(desc)];
			if (desc.blend_enabled)
			{
				[encoder setBlendColorRed:desc.blend_color_red
					green:desc.blend_color_green
					blue:desc.blend_color_blue
					alpha:desc.blend_color_alpha];
			}

			if (desc.depth_bounds_test_enabled)
			{
				[encoder setDepthTestMinBound:desc.depth_bounds_min maxBound:desc.depth_bounds_max];
			}
			else
			{
				[encoder setDepthTestMinBound:0.f maxBound:1.f];
			}

			if (desc.polygon_offset_fill_enabled && is_triangle_primitive(desc.primitive))
			{
				[encoder setDepthBias:desc.polygon_offset_bias slopeScale:desc.polygon_offset_scale clamp:0.f];
			}
			else
			{
				[encoder setDepthBias:0.f slopeScale:0.f clamp:0.f];
			}

			std::lock_guard lock(m_impl->m_mutex);
			if (desc.depth_test_enabled || desc.depth_write_enabled || desc.stencil_test_enabled)
			{
				id<MTLDepthStencilState> state = (__bridge id<MTLDepthStencilState>)get_depth_stencil_state_handle(desc);
				[encoder setDepthStencilState:state];
				frame.track_object((__bridge void*)state);

				if (desc.stencil_test_enabled)
				{
					[encoder setStencilFrontReferenceValue:desc.stencil_ref backReferenceValue:desc.two_sided_stencil_test_enabled ? desc.back_stencil_ref : desc.stencil_ref];
				}
			}
			else
			{
				[encoder setDepthStencilState:nil];
			}

			return;
		}

		fmt::throw_exception("Metal render state binding requires macOS 26.0 or newer");
	}

	void* render_state_cache::get_depth_stencil_state_handle(const dynamic_render_state_desc& desc)
	{
		rsx_log.trace("rsx::metal::render_state_cache::get_depth_stencil_state_handle(retained=%u)",
			static_cast<u32>(m_impl->m_depth_stencil_states.size()));

		const depth_stencil_state_key key = make_depth_stencil_key(desc);

		for (const depth_stencil_state_record& record : m_impl->m_depth_stencil_states)
		{
			if (record.key == key)
			{
				return (__bridge void*)record.state;
			}
		}

		void* state_handle = create_depth_stencil_state(m_impl->m_device, desc);
		id<MTLDepthStencilState> state = (__bridge_transfer id<MTLDepthStencilState>)state_handle;
		m_impl->m_depth_stencil_states.emplace_back(depth_stencil_state_record
		{
			.key = key,
			.state = state,
		});

		return (__bridge void*)state;
	}

	u32 render_state_cache::retained_depth_stencil_state_count() const
	{
		rsx_log.trace("rsx::metal::render_state_cache::retained_depth_stencil_state_count()");

		std::lock_guard lock(m_impl->m_mutex);
		if (m_impl->m_depth_stencil_states.size() > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal retained depth/stencil state count overflow");
		}

		return static_cast<u32>(m_impl->m_depth_stencil_states.size());
	}
}
