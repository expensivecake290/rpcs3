#pragma once

#include "MTLResourceState.h"
#include "util/types.hpp"

#include <memory>
#include <string>

namespace rsx::metal
{
	class buffer;
	class command_frame;
	class device;
	class sampler;
	struct shader_interface_layout;
	class texture;

	enum class argument_table_render_stage : u32
	{
		vertex = 1u << 0,
		fragment = 1u << 1,
		tile = 1u << 2,
		object = 1u << 3,
		mesh = 1u << 4,
	};

	struct argument_table_desc
	{
		std::string label;
		u32 max_buffers = 0;
		u32 max_textures = 0;
		u32 max_samplers = 0;
		b8 initialize_bindings = true;
		b8 support_attribute_strides = false;
	};

	class argument_table
	{
	public:
		argument_table(device& dev, const argument_table_desc& desc);
		~argument_table();

		argument_table(const argument_table&) = delete;
		argument_table& operator=(const argument_table&) = delete;

		void* handle() const;
		u32 max_buffers() const;
		u32 max_textures() const;
		u32 max_samplers() const;
		b8 supports_attribute_strides() const;

		void bind_buffer_address(u32 index, const buffer& buf, u64 offset = 0, resource_access access = resource_access::read);
		void bind_vertex_buffer_address(u32 index, const buffer& buf, u64 offset, u32 stride);
		void bind_texture(u32 index, const texture& tex, resource_access access = resource_access::read);
		void bind_sampler(u32 index, const sampler& sampler_state);

		void validate_shader_bindings(const shader_interface_layout& layout) const;
		void bind_to_render_encoder(command_frame& frame, void* render_encoder_handle, const shader_interface_layout& layout) const;
		void bind_to_render_encoder(command_frame& frame, void* render_encoder_handle, u32 stages) const;
		void bind_to_compute_encoder(command_frame& frame, void* compute_encoder_handle) const;

	private:
		void validate_shader_bindings_locked(const shader_interface_layout& layout) const;
		void validate_bound_resource_conflicts(resource_stage stage) const;
		void track_bound_resources(command_frame& frame, void* encoder_handle, resource_stage stage) const;
		void retain_bound_table_locked(command_frame& frame) const;

		struct argument_table_impl;
		std::unique_ptr<argument_table_impl> m_impl;
	};
}
