#pragma once

#include <memory>

#include "mtlutils/commands.h"
#include "mtlutils/image.h"

namespace mtl
{
	class MTLPipelineCompiler;
	class render_device;

	enum class resolve_direction : u8
	{
		multisample_to_expanded,
		expanded_to_multisample,
	};

	enum class resolve_execution_path : u8
	{
		color_compute,
		color_render,
		depth_render,
		stencil_render,
		depth_stencil_render,
	};

	struct resolve_sample_grid
	{
		u8 x = 1;
		u8 y = 1;

		[[nodiscard]] u32 count() const;
		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] bool operator==(const resolve_sample_grid&) const = default;
		[[nodiscard]] static resolve_sample_grid from_sample_count(u32 samples);
	};

	struct resolve_subresource
	{
		u32 mip_level = 0;
		u32 array_slice = 0;
		u32 origin_x = 0;
		u32 origin_y = 0;
		u32 width = 0;
		u32 height = 0;
		u32 layer_count = 1;

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] bool operator==(const resolve_subresource&) const = default;
	};

	struct resolve_request
	{
		image* multisampled = nullptr;
		image* expanded = nullptr;
		resolve_direction direction = resolve_direction::multisample_to_expanded;
		resolve_subresource multisampled_region;
		resolve_subresource expanded_region;
		u8 aspects = texture_aspect_none;
		u8 stencil_initial_value = 0;
		bool stencil_contents_initialized = true;

		void validate() const;
		[[nodiscard]] resolve_sample_grid sample_grid() const;
		[[nodiscard]] resolve_execution_path execution_path() const;
	};

	struct resolve_helper_statistics
	{
		u64 color_resolves = 0;
		u64 color_unresolves = 0;
		u64 depth_resolves = 0;
		u64 depth_unresolves = 0;
		u64 stencil_resolves = 0;
		u64 stencil_unresolves = 0;
		u64 combined_depth_stencil_passes = 0;
		u64 native_pipelines = 0;
		u64 pipeline_cache_hits = 0;
		u64 transient_bindings = 0;
	};

	class color_resolve_helper final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		color_resolve_helper();
		~color_resolve_helper();
		color_resolve_helper(const color_resolve_helper&) = delete;
		color_resolve_helper& operator=(const color_resolve_helper&) = delete;

		void initialize(render_device& device, MTLPipelineCompiler& compiler);
		void run(command_buffer& command, const resolve_request& request);
		void destroy();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] resolve_helper_statistics statistics() const;
	};

	class depth_stencil_resolve_helper final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		depth_stencil_resolve_helper();
		~depth_stencil_resolve_helper();
		depth_stencil_resolve_helper(const depth_stencil_resolve_helper&) = delete;
		depth_stencil_resolve_helper& operator=(const depth_stencil_resolve_helper&) = delete;

		void initialize(render_device& device, MTLPipelineCompiler& compiler);
		void run(command_buffer& command, const resolve_request& request);
		void destroy();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] resolve_helper_statistics statistics() const;
	};

	class resolve_helper final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		resolve_helper();
		~resolve_helper();
		resolve_helper(const resolve_helper&) = delete;
		resolve_helper& operator=(const resolve_helper&) = delete;

		void initialize(render_device& device, MTLPipelineCompiler& compiler);
		void resolve(command_buffer& command, image& destination, image& source,
			u8 aspects = texture_aspect_none);
		void unresolve(command_buffer& command, image& destination, image& source,
			u8 aspects = texture_aspect_none);
		void run(command_buffer& command, const resolve_request& request);
		void destroy();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] resolve_helper_statistics statistics() const;
	};

	[[nodiscard]] resolve_helper& get_resolve_helper();
	void initialize_resolve_helpers(render_device& device, MTLPipelineCompiler& compiler);
	void resolve_image(command_buffer& command, image& destination, image& source,
		u8 aspects = texture_aspect_none);
	void unresolve_image(command_buffer& command, image& destination, image& source,
		u8 aspects = texture_aspect_none);
	void clear_resolve_helpers();
}
