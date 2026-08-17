#pragma once

#include "Emu/IdManager.h"
#include "Emu/RSX/Common/simple_array.hpp"
#include "Emu/RSX/Overlays/overlay_controls.h"
#include "mtlutils/data_heap.h"
#include "mtlutils/graphics_pipeline_state.hpp"
#include "mtlutils/image.h"
#include "mtlutils/sampler.h"

#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace rsx::overlays
{
	enum class texture_sampling_mode;
	struct overlay;
}

namespace mtl
{
	class MTLPipelineCompiler;
	class render_device;
	class render_target;

	struct overlay_render_target
	{
		image* resource = nullptr;
		u32 mip_level = 0;
		u32 array_slice = 0;
		u8 write_aspects = texture_aspect_color;
		bool preserve_contents = true;

		void validate() const;
		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] u32 width() const;
		[[nodiscard]] u32 height() const;
		[[nodiscard]] u32 samples() const;
	};

	struct overlay_pipeline_key
	{
		u64 color_format = 0;
		u64 depth_stencil_format = 0;
		u32 sample_count = 1;
		primitive_topology topology = primitive_topology::triangle_strip;
		u8 color_write_mask = 0xf;
		bool blending = false;
		bool depth_write = false;
		bool stencil_write = false;

		[[nodiscard]] bool operator==(const overlay_pipeline_key&) const = default;
		[[nodiscard]] u64 hash() const;
	};

	struct overlay_pipeline_key_hash
	{
		[[nodiscard]] usz operator()(const overlay_pipeline_key& key) const noexcept;
	};

	struct overlay_pass_statistics
	{
		u64 draw_calls = 0;
		u64 vertices = 0;
		u64 pipeline_builds = 0;
		u64 pipeline_cache_hits = 0;
		u64 texture_uploads = 0;
		u64 uploaded_bytes = 0;
	};

	class overlay_pass
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	protected:
		explicit overlay_pass(std::string label);
		void set_shader_sources(std::string vertex_source, std::string fragment_source,
			std::string vertex_function, std::string fragment_function);
		void set_sampler_filter(sampler_filter filter);
		void set_source_texture_count(u32 count);
		void set_primitive_topology(primitive_topology topology);
		void set_color_write_mask(u8 mask);
		void set_blending(bool enabled);
		void set_depth_write(bool enabled);
		void set_stencil_write(bool enabled);
		void upload_vertex_bytes(std::span<const std::byte> bytes, u32 stride);
		void upload_constant_bytes(std::span<const std::byte> bytes);
		void draw(command_buffer& command, const areau& viewport,
			const overlay_render_target& target, std::span<image_view* const> sources,
			u32 vertex_count, u32 first_vertex = 0, u32 instance_count = 1,
			u32 base_instance = 0, const areau* scissor = nullptr,
			u32 stencil_reference = 0, u32 stencil_write_mask = 0xff);

		template <typename T>
		void upload_vertex_data(std::span<const T> vertices)
		{
			static_assert(std::is_trivially_copyable_v<T>);
			upload_vertex_bytes(std::as_bytes(vertices), sizeof(T));
		}

		template <typename T>
		void upload_constants(const T& constants)
		{
			static_assert(std::is_trivially_copyable_v<T>);
			upload_constant_bytes(std::as_bytes(std::span{&constants, 1}));
		}

	public:
		virtual ~overlay_pass();
		overlay_pass(const overlay_pass&) = delete;
		overlay_pass& operator=(const overlay_pass&) = delete;

		virtual void initialize(render_device& device, memory_allocator& allocator,
			MTLPipelineCompiler& compiler);
		virtual void destroy();
		void reclaim(u64 completed_submission);
		void trim(memory_pressure pressure);

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] overlay_pass_statistics statistics() const;
	};

	class ui_overlay_renderer final : public overlay_pass
	{
		struct ui_impl;
		std::unique_ptr<ui_impl> m_ui;

		[[nodiscard]] image_view* upload_simple_texture(command_buffer& command,
			data_heap& upload_heap, u64 key, u32 width, u32 height, u32 layers,
			bool font, bool temporary, const void* pixels, u32 owner_uid);
		[[nodiscard]] image_view* find_font(const rsx::overlays::font* font,
			command_buffer& command, data_heap& upload_heap);
		[[nodiscard]] image_view* find_temporary_image(
			const rsx::overlays::image_info_base* description, command_buffer& command,
			data_heap& upload_heap, u32 owner_uid);

	public:
		ui_overlay_renderer();
		~ui_overlay_renderer() override;

		void initialize(render_device& device, memory_allocator& allocator,
			MTLPipelineCompiler& compiler) override;
		void initialize_resources(command_buffer& command, data_heap& upload_heap);
		void destroy() override;
		void remove_temporary_resources(u32 owner_uid);
		void run(command_buffer& command, const areau& viewport,
			const overlay_render_target& target, data_heap& upload_heap,
			rsx::overlays::overlay& ui);
	};

	class attachment_clear_pass final : public overlay_pass
	{
	public:
		attachment_clear_pass();
		void run(command_buffer& command, const overlay_render_target& target,
			const areau& rectangle, u32 clear_mask, color4f color);
	};

	class stencil_clear_pass final : public overlay_pass
	{
	public:
		stencil_clear_pass();
		void run(command_buffer& command, render_target& target,
			const areau& rectangle, u32 stencil_value, u32 stencil_write_mask);
	};

	class video_out_calibration_pass final : public overlay_pass
	{
		struct alignas(16) calibration_config
		{
			f32 gamma = 1.f;
			s32 limited_range = 0;
			s32 stereo_display_mode = 0;
			s32 stereo_image_count = 0;
			color4_base<f32> left_anaglyph_matrix[3]{};
			color4_base<f32> right_anaglyph_matrix[3]{};
		};

		calibration_config m_config;

	public:
		video_out_calibration_pass();
		void run(command_buffer& command, const areau& viewport,
			const overlay_render_target& target,
			std::span<viewable_image* const> sources, f32 gamma,
			bool limited_rgb, bool stereo_enabled);
	};

	class overlay_pass_manager final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		overlay_pass_manager();
		~overlay_pass_manager();
		overlay_pass_manager(const overlay_pass_manager&) = delete;
		overlay_pass_manager& operator=(const overlay_pass_manager&) = delete;

		void initialize(render_device& device, memory_allocator& allocator,
			MTLPipelineCompiler& compiler);
		void destroy();
		void reclaim(u64 completed_submission);
		void trim(memory_pressure pressure);

		[[nodiscard]] ui_overlay_renderer& ui();
		[[nodiscard]] attachment_clear_pass& color_clear();
		[[nodiscard]] stencil_clear_pass& stencil_clear();
		[[nodiscard]] video_out_calibration_pass& video_calibration();
		[[nodiscard]] explicit operator bool() const;
	};
}
