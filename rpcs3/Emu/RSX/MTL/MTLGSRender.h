#pragma once

#include "Emu/RSX/RSXThread.h"
#include "Emu/RSX/Core/RSXVertexTypes.h"

#include "MTLCommandQueue.h"
#include "MTLDevice.h"
#include "MTLNativeWindow.h"
#include "MTLPipelineCache.h"
#include "MTLPipelineState.h"
#include "MTLPresentation.h"
#include "MTLRenderTargetCache.h"
#include "MTLRenderState.h"
#include "MTLSamplerCache.h"
#include "MTLShaderCache.h"
#include "MTLShaderCompiler.h"
#include "MTLShaderLibrary.h"
#include "MTLShaderRecompiler.h"
#include "MTLTextureCache.h"

#include <memory>
#include <vector>

namespace rsx::metal
{
	class argument_table_pool;
	class draw_resource_manager;
	struct prepared_draw_command;
}

class MTLGSRender : public rsx::thread
{
public:
	u64 get_cycles() final;

	MTLGSRender(utils::serial* ar) noexcept;
	MTLGSRender() noexcept : MTLGSRender(nullptr) {}
	~MTLGSRender() override;

private:
	void on_init_thread() override;
	void on_exit() override;
	void end() override;
	void clear_surface(u32 arg) override;
	void flip(const rsx::display_flip_info_t& info) override;
	f64 get_display_refresh_rate() const override;

	void report_backend_state() const;
	void present_clear_frame();
	void update_draw_target_state(const rsx::metal::draw_target_binding& binding);
	rsx::metal::render_pipeline_record get_current_render_pipeline(const rsx::metal::draw_target_binding& binding);
	rsx::metal::dynamic_render_state_desc get_current_dynamic_render_state(const rsx::metal::draw_render_encoder_scope& draw_encoder);
	rsx::metal::prepared_draw_command prepare_draw_vertex_inputs(rsx::metal::command_frame& frame, rsx::metal::argument_table& vertex_table, const rsx::metal::shader_interface_layout& layout);
	rsx::metal::prepared_draw_command preflight_draw_argument_tables(rsx::metal::command_frame& frame, void* render_encoder_handle, const rsx::metal::render_pipeline_record& pipeline);
	rsx::metal::argument_table_pool& get_fragment_argument_table_pool(const rsx::metal::shader_interface_layout& layout);
	void prepare_fragment_textures(rsx::metal::command_frame& frame, const rsx::metal::shader_interface_layout& layout);
	void bind_fragment_textures(rsx::metal::argument_table& fragment_table, const rsx::metal::shader_interface_layout& layout);
	void bind_fragment_samplers(rsx::metal::argument_table& fragment_table, const rsx::metal::shader_interface_layout& layout);
	u32 retained_fragment_argument_table_count() const;

	struct fragment_argument_table_pool_record
	{
		u32 max_buffers = 0;
		u32 max_textures = 0;
		u32 max_samplers = 0;
		b8 support_attribute_strides = false;
		std::unique_ptr<rsx::metal::argument_table_pool> pool;
	};

	std::unique_ptr<rsx::metal::device> m_device;
	std::unique_ptr<rsx::metal::native_window> m_window;
	std::unique_ptr<rsx::metal::command_queue> m_queue;
	std::unique_ptr<rsx::metal::presentation_surface> m_presentation;
	std::unique_ptr<rsx::metal::persistent_shader_cache> m_shader_cache;
	std::unique_ptr<rsx::metal::shader_compiler> m_shader_compiler;
	std::unique_ptr<rsx::metal::shader_library_cache> m_shader_library_cache;
	std::unique_ptr<rsx::metal::shader_recompiler> m_shader_recompiler;
	std::unique_ptr<rsx::metal::pipeline_cache> m_pipeline_cache;
	std::unique_ptr<rsx::metal::render_pipeline_cache> m_render_pipeline_cache;
	std::unique_ptr<rsx::metal::render_target_cache> m_render_target_cache;
	std::unique_ptr<rsx::metal::render_state_cache> m_render_state_cache;
	std::unique_ptr<rsx::metal::sampler_cache> m_sampler_cache;
	std::unique_ptr<rsx::metal::sampled_texture_cache> m_texture_cache;
	std::unique_ptr<rsx::metal::draw_resource_manager> m_draw_resources;
	std::unique_ptr<rsx::metal::argument_table_pool> m_vertex_argument_tables;
	std::vector<fragment_argument_table_pool_record> m_fragment_argument_table_pools;
	rsx::vertex_input_layout m_vertex_layout;
	rsx::metal::render_scissor m_draw_scissor;
	b8 m_draw_scissor_valid = false;
	u64 m_frame_count = 0;
};
