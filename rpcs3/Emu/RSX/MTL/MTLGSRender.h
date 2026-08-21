#pragma once

#include "MTLAsyncScheduler.h"
#include "MTLCommandStream.h"
#include "MTLFramebuffer.h"
#include "MTLGSRenderTypes.hpp"
#include "MTLHelpers.h"
#include "MTLOverlays.h"
#include "MTLRenderPass.h"
#include "MTLRenderTargets.h"
#include "MTLShaderInterpreter.h"
#include "MTLTextureCache.h"

#include "Emu/RSX/GSRender.h"
#include "Emu/RSX/Host/RSXDMAWriter.h"

#include <array>
#include <deque>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace mtl
{
	class upscaler;
	using host_data = rsx::host_gpu_context_t;

	struct viewport
	{
		f64 x = 0.0;
		f64 y = 0.0;
		f64 width = 0.0;
		f64 height = 0.0;
		f64 minimum_depth = 0.0;
		f64 maximum_depth = 1.0;
	};

	struct scissor_rectangle
	{
		u32 x = 0;
		u32 y = 0;
		u32 width = 0;
		u32 height = 0;
	};

}

class MTLGSRender : public GSRender, public ::rsx::reports::ZCULL_control
{
	enum frame_context_state : u32
	{
		frame_context_dirty = 1,
	};

	enum flush_queue_state : u32
	{
		flush_queue_ok = 0,
		flush_queue_active = 1,
		flush_queue_deadlock = 2,
	};

	const mtl::MTLFragmentProgram* m_fragment_program = nullptr;
	const mtl::MTLVertexProgram* m_vertex_program = nullptr;
	mtl::MTLProgramPipeline* m_program_template = nullptr;
	std::unique_ptr<mtl::MTLProgramPipeline> m_program_instance;
	mtl::MTLProgramPipeline* m_program = nullptr;
	bool m_program_interpreted = false;
	mtl::pipeline_properties m_pipeline_properties;
	const mtl::vertex_program_bindings* m_vertex_bindings = nullptr;
	const mtl::fragment_program_bindings* m_fragment_bindings = nullptr;

	mtl::texture_cache m_texture_cache;
	mtl::surface_cache m_render_targets;
	std::unique_ptr<mtl::buffer> m_null_buffer;
	std::unique_ptr<mtl::buffer_view> m_null_buffer_view;
	std::shared_ptr<mtl::upscaler> m_upscaler;
	output_scaling_mode m_output_scaling = output_scaling_mode::bilinear;

	std::unique_ptr<mtl::buffer> m_conditional_render_buffer;
	u64 m_conditional_render_sync_tag = 0;
	shared_mutex m_sampler_mutex;
	atomic_t<bool> m_samplers_dirty = true;
	std::array<std::shared_ptr<mtl::sampler>, rsx::limits::fragment_textures_count> m_fragment_samplers{};
	std::array<std::shared_ptr<mtl::sampler>, rsx::limits::vertex_textures_count> m_vertex_samplers{};

	std::unique_ptr<mtl::vertex_cache> m_vertex_cache;
	std::unique_ptr<mtl::shader_cache> m_shader_cache;
	std::unique_ptr<mtl::MTLProgramBuffer> m_program_buffer;

	mtl::shared_state* m_shared_state = nullptr;
	mtl::render_device* m_device = nullptr;
	mtl::memory_allocator* m_allocator = nullptr;
	std::unique_ptr<mtl::swapchain_interface> m_swapchain;
	mtl::command_stream m_command_stream;
	mtl::async_task_scheduler m_async_scheduler;
	mtl::MTLPipelineCompiler m_pipeline_compiler;
	mtl::renderer_resources m_resources;

	std::unique_ptr<mtl::query_pool_manager> m_occlusion_query_manager;
	bool m_occlusion_query_active = false;
	rsx::reports::occlusion_query_info* m_active_query_info = nullptr;
	std::vector<mtl::occlusion_data> m_occlusion_map;

	shared_mutex m_secondary_command_guard;
	mtl::command_buffer_chain<mtl::maximum_async_command_buffers> m_secondary_commands;
	mtl::command_buffer_chain<mtl::maximum_async_command_buffers> m_primary_commands;
	mtl::command_buffer_chunk* m_current_command_buffer = nullptr;

	std::unique_ptr<mtl::buffer> m_host_object_data;
	std::shared_ptr<mtl::framebuffer> m_draw_framebuffer;
	mtl::render_pass m_render_pass;
	mtl::render_pass_configuration m_render_pass_configuration;
	mtl::encoder_binding_state m_encoder_bindings;

	sizeu m_swapchain_dimensions{};
	bool m_swapchain_unavailable = false;
	bool m_should_reinitialize_swapchain = false;
	u64 m_last_heap_sync_time = 0;
	u32 m_texture_buffer_view_size = 0;

	mtl::data_heap m_attribute_heap;
	mtl::data_heap m_fragment_constants_heap;
	mtl::data_heap m_transform_constants_heap;
	mtl::data_heap m_fragment_environment_heap;
	mtl::data_heap m_vertex_environment_heap;
	mtl::data_heap m_fragment_texture_parameters_heap;
	mtl::data_heap m_vertex_layout_heap;
	mtl::data_heap m_index_heap;
	mtl::data_heap m_texture_upload_heap;
	mtl::data_heap m_raster_environment_heap;
	mtl::data_heap m_instancing_heap;
	mtl::data_heap m_fragment_instructions_heap;
	mtl::data_heap m_vertex_instructions_heap;
	rsx::simple_array<mtl::data_heap*> m_flushable_heaps;

	mtl::argument_buffer_binding m_vertex_environment_binding;
	mtl::argument_buffer_binding m_fragment_environment_binding;
	mtl::argument_buffer_binding m_vertex_layout_binding;
	mtl::argument_buffer_binding m_vertex_constants_binding;
	mtl::argument_buffer_binding m_fragment_constants_binding;
	mtl::argument_buffer_binding m_fragment_texture_parameters_binding;
	mtl::argument_buffer_binding m_raster_environment_binding;
	mtl::argument_buffer_binding m_instancing_indirection_binding;
	mtl::argument_buffer_binding m_instancing_constants_binding;
	mtl::argument_buffer_binding m_vertex_instructions_binding;
	mtl::argument_buffer_binding m_fragment_instructions_binding;
	mtl::argument_buffer_binding m_sampler_state_binding;
	mtl::argument_buffer_binding m_vertex_sampler_state_binding;

	u64 m_transform_constants_offset = 0;
	u64 m_vertex_environment_offset = 0;
	u64 m_vertex_layout_offset = 0;
	u64 m_fragment_constants_offset = 0;
	u64 m_fragment_environment_offset = 0;
	u64 m_texture_parameters_offset = 0;
	u64 m_stipple_array_offset = 0;

	std::vector<mtl::frame_context> m_frame_context_storage;
	u32 m_maximum_async_frames = 0;
	mtl::frame_context m_auxiliary_frame_context;
	u32 m_current_queue_index = 0;
	mtl::frame_context* m_current_frame = nullptr;
	std::deque<mtl::frame_context*> m_queued_frames;

	mtl::viewport m_viewport;
	mtl::scissor_rectangle m_scissor;
	std::vector<u8> m_draw_buffers;
	shared_mutex m_flush_queue_mutex;
	mtl::flush_request_task m_flush_requests;
	ullong m_last_conditional_render_evaluation = 0;
	rsx::atomic_bitmask_t<flush_queue_state> m_queue_status;
	utils::address_range32 m_offloader_fault_range;
	rsx::invalidation_cause m_offloader_fault_cause = rsx::invalidation_cause::read;
	mtl::draw_call m_current_draw;
	std::vector<mtl::image*> m_framebuffer_images;
	std::unique_ptr<mtl::viewable_image> m_overlay_recording_image;
	std::vector<std::unique_ptr<mtl::viewable_image>> m_present_temporary_images;
	std::vector<std::unique_ptr<mtl::buffer>> m_present_temporary_buffers;
	rsx::vertex_input_layout m_vertex_layout;
	mtl::MTLShaderInterpreter m_shader_interpreter;
	u32 m_interpreter_state = 0;
	mtl::overlay_pass_manager m_overlay_passes;
	rsx::simple_array<u8> m_scratch_memory;

	[[nodiscard]] std::pair<const mtl::vertex_program_bindings*,
		const mtl::fragment_program_bindings*> get_binding_tables() const;
	void prepare_render_targets(rsx::framebuffer_creation_context context);
	mtl::submission close_and_submit_command_buffer(std::span<const mtl::event_operation> waits = {},
		std::span<const mtl::event_operation> signal_operations = {}, bool wait_for_completion = false);
	void flush_command_queue(bool hard_sync = false, bool preserve_current = false);
	void queue_swap_request();
	void cleanup_frame_context(mtl::frame_context* context);
	void advance_queued_frames();
	void present(mtl::frame_context* context);
	[[nodiscard]] bool reinitialize_swapchain();
	[[nodiscard]] mtl::viewable_image* get_present_source(
		mtl::present_surface_info* information, const rsx::avconf& configuration);

	void begin_render_pass();
	void close_render_pass();
	[[nodiscard]] std::shared_ptr<mtl::framebuffer> get_framebuffer();
	void invalidate_render_pass();
	void update_draw_state();
	void check_present_status();

	[[nodiscard]] mtl::vertex_upload_info upload_vertex_data();
	[[nodiscard]] bool load_program();
	void load_program_environment();
	void update_vertex_environment(u32 id, const mtl::vertex_upload_info& vertex_information);
	void upload_transform_constants(const rsx::io_buffer& source);
	void load_texture_environment();
	[[nodiscard]] bool bind_texture_environment();
	[[nodiscard]] bool bind_interpreter_texture_environment();
	void reclaim_completed_resources();

public:
	MTLGSRender(utils::serial* archive) noexcept;
	MTLGSRender() noexcept : MTLGSRender(nullptr) {}
	~MTLGSRender() override;

	[[nodiscard]] u64 get_cycles() final;
	void initialize_buffers(rsx::framebuffer_creation_context context, bool skip_reading = false);
	void set_viewport();
	void set_scissor(bool clip_viewport);
	void bind_viewport();

	void write_barrier(u32 address, u32 range) override;
	void sync_hint(rsx::FIFO::interrupt_hint hint, rsx::reports::sync_hint_payload_t payload) override;
	bool release_GCM_label(u32 type, u32 address, u32 data) override;

	void begin_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	void end_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	bool check_occlusion_query_status(rsx::reports::occlusion_query_info* query) override;
	void get_occlusion_query_result(rsx::reports::occlusion_query_info* query) override;
	void discard_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	void emergency_query_cleanup(mtl::command_buffer* commands);

	[[nodiscard]] bool on_vram_exhausted(rsx::problem_severity severity);
	void begin_conditional_rendering(
		const std::vector<rsx::reports::occlusion_query_info*>& sources) override;
	void end_conditional_rendering() override;
	[[nodiscard]] std::pair<volatile mtl::host_data*, mtl::buffer_handle>
		map_host_object_data() const;
	void on_guest_texture_read(const mtl::command_buffer& command);
	void patch_transform_constants(rsx::context* context, u32 index, u32 count) override;
	[[nodiscard]] bool is_current_program_interpreted() const override;

protected:
	void clear_surface(u32 mask) override;
	void begin() override;
	void end() override;
	void emit_geometry(u32 sub_index) override;
	void on_init_thread() override;
	void on_exit() override;
	void flip(const rsx::display_flip_info_t& information) override;
	void renderctl(u32 request_code, void* arguments) override;
	void do_local_task(rsx::FIFO::state state) override;
	bool scaled_image_from_memory(const rsx::blit_src_info& source,
		const rsx::blit_dst_info& destination, bool interpolate) override;
	void notify_tile_unbound(u32 tile) override;
	bool on_access_violation(u32 address, bool is_writing) override;
	void on_invalidate_memory_range(const utils::address_range32& range,
		rsx::invalidation_cause cause) override;
	void on_semaphore_acquire_wait() override;
};
