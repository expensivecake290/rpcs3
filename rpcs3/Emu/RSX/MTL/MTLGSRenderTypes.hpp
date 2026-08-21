#pragma once

#include "MTLDataHeapManager.h"
#include "MTLFormats.h"
#include "MTLProgramBuffer.h"
#include "MTLQueryPool.h"
#include "MTLResourceManager.h"
#include "mtlutils/swapchain.h"

#include "Emu/RSX/Common/simple_array.hpp"
#include "Emu/RSX/rsx_cache.h"
#include "Emu/RSX/rsx_utils.h"
#include "Utilities/mutex.h"
#include "util/asm.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace mtl
{
	inline constexpr u64 attribute_ring_buffer_size = 64ull * 1024 * 1024;
	inline constexpr u64 texture_upload_ring_buffer_size = 64ull * 1024 * 1024;
	inline constexpr u64 uniform_ring_buffer_size = 16ull * 1024 * 1024;
	inline constexpr u64 transform_constants_buffer_size = 16ull * 1024 * 1024;
	inline constexpr u64 fragment_constants_buffer_size = 16ull * 1024 * 1024;
	inline constexpr u64 index_ring_buffer_size = 16ull * 1024 * 1024;
	inline constexpr u32 maximum_async_command_buffers = 512;
	inline constexpr auto frame_present_timeout = std::chrono::seconds(10);
	inline constexpr auto general_wait_timeout = std::chrono::seconds(2);

	using vertex_cache = rsx::vertex_cache::default_vertex_cache<rsx::vertex_cache::uploaded_range>;
	using weak_vertex_cache = rsx::vertex_cache::weak_vertex_cache;
	using null_vertex_cache = vertex_cache;
	using pipeline_properties = graphics_pipeline_configuration;
	using shader_cache = rsx::shaders_cache<pipeline_properties, MTLProgramBuffer>;

	struct vertex_upload_info
	{
		primitive_topology primitive = primitive_topology::triangle;
		u32 vertex_draw_count = 0;
		u32 allocated_vertex_count = 0;
		u32 first_vertex = 0;
		u32 vertex_index_base = 0;
		u32 vertex_index_offset = 0;
		u32 persistent_window_offset = 0;
		u32 volatile_window_offset = 0;
		std::optional<std::tuple<u64, index_element_type>> index_info;
		argument_buffer_binding line_mapping_binding;
		bool emulated_indices = false;
		bool primitive_restart = false;
		bool line_expansion = false;
	};

	struct command_buffer_chunk : command_buffer
	{
		u64 resource_event = 0;
		u64 reset_id = 0;
		mutable shared_mutex guard_mutex;

		command_buffer_chunk() = default;

		void tag()
		{
			std::unique_lock lock(guard_mutex);
			if (resource_event)
				fmt::throw_exception("Metal command buffer already has a resource event");
			resource_event = allocate_resource_event();
			const u64 completion_event = resource_event;
			notify_on_completion([completion_event](bool)
			{
				notify_resource_event_completed(completion_event);
			});
		}

		[[nodiscard]] bool poke()
		{
			std::unique_lock lock(guard_mutex);
			const submission current = last_submission();
			if (!current || !current.completed()) return !current;
			resource_event = 0;
			return true;
		}

		[[nodiscard]] bool wait(std::chrono::nanoseconds timeout = {})
		{
			std::unique_lock lock(guard_mutex);
			const submission current = last_submission();
			if (!current) return true;
			const bool completed = timeout.count() == 0 ? (current.wait(), true) : current.wait(timeout);
			if (completed) resource_event = 0;
			return completed && current.succeeded();
		}

		void reset()
		{
			if (!poke() && !wait(frame_present_timeout))
				fmt::throw_exception("Metal command buffer did not complete before reset");
			std::unique_lock lock(guard_mutex);
			if (last_submission()) reset_after_completion();
			++reset_id;
		}

		void flush() const
		{
			reader_lock lock(guard_mutex);
			const submission current = last_submission();
			if (current && current.completed() && !current.succeeded())
				fmt::throw_exception("Metal command buffer submission failed");
		}
	};

	struct occlusion_data
	{
		rsx::simple_array<query_handle> indices;
		command_buffer_chunk* command_buffer_to_wait = nullptr;
		u64 command_buffer_sync_id = 0;

		[[nodiscard]] bool is_current(const command_buffer_chunk* command) const
		{
			return command && command_buffer_to_wait == command &&
				command_buffer_sync_id == command->reset_id;
		}

		void set_sync_command_buffer(command_buffer_chunk* command)
		{
			if (!command) fmt::throw_exception("Cannot synchronize occlusion data with an empty command buffer");
			command_buffer_to_wait = command;
			command_buffer_sync_id = command->reset_id;
		}

		void sync() const
		{
			if (!command_buffer_to_wait)
				fmt::throw_exception("Occlusion data has no synchronization command buffer");
			if (command_buffer_to_wait->reset_id == command_buffer_sync_id)
			{
				command_buffer_to_wait->flush();
				if (!command_buffer_to_wait->wait(frame_present_timeout))
					fmt::throw_exception("Metal occlusion synchronization failed");
			}
		}
	};

	struct frame_context
	{
		rsx::flags32_t flags = 0;
		drawable_frame drawable;
		command_buffer_chunk* swap_command_buffer = nullptr;
		managed_heap_snapshot heap_snapshot;
		u64 last_frame_sync_time = 0;
		u64 frame_id = 0;
		u64 submission_id = 0;

		void initialize(u64 id)
		{
			if (!id) fmt::throw_exception("Metal frame context requires a nonzero identifier");
			*this = {};
			frame_id = id;
		}

		void destroy(swapchain_interface* swapchain = nullptr)
		{
			if (drawable && swapchain) swapchain->discard(drawable);
			drawable = {};
			swap_command_buffer = nullptr;
			heap_snapshot = {};
			flags = 0;
			last_frame_sync_time = 0;
			frame_id = 0;
			submission_id = 0;
		}

		void grab_resources(const frame_context& other)
		{
			flags = other.flags;
			heap_snapshot = other.heap_snapshot;
			last_frame_sync_time = other.last_frame_sync_time;
			submission_id = other.submission_id;
		}

		void tag_frame_end(u64 submission)
		{
			if (!submission) fmt::throw_exception("Metal frame completion requires a submission");
			submission_id = submission;
			heap_snapshot = get_data_heap_manager().capture_snapshot(submission);
			last_frame_sync_time = rsx::get_shared_tag();
		}

		void reset_heap_ptrs()
		{
			last_frame_sync_time = 0;
			submission_id = 0;
			heap_snapshot = {};
		}
	};

	struct flush_request_task
	{
		atomic_t<bool> pending_state{false};
		atomic_t<int> num_waiters{0};
		std::atomic_bool hard_sync{false};

		void post(bool require_hard_sync)
		{
			if (require_hard_sync) hard_sync.store(true, std::memory_order_release);
			pending_state.store(true);
			num_waiters++;
		}

		void remove_one()
		{
			const int previous = num_waiters.fetch_sub(1);
			if (previous <= 0)
			{
				num_waiters.fetch_add(1);
				fmt::throw_exception("Metal flush request waiter count underflowed");
			}
		}

		void clear_pending_flag()
		{
			hard_sync.store(false, std::memory_order_release);
			pending_state.store(false);
		}

		[[nodiscard]] bool pending() const
		{
			return pending_state.load();
		}

		[[nodiscard]] bool requires_hard_sync() const
		{
			return hard_sync.load(std::memory_order_acquire);
		}

		void consumer_wait() const
		{
			utils::spin_wait(num_waiters, [](auto value)
			{
				return value == 0;
			});
		}

		void producer_wait() const
		{
			utils::spin_wait(pending_state, [](auto value)
			{
				return !value;
			});
		}
	};

	struct present_surface_info
	{
		u32 address = 0;
		u32 format = 0;
		u32 width = 0;
		u32 height = 0;
		u32 pitch = 0;
		u8 eye = 0;
	};

	struct draw_call
	{
		u32 subdraw_id = 0;
	};
	using frame_context_t = frame_context;
	using draw_call_t = draw_call;

	struct encoder_binding_state
	{
		MTLProgramPipeline* pipeline = nullptr;
		u64 render_pass_signature = 0;
		u64 vertex_argument_signature = 0;
		u64 fragment_argument_signature = 0;
		u32 stencil_reference = 0;
		u32 sample_mask = 0xffffffff;
		bool viewport_valid = false;
		bool scissor_valid = false;
		bool depth_stencil_valid = false;
		bool blend_constant_valid = false;

		void invalidate()
		{
			*this = {};
			sample_mask = 0xffffffff;
		}
	};

	template <u32 Count>
	class command_buffer_chain
	{
		static_assert(Count != 0);
		atomic_t<u32> m_current_index = 0;
		std::array<command_allocator, Count> m_allocators;
		std::array<command_buffer_chunk, Count> m_commands;

	public:
		command_buffer_chain() = default;

		void create(render_device& device, std::string_view label)
		{
			if (!device || label.empty())
				fmt::throw_exception("Invalid Metal command-buffer chain configuration");
			for (u32 index = 0; index < Count; ++index)
			{
				const std::string indexed_label = fmt::format("%s %u", label, index);
				m_allocators[index].create(device, indexed_label);
				m_commands[index].create(m_allocators[index], indexed_label);
			}
			m_current_index.store(0);
		}

		void destroy()
		{
			wait_all();
			for (auto& command : m_commands) command.destroy();
			for (auto& allocator : m_allocators) allocator.destroy();
			m_current_index.store(0);
		}

		void poke_all()
		{
			for (auto& command : m_commands) static_cast<void>(command.poke());
		}

		void wait_all()
		{
			for (auto& command : m_commands)
			{
				if (!command.wait(frame_present_timeout))
					fmt::throw_exception("Metal command-buffer chain wait failed");
			}
		}

		[[nodiscard]] command_buffer_chunk* next()
		{
			const u32 index = ++m_current_index % Count;
			auto& command = m_commands[index];
			if (!command.poke() && !command.wait(frame_present_timeout))
				fmt::throw_exception("Metal command-buffer chain exhausted its in-flight entries");
			command.reset();
			return &command;
		}

		[[nodiscard]] command_buffer_chunk* get()
		{
			return &m_commands[m_current_index.load() % Count];
		}
	};
}
