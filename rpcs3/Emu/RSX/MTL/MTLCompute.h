#pragma once

#include "MTLPipelineCompiler.h"

#include <array>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>

namespace mtl
{
	enum class compute_kernel_kind : u8
	{
		shuffle,
		gather_depth_stencil,
		scatter_depth_stencil,
		format_convert,
		deswizzle_3d,
		aggregate,
		tile_copy,
	};

	enum class compute_shuffle_operation : u8
	{
		byte_swap_u16,
		byte_swap_u32,
		byte_swap_u16_u32,
		depth24_to_float32,
		float32_to_depth24_swapped,
		depth24_byte_swap,
	};

	enum class depth_stencil_operation : u8
	{
		gather_depth24,
		gather_depth32,
		scatter_depth24,
		scatter_depth32,
	};

	enum class rsx_detiler_operation : u8
	{
		decode,
		encode,
	};

	struct compute_kernel_specification
	{
		compute_kernel_kind kind = compute_kernel_kind::shuffle;
		compute_shuffle_operation shuffle = compute_shuffle_operation::byte_swap_u32;
		depth_stencil_operation depth_stencil = depth_stencil_operation::gather_depth24;
		rsx_detiler_operation detiler = rsx_detiler_operation::decode;
		u32 source_element_size = 4;
		u32 destination_element_size = 4;
		u32 block_size = 4;
		bool swap_source = false;
		bool swap_destination = false;
		bool depth_float = false;

		void validate() const;
		[[nodiscard]] u32 data_buffer_count() const;
		[[nodiscard]] std::string label() const;
		[[nodiscard]] bool operator==(const compute_kernel_specification&) const = default;
	};

	struct compute_buffer_range
	{
		const buffer* resource = nullptr;
		u64 offset = 0;
		u64 length = 0;
		argument_access access = argument_access::read_write;

		void validate() const;
	};

	struct compute_dispatch_size
	{
		u32 x = 1;
		u32 y = 1;
		u32 z = 1;

		void validate() const;
	};

	struct compute_task_statistics
	{
		u64 dispatches = 0;
		u64 invocations = 0;
		u64 parameter_bytes = 0;
	};

	class compute_task
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	protected:
		explicit compute_task(compute_kernel_specification specification);
		void dispatch(command_buffer& command, std::span<const compute_buffer_range> buffers,
			const void* parameters, usz parameter_size, compute_dispatch_size invocations);

	public:
		virtual ~compute_task();
		compute_task(const compute_task&) = delete;
		compute_task& operator=(const compute_task&) = delete;
		compute_task(compute_task&&) = delete;
		compute_task& operator=(compute_task&&) = delete;

		void create(render_device& device, MTLPipelineCompiler& compiler);
		void destroy();
		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] const compute_kernel_specification& specification() const;
		[[nodiscard]] compute_task_statistics statistics() const;
	};

	class cs_shuffle_base : public compute_task
	{
	protected:
		explicit cs_shuffle_base(compute_shuffle_operation operation);

	public:
		void run(command_buffer& command, const buffer& data, u32 data_length, u32 data_offset = 0);
	};

	class cs_shuffle_16 final : public cs_shuffle_base
	{
	public:
		cs_shuffle_16();
	};

	class cs_shuffle_32 final : public cs_shuffle_base
	{
	public:
		cs_shuffle_32();
	};

	class cs_shuffle_32_16 final : public cs_shuffle_base
	{
	public:
		cs_shuffle_32_16();
	};

	class cs_shuffle_d24x8_f32 final : public cs_shuffle_base
	{
	public:
		cs_shuffle_d24x8_f32();
	};

	class cs_shuffle_se_f32_d24x8 final : public cs_shuffle_base
	{
	public:
		cs_shuffle_se_f32_d24x8();
	};

	class cs_shuffle_se_d24x8 final : public cs_shuffle_base
	{
	public:
		cs_shuffle_se_d24x8();
	};

	class cs_interleave_task : public compute_task
	{
	protected:
		cs_interleave_task(depth_stencil_operation operation, bool swap_bytes, bool depth_float);

	public:
		void run(command_buffer& command, const buffer& data, u32 data_offset, u32 data_length,
			u32 depth_offset, u32 stencil_offset);
	};

	template <bool SwapBytes = false>
	class cs_gather_d24x8 final : public cs_interleave_task
	{
	public:
		cs_gather_d24x8()
			: cs_interleave_task(depth_stencil_operation::gather_depth24, SwapBytes, false)
		{
		}
	};

	template <bool SwapBytes = false, bool DepthFloat = false>
	class cs_gather_d32x8 final : public cs_interleave_task
	{
	public:
		cs_gather_d32x8()
			: cs_interleave_task(depth_stencil_operation::gather_depth32, SwapBytes, DepthFloat)
		{
		}
	};

	class cs_scatter_d24x8 final : public cs_interleave_task
	{
	public:
		cs_scatter_d24x8();
	};

	template <bool DepthFloat = false>
	class cs_scatter_d32x8 final : public cs_interleave_task
	{
	public:
		cs_scatter_d32x8()
			: cs_interleave_task(depth_stencil_operation::scatter_depth32, false, DepthFloat)
		{
		}
	};

	class cs_fconvert_base : public compute_task
	{
	protected:
		cs_fconvert_base(u32 source_element_size, u32 destination_element_size,
			bool swap_source, bool swap_destination);

	public:
		void run(command_buffer& command, const buffer& data, u32 source_offset,
			u32 source_length, u32 destination_offset);
	};

	template <typename From, typename To, bool SwapSource = false, bool SwapDestination = false>
	class cs_fconvert_task final : public cs_fconvert_base
	{
		static_assert((sizeof(From) == 4 && sizeof(To) == 2) ||
			(sizeof(From) == 2 && sizeof(To) == 4));

	public:
		cs_fconvert_task()
			: cs_fconvert_base(sizeof(From), sizeof(To), SwapSource, SwapDestination)
		{
		}
	};

	class cs_deswizzle_base : public compute_task
	{
	protected:
		cs_deswizzle_base(u32 block_size, u32 base_element_size, bool swap_bytes);

	public:
		void run(command_buffer& command, const buffer& destination, u32 destination_offset,
			const buffer& source, u32 source_offset, u32 data_length, u32 width,
			u32 height, u32 depth, u32 mipmaps);
	};

	template <typename BlockType, typename BaseType, bool SwapBytes>
	class cs_deswizzle_3d final : public cs_deswizzle_base
	{
		static_assert(std::is_trivially_copyable_v<BlockType> && std::is_trivially_copyable_v<BaseType>);
		static_assert(sizeof(BaseType) == 1 || sizeof(BaseType) == 2 || sizeof(BaseType) == 4);
		static_assert(sizeof(BlockType) >= sizeof(BaseType) && sizeof(BlockType) % sizeof(BaseType) == 0);

	public:
		cs_deswizzle_3d()
			: cs_deswizzle_base(sizeof(BlockType), sizeof(BaseType), SwapBytes)
		{
		}
	};

	class cs_aggregator final : public compute_task
	{
	public:
		cs_aggregator();
		void run(command_buffer& command, const buffer& destination, const buffer& source,
			u32 num_words, u32 destination_offset = 0, u32 source_offset = 0);
	};

	struct rsx_detiler_config
	{
		u32 tile_base_address = 0;
		u32 tile_base_offset = 0;
		u32 tile_rw_offset = 0;
		u32 tile_size = 0;
		u32 tile_pitch = 0;
		u32 bank = 0;
		const buffer* destination = nullptr;
		u32 destination_offset = 0;
		const buffer* source = nullptr;
		u32 source_offset = 0;
		u16 image_width = 0;
		u16 image_height = 0;
		u32 image_pitch = 0;
		u8 image_bytes_per_pixel = 0;

		void validate() const;
	};

	class cs_tile_memcpy_base : public compute_task
	{
	protected:
		explicit cs_tile_memcpy_base(rsx_detiler_operation operation);

	public:
		void run(command_buffer& command, const rsx_detiler_config& configuration);
	};

	template <rsx_detiler_operation Operation>
	class cs_tile_memcpy final : public cs_tile_memcpy_base
	{
	public:
		cs_tile_memcpy()
			: cs_tile_memcpy_base(Operation)
		{
		}
	};

	class compute_task_manager final
	{
		render_device* m_device = nullptr;
		MTLPipelineCompiler* m_compiler = nullptr;
		std::unordered_map<std::type_index, std::unique_ptr<compute_task>> m_tasks;
		mutable std::mutex m_mutex;

	public:
		compute_task_manager() = default;
		~compute_task_manager();
		compute_task_manager(const compute_task_manager&) = delete;
		compute_task_manager& operator=(const compute_task_manager&) = delete;

		void initialize(render_device& device, MTLPipelineCompiler& compiler);
		void reset();
		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] usz size() const;

		template <typename Task>
		Task& get()
		{
			static_assert(std::is_base_of_v<compute_task, Task>);
			std::lock_guard lock(m_mutex);
			if (!m_device || !m_compiler)
				fmt::throw_exception("Metal compute-task manager is not initialized");
			const std::type_index key(typeid(Task));
			if (const auto found = m_tasks.find(key); found != m_tasks.end())
				return static_cast<Task&>(*found->second);
			auto task = std::make_unique<Task>();
			task->create(*m_device, *m_compiler);
			Task& result = *task;
			m_tasks.emplace(key, std::move(task));
			return result;
		}
	};

	[[nodiscard]] compute_task_manager& compute_tasks();
	void initialize_compute_tasks(render_device& device, MTLPipelineCompiler& compiler);
	void reset_compute_tasks();

	template <typename Task>
	Task* get_compute_task()
	{
		return &compute_tasks().get<Task>();
	}
}
