#pragma once

#include <array>
#include <memory>
#include <string>

#include "device.h"

namespace mtl
{
	enum class allocation_pool : u8
	{
		system,
		surface_cache,
		texture_cache,
		swapchain,
		scratch,
		sampler,
		count,
	};

	enum class storage_mode : u8
	{
		automatic,
		shared,
		managed,
		private_,
		memoryless,
	};

	enum class cpu_cache_mode : u8
	{
		default_cache,
		write_combined,
	};

	enum class cpu_access : u8
	{
		none,
		read,
		write,
		read_write,
	};

	enum class hazard_tracking : u8
	{
		tracked,
		untracked,
	};

	enum class memory_pressure : u8
	{
		normal,
		warning,
		critical,
	};

	struct memory_allocation_request
	{
		u64 size = 0;
		u64 alignment = 1;
		storage_mode storage = storage_mode::automatic;
		cpu_cache_mode cache = cpu_cache_mode::default_cache;
		cpu_access access = cpu_access::none;
		hazard_tracking hazards = hazard_tracking::tracked;
		allocation_pool pool = allocation_pool::system;
		std::string label;
		bool use_placement_heap = true;
		bool purgeable = false;
		bool throw_on_failure = true;
		bool recover_on_failure = true;
	};

	struct memory_usage
	{
		u64 allocated = 0;
		u64 resident = 0;
		u64 peak = 0;
		u64 budget = 0;
		std::array<u64, static_cast<usz>(allocation_pool::count)> pools{};

		[[nodiscard]] f32 pressure_ratio() const
		{
			return budget ? static_cast<f32>(allocated) / static_cast<f32>(budget) : 0.f;
		}
	};

	using native_heap_handle = void*;
	using native_allocation_handle = void*;

	class memory_allocation
	{
		struct impl;
		std::shared_ptr<impl> m_impl;

		explicit memory_allocation(std::shared_ptr<impl> implementation);
		friend class memory_allocator;

	public:
		memory_allocation() = default;

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] u64 size() const;
		[[nodiscard]] u64 offset() const;
		[[nodiscard]] storage_mode storage() const;
		[[nodiscard]] allocation_pool pool() const;
		[[nodiscard]] bool is_cpu_visible() const;
		[[nodiscard]] bool is_heap_placed() const;
		[[nodiscard]] buffer_handle buffer() const;
		[[nodiscard]] native_heap_handle heap() const;
		[[nodiscard]] native_allocation_handle native_allocation() const;

		[[nodiscard]] void* map(u64 offset, u64 size);
		void unmap();
		void did_modify(u64 offset, u64 size);
		[[nodiscard]] bool make_volatile();
		[[nodiscard]] bool make_nonvolatile();
	};

	class residency_set
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		residency_set();
		~residency_set();
		residency_set(const residency_set&) = delete;
		residency_set& operator=(const residency_set&) = delete;
		residency_set(residency_set&&) noexcept;
		residency_set& operator=(residency_set&&) noexcept;

		void create(const render_device& device, std::string_view label, u64 initial_capacity);
		void destroy();
		void add(const memory_allocation& allocation);
		void add(native_allocation_handle allocation);
		void remove(const memory_allocation& allocation);
		void remove(native_allocation_handle allocation);
		void attach(command_queue_handle queue);
		void detach(command_queue_handle queue);
		void request_residency();
		void end_residency();
		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] native_allocation_handle native_handle() const;
	};

	class memory_allocator
	{
		struct impl;
		std::shared_ptr<impl> m_impl;

	public:
		explicit memory_allocator(const render_device& device);
		~memory_allocator();
		memory_allocator(const memory_allocator&) = delete;
		memory_allocator& operator=(const memory_allocator&) = delete;
		memory_allocator(memory_allocator&&) noexcept;
		memory_allocator& operator=(memory_allocator&&) noexcept;

		[[nodiscard]] memory_allocation allocate_buffer(const memory_allocation_request& request);
		[[nodiscard]] memory_allocation allocate_placement(const memory_allocation_request& request);
		void trim(memory_pressure pressure);
		[[nodiscard]] memory_usage usage() const;
		[[nodiscard]] const render_device& device() const;
	};

	class memory_block
	{
		memory_allocation m_allocation;

	public:
		memory_block(memory_allocator& allocator, const memory_allocation_request& request);

		[[nodiscard]] u64 size() const;
		[[nodiscard]] buffer_handle buffer() const;
		[[nodiscard]] void* map(u64 offset, u64 size);
		void unmap();
		void did_modify(u64 offset, u64 size);
		[[nodiscard]] const memory_allocation& allocation() const;
	};

	void notify_memory_allocated(native_allocation_handle handle, u64 size, allocation_pool pool);
	void notify_memory_freed(native_allocation_handle handle);
	void reset_memory_tracking();
	[[nodiscard]] u64 get_application_memory_usage();
	[[nodiscard]] u64 get_application_pool_usage(allocation_pool pool);
	[[nodiscard]] memory_pressure determine_memory_pressure(const memory_usage& usage);
}
