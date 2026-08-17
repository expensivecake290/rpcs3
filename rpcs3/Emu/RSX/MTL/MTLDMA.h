#pragma once

#include <functional>
#include <memory>
#include <span>

#include "Utilities/address_range.h"
#include "mtlutils/buffer_object.h"
#include "mtlutils/commands.h"

namespace mtl
{
	inline constexpr u32 dma_default_block_size = 0x10000;
	inline constexpr u32 dma_default_page_size = 0x1000;

	enum class dma_block_kind : u8
	{
		staging,
		host_no_copy,
		alias,
	};

	enum class dma_access : u8
	{
		gpu_read,
		gpu_write,
		gpu_read_write,
	};

	struct dma_host_callbacks
	{
		std::function<bool(u32 address, std::span<u8> destination)> read;
		std::function<bool(u32 address, std::span<const u8> source)> write;
		std::function<void*(u32 address, u64 length)> direct_pointer;

		[[nodiscard]] explicit operator bool() const
		{
			return static_cast<bool>(read) && static_cast<bool>(write);
		}
	};

	struct dma_pool_create_info
	{
		u32 block_size = dma_default_block_size;
		u32 page_size = dma_default_page_size;
		u64 maximum_cached_bytes = 256ull * 1024 * 1024;
		allocation_pool allocation = allocation_pool::system;
		bool allow_host_no_copy = true;
		bool preload_new_blocks = true;
	};

	struct dma_block_statistics
	{
		dma_block_kind kind = dma_block_kind::staging;
		u32 base_address = 0;
		u64 size = 0;
		u64 generation = 0;
		u64 loads = 0;
		u64 flushes = 0;
		u64 loaded_bytes = 0;
		u64 flushed_bytes = 0;
		u64 last_gpu_read_submission = 0;
		u64 last_gpu_write_submission = 0;
		u64 completed_submission = 0;
		bool host_dirty = false;
		bool gpu_dirty = false;
	};

	class dma_block;

	struct dma_mapping_handle
	{
		std::shared_ptr<dma_block> owner;
		buffer* resource = nullptr;
		u64 offset = 0;
		u32 address = 0;
		u32 length = 0;
		u64 generation = 0;

		[[nodiscard]] explicit operator bool() const
		{
			return owner && resource && length != 0 && generation != 0;
		}
	};

	class dma_block final : public std::enable_shared_from_this<dma_block>
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		dma_block();
		~dma_block();
		dma_block(const dma_block&) = delete;
		dma_block& operator=(const dma_block&) = delete;
		dma_block(dma_block&&) = delete;
		dma_block& operator=(dma_block&&) = delete;

		void create(memory_allocator& allocator, u32 base_address, u64 size,
			const dma_host_callbacks& host, const dma_pool_create_info& info);
		void create_alias(std::shared_ptr<dma_block> parent, u32 base_address, u64 size);
		void destroy();
		void extend(memory_allocator& allocator, u64 new_size,
			const dma_host_callbacks& host, const dma_pool_create_info& info);

		[[nodiscard]] dma_mapping_handle map(const utils::address_range32& range);
		void load(const utils::address_range32& range);
		void flush(const utils::address_range32& range);
		void mark_host_modified(const utils::address_range32& range);
		void mark_gpu_access(const utils::address_range32& range, dma_access access, u64 submission_value);
		void notify_completed(u64 completed_submission_value);

		void copy_to(command_buffer& command, buffer& destination, u64 destination_offset,
			const utils::address_range32& source_range);
		void copy_from(command_buffer& command, const buffer& source, u64 source_offset,
			const utils::address_range32& destination_range, u64 submission_value);

		[[nodiscard]] bool contains(const utils::address_range32& range) const;
		[[nodiscard]] bool overlaps(const utils::address_range32& range) const;
		[[nodiscard]] u32 start() const;
		[[nodiscard]] u32 end() const;
		[[nodiscard]] u64 size() const;
		[[nodiscard]] u64 generation() const;
		[[nodiscard]] dma_block_kind kind() const;
		[[nodiscard]] buffer* resource() const;
		[[nodiscard]] std::shared_ptr<dma_block> head();
		[[nodiscard]] dma_block_statistics statistics() const;
	};

	struct dma_pool_statistics
	{
		u64 mappings = 0;
		u64 cache_hits = 0;
		u64 cache_misses = 0;
		u64 merges = 0;
		u64 loads = 0;
		u64 flushes = 0;
		u64 invalidations = 0;
		u64 blocks = 0;
		u64 head_blocks = 0;
		u64 allocated_bytes = 0;
		u64 completed_submission = 0;
	};

	class dma_pool final
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		dma_pool();
		dma_pool(memory_allocator& allocator, dma_host_callbacks host,
			const dma_pool_create_info& info = {});
		~dma_pool();
		dma_pool(const dma_pool&) = delete;
		dma_pool& operator=(const dma_pool&) = delete;
		dma_pool(dma_pool&&) = delete;
		dma_pool& operator=(dma_pool&&) = delete;

		void create(memory_allocator& allocator, dma_host_callbacks host,
			const dma_pool_create_info& info = {});
		void destroy();

		[[nodiscard]] dma_mapping_handle map(u32 address, u32 length);
		void load(u32 address, u32 length);
		void flush(u32 address, u32 length);
		void mark_host_modified(u32 address, u32 length);
		void mark_gpu_access(const dma_mapping_handle& mapping, dma_access access, u64 submission_value);
		void notify_completed(u64 completed_submission_value);
		[[nodiscard]] usz invalidate(u32 address, u32 length);
		[[nodiscard]] usz trim(u32 required_idle_checks = 2);
		void clear();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] dma_pool_statistics statistics() const;
	};

	void initialize_dma_pool(memory_allocator& allocator, dma_host_callbacks host,
		const dma_pool_create_info& info = {});
	void shutdown_dma_pool();
	[[nodiscard]] dma_pool& get_dma_pool();
	[[nodiscard]] dma_mapping_handle map_dma(u32 address, u32 length);
	void load_dma(u32 address, u32 length);
	void flush_dma(u32 address, u32 length);
	[[nodiscard]] usz unmap_dma(u32 address, u32 length);
	void notify_dma_completed(u64 completed_submission_value);
	void clear_dma_resources();
}
