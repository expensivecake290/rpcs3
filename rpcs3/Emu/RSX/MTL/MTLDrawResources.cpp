#include "stdafx.h"
#include "MTLDrawResources.h"

#include "MTLBuffer.h"
#include "MTLCommandBuffer.h"
#include "MTLDevice.h"

#include "util/asm.hpp"

#include <cstring>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace
{
	constexpr u64 draw_buffer_alignment = 256;

	u64 checked_draw_buffer_size(u64 size)
	{
		rsx_log.trace("checked_draw_buffer_size(size=0x%llx)", size);

		if (!size)
		{
			fmt::throw_exception("Metal draw resource upload requires a non-zero size");
		}

		if (size > std::numeric_limits<u64>::max() - (draw_buffer_alignment - 1))
		{
			fmt::throw_exception("Metal draw resource upload size overflow: size=0x%llx", size);
		}

		return utils::align(size, draw_buffer_alignment);
	}

	void increment_draw_resource_counter(u32& counter, const char* name)
	{
		rsx_log.trace("increment_draw_resource_counter(name=%s, counter=%u)", name, counter);

		if (counter == std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal draw resource counter overflow: %s", name);
		}

		counter++;
	}

	void add_uploaded_draw_bytes(u64& counter, u64 size)
	{
		rsx_log.trace("add_uploaded_draw_bytes(counter=0x%llx, size=0x%llx)", counter, size);

		if (counter > std::numeric_limits<u64>::max() - size)
		{
			fmt::throw_exception("Metal draw resource uploaded byte counter overflow");
		}

		counter += size;
	}

	u32 checked_retained_draw_buffer_count(usz count)
	{
		rsx_log.trace("checked_retained_draw_buffer_count(count=0x%llx)", static_cast<u64>(count));

		if (count > std::numeric_limits<u32>::max())
		{
			fmt::throw_exception("Metal draw resource retained buffer count overflow");
		}

		return static_cast<u32>(count);
	}
}

namespace rsx::metal
{
	struct draw_buffer_slot
	{
		std::unique_ptr<buffer> storage;
		b8 in_flight = false;
	};

	struct draw_resource_manager::draw_resource_manager_impl
	{
		device& m_device;
		std::vector<std::unique_ptr<draw_buffer_slot>> m_buffers;
		std::unique_ptr<buffer> m_zero_vertex_buffer;
		mutable std::mutex m_mutex;
		draw_resource_stats m_stats{};

		explicit draw_resource_manager_impl(device& dev)
			: m_device(dev)
		{
		}
	};

	draw_resource_manager::draw_resource_manager(device& dev)
		: m_impl(std::make_unique<draw_resource_manager_impl>(dev))
	{
		rsx_log.notice("rsx::metal::draw_resource_manager::draw_resource_manager(device=*0x%x)", dev.handle());

		buffer_desc desc =
		{
			.size = 16,
			.label = "RPCS3 Metal zero vertex input",
			.storage = buffer_storage::shared,
			.cpu_cache = buffer_cpu_cache_mode::write_combined,
			.hazard_tracking = resource_hazard_tracking::untracked,
		};

		m_impl->m_zero_vertex_buffer = std::make_unique<buffer>(dev, std::move(desc));
		std::memset(m_impl->m_zero_vertex_buffer->map(), 0, 16);
	}

	draw_resource_manager::~draw_resource_manager()
	{
		rsx_log.notice("rsx::metal::draw_resource_manager::~draw_resource_manager()");
	}

	draw_buffer_binding draw_resource_manager::upload_generated_buffer(command_frame& frame, u64 size, const std::string& label, const std::function<void(void*, u64)>& fill)
	{
		rsx_log.trace("rsx::metal::draw_resource_manager::upload_generated_buffer(frame_index=%u, size=0x%llx, label=%s)",
			frame.frame_index(),
			size,
			label.c_str());

		if (!fill)
		{
			fmt::throw_exception("Metal draw resource upload requires a valid fill callback");
		}

		const u64 allocation_size = checked_draw_buffer_size(size);

		std::lock_guard lock(m_impl->m_mutex);

		draw_buffer_slot* selected = nullptr;
		for (const std::unique_ptr<draw_buffer_slot>& slot : m_impl->m_buffers)
		{
			if (!slot || slot->in_flight || !slot->storage || slot->storage->length() < allocation_size)
			{
				continue;
			}

			selected = slot.get();
			increment_draw_resource_counter(m_impl->m_stats.reused_buffer_count, "reused draw buffer");
			break;
		}

		if (!selected)
		{
			buffer_desc desc =
			{
				.size = allocation_size,
				.label = label,
				.storage = buffer_storage::shared,
				.cpu_cache = buffer_cpu_cache_mode::write_combined,
				.hazard_tracking = resource_hazard_tracking::untracked,
			};

			auto slot = std::make_unique<draw_buffer_slot>();
			slot->storage = std::make_unique<buffer>(m_impl->m_device, std::move(desc));
			selected = slot.get();
			m_impl->m_buffers.emplace_back(std::move(slot));
			increment_draw_resource_counter(m_impl->m_stats.created_buffer_count, "created draw buffer");
			m_impl->m_stats.retained_buffer_count = checked_retained_draw_buffer_count(m_impl->m_buffers.size());
		}

		void* data = selected->storage->map();
		std::memset(data, 0, static_cast<usz>(size));
		fill(data, size);
		selected->in_flight = true;

		draw_resource_manager_impl* impl = m_impl.get();
		frame.on_completed([impl, selected]()
		{
			std::lock_guard lock(impl->m_mutex);
			selected->in_flight = false;
		});

		increment_draw_resource_counter(m_impl->m_stats.uploaded_buffer_count, "uploaded draw buffer");
		add_uploaded_draw_bytes(m_impl->m_stats.uploaded_byte_count, size);

		return
		{
			.resource = selected->storage.get(),
			.offset = 0,
			.size = size,
		};
	}

	buffer& draw_resource_manager::zero_vertex_buffer() const
	{
		rsx_log.trace("rsx::metal::draw_resource_manager::zero_vertex_buffer()");

		if (!m_impl->m_zero_vertex_buffer)
		{
			fmt::throw_exception("Metal draw resource zero vertex buffer is not initialized");
		}

		return *m_impl->m_zero_vertex_buffer;
	}

	draw_resource_stats draw_resource_manager::stats() const
	{
		rsx_log.trace("rsx::metal::draw_resource_manager::stats()");

		std::lock_guard lock(m_impl->m_mutex);
		draw_resource_stats result = m_impl->m_stats;
		result.retained_buffer_count = checked_retained_draw_buffer_count(m_impl->m_buffers.size());
		return result;
	}

	void draw_resource_manager::report() const
	{
		rsx_log.notice("rsx::metal::draw_resource_manager::report()");

		const draw_resource_stats resource_stats = stats();
		rsx_log.notice("Metal draw resources: retained=%u, created=%u, reused=%u, uploads=%u, uploaded_bytes=0x%llx",
			resource_stats.retained_buffer_count,
			resource_stats.created_buffer_count,
			resource_stats.reused_buffer_count,
			resource_stats.uploaded_buffer_count,
			resource_stats.uploaded_byte_count);
	}
}
