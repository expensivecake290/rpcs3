#pragma once

#include <mutex>
#include <unordered_map>

#include "Utilities/StrFmt.h"
#include "swapchain.h"

namespace mtl
{
	class swapchain_core
	{
		struct tracked_drawable
		{
			drawable_handle drawable = nullptr;
			texture_handle texture = nullptr;
			u64 generation = 0;
		};

		swapchain_configuration m_configuration;
		swapchain_statistics m_statistics;
		std::unordered_map<u64, tracked_drawable> m_outstanding;
		mutable std::mutex m_mutex;
		u64 m_next_frame = 1;
		bool m_initialized = false;

		void require_initialized() const
		{
			if (!m_initialized)
			{
				fmt::throw_exception("Metal swapchain lifecycle is not initialized");
			}
		}

		static void validate_configuration(const swapchain_configuration& configuration)
		{
			if (!configuration.pixel_format || !configuration.width || !configuration.height ||
				configuration.maximum_drawables < 2 || configuration.maximum_drawables > 3 ||
				configuration.acquire_timeout < std::chrono::nanoseconds::zero() || configuration.label.empty())
			{
				fmt::throw_exception("Invalid Metal swapchain configuration");
			}
		}

		void invalidate(drawable_frame& frame) const
		{
			frame.drawable = nullptr;
			frame.texture = nullptr;
			frame.frame_id = 0;
			frame.status = drawable_acquire_status::unavailable;
		}

	public:
		swapchain_core() = default;
		swapchain_core(const swapchain_core&) = delete;
		swapchain_core& operator=(const swapchain_core&) = delete;
		swapchain_core(swapchain_core&&) = delete;
		swapchain_core& operator=(swapchain_core&&) = delete;

		void initialize(const swapchain_configuration& configuration, drawable_size actual_size)
		{
			validate_configuration(configuration);
			if (!actual_size)
			{
				fmt::throw_exception("Metal swapchain cannot initialize with an empty drawable size");
			}
			std::lock_guard lock(m_mutex);
			m_configuration = configuration;
			m_configuration.width = actual_size.width;
			m_configuration.height = actual_size.height;
			m_statistics = {};
			m_statistics.generation = 1;
			m_statistics.size = actual_size;
			m_outstanding.clear();
			m_next_frame = 1;
			m_initialized = true;
		}

		void shutdown()
		{
			std::lock_guard lock(m_mutex);
			m_outstanding.clear();
			m_configuration = {};
			m_statistics = {};
			m_next_frame = 1;
			m_initialized = false;
		}

		void reconfigure(const swapchain_configuration& configuration, drawable_size actual_size)
		{
			validate_configuration(configuration);
			if (!actual_size) fmt::throw_exception("Metal swapchain reconfiguration has an empty size");
			std::lock_guard lock(m_mutex);
			require_initialized();
			const bool changed = m_configuration != configuration ||
				m_statistics.size.width != actual_size.width || m_statistics.size.height != actual_size.height;
			m_configuration = configuration;
			m_configuration.width = actual_size.width;
			m_configuration.height = actual_size.height;
			if (changed)
			{
				m_statistics.generation++;
				m_statistics.resize_count++;
				m_statistics.size = actual_size;
			}
		}

		[[nodiscard]] bool resize(drawable_size actual_size)
		{
			if (!actual_size) return false;
			std::lock_guard lock(m_mutex);
			require_initialized();
			if (m_statistics.size.width == actual_size.width && m_statistics.size.height == actual_size.height)
			{
				return false;
			}
			m_statistics.size = actual_size;
			m_configuration.width = actual_size.width;
			m_configuration.height = actual_size.height;
			m_statistics.generation++;
			m_statistics.resize_count++;
			return true;
		}

		[[nodiscard]] drawable_frame acquire(drawable_handle drawable, texture_handle texture,
			drawable_size actual_size)
		{
			if (!drawable || !texture || !actual_size)
			{
				fmt::throw_exception("Cannot track an invalid Metal drawable");
			}
			std::lock_guard lock(m_mutex);
			require_initialized();
			if (m_statistics.size.width != actual_size.width || m_statistics.size.height != actual_size.height)
			{
				m_statistics.size = actual_size;
				m_configuration.width = actual_size.width;
				m_configuration.height = actual_size.height;
				m_statistics.generation++;
				m_statistics.resize_count++;
			}
			if (m_outstanding.size() >= m_configuration.maximum_drawables)
			{
				fmt::throw_exception("Metal drawable tracking exceeded the configured in-flight limit");
			}
			if (m_next_frame == 0) m_next_frame = 1;
			const u64 frame_id = m_next_frame++;
			m_outstanding.emplace(frame_id, tracked_drawable{drawable, texture, m_statistics.generation});
			m_statistics.acquired_frames++;
			m_statistics.in_flight_drawables = static_cast<u32>(m_outstanding.size());
			return {drawable, texture, actual_size, frame_id, m_statistics.generation,
				drawable_acquire_status::success};
		}

		void note_acquire_failure(drawable_acquire_status status)
		{
			std::lock_guard lock(m_mutex);
			require_initialized();
			if (status == drawable_acquire_status::timeout) m_statistics.timed_out_acquires++;
		}

		[[nodiscard]] bool is_current(const drawable_frame& frame) const
		{
			if (!frame) return false;
			std::lock_guard lock(m_mutex);
			if (!m_initialized) return false;
			const auto found = m_outstanding.find(frame.frame_id);
			return found != m_outstanding.end() && found->second.drawable == frame.drawable &&
				found->second.texture == frame.texture && found->second.generation == frame.generation;
		}

		[[nodiscard]] drawable_present_status complete(drawable_frame& frame,
			drawable_present_status status)
		{
			std::lock_guard lock(m_mutex);
			require_initialized();
			const auto found = m_outstanding.find(frame.frame_id);
			if (!frame || found == m_outstanding.end() || found->second.drawable != frame.drawable ||
				found->second.texture != frame.texture || found->second.generation != frame.generation)
			{
				fmt::throw_exception("Metal drawable is stale, foreign, or already completed");
			}
			if (status == drawable_present_status::success) m_statistics.presented_frames++;
			else m_statistics.dropped_frames++;
			if (frame.generation != m_statistics.generation && status == drawable_present_status::success)
			{
				status = drawable_present_status::resized;
			}
			m_outstanding.erase(found);
			m_statistics.in_flight_drawables = static_cast<u32>(m_outstanding.size());
			invalidate(frame);
			return status;
		}

		void discard(drawable_frame& frame)
		{
			static_cast<void>(complete(frame, drawable_present_status::dropped));
		}

		[[nodiscard]] bool initialized() const
		{
			std::lock_guard lock(m_mutex);
			return m_initialized;
		}

		[[nodiscard]] const swapchain_configuration& configuration() const
		{
			return m_configuration;
		}

		[[nodiscard]] drawable_size size() const
		{
			std::lock_guard lock(m_mutex);
			return m_statistics.size;
		}

		[[nodiscard]] u64 generation() const
		{
			std::lock_guard lock(m_mutex);
			return m_statistics.generation;
		}

		[[nodiscard]] swapchain_statistics statistics() const
		{
			std::lock_guard lock(m_mutex);
			return m_statistics;
		}
	};
}
