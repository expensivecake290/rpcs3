#pragma once

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "mtlutils/garbage_collector.h"
#include "mtlutils/memory.h"

namespace mtl
{
	class shared_state;

	enum class managed_resource_class : u8
	{
		unknown,
		buffer,
		texture,
		texture_view,
		sampler,
		pipeline,
		argument_table,
		query,
		command,
		native_object,
		count,
	};

	struct resource_retirement
	{
		u64 submission = 0;
		u64 frame = 0;

		[[nodiscard]] explicit operator bool() const
		{
			return submission != 0;
		}
	};

	struct deferred_resource_info
	{
		managed_resource_class resource_class = managed_resource_class::unknown;
		u64 bytes = 0;
		std::string label;
	};

	struct resource_debug_marker
	{
		u64 id = 0;
		u64 submission = 0;
		u64 frame = 0;
		std::string label;
		std::string detail;
	};

	struct resource_manager_statistics
	{
		u64 current_submission = 0;
		u64 completed_submission = 0;
		u64 retired_count = 0;
		u64 released_count = 0;
		u64 pending_count = 0;
		u64 pending_bytes = 0;
		u64 peak_pending_bytes = 0;
		u64 marker_count = 0;
		u64 trim_count = 0;
		u64 shutdown_callbacks = 0;
	};

	class resource_manager
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		resource_manager();
		~resource_manager();
		resource_manager(const resource_manager&) = delete;
		resource_manager& operator=(const resource_manager&) = delete;

		void initialize(shared_state& state);
		void destroy(u64 completed_submission, bool device_is_idle);

		void set_retirement_point(resource_retirement point);
		[[nodiscard]] resource_retirement retirement_point() const;

		void retire(disposable&& object, const deferred_resource_info& info = {});
		void retire(disposable&& object, resource_retirement point,
			const deferred_resource_info& info = {});

		template <typename T>
		void retire(std::unique_ptr<T>& object, const deferred_resource_info& info = {})
		{
			if (object)
			{
				retire(disposable::make(object.release()), info);
			}
		}

		template <typename T>
		void retire(std::unique_ptr<T>& object, resource_retirement point,
			const deferred_resource_info& info = {})
		{
			if (object)
			{
				retire(disposable::make(object.release()), point, info);
			}
		}

		void defer(std::function<void()> release, const deferred_resource_info& info = {});
		void defer(std::function<void()> release, resource_retirement point,
			const deferred_resource_info& info = {});

		[[nodiscard]] u64 add_debug_marker(std::string label, std::string detail = {});
		void remove_debug_marker(u64 marker_id);
		[[nodiscard]] std::vector<resource_debug_marker> debug_markers() const;

		void complete(u64 completed_submission);
		void flush_completed(u64 completed_submission);
		void trim(memory_pressure pressure, u64 completed_submission);
		void add_shutdown_callback(std::function<void()> callback);

		[[nodiscard]] resource_manager_statistics statistics() const;
		[[nodiscard]] explicit operator bool() const;
	};

	[[nodiscard]] resource_manager& get_resource_manager();
	[[nodiscard]] u64 allocate_resource_event();
	[[nodiscard]] u64 current_resource_event();
	[[nodiscard]] u64 last_completed_resource_event();
	void notify_resource_event_completed(u64 event);
}
