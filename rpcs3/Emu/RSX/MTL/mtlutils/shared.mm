#include "stdafx.h"
#include "shared.h"

#include "util/atomic.hpp"

#include <mutex>
#include <set>

namespace mtl
{
	struct shared_state::impl
	{
		mutable std::mutex lifecycle_mutex;
		std::mutex completion_mutex;
		render_device render;
		std::unique_ptr<memory_allocator> memory;
		residency_set residency_resources;
		garbage_collector garbage_resources;
		atomic_t<u64> frame{0};
		atomic_t<u64> submission{0};
		atomic_t<u64> completed{0};
		std::set<u64> completion_ready;
		bool initialized = false;
	};

	shared_state::shared_state()
		: m_impl(std::make_unique<impl>())
	{
	}

	shared_state::~shared_state()
	{
		shutdown();
	}

	void shared_state::initialize(std::string_view preferred_device)
	{
		std::lock_guard lock(m_impl->lifecycle_mutex);
		if (m_impl->initialized)
		{
			fmt::throw_exception("Metal shared state is already initialized");
		}

		const std::vector<physical_device> devices = enumerate_devices();
		physical_device selected = select_device(devices, preferred_device);
		try
		{
			m_impl->render.create(selected);
			m_impl->memory = std::make_unique<memory_allocator>(m_impl->render);
			m_impl->residency_resources.create(m_impl->render, "RPCS3 Metal renderer resources", 4096);
			m_impl->residency_resources.attach(m_impl->render.graphics_queue());
			if (m_impl->render.transfer_queue() != m_impl->render.graphics_queue())
			{
				m_impl->residency_resources.attach(m_impl->render.transfer_queue());
			}
			m_impl->residency_resources.request_residency();
			m_impl->frame.store(0);
			m_impl->submission.store(0);
			m_impl->completed.store(0);
			m_impl->completion_ready.clear();
			reset_memory_tracking();
			m_impl->initialized = true;
		}
		catch (...)
		{
			m_impl->residency_resources.destroy();
			m_impl->memory.reset();
			m_impl->render.destroy();
			throw;
		}
	}

	void shared_state::shutdown()
	{
		if (!m_impl)
		{
			return;
		}

		std::lock_guard lock(m_impl->lifecycle_mutex);
		if (!m_impl->initialized)
		{
			return;
		}

		const u64 submitted = m_impl->submission.load();
		const u64 completed = m_impl->completed.load();
		ensure(completed == submitted);

		m_impl->garbage_resources.complete(submitted);
		m_impl->garbage_resources.drain();
		m_impl->residency_resources.end_residency();
		m_impl->residency_resources.destroy();
		m_impl->memory.reset();
		reset_memory_tracking();
		m_impl->render.destroy();
		m_impl->frame.store(0);
		m_impl->submission.store(0);
		m_impl->completed.store(0);
		m_impl->completion_ready.clear();
		m_impl->initialized = false;
	}

	shared_state::operator bool() const
	{
		return m_impl && m_impl->initialized;
	}

	render_device& shared_state::device()
	{
		if (!*this)
		{
			fmt::throw_exception("Metal device requested before shared-state initialization");
		}
		return m_impl->render;
	}

	const render_device& shared_state::device() const
	{
		if (!*this)
		{
			fmt::throw_exception("Metal device requested before shared-state initialization");
		}
		return m_impl->render;
	}

	memory_allocator& shared_state::allocator()
	{
		if (!*this || !m_impl->memory)
		{
			fmt::throw_exception("Metal allocator requested before shared-state initialization");
		}
		return *m_impl->memory;
	}

	residency_set& shared_state::residency()
	{
		if (!*this)
		{
			fmt::throw_exception("Metal residency set requested before shared-state initialization");
		}
		return m_impl->residency_resources;
	}

	garbage_collector& shared_state::garbage()
	{
		if (!*this)
		{
			fmt::throw_exception("Metal garbage collector requested before shared-state initialization");
		}
		return m_impl->garbage_resources;
	}

	u64 shared_state::begin_frame()
	{
		if (!*this)
		{
			fmt::throw_exception("Cannot begin a Metal frame before shared-state initialization");
		}
		return m_impl->frame.fetch_add(1) + 1;
	}

	u64 shared_state::next_submission()
	{
		if (!*this)
		{
			fmt::throw_exception("Cannot allocate a Metal submission ID before shared-state initialization");
		}

		const u64 value = m_impl->submission.fetch_add(1) + 1;
		m_impl->garbage_resources.set_retirement_value(value);
		return value;
	}

	void shared_state::notify_submission_completed(u64 value)
	{
		if (!*this || value == 0 || value > m_impl->submission.load())
		{
			fmt::throw_exception("Invalid completed Metal submission value %llu", value);
		}

		{
			std::lock_guard lock(m_impl->completion_mutex);
			if (value <= m_impl->completed.load())
			{
				return;
			}

			m_impl->completion_ready.insert(value);
			u64 contiguous = m_impl->completed.load();
			while (m_impl->completion_ready.erase(contiguous + 1))
			{
				++contiguous;
			}
			m_impl->completed.store(contiguous);
		}
		m_impl->garbage_resources.complete(m_impl->completed.load());
	}

	u64 shared_state::current_frame() const
	{
		return m_impl ? m_impl->frame.load() : 0;
	}

	u64 shared_state::current_submission() const
	{
		return m_impl ? m_impl->submission.load() : 0;
	}

	u64 shared_state::completed_submission() const
	{
		return m_impl ? m_impl->completed.load() : 0;
	}

	shared_state& get_shared_state()
	{
		static shared_state state;
		return state;
	}

	garbage_collector* get_garbage_collector()
	{
		return &get_shared_state().garbage();
	}

	u64 get_frame_id()
	{
		return get_shared_state().current_frame();
	}

	u64 get_submission_id()
	{
		return get_shared_state().current_submission();
	}

	u64 get_completed_submission_id()
	{
		return get_shared_state().completed_submission();
	}
}
