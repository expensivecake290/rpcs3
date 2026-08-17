#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "util/types.hpp"

namespace mtl
{
	class disposable
	{
		void* m_pointer = nullptr;
		std::function<void(void*)> m_deleter;

		disposable(void* pointer, std::function<void(void*)> deleter)
			: m_pointer(pointer)
			, m_deleter(std::move(deleter))
		{
		}

	public:
		disposable() = default;
		disposable(const disposable&) = delete;
		disposable& operator=(const disposable&) = delete;

		disposable(disposable&& other) noexcept
			: m_pointer(std::exchange(other.m_pointer, nullptr))
			, m_deleter(std::move(other.m_deleter))
		{
		}

		disposable& operator=(disposable&& other) noexcept
		{
			if (this != &other)
			{
				reset();
				m_pointer = std::exchange(other.m_pointer, nullptr);
				m_deleter = std::move(other.m_deleter);
			}
			return *this;
		}

		~disposable()
		{
			reset();
		}

		void reset()
		{
			if (m_pointer)
			{
				m_deleter(m_pointer);
				m_pointer = nullptr;
			}
		}

		[[nodiscard]] explicit operator bool() const
		{
			return m_pointer != nullptr;
		}

		template <typename T>
		static disposable make(T* pointer)
		{
			return disposable(pointer, [](void* raw)
			{
				delete static_cast<T*>(raw);
			});
		}

		template <typename T, typename Deleter>
		static disposable make(T* pointer, Deleter deleter)
		{
			return disposable(pointer, [deleter = std::move(deleter)](void* raw) mutable
			{
				deleter(static_cast<T*>(raw));
			});
		}
	};

	class garbage_collector
	{
		mutable std::mutex m_mutex;
		std::map<u64, std::vector<disposable>> m_retired;
		std::vector<std::function<void()>> m_exit_callbacks;
		u64 m_completed_value = 0;
		u64 m_retirement_value = 0;

	public:
		garbage_collector() = default;
		~garbage_collector()
		{
			drain();
		}

		garbage_collector(const garbage_collector&) = delete;
		garbage_collector& operator=(const garbage_collector&) = delete;

		void set_retirement_value(u64 value)
		{
			std::lock_guard lock(m_mutex);
			ensure(value >= m_retirement_value);
			m_retirement_value = value;
		}

		void dispose(disposable&& object)
		{
			if (!object)
			{
				return;
			}

			std::lock_guard lock(m_mutex);
			m_retired[m_retirement_value].push_back(std::move(object));
		}

		template <typename T>
		void dispose(std::unique_ptr<T>& object)
		{
			if (object)
			{
				dispose(disposable::make(object.release()));
			}
		}

		void complete(u64 value)
		{
			std::vector<disposable> ready;
			{
				std::lock_guard lock(m_mutex);
				ensure(value >= m_completed_value);
				m_completed_value = value;

				for (auto iterator = m_retired.begin(); iterator != m_retired.end() && iterator->first <= value;)
				{
					for (auto& object : iterator->second)
					{
						ready.push_back(std::move(object));
					}
					iterator = m_retired.erase(iterator);
				}
			}
			// Run arbitrary destructors outside the collector lock.
			ready.clear();
		}

		void add_exit_callback(std::function<void()> callback)
		{
			ensure(callback);
			std::lock_guard lock(m_mutex);
			m_exit_callbacks.push_back(std::move(callback));
		}

		void drain()
		{
			std::map<u64, std::vector<disposable>> retired;
			std::vector<std::function<void()>> callbacks;
			{
				std::lock_guard lock(m_mutex);
				retired.swap(m_retired);
				callbacks.swap(m_exit_callbacks);
			}

			retired.clear();
			for (auto iterator = callbacks.rbegin(); iterator != callbacks.rend(); ++iterator)
			{
				(*iterator)();
			}
		}

		[[nodiscard]] u64 completed_value() const
		{
			std::lock_guard lock(m_mutex);
			return m_completed_value;
		}

		[[nodiscard]] usz pending_count() const
		{
			std::lock_guard lock(m_mutex);
			usz result = 0;
			for (const auto& [value, objects] : m_retired)
			{
				static_cast<void>(value);
				result += objects.size();
			}
			return result;
		}
	};

	[[nodiscard]] garbage_collector* get_garbage_collector();
}
