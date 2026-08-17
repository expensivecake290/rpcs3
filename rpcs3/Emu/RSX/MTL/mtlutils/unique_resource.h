#pragma once

#include <utility>

#include "util/types.hpp"

namespace mtl
{
	[[nodiscard]] u64 generate_resource_uid();
	[[nodiscard]] void* retain_native_object(void* object);
	void release_native_object(void* object);

	class unique_resource
	{
		u64 m_uid;

	public:
		unique_resource();
		unique_resource(const unique_resource&) = delete;
		unique_resource& operator=(const unique_resource&) = delete;
		unique_resource(unique_resource&& other) noexcept;
		unique_resource& operator=(unique_resource&& other) noexcept;

		[[nodiscard]] u64 uid() const
		{
			return m_uid;
		}

		[[nodiscard]] bool operator==(const unique_resource& other) const
		{
			return m_uid == other.m_uid;
		}
	};

	class unique_native_resource : public unique_resource
	{
		void* m_object = nullptr;

	public:
		unique_native_resource() = default;
		explicit unique_native_resource(void* object);
		~unique_native_resource();
		unique_native_resource(const unique_native_resource&) = delete;
		unique_native_resource& operator=(const unique_native_resource&) = delete;
		unique_native_resource(unique_native_resource&& other) noexcept;
		unique_native_resource& operator=(unique_native_resource&& other) noexcept;

		void reset(void* object = nullptr);
		[[nodiscard]] void* release();

		[[nodiscard]] void* get() const
		{
			return m_object;
		}

		[[nodiscard]] explicit operator bool() const
		{
			return m_object != nullptr;
		}
	};

	template <typename Handle>
	class unique_metal_resource : public unique_native_resource
	{
		static void* to_opaque(Handle object)
		{
#ifdef __OBJC__
			return (__bridge void*)object;
#else
			return static_cast<void*>(object);
#endif
		}

		static Handle from_opaque(void* object)
		{
#ifdef __OBJC__
			return (__bridge Handle)object;
#else
			return static_cast<Handle>(object);
#endif
		}

		static Handle from_released_opaque(void* object)
		{
#ifdef __OBJC__
			return (__bridge_transfer Handle)object;
#else
			return static_cast<Handle>(object);
#endif
		}

	public:
		unique_metal_resource() = default;

		explicit unique_metal_resource(Handle object)
			: unique_native_resource(to_opaque(object))
		{
		}

		void reset(Handle object = nullptr)
		{
			unique_native_resource::reset(to_opaque(object));
		}

		[[nodiscard]] Handle get() const
		{
			return from_opaque(unique_native_resource::get());
		}

		[[nodiscard]] Handle release()
		{
			return from_released_opaque(unique_native_resource::release());
		}
	};
}
