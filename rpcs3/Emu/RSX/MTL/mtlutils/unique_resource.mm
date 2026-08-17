#include "stdafx.h"
#include "unique_resource.h"

#import <CoreFoundation/CoreFoundation.h>
#import <Foundation/Foundation.h>

#include "util/atomic.hpp"

namespace mtl
{
	namespace
	{
		atomic_t<u64> s_resource_uid{1};
	}

	u64 generate_resource_uid()
	{
		const u64 result = s_resource_uid.fetch_add(1);
		if (result == 0)
		{
			fmt::throw_exception("Metal resource UID space was exhausted");
		}

		return result;
	}

	void* retain_native_object(void* object)
	{
		if (!object)
		{
			return nullptr;
		}

		id native_object = (__bridge id)object;
		return const_cast<void*>(static_cast<const void*>(CFBridgingRetain(native_object)));
	}

	void release_native_object(void* object)
	{
		if (object)
		{
			CFRelease(static_cast<CFTypeRef>(object));
		}
	}

	unique_resource::unique_resource()
		: m_uid(generate_resource_uid())
	{
	}

	unique_resource::unique_resource(unique_resource&& other) noexcept
		: m_uid(std::exchange(other.m_uid, 0))
	{
	}

	unique_resource& unique_resource::operator=(unique_resource&& other) noexcept
	{
		if (this != &other)
		{
			m_uid = std::exchange(other.m_uid, 0);
		}
		return *this;
	}

	unique_native_resource::unique_native_resource(void* object)
		: m_object(retain_native_object(object))
	{
	}

	unique_native_resource::~unique_native_resource()
	{
		release_native_object(m_object);
	}

	unique_native_resource::unique_native_resource(unique_native_resource&& other) noexcept
		: unique_resource(std::move(other))
		, m_object(std::exchange(other.m_object, nullptr))
	{
	}

	unique_native_resource& unique_native_resource::operator=(unique_native_resource&& other) noexcept
	{
		if (this != &other)
		{
			release_native_object(m_object);
			unique_resource::operator=(std::move(other));
			m_object = std::exchange(other.m_object, nullptr);
		}

		return *this;
	}

	void unique_native_resource::reset(void* object)
	{
		if (object == m_object)
		{
			return;
		}

		void* retained = retain_native_object(object);
		release_native_object(m_object);
		m_object = retained;
	}

	void* unique_native_resource::release()
	{
		return std::exchange(m_object, nullptr);
	}
}
