#include "stdafx.h"
#include "sampler.h"

#include "shared.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <bit>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace mtl
{
	struct sampler::impl
	{
		id<MTLSamplerState> state;
		sampler_description creation;
		sampler_shader_state shader_parameters;
		std::atomic<u64> last_frame = 0;
	};

	struct sampler_pool::impl
	{
		const render_device* render = nullptr;
		std::unordered_map<sampler_key, std::shared_ptr<sampler>, sampler_key_hash> entries;
		mutable std::mutex mutex;
	};

	namespace
	{
		bool uses_border(sampler_address_mode mode)
		{
			return mode == sampler_address_mode::border || mode == sampler_address_mode::clamp ||
				mode == sampler_address_mode::mirror_once_border || mode == sampler_address_mode::mirror_once_clamp;
		}

		bool finite_color(const std::array<f32, 4>& color)
		{
			return std::all_of(color.begin(), color.end(), [](f32 value)
			{
				return std::isfinite(value);
			});
		}

		sampler_description normalize_description(const sampler_description& input)
		{
			sampler_description result = input;
			if (!std::isfinite(result.lod_bias) || !std::isfinite(result.min_lod) || !std::isfinite(result.max_lod) ||
				result.min_lod > result.max_lod || result.max_anisotropy == 0 || result.max_anisotropy > 16 ||
				!finite_color(result.border.value))
			{
				fmt::throw_exception("Invalid Metal sampler numeric state");
			}

			if (!result.compare_enabled)
			{
				result.compare = sampler_compare_function::never;
			}
			if (result.mip_filter == sampler_mip_filter::none)
			{
				result.lod_bias = 0.f;
				result.min_lod = 0.f;
				result.max_lod = 0.f;
			}

			switch (result.border.kind)
			{
			case border_color_kind::transparent_black:
				result.border.value = {0.f, 0.f, 0.f, 0.f};
				result.border.native_format = 0;
				result.border.aspects = 0;
				result.border.integer = false;
				break;
			case border_color_kind::opaque_black:
				result.border.value = {0.f, 0.f, 0.f, 1.f};
				result.border.native_format = 0;
				result.border.aspects = 0;
				result.border.integer = false;
				break;
			case border_color_kind::opaque_white:
				result.border.value = {1.f, 1.f, 1.f, 1.f};
				result.border.native_format = 0;
				result.border.aspects = 0;
				result.border.integer = false;
				break;
			case border_color_kind::custom:
				if (result.border.aspects == 0)
				{
					fmt::throw_exception("A custom Metal border color requires texture aspects");
				}
				break;
			}

			if (!uses_border(result.address_s) && !uses_border(result.address_t) && !uses_border(result.address_r))
			{
				result.border = border_color::transparent_black();
			}
			return result;
		}

		MTLSamplerAddressMode native_address_mode(sampler_address_mode mode)
		{
			switch (mode)
			{
			case sampler_address_mode::wrap: return MTLSamplerAddressModeRepeat;
			case sampler_address_mode::mirror: return MTLSamplerAddressModeMirrorRepeat;
			case sampler_address_mode::clamp_to_edge: return MTLSamplerAddressModeClampToEdge;
			case sampler_address_mode::clamp: return MTLSamplerAddressModeClampToBorderColor;
			case sampler_address_mode::border: return MTLSamplerAddressModeClampToBorderColor;
			case sampler_address_mode::mirror_once_clamp_to_edge:
				return MTLSamplerAddressModeMirrorClampToEdge;
			case sampler_address_mode::mirror_once_border:
			case sampler_address_mode::mirror_once_clamp:
				return MTLSamplerAddressModeClampToBorderColor;
			}
			fmt::throw_exception("Invalid Metal sampler address mode %u", static_cast<u8>(mode));
		}

		MTLSamplerMinMagFilter native_filter(sampler_filter filter)
		{
			return filter == sampler_filter::linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
		}

		MTLSamplerMipFilter native_mip_filter(sampler_mip_filter filter)
		{
			switch (filter)
			{
			case sampler_mip_filter::none: return MTLSamplerMipFilterNotMipmapped;
			case sampler_mip_filter::nearest: return MTLSamplerMipFilterNearest;
			case sampler_mip_filter::linear: return MTLSamplerMipFilterLinear;
			}
			fmt::throw_exception("Invalid Metal sampler mip filter %u", static_cast<u8>(filter));
		}

		MTLCompareFunction native_compare(sampler_compare_function compare)
		{
			switch (compare)
			{
			case sampler_compare_function::never: return MTLCompareFunctionNever;
			case sampler_compare_function::less: return MTLCompareFunctionLess;
			case sampler_compare_function::equal: return MTLCompareFunctionEqual;
			case sampler_compare_function::less_equal: return MTLCompareFunctionLessEqual;
			case sampler_compare_function::greater: return MTLCompareFunctionGreater;
			case sampler_compare_function::not_equal: return MTLCompareFunctionNotEqual;
			case sampler_compare_function::greater_equal: return MTLCompareFunctionGreaterEqual;
			case sampler_compare_function::always: return MTLCompareFunctionAlways;
			}
			fmt::throw_exception("Invalid Metal sampler comparison function %u", static_cast<u8>(compare));
		}

		MTLSamplerBorderColor native_border(border_color_kind color)
		{
			switch (color)
			{
			case border_color_kind::transparent_black:
			case border_color_kind::custom:
				return MTLSamplerBorderColorTransparentBlack;
			case border_color_kind::opaque_black:
				return MTLSamplerBorderColorOpaqueBlack;
			case border_color_kind::opaque_white:
				return MTLSamplerBorderColorOpaqueWhite;
			}
			fmt::throw_exception("Invalid Metal sampler border color %u", static_cast<u8>(color));
		}

		u32 emulation_flags(const sampler_description& description)
		{
			u32 result = sampler_emulation_none;
			for (sampler_address_mode mode : {description.address_s, description.address_t, description.address_r})
			{
				if (mode == sampler_address_mode::mirror_once_border) result |= sampler_emulation_mirror_once_border;
				if (mode == sampler_address_mode::clamp) result |= sampler_emulation_rsx_clamp;
				if (mode == sampler_address_mode::mirror_once_clamp) result |= sampler_emulation_mirror_once_clamp;
			}
			if (description.border.is_custom() &&
				(uses_border(description.address_s) || uses_border(description.address_t) || uses_border(description.address_r)))
			{
				result |= sampler_emulation_custom_border;
				if (description.border.integer) result |= sampler_emulation_integer_border;
			}
			if (!description.normalized_coordinates) result |= sampler_emulation_unnormalized_coordinates;

			constexpr f32 minimum_bias = -16.f;
			constexpr f32 maximum_bias = 1023.f / 64.f;
			const f32 encoded_bias = std::round(description.lod_bias * 64.f) / 64.f;
			if (description.lod_bias < minimum_bias || description.lod_bias > maximum_bias || encoded_bias != description.lod_bias)
			{
				result |= sampler_emulation_lod_bias;
			}
			return result;
		}

		u64 canonical_float_bits(f32 value)
		{
			return value == 0.f ? 0 : std::bit_cast<u32>(value);
		}

		void hash_combine(usz& seed, u64 value) noexcept
		{
			seed ^= static_cast<usz>(value) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
		}
	}

	bool border_color::is_custom() const
	{
		return kind == border_color_kind::custom;
	}

	border_color border_color::transparent_black()
	{
		return {border_color_kind::transparent_black, {0.f, 0.f, 0.f, 0.f}, 0, 0, false};
	}

	border_color border_color::opaque_black()
	{
		return {border_color_kind::opaque_black, {0.f, 0.f, 0.f, 1.f}, 0, 0, false};
	}

	border_color border_color::opaque_white()
	{
		return {border_color_kind::opaque_white, {1.f, 1.f, 1.f, 1.f}, 0, 0, false};
	}

	border_color border_color::custom(std::array<f32, 4> color, u64 format, u8 aspects, bool integer)
	{
		border_color result{border_color_kind::custom, color, format, aspects, integer};
		if (!finite_color(color) || aspects == 0)
		{
			fmt::throw_exception("Invalid custom Metal sampler border color");
		}
		return result;
	}

	usz sampler_key_hash::operator()(const sampler_key& key) const noexcept
	{
		const sampler_description& value = key.description;
		usz result = static_cast<u8>(value.address_s) |
			(static_cast<usz>(static_cast<u8>(value.address_t)) << 4) |
			(static_cast<usz>(static_cast<u8>(value.address_r)) << 8) |
			(static_cast<usz>(static_cast<u8>(value.min_filter)) << 12) |
			(static_cast<usz>(static_cast<u8>(value.mag_filter)) << 13) |
			(static_cast<usz>(static_cast<u8>(value.mip_filter)) << 14) |
			(static_cast<usz>(static_cast<u8>(value.compare)) << 16) |
			(static_cast<usz>(value.compare_enabled) << 19) |
			(static_cast<usz>(value.normalized_coordinates) << 20);
		hash_combine(result, canonical_float_bits(value.lod_bias));
		hash_combine(result, canonical_float_bits(value.min_lod));
		hash_combine(result, canonical_float_bits(value.max_lod));
		hash_combine(result, value.max_anisotropy);
		hash_combine(result, static_cast<u8>(value.border.kind));
		for (f32 channel : value.border.value) hash_combine(result, canonical_float_bits(channel));
		hash_combine(result, value.border.native_format);
		hash_combine(result, value.border.aspects);
		hash_combine(result, value.border.integer);
		return result;
	}

	sampler::sampler()
		: m_impl(std::make_unique<impl>())
	{
	}

	sampler::sampler(const render_device& device, const sampler_description& description, std::string_view label)
		: sampler()
	{
		create(device, description, label);
	}

	sampler::~sampler()
	{
		destroy();
	}

	void sampler::create(const render_device& device, const sampler_description& description, std::string_view label)
	{
		destroy();
		if (!device)
		{
			fmt::throw_exception("Cannot create a Metal sampler without a render device");
		}
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		m_impl->creation = normalize_description(description);
		m_impl->shader_parameters.border_color = m_impl->creation.border.value;
		m_impl->shader_parameters.emulation_flags = mtl::emulation_flags(m_impl->creation);
		m_impl->shader_parameters.lod_bias =
			(m_impl->shader_parameters.emulation_flags & sampler_emulation_lod_bias) ? m_impl->creation.lod_bias : 0.f;
		m_impl->shader_parameters.address_modes = static_cast<u32>(m_impl->creation.address_s) |
			(static_cast<u32>(m_impl->creation.address_t) << 4) |
			(static_cast<u32>(m_impl->creation.address_r) << 8);
		m_impl->shader_parameters.border_metadata = m_impl->creation.border.aspects |
			(static_cast<u32>(m_impl->creation.border.integer) << 8) |
			(static_cast<u32>(m_impl->creation.min_filter == sampler_filter::linear) << 16) |
			(static_cast<u32>(m_impl->creation.mag_filter == sampler_filter::linear) << 17) |
			(static_cast<u32>(m_impl->creation.mip_filter) << 18) |
			((static_cast<u32>(static_cast<s32>(std::round(m_impl->creation.lod_bias * 64.f))) &
				0x7ffu) << 20);

		MTLSamplerDescriptor* native = [MTLSamplerDescriptor new];
		native.label = [NSString stringWithUTF8String:std::string(label).c_str()];
		native.minFilter = native_filter(m_impl->creation.min_filter);
		native.magFilter = native_filter(m_impl->creation.mag_filter);
		native.mipFilter = native_mip_filter(m_impl->creation.mip_filter);
		native.maxAnisotropy = m_impl->creation.max_anisotropy;
		native.sAddressMode = native_address_mode(m_impl->creation.address_s);
		native.tAddressMode = native_address_mode(m_impl->creation.address_t);
		native.rAddressMode = native_address_mode(m_impl->creation.address_r);
		native.borderColor = native_border(m_impl->creation.border.kind);
		native.normalizedCoordinates = (m_impl->shader_parameters.emulation_flags & sampler_emulation_unnormalized_coordinates) ? YES :
			m_impl->creation.normalized_coordinates;
		native.lodMinClamp = m_impl->creation.min_lod;
		native.lodMaxClamp = m_impl->creation.max_lod;
		native.lodBias = (m_impl->shader_parameters.emulation_flags & sampler_emulation_lod_bias) ? 0.f : m_impl->creation.lod_bias;
		native.compareFunction = m_impl->creation.compare_enabled ? native_compare(m_impl->creation.compare) : MTLCompareFunctionNever;
		native.supportArgumentBuffers = YES;
		m_impl->state = [device.native_handle() newSamplerStateWithDescriptor:native];
		if (!m_impl->state)
		{
			m_impl->creation = {};
			m_impl->shader_parameters = {};
			fmt::throw_exception("Metal failed to create sampler '%s'", label);
		}
		m_impl->last_frame.store(get_shared_state() ? get_frame_id() : 0, std::memory_order_relaxed);
	}

	void sampler::destroy()
	{
		if (m_impl)
		{
			m_impl->state = nil;
			m_impl->creation = {};
			m_impl->shader_parameters = {};
			m_impl->last_frame.store(0, std::memory_order_relaxed);
		}
	}

	sampler::operator bool() const { return m_impl && m_impl->state; }
	sampler_handle sampler::native_handle() const { return m_impl ? m_impl->state : nil; }

	const sampler_description& sampler::description() const
	{
		if (!*this) fmt::throw_exception("Description requested from an empty Metal sampler");
		return m_impl->creation;
	}

	sampler_shader_state sampler::shader_state() const
	{
		if (!*this) fmt::throw_exception("Shader state requested from an empty Metal sampler");
		return m_impl->shader_parameters;
	}

	u32 sampler::emulation_flags() const
	{
		return shader_state().emulation_flags;
	}

	bool sampler::matches(const sampler_description& description) const
	{
		return *this && m_impl->creation == normalize_description(description);
	}

	void sampler::touch(u64 frame_id)
	{
		if (!*this) fmt::throw_exception("Cannot touch an empty Metal sampler");
		m_impl->last_frame.store(frame_id, std::memory_order_relaxed);
	}

	u64 sampler::last_used_frame() const
	{
		return m_impl ? m_impl->last_frame.load(std::memory_order_relaxed) : 0;
	}

	sampler_pool::sampler_pool()
		: m_impl(std::make_unique<impl>())
	{
	}

	sampler_pool::~sampler_pool()
	{
		destroy();
	}

	void sampler_pool::create(const render_device& device)
	{
		destroy();
		if (!device)
		{
			fmt::throw_exception("Cannot create a Metal sampler pool without a render device");
		}
		if (!m_impl) m_impl = std::make_unique<impl>();
		m_impl->render = &device;
	}

	void sampler_pool::destroy()
	{
		if (!m_impl) return;
		clear();
		std::lock_guard lock(m_impl->mutex);
		m_impl->render = nullptr;
	}

	std::shared_ptr<sampler> sampler_pool::get(const sampler_description& description, std::string_view label)
	{
		if (!m_impl || !m_impl->render)
		{
			fmt::throw_exception("Cannot get a sampler from an empty Metal sampler pool");
		}
		const sampler_key key{normalize_description(description)};
		std::lock_guard lock(m_impl->mutex);
		if (const auto found = m_impl->entries.find(key); found != m_impl->entries.end())
		{
			found->second->touch(get_shared_state() ? get_frame_id() : 0);
			return found->second;
		}

		auto result = std::make_shared<sampler>(*m_impl->render, key.description, label);
		result->touch(get_shared_state() ? get_frame_id() : 0);
		m_impl->entries.emplace(key, result);
		return result;
	}

	std::shared_ptr<sampler> sampler_pool::find(const sampler_description& description) const
	{
		if (!m_impl || !m_impl->render) return {};
		const sampler_key key{normalize_description(description)};
		std::lock_guard lock(m_impl->mutex);
		const auto found = m_impl->entries.find(key);
		if (found == m_impl->entries.end()) return {};
		found->second->touch(get_shared_state() ? get_frame_id() : 0);
		return found->second;
	}

	std::vector<std::shared_ptr<sampler>> sampler_pool::collect(
		const std::function<bool(const sampler&)>& predicate)
	{
		if (!predicate)
		{
			fmt::throw_exception("Metal sampler collection requires a predicate");
		}
		std::vector<std::shared_ptr<sampler>> snapshot;
		{
			std::lock_guard lock(m_impl->mutex);
			for (const auto& [key, value] : m_impl->entries)
			{
				static_cast<void>(key);
				snapshot.push_back(value);
			}
		}

		std::unordered_set<const sampler*> selected;
		for (const auto& value : snapshot)
		{
			if (predicate(*value)) selected.insert(value.get());
		}

		std::vector<std::shared_ptr<sampler>> result;
		std::lock_guard lock(m_impl->mutex);
		for (auto iterator = m_impl->entries.begin(); iterator != m_impl->entries.end();)
		{
			if (!selected.contains(iterator->second.get()))
			{
				++iterator;
				continue;
			}
			result.push_back(std::move(iterator->second));
			iterator = m_impl->entries.erase(iterator);
		}
		return result;
	}

	void sampler_pool::clear()
	{
		if (!m_impl) return;
		std::unordered_map<sampler_key, std::shared_ptr<sampler>, sampler_key_hash> entries;
		{
			std::lock_guard lock(m_impl->mutex);
			entries.swap(m_impl->entries);
		}
		entries.clear();
	}

	usz sampler_pool::size() const
	{
		if (!m_impl) return 0;
		std::lock_guard lock(m_impl->mutex);
		return m_impl->entries.size();
	}
}
