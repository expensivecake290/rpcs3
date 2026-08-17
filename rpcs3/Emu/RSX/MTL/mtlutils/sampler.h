#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "device.h"
#include "unique_resource.h"

namespace mtl
{
	enum class sampler_address_mode : u8
	{
		wrap,
		mirror,
		clamp_to_edge,
		border,
		clamp,
		mirror_once_clamp_to_edge,
		mirror_once_border,
		mirror_once_clamp,
	};

	enum class sampler_filter : u8
	{
		nearest,
		linear,
	};

	enum class sampler_mip_filter : u8
	{
		none,
		nearest,
		linear,
	};

	enum class sampler_compare_function : u8
	{
		never,
		less,
		equal,
		less_equal,
		greater,
		not_equal,
		greater_equal,
		always,
	};

	enum class border_color_kind : u8
	{
		transparent_black,
		opaque_black,
		opaque_white,
		custom,
	};

	struct border_color
	{
		border_color_kind kind = border_color_kind::transparent_black;
		std::array<f32, 4> value{};
		u64 native_format = 0;
		u8 aspects = 0;
		bool integer = false;

		[[nodiscard]] bool operator==(const border_color&) const = default;
		[[nodiscard]] bool is_custom() const;
		[[nodiscard]] static border_color transparent_black();
		[[nodiscard]] static border_color opaque_black();
		[[nodiscard]] static border_color opaque_white();
		[[nodiscard]] static border_color custom(std::array<f32, 4> color, u64 format, u8 aspects, bool integer = false);
	};

	struct sampler_description
	{
		sampler_address_mode address_s = sampler_address_mode::wrap;
		sampler_address_mode address_t = sampler_address_mode::wrap;
		sampler_address_mode address_r = sampler_address_mode::wrap;
		sampler_filter min_filter = sampler_filter::nearest;
		sampler_filter mag_filter = sampler_filter::nearest;
		sampler_mip_filter mip_filter = sampler_mip_filter::nearest;
		f32 lod_bias = 0.f;
		f32 min_lod = 0.f;
		f32 max_lod = 1000.f;
		u32 max_anisotropy = 1;
		border_color border;
		sampler_compare_function compare = sampler_compare_function::never;
		bool compare_enabled = false;
		bool normalized_coordinates = true;

		[[nodiscard]] bool operator==(const sampler_description&) const = default;
	};

	enum sampler_emulation : u32
	{
		sampler_emulation_none = 0,
		sampler_emulation_custom_border = 1 << 0,
		sampler_emulation_mirror_once_border = 1 << 1,
		sampler_emulation_rsx_clamp = 1 << 2,
		sampler_emulation_mirror_once_clamp = 1 << 3,
		sampler_emulation_lod_bias = 1 << 4,
		sampler_emulation_unnormalized_coordinates = 1 << 5,
		sampler_emulation_integer_border = 1 << 6,
	};

	struct alignas(16) sampler_shader_state
	{
		std::array<f32, 4> border_color{};
		f32 lod_bias = 0.f;
		u32 emulation_flags = sampler_emulation_none;
		u32 address_modes = 0;
		u32 border_metadata = 0;
	};

	struct sampler_key
	{
		sampler_description description;

		[[nodiscard]] bool operator==(const sampler_key&) const = default;
	};

	struct sampler_key_hash
	{
		[[nodiscard]] usz operator()(const sampler_key& key) const noexcept;
	};

	class sampler : public unique_resource
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		sampler();
		sampler(const render_device& device, const sampler_description& description, std::string_view label);
		~sampler();
		sampler(const sampler&) = delete;
		sampler& operator=(const sampler&) = delete;
		sampler(sampler&&) = delete;
		sampler& operator=(sampler&&) = delete;

		void create(const render_device& device, const sampler_description& description, std::string_view label);
		void destroy();

		[[nodiscard]] explicit operator bool() const;
		[[nodiscard]] sampler_handle native_handle() const;
		[[nodiscard]] const sampler_description& description() const;
		[[nodiscard]] sampler_shader_state shader_state() const;
		[[nodiscard]] u32 emulation_flags() const;
		[[nodiscard]] bool matches(const sampler_description& description) const;
		void touch(u64 frame_id);
		[[nodiscard]] u64 last_used_frame() const;
	};

	class sampler_pool
	{
		struct impl;
		std::unique_ptr<impl> m_impl;

	public:
		sampler_pool();
		~sampler_pool();
		sampler_pool(const sampler_pool&) = delete;
		sampler_pool& operator=(const sampler_pool&) = delete;
		sampler_pool(sampler_pool&&) = delete;
		sampler_pool& operator=(sampler_pool&&) = delete;

		void create(const render_device& device);
		void destroy();
		[[nodiscard]] std::shared_ptr<sampler> get(const sampler_description& description, std::string_view label);
		[[nodiscard]] std::shared_ptr<sampler> find(const sampler_description& description) const;
		[[nodiscard]] std::vector<std::shared_ptr<sampler>> collect(
			const std::function<bool(const sampler&)>& predicate);
		void clear();
		[[nodiscard]] usz size() const;
	};
}
