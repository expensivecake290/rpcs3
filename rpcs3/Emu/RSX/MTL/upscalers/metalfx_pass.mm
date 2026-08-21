#include "stdafx.h"
#include "metalfx_pass.h"

#include "../MTLFormats.h"

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include <array>

namespace mtl
{
	namespace
	{
		struct scaler_configuration
		{
			u64 input_format = 0;
			u64 output_format = 0;
			u32 input_width = 0;
			u32 input_height = 0;
			u32 output_width = 0;
			u32 output_height = 0;
			metalfx_color_processing color_processing = metalfx_color_processing::perceptual;

			[[nodiscard]] bool operator==(const scaler_configuration&) const = default;
			[[nodiscard]] explicit operator bool() const
			{
				return input_format && output_format && input_width && input_height &&
					output_width && output_height;
			}
		};

		[[nodiscard]] MTLFXSpatialScalerColorProcessingMode native_color_processing(
			metalfx_color_processing mode)
		{
			switch (mode)
			{
			case metalfx_color_processing::perceptual:
				return MTLFXSpatialScalerColorProcessingModePerceptual;
			case metalfx_color_processing::linear:
				return MTLFXSpatialScalerColorProcessingModeLinear;
			case metalfx_color_processing::high_dynamic_range:
				return MTLFXSpatialScalerColorProcessingModeHDR;
			}
			fmt::throw_exception("Invalid MetalFX color-processing mode");
		}

		[[nodiscard]] image_create_info working_image_information(u64 format,
			u32 width, u32 height, rsx::format_class format_class, std::string label)
		{
			return {
				.type = texture_type::texture_2d,
				.formats = get_view_compatibility(format),
				.width = width,
				.height = height,
				.depth = 1,
				.mip_levels = 1,
				.array_layers = 1,
				.sample_count = 1,
				.usage = texture_usage_shader_read | texture_usage_shader_write |
					texture_usage_render_target | texture_usage_copy_source |
					texture_usage_copy_destination,
				.aspects = texture_aspect_color,
				.format_class = format_class,
				.storage = storage_mode::private_,
				.hazards = hazard_tracking::tracked,
				.pool = allocation_pool::swapchain,
				.label = std::move(label),
				.use_placement_heap = false,
			};
		}

		[[nodiscard]] bool matches(const std::unique_ptr<viewable_image>& image,
			u64 format, u32 width, u32 height)
		{
			return image && *image && image->format() == format && image->width() == width &&
				image->height() == height && image->samples() == 1;
		}
	}

	struct metalfx_upscale_pass::impl
	{
		render_device* device = nullptr;
		memory_allocator* allocator = nullptr;
		metalfx_color_processing processing = metalfx_color_processing::perceptual;
		scaler_configuration configuration;
		std::unique_ptr<viewable_image> input;
		std::array<std::unique_ptr<viewable_image>, 2> outputs;
		id<MTL4FXSpatialScaler> scaler = nil;

		void configure(const scaler_configuration& desired)
		{
			if (!desired) fmt::throw_exception("Invalid MetalFX spatial-scaler configuration");
			if (configuration == desired && scaler) return;

			MTLFXSpatialScalerDescriptor* descriptor = [MTLFXSpatialScalerDescriptor new];
			descriptor.colorTextureFormat = static_cast<MTLPixelFormat>(desired.input_format);
			descriptor.outputTextureFormat = static_cast<MTLPixelFormat>(desired.output_format);
			descriptor.inputWidth = desired.input_width;
			descriptor.inputHeight = desired.input_height;
			descriptor.outputWidth = desired.output_width;
			descriptor.outputHeight = desired.output_height;
			descriptor.colorProcessingMode = native_color_processing(desired.color_processing);
			id<MTL4FXSpatialScaler> created = [descriptor
				newSpatialScalerWithDevice:device->native_handle() compiler:device->compiler()];
			if (!created)
			{
				fmt::throw_exception(
					"MetalFX could not create a spatial scaler (input=%ux%u format=%llu, "
					"output=%ux%u format=%llu, color-mode=%u)",
					desired.input_width, desired.input_height, desired.input_format,
					desired.output_width, desired.output_height, desired.output_format,
					static_cast<u32>(desired.color_processing));
			}
			scaler = created;
			configuration = desired;
		}

		viewable_image& prepare_input(command_buffer& command, viewable_image& source,
			const upscale_request& request)
		{
			if (!matches(input, source.format(), request.source_width(), request.source_height()))
			{
				input = std::make_unique<viewable_image>(*allocator,
					working_image_information(source.format(), request.source_width(),
						request.source_height(), source.format_class(), "RPCS3 MetalFX input"));
			}
			const image_rectangle target_area{0, 0, static_cast<s32>(request.source_width()),
				static_cast<s32>(request.source_height())};
			copy_scaled_image(command, source, *input, request.source_area, target_area,
				1, true, image_filter::nearest);
			return *input;
		}

		viewable_image& prepare_output(u32 view, u64 format, u32 width, u32 height,
			rsx::format_class format_class)
		{
			if (view >= outputs.size())
				fmt::throw_exception("MetalFX output view %u is out of range", view);
			if (!matches(outputs[view], format, width, height))
			{
				outputs[view] = std::make_unique<viewable_image>(*allocator,
					working_image_information(format, width, height, format_class,
						view ? "RPCS3 MetalFX right output" : "RPCS3 MetalFX left output"));
			}
			return *outputs[view];
		}
	};

	metalfx_upscale_pass::metalfx_upscale_pass(render_device& device,
		memory_allocator& allocator, metalfx_color_processing color_processing)
		: m_impl(std::make_unique<impl>())
	{
		if (&allocator.device() != &device || !supported(device))
			fmt::throw_exception("MetalFX spatial scaling is unavailable on the active device");
		static_cast<void>(native_color_processing(color_processing));
		m_impl->device = &device;
		m_impl->allocator = &allocator;
		m_impl->processing = color_processing;
	}

	metalfx_upscale_pass::~metalfx_upscale_pass() = default;

	bool metalfx_upscale_pass::supported(const render_device& device)
	{
		return device && device.info().features.metal4 && device.compiler() &&
			[MTLFXSpatialScalerDescriptor supportsMetal4FX:device.native_handle()];
	}

	metalfx_color_processing metalfx_upscale_pass::color_processing() const
	{
		if (!m_impl) fmt::throw_exception("MetalFX spatial scaler has no implementation state");
		return m_impl->processing;
	}

	viewable_image* metalfx_upscale_pass::scale_output(command_buffer& command,
		viewable_image& source, viewable_image* destination,
		const upscale_request& request, rsx::flags32_t mode)
	{
		if (!m_impl || !m_impl->device || !m_impl->allocator)
			fmt::throw_exception("MetalFX spatial scaler is not initialized");
		validate_upscale_request(command, source, destination, request, mode);
		if (&command.allocator().owner() != m_impl->device)
			fmt::throw_exception("MetalFX command buffer belongs to a different device");

		const u32 view = (mode & upscale_right_view) ? 1u : 0u;
		const u64 output_format = destination ? destination->format() : source.format();
		const rsx::format_class output_class = destination
			? destination->format_class() : source.format_class();
		const scaler_configuration desired{
			.input_format = source.format(),
			.output_format = output_format,
			.input_width = request.source_width(),
			.input_height = request.source_height(),
			.output_width = request.destination_width(),
			.output_height = request.destination_height(),
			.color_processing = m_impl->processing,
		};
		m_impl->configure(desired);
		viewable_image& input = m_impl->prepare_input(command, source, request);
		viewable_image& output = m_impl->prepare_output(view, output_format,
			request.destination_width(), request.destination_height(), output_class);

		id<MTLTexture> input_texture = input.native_handle();
		id<MTLTexture> output_texture = output.native_handle();
		const MTLTextureUsage required_input_usage = m_impl->scaler.colorTextureUsage;
		const MTLTextureUsage required_output_usage = m_impl->scaler.outputTextureUsage;
		if ((input_texture.usage & required_input_usage) != required_input_usage ||
			(output_texture.usage & required_output_usage) != required_output_usage)
		{
			fmt::throw_exception("MetalFX working textures do not satisfy scaler usage requirements");
		}

		transition_image(command, input,
			{queue_kind::graphics, stage_machine_learning, access_shader_read,
				get_submission_id(), true});
		transition_image(command, output,
			{queue_kind::graphics, stage_machine_learning, access_shader_write,
				get_submission_id(), true});
		if (command.active_encoder() != encoder_kind::none) command.end_encoding();

		m_impl->scaler.inputContentWidth = desired.input_width;
		m_impl->scaler.inputContentHeight = desired.input_height;
		m_impl->scaler.colorTexture = input_texture;
		m_impl->scaler.outputTexture = output_texture;
		command.retain_native_object((__bridge void*)m_impl->scaler, false);
		command.retain_native_object((__bridge void*)input_texture, true);
		command.retain_native_object((__bridge void*)output_texture, true);
		[m_impl->scaler encodeToCommandBuffer:command.native_handle()];

		if (!(mode & upscale_and_commit)) return &output;
		const image_rectangle output_area{0, 0, static_cast<s32>(desired.output_width),
			static_cast<s32>(desired.output_height)};
		image_conversion conversion;
		const bool compatible = formats_are_bitcast_compatible(output, *destination);
		if (!compatible) conversion.kind = image_conversion_kind::color_to_color;
		copy_scaled_image(command, output, *destination, output_area,
			request.destination_area, 1, compatible, image_filter::nearest, conversion);
		return nullptr;
	}
}
