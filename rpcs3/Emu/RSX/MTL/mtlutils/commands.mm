#include "stdafx.h"
#include "commands.h"

#include "../MTLResourceManager.h"
#include "shared.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <condition_variable>
#include <mutex>

namespace mtl
{
	struct submission::impl
	{
		mutable std::mutex mutex;
		mutable std::condition_variable condition;
		u64 submission_value = 0;
		bool is_completed = false;
		bool is_successful = false;
		error failure_info;
		f64 start_time = 0.0;
		f64 end_time = 0.0;
	};

	struct command_allocator::impl
	{
		render_device* render = nullptr;
		id<MTL4CommandAllocator> allocator;
		std::mutex mutex;
		std::condition_variable condition;
		u32 encoding_buffers = 0;
		u32 pending_submissions = 0;
	};

	struct command_buffer::impl
	{
		command_allocator* command_allocator_owner = nullptr;
		id<MTL4CommandBuffer> buffer;
		id<MTL4CommandEncoder> encoder;
		NSMutableArray* retained_objects;
		std::vector<std::function<void(bool)>> completion_callbacks;
		std::vector<void*> resident_allocations;
		encoder_kind encoder_type = encoder_kind::none;
		submission most_recent_submission;
		u32 state_flags = 0;
		u32 debug_group_depth = 0;
		bool recording = false;
		bool ended = false;
	};

	namespace
	{
		error make_submission_error(NSError* native_error)
		{
			error result;
			result.native_code = native_error.code;
			result.domain = native_error.domain.UTF8String ?: "Metal";
			result.description = native_error.localizedDescription.UTF8String ?: "Metal command submission failed";

			if ([native_error.domain isEqualToString:MTL4CommandQueueErrorDomain])
			{
				switch (static_cast<MTL4CommandQueueError>(native_error.code))
				{
				case MTL4CommandQueueErrorOutOfMemory:
					result.code = error_code::out_of_memory;
					break;
				case MTL4CommandQueueErrorAccessRevoked:
				case MTL4CommandQueueErrorDeviceRemoved:
					result.code = error_code::device_lost;
					break;
				case MTL4CommandQueueErrorTimeout:
				case MTL4CommandQueueErrorNotPermitted:
				case MTL4CommandQueueErrorInternal:
					result.code = error_code::command_submission_failed;
					break;
				case MTL4CommandQueueErrorNone:
					result.code = error_code::none;
					break;
				}
			}
			else
			{
				result.code = error_code::command_submission_failed;
			}
			return result;
		}

		void validate_event_operation(const event_operation& operation, const char* operation_name)
		{
			if (!operation.event)
			{
				fmt::throw_exception("Metal queue %s contains a null event", operation_name);
			}
		}

		void release_retained_objects(auto& state)
		{
			if (get_shared_state())
			{
				for (void* allocation : state.resident_allocations)
				{
					get_shared_state().residency().remove(allocation);
				}
			}
			state.resident_allocations.clear();
			[state.retained_objects removeAllObjects];
		}
	}

	submission::submission(std::shared_ptr<impl> implementation)
		: m_impl(std::move(implementation))
	{
	}

	submission::operator bool() const
	{
		return m_impl && m_impl->submission_value != 0;
	}

	u64 submission::value() const
	{
		return m_impl ? m_impl->submission_value : 0;
	}

	bool submission::completed() const
	{
		if (!m_impl)
		{
			return false;
		}
		std::lock_guard lock(m_impl->mutex);
		return m_impl->is_completed;
	}

	bool submission::succeeded() const
	{
		if (!m_impl)
		{
			return false;
		}
		std::lock_guard lock(m_impl->mutex);
		return m_impl->is_completed && m_impl->is_successful;
	}

	error submission::failure() const
	{
		if (!m_impl)
		{
			return {error_code::invalid_operation, 0, "Metal", "Failure requested from an empty submission"};
		}
		std::lock_guard lock(m_impl->mutex);
		return m_impl->failure_info;
	}

	f64 submission::gpu_start_time() const
	{
		if (!m_impl)
		{
			return 0.0;
		}
		std::lock_guard lock(m_impl->mutex);
		return m_impl->start_time;
	}

	f64 submission::gpu_end_time() const
	{
		if (!m_impl)
		{
			return 0.0;
		}
		std::lock_guard lock(m_impl->mutex);
		return m_impl->end_time;
	}

	bool submission::wait(std::chrono::nanoseconds timeout) const
	{
		if (!m_impl)
		{
			return false;
		}

		std::unique_lock lock(m_impl->mutex);
		return m_impl->condition.wait_for(lock, timeout, [&]
		{
			return m_impl->is_completed;
		});
	}

	void submission::wait() const
	{
		if (!m_impl)
		{
			fmt::throw_exception("Cannot wait for an empty Metal submission");
		}

		std::unique_lock lock(m_impl->mutex);
		m_impl->condition.wait(lock, [&]
		{
			return m_impl->is_completed;
		});
	}

	command_allocator::command_allocator()
		: m_impl(std::make_unique<impl>())
	{
	}

	command_allocator::~command_allocator()
	{
		destroy();
	}

	void command_allocator::create(render_device& device, std::string_view label)
	{
		destroy();
		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}

		MTL4CommandAllocatorDescriptor* descriptor = [MTL4CommandAllocatorDescriptor new];
		descriptor.label = [NSString stringWithUTF8String:std::string(label).c_str()];
		NSError* native_error = nil;
		m_impl->allocator = [device.native_handle() newCommandAllocatorWithDescriptor:descriptor error:&native_error];
		if (!m_impl->allocator)
		{
			error value;
			value.code = error_code::resource_creation_failed;
			value.native_code = native_error.code;
			value.domain = native_error.domain.UTF8String ?: "Metal";
			value.description = native_error.localizedDescription.UTF8String ?: "Metal returned no command allocator";
			throw_error(value, "Metal command-allocator creation");
		}
		m_impl->render = &device;
	}

	void command_allocator::destroy()
	{
		if (!m_impl || !m_impl->allocator)
		{
			return;
		}

		std::unique_lock lock(m_impl->mutex);
		ensure(m_impl->encoding_buffers == 0);
		m_impl->condition.wait(lock, [&]
		{
			return m_impl->pending_submissions == 0;
		});
		m_impl->allocator = nil;
		m_impl->render = nullptr;
	}

	void command_allocator::reset()
	{
		if (!m_impl || !m_impl->allocator)
		{
			fmt::throw_exception("Cannot reset an empty Metal command allocator");
		}

		std::lock_guard lock(m_impl->mutex);
		ensure(m_impl->encoding_buffers == 0 && m_impl->pending_submissions == 0);
		[m_impl->allocator reset];
	}

	command_allocator::operator bool() const
	{
		return m_impl && m_impl->allocator;
	}

	render_device& command_allocator::owner() const
	{
		if (!m_impl || !m_impl->render)
		{
			fmt::throw_exception("Render device requested from an empty Metal command allocator");
		}
		return *m_impl->render;
	}

	command_allocator_handle command_allocator::native_handle() const
	{
		return m_impl ? m_impl->allocator : nil;
	}

	u64 command_allocator::allocated_size() const
	{
		return m_impl && m_impl->allocator ? m_impl->allocator.allocatedSize : 0;
	}

	command_buffer::command_buffer()
		: m_impl(std::make_unique<impl>())
	{
	}

	command_buffer::~command_buffer()
	{
		destroy();
	}

	command_buffer::command_buffer(command_buffer&& other) noexcept
		: m_impl(std::move(other.m_impl))
	{
	}

	command_buffer& command_buffer::operator=(command_buffer&& other) noexcept
	{
		if (this != &other)
		{
			destroy();
			m_impl = std::move(other.m_impl);
		}
		return *this;
	}

	void command_buffer::create(command_allocator& allocator, std::string_view label)
	{
		destroy();
		if (!allocator)
		{
			fmt::throw_exception("Cannot create a Metal command buffer without an allocator");
		}

		if (!m_impl)
		{
			m_impl = std::make_unique<impl>();
		}
		m_impl->buffer = [allocator.owner().native_handle() newCommandBuffer];
		if (!m_impl->buffer)
		{
			fmt::throw_exception("Metal returned no command buffer");
		}
		m_impl->buffer.label = [NSString stringWithUTF8String:std::string(label).c_str()];
		m_impl->command_allocator_owner = &allocator;
		m_impl->retained_objects = [NSMutableArray array];
	}

	void command_buffer::destroy()
	{
		if (!m_impl || !m_impl->buffer)
		{
			return;
		}

		ensure(!m_impl->recording && m_impl->encoder_type == encoder_kind::none);
		if (m_impl->most_recent_submission && !m_impl->most_recent_submission.completed())
		{
			m_impl->most_recent_submission.wait();
		}
		for (auto& callback : m_impl->completion_callbacks)
		{
			callback(false);
		}
		m_impl->completion_callbacks.clear();
		release_retained_objects(*m_impl);
		m_impl->most_recent_submission = {};
		m_impl->buffer = nil;
		m_impl->command_allocator_owner = nullptr;
		m_impl->ended = false;
	}

	void command_buffer::begin()
	{
		if (!m_impl || !m_impl->buffer || !m_impl->command_allocator_owner)
		{
			fmt::throw_exception("Cannot begin an empty Metal command buffer");
		}
		if (m_impl->recording || m_impl->encoder_type != encoder_kind::none)
		{
			fmt::throw_exception("Metal command buffer is already recording");
		}
		if (m_impl->most_recent_submission && !m_impl->most_recent_submission.completed())
		{
			m_impl->most_recent_submission.wait();
		}
		if (m_impl->most_recent_submission)
		{
			reset_after_completion();
		}

		auto& allocator_impl = *m_impl->command_allocator_owner->m_impl;
		{
			std::lock_guard lock(allocator_impl.mutex);
			ensure(allocator_impl.encoding_buffers == 0);
			allocator_impl.encoding_buffers++;
		}
		[m_impl->buffer beginCommandBufferWithAllocator:allocator_impl.allocator];
		m_impl->recording = true;
		m_impl->ended = false;
		m_impl->state_flags = 0;
		m_impl->debug_group_depth = 0;
	}

	void command_buffer::end()
	{
		if (!m_impl || !m_impl->recording)
		{
			fmt::throw_exception("Cannot end a Metal command buffer that is not recording");
		}
		if (m_impl->encoder_type != encoder_kind::none || m_impl->debug_group_depth != 0)
		{
			fmt::throw_exception("Cannot end a Metal command buffer with an active encoder or debug group");
		}

		[m_impl->buffer endCommandBuffer];
		m_impl->recording = false;
		m_impl->ended = true;
		auto& allocator_impl = *m_impl->command_allocator_owner->m_impl;
		{
			std::lock_guard lock(allocator_impl.mutex);
			ensure(allocator_impl.encoding_buffers == 1);
			allocator_impl.encoding_buffers--;
		}
	}

	void command_buffer::discard()
	{
		if (!m_impl || !m_impl->buffer || !m_impl->command_allocator_owner)
		{
			return;
		}
		if (!m_impl->recording)
		{
			for (auto& callback : m_impl->completion_callbacks)
			{
				callback(false);
			}
			m_impl->completion_callbacks.clear();
			m_impl->ended = false;
			return;
		}
		if (m_impl->encoder_type != encoder_kind::none)
		{
			[m_impl->encoder endEncoding];
			m_impl->encoder = nil;
			m_impl->encoder_type = encoder_kind::none;
		}
		while (m_impl->debug_group_depth)
		{
			[m_impl->buffer popDebugGroup];
			--m_impl->debug_group_depth;
		}
		[m_impl->buffer endCommandBuffer];
		m_impl->recording = false;
		m_impl->ended = false;
		for (auto& callback : m_impl->completion_callbacks)
		{
			callback(false);
		}
		m_impl->completion_callbacks.clear();
		auto& allocator_impl = *m_impl->command_allocator_owner->m_impl;
		{
			std::lock_guard lock(allocator_impl.mutex);
			ensure(allocator_impl.encoding_buffers == 1);
			allocator_impl.encoding_buffers--;
		}
	}

	native_encoder_handle command_buffer::begin_render_encoding(native_render_pass_descriptor descriptor)
	{
		if (!m_impl || !m_impl->recording || m_impl->encoder_type != encoder_kind::none || !descriptor)
		{
			fmt::throw_exception("Invalid Metal render encoder begin");
		}
		MTL4RenderPassDescriptor* native_descriptor = (__bridge MTL4RenderPassDescriptor*)descriptor;
		id<MTL4RenderCommandEncoder> encoder = [m_impl->buffer renderCommandEncoderWithDescriptor:native_descriptor];
		if (!encoder)
		{
			fmt::throw_exception("Metal returned no render command encoder");
		}
		m_impl->encoder = encoder;
		m_impl->encoder_type = encoder_kind::render;
		return (__bridge void*)encoder;
	}

	native_encoder_handle command_buffer::begin_compute_encoding()
	{
		if (!m_impl || !m_impl->recording || m_impl->encoder_type != encoder_kind::none)
		{
			fmt::throw_exception("Invalid Metal compute encoder begin");
		}
		id<MTL4ComputeCommandEncoder> encoder = [m_impl->buffer computeCommandEncoder];
		if (!encoder)
		{
			fmt::throw_exception("Metal returned no compute command encoder");
		}
		m_impl->encoder = encoder;
		m_impl->encoder_type = encoder_kind::compute;
		return (__bridge void*)encoder;
	}

	void command_buffer::end_encoding()
	{
		if (!m_impl || !m_impl->recording || !m_impl->encoder || m_impl->encoder_type == encoder_kind::none)
		{
			fmt::throw_exception("Cannot end an inactive Metal command encoder");
		}
		[m_impl->encoder endEncoding];
		m_impl->encoder = nil;
		m_impl->encoder_type = encoder_kind::none;
	}

	void command_buffer::push_debug_group(std::string_view label)
	{
		if (!m_impl || !m_impl->recording || label.empty())
		{
			fmt::throw_exception("Invalid Metal command-buffer debug group");
		}
		[m_impl->buffer pushDebugGroup:[NSString stringWithUTF8String:std::string(label).c_str()]];
		m_impl->debug_group_depth++;
	}

	void command_buffer::pop_debug_group()
	{
		if (!m_impl || !m_impl->recording || m_impl->debug_group_depth == 0)
		{
			fmt::throw_exception("Unbalanced Metal command-buffer debug group");
		}
		[m_impl->buffer popDebugGroup];
		m_impl->debug_group_depth--;
	}

	void command_buffer::retain_native_object(void* object, bool make_resident)
	{
		if (!m_impl || !m_impl->recording || !object)
		{
			fmt::throw_exception("Cannot retain an invalid native object outside Metal command recording");
		}

		id native_object = (__bridge id)object;
		if (make_resident)
		{
			if (![native_object conformsToProtocol:@protocol(MTLAllocation)])
			{
				fmt::throw_exception("Metal object requested for residency does not implement MTLAllocation");
			}
			if (!get_shared_state())
			{
				fmt::throw_exception("Metal residency is unavailable while retaining a GPU allocation");
			}
			get_shared_state().residency().add(object);
			m_impl->resident_allocations.push_back(object);
		}
		[m_impl->retained_objects addObject:native_object];
	}

	void command_buffer::notify_on_completion(std::function<void(bool)> callback)
	{
		if (!m_impl || !m_impl->buffer || (!m_impl->recording && !m_impl->ended) || !callback)
		{
			fmt::throw_exception("Invalid Metal command completion callback");
		}
		m_impl->completion_callbacks.emplace_back(std::move(callback));
	}

	submission command_buffer::submit(const submit_info& info)
	{
		if (!m_impl || !m_impl->buffer || m_impl->recording || !m_impl->ended)
		{
			fmt::throw_exception("Metal command buffer must be ended before submission");
		}
		if (m_impl->most_recent_submission && !m_impl->most_recent_submission.completed())
		{
			fmt::throw_exception("Metal command buffer is already pending");
		}
		if (has_flag(command_has_open_query))
		{
			fmt::throw_exception("Cannot submit a Metal command buffer with an open query");
		}

		for (const event_operation& operation : info.waits)
		{
			validate_event_operation(operation, "wait");
		}
		for (const event_operation& operation : info.signal_operations)
		{
			validate_event_operation(operation, "signal");
		}

		id<MTL4CommandQueue> queue = info.queue == queue_kind::graphics
			? m_impl->command_allocator_owner->owner().graphics_queue()
			: m_impl->command_allocator_owner->owner().transfer_queue();
		for (const event_operation& operation : info.waits)
		{
			[queue waitForEvent:operation.event value:operation.value];
		}

		auto state = std::make_shared<submission::impl>();
		auto completion_callbacks = std::make_shared<std::vector<std::function<void(bool)>>>(
			std::move(m_impl->completion_callbacks));
		state->submission_value = get_shared_state().next_submission();
		const u64 resource_event = current_resource_event();
		if (!resource_event)
		{
			fmt::throw_exception("Metal command submission has no resource-retirement event");
		}
		static_cast<void>(allocate_resource_event());
		auto& allocator_impl = *m_impl->command_allocator_owner->m_impl;
		{
			std::lock_guard lock(allocator_impl.mutex);
			allocator_impl.pending_submissions++;
		}

		MTL4CommitOptions* options = [MTL4CommitOptions new];
		[options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback)
		{
			const bool succeeded = feedback.error == nil;
			{
				std::lock_guard lock(state->mutex);
				state->start_time = feedback.GPUStartTime;
				state->end_time = feedback.GPUEndTime;
				state->is_completed = true;
				state->is_successful = succeeded;
				if (feedback.error)
				{
					state->failure_info = make_submission_error(feedback.error);
				}
			}
			state->condition.notify_all();
			for (auto& callback : *completion_callbacks)
			{
				callback(succeeded);
			}
			{
				std::lock_guard lock(allocator_impl.mutex);
				ensure(allocator_impl.pending_submissions != 0);
				allocator_impl.pending_submissions--;
			}
			allocator_impl.condition.notify_all();
			get_shared_state().notify_submission_completed(state->submission_value);
			notify_resource_event_completed(resource_event);
		}];

		id<MTL4CommandBuffer> native_buffer = m_impl->buffer;
		[queue commit:&native_buffer count:1 options:options];
		for (const event_operation& operation : info.signal_operations)
		{
			[queue signalEvent:operation.event value:operation.value];
		}

		m_impl->most_recent_submission = submission(state);
		m_impl->ended = false;
		m_impl->state_flags = 0;
		if (info.wait_for_completion)
		{
			m_impl->most_recent_submission.wait();
			if (!m_impl->most_recent_submission.succeeded())
			{
				throw_error(m_impl->most_recent_submission.failure(), "Metal command submission");
			}
		}
		return m_impl->most_recent_submission;
	}

	void command_buffer::reset_after_completion()
	{
		if (!m_impl || !m_impl->most_recent_submission || !m_impl->most_recent_submission.completed())
		{
			fmt::throw_exception("Cannot reset a Metal command buffer before completion");
		}
		if (!m_impl->most_recent_submission.succeeded())
		{
			throw_error(m_impl->most_recent_submission.failure(), "Metal command-buffer reuse");
		}
		release_retained_objects(*m_impl);
		m_impl->command_allocator_owner->reset();
		m_impl->most_recent_submission = {};
	}

	void command_buffer::clear_flags()
	{
		if (m_impl)
		{
			m_impl->state_flags = 0;
		}
	}

	void command_buffer::set_flag(command_buffer_flag flag)
	{
		if (!m_impl)
		{
			fmt::throw_exception("Cannot flag an empty Metal command buffer");
		}
		m_impl->state_flags |= flag;
	}

	void command_buffer::clear_flag(command_buffer_flag flag)
	{
		if (!m_impl)
		{
			fmt::throw_exception("Cannot clear a flag on an empty Metal command buffer");
		}
		m_impl->state_flags &= ~static_cast<u32>(flag);
	}

	bool command_buffer::has_flag(command_buffer_flag flag) const
	{
		return m_impl && (m_impl->state_flags & flag) != 0;
	}

	command_buffer::operator bool() const
	{
		return m_impl && m_impl->buffer;
	}

	bool command_buffer::is_recording() const
	{
		return m_impl && m_impl->recording;
	}

	bool command_buffer::is_pending() const
	{
		return m_impl && m_impl->most_recent_submission && !m_impl->most_recent_submission.completed();
	}

	encoder_kind command_buffer::active_encoder() const
	{
		return m_impl ? m_impl->encoder_type : encoder_kind::none;
	}

	native_encoder_handle command_buffer::active_native_encoder() const
	{
		return m_impl && m_impl->encoder ? (__bridge void*)m_impl->encoder : nullptr;
	}

	command_buffer_handle command_buffer::native_handle() const
	{
		return m_impl ? m_impl->buffer : nil;
	}

	command_allocator& command_buffer::allocator() const
	{
		if (!m_impl || !m_impl->command_allocator_owner)
		{
			fmt::throw_exception("Allocator requested from an empty Metal command buffer");
		}
		return *m_impl->command_allocator_owner;
	}

	submission command_buffer::last_submission() const
	{
		return m_impl ? m_impl->most_recent_submission : submission();
	}
}
