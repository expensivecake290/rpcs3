# Metal 4 renderer parity audit

This report records the final dependency, completion, file, and runtime audits required by the Metal 4 backend plan. It is documentation only; names of the reference API below are not renderer dependencies.

## Dependency audit

All 103 C++, Objective-C++, and header files under `MTL/` were searched case-insensitively for `VK`, `Vk`, `vulkan`, `MoltenVK`, and `SPIRV`. The code scan produced zero occurrences. The Metal target also configures and compiles with `USE_VULKAN=OFF`; it imports no reference-backend header and links no reference-backend runtime.

## Completion audit

All Metal code was searched for TODO/FIXME markers, “not implemented” or “unimplemented” messages, disabled `#if 0` sections, placeholder/stub/skeleton language, and placeholder shader text. The scan produced zero occurrences.

The 174 explicit `return {}`, `return nullptr`, `return true`, and `return false` sites were inspected. They are typed empty results for cache misses, invalid or absent optional resources, timeout/status predicates, capability probes, spatial-upscaler temporal-data queries, and generated shader comparison functions. None reports success without performing required work. Empty bodies are initializer-list-only constructors, move constructors, deliberate no-op cleanup catch blocks, or decompiler hooks whose Metal arguments are emitted by the function-signature builder. No operational backend method has an empty body.

## File parity audit

Every supplied reference-backend implementation/header is classified below. “Ported directly” means behavioral responsibility has a native Metal implementation, not a mechanical API translation.

| Reference file | Classification | Metal implementation or rationale |
|---|---|---|
| `VKAsyncScheduler.cpp` | PORTED DIRECTLY | `MTLAsyncScheduler.mm` |
| `VKAsyncScheduler.h` | PORTED DIRECTLY | `MTLAsyncScheduler.h` |
| `VKCommandStream.cpp` | PORTED DIRECTLY | `MTLCommandStream.mm` |
| `VKCommandStream.h` | PORTED DIRECTLY | `MTLCommandStream.h` |
| `VKCommonDecompiler.cpp` | PORTED DIRECTLY | `MTLCommonDecompiler.cpp` |
| `VKCommonDecompiler.h` | PORTED DIRECTLY | `MTLCommonDecompiler.h` |
| `VKCommonPipelineLayout.cpp` | PORTED DIRECTLY | `MTLCommonPipelineLayout.mm` |
| `VKCommonPipelineLayout.h` | PORTED DIRECTLY | `MTLCommonPipelineLayout.h` |
| `VKCompute.cpp` | PORTED DIRECTLY | `MTLCompute.mm` |
| `VKCompute.h` | PORTED DIRECTLY | `MTLCompute.h` |
| `VKDMA.cpp` | PORTED DIRECTLY | `MTLDMA.mm` |
| `VKDMA.h` | PORTED DIRECTLY | `MTLDMA.h` |
| `VKDataHeapManager.cpp` | PORTED DIRECTLY | `MTLDataHeapManager.cpp` |
| `VKDataHeapManager.h` | PORTED DIRECTLY | `MTLDataHeapManager.h` |
| `VKDraw.cpp` | PORTED DIRECTLY | `MTLDraw.mm` |
| `VKFormats.cpp` | PORTED DIRECTLY | `MTLFormats.mm` |
| `VKFormats.h` | PORTED DIRECTLY | `MTLFormats.h` |
| `VKFragmentProgram.cpp` | PORTED DIRECTLY | `MTLFragmentProgram.mm` |
| `VKFragmentProgram.h` | PORTED DIRECTLY | `MTLFragmentProgram.h` |
| `VKFramebuffer.cpp` | PORTED DIRECTLY | `MTLFramebuffer.mm` |
| `VKFramebuffer.h` | PORTED DIRECTLY | `MTLFramebuffer.h` |
| `VKGSRender.cpp` | PORTED DIRECTLY | `MTLGSRender.mm` |
| `VKGSRender.h` | PORTED DIRECTLY | `MTLGSRender.h` |
| `VKGSRenderTypes.hpp` | PORTED DIRECTLY | `MTLGSRenderTypes.hpp` |
| `VKHelpers.cpp` | PORTED DIRECTLY | `MTLHelpers.mm` |
| `VKHelpers.h` | PORTED DIRECTLY | `MTLHelpers.h` |
| `VKMemAlloc.cpp` | PORTED DIRECTLY | `MTLMemAlloc.mm` |
| `VKOverlays.cpp` | PORTED DIRECTLY | `MTLOverlays.mm` |
| `VKOverlays.h` | PORTED DIRECTLY | `MTLOverlays.h` |
| `VKPipelineCompiler.cpp` | PORTED DIRECTLY | `MTLPipelineCompiler.mm` |
| `VKPipelineCompiler.h` | PORTED DIRECTLY | `MTLPipelineCompiler.h` |
| `VKPresent.cpp` | PORTED DIRECTLY | `MTLPresent.mm`; presentation declarations live in `MTLGSRender.h` |
| `VKProcTable.h` | VULKAN-ONLY CONCEPT — NOT REQUIRED | Metal uses framework dispatch; initialization and native error translation live in `MetalAPI.*` |
| `VKProgramBuffer.h` | PORTED DIRECTLY | `MTLProgramBuffer.h` |
| `VKProgramPipeline.cpp` | PORTED DIRECTLY | `MTLProgramPipeline.mm` |
| `VKProgramPipeline.h` | PORTED DIRECTLY | `MTLProgramPipeline.h` |
| `VKQueryPool.cpp` | PORTED DIRECTLY | `MTLQueryPool.mm` |
| `VKQueryPool.h` | PORTED DIRECTLY | `MTLQueryPool.h` |
| `VKRenderPass.cpp` | PORTED DIRECTLY | `MTLRenderPass.mm` |
| `VKRenderPass.h` | PORTED DIRECTLY | `MTLRenderPass.h` |
| `VKRenderTargets.cpp` | PORTED DIRECTLY | `MTLRenderTargets.mm` |
| `VKRenderTargets.h` | PORTED DIRECTLY | `MTLRenderTargets.h` |
| `VKResolveHelper.cpp` | PORTED DIRECTLY | `MTLResolveHelper.mm` |
| `VKResolveHelper.h` | PORTED DIRECTLY | `MTLResolveHelper.h` |
| `VKResourceManager.cpp` | PORTED DIRECTLY | `MTLResourceManager.mm` |
| `VKResourceManager.h` | PORTED DIRECTLY | `MTLResourceManager.h` |
| `VKShaderInterpreter.cpp` | PORTED DIRECTLY | `MTLShaderInterpreter.mm` |
| `VKShaderInterpreter.h` | PORTED DIRECTLY | `MTLShaderInterpreter.h` |
| `VKTexture.cpp` | PORTED DIRECTLY | `MTLTexture.mm` |
| `VKTextureCache.cpp` | PORTED DIRECTLY | `MTLTextureCache.mm` |
| `VKTextureCache.h` | PORTED DIRECTLY | `MTLTextureCache.h` |
| `VKVertexBuffers.cpp` | PORTED DIRECTLY | `MTLVertexBuffers.mm` |
| `VKVertexProgram.cpp` | PORTED DIRECTLY | `MTLVertexProgram.mm` |
| `VKVertexProgram.h` | PORTED DIRECTLY | `MTLVertexProgram.h` |
| `VulkanAPI.cpp` | PORTED DIRECTLY | `MetalAPI.mm` |
| `VulkanAPI.h` | PORTED DIRECTLY | `MetalAPI.h` |
| `upscalers/bilinear_pass.hpp` | PORTED DIRECTLY | `upscalers/bilinear_pass.hpp` |
| `upscalers/fsr1/fsr_pass.cpp` | PORTED INTO DIFFERENT METAL FILE | Native enhanced scaling is implemented by `upscalers/metalfx_pass.mm` |
| `upscalers/fsr_pass.h` | PORTED INTO DIFFERENT METAL FILE | Spatial-scaler interface is implemented by `upscalers/metalfx_pass.h` |
| `upscalers/nearest_pass.hpp` | PORTED DIRECTLY | `upscalers/nearest_pass.hpp` |
| `upscalers/upscaling.h` | PORTED DIRECTLY | `upscalers/upscaling.h` |
| `vkutils/barriers.cpp` | PORTED DIRECTLY | `mtlutils/barriers.mm` |
| `vkutils/barriers.h` | PORTED DIRECTLY | `mtlutils/barriers.h` |
| `vkutils/buffer_object.cpp` | PORTED DIRECTLY | `mtlutils/buffer_object.mm` |
| `vkutils/buffer_object.h` | PORTED DIRECTLY | `mtlutils/buffer_object.h` |
| `vkutils/chip_class.cpp` | PORTED DIRECTLY | `mtlutils/chip_class.cpp` |
| `vkutils/chip_class.h` | PORTED DIRECTLY | `mtlutils/chip_class.h` |
| `vkutils/commands.cpp` | PORTED DIRECTLY | `mtlutils/commands.mm` |
| `vkutils/commands.h` | PORTED DIRECTLY | `mtlutils/commands.h` |
| `vkutils/data_heap.cpp` | PORTED DIRECTLY | `mtlutils/data_heap.mm` |
| `vkutils/data_heap.h` | PORTED DIRECTLY | `mtlutils/data_heap.h` |
| `vkutils/descriptors.cpp` | PORTED DIRECTLY | `mtlutils/descriptors.mm` |
| `vkutils/descriptors.h` | PORTED DIRECTLY | `mtlutils/descriptors.h` |
| `vkutils/device.cpp` | PORTED DIRECTLY | `mtlutils/device.mm` |
| `vkutils/device.h` | PORTED DIRECTLY | `mtlutils/device.h` |
| `vkutils/ex.cpp` | PORTED DIRECTLY | `mtlutils/ex.mm` |
| `vkutils/ex.h` | PORTED DIRECTLY | `mtlutils/ex.h` |
| `vkutils/framebuffer_object.hpp` | PORTED DIRECTLY | `mtlutils/framebuffer_object.hpp` |
| `vkutils/garbage_collector.h` | PORTED DIRECTLY | `mtlutils/garbage_collector.h` |
| `vkutils/graphics_pipeline_state.hpp` | PORTED DIRECTLY | `mtlutils/graphics_pipeline_state.hpp` |
| `vkutils/image.cpp` | PORTED DIRECTLY | `mtlutils/image.mm` |
| `vkutils/image.h` | PORTED DIRECTLY | `mtlutils/image.h` |
| `vkutils/image_helpers.cpp` | PORTED DIRECTLY | `mtlutils/image_helpers.mm` |
| `vkutils/image_helpers.h` | PORTED DIRECTLY | `mtlutils/image_helpers.h` |
| `vkutils/instance.cpp` | PORTED INTO DIFFERENT METAL FILE | Device discovery and capability validation live in `mtlutils/device.mm` and `MetalAPI.mm` |
| `vkutils/instance.h` | PORTED INTO DIFFERENT METAL FILE | Public discovery abstractions live in `mtlutils/device.h` and `MetalAPI.h` |
| `vkutils/memory.cpp` | PORTED DIRECTLY | `mtlutils/memory.mm` |
| `vkutils/memory.h` | PORTED DIRECTLY | `mtlutils/memory.h` |
| `vkutils/metal_layer.h` | PORTED DIRECTLY | `mtlutils/metal_layer.h` uses a native presentation layer directly |
| `vkutils/metal_layer.mm` | PORTED DIRECTLY | `mtlutils/metal_layer.mm` uses a native presentation layer directly |
| `vkutils/pipeline_binding_table.h` | PORTED DIRECTLY | `mtlutils/pipeline_binding_table.h` |
| `vkutils/query_pool.hpp` | PORTED DIRECTLY | `mtlutils/query_pool.hpp` |
| `vkutils/sampler.cpp` | PORTED DIRECTLY | `mtlutils/sampler.mm` |
| `vkutils/sampler.h` | PORTED DIRECTLY | `mtlutils/sampler.h` |
| `vkutils/scratch.cpp` | PORTED DIRECTLY | `mtlutils/scratch.mm` |
| `vkutils/scratch.h` | PORTED DIRECTLY | `mtlutils/scratch.h` |
| `vkutils/shared.cpp` | PORTED DIRECTLY | `mtlutils/shared.mm` |
| `vkutils/shared.h` | PORTED DIRECTLY | `mtlutils/shared.h` |
| `vkutils/swapchain.cpp` | PORTED DIRECTLY | `mtlutils/swapchain.mm` |
| `vkutils/swapchain.h` | PORTED DIRECTLY | `mtlutils/swapchain.h` |
| `vkutils/swapchain_android.hpp` | VULKAN-ONLY CONCEPT — NOT REQUIRED | Native Metal 4 backend targets macOS; Android window integration is out of scope |
| `vkutils/swapchain_core.h` | PORTED DIRECTLY | `mtlutils/swapchain_core.h` |
| `vkutils/swapchain_macos.hpp` | PORTED DIRECTLY | `mtlutils/swapchain_macos.hpp` |
| `vkutils/swapchain_unix.hpp` | VULKAN-ONLY CONCEPT — NOT REQUIRED | X11/Wayland presentation does not belong in the native macOS renderer |
| `vkutils/swapchain_win32.hpp` | VULKAN-ONLY CONCEPT — NOT REQUIRED | Win32 presentation does not belong in the native macOS renderer |
| `vkutils/sync.cpp` | PORTED DIRECTLY | `mtlutils/sync.mm` |
| `vkutils/sync.h` | PORTED DIRECTLY | `mtlutils/sync.h` |
| `vkutils/unique_resource.cpp` | PORTED DIRECTLY | `mtlutils/unique_resource.mm` |
| `vkutils/unique_resource.h` | PORTED DIRECTLY | `mtlutils/unique_resource.h` |

Inventory check: 109 reference files are listed above exactly once.

## Runtime parity audit

Validation was performed on Apple M1 / macOS 27 with the Metal HUD enabled, using a full
Debug application build configured with `USE_METAL=ON` and `USE_VULKAN=OFF`. The linked
application loads Metal and MetalFX and has no Vulkan or MoltenVK dynamic dependency or
entry point.

The following repository-provided RSX workloads were booted with `Renderer: Metal`, an
isolated portable configuration, interpreter CPU decoders, and a disabled on-disk shader
cache so that shader generation and pipeline creation occurred during the audit:

| Workload | Result |
| --- | --- |
| `bin/test/gs_gcm_basic_triangle.elf` | Completed without a Metal fatal |
| `bin/test/gs_gcm_cube.elf` | Completed without a Metal fatal |
| `bin/test/gs_gcm_hello_world.elf` | Sustained a 20-second validation run without a Metal fatal |
| `bin/test/gs_gcm_tetris.elf` | Sustained rendering at approximately 60 FPS; the Metal HUD reported active render and compute encoders, presentation, seven pipeline states, and seven cached shaders |
| `bin/test/gs_gcm_handle_system_cmd.elf` | Completed without a Metal fatal |

The runtime audit found and repaired concrete defects at the exercised boundaries:

- compute-to-render encoder transition at render-pass begin;
- constant-address-space preservation in generated fragment helpers;
- native interpreter framebuffer-fetch declarations for only active color attachments;
- point-size output compatibility across reusable topology variants;
- stale render-pass ownership after utility encoders transition command state;
- nullable render-target classification in the texture cache;
- deferred-resource event rollover and completion at command submission;
- mixed half/full RSX fragment-register emission through the lossless full-precision path;
- native interpreter format-string expansion for programmable blending.

No workload reported a shader compilation failure, pipeline compilation failure, command
buffer failure, device loss, or renderer fatal after those repairs. This runtime evidence
supplements the per-file dependency inventory above; it does not claim compatibility with
commercial titles that were not available in the repository test corpus.
