# Metal renderer integration audit

This ledger records the separate, current-checkout integration phases required
after the native backend was completed. Each row was reviewed against the
renderer-specific block in the target file and the corresponding native
backend entry point. Existing Vulkan behavior is left unchanged.

| Phase | Target file | Vulkan reference in that file | Metal result |
|---:|---|---|---|
| I-01 | `CMakeLists.txt` | `USE_VULKAN`, Vulkan package discovery, and the Vulkan language/build guards | Added the independent `USE_METAL` option and enables Objective-C++ only for an Apple Metal build. A Metal-only configuration does not require the Vulkan package. |
| I-02 | `3rdparty/CMakeLists.txt` | VulkanMemoryAllocator and shader-tool dependency targets | Inspected; no Metal dependency was added because Metal is supplied by the OS frameworks and the native backend does not consume VulkanMemoryAllocator. |
| I-03 | `3rdparty/protobuf/CMakeLists.txt` | Shared build dependency reached by the renderer-enabled application build | Made bundled protobuf headers take precedence over broad package-manager include roots, preventing unrelated host protobuf versions from corrupting any bundled build. |
| I-04 | `rpcs3/CMakeLists.txt` | Apple Vulkan packaging/runtime conditions | Inspected; the existing Vulkan conditions remain scoped to Vulkan. Metal uses OS frameworks and needs no packaged loader. |
| I-05 | `rpcs3/Emu/CMakeLists.txt` | `HAVE_VULKAN`, the full `VK/` source list, Vulkan target link, and renderer compile flags | Added `HAVE_METAL`, all native Metal implementation units, ARC/blocks/exception options at the Objective-C++ boundary, and Metal, MetalFX, QuartzCore, AppKit, and Foundation linkage under `APPLE AND USE_METAL`. |
| I-06 | `rpcs3/Emu/system_config_types.h` | `video_renderer::vulkan` enum value | Added the distinct `video_renderer::metal` value without reordering or changing existing values. |
| I-07 | `rpcs3/Emu/system_config_types.cpp` | Vulkan enum formatter/parser text | Added the canonical `Metal` renderer name used by YAML parsing and formatting. |
| I-08 | `rpcs3/Emu/system_config.h` | `node_vk`, Vulkan adapter configuration, and renderer default | Added `node_mtl` with its own adapter field and selects Metal by default only in a build that provides it; Vulkan and non-Metal defaults retain their prior conditions. |
| I-09 | `rpcs3/Emu/system_config.cpp` | Generic configuration load/save and renderer enum serialization | Inspected; the enum formatter and `node_mtl` registration make the existing generic serialization path complete, so no backend-specific branch is necessary. |
| I-10 | `rpcs3/Emu/System.cpp` | Vulkan SDK logging, default-adapter validation, and renderer-specific config transfer | Added Metal adapter validation and transfer through the same startup lifecycle, with no Vulkan SDK or loader dependency. |
| I-11 | `rpcs3/Emu/title.h` | Vulkan-specific adapter title field | Generalized the field to `graphics_adapter`, so diagnostics describe the active backend rather than hard-coding Vulkan. |
| I-12 | `rpcs3/Emu/title.cpp` | `video_renderer::vulkan` title switch | Added Metal to the adapter-bearing renderers while retaining the existing Vulkan output. |
| I-13 | `rpcs3/Emu/RSX/RSXThread.cpp` | Vulkan permission for the native user interface path | Permits the same native interface path for Metal. |
| I-14 | `rpcs3/Emu/RSX/rsx_cache.h` | Vulkan shader-cache worker selection | Enables the existing multithreaded shader cache for Metal as well. |
| I-15 | `rpcs3/Emu/RSX/Common/TextureUtils.cpp` | Vulkan wording around shared texture conversion semantics | Inspected; conversion behavior is already renderer-neutral and needs no Metal branch. |
| I-16 | `rpcs3/Emu/RSX/Common/BufferUtils.cpp` | Shared RSX buffer conversion used by the Vulkan renderer | Corrected the shared conversion path used by the native Metal upload implementation; behavior remains common to all renderers rather than being duplicated in Metal. |
| I-17 | `rpcs3/Emu/RSX/Core/RSXDrawCommands.cpp` | Vulkan half-pixel coordinate rule | Documents the identical Metal coordinate requirement while preserving the existing rule. |
| I-18 | `rpcs3/Emu/RSX/Core/RSXDriverState.h` | Vulkan-only driver-state dirty bit | Inspected; Metal invalidates its native pipeline state directly and does not claim the Vulkan-only bit. |
| I-19 | `rpcs3/Emu/RSX/GL/GLTextureCache.h` | A Vulkan mention inside OpenGL fallback policy | Inspected; this is an OpenGL/Vulkan interoperability decision and is not part of Metal selection. |
| I-20 | `rpcs3/Emu/RSX/Program/SPIRVCommon.cpp` | Vulkan shader-module infrastructure | Inspected; Metal compiles its own MSL and does not enter this backend-specific path. |
| I-21 | `rpcs3/main_application.cpp` | supported-renderer detection and Vulkan preference | Detects the Metal build and makes Metal the preferred Apple renderer when its build path is present. Device-level selection remains gated by Metal 4 enumeration. |
| I-22 | `rpcs3/headless_application.cpp` | rejection of real GPU renderers in the null-only headless application | Rejects Metal consistently with the other real renderers rather than constructing a windowless native device path. |
| I-23 | `rpcs3/module_verifier.cpp` | Windows Vulkan runtime-module verification | Inspected; Metal frameworks are OS-provided on macOS and have no redistributable module to verify. |
| I-24 | `rpcs3/rpcs3qt/gs_frame.cpp` | explicit Vulkan surface/frame handling | Inspected; Metal attaches a `CAMetalLayer` to the existing native `NSView` through the backend boundary, so no Vulkan-surface branch is copied. |
| I-25 | `rpcs3/rpcs3qt/render_creator.h` | Vulkan device enumeration declaration | Added a Metal renderer creator with its own device enumeration result. |
| I-26 | `rpcs3/rpcs3qt/render_creator.cpp` | `vk::render_device::enumerate_devices` and Vulkan renderer registration | Enumerates native devices, filters for `MTLGPUFamilyMetal4`, and exposes Metal only when the current OS/device can run the required path. |
| I-27 | `rpcs3/rpcs3qt/gui_application.cpp` | `VKGSRender` renderer factory and Vulkan frame selection | Constructs `MTLGSRender` for `video_renderer::metal` and uses the generic native frame consumed by the Metal layer abstraction. |
| I-28 | `rpcs3/rpcs3qt/emu_settings_type.h` | Vulkan adapter setting enum | Added the Metal adapter setting key. |
| I-29 | `rpcs3/rpcs3qt/emu_settings_type.cpp` | Vulkan adapter config-node mapping | Maps the Metal adapter key to `g_cfg.video.mtl.adapter`. |
| I-30 | `rpcs3/rpcs3qt/emu_settings.cpp` | localized Vulkan renderer display name | Adds the localized Metal display name. |
| I-31 | `rpcs3/rpcs3qt/settings_dialog.cpp` | renderer list, Vulkan adapter list, and Vulkan-only controls | Adds Metal choice/adapter handling, chooses the correct adapter for title preview, and disables Vulkan-only allocator/scheduler/ReBAR controls outside Vulkan. MetalFX exposes its own applicable scaling controls. |
| I-32 | `rpcs3/rpcs3qt/tooltips.h` | renderer, adapter, and FSR/Vulkan-specific tooltip text | Describes Metal/MetalFX behavior without changing the Vulkan descriptions. |

## Integration acceptance

- Metal is independently selectable through `USE_METAL`.
- A configuration with `USE_METAL=ON` and `USE_VULKAN=OFF` configures and
  generates successfully.
- The backend Objective-C++ boundary remains inside `MTL/` and the existing
  Apple frame/layer abstraction.
- Runtime exposure requires a native device reporting Metal 4 family support.
- Adapter serialization, validation, UI display, title diagnostics, factory
  creation, caches, and framework linkage all use the Metal-specific path.
