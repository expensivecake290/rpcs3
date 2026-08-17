# Native SwiftUI Frontend Replacement — Phase 0 Baseline

**Status:** Blocked — do not advance to Phase 1 or Phase 3.

## Repository

- Revision: `fc93d932c8560f763f5223c0a4165cc53bceeb3f`
- Branch: `metal-4-backend`
- Latest commit: `Swap PR number with PR text in update manager`
- Host: macOS 27.0, arm64
- Xcode: 27.0 (`27A5194q`)

## Working-tree safety

The checkout was already dirty before frontend work began. Existing changes were not discarded, staged, or modified:

- `rpcs3/Emu/Cell/PPUInterpreter.cpp`
- `rpcs3/Emu/RSX/Common/texture_cache_helpers.h`
- `rpcs3/util/v128.hpp`
- `rpcs3/Emu/RSX/MTL.zip`
- `rpcs3/Emu/RSX/MTL/`

The WIP Metal 4 renderer path (`rpcs3/Emu/RSX/MTL/`) and archive were explicitly excluded from frontend changes and must remain excluded from frontend compilation work until the user authorizes otherwise.

## Build attempt

Configuration used for a non-Metal baseline attempt:

```text
cmake -S . -B build-phase0-system-mvk -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUSE_NATIVE_INSTRUCTIONS=OFF \
  -DUSE_PRECOMPILED_HEADERS=OFF \
  -DUSE_FAUDIO=OFF \
  -DBUILD_LLVM=ON \
  -DSTATIC_LINK_LLVM=ON \
  -DUSE_LTO=OFF \
  -DUSE_SYSTEM_MVK=ON \
  -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build-phase0-system-mvk --target rpcs3 -- -j 8
```

CMake configuration completed. The build reached the RPCS3 C++ targets but stopped in existing Protobuf-generated sources with errors including:

```text
Protobuf C++ gencode is built with an incompatible version of Protobuf C++ headers/runtime
expected unqualified-id ... google::protobuf::internal::ClassDataLite<...>
```

No SwiftUI, Objective-C++, or frontend replacement files were compiled. The baseline therefore did not produce a runnable RPCS3 binary.

## Phase 0 gate checklist

| Requirement | Result |
|---|---|
| Record exact RPCS3 revision | Recorded |
| Build untouched macOS version | Blocked by pre-existing dirty checkout and Protobuf mismatch |
| Launch Qt frontend | Not reached |
| Confirm library loading | Not reached |
| Boot a known title | Not reached |
| Open all configuration categories | Not reached |
| Open controller configuration | Not reached |
| Confirm firmware workflow | Not reached |
| Confirm package installation workflow | Not reached |
| Confirm patches | Not reached |
| Confirm trophy manager | Not reached |
| Confirm logging | Not reached |
| Record current behavior | Not reached |

## Required next action

Resolve the baseline checkout/build condition without changing the WIP Metal 4 renderer or adding files to it. Then rerun the complete Phase 0 checklist. Phase 1 Qt inventory, Phase 2 configuration inventory, and Phase 3 Swift build foundation must wait until this gate passes.
