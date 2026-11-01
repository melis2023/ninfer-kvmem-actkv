# Build system

NInfer builds from its source tree with CMake 3.28+ and C++/CUDA 20.
`CMAKE_CUDA_ARCHITECTURES` lists one or more of `70`, `75`, `80`, `86` (the default), `89` and
`120a` ([several architectures](#builds-of-several-architectures)); configure refuses anything
else. CUDA 12.8 or newer builds `75` and the `sm_8x` targets, a build with `70` needs CUDA 12.x
([Tesla V100](../build-linux.md#tesla-v100-sm_70)), and a build with `120a` needs CUDA 13.1 or
newer, so `70` and `120a` cannot share a build.

## Code tiers

Each architecture of a build belongs to a code tier (`ops/common/device_tier.h`): Volta (`70`),
Turing (`75`), Ampere (`80`, `86`, `89`) and Blackwell (`120a`). CMake defines `NINFER_TIER_*` for
each tier the build carries. Device code selects its instructions by `__CUDA_ARCH__`
(`NINFER_SM70_DEVICE`, `NINFER_SM75_DEVICE`, `NINFER_PRE_SM80_DEVICE`,
`NINFER_PRE_BLACKWELL_DEVICE`). Host code that sizes a launch, picks a kernel variant or picks a
route asks for the tier of the device it launches on (`device_tier`, `current_device_tier`): a probe
kernel's `cudaFuncGetAttributes` reports which image CUDA loaded on the device, once per device.
A build of one tier answers without asking.

Kernel shapes come in two sets. The Turing set is sized for 64 KiB of shared memory per block; every
other tier runs the standard set, sized for 99 KiB. A kernel whose shape depends on the set takes it
as a template argument (`TuringShapes`) and its launcher instantiates the sets the build carries
(`with_shape_set`). A kernel of one shape for every card asserts against `kMaxSharedBytesPerBlock`
(`ops/common/device_limits.h`), the smaller limit of the sets the build carries.

### Turing tier (`75`)

The Turing tier runs on Turing (`sm_75`: the TITAN RTX and the RTX 20 series; it is measured on a
TITAN RTX). It compiles the `mma.sync` compatibility path. Turing lacks some instructions that the
path uses, and the device helpers replace them:

- `ops/common/pre_sm80.cuh` holds the forms that Volta and Turing share, under
  `NINFER_PRE_SM80_DEVICE` (device code below `sm_80`): `cp.async` as a plain copy with the same
  cache policy and zero fill (`cp_commit` and `cp_wait` are empty), `redux.sync` as a shuffle
  butterfly, the packed conversions in software, and the FP32 products of BF16, FP16 and TF32
  `m16n8` fragments in a fixed summation order.
- `ops/common/sm75_mma.cuh` holds the Turing tensor-core forms. FP16 `m16n8k16` runs as two
  `m16n8k8`, and INT8 `m16n8k32` as four `m8n8k16`. BF16 and TF32 products run on FP16 `m16n8k8`
  with FP32 accumulation for each output whose row of A and column of B hold only FP16 values, and
  as the FP32 product of the same operands for the other outputs. No value is narrowed, and an
  output's route depends on its own row and column only, so every tiling of a product gives the
  same bits. `ninfer_mma_fragments_test` qualifies each helper against an FP64 oracle through the
  PTX fragment layouts. They are defined in every image but Volta's (`NINFER_SM75_FORMS`), so a
  Turing kernel variant compiles in every image that may hold it.
- The Volta image keeps the Volta forms in `ops/common/sm70_emulation.cuh` (no `ldmatrix`, no
  `m16n8` shapes); the Turing image keeps the native `ldmatrix` and FP16/INT8 shapes.

`ops/common/device_limits.h` holds the 1,024 threads that a Turing SM holds:
`resident_ctas_bound()` lowers a `__launch_bounds__` minimum of resident CTAs that a Turing SM
cannot hold to the number it can (ptxas ignores such a minimum, which leaves the registers
unbounded). The Turing shape set differs from the standard one where a kernel does not fit 64 KiB:

- The standard INT8-family prompt kernel needs 93,184 bytes, so a Turing card does not run it
  (`causal_prompt_i8_standard`): every INT8-family prompt runs on the fast kernel with one K/V
  stage, and the `attn_pack_gqa` and `attn_prompt_fast` switches do not exist there.
- Small-T INT8 attention has no 64-key arena tier (`small_t_i8_key_block64`).
- The FP8 prompt kernel takes 16-key tiles (`CausalPromptFp8Shape`), whose separate widened K and V
  stages fit 64 KiB, so V widens during QK. The small-T FP8, NVFP4 and K8V4 shapes of two and three
  16-row tiles widen K and V into one shared tile (`kCausalSmallTCompactKV`); V overwrites K after
  QK.
- The BF16 prompt and small-T attention take their Turing kernels (`prompt_bf16_sm75.cuh`,
  `small_t_bf16_sm75.cuh`; the prompt kernel stages K and V through one buffer), context-KV
  materialization uses 64-key blocks, the rowsplit A8 GEMM runs one stage and the BF16 GEMM rings
  at most two, and the BF16 `n256_k5120` GEMV splits K eight ways at narrow widths.
- The GGUF flash attention of Gemma 4 (`ops/linear/gguf/ggml_fattn.cu`) takes, per tier, the most
  query rows per tile that the tier's device code has, computes without a race and its shared
  memory fits. llama.cpp's Turing kernel has no tile wider than 32 columns, so width 256 takes 16
  rows instead of 32 and width 512 takes 4 rows (32 columns) instead of 8. The 512-wide tiles keep
  Q in shared memory; they race without the combine-loop barrier that the vendored kernel carries
  ([README.ninfer.md](../../third_party/ggml-quants/README.ninfer.md)).
- The cooperative GDN gating projection reads the occupancy of its compiled kernels on the device
  (`bf16_gdn_gating_proj_mma_resident_ctas_per_sm`). A Turing SM holds fewer of these CTAs than an
  `sm_86` SM, so the widest cooperative routes go to the next route that is resident.

Some Ops round private operands differently on Turing. The Op contracts and the oracle criteria
are the same on every tier:

- BF16-cache attention multiplies Q, K and V as their FP16 roundings: a value from 2^-17 to below
  2^16 is unchanged, and a smaller value changes by at most 2^-25. A tile that holds a value of
  2^16 or more is swept again on the per-output FP32 route (`prompt_bf16_sm75.cuh`,
  `small_t_bf16_sm75.cuh`, `ops/softmax_attention/common/bf16_attention.cuh`). The fast sweep and
  the FP32 sweep are separate functions, because in one loop body the FP32 route made the fast
  route spill.
- Every attention PV product with BF16 values rounds the softmax probabilities to FP16 in the
  Turing image, as the INT8-family kernels do on every architecture: BF16-cache causal attention,
  the packed attention of the vision towers, and the context and sliding-window attention of the
  drafters (`pack_attention_p`).
- An FP16-accumulated product (`mma_f16_f16acc`) rounds its partial sum to FP16 once more per
  16-value step, because it runs as two 8-value steps.

A Turing card selects the legacy Linear route table only (`unified_linear_routes`). Most unified
schedules need more than 64 KiB; a build for Turing alone refuses `NINFER_LINEAR_ROUTES=unified`.
The Q8 G32 K-split contraction (`ops/linear/q8/q8_ksplit_mma.cuh`) forms its codes as FP16 pairs and
runs FP16 MMA on tiles whose live activations are FP16 values, with the operands and order of the
BF16 route, so the outputs are the same bits. It loads the next K group into registers during the
current group's MMAs. A device profile can run any 8-column K-split tile on four K warps at four
CTAs an SM (`q8_ksplit/<rows>x<k>` -> `kw4`, `ops/linear/q8/q8_ksplit_config.h`); the TITAN RTX
profile does so where `ninfer-calibrate` measured it faster
([TITAN RTX measurements](../performance/titan-rtx-2026-10.md)).

### Volta tier (`70`)

A Volta card runs its own projection routes (`ops/linear/sm70`, chosen by `sm70::enabled()`), its
decode attention (`small_t_sm70.cuh`) and its single-row prompt attention (`prompt_sm70.cu`);
those kernels' `mma.m8n8k4` bodies exist in the Volta image only. Its projection weights take the
panel layout on the ranks whose device runs Volta's code (`apply_device_layouts`). Its route
catalog holds the GDN prefill switches alone.

### Blackwell tier (`120a`)

A card that runs the `120a` code also takes the FP8 A8 and NVFP4 W4A4 routes and the fast NVFP4
prompt kernel (`NINFER_SM120_FP8`, `NINFER_SM120_NVFP4`: the build carries them). A stored A8 or
A4 permission is kept only for a weight on such a rank (`apply_rank_policies`); elsewhere the
weight runs its A16 route. The warp-specialized TMA kernels are compiled for `120a` alone
(`ninfer_nvfp4_non_rdc`); the other Blackwell-only instructions compile as traps in the other
images of the build (`NINFER_PRE_BLACKWELL_DEVICE`), which no launch reaches.

### Builds of several architectures

A build may list several architectures, for a layer pipeline across cards of different tiers in
one process (`--devices`): for example `70;75;80;86;89` with CUDA 12.8 or `75;80;86;89;120a` with
CUDA 13.1. CUDA loads on each card the image of the newest listed architecture that the card runs,
and the host follows that image's tier for every launch, so each card runs the kernels, routes,
route catalog and built-in profile of its own single-architecture build. The route catalog and its
digest are per tier (`calibration_route_catalog`), so a calibrated profile entry applies to the
same card in any build that runs it on the same tier. `DeviceContext` admits ranks of different
compute capabilities when each runs the build's code (`docs/maintainer/pipeline-stages.md`).
Compile time and binary size grow with the number of architectures.

`NINFER_CUDA_PRUNE_PTX=ON` runs NVIDIA `nvprune` on CUDA static archives before host linking.
It retains all configured SASS targets and host code, and removes PTX. Object-cache entries remain
unmodified. Windows release builds enable it to reduce image size; local and Linux builds default
to OFF. A pruned build needs a GPU compatible with one of its SASS images, with no PTX fallback
for other targets.

Windows releases also enable `NINFER_WINDOWS_KERNEL_DLL` (local default OFF). The GGML kernels
use a shared library with the same architecture set, while the other components remain static.
The existing bridge exposes functions and plain values; it imports no mutable globals or virtual
tables. CMake exports those functions and stages the DLL beside consumers. The DLL is not passed
to `nvprune`, which accepts static archives and relocatable objects, so it retains PTX.
Compilation uses an object library with the static build's target name and flags; a separate
target links those objects into the DLL. DLL export definitions therefore do not affect the
CUDA compilation target.
The DLL also owns the device-tier probe and its cache, compiled as a separate object library.
Both the GGML bridge and static Ops call its exported functions, so there is one cache and no
circular library dependency. With the DLL option off, the probe remains in the Ops archive.
The compile PDB is fixed to `CMakeFiles/ninfer_ggml_quants.dir/ninfer_ggml_quants.pdb` in both modes:
CMake otherwise changes `/Fd`
from the static library's named PDB to a directory for object libraries, changing the cache key.

Product commands and prerequisites are in the README's [Quick start](../../README.md#quick-start) and
[Building](../../README.md#building) sections; test and measurement workflows live in
[tests](../../tests/README.md) and [benchmarks](../../bench/README.md).

## Configuration

The root `CMakeLists.txt` owns language/toolchain constraints, build options and top-level
composition. Architecture selection and validation happen before `project()` detects CUDA.
The default configuration is Release. Ninja links and archives share the single-slot
`ninfer_link` pool; compilation uses the build command's parallelism.

| Cache option | Default | Scope |
|---|---|---|
| `NINFER_BUILD_APPS` | ON | CLI, HTTP server, perplexity evaluator and `ninfer-calibrate` |
| `BUILD_TESTING` | OFF | C++ tests and registered Python interoperability tests |
| `NINFER_BUILD_BENCHMARKS` | OFF | Op, model, Engine and context-cost benchmarks |

The opt-in kernel, build and platform options (`NINFER_SM120_NATIVE`, `NINFER_PDL`,
`NINFER_MULTICALL`, the SFU and Windows options, and the rest) are listed in the
[Linux build guide](../build-linux.md#build-options).

Apps or tests enable the internal product support components: media acquisition, prompt input,
logging and serving. This is one derived condition, not a separate user option. FFmpeg belongs
to model media decoding and remains required when apps are disabled. Curl and spdlog are needed
only for product support. A Python 3 interpreter is found when tests are configured. The larger Python
pytest suites and conversion/evaluation scripts run separately from CMake and CTest.

Python 3.11 describes the maintained environment, not a configuration version gate. The one CUDA
version gate is 13.1 for a `120a` build: CUDA 12.8 and 12.9 miscompile `sm_120a` kernels.
CMake also discovers FFmpeg without imposing library version floors. Actual language/API support
is exercised by compilation and tests. The libcurl 7.85 minimum has a concrete API basis:
media acquisition uses `CURLOPT_PROTOCOLS_STR` and `CURLOPT_REDIR_PROTOCOLS_STR`, introduced in
[that release](https://curl.se/libcurl/c/CURLOPT_PROTOCOLS_STR.html).

The checked-in `CMakePresets.json` provides:

- `release`: Release product binaries, tests and benchmarks disabled;
- `dev`: Release products, tests and benchmarks enabled;
- matching build presets and a `dev` test preset with failure output enabled.

Both configure presets use `build/`. Switching between them explicitly resets all three build
options; it does not create independent build trees. Build with `cmake --build build -j`.
For a separate configuration, override the binary directory with `-B build-<name>` and use that
directory in subsequent build/test commands.

Keep machine paths in the ignored `CMakeUserPresets.json`. For example, replacing the interpreter
placeholder with the interpreter from the selected environment:

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local",
      "inherits": "dev",
      "cacheVariables": {
        "Python3_EXECUTABLE": "/path/to/python3"
      }
    }
  ]
}
```

Configure that environment with `cmake --preset local`. Compiler paths may also be supplied here;
changing compilers requires a fresh build directory. There is no tools option, installed SDK,
package export or configure-time dependency download.

## Targets and dependencies

`cmake/Dependencies.cmake` discovers system CUDA, Threads, FFmpeg and conditional curl, and
exposes the repository-pinned JSON/HTTP headers and conditional spdlog library. External include
requirements follow their consuming targets. `cmake/NinferTargets.cmake` provides private
project includes and the two CUDA archive policies.

`src/CMakeLists.txt` explicitly enters the component directories. Each component's
`CMakeLists.txt` owns its targets, dependency visibility and compile properties:

| Component | Targets |
|---|---|
| `core/` | `ninfer_core` |
| `artifact/` | `ninfer_artifact` |
| `text/` | `ninfer_text`, including bundled utf8proc C source |
| `media/` | `ninfer_media_decode` |
| `ops/` | `ninfer_ops`, `ninfer_nvfp4_non_rdc` |
| `models/` | `ninfer_model_loading`, `ninfer_model_runtime` |
| `runtime/` | `ninfer_runtime_support`, `ninfer_engine` and its `ninfer::engine` alias |
| `product/` | media acquisition, prompt input and logging libraries |
| `serve/` | `ninfer_serve` |

Component declaration order is not the execution dependency graph: CMake resolves named target
links during generation. Model runtime can refer to the runtime support target declared later;
support links core, while Engine links model runtime. The ownership and dependency contracts are
defined by [Engine architecture](engine-architecture.md) and [Op development](op-development.md).

## Explicit source ownership

Small components list their sources directly. Larger components include local `sources.cmake`
manifests which use `target_sources(<owner> PRIVATE ...)`. Included manifests anchor file paths
with `CMAKE_CURRENT_LIST_DIR`; `include()` does not change `CMAKE_CURRENT_SOURCE_DIR`.

- Every implementation under `src/` has one target owner and one explicit registration.
- Ops register simple horizontal implementations in `basic_sources.cmake` and larger semantic
  families in their own manifests. A family also registers its files under `wrapper/`.
- Linear includes one manifest for each of BF16, FP8, NVFP4, Q4, Q5, Q6 and Q8. Each manifest lists
  all its shape filenames and their actual `.cpp`/`.cu` extensions.
- Models separate loading, frontend, execution and Program source lists while retaining the
  distinct immutable-loading and mutable-runtime targets. Checkpoints/recipes do not define
  additional compile owners.
- New files require an explicit manifest edit. New families require an explicit parent include.
  There is no recursive glob, directory discovery or generated shape/dispatch registry.

Adding a source manifest does not require another library. Add a target only for a meaningful
dependency, reuse or compilation boundary. Shape semantics, supported domains and finite
dispatch remain in C++; CMake only selects translation units.

## CUDA compilation boundaries

`ninfer_core` and `ninfer_ops` compile whole-program: no relocatable device code (RDC) and no
device link. Every device symbol is defined and used in one translation unit. A change that needs a
device symbol across translation units has to bring RDC back to its target and measure the cost:
under RDC ptxas lowers a shuffle from a computed lane to an out-of-line call and stops pipelining
loads across loop back edges.

`ninfer_nvfp4_non_rdc` explicitly disables separable compilation and device-symbol resolution. It
contains the warp-specialized Linear, LinearSwiGLU and causal-attention NVFP4 sources that depend
on `setmaxnreg` register transfer, which RDC would discard. Their owning family manifests register
these sources into that target; host launchers connect them to the normal Ops. All three CUDA
archive targets retain `-lineinfo`.

Whole-program compilation is not a speed lever on every card. On the RTX 3090 (sm_86, CUDA 12.8, 27B
DFlash2 bundle, rk4v4 KV, three interleaved rounds) the base line measured the switch at
pp512/pp2048/pp8192 +0.4/-0.6/-1.4% and decode within +-0.3%, all inside run-to-run spread, while
the server binary grows from 498 to 637 MB; the sm_89 fork reported +2.7-4.9% prefill from it. What
it buys everywhere is the code generation described above; re-measure speed before attributing a
gain to it on a new architecture or toolchain.

Source-list maintenance must preserve language, architecture, RDC mode, device-link ownership
and numerical compiler options. Splitting a manifest does not reduce kernel instantiation work
or justify combining shape translation units or enabling unity builds.

## Include boundaries

The value types of the in-tree interface live in one header per domain under `include/ninfer/`:

| Header | Types |
|---|---|
| `base.h` | `TokenId`, shared limits and defaults |
| `model_enums.h` | Engine purpose, KV capacity mode, speculative drafting, weight residency |
| `kv_cache_storage.h` | `KvCacheStorage` (`--kv-dtype`) |
| `startup.h` | startup phases, events and observer |
| `engine_options.h` | `EngineOptions` and the options it holds |
| `request.h` | requests, prompts, media, tool calls, streamed output, `GenerationResult` |
| `observability/execution.h` | per-request timings, speculative counters, materialization diagnostics |
| `observability/resources.h` | memory and resource summaries |
| `observability/snapshot.h` | `RuntimeStats`, model metadata and residency, slot state, `LoadSummary` |

`include/ninfer/types_fwd.h` declares every struct, class and enum of these headers. A header that
names a type only by reference or pointer, or an enum without its enumerators, includes
`types_fwd.h` instead of the defining header; the file that reads the members includes the
definition. `include/ninfer/types.h` lists all of them, and only `include/ninfer/engine.h` includes
it, which also keeps the declarations consistent with the definitions. Other code includes the
header that defines the types it uses, so a change to one domain recompiles only the code of that
domain. CUDA translation units under `src/` take only `base.h`, `model_enums.h`,
`kv_cache_storage.h` and `startup.h`. `scripts/check-include-boundaries.py` fails when a file other
than `engine.h` includes `ninfer/types.h`, when a `.cu` under `src/` reaches `engine.h`,
`engine_options.h`, `request.h` or `observability/`, or when `types_fwd.h` and the definitions
disagree; CI runs it.

## Consumers and verification

`apps/CMakeLists.txt` explicitly defines the four product executables, or with `NINFER_MULTICALL`
one multi-call executable that bundles them. Tests and benchmarks
include their domain registrations without creating new CMake subdirectories, preserving
`build/tests/`, `build/bench/` and CTest working directories. The helpers are local to those
consumers. Tests explicitly name their linked libraries; public-header and pure-host checks keep
their isolated include/link requirements.

Linear, LinearAdd and LinearSwiGLU reuse compiled test support libraries. Oracle options apply
to those libraries themselves as well as the test executables. Test variants, skip return codes,
interop commands and special linker options belong with the affected registrations.

For build-organization changes, compare configured target/source ownership, compile flags,
device links and CTest commands/properties; check the relevant option combinations, build the
affected targets and run their behavioral tests. A shape-source edit should rebuild its own
translation unit plus necessary downstream links. Build-time or inference-speed claims require
measurements at that scope.
