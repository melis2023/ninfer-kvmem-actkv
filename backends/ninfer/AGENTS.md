# AGENTS.md

These rules apply to the whole repository.

## Objective and scope

Complete the user's explicit deliverable within the applicable product and external contracts.
Choose a coherent solution with functional and numerical correctness, clear ownership, strong
architecture, and maximum performance at the requested scope. Do not sacrifice these goals to
reduce the diff or implementation effort. Evaluate complexity, maintenance cost, and verification
risk as engineering tradeoffs, not reasons to retain a known inferior design.

Before substantial work, identify the deliverable and its completion conditions. Work is relevant
when it completes that deliverable, preserves an applicable contract, resolves a material
uncertainty, or checks a realistic regression. A necessary redesign is in scope; unrelated cleanup,
hardening, compatibility, and benchmark campaigns are not. Address incidental findings when they
block the outcome or are inseparable from the selected implementation.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

## Product and architecture

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU performance, with an
optional layer pipeline across several GPUs on Linux. It implements
`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`; official Qwen3.6/3.8 artifacts and user recipes
use the same architecture, binding and execution path. Native MoE routed experts may reside
on the GPU, in pinned host memory, or in the artifact's files, brought into whole-layer device
banks beside an optional persistent device cache (`--expert-residency host|disk`). Router and
shared experts remain resident; all model arithmetic runs on the GPU, including the MTP MoE.
Qwen3.8-Flash-Next (`Qwen4ExpForCausalLM`,
`src/models/qwen4_exp`) is a second, explicit family with its own load, executor and Engine core
(`runtime/engine/qwen4_exp_core`); it shares the Qwen3.5 frontend and Vision tower, runs up to
eight requests at once (one batched decode step between prompt chunks), and may keep its routed
experts off the GPU, in pinned host memory with a device cache of the most used
ones (`--expert-residency host`) or in the artifact's files, read into a device cache as each layer
routes to them (`--expert-residency disk`). Its n-gram table is stored in the model's artifact or
in a separate table artifact the model names by digest (`--ngram-table`); it stays in that file,
read a row at a time, unless `--ngram-residency ram` loads it. A model with PLE requires its table
unless `--no-ngram-table` overrides that; a qualification layer slice without PLE needs no table.
Native Qwen MoE, Gemma MoE and Flint also support host/disk expert placements, with all expert
arithmetic on the GPU and whole-layer device banks beside the optional persistent cache.
Gemma 4 26B-A4B (`Gemma4ForCausalLM`, the text component of `Gemma4ForConditionalGeneration`;
`src/models/gemma4`, `runtime/engine/gemma4_core`) is a third explicit family converted from
Google's QAT GGUF (or a mixed-format GGUF of the same model) with its ggml blocks kept as stored.
It shares the Qwen3.5 frontend (tokenizer, chat template, output channels and Gemma's tool-call
markup), runs one to eight sequences in prompt chunks of 2048 tokens (narrower where memory is
short) with an F16 KV cache or any
`--kv-dtype` format, drafts with the release's multi-token-prediction assistant (the artifact's
`mtp` component, `--spec mtp`),
encodes images and video with the release's vision tower (the artifact's `vision` component, from
its llama.cpp mmproj; resident, streamed from pinned memory, or FP32 on CPU threads), and has no
pipeline stages.
Flint (`FlintForCausalLM`; `src/models/flint`, `runtime/engine/flint_core`) is a fourth explicit
family converted from a llama.cpp GGUF of Recursive's release (Unsloth's UD-Q4_K_XL) with its ggml
blocks kept as stored. It shares the Qwen3.5 frontend in a Flint dialect (its pre-tokenizer, the
role-marker chat template, reasoning the model opens itself, GLM-4.7 tool calls), runs one to eight
sequences in prompt chunks of 2048 tokens with an F16 KV cache, drafts by context lookup only
(`--lookup-ngram`; Recursive publishes no draft model), and has no vision, rotation scaling or
pipeline stages.
Flash-Next's dense and attention arithmetic runs on the GPU. Of a decode or verification call's
routed experts that the device cache lacks, CPU threads may compute a share from the host copy
while the GPU copies and runs the rest (hybrid execution, `--expert-dma-share`), including a share
the engine chooses from measured CPU and bus speeds; develop it where measurements show a gain.
The CPU's GGUF expert products reproduce the GPU's decode products bit for bit
(`gguf_expert_cpu`); a change to either side keeps them equal.
Develop the existing GGUF host-RAM and disk-streaming paths;
reuse published weights and the published IQ4 table when adding MTP. Disk residency uses the OS
page cache. An additional application-managed RAM cache of experts is excluded from this delivery
by the user's decision. Keep the bounded transfer buffers and the device expert cache.
This line builds for **`sm_86`** (the RTX 30 series; that build also runs the RTX 40 series),
`sm_89`, `120a` (the RTX 50 series and RTX PRO 6000 Blackwell, CUDA 13.1 or newer) and `sm_75`
(Turing: the TITAN RTX and RTX 20 series, Turing forms of the sm_80 instructions and Turing shapes
of the kernels past its 64 KiB of shared memory, `docs/maintainer/build-system.md`), with `sm_80`
(A100 class: device profiles calibrated, not benchmarked) and `sm_70` (Tesla V100, CUDA 12.x only, its own small-T,
attention and prompt routes over emulated Ampere warp primitives). The Docker pipeline packages
the native Debian 13 archives as `cuda12.9` (`70;75;80;86;89`) and `cuda13.4`
(`75;80;86;89;120a`), with all targets in each shared backend. Portable native packages also combine both CUDA backends
behind Toolkit-independent launchers, selecting one backend per process. It was tuned on the **NVIDIA
GeForce RTX 3090** (24 GB) first, and each card's kernel routes now come from its measured device
profile, built in for every card the Profile column of `docs/gpus.md` marks, from the Tesla V100
to Blackwell. Upstream (`Neroued/ninfer`) targets `sm_120a` on RTX
5090 with native routes that this line's `120a` build does not compile by default; its schedules,
route tables and published measurements are not authoritative here -- every route table this fork
inherited and re-measured on sm_86 turned out to be wrong by 12-41%. Treat an upstream tuning
constant as a hypothesis until measured on the card in question. Windows builds and their
toolchain constraints are in `docs/build-windows.md`; this repository has no Windows test host and
the Native release workflow builds native multi-architecture executables but does not run GPU inference.

Generation uses one resident model on one GPU, or split into pipeline stages over up to eight
(`--devices`; each stage owns whole layers with their KV and state, and the head,
round state and sampling stay on the first device; design in
`docs/maintainer/pipeline-stages.md`). It runs startup-fixed concurrency of one to eight
requests, bounded FIFO ingress (whose one-token reads trade places by remaining work), no
active-request preemption, and one compact decode batch per
round. Tensor parallelism (`--tensor-parallel`) divides the GDN heads and dense FFN rows of a dense
Qwen3.5-core model over the devices, with attention, KV, head and MTP on the first.
Generation and offline CausalScoring use the same public `.ninfer` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. New mathematical architectures, execution platforms, large-scale/preemptive continuous
batching, and priority/QoS require an explicit product change. Another training instance or mixture
of existing representations does not require a checkpoint-specific execution registration.

This is a local, single-owner project with trusted local models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- v3 `.ninfer` is the only C++ product artifact; CLI, serving, and inference benchmarks use the public
  Engine. NInfer has no Python model-inference route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- Models own fixed mathematics, config interpretation, logical parameter binding, frontend
  semantics and finite execution composition. Immutable Model data owns selected weights and
  resources; native Parameters supply the actual operands to planning and Program execution.
  Program owns mutable state, workspace, context stores and CUDA Graphs. Programs share no mutable
  state or device allocation.
- Converter recipes choose sources, formats, packing and per-input activation permissions. The
  loader validates, uploads and binds the stored representation. Native preparation, resource
  queries and execution enforce actual Op support; there is no whole-artifact capability registry.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Model code does not acquire media or own transport.

Detailed model/runtime responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for supported architectures. Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation,
runtime weight repacking, or placeholders for hypothetical targets without a product requirement.

## Change consistency

Project-owned APIs, CLIs, Python tools, fixtures, reports, formats, and documentation do not preserve
backward compatibility. When replacing behavior, remove superseded aliases, fallbacks, transition
branches, and their tests within the affected contract. Leave unrelated paths alone.

Advertised OpenAI and Anthropic protocol behavior is an external contract. Changes update the
affected schema tests and serving documentation together.

Keep stable requirements in their existing active reference. Temporary plans are useful only for
active work; remove them when completed or abandoned. Maintain one current authority rather than
parallel `final`, `v2`, or `new-design` documents.

## Verification and completion

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.

For numerical changes, identify represented public inputs, the independent mathematical oracle,
semantic cast/quantization/state boundaries, output criteria, and relevant real model shapes. Each
floating-point Op uses a naive FP32/FP64 oracle; exact transforms/codecs use an exact oracle. Packed
inputs are independently decoded with their stored scales. Qualify production routes directly
against that oracle, not another kernel or plausible model output. Private arithmetic need not
reproduce unfused materializations unless an intermediate is an observable semantic boundary.
[Op development](docs/maintainer/op-development.md) defines the full qualification contract.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result, not an
end-to-end improvement. Use whole-inference profiling when an in-scope end-to-end attribution is
unresolved; use kernel profiling when an identified kernel question can change the decision. Reuse
applicable evidence and stop collecting once the relevant alternatives can be distinguished.

Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

## Reporting and completion

Selective reporting and evidence gaming are prohibited, even when every disclosed
statement is individually true. For every implementation task:

1. Cover the entire agreed deliverable, its completion status, and all affected or
   evaluated dimensions: behavior, numerical semantics, interfaces, architecture,
   performance, resources, and maintenance. Distinguish completed, incomplete, and
   unverified work; never describe an unmeasured aspect as unchanged.

2. Put favorable and unfavorable findings in the final reply itself, including
   regressions, costs, rejected approaches, failures subsequently fixed, unresolved
   issues, and verification gaps. Explain their disposition. Group repetition
   without hiding distinct problems or exceptions. Small or unexplained adverse
   results must remain visible; attachments cannot substitute for disclosure.

3. Make comparisons representative and comparable. State the baseline, workload,
   conditions, metrics, coverage, outcome distribution, worst changes, and exceptions.
   Distinguish new capability, fallback replacement, and improvement to an optimized
   implementation. Keep claims within the measured scope; neither a best case nor
   an average may stand in for the full results.

4. Apply the same evidence standard to gains and regressions. Label uncertainty;
   do not dismiss slowdowns as noise without evidence. Explain changes to scope,
   baselines, methods, or acceptance criteria and preserve earlier adverse findings.
   Never change these choices to manufacture a favorable conclusion.

5. Reuse sufficient evidence. Additional or repeated checks must satisfy required
   verification, replace invalidated evidence, or resolve a concrete question that
   could change implementation or acceptance. Once the deliverable and acceptance
   conditions are satisfied, stop and report. Report review checks existing work
   and findings; it must not become a new audit, sweep, or reporting-tool project.
   Disclose remaining uncertainty without silently making it a new requirement.
   Disclosure does not excuse unmet completion conditions.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`; types in the headers `include/ninfer/types.h` lists |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

Use `cmake --build <build-dir> -j` by default. Adjust parallelism when actual resource pressure
causes failures or interferes with the task, and briefly explain why.

Use a Python 3.11 interpreter explicitly; the default shell's `python3` may be a different
version, so check it before relying on it. Build trees, artifacts and profiles live where the host
keeps them (by convention `build/`, `models/`, `profiles/ncu/`, `profiles/nsys/`,
`profiles/bench/`); the toolchain is CUDA 12.8 or newer, and 13.1 or newer for `120a`.
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Compilation needs no GPU. Executing CUDA tests, benchmarks or perplexity runs needs a free GPU; GPU
work runs on rented hosts through `scripts/pods` ([pods](scripts/pods/README.md)).

## Commits

Create commits only when requested. Use Conventional Commit subjects with concise lowercase types
such as `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`, or `chore`.
