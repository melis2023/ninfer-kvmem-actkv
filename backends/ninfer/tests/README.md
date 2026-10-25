# Tests

The retained tests protect current `.ninfer`, numerical operator, model, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Organization

- `artifact/` — v3 framing, directory/binding records, codecs, sharding, selected-object
  materialization and Python-writer/C++-reader interoperability;
- `convert/` — source interpretation, Qwen logical mapping, recipe overrides/sharing, optional
  components, resources, proposals and numerical conversion methods;
- `models/qwen3_5/` — config/binding, frontend, state/context stores, workspace, MTP alignment and
  opt-in real Engine integration;
- `ops/` — semantic Op qualification with independent mathematical or state-transition oracles;
  Linear and fused Linear suites are separated by their supported weight/activation paths;
- root C++ tests — core storage, runtime admission/resource policy, public API, serving protocols,
  logging, benchmark reports and causal-scoring evaluation;
- `test_serve_corpus.py` — agreement between the serving request-log schema and its measurement
  consumer.

Tests are grouped by observable risk, not by mirroring every source file or class.
`CMakeLists.txt` includes explicit registrations from `cmake/`, `artifact/`, `models/qwen3_5/`
and `ops/`. Registration helpers live in `cmake/NinferTests.cmake`; included manifests keep
executables and CTest working directories under `build/tests/`.

Tests link into one executable, `ninfer_tests`, rather than one each: every test that links the
Op library would otherwise carry its own copy of the whole kernel image (~450 MB on sm_86), which
added up to tens of GB. Each test keeps its name as a program in the bundle and still runs in its
own process under CTest. Run one by hand with `build/tests/ninfer_tests <name> [args...]`;
`ninfer_tests --list` prints the names. `ninfer_jinja_test` and
`ninfer_artifact_materialization_test` stay standalone because Python tests invoke them by path.
A test's helpers and entry function must be internal (anonymous namespace) so that programs do not
collide at link time. The mechanism is `cmake/NinferBundles.cmake`.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/Q8, FP8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

Select a Python environment with the dependencies for the tests first. The maintained environment
uses Python 3.11; CMake finds Python 3 without restricting its minor version.
`Python3_EXECUTABLE` selects the interpreter used by interop and frontend tests explicitly.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On a machine without a usable CUDA device -- no GPU, an old driver, or only the CUDA toolkit's stub
`libcuda` (`LD_LIBRARY_PATH` pointing at a directory with `libcuda.so.1` linked to
`$CUDA_HOME/lib64/stubs/libcuda.so`, as CI does) -- every test that needs a device reports as
skipped and the host tests still run. A test asks `tests/cuda_availability.h`; one that fails on its
first CUDA call with the runtime's no-device message is skipped by CTest the same way.

Alternatively, `cmake --preset dev` enables products, tests and benchmarks together.
After building, `ctest --preset dev` runs the same CTest suite. See
[Build system](../docs/maintainer/build-system.md) for local interpreter presets.

The chat-template reference test uses Python Jinja2.

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target ninfer_tests
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|softmax_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

## Running a test under compute-sanitizer

`memcheck` and `initcheck` both work on these binaries, including the large statically linked ones.
Invoke it unqualified so `PATH` resolves it:

```bash
compute-sanitizer --tool initcheck --error-exitcode 9 \
  build/tests/ninfer_tests ninfer_softmax_attention_test
```

**Do not call the copy under `CUDA/v12.4/compute-sanitizer/` by absolute path.** Two copies are
installed on a typical toolkit layout, and the 2024.1.0 one in v12.4 prints its banner and
`ERROR SUMMARY: 0 errors`, exits 0, and **never executes the binary** — a clean report having
checked nothing. `PATH` resolves `compute-sanitizer` to the v12.8 launcher, which is the working
one, so the unqualified form above is safe.

Two messages that look like failures and are not:

- *"Target application terminated before first instrumented API call"* — the test made no CUDA
  calls, usually because it skipped for a missing `NINFER_*_WEIGHTS` environment variable. Set the
  variable, or pick a test that does not need one.
- A single `-k 'regex:a|b'`-style argument disappearing in PowerShell: `|` is parsed as a pipeline
  before the tool sees it. One pattern per invocation.

`ncu` needs one thing more: GPU performance counters are administrator-only by default on Windows
and it fails with `ERR_NVGPUCTRPERM` otherwise. Run it from an elevated shell — that satisfies the
permission per-run, with nothing persistent and no reboot.
`scripts/sweeps/admin-profile.ps1` does this for the profiles the maintainer notes reference.

The variable-width DFlash2 target-attention subset can be run with
`./build/tests/ninfer_tests ninfer_softmax_attention_test --dflash2-only`. It covers D256/Q24/KV4 across all five
cache codecs, W=2..16, B=1..8, request-local prefixes, cache effects, and Graph metadata/input
updates. The default executable also runs the existing attention geometries and prefill tests.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target ninfer_tests
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
```

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

The Linear, LinearAdd and LinearSwiGLU common `.cpp` implementations each compile once into a
test support library. Both those libraries and the Op test executables receive the oracle's
`-fno-fast-math` and `-ffp-contract=off` options on GNU/Clang C++ compilers.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/convert \
  tests/test_serve_corpus.py tests/test_speed_of_light.py
```

The Python suites exercise conversion and encoded output, without running model inference.
The maintained environment uses Python 3.11 with the dependencies for those suites. C++ binding and Engine tests
cover consumption of their resulting representation.

The real loading test accepts an explicit artifact path and optional component selection:

```bash
./build/tests/ninfer_tests ninfer_qwen3_5_loading_real_test \
  --artifact out/qwen3_6_27b.ninfer --vision --speculative mtp --proposal optimized
```

Add `--host-only` to check semantic binding without uploading weights. This does not construct a
Program or establish native Op support.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine. It accepts Qwen3.6 and Qwen3.8 artifacts; its prompt-token goldens are kept
per model generation because each registers its own chat template:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_prefix_real_test --output-on-failure
```

The causal-scoring integration test uses the same artifact variable and checks a full 1,024-column
score tile, overlapping target suffixes, and repeated-window State/KV isolation:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_score_real_test --output-on-failure
```

Run the 35B-A3B MoE route independently:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_moe_real_test --output-on-failure
```

Qwen3.8-Flash-Next runs end to end from a converted GSQ-RCO release. The generate test answers
fact prompts and a 4,463-token needle, decodes three sequences as one batch and from a restored
snapshot, and, where decode replays CUDA graphs, checks every token against eager decode.
`NINFER_QWEN4_EXP_EXPERTS` picks `device`, `host` or `disk`, `NINFER_QWEN4_EXP_DEVICES` the pipeline
devices, and `NINFER_QWEN4_EXP_EXPERT_CACHE_MIB` the device expert cache of host or disk experts
(a 24 GB card's on a larger one); a model stored without its n-gram table takes the table artifact
from `NINFER_QWEN4_EXP_NGRAM_TABLE`:

```bash
NINFER_QWEN4_EXP_ARTIFACT=$PWD/models/flash-next-q2_0.ninfer \
NINFER_QWEN4_EXP_NGRAM_TABLE=$PWD/models/flash-next-ngram-table.ninfer \
NINFER_QWEN4_EXP_EXPERTS=host \
  ctest --test-dir build -R ninfer_qwen4_exp_generate_real --output-on-failure
```

Its Ops are also composed over a slice of the BF16 checkpoint against an FP64 golden.
`tools/reference/fetch_slice.py` range-fetches the first blocks, embedding, final mixer, head and
the needed n-gram rows (`--exclude ngram_embedding` keeps the 102 GB table out), and
`tools/reference/qwen4_exp.py --layers 4 --head --out DIR/golden` writes the golden:

```bash
NINFER_QWEN4_EXP_SLICE=$PWD/fn_slice \
  ctest --test-dir build -R ninfer_qwen4_exp_slice_real --output-on-failure
```

Token log probabilities run through the public Engine on either family. The test checks that each
record spells its token's text and that the drawn token tops its distribution under greedy
decoding, that gathering them changes no generated token, that streaming, a stop string, three
requests decoded as one batch and a JSON grammar's mask all reach the records, and that a request
without them gathers none. `NINFER_LOGPROBS_ARTIFACT` names the artifact, and the test is skipped
without it; `NINFER_LOGPROBS_NGRAM_TABLE` names a Qwen3.8-Flash-Next table artifact, and
`NINFER_LOGPROBS_SPECULATIVE=mtp` or `dflash2` verifies the tokens in speculative rounds:

```bash
NINFER_LOGPROBS_ARTIFACT=$PWD/out/qwen3_8_27b.ninfer NINFER_LOGPROBS_SPECULATIVE=dflash2 \
  ctest --test-dir build -R ninfer_engine_logprobs_real_test --output-on-failure
```

Without `NINFER_TEST_ARTIFACT`, CTest marks these real Engine tests as skipped. Run GPU integration
tests serially. `NINFER_PREFIX_REAL_SCENARIO` selects a focused prefix scenario such as `vision`,
`pressure-resume` or `concurrent`; the default is `all`. These integration checks
use behavior and state accounting rather than another numerical path's generated tokens as a golden.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares stripped and preserved closed-turn prompt lengths, and verifies compatible
prefix reuse, speculative execution, frontier bounds and Responses inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- model Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.

## DFlash2 Engine integration

The DFlash prefill regression checks actual draft KV contents after a StateImage fork and a
conflicting decode binding, including shortened chunks and oversized local/full KV appends. It
uses native Program storage and the production prefill route; select the draft component stored in
the artifact:

```bash
cmake --build build -j --target ninfer_qwen3_5_dflash_prefill_real_test
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_qwen3_5_dflash_prefill_real_test dflash2
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_35b_a3b.ninfer \
  build/tests/ninfer_qwen3_5_dflash_prefill_real_test dflash
```

The Engine test uses an artifact containing DFlash2 and checks output budgets, speculative activity,
a forced thinking-control append, penalty-enabled sampling, compact batches with unequal budgets,
same-route same-seed replay,
retained/fresh prefix behavior and absence of a full backend KV pool. A shared DFlash/DFlash2 fixture starts decode at token 63, verifies across the page
boundary, stops after one target column at token 64, and checks the exact retained frontier and
subsequent generation with and without reuse.
The KV Store test checks exact mapping and reservation accounting for the same transition.
K>=7 also exercises a stop inside a licensed block; K=15 additionally checks oversized prefill,
local ring wrap, and the logical context-capacity tail. Optional Vision runs image/video capture
and prefix restore. Zero extra Device StateImage slots exercise Host snapshot/restore.

```bash
cmake --build build -j --target ninfer_tests
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_tests ninfer_qwen3_5_dflash2_real_test 15 1 1 8
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_tests ninfer_qwen3_5_dflash2_real_test 7 1 0 2 bf16 1 0
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_tests ninfer_qwen3_5_dflash2_real_test 2 0 0 2 int8
```

Arguments are K, Graph enabled, optimized head enabled, maximum B, target KV (`bf16` or `int8`),
Vision enabled, and extra Device StateImage slots. Defaults are `15 1 1 8 bf16 0 3`. Run GPU
integration tests serially. The individual Op suites remain the numerical/state-transition oracle;
the fixed Engine fixture does not define bit parity across arbitrary floating-point routes.

## Structured output

The CPU grammar and protocol tests and the GPU sampling/speculative tests qualify this path:

```bash
ctest --test-dir build --output-on-failure -R 'ninfer_(structured_output|sampling|speculative_round|openai_schema|openai_responses|anthropic_schema|cli_options|prompt_input|tool_call_parser|qwen3_5_frontend|qwen3_5_structured_round)_test$'
```

`tests/test_structured_output_live.py` runs a temporary loopback server and validates completed
outputs with the independent Python `jsonschema` validator. Use Python 3.11 with `requests` and
`jsonschema` installed in a test environment. Supply an explicit v3 artifact containing the
selected speculative backends; the test never downloads or converts weights.

```bash
python tests/test_structured_output_live.py \
  --server build/apps/ninfer-serve --artifact /absolute/path/model.ninfer \
  --output-dir work/structured-live --modes none mtp dflash2 dflash2-eager
```

It checks conflicting prompts, greedy and stochastic generation, schema versus JSON object mode,
concurrent and mixed traffic, prefix reuse, SSE, token limits, disconnect cleanup, compile errors,
Responses, and Anthropic Messages. It also runs a tool-result-to-bounded-JSON exchange while
reasoning and multiple tools remain enabled, including a Markdown-formatted example in the prompt.
CPU tests cover numeric endpoints, schema enforcement with tool alternatives, split and mixed-token
reasoning transitions, speculative mask previews, injected thinking-budget closure, and prompt cache
boundary preservation. `--concurrency 8 --draft-tokens 15 --modes dflash2` exercises
the largest draft and batch dimensions. A separate DFlash-capable artifact can use `--modes dflash`.
The server is terminated on success or failure. An occupied test port causes the test to stop.


The old/new Qwen3.8 binding matrix uses `out/qwen3_8_27b_old.ninfer` and
`out/qwen3_8_27b_nvfp4_old.ninfer` for the legacy inventories, and the canonical filenames above
for the companion artifacts. The legacy paths may be overridden with
`NINFER_QWEN3_8_27B_OLD_WEIGHTS` and `NINFER_QWEN3_8_27B_NVFP4_OLD_WEIGHTS`.

## KVMem canonical-input fallback

`ninfer_p4_history` accepts the `input-fallback` mode with the usual artifact,
KV format, execution mode, output file and MTP width arguments:

```powershell
./build/tests/ninfer_p4_history.exe C:/models/model.ninfer rk8v4 graph C:/work/fallback.tokens 4 input-fallback
```

The fixture exceeds the physical window, crosses a controlled thinking-close
boundary, then discards generated answers and appends synthetic tool records.
It checks input-prefix reuse on successive turns, repeated input and cancellation
recovery, with exact native transfers and bounded Host/device occupancy. Width 0
checks ordinary generation; width 4 also enables ngram proposals.

`input-fallback-disk-write` and `input-fallback-disk-read` use separate processes
with output files in the same directory; the disk cache is its `disk` subdirectory.
The read case must restore the earlier input after a generated reconstruction
was saved. `input-fallback-disk-corrupt` expects a damaged snapshot to be rejected
and recomputed before exercising subsequent reuse.

## KVMem NVFP4 integration

The source accepts `nvfp4` with KVMem and fixed MTP1..4. On 2026-10-04 the focused
codec, Attention, page payload, options and MTP graph-profile tests passed on the
RTX 5050 Laptop. Sparse/reordered/cross-page cases passed for eager and graph.
Bonsai ordinary, MTP1 eager and MTP4 CUDA Graph execution also passed at B128/R128,
including Host restoration/reuse, 129-token reserve crossing, output limits 1..5
and HTTP cancellation/recovery. MTP used the existing D3D12/WDDM build/runtime option,
FP16 GDN and disabled ngram; the default CUDA budget rejected the Q8 MTP load.
On the same date, RTX 5060 Ti / IQ3S completed 229 numerical, functional, quality
and fault-injection cases: ordinary/MTP1..4 eager/Graph, retained sessions, C2/C4
text, resident vision, cold snapshots and 8K/32K/64K/128K NVFP4 retrieval. Three
independent B32768/R16384 MTP4 Graph service starts and long-input HTTP reuse,
output limits and cancellation also passed under the default CUDA policy. Exact
commands, failed attempts and test corrections are in the local integration
[qualification report](../../../docs/nvfp4-kvmem-validation-20261004.md).
This qualifies the tested source/build/model combination, not an existing package.
Use an explicit text+MTP artifact and a free GPU selected by UUID. Functional
tests can use either local card; performance comparisons use the RTX 5060 Ti.
Keep ngram and lookup-ngram disabled for this qualification.

Build the affected targets with eight jobs. The existing
`ninfer_p1_options`, serve-options and Windows launcher tests cover NVFP4
ordinary/MTP1..4, 1..4 text lanes, resident vision, cold snapshots and rejected
combinations. `ninfer_p1_payload` adds an independent four-layer Main/MTP byte
layout in both device plane orders; `ninfer_p1_sparse nvfp4` checks fragmented,
reordered compact mappings against FP64 attention over independently decoded
stored codes/scales, including widths 1..5, 13, 53 and 63..65 in eager/graph execution.
NVFP4 future unwritten K/V scales contain NaNs, so the 90 cases also check that
masked tile tails cannot contaminate valid outputs.
The 8K quality fixture checks same-seed token equality separately for cold
prefill and cached short-suffix execution, then changes the seed on the cached
route. Cold-versus-cached equality is reported, not required: IQ3S INT8 MTP4
produces two different but individually repeatable sampled outputs. Native
payload/state transfers and greedy access-code recovery remain exact checks;
numerical route changes do not promise identical stochastic token choices.
`ninfer_kvmem_nvfp4_tests` bundles the native append, Attention, MTP graph-profile
and serve-options tests for focused runs. Pass each program name to this executable;
the codec and Attention programs accept `--nvfp4-only`.

Real-model fixture commands below show parameter syntax; the report records
the executed paths, commands and outcomes:

```powershell
# Repeat fixed drafts 0..4 and both execution modes; the last argument keeps ngram off.
./build/tests/ninfer_p4_history.exe C:/models/model-mtp.ninfer nvfp4 graph C:/work/history.tokens 4 input-fallback 0
# Optional final "wddm" uses the existing NINFER_D3D12_RESIDENCY build capability.
./build/tests/ninfer_p4_history.exe C:/models/model-mtp.ninfer nvfp4 eager C:/work/history.tokens 1 input-fallback 0 wddm
./build/tests/ninfer_memory_sessions.exe C:/models/model-mtp.ninfer 4 nvfp4 eager
./build/tests/ninfer_memory_concurrency.exe C:/models/model-mtp.ninfer 4 4 512 nvfp4 graph
./build/tests/ninfer_memory_concurrency.exe C:/models/model-mtp.ninfer 4 2 52 nvfp4 eager
# Eight mixed text/image requests: independent archives, colors, cancellation, reuse and image replacement.
./build/tests/ninfer_memory_concurrency.exe C:/models/model-vision-mtp.ninfer 3 8 1024 int8 graph adaptive C:/work/red.png C:/work/blue.png cpu 1024
./build/tests/ninfer_memory_disk.exe C:/models/model-mtp.ninfer 4 nvfp4 C:/work/nvfp4-disk cold 2048 0 long graph
./build/tests/ninfer_memory_disk.exe C:/models/model-mtp.ninfer 4 nvfp4 C:/work/nvfp4-disk hit 2048 0 long graph
# Requires an artifact with resident vision and the existing red/blue image fixtures.
./build/tests/ninfer_memory_vision.exe C:/models/model-vision-mtp.ninfer 4 kvmem C:/work/red.png C:/work/blue.png nvfp4 graph
```

History covers repeated requests, edits, cancellation, endpoint/input-fallback
rewinds and output truncation. Sessions scale their existing INT8 Host-pressure
quota by native payload size; concurrency keeps an explicit actual MiB quota and
uses a format-adjusted pressure threshold. For the IQ3S fixture, 96 MiB queues
the INT8 requests, while NVFP4 needs 52 MiB to keep the same pressure: 96 MiB can
admit two NVFP4 requests. Check native Main/MTP payload and GDN
state equality with `verify_transfers`, committed mean-K/frontiers, no active page
leaks and bounded Host usage. Cold qualification also needs cancellation, explicit
clear, corruption/truncation fallback and a format-mismatch restart; use a separate
directory for each positive cold/hit pair. An INT8 snapshot restarted under NVFP4
must report incompatibility and recompute rather than reuse it. Capacity was
separately verified with IQ3S MTP4 at 32K/16K on the 5060 Ti; Bonsai without MTP
was verified at 20K/10K on the 5050. Bonsai Q8 MTP under the default CUDA budget
does not fit that 8GB card, so the low-memory WDDM passes are recorded separately.

## Container build cache

Check the Dockerfile's incremental-build and configuration-invalidation behavior without compiling
or running NInfer:

```bash
bash tests/test_docker_build_cache.sh docker
```

This uses the real build stage with a tiny CMake fixture and isolated cache mounts. It checks added
and removed compiler flags, changed and restored cached defaults, environment flags, old header
timestamps, non-code edits, and one-file incremental compilation. `NINFER_TOOLCHAIN_IMAGE=<image>`
stands in for the Dockerfile's toolchain stage, as CI does with the one it publishes. Docker BuildKit is required;
`podman` can be passed instead to check that builder. Logs and the fixture remain in an ignored
`build-cache-test.*` directory; the test image and its small build caches remain in the builder.
The check needs the Dockerfile's build dependencies but no GPU or model weights.

## CPU structured-output qualification

`ninfer_structured_output_test` and `ninfer_unicode_scalar_output_test` use a byte
vocabulary and EOS without loading an artifact or executing CUDA. The stdin JSONL
`ninfer_native_schema_probe` preserves schema declaration order and tests native
compilation, token masks, accepted transitions and EOS. It is an executable fixture,
not an unattended CTest target.

`text/native_schema_oracle.py` generates and independently checks the corpus using
Python `jsonschema`, calendar/time arithmetic, strict UTF-8 and Node.js ECMA262 regexes.
Node is found on PATH, or selected through `NINFER_ORACLE_NODE`. Pass `--schemas` for
an optional JSON list of `{name, schema}` application schemas; these are compile-only.
No model or inference HTTP server is used by these tools.

```bash
cmake --build build --target ninfer_structured_output_test ninfer_unicode_scalar_output_test ninfer_native_schema_probe --parallel 2
ctest --test-dir build -R '^ninfer_(structured_output|unicode_scalar_output)_test$' --output-on-failure
python3 tests/text/native_schema_oracle.py --emit-corpus corpus.jsonl --report oracle.json
build/tests/ninfer_native_schema_probe < corpus.jsonl > native.jsonl
python3 tests/text/native_schema_oracle.py --native-results native.jsonl --report comparison.json
```

These host tests establish supported grammar semantics. Actual tokenizer,
speculative decoding and concurrent serving behavior are exercised separately
by the real-model structured-output tests described above.

The `ninfer_unique_strings_test`, `ninfer_structured_unique_test` and
`ninfer_unique_strings_masks_test` are CPU-only. They exercise decoded-string
equality, remaining enum prefixes, nested scopes, rollback/forks, actual mask versus
commit consistency, multi-item vocabulary tokens, EOS and reasoning/MTP boundaries.
The JSONL `ninfer_schema_normalization_probe` reports normalized schemas for an
independent JSON Schema validator to compare accepted instances. Neither probe
loads model artifacts.

`schema_semantics_oracle.py` generates independent Draft 2020-12 cases for
string uniqueness, escaped equality, finite-domain exhaustion, nested scopes,
union intersections, local references and explicit unsupported contracts.
`compare_schema_semantics.py` compares the native byte-vocabulary probe with
the unchanged input schema. Accepting an invalid value always fails; expected
compile errors and declared serialization restrictions are reported separately.

```bash
python3 tests/text/schema_semantics_oracle.py --output semantic-corpus.jsonl
build/tests/ninfer_native_schema_probe < semantic-corpus.jsonl > semantic-native.jsonl
python3 tests/text/compare_schema_semantics.py --cases semantic-corpus.jsonl --results semantic-native.jsonl --report semantic-comparison.json
ctest --test-dir build -R '^ninfer_(structured_output|unicode_scalar_output|unique_strings|structured_unique|unique_strings_masks)_test$' --output-on-failure
```

## KVMem K8V4 compact qualification

K8V4 reuses native FP8 K / NVFP4 V CUDA kernels. The independent payload oracle
checks three Main layers plus one MTP layer in both page orders, including the
FP16 row K scale and raw E4M3 group V scales. Sparse Attention uses FP8-quantized
rotated Q, represented K/V cast boundaries and inverse Hadamard; future K FP16
and V E4M3 scales are poisoned with NaNs under causal/tile masks.

The existing focused bundle supports `ninfer_kv_cache_append_test --k8v4-only`
and `ninfer_softmax_attention_test --k8v4-only`; use `ninfer_p1_sparse k8v4` for
fragmented/reordered pages, widths 1..5, 13, 53, 63..65 and eager/Graph masking.
The options and Windows launcher checks cover ordinary/MTP1..4, lane budgets,
vision/cold settings and rejected combinations for both NVFP4 and K8V4.

The 2026-10-04 real IQ3S/5060 Ti qualification deliberately uses 26 processes:
12 history, sessions, C4 normal/pressure, resident vision, five cold-disk runs
(including an NVFP4 mismatch seed), two fallback cold runs, 8K/32K quality and
one 32K/16K MTP4 Graph serving start. All ngram/lookup paths are disabled. Each
process can issue several requests; foundational numerical cases are counted
separately and no unrelated suite is repeated. Commands, scope and timings are
in the [report](../../../docs/k8v4-kvmem-validation-20261004.md).


## KVMem parameter-extension qualification

The 2026-10-04 follow-up reused the existing fixtures with 12 configurations, three native
diagnostic launches and 18 model launches total, retaining all four failed attempts.
`ninfer_p4_history` accepts a final `adaptive` or
`fast` argument in its existing feature slot. `ninfer_memory_concurrency` accepts optional
`fast`; `ninfer_memory_vision` accepts `resident|cpu`, then optional `query-only|native-boundary`.
The query slice checks replay, repeated reuse, multiple query images, cancellation and recovery;
the boundary slice checks one native request whose first visual column starts at the unit boundary.
The focused Attention bundle
supports `--rk8v4-only`; `NINFER_PROMPT_FAST=1` qualifies its existing fast prompt route
against the independent represented-input oracle. `--int8-prompt-only` covers both routes.

The chat endpoint fixture honors model stop tokens and compares reuse with the actual
generated prefix, allowing its bounded template closure. Forcing a visible assistant response
past EOS into new role tokens does not create a reconstructible chat prefix. Other history
checks retain their exact output-budget assertions. Details and supported bounds are in the
parent [report](../../../docs/parameter-admission-validation-20261004.md).

## KVMem adaptive-combination follow-up

The focused follow-up uses ten model processes on IQ3S/RTX 5060 Ti, reusing
existing numerical evidence. The final feature slot in `ninfer_p4_history` and
`ninfer_memory_concurrency` accepts `adaptive-fast`; concurrency also accepts
`adaptive`. The ngram fixture accepts a final `adaptive` after its MTP draft
width. Vision accepts `adaptive` after `resident|cpu` and `query-only`;
disk accepts it after `ngram-width short|long eager|graph`.
Each fixture checks the adaptive observation and records its window histogram;
existing reuse, isolation, exact ngram greedy comparisons, cancellation,
resource retirement and transfer assertions remain active. Disk cold and hit
are separate processes sharing one dedicated directory. Fast prefill remains
INT8/RK8V4 text-only without cold disk. Ngram supports one text or INT8/NVFP4 resident/CPU Vision lane, max63;
That follow-up qualified one resident/CPU image lane; cold snapshots require profile off.
The new bounds and measured configurations are recorded in the parent
[report](../../../docs/adaptive-combinations-validation-20261005.md).

## KVMem images with ngram

`ninfer_memory_vision` accepts `all|query-only|native-boundary`, then
`fixed|adaptive`, then an optional ngram width. `copy-compare` runs only the
cold/restored copy pair, including ngram-off or native controls. A nonzero width adds real image
and text copy requests with positive wide-draft acceptance, exact cold/restored
greedy and seeded-sampling comparisons, cancellation/recovery, output limits
and edited-prefix rejection. Copy requests honor model stop tokens; the existing
fixed-length color fixtures keep their original termination policy. Transfer
verification logs must prove query replay, not merely a long input or an enabled
flag. The parent [qualification report](../../../docs/vision-ngram-validation-20261005.md)
records the initial two admitted configurations, the K8V4 observation and controls.
The current [all-KV qualification](../../../docs/vision-ngram-all-kv-validation-20261006.md)
records the subsequent admission of all five KV formats and concurrent checks.

## KVMem image concurrency (2026-10-06)

The Engine, serving CLI and Windows launcher now admit 1..8 text or resident/CPU
image lanes. Ngram widths 1..15 also admit 1..8 lanes with MTP. Valid requested
widths 16..63 automatically reduce to 15 at startup with C2..8, with a warning;
C1 preserves those widths. Image ngram admits all nine native KV profiles (including FP8, RK4V4,
RK4V4-E8 and RK2V4-E8). CPU image encoding remains
serialized; active image and text requests share the native compact decode batch.
The launcher's default image limit is 1024 merged tokens.

`ninfer_memory_concurrency` accepts the optional four final arguments
`red.png blue.png resident|cpu vision-tokens [selected-tokens]`, after the KV, Graph and feature
arguments. The feature can be `fixed` for ordinary or fixed-MTP execution.
Image fixtures use B1536/R128 by default; two actual 1024-token images require
a larger selection budget, for example a final `2560`. Preparation logs record
the actual vision-token count separately from the configured per-image limit.
The fixture mixes red images, blue images and text, with distinct access codes,
then checks lane isolation, cancellation, restored peers, image replacement
under the same source name, compact batching, configured active lanes, global
Host admission and resource retirement. The current build and validation limits
are recorded in the parent [report](../../../docs/vision-concurrency-validation-20261006.md).

`ninfer_memory_ngram` now accepts `fixed|adaptive` after the MTP draft count,
then an optional concurrency in 2..8, then optional `red.png blue.png resident|cpu`.
Its concurrent route mixes distinct exact Python copies with a free-form neural
lane, requires positive copy acceptance and compact batching, and checks full
configured overlap after restore, exact cold/restored/recovered copy tokens,
cancellation isolation, global Host bounds and retirement of Main/MTP pages.
The image route uses a 1024-token limit with 512x512 color fixtures.
Text copies use B768/R128 so the complete copy query survives selection of
the longer history. Image copies use B1536/R128. The optional environment
`NINFER_NGRAM_TEST_COLORS_ONLY=1` runs only the color diagnostic and prints
`MEMORY_NGRAM_COLORS_ONLY PASS`; it does not qualify concurrent copying.
MTP/ngram Graph families now eagerly warm every batch shape before capture,
including GGUF linear routes that first need cuBLAS above 64 verify columns.

## Additional KVMem KV profiles (2026-10-06)

`ninfer_p1_payload` covers the independently specified FP8, RK4V4, RK4V4-E8 and
RK2V4-E8 Main/MTP plane schemas in page-major and head-major order.
`ninfer_p1_sparse <format>` checks each against the existing independent represented-value
attention oracle across 90 sparse/page-boundary/eager/Graph cases. RK4V4 selects the
maintained packed-key `CachePlan` fixture. E8 profiles additionally require exact
fused/standalone append byte parity; numerical criteria are unchanged.
The Engine/serving/launcher tests admit all nine profiles, retain unknown-format
rejection and preserve disabled cold snapshots and the fast-prefill restrictions.
Representative real-model history, concurrent image/ngram and cancellation evidence
is in the [additional KV report](../../../docs/other-kv-admission-validation-20261006.md).

## KVMem fast-prefill admission (2026-10-06)

`ninfer_p1_sparse <format> fast` selects INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8
and checks 36 cases per format (T65/127/128/129, sparse/permuted pages, offset0/63,
eager/Graph). It asserts that the native route selects prompt Attention and the fast
kernel, then applies the existing independent represented-value oracle and cache/guard checks.
`NINFER_TEST_FAST_PREFILL=1` explicitly sets `EngineOptions.fast_prefill_kernel`
in `ninfer_memory_vision` and `ninfer_memory_ngram`; 0 disables it and other values reject.
The fixtures publish the setting as `MEMORY_FEATURE fast_prefill=1`. Normal test defaults
remain off. Parameter tests admit the five fast profiles with C1..8 resident/CPU images
and MTP/ngram while preserving rejection of unsupported KV, concurrent prefill and cold disk.
Cold prefills are serialized, so `ninfer_memory_ngram` records their actual overlap without
requiring it. Each of the three restored phases still requires full configured concurrency
and more decode row-rounds than decode rounds; exact token/byte/resource checks remain unchanged.
See the [fast-prefill report](../../../docs/fast-prefill-admission-validation-20261006.md).
