# NInfer documentation

Start with the [project README](../README.md) to download an artifact and run the HTTP server from
the Docker image or from a build. The executables' `--help` output is the exact source for option
spelling and defaults.

## User guides

| Document | Purpose |
|---|---|
| [Linux build](rtx-3090-linux.md) | the container image, native Ubuntu builds for `86`, `89` and `120a`, build options and the Bash launchers |
| [Windows](rtx-3090-windows.md) | native Windows builds and the launcher scripts |
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, speculative decoding, and the runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, llama.cpp endpoints, the router, model suspend, state, streaming, token counting, authentication, tool calls, and the server options |
| [Ngram copy proposals](ngram.md) | copy acceleration alongside MTP, DFlash or DFlash2, on by default with a drafter |
| [Performance](performance.md) | the reference measurements on RTX 3090, 4090, 5090 and RTX PRO 6000, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [GGUF block formats](gguf.md) | GGUF releases with a ggml type per tensor: the eighteen block formats, their products, serving and measurements |
| [Qwen3.8-Flash-Next](qwen3-8-flash-next.md) | converting the GSQ-RCO GGUF releases and their n-gram table; experts on one or several GPUs, in host memory or on disk |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [Device profiles](device-profiles.md) | per-GPU route profiles: the built-in RTX 3090/4090/5090/PRO 6000 table, calibration at first start, `ninfer-calibrate` |
| [Configuration calculator](config-calculator.html) | which context, KV format and speculation fit an RTX 3090 with Qwen3.8-27B or Qwen3.6-35B-A3B, from measurements |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |
| [Release archives](release-archive-linux.md) ([Windows](release-archive-windows.md)) | the README that `scripts/package-release.sh` and `.ps1` put in a release archive |

## Model artifacts

This line's own artifacts (Ternary Bonsai 2, the GSQ-RCO and Flash-Next conversions, the
Qwen3.8-27B fine-tunes and Qwen3.6-35B-A3B NVFP4) are in the README's
[artifact table](../README.md#artifacts); their cards live on Hugging Face. The official upstream
artifacts load here too, and their cards are versioned in this repository as upstream publishes
them: their requirements and measurements are upstream NInfer's on one RTX 5090. On this line the
`groupwise-int` artifacts run on every supported card.

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |
| Qwen3.6-35B-A3B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | [model card](../model-cards/Qwen3.6-35B-A3B-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [Qwen3.8-Flash-Next plan](maintainer/qwen3-8-flash-next-plan.md) | the `Qwen4ExpForConditionalGeneration` family: its mathematics, byte census, mapping onto the engine, the milestones done and the work not started |
| [Pipeline stages](maintainer/pipeline-parallel-plan.md) | `--devices`: whole-layer stages over several GPUs, what they cover, verification and measurements |
| [Quality trades](maintainer/quality-trade-experiments.md) | the precision flags (`--lm-head-q4/q6`, `--embedding-q4/q6`, `--mtp-experts-q4`, `--gdn-state-fp16`, `--mlp-a8-decode`) and what each costs |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | candidate selection, retention, materialization and Device/Host checkpoint policy |
| [Hybrid prefix cache](maintainer/hybrid-prefix-cache-spec.md) | the `--use-alt-prefix-caching` mode: block tree, sparse state snapshots, tap placement, Host slab tier, eviction and automatic configuration |
| [Research notes](maintainer/research-notes.md) | candidates that were built, measured and not routed, with the evidence, so a later round does not rebuild them |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |
| [Cutting a release](maintainer/release-process.md) | the release flow, the packagers, smoke tests, and the gotchas that nearly shipped a broken archive |
| [Launcher profiles](maintainer/launcher-profiles.md) | `run`, `download-model` and `package-release`, the serving profile per model, and the measurements behind each default |
| [The consolidated line](maintainer/consolidated-line.md) | the maintainer map: every change this line carries over its base, its author, the files it touches and the tests that cover it |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.
