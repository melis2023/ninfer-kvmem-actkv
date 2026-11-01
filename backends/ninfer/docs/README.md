# NInfer documentation

Start with the [project README](../README.md) to download an artifact and run the HTTP server from
the Docker image or from a build. The executables' `--help` output is the exact source for option
spelling and defaults.

## User guides

| Document | Purpose |
|---|---|
| [Linux build](build-linux.md) | the container image, native Ubuntu builds for every target (`70`, `75`, `80`, `86`, `89`, `120a`), build options and the Bash launchers |
| [Windows build](build-windows.md) | native Windows builds, their toolchain constraints and the launcher scripts |
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, speculative decoding, and the runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, llama.cpp endpoints, rerank and typed decisions, the router, model suspend, state, streaming, token counting, authentication, tool calls, and the server options |
| [Ngram copy proposals](ngram.md) | copy acceleration alongside MTP, DFlash or DFlash2, on by default with a drafter |
| [Launch recipes](recipes.md) | the `ninfer-serve` commands for every model and memory size, from an 8 GB card to the largest context, and the flags they use |
| [Supported GPUs](gpus.md) | architectures and their builds, every card of 8 GB and more with what has run on it, and what fits in a given amount of GPU memory |
| [Performance](performance.md) | every measured result card by card: largest context, decode and prompt rates by model, KV format, depth and drafter, several requests at once, Flash-Next's expert placements, quality; the measurement pages and users' field reports |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [GGUF block formats](gguf.md) | GGUF releases with a ggml type per tensor: the eighteen block formats, their products, serving and measurements |
| [Qwen3.8-Flash-Next](qwen3-8-flash-next.md) | converting the GSQ-RCO GGUF releases and their n-gram table; experts on one or several GPUs, in host memory or on disk; measurements |
| [Gemma 4 26B-A4B](gemma4.md) | converting Google's QAT GGUF release with its MTP assistant and vision tower, what runs, speculative decoding, images and video, the defaults, measurements against llama.cpp and the numerical checks |
| [Flint](flint.md) | converting a GGUF of Recursive's Flint, what runs, reasoning and GLM-4.7 tool calls, expert residency, measurements against llama.cpp and the numerical checks |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [Device profiles](device-profiles.md) | per-GPU route profiles: the built-in RTX 3090/4090/5090/PRO 6000/TITAN RTX/Tesla V100 table, calibration at first start, `ninfer-calibrate` |
| [Configuration calculator](config-calculator.html) | which context, KV format and speculation fit an RTX 3090 with Qwen3.8-27B or Qwen3.6-35B-A3B, from measurements |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |
| [Release archives](release-archive-linux.md) ([Windows](release-archive-windows.md)) | the README that `scripts/package-release.sh` and `.ps1` put in a release archive |

## Model artifacts

The artifacts converted for this engine (Ternary Bonsai 2, the GSQ-RCO and Flash-Next conversions,
the Qwen3.8-27B fine-tunes and Qwen3.6-35B-A3B NVFP4) are in the README's
[artifact table](../README.md#artifacts); their cards live on Hugging Face. The official NInfer
artifacts load as well; their cards on Hugging Face give upstream's requirements and its
measurements on one RTX 5090. The `groupwise-int` artifacts run on every supported card, the
`nvfp4` ones need a `120a` build.

| Model | Weights | Download |
|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [neroued/Qwen3.6-27B-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | [neroued/Qwen3.6-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | [neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)
- [Rented GPU hosts](../scripts/pods/README.md)

## Maintainer references

The references under [`maintainer/`](maintainer/) record current architecture, model, artifact,
and maintenance contracts. These files are not additional user workflows or installed API
documentation. Several of them (engine architecture, artifact container, paged KV, ReplaySSM,
resource scheduling) are written in Chinese, as upstream wrote them.

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
| [Qwen4Exp model](maintainer/qwen4-exp-model.md) | Flash-Next mathematics, byte census, state sizes, the n-gram table's storage and the implementation status |
| [Pipeline stages](maintainer/pipeline-stages.md) | `--devices`: whole-layer stages over several GPUs, and `--tensor-parallel`: every layer divided over them; what each covers, verification and measurements |
| [Quality trades](maintainer/quality-trade-experiments.md) | the precision options and what each costs and buys |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | candidate selection, retention, materialization and Device/Host checkpoint policy |
| [Hybrid prefix cache](maintainer/hybrid-prefix-cache-spec.md) | the default prefix cache of `ninfer-serve`: block tree, sparse state snapshots, tap placement, Host slab tier, eviction and automatic configuration |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Cutting a release](maintainer/release-process.md) | the release-archive flow, the packagers and their checks |
| [Launcher profiles](maintainer/launcher-profiles.md) | `run` and `download-model`, the serving profile per model, and the reasons behind each flag |

[CONTRIBUTORS.md](../CONTRIBUTORS.md) credits the work this repository carries from its base and
other forks. The [conversion guide](weight-conversion.md) is the entry point for making an artifact.
Exact config fields, parameter expansion and native supported domains are maintained by the code
linked from these references.
