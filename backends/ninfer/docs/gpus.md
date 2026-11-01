# Supported GPUs

Which NVIDIA GPUs NInfer runs on: the architectures and their builds, which cards have been
measured and with which models, and what fits in a given amount of GPU memory. The rates measured on
each card are on the [performance](performance.md) page.

## Architectures

A build targets one architecture (`CMAKE_CUDA_ARCHITECTURES`), or several for a layer pipeline across
cards of different architectures ([several architectures](build-linux.md#cards-of-several-architectures-in-one-pipeline));
configure refuses any other target.
The Docker pipeline packages two multiarch images: `cuda12.9` contains `70;75;80;86;89`, and
`cuda13.4` contains `75;80;86;89;120a`. Each uses one executable for all included architectures,
including a pipeline across different GPU families. See [Docker](../README.md#docker) for tags
and verification status.

| Architecture | Compute capability | Build | CUDA | Docker tags | Status |
|---|---|---|---|---|---|
| Blackwell: GeForce RTX 50, RTX PRO Blackwell | 12.0 | `120a` | 13.1 or newer | `cuda13.4` | measured; built-in device profiles |
| Ada Lovelace | 8.9 | `89`, or the `86` build | 12.8 or newer | both, native `89` | measured; built-in device profiles |
| Ampere | 8.6 | `86` (the default) | 12.8 or newer | both | measured (the reference card); built-in device profiles |
| Ampere, A100 class | 8.0 | `80` | 12.8 or newer | both | built-in device profiles; not benchmarked |
| Turing | 7.5 | `75` | 12.8 or newer | both | measured; built-in device profiles |
| Volta | 7.0 | `70` | 12.8 or 12.9 | `cuda12.9` | measured; built-in device profiles |
| Hopper; data-center Blackwell; GB10 | 9.0; 10.x; 12.1 | none | | | no build |
| Pascal and older | 6.x and older | none | | | no build |

- **Device profiles.** A profile picks the kernel route of every operation and width. The binary
  carries a profile measured on every card the Profile column of the [card tables](#cards) marks; a
  built-in profile applies to the same GPU name, compute capability and SM count. Any other card of
  these architectures calibrates once at first start and stores the result
  ([device profiles](device-profiles.md)).
- **Architecture limits.** NVFP4 expert weights (the Qwen3.6-35B-A3B NVFP4 artifact) need a `120a`
  build. The Turing and Volta builds replace the instructions their GPUs lack and shape kernels to
  their shared memory ([Turing](build-linux.md#turing-sm_75),
  [Tesla V100](build-linux.md#tesla-v100-sm_70)).

## What fits by memory

Measured, one request; context in tokens is the largest `--max-context` the server starts with
(details on the [performance](performance.md#largest-context) page).

| GPU memory | Runs | Measured on |
|---|---|---|
| 8 GB | Ternary Bonsai 2 27B (24,576 tokens with `rk4v4`, no MTP); Qwen3.8-Flash-Next with its experts in host memory or on disk | RTX 3060 Ti |
| 11 GB | Ternary Bonsai 2 27B (200,704 tokens with `rk4v4`; 143,360 with MTP) | RTX 2080 Ti |
| 12 GB | Ternary Bonsai 2 27B (266,240 tokens); Qwen3.8-27B GSQ-RCO IQ3_S (8,192 tokens, no MTP) | RTX 3060 |
| 16 GB | Gemma 4 26B-A4B Q4_0 (61,440 tokens); Qwen3.8-27B IQ3_S (241,664 tokens; 122,880 with `rk4v4-e8` and 98,304 with MTP on a card that also drives a desktop); the official Qwen3.8-27B with `--embedding-q4 --lm-head-q4` | Tesla V100-SXM2-16GB, RTX 5070 Ti ([field report](performance/field-reports.md#rtx-5070-ti-windows-11)) |
| 24 GB | the official Qwen3.8-27B (20 GiB) and its MXFP8 tune (17 GiB); Qwen3.6-35B-A3B `groupwise-int` (19.6 GiB of weights) on an RTX 3090, over two cards on the TITAN RTX; Flash-Next with every expert on the GPU over two cards | RTX 3090, RTX 4090, TITAN RTX |
| 32 GB | Qwen3.6-35B-A3B NVFP4 (Blackwell only) | RTX 5090 |
| 48 GB | Flash-Next Q2_0 with every expert on one card | L40S, RTX PRO 6000 |

Gemma 4 does not fit 12 GB, and IQ3_S does not fit 11 GB: their weights alone are larger.

## Cards

Desktop, workstation and data-center cards with 8 GB or more; laptop GPUs are not listed.
✅ ran the models listed: measured by the project, or reported by a user in a
[field report](performance/field-reports.md) · ⬜ no result yet. **Profile:** ✓ the binary carries a
[device profile](device-profiles.md) measured on this card (or on the variant named); an empty cell
means the card calibrates once at its first start.

### Blackwell (12.0, build `120a`)

| | GPU | Memory | Line | Profile | Models run |
|:-:|---|---:|---|---|---|
| ✅ | RTX 5090 | 32 GB | GeForce | ✓ | Bonsai 2, Qwen3.8-27B, IQ3_S, Flash-Next, 35B-A3B NVFP4, Flint |
| ⬜ | RTX 5090 D | 32 GB | GeForce | | |
| ⬜ | RTX 5090 D V2 | 24 GB | GeForce | ✓ | |
| ⬜ | RTX 5080 | 16 GB | GeForce | ✓ | |
| ✅ | RTX 5070 Ti | 16 GB | GeForce | ✓ | IQ3_S, Swift 1.5 IQ3_S ([field report](performance/field-reports.md#rtx-5070-ti-windows-11)) |
| ⬜ | RTX 5070 | 12 GB | GeForce | ✓ | |
| ⬜ | RTX 5060 Ti | 16 / 8 GB | GeForce | ✓ | |
| ⬜ | RTX 5060 | 8 GB | GeForce | | |
| ⬜ | RTX 5050 | 8 GB | GeForce | | |
| ✅ | RTX PRO 6000 Blackwell Workstation Edition | 96 GB | Workstation | ✓ | Bonsai 2, Qwen3.8-27B, Flash-Next |
| ⬜ | RTX PRO 6000 Blackwell Max-Q | 96 GB | Workstation | ✓ | |
| ⬜ | RTX PRO 5000 Blackwell | 72 / 48 GB | Workstation | 48 GB | |
| ⬜ | RTX PRO 4500 Blackwell | 32 GB | Workstation | ✓ | |
| ⬜ | RTX PRO 4000 Blackwell | 24 GB | Workstation | ✓ | |
| ⬜ | RTX PRO 4000 Blackwell SFF | 24 GB | Workstation | | |
| ⬜ | RTX PRO 2000 Blackwell | 16 GB | Workstation | | |
| ⬜ | RTX PRO 6000 Blackwell Server Edition | 96 GB | Data center | ✓ | |
| ⬜ | RTX PRO 4500 Blackwell Server Edition | 32 GB | Data center | ✓ | |

### Ada Lovelace (8.9, the image's `86` build or `89`)

| | GPU | Memory | Line | Profile | Models run |
|:-:|---|---:|---|---|---|
| ✅ | RTX 4090 | 24 GB | GeForce | ✓ | Bonsai 2, Qwen3.8-27B, IQ3_S, Flash-Next, Flint (experts in host memory); Swift 1.5 IQ3_S ([field reports](performance/field-reports.md#rtx-4090-qwen-code-with-dflash2)) |
| ⬜ | RTX 4090 D | 24 GB | GeForce | | |
| ⬜ | RTX 4080 Super | 16 GB | GeForce | ✓ | |
| ✅ | RTX 4080 | 16 GB | GeForce | ✓ | IQ3_S ([field report](performance/field-reports.md#rtx-4080-and-rtx-4090-windows-11)) |
| ⬜ | RTX 4070 Ti Super | 16 GB | GeForce | ✓ | |
| ⬜ | RTX 4070 Ti | 12 GB | GeForce | ✓ | |
| ⬜ | RTX 4070 Super | 12 GB | GeForce | | |
| ⬜ | RTX 4070 | 12 GB | GeForce | ✓ | |
| ⬜ | RTX 4060 Ti | 16 / 8 GB | GeForce | ✓ | |
| ⬜ | RTX 4060 | 8 GB | GeForce | | |
| ⬜ | RTX 6000 Ada | 48 GB | Workstation | ✓ | |
| ⬜ | RTX 5880 Ada | 48 GB | Workstation | | |
| ⬜ | RTX 5000 Ada | 32 GB | Workstation | ✓ | |
| ⬜ | RTX 4500 Ada | 24 GB | Workstation | | |
| ⬜ | RTX 4000 Ada | 20 GB | Workstation | ✓ | |
| ⬜ | RTX 4000 SFF Ada | 20 GB | Workstation | | |
| ⬜ | RTX 2000 Ada | 16 GB | Workstation | | |
| ✅ | L40S | 48 GB | Data center | ✓ | Flash-Next |
| ⬜ | L40 | 48 GB | Data center | | |
| ⬜ | L20 | 48 GB | Data center | | |
| ⬜ | L4 | 24 GB | Data center | ✓ | |

### Ampere (8.6, build `86`)

| | GPU | Memory | Line | Profile | Models run |
|:-:|---|---:|---|---|---|
| ✅ | RTX 3090 Ti | 24 GB | GeForce | | Flash-Next over two cards |
| ✅ | RTX 3090 | 24 GB | GeForce | ✓ | Bonsai 2, Qwen3.8-27B, IQ3_S, 35B-A3B, Flash-Next, Gemma 4, Flint (experts in host memory, on disk; the reference card) |
| ⬜ | RTX 3080 Ti | 12 GB | GeForce | ✓ | |
| ⬜ | RTX 3080 | 12 / 10 GB | GeForce | 10 GB (68 SMs) | |
| ⬜ | RTX 3070 Ti | 8 GB | GeForce | | |
| ⬜ | RTX 3070 | 8 GB | GeForce | ✓ | |
| ✅ | RTX 3060 Ti | 8 GB | GeForce | ✓ | Bonsai 2, Flash-Next (experts in host memory, on disk) |
| ✅ | RTX 3060 | 12 GB | GeForce | ✓ | Bonsai 2, IQ3_S, Flash-Next (experts in host memory, on disk) |
| ⬜ | RTX 3060 | 8 GB | GeForce | the 12 GB card's (28 SMs) | |
| ⬜ | RTX 3050 | 8 GB | GeForce | | |
| ⬜ | RTX A6000 | 48 GB | Workstation | | |
| ⬜ | RTX A5500 | 24 GB | Workstation | | |
| ⬜ | RTX A5000 | 24 GB | Workstation | ✓ | |
| ⬜ | RTX A4500 | 20 GB | Workstation | | |
| ⬜ | RTX A4000 | 16 GB | Workstation | | |
| ⬜ | RTX A2000 | 12 GB | Workstation | | |
| ⬜ | RTX A1000 | 8 GB | Workstation | | |
| ✅ | A40 | 48 GB | Data center | ✓ | Flint |
| ⬜ | A10 | 24 GB | Data center | ✓ | |
| ⬜ | A10G | 24 GB | Data center | | |
| ⬜ | A16 | 16 GB per GPU | Data center | | |
| ⬜ | A2 | 16 GB | Data center | | |

### Ampere, A100 class (8.0, build `80`, not in the image)

| | GPU | Memory | Line | Profile | Models run |
|:-:|---|---:|---|---|---|
| ⬜ | A100 (PCIe, SXM4) | 80 / 40 GB | Data center | ✓ | |
| ⬜ | A800 | 80 / 40 GB | Data center | | |
| ⬜ | A30 | 24 GB | Data center | | |

### Turing (7.5, build `75`)

| | GPU | Memory | Line | Profile | Models run |
|:-:|---|---:|---|---|---|
| ✅ | TITAN RTX | 24 GB | GeForce | ✓ | Bonsai 2, Qwen3.8-27B, IQ3_S, MXFP8, Gemma 4, Flash-Next; 35B-A3B over two cards |
| ✅ | RTX 2080 Ti | 11 GB | GeForce | ✓ | Bonsai 2 |
| ⬜ | RTX 2080 Super | 8 GB | GeForce | | |
| ⬜ | RTX 2080 | 8 GB | GeForce | | |
| ⬜ | RTX 2070 Super | 8 GB | GeForce | ✓ | |
| ⬜ | RTX 2070 | 8 GB | GeForce | ✓ | |
| ⬜ | RTX 2060 | 12 GB | GeForce | | |
| ⬜ | RTX 2060 Super | 8 GB | GeForce | ✓ | |
| ✅ | Quadro RTX 8000 | 48 GB | Workstation | ✓ | Bonsai 2, Gemma 4, Flash-Next |
| ⬜ | Quadro RTX 6000 | 24 GB | Workstation | ✓ | |
| ⬜ | Quadro RTX 5000 | 16 GB | Workstation | | |
| ⬜ | Quadro RTX 4000 | 8 GB | Workstation | | |
| ⬜ | T1000 | 8 GB | Workstation | | no tensor cores; whether the `75` build runs on it is unknown |
| ⬜ | Tesla T4 | 16 GB | Data center | ✓ | |

### Volta (7.0, build `70` with CUDA 12.8 or 12.9, built from source)

| | GPU | Memory | Line | Profile | Models run |
|:-:|---|---:|---|---|---|
| ⬜ | TITAN V CEO Edition | 32 GB | GeForce | | |
| ⬜ | TITAN V | 12 GB | GeForce | | |
| ⬜ | Quadro GV100 | 32 GB | Workstation | | |
| ✅ | Tesla V100-SXM2 | 16 GB | Data center | ✓ | Bonsai 2, IQ3_S, Qwen3.8-27B (Q4 embedding and head), Flash-Next (experts in host memory, on disk), Gemma 4 |
| ⬜ | Tesla V100-SXM2 | 32 GB | Data center | ✓ | |
| ⬜ | Tesla V100 PCIe | 32 / 16 GB | Data center | | |
| ⬜ | Tesla V100S PCIe | 32 GB | Data center | | |

The RTX 3060 also ran the Windows test suite for a contributor
([Windows build](build-windows.md)).
