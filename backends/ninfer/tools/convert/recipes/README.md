# Fork Qwen3.8-27B conversion profiles

These Python recipes use the v3 logical conversion API, without restoring target
inventories or changing runtime identities. Sources are explicit local paths;
conversion does not download checkpoints or calibration data.

## Full and QAT

```text
python -m tools.convert --model BF16_BASE --source quantized=NVFP4_SOURCE --source calibration=CALIBRATION.json --recipe tools/convert/recipes/qwen3_8_27b_nvfp4full.py --out OUTPUT.ninfer
python -m tools.convert --model BF16_BASE --source quantized=QAT_SOURCE --recipe tools/convert/recipes/qwen3_8_27b_nvfp4qat.py --out OUTPUT.ninfer
```

- **full** preserves the saved fork converter: source NVFP4 MLP parents on layers
  0–55; local NVFP4 attention/GDN and layers 56–63 MLP; six early attention-input,
  two attention-output and one GDN-output BF16 parent exceptions; Q8 vocabulary.
  Its calibration JSON has exactly 135 `measured_sites` entries, each containing
  `input_scale_divisor`, using the saved converter's site names. The custom encoder
  uses one global FP32 divisor per complete parent, E4M3FN block scales, and E2M1
  round-to-nearest-even codes, independent of streaming chunk size.
- **qat** imports all Text attention/GDN/MLP NVFP4 packed words, block scales,
  global weight divisors and activation divisors. Quantized GDN controls are
  decoded and cast to BF16; Q8 vocabulary and other direct tensors come from the
  BF16 base. It is **not** an alias of the official mixed-FP8 recipe.

Optional components retain the saved converter's allocation. Add
`--components text,vision,mtp,dflash2 --source dflash2=BF16_DFLASH2` as needed.
The saved converter uses a W8 DFlash2 module. To select the shipped fork's
weight-only NVFP4 module, also pass:

```text
--override tools/convert/recipes/dflash2_nvfp4.py
```

That override has 31 NVFP4 parents: each of five layers' QKV, output, gate/up,
down and two convolution-control parents, plus the selector hidden projection.
Feature projection stays W8; norms, convolution bases and codebooks stay BF16.
Context key/value bindings share their corresponding projection weights. All
module uses remain A16Only, without fabricated activation calibration.

The NVIDIA ModelOpt all-64-layer MLP experiment and its tolerance of 16 retired
calibration entries were uncommitted and explicitly reverted before the saved
baseline. These recipes do not silently revive that experiment.

## Replace only the module with W8

```text
python -m tools.convert.qwen3_8_27b.splice_w8_module SOURCE.v3.ninfer --dflash2-model BF16_DFLASH2 --out OUTPUT.v3.ninfer --device cpu
```

This reads the resident v3 component configuration, encodes only the companion
module, and copies every non-module payload byte-exact, retaining its descriptor,
binding, Use and metadata. No base checkpoint or base-weight regeneration is
required. The output must be new; multi-file artifacts are supported. Physical
container offsets and framing change, so the entire files are not byte-identical.

## Verification boundary

Focused CPU synthetic tests cover real-shape logical recipe preparation, exact
source-word/divisor imports, BF16 control decode, E2M1 ties and signed zero,
parent-global divisor/chunk invariance, the 31-parent module allocation, and
multi-file splice preservation. They do not establish full-checkpoint conversion,
GPU numerical parity, inference quality, or performance. Use an existing managed
Python 3.11 environment with Torch; no new dependency installation is required by
the recipes themselves.
