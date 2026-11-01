from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from tools.convert.sources.compressed_tensors import matrix_source
from tools.convert.sources.safetensors import SafetensorsSource


_CONFIG = {"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}


def _decode_word(word):
    """Scalar E4M3FN definition, independent of Torch's FP8 conversion."""
    exponent, mantissa = (word >> 3) & 15, word & 7
    if exponent == 15 and mantissa == 7:
        return math.nan
    magnitude = (mantissa * 2.0**-9 if exponent == 0 else
                 (1 + mantissa / 8) * 2.0**(exponent - 7))
    return -magnitude if word & 128 else magnitude


def _oracle(words, scales):
    table = torch.tensor([_decode_word(i) for i in range(256)], dtype=torch.float64)
    n, k = words.shape
    # Expand independently along matrix axes, not the source reader's flat index map.
    multipliers = scales.double().repeat_interleave(128, 0).repeat_interleave(128, 1)[:n, :k]
    return (table[words.long()] * multipliers).float()


def _checkpoint(path, words, scales, config=_CONFIG):
    (path / "config.json").write_text(json.dumps(config))
    save_file({"proj.weight": words.view(torch.float8_e4m3fn),
               "proj.weight_scale_inv": scales}, path / "model.safetensors")
    return SafetensorsSource(path)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_block_fp8_partial_reads_match_independent_oracle(tmp_path, dtype):
    finite = torch.tensor([i for i in range(256) if (i & 127) != 127], dtype=torch.uint8)
    words = finite[torch.arange(129 * 259) % len(finite)].reshape(129, 259)
    scales = torch.tensor([[1.00390625, 0.125, 0.0], [0.002731, 2.0, 0.007813]], dtype=dtype)
    expected = _oracle(words, scales).flatten()
    cuts = [0, 1, 127, 129, 258, 260, 127 * 259 + 126, 128 * 259 + 129, words.numel()]
    with _checkpoint(tmp_path, words, scales) as store:
        source = matrix_source(store, "proj.weight", tuple(words.shape))
        assert source.values(259, 259).numel() == 0
        assert store.bytes_read == 0
        for begin, end in zip(cuts, cuts[1:]):
            before = store.bytes_read
            result = source.values(begin, end)
            assert result.dtype == torch.float32
            assert torch.equal(result.view(torch.int32), expected[begin:end].view(torch.int32))
            block_rows = (end - 1) // 259 // 128 - begin // 259 // 128 + 1
            assert store.bytes_read - before == end - begin + block_rows * 3 * scales.element_size()
        assert torch.equal(source.rows(127, 129), expected.reshape(129, 259)[127:])
        with pytest.raises(ValueError, match="exceeds"):
            source.values(-1, 2)
        with pytest.raises(ValueError, match="encoded rows"):
            source.read_encoded(0, 1)
        with pytest.raises(ValueError, match="decoded values"):
            matrix_source(store, "proj.weight", tuple(words.shape), "fp8_e4m3fn_row_bf16").values()


@pytest.mark.parametrize("fault,message", [
    ("config", "128x128"), ("weight_shape", "expected F8_E4M3"),
    ("scale_shape", "expected BF16/F32"), ("scale_dtype", "expected BF16/F32"),
    ("positive_nan", "codes must be finite"), ("negative_nan", "codes must be finite"),
    ("negative_scale", "finite and nonnegative"), ("infinite_scale", "finite and nonnegative"),
    ("nan_scale", "finite and nonnegative"), ("overflow", "overflow FP32"),
])
def test_block_fp8_refuses_invalid_representations(tmp_path, fault, message):
    words = torch.full((2, 128), 0x7E, dtype=torch.uint8)
    scales = torch.ones(1, 1)
    config = _CONFIG
    if fault == "config":
        config = {"quantization_config": {"quant_method": "fp8", "weight_block_size": [64, 128]}}
    elif fault == "weight_shape":
        words = words[:, :64].contiguous()
    elif fault == "scale_shape":
        scales = torch.ones(1, 2)
    elif fault == "scale_dtype":
        scales = scales.half()
    elif fault.endswith("nan"):
        words[0, 0] = 0x7F if fault == "positive_nan" else 0xFF
    elif fault == "negative_scale":
        scales.fill_(-1)
    elif fault == "infinite_scale":
        scales.fill_(math.inf)
    elif fault == "nan_scale":
        scales.fill_(math.nan)
    elif fault == "overflow":
        scales.fill_(torch.finfo(torch.float32).max)
    with _checkpoint(tmp_path, words, scales, config) as store:
        with pytest.raises(ValueError, match=message):
            matrix_source(store, "proj.weight", (2, 128)).values()


@pytest.mark.parametrize("family", ["qwen3_5", "qwen4_exp", "qwen4_exp_mtp"])
def test_expert_binding_resolves_split_and_fused_selected_sources(tmp_path, family):
    from copy import deepcopy

    from .test_qwen3_5 import _checkpoint as model_checkpoint, _config
    from .test_qwen4_exp import _tiny_config
    from tools.convert import qwen3_5, qwen4_exp

    config = _tiny_config() if family.startswith("qwen4_exp") else _config(moe=True)
    build = qwen4_exp.build_model if family.startswith("qwen4_exp") else qwen3_5.build_model
    text = config["text_config"]
    h, ir, e = text["hidden_size"], text["moe_intermediate_size"], text["num_experts"]
    mtp = family.endswith("_mtp")
    if mtp:
        e += 2
        text.update(mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False,
                    mtp={"num_hidden_layers": 1, "hybrid": True, "layer_types": ["full_attention"],
                         "rope_theta": 10000000, "num_experts": e})
    split_config = deepcopy(config)
    split_config.update(_CONFIG)
    bank = torch.arange(e * 2 * ir * h).bfloat16().reshape(e, 2 * ir, h)
    down = torch.arange(e * h * ir).bfloat16().reshape(e, h, ir)
    prefix = "mtp.layers.0.mlp.experts" if mtp else "model.language_model.layers.0.mlp.experts"
    logical_prefix = "mtp/layer/moe/experts" if mtp else "text/layers/0/moe/experts"
    fused = {prefix + ".gate_up_proj": bank, prefix + ".down_proj": down}
    split, expected = {}, {}
    for expert in range(e):
        for role, shape in (("gate", (ir, h)), ("up", (ir, h)), ("down", (h, ir))):
            name = f"{prefix}.{expert}.{role}_proj"
            words = torch.full(shape, 0x38 + expert, dtype=torch.uint8)
            scales = torch.tensor([[0.125 * (1 + len(split))]], dtype=torch.bfloat16)
            split[name + ".weight"] = words.view(torch.float8_e4m3fn)
            split[name + ".weight_scale_inv"] = scales
            expected[expert, role] = _oracle(words, scales)
    with model_checkpoint(tmp_path / "split", split_config, split) as split_store, \
            model_checkpoint(tmp_path / "fused", config, fused) as fused_store:
        for base in (split_store, fused_store):
            model = build(base, components=("text", "mtp") if mtp else ("text",))
            for expert in range(e):
                for role in ("gate", "up", "down"):
                    name = f"{logical_prefix}/{expert}/{role}"
                    shape = model.parameters[name].shape
                    for selected in (split_store, fused_store):
                        actual = model.source(name, selected).values().reshape(shape)
                        want = (expected[expert, role] if selected is split_store else
                                down[expert] if role == "down" else
                                bank[expert, :ir] if role == "gate" else bank[expert, ir:])
                        assert torch.equal(actual, want)
                    assert torch.equal(model.parameters[name].source.values().reshape(shape),
                                       expected[expert, role] if base is split_store else
                                       down[expert] if role == "down" else
                                       bank[expert, :ir] if role == "gate" else bank[expert, ir:])
        split_store.config["text_config"]["moe_intermediate_size"] += 1
        with pytest.raises(ValueError, match="moe_intermediate_size differs"):
            model.source(f"{logical_prefix}/0/down", split_store)


def qualify_real_source(path: Path):
    results = []
    with SafetensorsSource(path) as store:
        for name in sorted(store.weight_map):
            if not name.endswith(".weight"):
                continue
            info = store.describe(name)
            words = store.read_flat(name).view(torch.uint8).reshape(info.shape)
            scales_name = name.removesuffix(".weight") + ".weight_scale_inv"
            scales_info = store.describe(scales_name)
            scales = store.read_flat(scales_name).reshape(scales_info.shape)
            expected = _oracle(words, scales).flatten()
            source = matrix_source(store, name, info.shape)
            for chunk in (65537, 131071):
                for begin in range(0, words.numel(), chunk):
                    end = min(begin + chunk, words.numel())
                    actual = source.values(begin, end)
                    assert torch.equal(actual.view(torch.int32), expected[begin:end].view(torch.int32)), name
            results.append({"tensor": name, "shape": info.shape, "scale_dtype": scales_info.dtype,
                            "values": words.numel(), "oracle": "FP64 codebook and block product; FP32 RNE",
                            "exact": True})
    print(json.dumps({"block_fp8_source_qualification": results}), flush=True)


if __name__ == "__main__":
    import sys

    qualify_real_source(Path(sys.argv[1]))
