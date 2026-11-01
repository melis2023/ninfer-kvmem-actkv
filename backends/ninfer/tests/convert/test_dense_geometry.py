"""Dense Qwen3.5 shape contracts; no model-weight downloads are needed."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.convert.dense_geometry import DenseGeometry
from tools.convert.gguf_blocks import expected_tensors, gdn_control_source, tiled_input_columns, validate
from tools.convert.sources.gguf import GGUFFile, write_gguf

# Published Qwen/Qwen3.5-{size}/config.json text tower dimensions.
SIZES = [
    ("0.8B", 1024, 3584, 24, 8, 2, 16, True),
    ("2B", 2048, 6144, 24, 8, 2, 16, True),
    ("4B", 2560, 9216, 32, 16, 4, 32, True),
    ("9B", 4096, 12288, 32, 16, 4, 32, False),
    ("27B", 5120, 17408, 64, 24, 4, 48, False),
]


def dense_config(size):
    _, hidden, intermediate, layers, heads, kv, values, tied = size
    config = dict(hidden_size=hidden, intermediate_size=intermediate, vocab_size=248320,
                  num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv,
                  head_dim=256, linear_num_key_heads=16, linear_num_value_heads=values,
                  linear_key_head_dim=128, linear_value_head_dim=128,
                  linear_conv_kernel_dim=4, tie_word_embeddings=tied,
                  layer_types=["linear_attention"] * 3 + ["full_attention"])
    config["layer_types"] *= layers // 4
    return config


@pytest.fixture(params=SIZES, ids=[x[0] for x in SIZES])
def geometry(request):
    return DenseGeometry.from_config(dense_config(request.param))


@pytest.mark.parametrize("change", [
    {"linear_num_key_heads": 8}, {"linear_num_value_heads": 64},
    {"head_dim": 128}, {"linear_conv_kernel_dim": 3}, {"num_experts": 8},
])
def test_reject_unimplemented_runtime_geometry(change):
    config = dense_config(SIZES[2]) | change
    with pytest.raises(ValueError):
        DenseGeometry.from_config(config)


def test_exporter_order_round_trip(geometry):
    g = geometry
    # GGUF repeats all key heads before the next value-head repeat. Runtime
    # groups all repeats of a key head; construct the expected order explicitly.
    repeats = g.value_heads // g.key_heads
    grouped = [repeat * g.key_heads + key for key in range(g.key_heads)
               for repeat in range(repeats)]
    tiled = np.arange(g.value_dim).reshape(g.value_heads, 128)
    expected = tiled[grouped].reshape(-1)
    np.testing.assert_array_equal(g.untile(tiled.reshape(-1), 128), expected)
    np.testing.assert_array_equal(g.untiled_rows(4096)(0, g.value_dim), expected + 4096)
    np.testing.assert_array_equal(expected[tiled_input_columns(g)], np.arange(g.value_dim))
    for gate in (False, True):
        expected_attention = [head * 512 + int(gate) * 256 + lane
                              for head in range(g.attention_heads) for lane in range(256)]
        np.testing.assert_array_equal(g.attention_rows(gate)(0, g.query_rows), expected_attention)


@pytest.mark.parametrize("mtp", [False, True])
def test_dense_tensor_contract_and_rejection(geometry, mtp):
    g = geometry
    shapes = expected_tensors(mtp, g)
    assert shapes["token_embd.weight"][0] == (248320, g.hidden)
    assert ("output.weight" in shapes) == (not g.tied)
    assert shapes["blk.0.attn_qkv.weight"][0] == (4096 + g.value_dim, g.hidden)
    assert shapes["blk.3.attn_q.weight"][0] == (g.attention_heads * 512, g.hidden)
    if mtp:
        assert shapes[f"blk.{g.layers}.nextn.eh_proj.weight"][0] == (g.hidden, 2 * g.hidden)
    tensors = {}
    for name, (shape, kind) in shapes.items():
        tensors[name] = SimpleNamespace(shape=shape, type_id=12 if kind == "blocks" else 0,
                                       type_name="Q4_K" if kind == "blocks" else "F32", offset=0, nbytes=1)
    source = SimpleNamespace(path="synthetic.gguf", tensors=tensors, data_bytes_available=1,
                             kv={**g.expected_header(), "qwen35.block_count": g.layers + int(mtp)})
    validate(source, g)
    source.kv["qwen35.ssm.time_step_rank"] += 1
    with pytest.raises(ValueError, match="time_step_rank"):
        validate(source, g)
    source.kv["qwen35.ssm.time_step_rank"] -= 1
    tensors["blk.0.attn_qkv.weight"].shape = (g.channels + 128, g.hidden)
    with pytest.raises(ValueError, match="attn_qkv"):
        validate(source, g)


def test_f32_control_cast_and_head_order(geometry, tmp_path):
    g = geometry
    # Nonrepresentable values exercise BF16 rounding, not just integer row IDs.
    values = (np.arange(g.value_heads * 128, dtype=np.float32) * 0.00137).reshape(g.value_heads, 128)
    path = tmp_path / "controls.gguf"
    write_gguf(path, {"general.architecture": "qwen35"}, [("w", values.shape, 0, values.tobytes())])
    order = [repeat * 16 + key for key in range(16) for repeat in range(g.value_heads // 16)]
    with GGUFFile(path) as source:
        actual = gdn_control_source(source, "w", g).rows(0, g.value_heads)
    assert torch.equal(actual, torch.from_numpy(values[order]).to(torch.bfloat16))
