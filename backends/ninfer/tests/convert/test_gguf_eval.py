from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.convert.gguf_blocks import tiled_input_columns
from tools.convert.quantization.groupwise import search_quantize_matrix
from tools.eval import gguf_eval


def _dequantize_q8_0(data: bytes, rows: int, k: int) -> np.ndarray:
    blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, k // 32, 34)
    scales = blocks[:, :, :2].copy().view(np.float16).astype(np.float32)
    codes = blocks[:, :, 2:].copy().view(np.int8).astype(np.float32)
    return (codes * scales).reshape(rows, k)


def test_v_head_order_matches_the_gguf_import_mapping():
    # tiled_input_columns()[c] is the grouped (HF) element read by GGUF column c.
    order = gguf_eval.v_head_order().numpy()
    assert np.array_equal(order, tiled_input_columns())


def test_q8_0_export_reproduces_signed_grouped_values_in_gguf_order():
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn((4, 384), generator=generator) * 0.03
    weight[1, 7] = 0.5
    encoded = search_quantize_matrix(weight, "q4_g64_fp16", negative_scales=True, device="cpu")
    assert bool((encoded.scales < 0).any())
    represented = (encoded.codes.float() * encoded.scales.float().unsqueeze(-1)).reshape(4, 384)

    # Permute whole 128-column heads the way ssm_out's tiled value heads are stored.
    heads = torch.tensor([1, 2, 0])
    column_order = (heads[:, None] * 128 + torch.arange(128)).reshape(-1)
    target = gguf_eval.Target("t", "t", (), row_order=torch.tensor([2, 0, 3, 1]),
                              column_order=column_order)
    codes, scales = gguf_eval.q8_0_blocks(encoded.codes, encoded.scales, 384)
    data = gguf_eval.q8_0_bytes(target, codes, scales)
    exported = _dequantize_q8_0(data, 4, 384)
    expected = represented[target.row_order][:, column_order].numpy()
    assert np.array_equal(exported, expected)


def test_template_offsets_locate_q8_0_tensors(tmp_path):
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "template.gguf"
    blocks = np.arange(3 * 2 * 34, dtype=np.uint8).reshape(3, 68)
    writer = gguf.GGUFWriter(str(path), "qwen35")
    writer.add_tensor("norm.weight", np.ones(5, dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down.weight", blocks,
                      raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    offsets = gguf_eval.template_offsets(path)
    assert set(offsets) == {"blk.0.ffn_down.weight"}
    offset, shape = offsets["blk.0.ffn_down.weight"]
    assert shape == (3, 64)
    assert path.read_bytes()[offset : offset + blocks.size] == blocks.tobytes()
