from __future__ import annotations

import struct
import sys

import numpy as np
import pytest
import torch

from tools.artifact.formats import GGUF_FORMATS, GGUF_FORMATS_BY_TYPE
from tools.artifact.layouts import encoded_size, gguf_blocks_geometry
from tools.convert import gguf_blocks
from tools.convert.sources.gguf import TYPE_BF16, GGUFFile, write_gguf

TYPE_Q8_0 = 8

# ggml-common.h: block values and bytes of every type the importer keeps.
GGML_BLOCKS = {
    2: (32, 18),
    6: (32, 22),
    8: (32, 34),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 50),
    20: (32, 18),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    29: (256, 56),
    42: (64, 18),
}


def _q8_0_file(tmp_path, rows: int, columns: int, seed: int):
    generator = np.random.default_rng(seed)
    blocks = generator.integers(0, 256, size=(rows, columns // 32, 34), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 32)).astype(np.float16)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(rows, columns // 32, 2)
    path = tmp_path / "blocks.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [("w", (rows, columns), TYPE_Q8_0, blocks.tobytes())],
    )
    return path, blocks


def _small_block_file(tmp_path, type_id: int, rows: int = 4):
    elements, size = GGML_BLOCKS[type_id]
    columns = 128
    bits, zero = {42: (2, 1), 2: (4, 8), 6: (5, 16)}[type_id]
    blocks = np.empty((rows, columns // elements, size), dtype=np.uint8)
    expected = np.empty((rows, columns), dtype=np.float32)
    for row in range(rows):
        for block in range(columns // elements):
            scale = (0.5, -0.25, 0.0, 2**-24)[(row + block) % 4]
            codes = [(5 * i + 7 * (i // 16) + row * 3 + block * 7) % (1 << bits)
                     for i in range(elements)]
            if type_id == 42:
                packed = bytes(sum(codes[i + j] << (2 * j) for j in range(4))
                               for i in range(0, elements, 4))
            else:
                packed = bytes((codes[i] & 15) | ((codes[i + 16] & 15) << 4)
                               for i in range(16))
                if type_id == 6:
                    high = sum(((code >> 4) & 1) << i for i, code in enumerate(codes))
                    packed = high.to_bytes(4, "little") + packed
            blocks[row, block] = np.frombuffer(struct.pack("<e", scale) + packed, dtype=np.uint8)
            expected[row, block * elements : (block + 1) * elements] = [
                (code - zero) * scale for code in codes
            ]
    path = tmp_path / f"blocks-{type_id}.gguf"
    write_gguf(path, {"general.architecture": "qwen35"},
               [("w", (rows, columns), type_id, blocks.tobytes())])
    return path, blocks.reshape(rows, -1), expected


def test_formats_are_ggml_blocks():
    assert {spec.ggml_type for spec in GGUF_FORMATS.values()} == set(GGML_BLOCKS)
    for type_id, (elements, size) in GGML_BLOCKS.items():
        spec = GGUF_FORMATS_BY_TYPE[type_id]
        assert (spec.block_elements, spec.block_bytes) == (elements, size)
        geometry = gguf_blocks_geometry(spec, (3, 4 * elements))
        assert geometry.row_bytes == 4 * size
        assert encoded_size("gguf_blocks_v1", spec, (3, 4 * elements)) == 12 * size
    with pytest.raises(ValueError):
        gguf_blocks_geometry("gguf_q4_k", (2, 128))


def test_rows_are_copied_byte_for_byte_through_the_row_map(tmp_path):
    path, blocks = _q8_0_file(tmp_path, 12, 64, 1)
    order = np.array([7, 2, 11, 2], dtype=np.int64)
    with GGUFFile(path) as gguf:
        source = gguf_blocks.block_source(gguf, "w", (4, 64), lambda b, e: order[b:e])
        words = source.read_encoded(0, 4)
        assert words.format == "gguf_q8_0"
        assert torch.equal(words.codes, torch.from_numpy(blocks.reshape(12, -1)[order]))
        assert torch.equal(
            source.read_encoded(1, 3).codes, torch.from_numpy(blocks.reshape(12, -1)[order[1:3]])
        )


@pytest.mark.parametrize("type_id", (42, 2, 6), ids=("q2_0", "q4_0", "q5_0"))
def test_small_blocks_preserve_rows_and_values(tmp_path, monkeypatch, type_id):
    if type_id == 42:
        # Q2_0 must decode even when gguf-py is absent or lacks this implementation.
        monkeypatch.setitem(sys.modules, "gguf", None)
    else:
        pytest.importorskip("gguf")
    path, blocks, values = _small_block_file(tmp_path, type_id)
    order = np.array([3, 0, 2, 0], dtype=np.int64)
    with GGUFFile(path) as gguf:
        source = gguf_blocks.block_source(gguf, "w", (4, 128), lambda b, e: order[b:e])
        encoded = source.read_encoded(1, 4)
        assert encoded.format == GGUF_FORMATS_BY_TYPE[type_id].name
        assert torch.equal(encoded.codes, torch.from_numpy(blocks[order[1:4]]))
        assert torch.equal(source.rows(1, 4), torch.from_numpy(values[order[1:4]]))
        with pytest.raises(ValueError, match="row range"):
            gguf.read_blocks("w", 0, 5)


def test_q2_0_gates_decode_before_grouped_bf16_cast(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "gguf", None)
    path, _, values = _small_block_file(tmp_path, 42, rows=48)
    grouped = values.reshape(3, 16, 128).transpose(1, 0, 2).reshape(48, 128)
    with GGUFFile(path) as gguf:
        source = gguf_blocks._gate_projection(gguf, "w")
        assert torch.equal(source.rows(0, 48), torch.from_numpy(grouped.copy()).to(torch.bfloat16))


def test_row_values_are_ggml_dequantization(tmp_path):
    pytest.importorskip("gguf")
    path, blocks = _q8_0_file(tmp_path, 6, 64, 2)
    with GGUFFile(path) as gguf:
        source = gguf_blocks.block_source(gguf, "w", (6, 64), lambda b, e: np.arange(b, e))
        scales = blocks[3:5, :, :2].copy().view(np.float16).astype(np.float32)
        codes = blocks[3:5, :, 2:].view(np.int8).astype(np.float32)
        assert torch.equal(source.rows(3, 5), torch.from_numpy((codes * scales).reshape(2, 64)))


def test_output_projection_columns_read_the_grouped_value_heads():
    columns = gguf_blocks.tiled_input_columns()
    assert sorted(columns.tolist()) == list(range(6144))
    for tiled_head in (0, 1, 15, 16, 31, 47):
        key_head, repeat = tiled_head % 16, tiled_head // 16
        assert columns[tiled_head * 128 + 5] == (key_head * 3 + repeat) * 128 + 5


def test_quantized_gates_decode_before_grouped_bf16_cast(tmp_path):
    pytest.importorskip("gguf")
    path, blocks = _q8_0_file(tmp_path, 48, 64, 3)
    scales = blocks[:, :, :2].copy().view(np.float16).astype(np.float32)
    codes = blocks[:, :, 2:].view(np.int8).astype(np.float32)
    tiled = (codes * scales).reshape(48, 64)
    # GGUF tiles by repeat then key; execution groups the three repeats of each key.
    grouped = tiled.reshape(3, 16, 64).transpose(1, 0, 2).reshape(48, 64)
    with GGUFFile(path) as gguf:
        source = gguf_blocks._gate_projection(gguf, "w")
        assert source.shape == (48, 64)
        assert torch.equal(source.rows(0, 48), torch.from_numpy(grouped.copy()).to(torch.bfloat16))


def test_bf16_gates_preserve_words_through_grouping(tmp_path):
    words = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) + 0x3800
    path = tmp_path / "gates.gguf"
    write_gguf(path, {"general.architecture": "qwen35"},
               [("w", (48, 64), TYPE_BF16, words.tobytes())])
    expected = words.reshape(3, 16, 64).transpose(1, 0, 2).reshape(48, 64)
    with GGUFFile(path) as gguf:
        source = gguf_blocks._gate_projection(gguf, "w")
        actual = source.rows(0, 48).view(torch.int16).numpy().view(np.uint16)
        assert np.array_equal(actual, expected)
