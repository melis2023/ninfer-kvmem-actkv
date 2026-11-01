from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.codecs.row_split import dequantize_row_split
from tools.artifact.layouts import row_split_geometry
from tools.artifact.reader import Artifact
from tools.convert.gguf_blocks import q2_native_source, q8_native_source
from tools.convert.methods import grouped_absmax, grouped_search, import_encoded
from tools.convert.model import Model, Parameter
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf import GGUFFile, write_gguf
from tools.convert.sources.logical import select_rows


@pytest.mark.parametrize("k", [64, 640, 2560])
def test_q2_import_preserves_selected_expert_codes_and_scale_bits(tmp_path, k):
    rng = np.random.default_rng(42)
    blocks = rng.integers(0, 256, (12, k // 64, 18), dtype=np.uint8)
    words = np.resize(np.array([0x3800, 0xBC00, 0x0001, 0x8000, 0x0000], dtype="<u2"),
                      (12, k // 64))
    blocks[..., :2] = words.view(np.uint8).reshape(12, k // 64, 2)
    path = tmp_path / "experts.gguf"
    write_gguf(path, {"general.architecture": "qwen4exp"},
               [("experts", (3, 4, k), 42, blocks.tobytes())])
    # Read one expert's rows out of order, including a duplicate. Chunking must retain that map.
    order = np.array([10, 8, 11, 8])
    output = tmp_path / "experts.ninfer"
    with GGUFFile(path) as gguf:
        source = q2_native_source(gguf, "experts", (4, k), lambda b, e: order[b:e])
        model = Model({"text": {"config": {}}})
        model.add(Parameter("expert", (4, k), source, inputs=("input",)))
        recipe = Recipe(model)
        recipe.assign("expert", format="q2_g64_fp16", method=import_encoded,
                      activation_policy="AllowA8")
        convert(model, recipe, output, device="cpu", rows_per_chunk=3)
    with Artifact(output) as artifact:
        obj = artifact.directory.bindings["expert"]["object"]
        payload = artifact.read_object(obj)
        geometry = row_split_geometry("q2_g64_fp16", (4, k))
        packed = np.frombuffer(payload[:geometry.base_bytes], dtype=np.uint8).reshape(4, -1)
        assert packed[:, :k // 4].tobytes() == blocks[order, :, 2:].tobytes()
        if k == 64:
            assert (packed[:, k // 4:] == 0x55).all()  # stored integer zero is offset code 1
        scales = np.frombuffer(payload[geometry.scale_offset:], dtype="<u2").reshape(4, -1)
        assert scales[:, :k // 64].tobytes() == blocks[order, :, :2].tobytes()
        if k == 64:
            assert (scales[:, 1:] == 0).all()
        selected = blocks[order]
        oracle = np.empty((4, k), dtype=np.float64)
        for r in range(4):
            for j in range(k):
                block = selected[r, j // 64]
                scale = float(block[:2].copy().view("<f2")[0])
                code = (int(block[2 + (j % 64) // 4]) >> (2 * (j % 4))) & 3
                oracle[r, j] = scale * (code - 1)
        assert torch.equal(dequantize_row_split(payload, "q2_g64_fp16", (4, k),
                                               dtype=torch.float64), torch.from_numpy(oracle))


@pytest.mark.parametrize("method", [grouped_absmax, grouped_search])
def test_q2_recipe_refuses_requantization(tmp_path, method):
    from tools.convert.sources.logical import array_source

    model = Model({"text": {"config": {}}})
    model.add(Parameter("expert", (2, 128), array_source(torch.ones(2, 128), "untrained")))
    recipe = Recipe(model)
    recipe.assign("expert", format="q2_g64_fp16", method=method)
    with pytest.raises(ValueError, match="import_encoded"):
        recipe.prepare(device="cpu")


def test_packed_q2_import_preserves_selected_rows_across_grouped_sources(tmp_path):
    k = 128
    blocks = np.arange(6 * 2 * 18, dtype=np.uint8).reshape(6, 2, 18)
    words = np.arange(12, dtype="<u2").reshape(6, 2) + 0x3800
    blocks[..., :2] = words.view(np.uint8).reshape(6, 2, 2)
    path, output = tmp_path / "group.gguf", tmp_path / "group.ninfer"
    write_gguf(path, {}, [("w", (6, k), 42, blocks.tobytes())])
    with GGUFFile(path) as gguf:
        whole = q2_native_source(gguf, "w", (6, k), lambda b, e: np.arange(b, e))
        model = Model({"text": {"config": {}}})
        for name, selection in (("gate", ((4, 6), (1, 2))), ("up", ((0, 2),))):
            selected = select_rows(whole, selection)
            model.add(Parameter(name, selected.shape, selected, inputs=("input",)))
        recipe = Recipe(model)
        recipe.assign(("gate", "up"), format="q2_g64_fp16", method=import_encoded)
        recipe.group(("gate", "up"))
        # One chunk crosses both selection spans and the parent source boundary.
        convert(model, recipe, output, device="cpu", rows_per_chunk=4)
    with Artifact(output) as artifact:
        from tools.artifact.schema import binding_parts

        gate = binding_parts(artifact.directory.bindings["gate"], artifact.by_id)[0]
        up = binding_parts(artifact.directory.bindings["up"], artifact.by_id)[0]
        assert gate[0] == up[0] and gate[1:] == (0, 3 * k) and up[1:] == (3 * k, 5 * k)
        payload = artifact.read_object(gate[0])
        geometry = row_split_geometry("q2_g64_fp16", (5, k))
        order = [4, 5, 1, 0, 1]
        assert payload[:geometry.base_bytes] == blocks[order, :, 2:].tobytes()
        assert payload[geometry.scale_offset:] == blocks[order, :, :2].tobytes()


@pytest.mark.parametrize("word", [0x7C00, 0xFC00, 0x7E01])
def test_q2_import_refuses_nonfinite_scales(tmp_path, word):
    block = np.zeros((1, 18), dtype=np.uint8)
    block[:, :2] = np.array([word], dtype="<u2").view(np.uint8)
    path = tmp_path / "invalid-q2.gguf"
    write_gguf(path, {}, [("w", (1, 64), 42, block.tobytes())])
    with GGUFFile(path) as gguf:
        source = q2_native_source(gguf, "w", (1, 64), lambda b, e: np.arange(b, e))
        with pytest.raises(ValueError, match="finite"):
            source.read_encoded(0, 1)


def test_q2_import_refuses_a_different_ggml_grid(tmp_path):
    path = tmp_path / "q4.gguf"
    write_gguf(path, {}, [("w", (2, 64), 2, bytes(72))])
    with GGUFFile(path) as gguf:
        with pytest.raises(ValueError, match="Q2_0"):
            q2_native_source(gguf, "w", (2, 64), lambda b, e: np.arange(b, e))


@pytest.mark.parametrize("k", [64, 640, 2560])
def test_q8_import_preserves_signed_codes_scales_and_padding(tmp_path, k):
    rng = np.random.default_rng(71)
    codes = rng.integers(-127, 128, (4, k // 32, 32), dtype=np.int8)
    words = np.resize(np.array([0x3800, 0xBC00, 0x0001, 0x0000], dtype="<u2"),
                      (4, k // 32))
    codes[words == 0] = 0
    blocks = np.empty((4, k // 32, 34), dtype=np.uint8)
    blocks[..., :2] = words.view(np.uint8).reshape(4, k // 32, 2)
    blocks[..., 2:] = codes.view(np.uint8)
    path = tmp_path / "shared.gguf"
    write_gguf(path, {}, [("shared", (4, k), 8, blocks.tobytes())])
    order = np.array([3, 1, 3])
    output = tmp_path / "shared.ninfer"
    with GGUFFile(path) as gguf:
        source = q8_native_source(gguf, "shared", (3, k), lambda b, e: order[b:e])
        model = Model({"text": {"config": {}}})
        model.add(Parameter("shared", (3, k), source, inputs=("input",)))
        recipe = Recipe(model)
        recipe.assign("shared", format="q8_g32_fp16", method=import_encoded,
                      activation_policy="AllowA8")
        convert(model, recipe, output, device="cpu", rows_per_chunk=2)
    with Artifact(output) as artifact:
        payload = artifact.read_object(artifact.directory.bindings["shared"]["object"])
        geometry = row_split_geometry("q8_g32_fp16", (3, k))
        packed = np.frombuffer(payload[:geometry.base_bytes], dtype=np.uint8).reshape(3, -1)
        assert packed[:, :k].tobytes() == codes[order].tobytes()
        assert (packed[:, k:] == 0).all()
        scales = np.frombuffer(payload[geometry.scale_offset:], dtype="<u2").reshape(3, -1)
        assert scales[:, :k // 32].tobytes() == words[order].tobytes()
        assert (scales[:, k // 32:] == 0).all()
        oracle = (codes[order].astype(np.float64) *
                  words[order].view("<f2").astype(np.float64)[..., None]).reshape(3, k)
        assert torch.equal(dequantize_row_split(payload, "q8_g32_fp16", (3, k),
                                               dtype=torch.float64), torch.from_numpy(oracle))


@pytest.mark.parametrize("word,code", [(0x8000, 0), (0, 1), (0x7C00, 0), (0x3800, -128)])
def test_q8_import_refuses_incompatible_source_without_canonicalizing(tmp_path, word, code):
    block = np.zeros((1, 34), dtype=np.uint8)
    block[:, :2] = np.array([word], dtype="<u2").view(np.uint8)
    block[:, 2] = np.array([code], dtype=np.int8).view(np.uint8)
    path = tmp_path / "invalid.gguf"
    write_gguf(path, {}, [("w", (1, 32), 8, block.tobytes())])
    with GGUFFile(path) as gguf:
        source = q8_native_source(gguf, "w", (1, 32), lambda b, e: np.arange(b, e))
        with pytest.raises(ValueError, match="contract"):
            source.read_encoded(0, 1)


def test_native_gsq_recipe_refuses_a_wrong_grid_before_reading_the_table(tmp_path):
    from tools.convert.qwen4_exp_gguf import qwen3_8_flash_next_gsq_q2

    path = tmp_path / "wrong-grid.gguf"
    write_gguf(path, {}, [("blk.0.ffn_gate_exps.weight", (1, 32), 8, bytes(34))])
    model = Model({"text": {"config": {"num_experts": 512}}})
    with GGUFFile(path) as source:
        # No table is provided: reaching that source would fail with a different exception.
        with pytest.raises(ValueError, match="stored Q2_0 experts"):
            qwen3_8_flash_next_gsq_q2(model, Recipe(model), {"gguf": source})
