from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.convert import gguf_blocks, qwen4_exp, qwen4_exp_gguf
from tools.convert.sources.gguf import GGUFFile, write_gguf

TYPE_Q2_0 = 42
TYPE_IQ4_NL = 20

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "qwen4_exp"


def _checkpoint():
    return json.loads((_FIXTURES / "config.json").read_text())


def _q2_0_blocks(generator, rows: int, columns: int) -> np.ndarray:
    blocks = generator.integers(0, 256, size=(rows, columns // 64, 18), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 64)).astype(np.float16)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(rows, columns // 64, 2)
    return blocks


def test_q2_0_rows_decode_as_ggml_defines_them():
    generator = np.random.default_rng(5)
    blocks = _q2_0_blocks(generator, 3, 128)
    values = gguf_blocks.dequantize_q2_0(blocks.reshape(3, -1))
    for row in range(3):
        for block in range(2):
            b = blocks[row, block]
            d = float(b[:2].copy().view(np.float16)[0])
            for j in (0, 1, 2, 3, 4, 31, 63):
                code = (int(b[2 + j // 4]) >> (2 * (j % 4))) & 3
                assert values[row, block * 64 + j] == np.float32(d * (code - 1))


def test_expert_bank_rows_are_addressed_matrix_major(tmp_path):
    # A [experts, rows, k] bank: expert e's row r is flattened row e * rows + r.
    generator = np.random.default_rng(6)
    experts, rows, columns = 3, 4, 128
    blocks = _q2_0_blocks(generator, experts * rows, columns)
    path = tmp_path / "bank.gguf"
    write_gguf(path, {"general.architecture": "qwen4exp"},
               [("bank", (experts, rows, columns), TYPE_Q2_0, blocks.tobytes())])
    with GGUFFile(path) as gguf:
        assert gguf.info("bank").shape == (experts, rows, columns)
        source = gguf_blocks.block_source(gguf, "bank", (rows, columns), qwen4_exp_gguf.rows(2 * rows))
        encoded = source.read_encoded(0, rows)
        assert encoded.format == "gguf_q2_0"
        assert torch.equal(encoded.codes, torch.from_numpy(blocks.reshape(experts * rows, -1)[2 * rows:]))
        assert torch.equal(source.rows(1, 3),
                           torch.from_numpy(gguf_blocks.dequantize_q2_0(blocks[2 * rows + 1:2 * rows + 3])))


def test_gdn_gate_rows_return_to_the_grouped_head_order():
    select = qwen4_exp_gguf.untiled_heads()
    order = select(0, 48)
    assert sorted(order.tolist()) == list(range(48))
    for grouped in (0, 1, 2, 3, 46, 47):
        key_head, repeat = grouped // 3, grouped % 3
        assert order[grouped] == repeat * 16 + key_head


def test_expected_tensors_cover_the_release_layout():
    tensors = qwen4_exp_gguf.expected_tensors()
    # Global 5, 48 layers of HC (8) and MoE (8), 36 GDN layers (9), 12 QSA layers (10), PLE (6).
    assert len(tensors) == 5 + 48 * 16 + 36 * 9 + 12 * 10 + 6
    assert tensors["blk.0.ffn_down_exps.weight"] == ((512, 2560, 640), "matrix")
    assert tensors["blk.3.indexer.k_proj.weight"] == ((128, 2560), "matrix")
    assert "blk.1.ple_key.weight" in tensors and "blk.5.ple_key.weight" not in tensors


def test_ngram_config_has_the_reference_constants():
    config = qwen4_exp.ngram_config(_checkpoint())
    # The constants the checkpoint stores (plan section 1.4.1), derived here from the config.
    assert config["multipliers"] == [23703573157769, 20109073645365, 8052911324071]
    assert config["head_vocab"][:3] == [20000003, 20000023, 20000033]
    assert config["head_vocab"][-1] == 20000171
    assert config["head_offset"][1] == 20000003 and config["head_offset"][15] == 300001275
    assert config["rows"] == 320001536 and config["row_width"] == 160
    assert config["architectures"] == ["Qwen4ExpNgramTable"]


def test_flash_next_converts_a_model_a_table_or_both(tmp_path):
    # Nothing converts beside the text model and its n-gram table.
    class Base:
        config = _checkpoint()
        root = tmp_path

    for components in (("text", "mtp"), ("vision",), ("text", "ngram", "mtp"), ()):
        with pytest.raises(ValueError, match="--components text,ngram"):
            qwen4_exp.build_model(Base(), components=components)
    # The table alone: its descriptor and its rows, no text component.
    table = qwen4_exp.build_model(Base(), components=("ngram",))
    assert set(table.components) == {"ngram"} and set(table.parameters) == {"ngram/table"}
    assert table.parameters["ngram/table"].shape == (320001536, 160)


def _tiny_table_config():
    # Four hash heads of 101, 103, 107 and 109 rows (padded to 424), 160 values per row as in the
    # release.
    config = deepcopy(_checkpoint())
    text = config["text_config"]
    text.update(ngram_vocab_size_base=100, make_ngram_vocab_size_divisible_by=8,
                heads_per_ngram=2, ple_embed_dim=640)
    return config


def test_the_table_artifact_stores_the_rows_and_their_digest(tmp_path):
    from tools.artifact.reader import Artifact
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe

    class Base:
        config = _tiny_table_config()
        root = tmp_path

    generator = np.random.default_rng(8)
    rows = 424
    blocks = generator.integers(0, 256, size=(rows, 5, 18), dtype=np.uint8)
    blocks[:, :, :2] = generator.uniform(0.01, 0.02, size=(rows, 5)).astype(np.float16).view(
        np.uint8).reshape(rows, 5, 2)
    shard = tmp_path / "shard2.gguf"
    write_gguf(shard, {"general.architecture": "qwen4exp"},
               [(qwen4_exp_gguf.NGRAM_TENSOR, (rows, 160), TYPE_IQ4_NL, blocks.tobytes())])
    expected = hashlib.sha256(blocks.tobytes()).hexdigest()
    with GGUFFile(shard) as table:
        assert qwen4_exp_gguf.table_digest(table) == expected
        model = qwen4_exp.build_model(Base(), components=("ngram",))
        recipe = Recipe(model)
        qwen4_exp_gguf.qwen3_8_flash_next_gguf(model, recipe, {"ngram": table})
        output = tmp_path / "table.ninfer"
        convert(model, recipe, output, device="cpu")
    with Artifact(output) as artifact:
        directory = artifact.directory
        descriptor = directory.components["ngram"]["config"]
        assert set(directory.components) == {"ngram"}
        assert descriptor["format"] == "gguf_iq4_nl" and descriptor["table_sha256"] == expected
        assert descriptor["rows"] == rows and descriptor["row_width"] == 160
        stored = directory.bindings["ngram/table"]["object"]
        assert hashlib.sha256(artifact.read_object(stored)).hexdigest() == expected


def test_an_expert_pruned_release_takes_its_own_expert_count(tmp_path):
    # The Coder release keeps 256 of the 512 experts per layer, and one router row for each.
    path = tmp_path / "pruned.gguf"
    write_gguf(path, {"general.architecture": "qwen4exp", "qwen4exp.expert_count": 256}, [])
    checkpoint = _checkpoint()
    pruned = qwen4_exp_gguf.with_gguf_expert_count(checkpoint, path)
    assert qwen4_exp.text_config(pruned)["num_experts"] == 256
    assert qwen4_exp.text_config(checkpoint)["num_experts"] == 512
    tensors = qwen4_exp_gguf.expected_tensors((1,), 256)
    assert tensors["blk.0.ffn_gate_inp.weight"] == ((256, 2560), "matrix")
    assert tensors["blk.0.ffn_gate_exps.weight"] == ((256, 640, 2560), "matrix")
    write_gguf(path, {"general.architecture": "qwen4exp", "qwen4exp.expert_count": 600}, [])
    with pytest.raises(ValueError):
        qwen4_exp_gguf.with_gguf_expert_count(checkpoint, path)
