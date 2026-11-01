from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.convert import gguf_blocks, qwen4_exp, qwen4_exp_gguf
from tools.convert.sources.gguf import TYPE_BF16, GGUFFile, write_gguf

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
    # Vision and the MTP block convert only beside the text model.
    class Base:
        config = _checkpoint()
        root = tmp_path

    for components in (("mtp",), ("vision",), ("ngram", "mtp"), (), ("text", "dflash")):
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


TYPE_Q8_0 = 8


def _q8_0_blocks(generator, rows: int, columns: int) -> np.ndarray:
    blocks = generator.integers(-127, 128, size=(rows, columns // 32, 34)).astype(np.int8).view(
        np.uint8)
    scales = generator.uniform(0.001, 0.002, size=(rows, columns // 32)).astype(np.float16)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(rows, columns // 32, 2)
    return blocks


def test_expected_mtp_tensors_are_one_qsa_block_with_its_input_and_mixer():
    tensors = qwen4_exp_gguf.expected_mtp_tensors()
    # HC (8), attention and indexer (10), MoE (8), eh_proj, two input norms, the mixer (3).
    assert len(tensors) == 8 + 10 + 8 + 6
    assert tensors["blk.48.nextn.eh_proj.weight"] == ((2560, 5120), "matrix")
    assert tensors["blk.48.nextn.hnorm.weight"] == ((10240,), "direct")
    assert tensors["blk.48.ffn_down_exps.weight"] == ((512, 2560, 640), "matrix")
    assert not any(".ssm_" in name or ".ple_" in name for name in tensors)
    assert qwen4_exp_gguf.expected_mtp_tensors(256)["blk.48.ffn_gate_inp.weight"] == (
        (256, 2560), "matrix")


def test_eh_proj_splits_into_its_two_projections_between_blocks(tmp_path):
    # [fc_embedding | fc_hidden]: the embedding's columns first, cut between Q8_0 blocks.
    generator = np.random.default_rng(9)
    rows, columns = 3, 128
    blocks = _q8_0_blocks(generator, rows, columns)
    path = tmp_path / "eh.gguf"
    write_gguf(path, {"general.architecture": "qwen4exp"},
               [("eh", (rows, columns), TYPE_Q8_0, blocks.tobytes())])
    with GGUFFile(path) as gguf:
        first, encoded = qwen4_exp_gguf.column_source(gguf, "eh", 0, 64)
        second, _ = qwen4_exp_gguf.column_source(gguf, "eh", 64, 128)
        assert encoded and first.shape == (3, 64)
        assert torch.equal(first.read_encoded(1, 3).codes,
                           torch.from_numpy(blocks.reshape(rows, -1)[1:3, :68]))
        assert torch.equal(second.read_encoded(0, 3).codes,
                           torch.from_numpy(blocks.reshape(rows, -1)[:, 68:]))
        assert first.read_encoded(0, 1).format == "gguf_q8_0"
        with pytest.raises(ValueError, match="between"):
            qwen4_exp_gguf.column_source(gguf, "eh", 0, 48)
        pytest.importorskip("gguf")
        scales = blocks[..., :2].copy().view(np.float16).astype(np.float32)
        codes = blocks[..., 2:].view(np.int8).astype(np.float32)
        values = (codes * scales).reshape(rows, columns)
        assert torch.equal(second.rows(0, 3), torch.from_numpy(values[:, 64:]))


def test_mtp_gguf_must_be_a_flash_next_mtp_block(tmp_path):
    path = tmp_path / "mtp.gguf"
    header = dict(qwen4_exp_gguf.MTP_HEADER) | {"qwen4exp.expert_count": 512}
    write_gguf(path, header | {"qwen4exp.nextn_shared_target_tensors": False}, [])
    with GGUFFile(path) as gguf, pytest.raises(ValueError, match="nextn_shared_target_tensors"):
        qwen4_exp_gguf.validate_mtp(gguf, 512)
    write_gguf(path, header, [])
    with GGUFFile(path) as gguf:
        with pytest.raises(ValueError, match="512 experts in the MTP block"):
            qwen4_exp_gguf.validate_mtp(gguf, 256)
        with pytest.raises(ValueError, match="tensor set mismatch"):
            qwen4_exp_gguf.validate_mtp(gguf, 512)
    # The MTP block keeps its GGUF's experts whatever the converted model kept.
    pruned = qwen4_exp_gguf.with_gguf_expert_count(_checkpoint(), _pruned(tmp_path))
    with_mtp = qwen4_exp_gguf.with_mtp_expert_count(pruned, path)
    assert qwen4_exp.text_config(with_mtp)["num_experts"] == 256
    assert qwen4_exp.mtp_config(with_mtp)["num_experts"] == 512


def _pruned(tmp_path):
    path = tmp_path / "pruned.gguf"
    write_gguf(path, {"general.architecture": "qwen4exp", "qwen4exp.expert_count": 256}, [])
    return path


def _q8_0_reference(row: np.ndarray) -> bytes:
    """ggml quantize_row_q8_0_ref, one block at a time in FP32 scalars."""

    out = bytearray()
    for block in row.astype(np.float32).reshape(-1, 32):
        d = np.float32(max(abs(float(x)) for x in block)) / np.float32(127)
        inverse = np.float32(1) / d if d else np.float32(0)
        out += np.float16(d).tobytes()
        for x in block:
            scaled = float(np.float32(x) * inverse)
            code = int(np.sign(scaled) * np.floor(abs(scaled) + 0.5))
            out += code.to_bytes(1, "little", signed=True)
    return bytes(out)


def test_hyper_connection_matrices_become_ggml_q8_0_rows(tmp_path):
    # A BF16 matrix is quantized by ggml's reference rounding; a Q8_0 one keeps its blocks.
    generator = np.random.default_rng(10)
    rows, columns = 5, 96
    values = (generator.standard_normal((rows, columns)) * 0.05).astype(np.float32)
    values[1, :32] = 0.0
    words = (values.view(np.uint32) >> 16).astype(np.uint16)
    blocks = _q8_0_blocks(generator, rows, columns)
    path = tmp_path / "hc.gguf"
    write_gguf(path, {"general.architecture": "qwen4exp"},
               [("bf16", (rows, columns), TYPE_BF16, words.tobytes()),
                ("q8", (rows, columns), TYPE_Q8_0, blocks.tobytes())])
    represented = (words.astype(np.uint32) << 16).view(np.float32)
    with GGUFFile(path) as gguf:
        quantized = qwen4_exp_gguf.hc_matrix_source(gguf, "bf16", (rows, columns))
        encoded = quantized.read_encoded(1, 4)
        assert encoded.format == "gguf_q8_0"
        assert encoded.codes.numpy().tobytes() == b"".join(
            _q8_0_reference(represented[r]) for r in range(1, 4))
        assert torch.equal(quantized.rows(1, 4),
                           torch.from_numpy(gguf_blocks.dequantize_q8_0(encoded.codes.numpy())))
        kept = qwen4_exp_gguf.hc_matrix_source(gguf, "q8", (rows, columns))
        assert torch.equal(kept.read_encoded(0, rows).codes,
                           torch.from_numpy(blocks.reshape(rows, -1)))
