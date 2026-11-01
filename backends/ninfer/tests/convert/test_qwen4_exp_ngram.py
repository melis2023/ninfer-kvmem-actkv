from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import pytest
from safetensors.torch import save_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.reader import Artifact
from tools.convert.official_recipes import RECIPES
from tools.convert.pipeline import convert
from tools.convert.qwen4_exp import build_model, ngram_config
from tools.convert.qwen4_exp_gguf import qwen3_8_flash_next_gguf
from tools.convert.qwen4_exp_ngram import attach_hot_profile, fp8_table_digest, hf_ngram_source
from tools.convert.quantization.fp8_row import quantize_bf16_rows
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource

_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"


def _hot_profile(rows=(17, 3, 99)):
    # NFNGHOT1 fixture for _fixture's four-head hash. The C++ writer interop checks this
    # fingerprint against independently derived constants, including across file segments.
    return b"NFNGHOT1" + struct.pack("<4Q", 424, 0xCFC0CA182769A039, 12345, len(rows)) + \
        struct.pack(f"<{len(rows)}I", *rows)


@pytest.mark.parametrize("explicit", [False, True])
def test_table_embeds_default_or_selected_profile(tmp_path, explicit):
    _fixture(tmp_path)
    data = _hot_profile()
    (tmp_path / "ngram.hot").write_bytes(b"ignored by override" if explicit else data)
    selected = tmp_path / "selected.hot"
    selected.write_bytes(data)
    with SafetensorsSource(tmp_path) as source:
        model = build_model(source, components=("ngram",),
                            resource_overrides={"ngram.hot": selected} if explicit else None)
        reference = model.components["ngram"]["resources"]["hot_profile"]
        assert model.resources[reference] == data
        recipe = Recipe(model)
        RECIPES["qwen3_8_flash_next"](model, recipe, {"ngram": source})
        path = tmp_path / "profiled.ninfer"
        convert(model, recipe, path, device="cpu", rows_per_chunk=17, max_file_bytes=12288)
    with Artifact(path) as artifact:
        role = artifact.directory.components["ngram"]["resources"]["hot_profile"]
        assert artifact.read_object(role) == data


@pytest.mark.parametrize("data, message", [
    (b"NFNGHOT1", "profile"),
    (_hot_profile()[:-1], "row count"),
    (_hot_profile() + b"x", "row count"),
    (_hot_profile((17, 17)), "duplicate"),
    (_hot_profile((424,)), "out-of-range"),
    (b"NFNGHOT1" + struct.pack("<4Q", 424, 1, 12345, 0), "another n-gram hash"),
    (b"NFNGHOT1" + struct.pack("<4Q", 425, 0xCFC0CA182769A039, 12345, 0), "another n-gram hash"),
])
def test_table_refuses_invalid_embedded_profile(tmp_path, data, message):
    _fixture(tmp_path)
    (tmp_path / "ngram.hot").write_bytes(data)
    with SafetensorsSource(tmp_path) as source, pytest.raises(ValueError, match=message):
        build_model(source, components=("ngram",))


def test_profile_requires_a_stored_table(tmp_path):
    from tools.convert.model import Model
    path = tmp_path / "ngram.hot"
    path.write_bytes(_hot_profile())
    with pytest.raises(ValueError, match="stored n-gram table"):
        attach_hot_profile(Model({}), path)


def _e4(word):
    e, m = (word >> 3) & 15, word & 7
    value = math.ldexp(m, -9) if e == 0 else math.ldexp(1 + m / 8, e - 7)
    return -value if word & 128 else value


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _oracle_row(values):
    maximum = max(abs(v) for v in values)
    if maximum == 0:
        return bytes(len(values) + 2)
    word = max(1, struct.unpack("<H", struct.pack("<e", _f32(maximum / 448)))[0])
    scale = struct.unpack("<e", struct.pack("<H", word))[0]
    reciprocal = _f32(1 / scale)
    codes = []
    for value in values:
        normalized = min(448, abs(_f32(value * reciprocal)))
        # FP64 enumeration of the finite positive codebook, with even-code midpoint ties.
        code = min(range(127), key=lambda c: (abs(_e4(c) - normalized), c & 1))
        codes.append(code | (128 if math.copysign(1, value) < 0 else 0))
    return bytes(codes) + struct.pack("<H", word)


def _fixture(root):
    config = json.loads((Path(__file__).resolve().parents[1] /
                         "fixtures/qwen4_exp/config.json").read_text())
    config["text_config"].update(ngram_vocab_size_base=100,
                                 make_ngram_vocab_size_divisible_by=8, heads_per_ngram=2,
                                 ple_embed_dim=640)
    (root / "config.json").write_text(json.dumps(config))
    rows = ngram_config(config)["rows"]
    assert rows == 424
    values = (torch.arange(rows * 160).float() * .017).sin().reshape(rows, 160).bfloat16()
    values[0].zero_()
    values[1, :5] = torch.tensor([448, 1.0625, 1.1875, -1.0625, -1.1875])
    # Nonuniform shards, deliberately crossing writer chunks and artifact part boundaries.
    edges = [0, 3, 131, rows]
    index = {}
    for s in range(3):
        name = f"{_PREFIX}{s}.weight"
        filename = f"source-{s}.safetensors"
        save_file({name: values[edges[s]:edges[s+1]].clone()}, root / filename)
        index[name] = filename
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return config, values


def _convert_table(source, path, recipe_name, chunk, max_file_bytes=None):
    model = build_model(source, components=("ngram",))
    recipe = Recipe(model)
    apply_recipe = (qwen3_8_flash_next_gguf if recipe_name == "qwen3_8_flash_next_gguf"
                    else RECIPES[recipe_name])
    apply_recipe(model, recipe, {"ngram": source})
    convert(model, recipe, path, device="cpu", rows_per_chunk=chunk,
            max_file_bytes=max_file_bytes)
    with Artifact(path) as artifact:
        obj = artifact.object(artifact.directory.bindings["ngram/table"]["object"])
        payload = artifact.read_object(obj.id)
        assert obj.format == "fp8_e4m3fn_row_fp16" and obj.layout == "row_interleaved_v1"
        assert len(payload) == 424 * 162
        assert artifact.directory.components["ngram"]["config"]["table_sha256"] == hashlib.sha256(payload).hexdigest()
    return payload


def test_fp16_scale_quantizer_against_independent_scalar_oracle():
    # Enumerate finite E4M3 magnitudes and midpoints, signs, zero, half-scale subnormals and
    # near-overflow scales. Inputs are BF16; FP32 normalization and the FP16 scale are boundaries.
    values = [_e4(i) for i in range(127)] + [(_e4(i) + _e4(i + 1)) / 2 for i in range(126)]
    rows = []
    for multiplier in (2.0**-22, 2.0**-14, .125, 1, 64, 32768):
        for sign in (1, -1):
            row = [sign * v * multiplier for v in values]
            row[0] = -0.0
            rows.append(row)
    rows.append([0.] * len(values))
    rows.append([2.0**-40] * len(values))
    inputs = torch.tensor(rows, dtype=torch.bfloat16)
    actual = quantize_bf16_rows(inputs, scale_dtype=torch.float16)
    for row, codes, scale in zip(inputs.float().tolist(), actual.codes.tolist(), actual.scales.tolist()):
        assert bytes(codes) + struct.pack("<e", scale) == _oracle_row(row)
    for value in (float("nan"), float("inf"), 2.0**30):
        with pytest.raises(ValueError, match="finite|NaN"):
            quantize_bf16_rows(torch.full((1, 160), value, dtype=torch.bfloat16),
                               scale_dtype=torch.float16)


def test_shards_stream_across_boundaries_and_hash_independently_of_chunks(tmp_path):
    config, values = _fixture(tmp_path)
    with SafetensorsSource(tmp_path) as store:
        source = hf_ngram_source(store, ngram_config(config))
        for begin, end in ((0, 0), (477, 485), (130 * 160 + 7, 132 * 160 - 3),
                           (423 * 160, 424 * 160)):
            before = store.bytes_read
            assert torch.equal(source.values(begin, end), values.flatten()[begin:end])
            assert store.bytes_read - before == (end - begin) * 2
        digest = fp8_table_digest(source, rows_per_chunk=7)
        assert digest == fp8_table_digest(source, rows_per_chunk=131)
        one = _convert_table(store, tmp_path / "one.ninfer", "qwen3_8_flash_next", 17)
        many = _convert_table(store, tmp_path / "many.ninfer", "qwen3_8_flash_next_gguf", 64, 12288)
        assert one == many and hashlib.sha256(one).hexdigest() == digest
        for row in (0, 1, 2, 3, 25, 75, 130, 131, 423):
            assert one[row*162:(row+1)*162] == _oracle_row(values[row].float().tolist())


@pytest.mark.parametrize("defect, message", [
    ("gap", "consecutively"), ("dtype", "BF16"), ("width", "width"),
    ("rows", "expected 424"), ("constants", "hash constants"),
])
def test_hf_table_refuses_incompatible_shards(tmp_path, defect, message):
    config, _ = _fixture(tmp_path)
    descriptor = ngram_config(config)
    if defect in ("dtype", "width", "rows"):
        shape = (4 if defect == "rows" else 3, 159 if defect == "width" else 160)
        dtype = torch.float32 if defect == "dtype" else torch.bfloat16
        save_file({_PREFIX + "0.weight": torch.zeros(shape, dtype=dtype)},
                  tmp_path / "source-0.safetensors")
    with SafetensorsSource(tmp_path) as store:
        if defect == "gap":
            del store.weight_map[_PREFIX + "1.weight"]
        if defect == "constants":
            descriptor = deepcopy(descriptor)
            descriptor["head_offset"][1] += 1
        with pytest.raises(ValueError, match=message):
            hf_ngram_source(store, descriptor)


def _bf16_word(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def interop_main(binary):
    """Python writer -> artifact file segments -> C++ row reader -> GPU decode."""
    with tempfile.TemporaryDirectory(prefix="ninfer-ngram-fp8-") as temporary:
        root = Path(temporary)
        _fixture(root)
        (root / "ngram.hot").write_bytes(_hot_profile())
        with SafetensorsSource(root) as source:
            for label, limit in (("single", None), ("sharded", 12288)):
                path = root / f"{label}.ninfer"
                payload = _convert_table(source, path, "qwen3_8_flash_next", 17, limit)
                decoded = bytearray()
                for row in range(424):
                    raw = payload[row*162:(row+1)*162]
                    scale = struct.unpack("<e", raw[160:])[0]
                    for code in raw[:160]:
                        decoded.extend(struct.pack("<H", _bf16_word(_e4(code) * scale)))
                oracle = root / f"{label}.bf16"
                oracle.write_bytes(decoded)
                result = subprocess.run([binary, "ninfer_qwen4_exp_ngram_component_test",
                                         "--writer-fixture", str(path), str(oracle)])
                if result.returncode:
                    return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(interop_main(sys.argv[1]))
