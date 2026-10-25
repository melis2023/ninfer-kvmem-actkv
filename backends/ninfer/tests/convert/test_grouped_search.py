from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.row_split import decode_row_split_codes
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert import imatrix
from tools.convert.gguf_blocks import tiled_input_columns
from tools.convert.methods import grouped_search
from tools.convert.model import Model, Parameter
from tools.convert.quantization.groupwise import search_quantize_matrix
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

from .test_recipe import _write


def _model(values):
    model = Model({"text": {"config": {}}})
    for name, value in values.items():
        model.add(Parameter(name, tuple(value.shape), array_source(value, name), inputs=("input",)))
    model.packing_groups = [tuple(values)]
    return model


def test_grouped_search_packs_inputs_and_matches_the_weighted_search(tmp_path):
    generator = torch.Generator().manual_seed(3)
    values = {
        name: (torch.randn((6, 192), generator=generator) * 0.05).to(torch.bfloat16)
        for name in ("query", "key")
    }
    importance = torch.rand(192, generator=generator) * 4
    path = tmp_path / "imatrix.safetensors"
    save_file({"query": importance, "key": importance.clone()}, str(path))
    imatrix.load.cache_clear()

    model = _model(values)
    recipe = Recipe(model)
    recipe.assign(
        ("query", "key"),
        format="q4_g64_fp16",
        method="grouped_search",
        parameters={"imatrix": str(path), "negative_scales": True},
    )
    prepared = recipe.prepare(device="cpu", rows_per_chunk=4)
    artifact_path = tmp_path / "search.ninfer"
    _write(artifact_path, model, prepared)

    expected = search_quantize_matrix(
        torch.cat(list(values.values())),
        "q4_g64_fp16",
        importance=importance,
        negative_scales=True,
        device="cpu",
    )
    with Artifact(artifact_path) as artifact:
        parts = {
            name: binding_parts(binding, artifact.by_id)[0]
            for name, binding in artifact.directory.bindings.items()
        }
        assert parts["query"][0] == parts["key"][0]
        obj = artifact.object(parts["query"][0])
        scales, codes = decode_row_split_codes(
            artifact.read_object(obj.id), obj.format, obj.shape
        )
    assert torch.equal(scales.view(torch.int16), expected.scales.view(torch.int16))
    assert torch.equal(codes, expected.codes)
    assert bool((scales < 0).any())


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"clip": 1.0}, "unknown grouped_search parameters"),
        ({"negative_scales": 1}, "must be a bool"),
    ],
)
def test_grouped_search_rejects_bad_parameters(parameters, message):
    model = _model({"query": torch.zeros((2, 64), dtype=torch.bfloat16)})
    recipe = Recipe(model)
    recipe.assign("query", format="q4_g64_fp16", method=grouped_search, parameters=parameters)
    with pytest.raises(ValueError, match=message):
        recipe.prepare(device="cpu", rows_per_chunk=4)


def _write_imatrix(path, sums: dict[str, np.ndarray], count: float = 2.0) -> None:
    gguf = pytest.importorskip("gguf")
    writer = gguf.GGUFWriter(str(path), "imatrix")
    for name, values in sums.items():
        writer.add_tensor(name + ".in_sum2", values.astype(np.float32))
        writer.add_tensor(name + ".counts", np.array([count], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_imatrix_import_maps_roles_and_averages_by_count(tmp_path):
    config = {
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 2,
    }
    path = tmp_path / "imatrix.gguf"
    _write_imatrix(
        path,
        {
            "blk.0.attn_qkv.weight": np.full(4, 6.0),
            "blk.0.ssm_out.weight": np.arange(8) * 2.0,
            "blk.1.attn_q.weight": np.full(4, 9.0),
            "blk.1.attn_qkv.weight": np.full(4, 1.0),  # not a full-attention role
        },
    )
    vectors = imatrix.from_gguf(path, config)
    assert set(vectors) == {
        "text/layers/0/gdn/query",
        "text/layers/0/gdn/key",
        "text/layers/0/gdn/value",
        "text/layers/0/gdn/output",
        "text/layers/1/attention/query",
        "text/layers/1/attention/gate",
    }
    assert torch.equal(vectors["text/layers/0/gdn/key"], torch.full((4,), 3.0))
    assert torch.equal(vectors["text/layers/1/attention/gate"], torch.full((4,), 4.5))


def test_imatrix_import_restores_grouped_value_heads_of_the_real_model(tmp_path):
    # GGUF ssm_out column c multiplies grouped activation element tiled_input_columns()[c]
    # (Qwen3.8-27B geometry), so the importance of that column belongs at that element.
    config = {
        "layer_types": ["linear_attention"],
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
    }
    tiled = np.arange(6144, dtype=np.float64)
    path = tmp_path / "imatrix.gguf"
    _write_imatrix(path, {"blk.0.ssm_out.weight": tiled}, count=1.0)
    grouped = imatrix.from_gguf(path, config)["text/layers/0/gdn/output"]
    expected = torch.empty(6144)
    expected[torch.from_numpy(tiled_input_columns().astype(np.int64))] = torch.from_numpy(
        tiled.astype(np.float32)
    )
    assert torch.equal(grouped, expected)
    assert not torch.equal(grouped, torch.from_numpy(tiled.astype(np.float32)))


def test_imatrix_cli_writes_loadable_vectors(tmp_path):
    config = {
        "text_config": {
            "layer_types": ["full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 2,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    _write_imatrix(tmp_path / "imatrix.gguf", {"blk.0.ffn_down.weight": np.full(8, 4.0)})
    out = tmp_path / "imatrix.safetensors"
    imatrix.main(
        ["--gguf", str(tmp_path / "imatrix.gguf"), "--config", str(tmp_path / "config.json"),
         "--out", str(out)]
    )
    imatrix.load.cache_clear()
    assert torch.equal(imatrix.load(str(out))["text/layers/0/mlp/down"], torch.full((8,), 2.0))
