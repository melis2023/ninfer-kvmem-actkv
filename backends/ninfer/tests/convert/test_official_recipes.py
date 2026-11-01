from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.convert.methods import fp8_row_maxabs, grouped_search, import_encoded
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import (
    RECIPES,
    qwen3_8_27b,
    qwen3_8_27b_imatrix,
    qwen3_8_27b_nvfp4_nvidia,
    qwen3_8_27b_q6,
)
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"

LAYER = "text/layers/0"


def _dense_model() -> Model:
    model = Model({"text": {"config": {}}})
    names = (
        "text/token_embedding",
        "text/output_head",
        f"{LAYER}/gdn/query",
        f"{LAYER}/gdn/output",
        f"{LAYER}/attention/query",
        f"{LAYER}/attention/key",
        f"{LAYER}/attention/output",
        f"{LAYER}/mlp/gate",
        f"{LAYER}/mlp/up",
        f"{LAYER}/mlp/down",
    )
    for name in names:
        inputs = () if name in ("text/token_embedding", "text/output_head") else ("input",)
        source = array_source(torch.ones((4, 8), dtype=torch.bfloat16), name)
        model.add(Parameter(name, (4, 8), source, inputs=inputs))
    return model


def _formats(recipe_function) -> dict[str, set[str]]:
    model = _dense_model()
    recipe = Recipe(model)
    recipe_function(model, recipe, {})
    return {
        name: {selection.format for selection in selections}
        for name, selections in recipe.selections.items()
    }


def _single(formats: dict[str, set[str]], name: str) -> str:
    values = formats[name]
    assert len(values) == 1, f"{name} is split across formats {values}"
    return next(iter(values))


def test_q6_recipe_is_registered() -> None:
    assert RECIPES["qwen3_8_27b_q6"] is qwen3_8_27b_q6


def test_flash_next_quality_recipe_writes_grouped_experts_and_preserves_sensitive_values(tmp_path):
    from tools.artifact.codecs.direct import decode_direct
    from tools.artifact.codecs.row_split import dequantize_row_split
    from tools.artifact.reader import Artifact
    from tools.artifact.schema import binding_parts
    from tools.convert.pipeline import convert
    from tools.convert.qwen4_exp_gguf import NGRAM_TENSOR
    from tools.convert.sources.gguf import GGUFFile, write_gguf

    model = Model({
        "text": {"config": {"architectures": ["Qwen4ExpForCausalLM"]}},
        "ngram": {"config": {"rows": 2, "row_width": 160}},
        "mtp": {"config": {"architectures": ["Qwen4ExpMTP"]}, "target": "text"},
        "vision": {"config": {}, "target": "text"},
    })
    values, expected = {}, {}
    for prefix in ("text/layers/0/moe/", "mtp/layer/moe/"):
        gate_up, downs = [], []
        for expert in range(2):
            for role in ("gate", "up", "down"):
                name = prefix + f"experts/{expert}/{role}"
                expected[name] = Q5 if prefix.startswith("text/") and role == "down" else Q4
                (downs if role == "down" else gate_up).append(name)
        model.packing_groups.extend((tuple(gate_up), tuple(downs)))
    expected.update({
        "text/token_embedding": Q8, "text/output_head": Q6,
        "text/layers/0/attention/query": Q8, "text/layers/0/moe/shared/gate": Q8,
        "text/layers/0/gdn/a_projection": "bf16", "text/layers/0/attn_hc/down": "bf16",
        "text/layers/0/moe/router": "bf16", "text/layers/0/ple/key": "bf16",
        "text/layers/0/indexer/query": "bf16", "mtp/fc_hidden": "bf16",
        "mtp/layer/moe/shared/down": "bf16", "vision/layers/0/mlp/fc1": "bf16",
    })
    for index, name in enumerate(expected):
        value = (torch.arange(256).float() * .17 + index).sin().reshape(2, 128).bfloat16()
        values[name] = value
        inputs = () if name == "text/token_embedding" else (name + "/input",)
        model.add(Parameter(name, value.shape, array_source(value, name), inputs=inputs))
    path = tmp_path / "table.gguf"
    write_gguf(path, {}, [(NGRAM_TENSOR, (2, 160), 20, bytes(180))])
    with GGUFFile(path) as table:
        recipe = Recipe(model)
        RECIPES["qwen3_8_flash_next"](model, recipe, {"ngram": table})
        output = tmp_path / "quality.ninfer"
        convert(model, recipe, output, device="cpu", rows_per_chunk=1)
    with Artifact(output) as artifact:
        parents = {}
        for name, format in expected.items():
            (parent, begin, end), = binding_parts(artifact.directory.bindings[name], artifact.by_id)
            obj = artifact.object(parent)
            assert obj.format == format
            raw = artifact.read_object(obj.id)
            decoded = (decode_direct(raw, format, obj.shape) if format == "bf16" else
                       dequantize_row_split(raw, format, obj.shape, dtype=torch.float32))
            value = decoded.flatten()[begin:end].reshape(values[name].shape)
            if format == "bf16":
                assert torch.equal(value, values[name])
            else:
                levels = {Q4: 7, Q5: 15, Q6: 31, Q8: 127}[format]
                assert (value.float() - values[name].float()).abs().max() <= .501 / levels
            parents[name] = obj.id
        for group in model.packing_groups:
            assert len({parents[name] for name in group}) == 1
        uses = {u["parameter"]: u for u in artifact.directory.uses}
        assert uses["text/layers/0/moe/experts/1/down"]["activation_policy"] == "AllowA8"
        assert uses["text/layers/0/moe/router"]["activation_policy"] == "A16Only"


def test_flash_next_quality_recipe_refuses_another_family_before_reading_the_table():
    model = _dense_model()
    with pytest.raises(ValueError, match="Flash-Next mathematics"):
        RECIPES["qwen3_8_flash_next"](model, Recipe(model), {})


def test_flash_next_mtp_refuses_unscaled_fused_fp8_experts():
    from tools.convert.official_recipes import flash_next_mtp_formats

    model = Model({"mtp": {"config": {}}})
    name = "mtp/layer/moe/experts/0/gate"
    codes = torch.ones((2, 128)).to(torch.float8_e4m3fn)
    model.add(Parameter(name, codes.shape, array_source(codes, "HF fused FP8"), inputs=("x",)))
    recipe = Recipe(model)
    flash_next_mtp_formats(model, recipe)
    with pytest.raises(ValueError, match="fused FP8 source scales"):
        recipe.prepare(device="cpu")


def test_flash_next_quality_recipe_does_not_ignore_a_selected_gguf_mtp(tmp_path):
    from tools.convert.sources.gguf import GGUFFile, write_gguf

    path = tmp_path / "mtp.gguf"
    write_gguf(path, {}, [])
    model = Model({"text": {"config": {"architectures": ["Qwen4ExpForCausalLM"]}},
                   "mtp": {"config": {}, "target": "text"}})
    with GGUFFile(path) as mtp, pytest.raises(ValueError, match="HF safetensors MTP source"):
        RECIPES["qwen3_8_flash_next"](model, Recipe(model), {"mtp": mtp})


def test_registered_recipe_gives_the_mlp_pair_q4() -> None:
    formats = _formats(qwen3_8_27b)
    assert _single(formats, f"{LAYER}/mlp/gate") == Q4
    assert _single(formats, f"{LAYER}/mlp/up") == Q4
    assert _single(formats, f"{LAYER}/mlp/down") == Q5
    assert _single(formats, "text/token_embedding") == Q8
    assert _single(formats, "text/output_head") == Q8


def test_q6_recipe_moves_only_the_mlp_pair() -> None:
    registered = _formats(qwen3_8_27b)
    q6 = _formats(qwen3_8_27b_q6)

    assert set(registered) == set(q6)
    moved = {name for name in q6 if q6[name] != registered[name]}
    assert moved == {f"{LAYER}/mlp/gate", f"{LAYER}/mlp/up"}

    for name in moved:
        assert registered[name] == {Q4}
        assert q6[name] == {Q6}


def test_q6_recipe_keeps_the_vocabulary_at_q8() -> None:
    """Q8 is 8.5 bits per weight, so the vocabulary endpoints already outrank Q6."""
    formats = _formats(qwen3_8_27b_q6)
    assert _single(formats, "text/token_embedding") == Q8
    assert _single(formats, "text/output_head") == Q8


def test_q6_recipe_leaves_the_other_projections_alone() -> None:
    formats = _formats(qwen3_8_27b_q6)
    assert _single(formats, f"{LAYER}/attention/query") == Q4
    assert _single(formats, f"{LAYER}/attention/key") == Q4
    assert _single(formats, f"{LAYER}/gdn/query") == Q4
    assert _single(formats, f"{LAYER}/attention/output") == Q5
    assert _single(formats, f"{LAYER}/gdn/output") == Q5
    assert _single(formats, f"{LAYER}/mlp/down") == Q5


def _nvidia_model() -> Model:
    model = Model({"text": {"config": {}}})
    names = (
        "text/token_embedding",
        "text/output_head",
        f"{LAYER}/gdn/a_projection",
        f"{LAYER}/gdn/b_projection",
        f"{LAYER}/gdn/query",
        f"{LAYER}/gdn/output",
        f"{LAYER}/attention/query",
        f"{LAYER}/attention/output",
        f"{LAYER}/mlp/gate",
        f"{LAYER}/mlp/up",
        f"{LAYER}/mlp/down",
    )
    for name in names:
        inputs = () if name == "text/token_embedding" else ("input",)
        source = array_source(torch.ones((4, 8), dtype=torch.bfloat16), name)
        model.add(
            Parameter(
                name,
                (4, 8),
                source,
                source_factory=lambda store, fmt, _source=source: _source,
                inputs=inputs,
            )
        )
    return model


def test_nvidia_recipe_stores_the_head_as_fp8() -> None:
    """The runtime's nvfp4 linear op has no vocabulary-shape kernel, so the
    source's NVFP4 head must be dequantised and re-quantised to FP8."""
    model = _nvidia_model()
    recipe = Recipe(model)
    qwen3_8_27b_nvfp4_nvidia(model, recipe, {"quantized": object()})

    head = recipe.selections["text/output_head"]
    assert len(head) == 1
    assert head[0].format == "fp8_e4m3fn_row_bf16"
    assert head[0].method is fp8_row_maxabs

    assert _single(_selection_formats(recipe), f"{LAYER}/mlp/gate") == "nvfp4"
    assert _single(_selection_formats(recipe), f"{LAYER}/mlp/down") == "nvfp4"
    mlp_gate = recipe.selections[f"{LAYER}/mlp/gate"][0]
    assert mlp_gate.method is import_encoded
    assert _single(_selection_formats(recipe), f"{LAYER}/attention/query") == (
        "fp8_e4m3fn_row_bf16"
    )
    assert _single(_selection_formats(recipe), f"{LAYER}/gdn/query") == (
        "fp8_e4m3fn_row_bf16"
    )


def _selection_formats(recipe: Recipe) -> dict[str, set[str]]:
    return {
        name: {selection.format for selection in selections}
        for name, selections in recipe.selections.items()
    }


def _imatrix_recipe() -> Recipe:
    model = Model({"text": {"config": {}}})
    names = ["text/token_embedding", "text/output_head"]
    for layer in (35, 36):
        for role in (
            "gdn/a_projection", "gdn/query", "gdn/value", "gdn/output", "attention/query",
            "attention/gate", "attention/output", "mlp/gate", "mlp/up", "mlp/down",
        ):
            names.append(f"text/layers/{layer}/{role}")
    for name in names:
        inputs = () if name == "text/token_embedding" else ("input",)
        source = array_source(torch.ones((4, 8), dtype=torch.bfloat16), name)
        model.add(Parameter(name, (4, 8), source, inputs=inputs))
    recipe = Recipe(model)
    qwen3_8_27b_imatrix(model, recipe, {"imatrix": SimpleNamespace(path=Path("im.safetensors"))})
    return recipe


def test_imatrix_recipe_searches_every_text_matrix_with_signed_scales() -> None:
    recipe = _imatrix_recipe()
    assert RECIPES["qwen3_8_27b_imatrix"] is qwen3_8_27b_imatrix
    for name, selections in recipe.selections.items():
        if name.endswith("/gdn/a_projection"):
            assert name in recipe.separate_parameters
            continue
        (selection,) = selections
        assert selection.method is grouped_search, name
        assert selection.parameters == {"imatrix": "im.safetensors", "negative_scales": True}


@pytest.mark.parametrize(
    "name, format",
    [
        ("text/token_embedding", Q4),
        ("text/output_head", Q6),
        ("text/layers/35/gdn/query", Q4),
        ("text/layers/35/gdn/value", Q5),
        ("text/layers/35/attention/gate", Q5),
        ("text/layers/35/mlp/up", Q4),
        ("text/layers/35/gdn/output", Q5),
        ("text/layers/35/attention/output", Q5),
        ("text/layers/35/mlp/down", Q5),
        ("text/layers/36/gdn/output", Q4),
        ("text/layers/36/attention/output", Q4),
        ("text/layers/36/mlp/down", Q4),
        ("text/layers/36/gdn/value", Q5),
    ],
)
def test_imatrix_recipe_formats(name, format) -> None:
    assert _single(_selection_formats(_imatrix_recipe()), name) == format


def test_imatrix_recipe_requires_the_imatrix_source() -> None:
    model = _dense_model()
    with pytest.raises(KeyError):
        qwen3_8_27b_imatrix(model, Recipe(model), {})
