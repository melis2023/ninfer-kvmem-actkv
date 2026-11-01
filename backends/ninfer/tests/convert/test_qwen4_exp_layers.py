from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert.__main__ import _layers, main
from tools.convert.official_recipes import qwen3_8_flash_next
from tools.convert.qwen4_exp import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource
from tests.convert.test_qwen4_exp import _tiny_config


def checkpoint(root: Path, *, offset=0):
    root.mkdir()
    config = _tiny_config()
    config["text_config"].update(
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        ple_layer_ids=[2],
    )
    (root / "config.json").write_text(json.dumps(config))
    values = {
        "model.language_model.layers.1.ple.conv1d.weight":
            torch.arange(64).reshape(16, 1, 4).to(torch.bfloat16) + offset,
        "model.language_model.layers.2.linear_attn.norm.weight":
            torch.tensor([2, 3], dtype=torch.bfloat16) + offset,
        "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight":
            torch.arange(96).reshape(12, 8).to(torch.bfloat16) + offset,
    }
    save_file(values, root / "model.safetensors")
    for name, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (root / name).write_text(json.dumps(value))
    (root / "chat_template.jinja").write_text("{{ messages }}")
    return values


@pytest.mark.parametrize("value", ["", "1", "1:3", "-1..3", "3..3", "4..2", "0..1.5"])
def test_cli_layer_range_refuses_ambiguous_or_empty_ranges(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _layers(value)


def test_cli_layer_range_is_half_open():
    assert _layers("0..4") == (0, 4)
    assert _layers("3..4") == (3, 4)


def test_hf_slice_rebases_weights_uses_and_source_overrides(tmp_path):
    values = checkpoint(tmp_path / "base")
    # A slice without PLE must not load a checkpoint's unused default hot-row profile.
    (tmp_path / "base" / "ngram.hot").write_bytes(b"unused default profile")
    override_values = checkpoint(tmp_path / "override", offset=100)
    with SafetensorsSource(tmp_path / "base") as base, \
            SafetensorsSource(tmp_path / "override") as override:
        original = deepcopy(base.config)
        model = build_model(base, layers=(2, 4))
        assert model.source_layers == (2, 3)
        assert model.config["num_hidden_layers"] == 2
        assert model.config["layer_types"] == ["linear_attention", "full_attention"]
        assert model.config["ple_layers"] == []
        assert set(model.components) == {"text"}
        with pytest.raises(ValueError, match="stored n-gram table"):
            build_model(base, layers=(2, 4),
                        resource_overrides={"ngram.hot": tmp_path / "base" / "ngram.hot"})
        assert not any("/ple/" in p or p.startswith("ngram/") for p in model.parameters)
        assert "text/layers/2/gdn/norm" not in model.parameters
        norm = model.parameters["text/layers/0/gdn/norm"]
        assert torch.equal(norm.source.values(), values[
            "model.language_model.layers.2.linear_attn.norm.weight"])
        key = "text/layers/1/indexer/key"
        source_key = "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"
        assert torch.equal(model.parameters[key].source.rows(0, 4), values[source_key][8:])
        assert model.parameters[key].inputs == ("text/layers/1/mixer_input",)
        assert torch.equal(model.source(key, override).rows(0, 4), override_values[source_key][8:])
        assert base.config == original
        # The official HF recipe must not open or hash a table that this slice cannot use.
        recipe = Recipe(model)
        qwen3_8_flash_next(model, recipe, {})
        assert recipe.policies[(key, "text/layers/1/mixer_input")] == "A16Only"
        override.config["text_config"]["layer_types"][3] = "linear_attention"
        with pytest.raises(ValueError, match="topology differs"):
            model.source(key, override)


def test_slice_rebases_ple_and_preserves_its_table_constants(tmp_path):
    values = checkpoint(tmp_path / "source")
    with SafetensorsSource(tmp_path / "source") as source:
        full = build_model(source)
        sliced = build_model(source, layers=(1, 3))
        assert sliced.config["ple_layers"] == [0]
        assert sliced.components["ngram"] == full.components["ngram"]
        assert sliced.parameters["ngram/table"].shape == full.parameters["ngram/table"].shape
        convolution = sliced.parameters["text/layers/0/ple/convolution"].source.values()
        assert torch.equal(convolution, values[
            "model.language_model.layers.1.ple.conv1d.weight"].reshape(-1))
        assert not any(name.startswith("text/layers/1/ple/") for name in sliced.parameters)


@pytest.mark.parametrize("layers", [(-1, 2), (1, 1), (3, 2), (0, 5), (True, 2), (0,), "0..2"])
def test_slice_refuses_invalid_bounds_before_loading_resources(tmp_path, layers):
    class Source:
        config = _tiny_config()
        root = tmp_path

    with pytest.raises(ValueError, match="--layers requires"):
        build_model(Source(), layers=layers)
    with pytest.raises(ValueError, match="requires the text component"):
        build_model(Source(), components=("ngram",), layers=(0, 1))


def test_cli_slice_writes_selected_source_bytes_and_provenance(tmp_path):
    values = checkpoint(tmp_path / "source")
    recipe = tmp_path / "recipe.py"
    recipe.write_text('''import torch
from tools.convert.sources.logical import array_source

def configure(model, recipe, sources):
    for name, parameter in model.parameters.items():
        if name != "text/layers/0/indexer/key":
            recipe.assign(name, source=array_source(torch.ones(parameter.shape), "fixture"))
''')
    output = tmp_path / "slice.ninfer"
    main(["--model", str(tmp_path / "source"), "--recipe", str(recipe),
          "--layers", "3..4", "--out", str(output), "--device", "cpu",
          "--rows-per-chunk", "3"])
    with Artifact(output) as artifact:
        directory = artifact.directory
        assert directory.provenance["source_layers"] == [3]
        assert set(directory.components) == {"text"}
        assert directory.components["text"]["config"]["layer_types"] == ["full_attention"]
        assert directory.components["text"]["config"]["ple_layers"] == []
        name = "text/layers/0/indexer/key"
        payload = b"".join(artifact.read_object(parent)[begin * 2:end * 2]
                           for parent, begin, end in binding_parts(directory.bindings[name], artifact.by_id))
        expected = values["model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"][8:]
        assert payload == expected.view(torch.uint16).numpy().astype("<u2").tobytes()
        assert any(use["parameter"] == name and use["input"] == "text/layers/0/mixer_input"
                   for use in directory.uses)
        assert not any(key.startswith("text/layers/1/") for key in directory.bindings)


def test_cli_refuses_slicing_another_architecture(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"]}))
    with pytest.raises(ValueError, match="implemented for Qwen3.8-Flash-Next"):
        main(["--model", str(tmp_path), "--recipe", "unused", "--layers", "0..2",
              "--out", str(tmp_path / "invalid.ninfer")])
