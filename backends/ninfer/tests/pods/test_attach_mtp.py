from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from scripts.pods.attach_mtp import attach, plan, verify_copy
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter


def inputs(root):
    config = {"architectures": ["Qwen4ExpForCausalLM"], "hidden_size": 2, "num_experts": 256}
    base_path, donor_path = root / "base.ninfer", root / "donor.ninfer"
    spec = TensorSpec("weight", (2, 2), "bf16", "contiguous_le_v1")
    components = {"text": {"config": config, "resources": {"tokenizer": "tokenizer"}},
                  "vision": {"config": {"hidden_size": 2}},
                  "ngram": {"config": {"table_sha256": "a" * 64}}}
    with ArtifactWriter(base_path, [spec, ResourceSpec("tokenizer", 5)], components=components,
                        bindings={"text/weight": {"object": "weight"},
                                  "text/output_head": {"object": "weight"},
                                  "vision/weight": {"object": "weight"}},
                        uses=[{"parameter": "text/weight", "input": "text/x", "activation_policy": "AllowA8"},
                              {"parameter": "text/output_head", "input": "text/final_hidden",
                               "activation_policy": "A16Only"}]) as w:
        w.write_object("weight", bytes(range(8)))
        w.write_object("tokenizer", b"hello")
    components = deepcopy(components)
    components.pop("vision")
    components["text"]["config"]["num_experts"] = 512
    components["mtp"] = {"target": "text", "config": {"architectures": ["Qwen4ExpMTP"]}}
    with ArtifactWriter(donor_path, [spec, ResourceSpec("tokenizer", 5)], components=components,
                        bindings={"text/output_head": {"object": "weight"},
                                  "mtp/weight": {"parts": [{"object": "weight", "range": [2, 4]},
                                                           {"object": "weight", "range": [0, 2]}]}},
                        uses=[{"parameter": "mtp/weight", "input": "mtp/x", "activation_policy": "AllowA8",
                               "auxiliaries": {"scale": {"object": "weight"}}},
                              {"parameter": "text/output_head", "input": "mtp/final_hidden",
                               "activation_policy": "AllowA8"}]) as w:
        w.write_object("weight", bytes(reversed(range(8))))
        w.write_object("tokenizer", b"other")
    return base_path, donor_path


def test_attachment_preserves_base_and_remaps_mtp(tmp_path):
    base_path, donor_path = inputs(tmp_path)
    out = tmp_path / "result.ninfer"
    with Artifact(base_path) as base, Artifact(donor_path) as donor:
        report = attach(base, donor, out)
        assert report['file_sha256'] == hashlib.sha256(out.read_bytes()).hexdigest()
        with Artifact(out) as result:
            for name, part in base.directory.components.items():
                assert result.directory.components[name] == part
            for obj in base.objects:
                assert result.read_object(obj.id) == base.read_object(obj.id)
            assert result.directory.bindings["text/weight"] == base.directory.bindings["text/weight"]
            assert result.read_object("mtp-import/weight") == donor.read_object("weight")
            assert result.directory.bindings["mtp/weight"]["parts"][0] == {
                "object": "mtp-import/weight", "range": [2, 4]}
            mtp_use = next(u for u in result.directory.uses if u["parameter"] == "mtp/weight")
            assert mtp_use["auxiliaries"]["scale"]["object"] == "mtp-import/weight"
            assert result.directory.uses[-1] == {"parameter": "text/output_head",
                "input": "mtp/final_hidden", "activation_policy": "A16Only"}
            assert result.read_object("tokenizer") == b"hello"
            assert "mtp-import/tokenizer" not in result.by_id
            assert report["base_objects_verified"] == 2
            assert report["mtp_objects_verified"] == 1
            with pytest.raises(ValueError, match="already contains"):
                plan(result.directory, donor.directory)
    assert json.loads(out.with_suffix(".ninfer.conversion.json").read_text())["gpu_qualification"] == "pending"


@pytest.mark.parametrize("change", ["text", "ngram"])
def test_incompatible_donor_refused_before_output(tmp_path, change):
    base_path, donor_path = inputs(tmp_path)
    with Artifact(base_path) as base, Artifact(donor_path) as donor:
        parts = deepcopy(donor.directory.components)
        parts[change]["config"]["incompatible"] = True
        donor.directory = replace(donor.directory, components=parts)
        with pytest.raises(ValueError):
            attach(base, donor, tmp_path / "refused.ninfer")
    assert not (tmp_path / "refused.ninfer").exists()


def test_existing_output_not_overwritten(tmp_path):
    base_path, donor_path = inputs(tmp_path)
    out = tmp_path / "keep.ninfer"
    out.write_bytes(b"keep")
    with Artifact(base_path) as base, Artifact(donor_path) as donor:
        with pytest.raises(FileExistsError):
            attach(base, donor, out)
    assert out.read_bytes() == b"keep"


def test_readback_rejects_corrupted_payload(tmp_path):
    base_path, donor_path = inputs(tmp_path)
    out = tmp_path / 'result.ninfer'
    with Artifact(base_path) as base, Artifact(donor_path) as donor:
        attach(base, donor, out)
        with Artifact(out) as result:
            offset = result.payload_offset + result.by_id['weight'].offset
            with out.open('r+b') as stream:
                stream.seek(offset)
                stream.write(b'\xff')
            sources = {obj.id: (base, obj.id) for obj in base.objects}
            sources['mtp-import/weight'] = (donor, 'weight')
            with pytest.raises(ValueError, match='output payload mismatch'):
                verify_copy(result, out, sources)
