from copy import deepcopy
import hashlib
import json

import pytest

from scripts.pods.fp8_table_model import rebind
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter


def inputs(root, *, different_hash=False, stale_report=False):
    model, table = root / "model.ninfer", root / "table.ninfer"
    config = {"rows": 2, "row_width": 160, "multipliers": [13, 17],
              "format": "gguf_iq4_nl", "table_sha256": "0" * 64}
    parts = {"text": {"config": {}, "resources": {"template": "template"}},
             "ngram": {"config": config}}
    with ArtifactWriter(model, [TensorSpec("weight", (2, 2), "bf16", "contiguous_le_v1"),
                               ResourceSpec("template", 5)], components=parts,
                        bindings={"text/weight": {"object": "weight"},
                                  "text/reordered": {"parts": [
                                      {"object": "weight", "range": [2, 4]},
                                      {"object": "weight", "range": [0, 2]}]}},
                        uses=[{"parameter": "text/weight", "input": "text/x",
                               "activation_policy": "AllowA8"}]) as writer:
        writer.write_object("weight", bytes(range(8)))
        writer.write_object("template", b"hello")
        identity = writer.artifact_id.hex()
    model.with_suffix(".ninfer.conversion.json").write_text(json.dumps({"artifact_id": identity}))
    table_config = deepcopy(config)
    payload = bytes(324)
    table_config.update(format="fp8_e4m3fn_row_fp16", table_sha256=hashlib.sha256(payload).hexdigest())
    if different_hash:
        table_config["multipliers"][0] += 1
    components = {"ngram": {"config": table_config}}
    with ArtifactWriter(table, [TensorSpec("table", (2, 160), "fp8_e4m3fn_row_fp16",
                                          "row_interleaved_v1")], components=components,
                        bindings={"ngram/table": {"object": "table"}}) as writer:
        writer.write_object("table", payload)
        identity = writer.artifact_id.hex()
    table.with_suffix(".ninfer.conversion.json").write_text(json.dumps({
        "artifact_id": "stale" if stale_report else identity, "components": components}))
    return model, table


def test_rebind_preserves_objects_resources_bindings_and_uses(tmp_path):
    model, table = inputs(tmp_path)
    output = tmp_path / "rebound.ninfer"
    rebind(model, table, output)
    with Artifact(model) as before, Artifact(output) as after, Artifact(table) as rows:
        assert before.artifact_id != after.artifact_id
        assert after.directory.components["ngram"] == rows.directory.components["ngram"]
        assert before.directory.components["text"] == after.directory.components["text"]
        assert before.directory.bindings == after.directory.bindings
        assert before.directory.uses == after.directory.uses
        assert before.objects == after.objects
        for obj in before.objects:
            assert before.read_object(obj.id) == after.read_object(obj.id)
        report = json.loads(output.with_suffix(".ninfer.conversion.json").read_text())
        assert report["artifact_id"] == after.artifact_id.hex()
        assert report["transformation"]["parent_artifact_id"] == before.artifact_id.hex()


@pytest.mark.parametrize("mode", ["different_hash", "stale_report"])
def test_refuses_incompatible_table_before_creating_output(tmp_path, mode):
    model, table = inputs(tmp_path, **{mode: True})
    output = tmp_path / "refused.ninfer"
    with pytest.raises(ValueError):
        rebind(model, table, output)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    model, table = inputs(tmp_path)
    output = tmp_path / "existing.ninfer"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        rebind(model, table, output)
    assert output.read_bytes() == b"preserve"
