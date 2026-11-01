#!/usr/bin/env python3
"""Bind the retained native-Q2 model to an FP8 table without requantizing its weights."""
from copy import deepcopy
import json
from pathlib import Path
import time

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec, binding_parts
from tools.artifact.writer import ArtifactWriter


def rebind(source: Path, table: Path, output: Path) -> None:
    report_path = Path(str(output) + ".conversion.json")
    if output.exists() or report_path.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    with Artifact(source) as model, Artifact(table) as rows:
        if "text" not in model.directory.components or "ngram/table" in model.directory.bindings:
            raise ValueError("expected a text model with an external n-gram table")
        if set(rows.directory.components) != {"ngram"}:
            raise ValueError("expected a standalone n-gram table")
        before = model.directory.components["ngram"]["config"]
        after = rows.directory.components["ngram"]["config"]
        unchanged = lambda config: {k: v for k, v in config.items()
                                    if k not in {"format", "table_sha256"}}
        if unchanged(before) != unchanged(after):
            raise ValueError("FP8 table geometry or hash constants differ from the model")
        if after["format"] != "fp8_e4m3fn_row_fp16":
            raise ValueError("expected an FP8 table with FP16 row scales")
        parts = binding_parts(rows.directory.bindings["ngram/table"], rows.by_id)
        if len(parts) != 1:
            raise ValueError("expected one whole-table binding")
        parent, begin, end = parts[0]
        stored = rows.by_id[parent]
        if (begin != 0 or end != after["rows"] * after["row_width"] or
                not isinstance(stored, TensorObject) or stored.format != after["format"] or
                stored.shape != (after["rows"], after["row_width"])):
            raise ValueError("FP8 table binding differs from its descriptor")
        report = json.loads(Path(str(source) + ".conversion.json").read_text())
        table_report = json.loads(Path(str(table) + ".conversion.json").read_text())
        if (report["artifact_id"] != model.artifact_id.hex() or
                table_report["artifact_id"] != rows.artifact_id.hex() or
                table_report["components"]["ngram"]["config"] != after):
            raise ValueError("conversion report belongs to another artifact")
        components = deepcopy(model.directory.components)
        components["ngram"]["config"] = deepcopy(after)
        transformation = {"type": "replace_external_ngram_descriptor",
                          "parent_artifact_id": model.artifact_id.hex(),
                          "table_artifact_id": rows.artifact_id.hex()}
        provenance = {**model.directory.provenance, "ngram_rebinding": transformation}
        specs = [TensorSpec(o.id, o.shape, o.format, o.layout, o.divisors)
                 if isinstance(o, TensorObject) else ResourceSpec(o.id, o.bytes, o.encoding)
                 for o in model.objects]
        with ArtifactWriter(output, specs, components=components,
                            bindings=model.directory.bindings, uses=model.directory.uses,
                            metadata={**model.directory.metadata, "name": output.stem},
                            provenance=provenance) as writer:
            for obj in model.objects:
                writer.write_object(obj.id, model.iter_object(obj.id))
            report.update(components=components, name=output.stem, output=str(output),
                          artifact_id=writer.artifact_id.hex(), provenance=provenance,
                          transformation=transformation, payload_bytes=writer.directory.payload_bytes,
                          files=[{"path": str(output),
                                  "payload_bytes": writer.directory.payload_bytes}])
        report["seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print("FP8_TABLE_MODEL_BOUND " + json.dumps({
            "artifact": output.name, "artifact_id": report["artifact_id"],
            "bytes": output.stat().st_size, "seconds": report["seconds"],
            "table_sha256": after["table_sha256"]}), flush=True)


if __name__ == "__main__":
    models = Path("/workspace/ninfer-work/models")
    rebind(models / "flash-next-native-q2-mtp.ninfer",
           models / "flash-next-fp8-table.ninfer",
           models / "flash-next-native-q2-mtp-fp8-table.ninfer")
