#!/usr/bin/env python3
"""Restore the pinned IQ4_NL shard and write a byte-preserving standalone table artifact."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts

ROOT = Path("/workspace/ninfer-work")
MODELS = ROOT / "models"
OUTPUT = MODELS / "flash-next-iq4-table.ninfer"


def main():
    revisions = json.loads((Path(os.environ["NINFER_JOB_DIR"]) / "inputs/model-inputs.json").read_text())
    repo = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
    revision = revisions[repo]
    info = HfApi(token=False).model_info(repo, revision=revision, files_metadata=True)
    files = [item for item in info.siblings if item.rfilename.endswith("Q2_0-00002-of-00002.gguf")]
    if len(files) != 1:
        raise RuntimeError("expected one pinned IQ4_NL table shard")
    item = files[0]
    print(json.dumps({"download_table": item.rfilename, "revision": revision,
                      "bytes": item.size}), flush=True)
    source = Path(hf_hub_download(repo, item.rfilename, revision=revision, local_dir=MODELS, token=False))
    if source.stat().st_size != item.size:
        raise RuntimeError("table source size mismatch")
    with source.open("rb") as stream:
        if not item.lfs or hashlib.file_digest(stream, "sha256").hexdigest() != item.lfs.sha256:
            raise RuntimeError("table source digest mismatch")
    if not OUTPUT.exists():
        subprocess.run([sys.executable, "-m", "tools.convert", "--model", str(MODELS / "config"),
                        "--recipe", "qwen3_8_flash_next_gguf", "--components", "ngram",
                        "--source", f"ngram={source}", "--device", "cpu",
                        "--rows-per-chunk", "65536", "--name", "flash-next-iq4-table",
                        "--out", str(OUTPUT)], check=True)
    conversion = json.loads(Path(str(OUTPUT) + ".conversion.json").read_text())
    with Artifact(OUTPUT) as artifact:
        if set(artifact.directory.components) != {"ngram"}:
            raise RuntimeError("expected a standalone n-gram artifact")
        if conversion["artifact_id"] != artifact.artifact_id.hex():
            raise RuntimeError("table conversion report mismatch")
        descriptor = artifact.directory.components["ngram"]["config"]
        if descriptor["format"] != "gguf_iq4_nl":
            raise RuntimeError("expected stored IQ4_NL table rows")
        parts = binding_parts(artifact.directory.bindings["ngram/table"], artifact.by_id)
        if len(parts) != 1:
            raise RuntimeError("expected one complete table object")
        parent, begin, end = parts[0]
        if begin != 0 or end != descriptor["rows"] * descriptor["row_width"]:
            raise RuntimeError("expected a whole-table binding")
        digest = hashlib.sha256()
        for chunk in artifact.iter_object(parent):
            digest.update(chunk)
        if digest.hexdigest() != descriptor["table_sha256"]:
            raise RuntimeError("stored table bytes differ from the converter's source digest")
        print("STANDALONE_TABLE_BYTES_PASS " + json.dumps({"artifact": str(OUTPUT),
              "artifact_id": artifact.artifact_id.hex(), "bytes": OUTPUT.stat().st_size,
              "table_sha256": digest.hexdigest()}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        print(json.dumps({"table_error": type(error).__name__, "http_status": status}), flush=True)
        raise SystemExit(1) from None
