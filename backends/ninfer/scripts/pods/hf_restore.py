#!/usr/bin/env python3
"""Restore exact private model revisions from verified archive receipts."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download
from tools.artifact.reader import Artifact

ROOT = Path("/workspace/ninfer-work")


def restore(item):
    repo, revision, name = item["repo"], item["revision"], item["file"]
    if HfApi().model_info(repo, revision=revision).private is not True:
        raise RuntimeError("archive privacy differs from the verified receipt")
    print(json.dumps({"restore_started": repo, "revision": revision, "file": name}), flush=True)
    source = Path(hf_hub_download(repo, name, revision=revision, local_dir=ROOT / "models"))
    report_path = Path(hf_hub_download(repo, name + ".conversion.json", revision=revision,
                                     local_dir=ROOT / "models"))
    if source.stat().st_size != item["bytes"]:
        raise RuntimeError("restored artifact size differs from the verified receipt")
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
            raise RuntimeError("restored artifact digest differs from the verified receipt")
    with Artifact(source) as artifact:
        if artifact.artifact_id.hex() != json.loads(report_path.read_text())["artifact_id"]:
            raise RuntimeError("restored conversion report names another artifact")
    print(json.dumps({"restore_verified": name, "bytes": item["bytes"]}), flush=True)


def main():
    job = Path(os.environ["NINFER_JOB_DIR"])
    items = json.loads((job / "inputs/verified.json").read_text())
    if not items or len({item["file"] for item in items}) != len(items):
        raise ValueError("restore receipts must name distinct artifacts")
    (ROOT / "models").mkdir(exist_ok=True)
    required = sum(item["bytes"] for item in items if not (ROOT / "models" / item["file"]).exists())
    if shutil.disk_usage(ROOT / "models").free < required + 512 * 1024**2:
        raise RuntimeError("restore needs the artifact sizes plus a 512 MiB disk reserve")
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(restore, items):
            pass
    (job / "restored.json").write_text(json.dumps(items, indent=2) + "\n")
    print("HF_RESTORE_PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        print(json.dumps({"restore_error": type(error).__name__, "http_status": status}), flush=True)
        raise SystemExit(1) from None
