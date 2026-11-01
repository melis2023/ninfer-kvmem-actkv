#!/usr/bin/env python3
"""Fetch one staged MTP artifact and verify its receipt on the GPU host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path("/workspace/ninfer-work")
BUCKET = "WaveCut/ninfer-cache"
PREFIX = "flash-next-mtp-20261008"


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or not ROOT.is_dir():
        raise RuntimeError("model transfers must run on the Linux Pod")
    variant = json.loads(args.config.read_text())["variant"]
    if variant not in ("q2", "iq3", "coder"):
        raise ValueError("unknown MTP candidate")
    job = Path(os.environ["NINFER_JOB_DIR"])
    table_dir = ROOT / "models"
    table = table_dir / "flash-next-iq4-table.ninfer"
    table_receipt = table_dir / "iq4-public-verified.json"
    if not table.exists() or not table_receipt.exists():
        table_dir.mkdir(exist_ok=True)
        print(json.dumps({"stage": "download_public_iq4_table"}), flush=True)
        downloaded = Path(hf_hub_download("WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3",
            "Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer",
            revision="6d7c2a69e8c5fd7813c864c6f6fa4be6f0e17cc3", local_dir=table_dir))
        expected = "261c5beae5dcec0a0dfe1914e15fc09c398099fdb08f2715379eb89867f6673a"
        if downloaded.stat().st_size != 28800142336 or file_sha(downloaded) != expected:
            raise ValueError("public IQ4 table failed its pinned SHA-256 check")
        if not table.exists():
            table.symlink_to(downloaded.name)
        elif file_sha(table) != expected:
            raise ValueError("retained table differs from the selected public IQ4 artifact")
        table_receipt.write_text(json.dumps({"sha256": expected, "bytes": 28800142336}) + "\n")
    destination = ROOT / "mtp-candidates" / variant
    destination.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    if api.bucket_info(BUCKET).private is not True:
        raise RuntimeError("candidate bucket must remain private")
    receipt_path = destination / "receipt.json"
    api.download_bucket_files(BUCKET, files=[(f"{PREFIX}/{variant}/receipt.json", receipt_path)],
                              raise_on_missing_files=True)
    receipt = json.loads(receipt_path.read_text())
    remote = receipt["remote_path"]
    if (receipt["variant"] != variant or receipt["bucket"] != BUCKET or
            not remote.startswith(f"{PREFIX}/{variant}/") or not remote.endswith(".ninfer")):
        raise ValueError("receipt does not identify the requested candidate")
    entry, = api.get_bucket_paths_info(BUCKET, [remote])
    if entry.size != receipt["bytes"] or entry.xet_hash != receipt["xet_hash"]:
        raise ValueError("bucket content differs from the assembly receipt")
    model = destination / Path(remote).name
    if not model.exists() and shutil.disk_usage(destination).free < receipt["bytes"] + (2 << 30):
        raise RuntimeError("candidate download needs more free disk space")
    print(json.dumps({"stage": "download", "variant": variant, "bytes": entry.size}), flush=True)
    api.download_bucket_files(BUCKET, files=[(remote, model),
        (remote + ".conversion.json", Path(str(model) + ".conversion.json"))],
        raise_on_missing_files=True)
    print(json.dumps({"stage": "verify_sha256", "variant": variant}), flush=True)
    if model.stat().st_size != receipt["bytes"] or file_sha(model) != receipt["sha256"]:
        raise ValueError("downloaded model differs from the verified assembly")
    report = json.loads(Path(str(model) + ".conversion.json").read_text())
    if (report["artifact_id"] != receipt["artifact_id"] or
            report["mtp_objects_verified"] != 28 or report["mtp_bindings"] != 1567 or
            set(report["components"]) != {"text", "vision", "ngram", "mtp"}):
        raise ValueError("candidate report has an unexpected component contract")
    verified = {**receipt, "model_path": str(model), "artifact_sha256_verified": True}
    (destination / "download-verified.json").write_text(json.dumps(verified, indent=2) + "\n")
    (job / "download-verified.json").write_text(json.dumps(verified, indent=2) + "\n")
    print(json.dumps({"stage": "MTP_CANDIDATE_VERIFIED", "variant": variant,
                      "bytes": receipt["bytes"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Network exceptions can include signed URLs. Keep them out of job logs.
        print(json.dumps({"error_type": type(error).__name__,
                          "reason": str(error) if type(error) in (ValueError, RuntimeError)
                          else "transfer failed; diagnostic withheld"}), flush=True)
        raise SystemExit(1) from None
