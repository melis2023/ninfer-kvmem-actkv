"""Preserve verified Linux release volumes in the private HF bucket before runner cleanup."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time

BUCKET = "WaveCut/ninfer-cache"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive(api, manifest, revision, run_id):
    name = manifest.name.removeprefix("SHA256SUMS-").removesuffix(".txt")
    if not re.fullmatch(r"ninfer-linux-x64-cuda(?:[0-9.]+-sm[0-9a-]+|12\.9-13\.4)-[0-9a-f]{12}", name):
        raise ValueError("Unexpected Linux artifact name")
    metadata = manifest.parent / name / "BUILD-INFO.json"
    info = json.loads(metadata.read_text())
    if info["source_revision"] != revision or info["platform"] != "linux-x64":
        raise ValueError("Artifact does not belong to this source revision")
    files = [(manifest, manifest.name), (metadata, "BUILD-INFO.json")]
    for line in manifest.read_text().splitlines():
        expected, filename = line.split("  ", 1)
        if not re.fullmatch(re.escape(name) + r"\.tar\.gz\.[0-9]{3}", filename):
            raise ValueError("Unexpected Linux archive volume")
        path = manifest.parent / filename
        if digest(path) != expected:
            raise ValueError("Local archive checksum mismatch")
        files.append((path, filename))
    if len(files) < 3:
        raise ValueError("Artifact contains no archive volumes")
    base = f"native-artifacts/v1/iamwavecut/ninfer-all/{run_id}/{name}"
    records = [{"name": name, "bytes": path.stat().st_size, "sha256": digest(path)}
               for path, name in files]
    api.batch_bucket_files(BUCKET, add=[(path, f"{base}/{name}") for path, name in files])
    with tempfile.TemporaryDirectory(dir=manifest.parent) as temporary:
        check = Path(temporary) / "verify"
        for record in records:
            api.download_bucket_files(BUCKET, files=[(f"{base}/{record['name']}", check)],
                                      raise_on_missing_files=True)
            if check.stat().st_size != record["bytes"] or digest(check) != record["sha256"]:
                raise ValueError("HF artifact round-trip checksum mismatch")
            check.unlink()
    receipt = manifest.parent / f"HF-{name}.json"
    receipt.write_text(json.dumps({"schema": 1, "run_id": run_id, "bucket": BUCKET,
                                  "prefix": base, "verified_at": time.time(), "files": records}, indent=2) + "\n")
    api.batch_bucket_files(BUCKET, add=[(receipt, f"{base}/COMPLETE.json")])
    print(json.dumps({"state": "verified", "prefix": base}), flush=True)


def main():
    workflow = os.environ.get("GITHUB_WORKFLOW_REF", "").split("@", 1)
    allowed = ("iamwavecut/ninfer-all/.github/workflows/native-release.yml",
               "iamwavecut/ninfer-all/.github/workflows/linux-cache-bootstrap.yml")
    if (len(workflow) != 2 or workflow[0] not in allowed or not workflow[1].startswith("refs/heads/")
            or os.environ.get("GITHUB_JOB") not in ("linux", "universal-linux")):
        raise ValueError("Only a native Linux artifact job may archive release volumes")
    from huggingface_hub import HfApi
    api = HfApi()
    if not api.bucket_info(BUCKET).private:
        raise ValueError("Artifact bucket must be private")
    manifest, = Path("dist").glob("SHA256SUMS-ninfer-linux-*.txt")
    archive(api, manifest, os.environ["GITHUB_SHA"], os.environ["GITHUB_RUN_ID"])


if __name__ == "__main__":
    done = threading.Event()
    def heartbeat():
        while not done.wait(60):
            print("Linux artifact upload or checksum verification is still running.", flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        main()
    except Exception as error:
        # Storage errors may contain signed URLs.
        print(f"::error::Linux artifact backup failed ({type(error).__name__})", flush=True)
        sys.exit(1)
    finally:
        done.set()
