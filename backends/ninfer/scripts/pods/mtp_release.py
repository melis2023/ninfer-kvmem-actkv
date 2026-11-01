#!/usr/bin/env python3
"""Build the three published NInfer models with MTP and stage them in the existing private bucket."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from huggingface_hub import HfApi, get_token, hf_hub_download, hf_hub_url
from huggingface_hub.utils import get_session

from scripts.pods.attach_mtp import attach
from scripts.pods.hf_artifact_audit import read_range
from tools.artifact.framing import HEADER, MAGIC, PAYLOAD_ALIGNMENT
from tools.artifact.layouts import align_up
from tools.artifact.reader import Artifact
from tools.artifact.schema import decode_directory

DONOR = "WaveCut/Qwen3.8-Flash-Next-GGUF-Q2_0-MTP-NInfer-exp-20261008"
BUCKET = "WaveCut/ninfer-cache"
BUCKET_PREFIX = "flash-next-mtp-20261008"
BASES = {
    "q2": "WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-NInfer-v3",
    "iq3": "WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-NInfer-v3",
    "coder": "WaveCut/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M-NInfer-v3",
}
CHUNK = 8 << 20


def log(stage, **values):
    print(json.dumps({"stage": stage, **values}), flush=True)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RemoteDonor:
    """Read selected objects of one pinned artifact, retaining only those objects on the Pod."""
    def __init__(self, repo, record, cache):
        self.url = hf_hub_url(repo["repo"], record["file"], revision=repo["revision"])
        self.token = get_token()
        magic, length, self.artifact_id = HEADER.unpack(read_range(self.url, 0, HEADER.size, self.token))
        if magic != MAGIC or not 0 < length <= 128 << 20:
            raise ValueError("invalid donor header")
        raw = read_range(self.url, HEADER.size, HEADER.size + length, self.token)
        self.directory = decode_directory(raw, entry_name=record["file"])
        self.payload_offset = align_up(HEADER.size + length, PAYLOAD_ALIGNMENT)
        self.objects = self.directory.objects
        self.by_id = {obj.id: obj for obj in self.objects}
        if (len(self.directory.files) != 1 or self.artifact_id.hex() != record["artifact_id"] or
                self.payload_offset + self.directory.payload_bytes != record["bytes"]):
            raise ValueError("donor differs from the pinned audited artifact")
        self.cache = cache / self.artifact_id.hex()
        self.cache.mkdir(parents=True, exist_ok=True)

    def iter_object(self, object_id):
        obj = self.by_id[object_id]
        local = self.cache / hashlib.sha256(object_id.encode()).hexdigest()
        checksum = local.with_suffix(".sha256")
        if local.exists() and checksum.exists():
            if local.stat().st_size != obj.bytes or file_sha(local) != checksum.read_text().strip():
                raise ValueError("cached donor object is corrupt")
            with local.open("rb") as stream:
                while chunk := stream.read(CHUNK):
                    yield chunk
            return
        temporary = local.with_suffix(".partial")
        digest = hashlib.sha256()
        cursor = self.payload_offset + obj.offset
        end = cursor + obj.bytes
        attempts = 0
        with temporary.open("wb") as output:
            while cursor < end:
                headers = {"Range": f"bytes={cursor}-{end - 1}"}
                if self.token:
                    headers["Authorization"] = f"Bearer {self.token}"
                try:
                    with get_session().stream("GET", self.url, headers=headers,
                                              follow_redirects=True, timeout=120) as response:
                        expected = f"bytes {cursor}-{end - 1}/"
                        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(expected):
                            raise ValueError("donor returned an unexpected byte range")
                        for chunk in response.iter_bytes(chunk_size=CHUNK):
                            if cursor + len(chunk) > end:
                                raise ValueError("donor exceeded requested range")
                            output.write(chunk)
                            digest.update(chunk)
                            cursor += len(chunk)
                            yield chunk
                        if cursor != end:
                            raise OSError("truncated donor range")
                except ValueError:
                    raise
                except Exception:
                    attempts += 1
                    if attempts >= 4:
                        raise RuntimeError("donor transfer failed after four attempts") from None
                    time.sleep(attempts * 2)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(local)
        checksum.write_text(digest.hexdigest() + "\n")


def build_variant(label, catalog, donor, root, receipts, api, local_only=False):
    repo = catalog[BASES[label]]
    record, = [a for a in repo["artifacts"] if "text" in a.get("components", [])]
    receipt_path = receipts / f"{label}.json"
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if local_only:
            if file_sha(Path(saved["model_path"])) != saved["sha256"]:
                raise ValueError("retained local candidate differs from its receipt")
            log("already_assembled", variant=label)
            return
        entry, = api.get_bucket_paths_info(BUCKET, [saved["remote_path"]])
        if entry.xet_hash != saved["xet_hash"] or entry.size != saved["bytes"]:
            raise ValueError("previous upload receipt does not match the bucket")
        log("already_staged", variant=label, xet_hash=saved["xet_hash"])
        return
    workspace = root / label
    source_dir, output_dir = workspace / "source", workspace / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(root).free
    source_expected = source_dir / record["file"]
    output_expected = output_dir / (Path(record["file"]).stem + "-MTP.ninfer")
    required = (0 if source_expected.exists() else record["bytes"])
    required += 0 if output_expected.exists() else record["bytes"] + (3 << 30)
    if available < required + (3 << 30):
        raise RuntimeError("insufficient remote storage for this model and its output")
    log("download_base", variant=label, bytes=record["bytes"])
    source = Path(hf_hub_download(repo["repo"], record["file"], revision=repo["revision"], local_dir=source_dir))
    if source.stat().st_size != record["bytes"] or file_sha(source) != record["file_sha256"]:
        raise ValueError("base artifact does not match its audited SHA-256")
    output = output_dir / (source.stem + "-MTP.ninfer")
    report_path = Path(str(output) + ".conversion.json")
    reused_output = output.exists()
    if reused_output:
        # A report is written only after all objects and metadata have passed readback.
        if not report_path.exists():
            raise RuntimeError("unverified output exists; inspect before retrying")
        report = json.loads(report_path.read_text())
        with Artifact(output) as result:
            if result.artifact_id.hex() != report["artifact_id"]:
                raise ValueError("existing report belongs to another output")
            for obj in result.objects:
                digest = hashlib.sha256()
                for chunk in result.iter_object(obj.id):
                    digest.update(chunk)
                if digest.hexdigest() != report["object_sha256"][obj.id]:
                    raise ValueError("existing output failed object verification")
    else:
        with Artifact(source) as base:
            if base.artifact_id.hex() != record["artifact_id"]:
                raise ValueError("base artifact identity differs from audit")
            report = attach(base, donor, output, provenance={
                "base_repo": repo["repo"], "base_revision": repo["revision"],
                "base_file_sha256": record["file_sha256"], "donor_repo": DONOR,
                "donor_revision": catalog[DONOR]["revision"]})
    digest = (file_sha(output) if reused_output or "file_sha256" not in report
              else report["file_sha256"])
    if "file_sha256" in report and digest != report["file_sha256"]:
        raise ValueError("retained artifact file differs from its verified report")
    if local_only:
        receipt = {"variant": label, "model_path": str(output), "sha256": digest,
                   "bytes": output.stat().st_size, "artifact_id": report["artifact_id"],
                   "mtp_bindings": report["mtp_bindings"],
                   "base_objects_verified": report["base_objects_verified"],
                   "mtp_objects_verified": report["mtp_objects_verified"],
                   "artifact_sha256_verified": True, "gpu_qualification": "pending"}
        shutil.copyfile(report_path, receipts / f"{label}.conversion.json")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        candidate = Path("/workspace/ninfer-work/mtp-candidates") / label
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / "download-verified.json").write_text(json.dumps(receipt, indent=2) + "\n")
        # All source objects passed output readback. The pinned public source can be downloaded again.
        shutil.rmtree(source_dir)
        log("MTP_LOCAL_CANDIDATE_VERIFIED", **receipt)
        return
    if api.bucket_info(BUCKET).private is not True:
        raise ValueError("staging bucket is not private")
    (output_dir / "README.md").write_text(
        f"# {label}: existing published NInfer model with MTP\n\n"
        f"Base: `{repo['repo']}` at `{repo['revision']}`.\n\n"
        "Every existing weight, Vision object, tokenizer resource and external IQ4 table descriptor "
        "is preserved. MTP objects are copied without requantization from the pinned donor recorded "
        "in the conversion report. All output object bytes passed SHA-256 readback.\n\n"
        "GPU qualification is pending. This private candidate is not a production release. "
        "The original model's license and restrictions continue to apply.\n")
    log("upload_bucket", variant=label, bytes=output.stat().st_size)
    prefix = f"{BUCKET_PREFIX}/{label}"
    api.batch_bucket_files(BUCKET, add=[(path, f"{prefix}/{path.name}")
                                      for path in sorted(output_dir.iterdir()) if path.is_file()])
    remote_path = f"{prefix}/{output.name}"
    uploaded, = api.get_bucket_paths_info(BUCKET, [remote_path])
    if uploaded.size != output.stat().st_size or not uploaded.xet_hash:
        raise ValueError("bucket artifact has the wrong size or no Xet content identity")
    receipt = {"variant": label, "bucket": BUCKET, "xet_hash": uploaded.xet_hash, "remote_path": remote_path,
               "sha256": digest, "bytes": uploaded.size, "artifact_id": report["artifact_id"],
               "mtp_bindings": report["mtp_bindings"], "base_objects_verified": report["base_objects_verified"],
               "mtp_objects_verified": report["mtp_objects_verified"], "gpu_qualification": "pending"}
    shutil.copyfile(report_path, receipts / f"{label}.conversion.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    api.batch_bucket_files(BUCKET, add=[(receipt_path, f"{prefix}/receipt.json")])
    # Only this task's source and output copies, after confirmed remote archival.
    shutil.rmtree(workspace)
    log("staged_and_local_copies_removed", **receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--variant", choices=list(BASES), action="append")
    parser.add_argument("--local-only", action="store_true",
                        help="retain the verified candidate on this GPU host for qualification")
    args = parser.parse_args()
    if sys.platform != "linux" or not str(args.root.resolve()).startswith("/workspace/"):
        raise RuntimeError("large artifact work must run on the Linux Pod under /workspace")
    args.root.mkdir(parents=True, exist_ok=True)
    args.receipts.mkdir(parents=True, exist_ok=True)
    catalog = {repo["repo"]: repo for repo in json.loads(args.audit.read_text())}
    donor_record, = catalog[DONOR]["artifacts"]
    donor = RemoteDonor(catalog[DONOR], donor_record, args.root / "mtp-cache")
    api = HfApi()
    for label in args.variant or BASES:
        build_variant(label, catalog, donor, args.root, args.receipts, api, args.local_only)
    log("MTP_ARTIFACTS_ASSEMBLED" if args.local_only else "MTP_ARTIFACTS_STAGED")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Hub and HTTP exception strings may contain signed URLs; preserve the stage and type only.
        log("failed", error_type=type(error).__name__, errno=getattr(error, "errno", None),
            reason=str(error) if type(error) in (ValueError, RuntimeError) else "diagnostic withheld")
        raise SystemExit(1) from None
