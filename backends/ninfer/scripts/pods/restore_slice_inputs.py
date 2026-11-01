#!/usr/bin/env python3
"""Restore the pinned inputs needed for native-Q2 slice qualification from HF."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import os

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path("/workspace/ninfer-work")
MODELS = ROOT / "models"
NATIVE_REPO = "WaveCut/Qwen3.8-Flash-Next-native-Q2-MTP-NInfer-exp-20261008"
NATIVE_REVISION = "18d80d97a99c72a91e72ab69b89b08ab72f4aa76"
NATIVE_NAME = "flash-next-native-q2-mtp.ninfer"
NATIVE_SHA256 = "a11c94cfa757b243bc8f54737e99a6800cac3135625d88af2d2766dd9eedfda1"


def main():
    manifest = Path(os.environ["NINFER_JOB_DIR"]) / "inputs/model-inputs.json"
    revisions = json.loads(manifest.read_text())
    MODELS.mkdir(exist_ok=True)

    def fetch(repo, revision, name, directory):
        print(json.dumps({"download": repo, "file": name, "revision": revision}), flush=True)
        downloaded = Path(hf_hub_download(repo, name, revision=revision, local_dir=directory))
        print(json.dumps({"downloaded": name, "bytes": downloaded.stat().st_size}), flush=True)
        return downloaded

    def native():
        info = HfApi().model_info(NATIVE_REPO, revision=NATIVE_REVISION)
        if info.private is not True:
            raise RuntimeError("native archive privacy changed")
        source = fetch(NATIVE_REPO, NATIVE_REVISION, NATIVE_NAME, MODELS)
        report = fetch(NATIVE_REPO, NATIVE_REVISION, NATIVE_NAME + ".conversion.json", MODELS)
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != NATIVE_SHA256:
                raise RuntimeError("native archive digest differs from the verified upload")
        from tools.artifact.reader import Artifact
        with Artifact(source) as artifact:
            if artifact.artifact_id.hex() != json.loads(report.read_text())["artifact_id"]:
                raise RuntimeError("native conversion report mismatch")

    def gguf():
        repo = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
        revision = revisions[repo]
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        files = [item for item in info.siblings
                 if item.rfilename.endswith("Q2_0-00001-of-00002.gguf")]
        if len(files) != 1:
            raise RuntimeError("expected one pinned Q2_0 model shard")
        item = files[0]
        source = fetch(repo, revision, item.rfilename, MODELS)
        if source.stat().st_size != item.size:
            raise RuntimeError("GGUF download size mismatch")
        if item.lfs:
            with source.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != item.lfs.sha256:
                    raise RuntimeError("GGUF download digest mismatch")
        target = MODELS / Path(item.rfilename).name
        if source != target:
            if target.exists():
                raise RuntimeError("refusing to replace an existing GGUF input")
            os.link(source, target)  # keep HF's resume/cache path without duplicating the payload

    def config():
        repo = "Qwen/Qwen3.8-Flash-Next"
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "generation_config.json", "chat_template.jinja"):
            fetch(repo, revisions[repo], name, MODELS / "config")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(task) for task in (native, gguf, config)]
        for future in futures:
            future.result()
    retained = ROOT / "jobs/model-inputs.json"
    if retained.exists() and json.loads(retained.read_text()) != revisions:
        raise RuntimeError("existing input revisions differ")
    retained.write_text(json.dumps(revisions, indent=2) + "\n")
    print("SLICE_INPUTS_RESTORED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        number = getattr(error, "errno", None)
        print(json.dumps({"restore_error": type(error).__name__, "http_status": status,
                          "errno": number, "os_error": os.strerror(number) if number else None}),
              flush=True)
        raise SystemExit(1) from None
