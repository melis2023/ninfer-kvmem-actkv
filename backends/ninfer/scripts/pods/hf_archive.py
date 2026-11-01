#!/usr/bin/env python3
"""Archive explicitly selected NInfer artifacts to private HF model repositories."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

TOKEN_PATH = "/run/ninfer-hf/token"


def stage_auth():
    """Pass the local HF credential through SSH stdin, outside jobs and upload folders."""
    from huggingface_hub import HfApi, get_token
    from harness import ssh_args, state

    token = get_token()
    if not token:
        raise RuntimeError("local Hugging Face authentication is unavailable")
    account = HfApi(token=token).whoami()["name"]
    command = ("umask 077; mkdir -p /run/ninfer-hf; chmod 700 /run/ninfer-hf; "
               f"cat > {TOKEN_PATH}; chmod 600 {TOKEN_PATH}")
    subprocess.run(["ssh", *ssh_args(state()), command], input=token, text=True, check=True,
                   stdout=subprocess.DEVNULL)
    print(json.dumps({"credential_staged_for": account, "transport": "SSH stdin"}), flush=True)


def private_repo(api, repo):
    if api.repo_exists(repo, repo_type="model"):
        if api.model_info(repo).private is not True:
            raise RuntimeError(f"refusing upload to a non-private repository: {repo}")
    else:
        api.create_repo(repo, repo_type="model", private=True)
    if api.model_info(repo).private is not True:
        raise RuntimeError(f"repository privacy was not confirmed: {repo}")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def model_card(repo, summary):
    text = (
        "---\nlibrary_name: ninfer\ntags:\n- ninfer\n- qwen3.8\n- experiment\n---\n\n"
        f"# {repo.split('/')[1]}\n\n"
        "Private intermediate NInfer v3 model artifact. Qualification is limited to the checks below.\n\n"
        f"Artifact: `{summary['artifact']}` ({summary['bytes']:,} bytes).\n\n"
        f"Recipe: `{summary['recipe']}`. Components: {', '.join(summary['components'])}.\n\n"
        "Verified: " + summary["qualification"] + "\n\n"
        "Limitations: " + summary["limitations"] + "\n\n"
        "The conversion report and artifact.json retain source revisions, formats and table identity.\n"
    )
    if companion := summary.get("ngram_companion"):
        text += (f"\nN-gram rows are external. Download `{companion['file']}` from "
                 f"[{companion['repo']}](https://huggingface.co/{companion['repo']}) "
                 "and pass its path with `--ngram-table`. The loader checks the table identity.\n")
    return text


def refresh_metadata(manifest_path: Path, output: Path):
    """Update qualifications of existing private archives without downloading model payloads."""
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    owner = api.whoami()["name"]
    results = []
    for item in json.loads(manifest_path.read_text())["models"]:
        repo = item["repo"]
        if repo.split("/")[0] != owner:
            raise RuntimeError("archive repository must belong to the authenticated account")
        before = api.model_info(repo, files_metadata=True)
        if before.private is not True:
            raise RuntimeError("metadata refresh requires an existing private repository")
        name = Path(item["artifact"]).name
        files = {f.rfilename: f for f in before.siblings}
        payload = files[name]
        summary = json.loads(Path(hf_hub_download(repo, "artifact.json", revision=before.sha)).read_text())
        if summary["artifact"] != name or summary["bytes"] != payload.size or not payload.lfs:
            raise RuntimeError("archive metadata does not identify its model payload")
        for field in ("qualification", "limitations", "ngram_companion"):
            summary[field] = item.get(field)
        expected = {"artifact.json": (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
                    "README.md": model_card(repo, summary).encode()}
        operations = []
        for path, content in expected.items():
            existing = Path(hf_hub_download(repo, path, revision=before.sha)).read_bytes()
            if existing != content:
                operations.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=content))
        if operations:
            commit = api.create_commit(repo, operations=operations, parent_commit=before.sha,
                                       commit_message="docs: record measured artifact qualifications")
            revision = commit.oid
        else:
            revision = before.sha
        after = api.model_info(repo, revision=revision, files_metadata=True)
        stored = {f.rfilename: f for f in after.siblings}[name]
        if after.private is not True or stored.size != payload.size or not stored.lfs or \
                stored.lfs.sha256 != payload.lfs.sha256:
            raise RuntimeError("archive payload or privacy changed during metadata refresh")
        for path, content in expected.items():
            if Path(hf_hub_download(repo, path, revision=revision)).read_bytes() != content:
                raise RuntimeError("archive metadata readback differs")
        results.append({"repo": repo, "private": True, "revision": revision, "file": name,
                        "bytes": stored.size, "sha256": stored.lfs.sha256,
                        "metadata_changed": bool(operations)})
        write_json(output / "verified.json", results)
        print(json.dumps({"metadata_verified": results[-1]}), flush=True)


def archive(manifest_path: Path, output: Path):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub._local_folder import read_upload_metadata
    from tools.artifact.reader import Artifact

    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"])
    api = HfApi()
    owner = api.whoami()["name"]
    results = []
    for item in manifest["models"]:
        repo = item["repo"]
        if repo.split("/")[0] != owner:
            raise RuntimeError("archive repository must belong to the authenticated account")
        source = root / item["artifact"]
        before = source.stat()
        report_path = Path(str(source) + ".conversion.json")
        report = json.loads(report_path.read_text())
        with Artifact(source) as artifact:
            if len(artifact.directory.files) != 1:
                raise RuntimeError("this archive manifest requires a single-file artifact")
            artifact_id = artifact.artifact_id.hex()
            if report["artifact_id"] != artifact_id:
                raise RuntimeError("conversion report belongs to another artifact")
            ngram_component = artifact.directory.components.get("ngram")
            ngram = ngram_component["config"] if ngram_component is not None else None
            summary = {
                "artifact_id": artifact_id,
                "artifact": source.name,
                "bytes": before.st_size,
                "recipe": artifact.directory.provenance.get("recipe"),
                "components": sorted(artifact.directory.components),
                "formats": dict(Counter(o.format for o in artifact.directory.objects
                                         if hasattr(o, "format"))),
                "ngram_format": ngram["format"] if ngram is not None else None,
                "ngram_table_sha256": ngram["table_sha256"] if ngram is not None else None,
                "contains_ngram_rows": "ngram/table" in artifact.directory.bindings,
                "source_layers": artifact.directory.provenance.get("source_layers"),
                "source_revisions": json.loads((root / "jobs/model-inputs.json").read_text()),
                "qualification": item["qualification"],
                "limitations": item["limitations"],
                "ngram_companion": item.get("ngram_companion"),
            }
        private_repo(api, repo)
        stage = root / "hf-archive" / repo.split("/")[1]
        stage.mkdir(parents=True, exist_ok=True)
        linked = stage / source.name
        if linked.exists():
            if not os.path.samefile(source, linked):
                raise RuntimeError("archive staging contains another artifact; preserve it and use a new repo")
        else:
            os.link(source, linked)  # no second 40-69 GB local copy
        write_json(stage / report_path.name, report)
        write_json(stage / "artifact.json", summary)
        (stage / "README.md").write_text(model_card(repo, summary))
        names = [source.name, report_path.name, "artifact.json", "README.md"]
        print(json.dumps({"archive_started": repo, "bytes": before.st_size}), flush=True)
        api.upload_large_folder(repo, stage, repo_type="model", private=True,
                                allow_patterns=names, num_workers=4, print_report_every=30)
        info = api.model_info(repo, files_metadata=True)
        if info.private is not True:
            raise RuntimeError("repository privacy changed during upload")
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError("source artifact changed during upload")
        files = {f.rfilename: f for f in info.siblings}
        remote = files[source.name]
        # Reuse the upload client's SHA-256 pass; do not scan the model a second time.
        cached = read_upload_metadata(stage, source.name)
        if (remote.size != before.st_size or not cached.sha256 or not remote.lfs or
                remote.lfs.sha256 != cached.sha256 or not cached.is_committed):
            raise RuntimeError("remote model size or SHA-256 differs from the uploaded artifact")
        for name in names[1:]:
            downloaded = hf_hub_download(repo, name, revision=info.sha)
            if Path(downloaded).read_bytes() != (stage / name).read_bytes():
                raise RuntimeError(f"remote metadata differs: {name}")
        result = {"repo": repo, "private": True, "revision": info.sha,
                  "file": source.name, "bytes": remote.size, "sha256": cached.sha256}
        results.append(result)
        write_json(output / "verified.json", results)
        print(json.dumps({"archive_verified": result}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-auth", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refresh-metadata", action="store_true",
                        help="update existing private model cards and qualifications without model downloads")
    parser.add_argument("--select", type=Path, help="write a subset of the canonical archive manifest")
    parser.add_argument("--only", action="append", default=[], help="artifact basename to include with --select")
    args = parser.parse_args()
    if args.select and args.manifest and args.only:
        manifest = json.loads(args.manifest.read_text())
        requested = set(args.only)
        selected = [item for item in manifest["models"] if Path(item["artifact"]).name in requested]
        if len(selected) != len(requested):
            raise ValueError("archive selection is missing or ambiguous")
        manifest["models"] = selected
        args.select.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.select, manifest)
    elif args.stage_auth:
        stage_auth()
    elif args.manifest and args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        if args.refresh_metadata:
            refresh_metadata(args.manifest, args.output)
        else:
            archive(args.manifest, args.output)
    else:
        parser.error("supply --stage-auth or both --manifest and --output")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # HTTP exception strings may carry signed URLs; retain the type/status without credentials.
        status = getattr(getattr(error, "response", None), "status_code", None)
        print(json.dumps({"archive_error": type(error).__name__, "http_status": status}), flush=True)
        raise SystemExit(1) from None
