#!/usr/bin/env python3
"""Remove only the explicitly abandoned Flash-Next experiments after preserving their reports."""
import argparse
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError

OLD_TABLE = "WaveCut/Qwen3.8-Flash-Next-IQ4-table-NInfer-exp-20261008"
PUBLIC_TABLE = "WaveCut/Qwen3.8-Flash-Next-ngram-table-NInfer-v3"
PUBLIC_FILE = "Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer"
TARGETS = {
    "WaveCut/Qwen3.8-Flash-Next-native-Q2-MTP-FP8-table-NInfer-exp-20261008": "FP8 table rejected by the user",
    "WaveCut/Qwen3.8-Flash-Next-FP8-table-NInfer-exp-20261008": "FP8 table rejected by the user",
    OLD_TABLE: "same row digest and descriptor as the retained public IQ4 table",
}
RETAINED = [
    "WaveCut/Qwen3.8-Flash-Next-GGUF-Q2_0-MTP-NInfer-exp-20261008",
    "WaveCut/Qwen3.8-Flash-Next-native-Q2-MTP-NInfer-exp-20261008",
    "WaveCut/Qwen3.8-Flash-Next-native-Q2-layers-1-4-NInfer-exp-20261008",
    "WaveCut/Qwen3.8-Flash-Next-native-Q2-layers-3-5-NInfer-exp-20261008",
]


def metadata_file(name):
    return name in ("README.md", "artifact.json", "LICENSE", "NOTICE") or name.endswith(".conversion.json")


def prepare(root, api, audit_path):
    audit = {r["repo"]: r for r in json.loads(audit_path.read_text())}
    private, = audit[OLD_TABLE]["artifacts"]
    public, = audit[PUBLIC_TABLE]["artifacts"]
    def directory(repo, artifact):
        path = audit_path.parent / repo.replace("/", "--") / (artifact["file"] + ".directory.json")
        return json.loads(path.read_text())
    private_dir, public_dir = directory(OLD_TABLE, private), directory(PUBLIC_TABLE, public)
    if private_dir["components"] != public_dir["components"]:
        raise ValueError("the IQ4 descriptors are not identical")
    records = []
    for repo, reason in TARGETS.items():
        info = api.model_info(repo, files_metadata=True)
        if not info.private or info.sha != audit[repo]["revision"]:
            raise ValueError(f"repository changed since audit: {repo}")
        dest = root / repo.replace("/", "--")
        dest.mkdir(parents=True, exist_ok=True)
        saved = []
        for entry in info.siblings:
            if not metadata_file(entry.rfilename):
                continue
            if entry.size is None or entry.size > 32 << 20:
                raise ValueError("unexpectedly large metadata; inspect before deleting")
            path = hf_hub_download(repo, entry.rfilename, revision=info.sha, local_dir=dest)
            saved.append(str(Path(path).relative_to(root)))
        if "README.md" not in [Path(p).name for p in saved] or "artifact.json" not in [Path(p).name for p in saved]:
            raise ValueError("experiment metadata was not preserved")
        records.append({"repo": repo, "revision": info.sha, "reason": reason, "saved_metadata": saved,
                        "artifact_bytes": sum(f.size or 0 for f in info.siblings if f.rfilename.endswith(".ninfer"))})
    plan = {"created": time.time(), "targets": records, "retained": RETAINED,
            "replacement_table": PUBLIC_TABLE, "replacement_file": PUBLIC_FILE,
            "table_row_sha256": public["table_sha256"]}
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


def apply(root, api):
    plan = json.loads((root / "plan.json").read_text())
    if {r["repo"] for r in plan["targets"]} != set(TARGETS):
        raise ValueError("cleanup plan differs from the fixed authorized targets")
    public = api.model_info(PUBLIC_TABLE, files_metadata=True)
    if public.private or not any(f.rfilename == PUBLIC_FILE for f in public.siblings):
        raise ValueError("public replacement table is unavailable")
    # Fix live companion links before deleting the duplicated private table.
    for repo in RETAINED:
        info = api.model_info(repo, files_metadata=True)
        for entry in info.siblings:
            if entry.rfilename not in ("README.md", "artifact.json"):
                continue
            if entry.size is None or entry.size > 1 << 20:
                raise ValueError("unexpected retained metadata size")
            cached = hf_hub_download(repo, entry.rfilename, revision=info.sha)
            text = Path(cached).read_text()
            if OLD_TABLE not in text:
                continue
            updated = text.replace(OLD_TABLE, PUBLIC_TABLE).replace("flash-next-iq4-table.ninfer", PUBLIC_FILE)
            dest = root / "retained-metadata" / repo.replace("/", "--")
            dest.mkdir(parents=True, exist_ok=True)
            (dest / (entry.rfilename + ".before")).write_text(text)
            path = dest / entry.rfilename
            path.write_text(updated)
            commit = api.upload_file(repo_id=repo, path_or_fileobj=path, path_in_repo=entry.rfilename,
                                     commit_message="Use the identical published IQ4 companion table")
            actual = hf_hub_download(repo, entry.rfilename, revision=commit.oid)
            if Path(actual).read_text() != updated:
                raise ValueError("companion metadata update did not verify")
    receipts = []
    for target in plan["targets"]:
        repo = target["repo"]
        info = api.model_info(repo)
        if not info.private or info.sha != target["revision"]:
            raise ValueError(f"refusing deletion after a repository change: {repo}")
        if not all((root / path).is_file() for path in target["saved_metadata"]):
            raise ValueError("saved experiment metadata is missing")
        api.delete_repo(repo_id=repo, repo_type="model")
        try:
            api.model_info(repo)
        except RepositoryNotFoundError:
            pass
        else:
            raise RuntimeError("repository still exists after deletion")
        receipts.append({**target, "deleted": time.time()})
        (root / "deleted.json").write_text(json.dumps(receipts, indent=2) + "\n")
        print(json.dumps({"deleted": repo, "artifact_bytes": target["artifact_bytes"]}), flush=True)
    remaining = {m.id for m in api.list_models(author="WaveCut")}
    if set(TARGETS) & remaining or not set(RETAINED).issubset(remaining):
        raise RuntimeError("final owner inventory does not match the intended cleanup")
    print(json.dumps({"verified_deleted": len(receipts),
                      "artifact_bytes": sum(r["artifact_bytes"] for r in receipts),
                      "retained_repositories": RETAINED}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "apply"))
    parser.add_argument("--root", type=Path, default=Path(".local/hf-cleanup-20261008"))
    parser.add_argument("--audit", type=Path, default=Path(".local/hf-artifact-audit/audit.json"))
    args = parser.parse_args()
    api = HfApi()
    if api.whoami()["name"] != "WaveCut":
        raise ValueError("expected the owning Hub account")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        prepare(args.root, api, args.audit)
    else:
        apply(args.root, api)


if __name__ == "__main__":
    main()
