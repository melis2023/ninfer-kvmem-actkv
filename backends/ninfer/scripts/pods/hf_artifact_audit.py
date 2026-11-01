#!/usr/bin/env python3
"""Inspect pinned NInfer artifact directories on HF without downloading weight payloads."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

from huggingface_hub import HfApi, get_token, hf_hub_url
from huggingface_hub.utils import get_session

from tools.artifact.framing import HEADER, MAGIC
from tools.artifact.schema import decode_directory


def read_range(url: str, begin: int, end: int, token: str | None) -> bytes:
    headers = {"Range": f"bytes={begin}-{end - 1}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with get_session().stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=90) as response:
        if response.status_code != 206:
            raise RuntimeError(f"range read returned HTTP {response.status_code}")
        expected = f"bytes {begin}-{end - 1}/"
        if not response.headers.get("Content-Range", "").startswith(expected):
            raise RuntimeError("server returned a different byte range")
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > end - begin:
                raise RuntimeError("range response exceeded the requested length")
        if len(data) != end - begin:
            raise RuntimeError("truncated range response")
        return bytes(data)


def inspect_repo(repo: str, output: Path, token: str | None) -> dict:
    info = HfApi(token=token).model_info(repo, files_metadata=True)
    result = {"repo": repo, "revision": info.sha, "private": info.private, "artifacts": []}
    destination = output / repo.replace("/", "--")
    destination.mkdir(parents=True, exist_ok=True)
    for entry in info.siblings:
        if not entry.rfilename.endswith(".ninfer"):
            continue
        record = {"file": entry.rfilename, "bytes": entry.size,
                  "file_sha256": entry.lfs.sha256 if entry.lfs else None}
        try:
            url = hf_hub_url(repo, entry.rfilename, revision=info.sha)
            magic, length, identity = HEADER.unpack(read_range(url, 0, HEADER.size, token))
            if magic != MAGIC or not 0 < length <= 128 << 20:
                raise RuntimeError("not a supported v3 directory")
            raw = read_range(url, HEADER.size, HEADER.size + length, token)
            directory = decode_directory(raw, entry_name=Path(entry.rfilename).name)
            mtp_bindings = [name for name in directory.bindings if name.startswith("mtp/")]
            descriptor = directory.components.get("ngram", {}).get("config", {})
            record.update(artifact_id=identity.hex(), components=sorted(directory.components),
                          has_mtp="mtp" in directory.components,
                          mtp_bindings=len(mtp_bindings),
                          mtp_config=directory.components.get("mtp"),
                          contains_ngram_rows="ngram/table" in directory.bindings,
                          table_sha256=descriptor.get("table_sha256"),
                          directory_bytes=length,
                          formats=sorted({obj.format for obj in directory.objects
                                          if hasattr(obj, "format")}))
            (destination / (Path(entry.rfilename).name + ".directory.json")).write_bytes(raw)
        except Exception as error:
            # HTTP exceptions may include signed CDN URLs. Keep credentials and URLs out of logs.
            record["error"] = type(error).__name__
            if isinstance(error, RuntimeError):
                record["reason"] = str(error)
        result["artifacts"].append(record)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", required=True)
    parser.add_argument("--name-contains", default="flashnext")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    token = get_token()
    api = HfApi(token=token)
    if not token or api.whoami()["name"].casefold() != args.author.casefold():
        raise RuntimeError("authenticated owner is required to include private repositories")
    key = re.sub(r"[^a-z0-9]", "", args.name_contains.casefold())
    repos = sorted(model.id for model in api.list_models(author=args.author)
                   if "ninfer" in model.id.casefold()
                   and key in re.sub(r"[^a-z0-9]", "", model.id.casefold()))
    if not repos:
        raise RuntimeError("no matching repositories returned")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "repositories.json").write_text(json.dumps(repos, indent=2) + "\n")
    print(json.dumps({"repositories": repos}), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda repo: inspect_repo(repo, args.output, token), repos))
    (args.output / "audit.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    if any(artifact.get("error") for repo in results for artifact in repo["artifacts"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
