#!/usr/bin/env python3
"""Carry the pod's compiler cache between rentals through the private bucket WaveCut/ninfer-cache.

usage: ccache_sync.py pull|push [--dir /workspace/ccache] [--arch 86]

The cache is one uncompressed tar (ccache already compresses its entries, and the Hub's chunk
deduplication then stores only what changed since the last push) under
`ccache/cuda<nvcc release>-sm<arch>-<distribution>.tar`. ccache hashes every compiler command
line and input, so a shared key only groups objects that can be reused. The credential is the
token `hf_archive.py --stage-auth` leaves at /run/ninfer-hf/token. A missing credential or remote
cache is reported and is not an error: the build then simply compiles cold.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

BUCKET = "WaveCut/ninfer-cache"
TOKEN = Path("/run/ninfer-hf/token")


def key(arch: str) -> str:
    release = re.search(r"release (\d+\.\d+)", subprocess.run(
        ["nvcc", "--version"], capture_output=True, text=True, check=True).stdout)
    if release is None:
        raise RuntimeError("nvcc --version names no release")
    distribution = "unknown"
    os_release = Path("/etc/os-release")
    if os_release.exists():
        fields = dict(line.split("=", 1) for line in os_release.read_text().splitlines() if "=" in line)
        distribution = (fields.get("ID", "x") + fields.get("VERSION_ID", "")).replace('"', "")
    return f"ccache/cuda{release.group(1)}-sm{arch}-{distribution}-{platform.machine()}.tar"


def api():
    if not TOKEN.is_file():
        return None
    os.environ.setdefault("HF_TOKEN_PATH", str(TOKEN))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # keeps the build log readable
    from huggingface_hub import HfApi

    client = HfApi()
    if client.bucket_info(BUCKET).private is not True:
        raise RuntimeError("the cache bucket is not private")
    return client


def report(**fields) -> None:
    print(json.dumps(fields), flush=True)


def pull(directory: Path, remote: str) -> None:
    client = api()
    if client is None:
        report(ccache="cold", reason="no staged Hub credential")
        return
    if directory.exists() and any(directory.iterdir()):
        report(ccache="kept", reason="the local cache is not empty", path=str(directory))
        return
    started = time.monotonic()
    archive = directory.parent / (directory.name + ".tar")
    try:
        client.download_bucket_files(BUCKET, files=[(remote, archive)], raise_on_missing_files=True)
    except Exception as error:
        if "NotFound" not in type(error).__name__:
            raise
        report(ccache="cold", reason="no remote cache", remote=remote)
        return
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as package:
        for member in package:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                raise RuntimeError(f"unexpected cache archive member: {member.name}")
        package.extractall(directory, filter="data")
    report(ccache="pulled", remote=remote, bytes=archive.stat().st_size,
           seconds=round(time.monotonic() - started, 1))
    archive.unlink()


def push(directory: Path, remote: str) -> None:
    client = api()
    if client is None:
        report(ccache="not pushed", reason="no staged Hub credential")
        return
    if not directory.is_dir():
        report(ccache="not pushed", reason="no local cache", path=str(directory))
        return
    subprocess.run(["ccache", "--cleanup"], check=False, stdout=subprocess.DEVNULL,
                   env={**os.environ, "CCACHE_DIR": str(directory)})
    started = time.monotonic()
    archive = directory.parent / (directory.name + ".tar")
    with tarfile.open(archive, "w") as package:
        for path in sorted(directory.rglob("*")):
            # Lock and temporary files are host-local state.
            if path.is_file() and not path.name.endswith((".lock", ".tmp")) and "/tmp/" not in str(path):
                package.add(path, arcname=str(path.relative_to(directory)), recursive=False)
    client.batch_bucket_files(BUCKET, add=[(archive, remote)])
    entry, = client.get_bucket_paths_info(BUCKET, [remote])
    if entry.size != archive.stat().st_size:
        raise RuntimeError("the pushed cache's size differs on the Hub")
    report(ccache="pushed", remote=remote, bytes=entry.size,
           seconds=round(time.monotonic() - started, 1))
    archive.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("pull", "push"))
    parser.add_argument("--dir", type=Path, default=Path("/workspace/ccache"))
    parser.add_argument("--arch", default="86")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("the compiler cache is synchronized on the remote host")
    remote = key(args.arch)
    (pull if args.action == "pull" else push)(args.dir, remote)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # A cache problem must never fail the build that called it.
        report(ccache="failed", error=type(error).__name__, detail=str(error)[:200])
