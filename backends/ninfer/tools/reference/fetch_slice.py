"""Fetch a slice of a safetensors checkpoint from the Hugging Face Hub by HTTP range requests.

Downloads only the named tensors (or every tensor whose name starts with a given prefix) into a
local directory laid out like the checkpoint (config.json, model.safetensors.index.json and one
safetensors file per source shard holding just the fetched tensors), so tools/reference can run a
prefix of blocks without the full download. --ngram-rows also fetches single rows of a 2-D
sharded table (Qwen3.8-Flash-Next's n-gram embedding) into a sparse .npz {row id: bytes}.

    python -m tools.reference.fetch_slice --repo Qwen/Qwen3.8-Flash-Next --out DIR \\
        --prefix model.language_model.layers.0. --prefix lm_head. --name model.language_model.embed_tokens.weight
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _url(repo: str, revision: str, file: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{file}"


def _open_response(url: str, start: int | None = None, end: int | None = None):
    request = urllib.request.Request(url, headers={"User-Agent": "ninfer-fetch-slice"})
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end - 1}")
    response = urllib.request.urlopen(request, timeout=120)
    if start is not None:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(f"bytes {start}-{end - 1}/"):
            response.close()
            raise RuntimeError(f"{url}: server did not honor the requested byte range")
    return response


def _request(url: str, start: int | None = None, end: int | None = None) -> bytes:
    with _open_response(url, start, end) as response:
        data = response.read() if start is None else response.read(end - start + 1)
    if start is not None and len(data) != end - start:
        raise RuntimeError(f"{url}: expected {end - start} bytes, got {len(data)}")
    return data


def _range_chunks(url: str, start: int, end: int, chunk_bytes: int):
    # One HTTP request per tensor; chunking bounds RAM without adding an RTT per chunk.
    with _open_response(url, start, end) as response:
        left = end - start
        while left:
            size = min(left, chunk_bytes)
            block = response.read(size)
            if len(block) != size:
                raise RuntimeError(f"{url}: truncated tensor range with {left} bytes remaining")
            yield block
            left -= size
        if response.read(1):
            raise RuntimeError(f"{url}: tensor range exceeds the requested length")


class Shard:
    def __init__(self, repo: str, revision: str, file: str):
        self.url = _url(repo, revision, file)
        size = struct.unpack("<Q", _request(self.url, 0, 8))[0]
        self.header = json.loads(_request(self.url, 8, 8 + size))
        self.header.pop("__metadata__", None)
        self.data_start = 8 + size

    def row_bytes(self, name: str, row: int) -> bytes:
        info = self.header[name]
        rows, width = info["shape"]
        element = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}[info["dtype"]]
        if not 0 <= row < rows:
            raise IndexError(f"{name}: row {row} of {rows}")
        begin = self.data_start + info["data_offsets"][0] + row * width * element
        return _request(self.url, begin, begin + width * element)


def _write_safetensors(path: Path, shard: Shard, names: list[str], chunk_bytes: int) -> dict:
    """Copy selected tensor ranges with bounded memory and hash the exact stored bytes."""
    header, offset = {}, 0
    for name in names:
        info = shard.header[name]
        begin, end = info["data_offsets"]
        header[name] = {"dtype": info["dtype"], "shape": info["shape"],
                        "data_offsets": [offset, offset + end - begin]}
        offset += end - begin
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(path.suffix + ".partial")
    file_digest, tensors = hashlib.sha256(), {}
    with partial.open("xb") as file:
        prefix = struct.pack("<Q", len(encoded)) + encoded
        file.write(prefix)
        file_digest.update(prefix)
        for name in names:
            digest = hashlib.sha256()
            begin, end = shard.header[name]["data_offsets"]
            for block in _range_chunks(shard.url, shard.data_start + begin,
                                       shard.data_start + end, chunk_bytes):
                file.write(block)
                file_digest.update(block)
                digest.update(block)
            tensors[name] = {"bytes": end - begin, "sha256": digest.hexdigest()}
    partial.replace(path)
    return {"bytes": len(prefix) + offset, "sha256": file_digest.hexdigest(), "tensors": tensors}


def fetch(repo: str, revision: str, out: Path, names: list[str], prefixes: list[str],
          excludes: list[str] = (), *, chunk_bytes: int = 8 << 20,
          workers: int = 1) -> list[str]:
    if chunk_bytes <= 0 or not 1 <= workers <= 32:
        raise ValueError("chunk_bytes must be positive and workers must be 1..32")
    config = _request(_url(repo, revision, "config.json"))
    index = json.loads(_request(_url(repo, revision, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    selected = sorted(n for n in weight_map
                      if (n in names or any(n.startswith(p) for p in prefixes))
                      and not any(e in n for e in excludes))
    missing = set(names) - set(weight_map)
    if missing:
        raise KeyError(f"not in the checkpoint: {sorted(missing)}")
    if not selected:
        raise ValueError("no checkpoint tensors match the requested selection")
    out.mkdir(parents=True, exist_ok=True)
    for file in ("config.json", "model.safetensors.index.json", "fetch-manifest.json"):
        if (out / file).exists():
            raise FileExistsError(out / file)
    by_shard: dict[str, list[str]] = {}
    for name in selected:
        by_shard.setdefault(weight_map[name], []).append(name)
    local_map, files = {}, {}
    def transfer(item):
        file, tensors = item
        shard = Shard(repo, revision, file)
        (out / file).parent.mkdir(parents=True, exist_ok=True)
        result = _write_safetensors(out / file, shard, tensors, chunk_bytes)
        print(json.dumps({"fetched": file, "bytes": result["bytes"]}), flush=True)
        return file, tensors, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(transfer, by_shard.items()))
    for file, tensors, result in results:
        files[file] = result
        local_map.update({name: file for name in tensors})
    (out / "config.json").write_bytes(config)
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": local_map}, indent=1))
    (out / "fetch-manifest.json").write_text(json.dumps({
        "repo": repo, "revision": revision, "files": files,
        "config_sha256": hashlib.sha256(config).hexdigest(),
    }, indent=2) + "\n")
    return selected


def fetch_rows(repo: str, revision: str, out: Path, shard_pattern: str, rows: list[int]) -> None:
    """Rows of a table stored as row shards named shard_pattern.format(s), into out (.npz)."""
    import numpy as np

    index = json.loads(_request(_url(repo, revision, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    first = shard_pattern.format(0)
    shard = Shard(repo, revision, weight_map[first])
    rows_per_shard = shard.header[first]["shape"][0]
    shards: dict[str, Shard] = {weight_map[first]: shard}
    values = {}
    for row in sorted(set(rows)):
        s, local = divmod(row, rows_per_shard)
        name = shard_pattern.format(s)
        file = weight_map[name]
        if file not in shards:
            shards[file] = Shard(repo, revision, file)
        values[str(row)] = np.frombuffer(shards[file].row_bytes(name, local), dtype=np.uint8)
    np.savez(out, **values)
    # The same rows for C++ readers: u64 count, then per row a u64 id and its raw bytes.
    with open(out.with_suffix(".bin"), "wb") as file:
        file.write(struct.pack("<Q", len(values)))
        for row in sorted(values, key=int):
            file.write(struct.pack("<Q", int(row)))
            file.write(values[row].tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel source shards (1..32); each transfer keeps bounded memory")
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[],
                        help="skip tensors whose name contains this (e.g. ngram_embedding)")
    parser.add_argument("--mtp", action="store_true",
                        help="fetch the mtp.* tensors only; requires a pinned 40-digit commit SHA")
    parser.add_argument("--ngram-rows", help="comma-separated table rows to fetch")
    parser.add_argument("--ngram-pattern",
                        default="model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{}.weight")
    args = parser.parse_args()
    if args.mtp:
        if args.name or args.prefix or args.exclude or args.ngram_rows:
            parser.error("--mtp selects only the MTP block; do not combine it with other selectors")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
            parser.error("--mtp requires --revision with a pinned 40-digit commit SHA")
        args.prefix = ["mtp."]
    if args.name or args.prefix:
        fetched = fetch(args.repo, args.revision, args.out, args.name, args.prefix, args.exclude,
                        workers=args.workers)
        print(f"fetched {len(fetched)} tensors into {args.out}")
    if args.ngram_rows:
        rows = [int(r) for r in args.ngram_rows.split(",")]
        fetch_rows(args.repo, args.revision, args.out / "ngram_rows.npz", args.ngram_pattern, rows)
        print(f"fetched {len(set(rows))} n-gram rows")


if __name__ == "__main__":
    main()
