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
import json
import os
import struct
import urllib.request
from pathlib import Path


def _url(repo: str, revision: str, file: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{file}"


def _request(url: str, start: int | None = None, end: int | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ninfer-fetch-slice"})
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end - 1}")
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if start is not None and len(data) != end - start:
        raise RuntimeError(f"{url}: expected {end - start} bytes, got {len(data)}")
    return data


class Shard:
    def __init__(self, repo: str, revision: str, file: str):
        self.url = _url(repo, revision, file)
        size = struct.unpack("<Q", _request(self.url, 0, 8))[0]
        self.header = json.loads(_request(self.url, 8, 8 + size))
        self.header.pop("__metadata__", None)
        self.data_start = 8 + size

    def tensor_bytes(self, name: str) -> bytes:
        begin, end = self.header[name]["data_offsets"]
        return _request(self.url, self.data_start + begin, self.data_start + end)

    def row_bytes(self, name: str, row: int) -> bytes:
        info = self.header[name]
        rows, width = info["shape"]
        element = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1}[info["dtype"]]
        if not 0 <= row < rows:
            raise IndexError(f"{name}: row {row} of {rows}")
        begin = self.data_start + info["data_offsets"][0] + row * width * element
        return _request(self.url, begin, begin + width * element)


def _write_safetensors(path: Path, tensors: dict[str, tuple[dict, bytes]]) -> None:
    header, offset = {}, 0
    for name, (info, data) in tensors.items():
        header[name] = {"dtype": info["dtype"], "shape": info["shape"],
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with open(path, "wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        for _, data in tensors.values():
            file.write(data)


def fetch(repo: str, revision: str, out: Path, names: list[str], prefixes: list[str],
          excludes: list[str] = ()) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_bytes(_request(_url(repo, revision, "config.json")))
    index = json.loads(_request(_url(repo, revision, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    selected = sorted(n for n in weight_map
                      if (n in names or any(n.startswith(p) for p in prefixes))
                      and not any(e in n for e in excludes))
    missing = set(names) - set(weight_map)
    if missing:
        raise KeyError(f"not in the checkpoint: {sorted(missing)}")
    by_shard: dict[str, list[str]] = {}
    for name in selected:
        by_shard.setdefault(weight_map[name], []).append(name)
    local_map = {}
    for file, tensors in by_shard.items():
        shard = Shard(repo, revision, file)
        payload = {name: (shard.header[name], shard.tensor_bytes(name)) for name in tensors}
        _write_safetensors(out / file, payload)
        local_map.update({name: file for name in tensors})
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": local_map}, indent=1))
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
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[],
                        help="skip tensors whose name contains this (e.g. ngram_embedding)")
    parser.add_argument("--ngram-rows", help="comma-separated table rows to fetch")
    parser.add_argument("--ngram-pattern",
                        default="model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{}.weight")
    args = parser.parse_args()
    if args.name or args.prefix:
        fetched = fetch(args.repo, args.revision, args.out, args.name, args.prefix, args.exclude)
        print(f"fetched {len(fetched)} tensors into {args.out}")
    if args.ngram_rows:
        rows = [int(r) for r in args.ngram_rows.split(",")]
        fetch_rows(args.repo, args.revision, args.out / "ngram_rows.npz", args.ngram_pattern, rows)
        print(f"fetched {len(set(rows))} n-gram rows")


if __name__ == "__main__":
    main()
