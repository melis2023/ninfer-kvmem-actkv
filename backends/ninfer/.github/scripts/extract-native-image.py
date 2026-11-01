"""Verify a Linux release artifact and prepare its existing Docker build context."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import tarfile


VARIANTS = {"12.9": ["70", "75", "80", "86", "89"],
            "13.4": ["75", "80", "86", "89", "120a"]}


def extract(source, destination, revision, cuda):
    architectures = VARIANTS[cuda]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid source revision")
    name = f"ninfer-linux-x64-cuda{cuda}-sm{'-'.join(architectures)}-{revision[:12]}"
    lines = (source / f"SHA256SUMS-{name}.txt").read_text().splitlines()
    if not lines:
        raise ValueError("Empty archive manifest")
    volumes = []
    for index, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        expected = f"{name}.tar.gz.{index:03d}"
        if not match or match[2] != expected:
            raise ValueError("Invalid archive volume order or name")
        path = source / expected
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != match[1]:
                raise ValueError(f"Checksum mismatch: {expected}")
        volumes.append(path)
    if set(source.iterdir()) != {*volumes, source / f"SHA256SUMS-{name}.txt"}:
        raise ValueError("Unexpected files in the release artifact")
    destination.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        streams = iter(stack.enter_context(p.open("rb")) for p in volumes)

        class Volumes:
            current = next(streams)

            def read(self, size):
                chunks = []
                while size and self.current is not None:
                    chunk = self.current.read(size)
                    if chunk:
                        chunks.append(chunk)
                        size -= len(chunk)
                    else:
                        self.current = next(streams, None)
                        if self.current is None:
                            break
                return b"".join(chunks)

        with tarfile.open(fileobj=Volumes(), mode="r|gz") as archive:
            archive.extractall(destination, filter="data")
    if list(destination.iterdir()) != [destination / name]:
        raise ValueError("Unexpected archive root")
    root = destination / name
    info = json.loads((root / "BUILD-INFO.json").read_text())
    if (info["source_revision"] != revision or info["modified_sources"]
            or info["cuda"] != cuda or info["platform"] != "linux-x64"
            or info["architectures"] != architectures or info.get("layout") != "shared-runtime-v1"):
        raise ValueError("Archive metadata does not match the requested source and variant")
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Checksum path escapes the archive")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Packaged file checksum mismatch: {relative}")
    commands = ('ninfer', 'ninfer-serve', 'ninfer_bench', 'ninfer-calibrate',
                'ninfer-perplexity', 'ninfer-ngram-profile')
    for required in ("Dockerfile", "packages/multi.txt", *commands,
                     *(f"help/{command}.txt" for command in commands),
                     f"backends/cuda{cuda.split('.')[0]}/libninfer-runtime.so",
                     f"backends/cuda{cuda.split('.')[0]}/architectures.txt"):
        if not (root / required).is_file():
            raise ValueError(f"Missing Docker input: {required}")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("revision")
    parser.add_argument("cuda", choices=VARIANTS)
    args = parser.parse_args()
    print(extract(args.source, args.destination, args.revision, args.cuda))
