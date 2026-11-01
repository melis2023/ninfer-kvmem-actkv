#!/usr/bin/env python3
"""Convert retained layer slices and compare every stored object with the full artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts

ROOT = Path("/workspace/ninfer-work")
MODELS = ROOT / "models"
BASELINE = MODELS / "flash-next-native-q2-mtp.ninfer"
GGUF = MODELS / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
GGUF_TABLE = MODELS / "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"


def original(name, begin):
    if name.startswith("text/layers/"):
        parts = name.split("/")
        parts[2] = str(int(parts[2]) + begin)
        return "/".join(parts)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", choices=("3..5", "1..4"), default="3..5")
    args = parser.parse_args()
    begin, end = map(int, args.layers.split(".."))
    output = MODELS / f"flash-next-native-q2-layers-{begin}-{end}.ninfer"
    has_ple = begin <= 1 < end
    with Artifact(BASELINE) as base:
        selected = [name for name in base.directory.bindings if name.startswith("text/") and
                    (not name.startswith("text/layers/") or
                     begin <= int(name.split("/")[2]) < end)]
        objects = {parent for name in selected for parent, _, _ in
                   binding_parts(base.directory.bindings[name], base.by_id)}
        estimated = sum(base.by_id[parent].bytes for parent in objects) + 16 * 1024**2
    if not output.exists():
        if shutil.disk_usage(MODELS).free < estimated + 512 * 1024**2:
            raise RuntimeError(f"slice needs {estimated} bytes plus a 512 MiB reserve")
        subprocess.run([
            sys.executable, "-m", "tools.convert", "--model", str(MODELS / "config"),
            "--recipe", "qwen3_8_flash_next_gsq_q2", "--layers", args.layers,
            "--components", "text", "--source", f"gguf={GGUF}", "--device", "cpu",
            "--rows-per-chunk", "65536", "--name", f"flash-next-layer-slice-{begin}-{end}",
            "--out", str(output),
            *(["--source", f"ngram={GGUF_TABLE}"] if has_ple else []),
        ], check=True)
    report = json.loads(Path(str(output) + ".conversion.json").read_text())
    with Artifact(output) as sliced, Artifact(BASELINE) as base:
        assert report["artifact_id"] == sliced.artifact_id.hex()
        assert sliced.directory.provenance["source_layers"] == list(range(begin, end))
        assert sliced.directory.provenance["recipe"] == "qwen3_8_flash_next_gsq_q2"
        assert set(sliced.directory.components) == ({"text", "ngram"} if has_ple else {"text"})
        assert sliced.directory.components["text"]["config"]["ple_layers"] == ([1 - begin] if has_ple else [])
        if has_ple:
            assert sliced.directory.components["ngram"]["config"] == base.directory.components["ngram"]["config"]
        assert {original(name, begin) for name in sliced.directory.bindings} == set(selected)
        checked = set()

        def compare_objects(left, right):
            key = (left, right)
            if key in checked:
                return
            a, b = sliced.by_id[left], base.by_id[right]
            assert (a.format, a.layout, a.shape, a.bytes, a.divisors) == (
                b.format, b.layout, b.shape, b.bytes, b.divisors), key
            # Equal-sized single-file objects yield the same bounded chunks.
            for actual, expected in zip(sliced.iter_object(left), base.iter_object(right), strict=True):
                assert actual == expected, key
            checked.add(key)

        for name, binding in sliced.directory.bindings.items():
            left = binding_parts(binding, sliced.by_id)
            right = binding_parts(base.directory.bindings[original(name, begin)], base.by_id)
            for (a, low, high), (b, first, last) in zip(left, right, strict=True):
                assert (low, high) == (first, last), name
                compare_objects(a, b)
        base_uses = {(u["parameter"], u["input"]): u for u in base.directory.uses}
        for use in sliced.directory.uses:
            other = base_uses[(original(use["parameter"], begin), original(use["input"], begin))]
            assert use["activation_policy"] == other["activation_policy"], use
            assert use.get("auxiliaries", {}).keys() == other.get("auxiliaries", {}).keys()
            for role, ref in use.get("auxiliaries", {}).items():
                compare_objects(ref["object"], other["auxiliaries"][role]["object"])
        for role, ref in sliced.directory.components["text"]["resources"].items():
            assert sliced.read_object(ref) == base.read_object(
                base.directory.components["text"]["resources"][role]), role
        result = {"source_layers": list(range(begin, end)), "bindings": len(sliced.directory.bindings),
                  "objects": len(checked), "bytes": output.stat().st_size, "ple": has_ple}
        print("LAYER_SLICE_BYTES_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
