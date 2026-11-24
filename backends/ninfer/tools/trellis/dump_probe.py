"""Dumps a few real coded matrices of the NariQ artifact as raw planes for the GPU probe test.

``tests/ops/test_trellis_real.cpp`` runs `trellis_linear` on these bytes and compares it with the
host reference (`ops/trellis/trellis_reference.cpp`), so the GPU kernel is checked against real
tile data instead of synthetic tiles.

Usage: python -m tools.trellis.dump_probe <artifact> <destination> [tensor ...]
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from tools.artifact.reader import Artifact

DEFAULT_TENSORS = ("blk.0.ffn_up.weight", "blk.0.attn_qkv.weight", "output.band00.weight")


def slug(name: str) -> str:
    return name.replace("/", "_")


def dump(artifact_path: str | Path, destination: str | Path, tensors=DEFAULT_TENSORS) -> int:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = Artifact.open(artifact_path)
    descriptors = artifact.directory.metadata["trellis.descriptors"]
    for name in tensors:
        obj = artifact.object(name)
        base = destination / slug(name)
        base.mkdir(parents=True, exist_ok=True)
        (base / "payload.bin").write_bytes(artifact.read_object(name))
        (base / "s16.f32").write_bytes(artifact.read_object(name + "~s16"))
        (base / "pre.f32").write_bytes(artifact.read_object(name + "~pre"))
        (base / "post.f32").write_bytes(artifact.read_object(name + "~post"))
        pin = name + "~pin"
        pout = name + "~pout"
        if pin in artifact.by_id:
            (base / "pin.i32").write_bytes(artifact.read_object(pin))
        if pout in artifact.by_id:
            (base / "pout.i32").write_bytes(artifact.read_object(pout))
        desc = np.asarray(descriptors[name], dtype="<i4")
        (base / "desc.i32").write_bytes(desc.tobytes())
        meta = {"name": name, "format": obj.format, "n": obj.shape[0], "k": obj.shape[1],
                "bits": int(obj.format[len("trellis_trl"):]), "desc": descriptors[name],
                "pin": pin in artifact.by_id, "pout": pout in artifact.by_id}
        (base / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{name} -> {base} ({obj.format} {obj.shape[0]}x{obj.shape[1]})")
    artifact.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2:
        print(__doc__)
        return 2
    return dump(arguments[0], arguments[1], tuple(arguments[2:]) or DEFAULT_TENSORS)


if __name__ == "__main__":
    raise SystemExit(main())
