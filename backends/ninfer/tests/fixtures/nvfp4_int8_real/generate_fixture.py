# -*- coding: utf-8 -*-
"""Generates the real-artifact NVFP4 weight fixture for the int8-grouped wrapper tests.

Each entry is one fused NVFP4 parent exactly as the artifact stores it: the E2M1 code plane, the
BlockScaleK16M128x4 scale plane, and the trailing FP32 weight divisor. The test binary uploads the
payload verbatim, runs the wrapper's int8 route, and checks sampled rows against an FP64 oracle it
decodes from the same bytes.

Usage:
    python generate_fixture.py <artifact.ninfer> <output.bin> [layer]

The default layer pick (3 for the attention block, 1 for the GDN block) matches the Qwen3.5
hybrid layout this artifact family ships.
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.artifact.reader import Artifact  # noqa: E402


def binding_object(art, name):
    rec = art.directory.bindings[name]
    parts = rec.get("parts") or [{"object": rec["object"]}]
    if len(parts) != 1:
        raise SystemExit(f"{name}: expected a single-part binding, got {len(parts)}")
    return parts[0]["object"]


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    artifact_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else None

    with Artifact(artifact_path) as art:
        objects = {o.id: o for o in art.objects if hasattr(o, "shape")}
        picks = {
            "attn_fused_14336x5120": f"text/layers/{layer or 3}/attention/query",
            "gdn_fused_16384x5120": f"text/layers/{layer or 1}/gdn/query",
            "down_5120x17408": f"text/layers/{layer or 3}/mlp/down",
            "attn_output_5120x6144": f"text/layers/{layer or 3}/attention/output",
            "gdn_output_5120x6144": f"text/layers/{layer or 1}/gdn/output",
        }
        entries = []
        for name, binding in picks.items():
            if binding not in art.directory.bindings:
                raise SystemExit(f"missing binding: {binding}")
            object_id = binding_object(art, binding)
            obj = objects[object_id]
            if (obj.format or "").lower() != "nvfp4":
                raise SystemExit(f"{binding}: object {object_id} is {obj.format}, expected nvfp4")
            n, k = (int(extent) for extent in obj.shape)
            expected = n * k // 2 + n * k // 16 + 4
            if obj.bytes != expected:
                raise SystemExit(
                    f"{binding}: object {object_id} bytes {obj.bytes} != code+scale+divisor {expected}")
            payload = art.read_range(obj.offset, obj.bytes)
            entries.append((name, n, k, payload))
            print(f"{name}: [{n}x{k}] from {object_id} ({obj.bytes} bytes)")

    with open(output_path, "wb") as stream:
        stream.write(b"NVRW")
        stream.write(struct.pack("<I", len(entries)))
        for name, n, k, payload in entries:
            encoded = name.encode("utf-8")
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)
            stream.write(struct.pack("<IIQ", n, k, len(payload)))
            stream.write(payload)
    print(f"wrote {len(entries)} entries -> {output_path}")


if __name__ == "__main__":
    main()
