"""Split a DFlash drafter out of a target artifact into its own ``--draft-model`` artifact.

The target converter writes the drafter as a ``dflash``/``dflash2`` component beside ``text``,
because a release ships the drafter with the target it was trained for. ``--draft-model`` instead
loads a drafter alone, from an artifact whose only component is the drafter and which targets
nothing (the target keeps its own embedding, head and KV).

This tool copies that component — its config, its parameters, the Uses they read and the objects
those bindings name — into such an artifact. The bytes are the ones the target converter wrote, so
the drafter the engine runs is bit-for-bit the one it ran embedded.

The target's ``text`` config supplies ``hidden_size``, ``vocab_size`` and ``num_target_layers``,
which an embedded component leaves implicit because it reads them from the text beside it. A
standalone drafter states them, and ``models/qwen3_5/config.cpp`` compares them with the model it is
attached to, so attaching the wrong drafter fails at startup by name instead of at the first draft.

Usage:
  python -m tools.convert.draft_standalone \\
      --source target.ninfer --component dflash2 --out dflash2-drafter.ninfer
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceObject, TensorSpec
from tools.artifact.writer import ArtifactWriter

# The components a target may carry a drafter in, and the name each keeps standalone (the name
# models/draft_artifact.cpp keys the backend on).
COMPONENTS = ("dflash", "dflash2")


def _objects_of(bindings: dict) -> list[str]:
    """Every stored object a set of parameter bindings reads, in first-seen order."""
    out: list[str] = []
    for binding in bindings.values():
        if "object" in binding:
            out.append(binding["object"])
        for part in binding.get("parts", ()):
            out.append(part["object"])
    return out


def component_record(directory, component: str) -> dict:
    """The component's directory entry as a standalone one: config with the target's geometry, no
    target reference."""
    if component not in directory.components:
        raise ValueError(
            f"no {component!r} component (the artifact has {sorted(directory.components)})"
        )
    record = dict(directory.components[component])
    # An embedded drafter targets the text it drafts for; standalone it targets nothing, because the
    # artifact holds no text component and the model it attaches to supplies the rest.
    if record.get("target") != "text":
        raise ValueError(
            f"the {component!r} component targets {record.get('target')!r}, not the text it "
            "drafts for"
        )
    record.pop("target")
    config = dict(record["config"])
    text = directory.components["text"]["config"]
    for field, value in (
        ("hidden_size", text.get("hidden_size")),
        ("vocab_size", text.get("vocab_size")),
        ("num_target_layers", text.get("num_hidden_layers")),
    ):
        if value is None:
            raise ValueError(f"the target's text config has no {field}")
        if field in config and config[field] != value:
            raise ValueError(
                f"{component}.{field} is {config[field]}, the target's {value}"
            )
        config[field] = value
    return {"config": config}


def extract(source: Path, component: str, out: Path, name: str | None) -> dict:
    with Artifact(source) as artifact:
        directory = artifact.directory
        record = component_record(directory, component)
        prefix = component + "/"
        bindings = {
            key: value
            for key, value in directory.bindings.items()
            if key.startswith(prefix)
        }
        if not bindings:
            raise ValueError(f"the {component!r} component binds no parameters")
        uses = [
            use
            for use in directory.uses
            if str(use.get("parameter", "")).startswith(prefix)
        ]
        wanted = set(_objects_of(bindings))
        specs: list[TensorSpec] = []
        for obj in directory.objects:
            if obj.id not in wanted:
                continue
            if isinstance(obj, ResourceObject):
                raise ValueError(f"{obj.id} is a resource, not a drafter weight")
            specs.append(
                TensorSpec(obj.id, obj.shape, obj.format, obj.layout, obj.divisors)
            )
        absent = wanted - {spec.id for spec in specs}
        if absent:
            raise ValueError(f"bindings name absent objects {sorted(absent)[:4]}")
        payload = sum(artifact.object(spec.id).bytes for spec in specs)
        print(
            f"{len(bindings)} parameters, {len(uses)} uses, {len(specs)} objects, "
            f"{payload / 1024 ** 2:.1f} MiB",
            flush=True,
        )
        with ArtifactWriter(
            out,
            specs,
            components={component: record},
            bindings=bindings,
            uses=uses,
            metadata={} if name is None else {"name": name},
            provenance={"source": str(source), "component": component},
        ) as writer:
            for spec in specs:
                writer.write_object(spec.id, artifact.iter_object(spec.id))
    return {"parameters": len(bindings), "objects": len(specs), "payload_bytes": payload}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A target artifact's drafter component as a standalone .ninfer"
    )
    parser.add_argument("--source", required=True, help="the target artifact carrying the drafter")
    parser.add_argument(
        "--component", default="dflash2", choices=COMPONENTS, help="which drafter to split out"
    )
    parser.add_argument("--out", required=True, help="the artifact to write")
    parser.add_argument("--name", default=None, help="artifact name in its metadata")
    args = parser.parse_args()

    report = extract(Path(args.source), args.component, Path(args.out), args.name)
    print(
        f"wrote {args.out}: {report['payload_bytes'] / 1024 ** 2:.1f} MiB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
