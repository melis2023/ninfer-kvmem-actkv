#!/usr/bin/env python3
"""Append a qualified Flash-Next MTP component without changing the target model's bytes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec, binding_parts
from tools.artifact.writer import ArtifactWriter


def plan(base, donor):
    """Keep all target objects and remap the donor's physical IDs, including split bindings."""
    if "mtp" in base.components or any(k.startswith("mtp/") for k in base.bindings):
        raise ValueError("target already contains MTP; refusing to replace it")
    if "text" not in base.components or "text" not in donor.components:
        raise ValueError("both artifacts must describe a text model")
    left, right = base.components["text"]["config"], donor.components["text"]["config"]
    if left.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("expected the Flash-Next architecture")
    # A pruned target and its full-size MTP own separate expert banks in the loader.
    ignore = {"num_experts"}
    if {k: v for k, v in left.items() if k not in ignore} != {
            k: v for k, v in right.items() if k not in ignore}:
        raise ValueError("target and MTP donor have incompatible text geometry or semantics")
    if base.components.get("ngram", {}).get("config") != donor.components.get("ngram", {}).get("config"):
        raise ValueError("target and MTP donor require different n-gram tables")
    component = deepcopy(donor.components.get("mtp", {}))
    if (component.get("target") != "text" or
            component.get("config", {}).get("architectures") != ["Qwen4ExpMTP"]):
        raise ValueError("donor has no supported MTP component")
    mtp_bindings = {k: deepcopy(v) for k, v in donor.bindings.items() if k.startswith("mtp/")}
    if not mtp_bindings:
        raise ValueError("donor MTP component has no weights")
    mtp_uses = [deepcopy(u) for u in donor.uses if u["parameter"].startswith("mtp/")]
    donor_index = {obj.id: obj for obj in donor.objects}
    referenced = set(component.get("resources", {}).values())
    all_bindings = list(mtp_bindings.values()) + [
        binding for use in mtp_uses for binding in use.get("auxiliaries", {}).values()]
    for binding in all_bindings:
        referenced.update(parent for parent, _, _ in binding_parts(binding, donor_index))
    selected = [obj for obj in donor.objects if obj.id in referenced]
    existing = {obj.id for obj in base.objects}
    remap = {}
    for obj in selected:
        name = "mtp-import/" + obj.id
        while name in existing:
            name = "mtp-import/" + name
        existing.add(name)
        remap[obj.id] = name
    for binding in all_bindings:
        for part in binding.get("parts", [binding]):
            part["object"] = remap[part["object"]]
    if "resources" in component:
        component["resources"] = {k: remap[v] for k, v in component["resources"].items()}
    components = {**deepcopy(base.components), "mtp": component}
    bindings = {**deepcopy(base.bindings), **mtp_bindings}
    uses = [*deepcopy(base.uses), *mtp_uses]
    # MTP also invokes the target's output head. Its activation policy and auxiliaries must
    # come from this target representation, which may differ from the donor's quantization.
    shared_uses = [u for u in donor.uses if not u["parameter"].startswith("mtp/")
                   and u["input"].startswith("mtp/")]
    if len(shared_uses) != 1:
        raise ValueError("donor must declare the shared MTP output head use")
    for donor_use in shared_uses:
        if donor_use["parameter"] != "text/output_head" or donor_use["input"] != "mtp/final_hidden":
            raise ValueError("unsupported shared MTP parameter use")
        original = [u for u in base.uses if u["parameter"] == "text/output_head"
                    and u["input"] == "text/final_hidden"]
        if len(original) != 1:
            raise ValueError("target output head has no unambiguous activation contract")
        use = deepcopy(original[0])
        use["input"] = "mtp/final_hidden"
        uses.append(use)
    return selected, remap, components, bindings, uses


def spec(obj, name):
    if isinstance(obj, TensorObject):
        return TensorSpec(name, obj.shape, obj.format, obj.layout, obj.divisors)
    return ResourceSpec(name, obj.bytes, obj.encoding)


def verify_copy(result, output, sources):
    """Compare payload bytes with their sources and hash the file in the same readback."""
    if len(result.directory.files) != 1:
        raise ValueError("MTP attachment verification requires one output file")
    whole, objects = hashlib.sha256(), {}
    with output.open('rb') as stream:
        for obj in sorted(result.objects, key=lambda value: value.offset):
            target = result.payload_offset + obj.offset
            if stream.tell() > target:
                raise ValueError("overlapping output objects")
            while stream.tell() < target:
                data = stream.read(min(8 << 20, target - stream.tell()))
                if not data:
                    raise ValueError("truncated output framing")
                whole.update(data)
            source, identity = sources[obj.id]
            digest, size = hashlib.sha256(), 0
            for expected in source.iter_object(identity):
                actual = stream.read(len(expected))
                if actual != expected:
                    raise ValueError(f"output payload mismatch: {obj.id}")
                size += len(actual)
                whole.update(actual)
                digest.update(actual)
            if size != obj.bytes:
                raise ValueError(f"output object size mismatch: {obj.id}")
            objects[obj.id] = digest.hexdigest()
        while data := stream.read(8 << 20):
            whole.update(data)
    return objects, whole.hexdigest()


def attach(base, donor, output: Path, *, provenance=None):
    report_path = Path(str(output) + ".conversion.json")
    if output.exists() or report_path.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    selected, remap, components, bindings, uses = plan(base.directory, donor.directory)
    transformation = {"type": "append_existing_mtp", "parent_artifact_id": base.artifact_id.hex(),
                      "donor_artifact_id": donor.artifact_id.hex(),
                      "source": provenance or {}, "mtp_bytes": sum(o.bytes for o in selected)}
    specs = [spec(obj, obj.id) for obj in base.objects] + [spec(obj, remap[obj.id]) for obj in selected]
    sources = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with ArtifactWriter(output, specs, components=components, bindings=bindings, uses=uses,
                        metadata=base.directory.metadata,
                        provenance={**base.directory.provenance, "mtp_attachment": transformation}) as writer:
        for source, objects, names in ((base, base.objects, {}), (donor, selected, remap)):
            total = sum(obj.bytes for obj in objects)
            done, last = 0, time.monotonic()
            for obj in objects:
                name = names.get(obj.id, obj.id)
                writer.write_object(name, source.iter_object(obj.id))
                sources[name] = (source, obj.id)
                done += obj.bytes
                if time.monotonic() - last >= 15:
                    print(json.dumps({"stage": "mtp" if names else "base", "bytes": done,
                                      "total": total}), flush=True)
                    last = time.monotonic()
        identity = writer.artifact_id.hex()
    # Exact payload comparison also computes the publication digest, avoiding separate full
    # artifact scans for copying hashes, readback hashes and the final file digest.
    with Artifact(output) as result:
        digests, file_digest = verify_copy(result, output, sources)
        if (result.directory.components != components or result.directory.bindings != bindings or
                list(result.directory.uses) != uses):
            raise ValueError("output metadata mismatch")
    report = {"artifact_id": identity, "file": output.name, "bytes": output.stat().st_size,
              "transformation": transformation, "components": components,
              "base_objects_verified": len(base.objects), "mtp_objects_verified": len(selected),
              "mtp_bindings": sum(k.startswith("mtp/") for k in bindings),
              "object_sha256": digests, "file_sha256": file_digest,
              "elapsed_seconds": time.monotonic() - started,
              "gpu_qualification": "pending"}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
