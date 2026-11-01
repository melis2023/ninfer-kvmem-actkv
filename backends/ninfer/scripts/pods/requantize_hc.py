#!/usr/bin/env python3
"""Rewrite a Flash-Next artifact with its hyper-connection matrices in ggml Q8_0.

Every hyper-connection `down`, `up` and `inject` matrix (text layers, final mixer, MTP layer and
MTP final mixer) stored as BF16 becomes `gguf_q8_0` rows, as the qwen3_8_flash_next_gguf recipe
now stores them: the text matrices, BF16 in the GSQ-RCO GGUFs, quantized by ggml's
quantize_row_q8_0_ref (per 32 inputs of a row a binary16 d = amax / 127 and codes round(x / d));
the MTP block's, which earlier conversions decoded from an MTP GGUF to BF16, taken from that GGUF
(`--mtp`) after checking that its values rounded to BF16 are the artifact's words. Every other
object is copied byte for byte and verified against its source. The norms stay BF16.

usage: requantize_hc.py INPUT.ninfer OUTPUT.ninfer [--mtp mtp-*.gguf]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec
from tools.convert import qwen4_exp_gguf
from tools.convert.gguf_blocks import dequantize_q8_0, quantize_q8_0
from tools.convert.sources.gguf import GGUFFile

HC_MATRIX = re.compile(
    r"^(text/layers/\d+/(attn|mlp)_hc|text/final_mixer|mtp/layer/(attn|mlp)_hc|mtp/final_mixer)"
    r"/(down|up|inject)$"
)
Q8_VALUES = 32
MTP_TENSORS = {
    "mtp/final_mixer": f"blk.{qwen4_exp_gguf.MTP_BLOCK}.nextn.hc_head_",
    "mtp/layer/attn_hc": f"blk.{qwen4_exp_gguf.MTP_BLOCK}.hc_attn_",
    "mtp/layer/mlp_hc": f"blk.{qwen4_exp_gguf.MTP_BLOCK}.hc_ffn_",
}


def bf16_words(data: bytes, shape) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint16).reshape(shape)


def bf16_values(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32)


def plan(base) -> dict[str, str]:
    """The BF16 objects bound whole to a hyper-connection matrix: object id -> parameter."""
    objects = {obj.id: obj for obj in base.objects}
    targets = {}
    for name, binding in base.directory.bindings.items():
        if not HC_MATRIX.match(name):
            continue
        if set(binding) != {"object"}:
            raise ValueError(f"{name}: expected a whole-object binding, got {binding}")
        obj = objects[binding["object"]]
        if obj.format == "gguf_q8_0":
            continue
        if obj.format != "bf16" or len(obj.shape) != 2 or obj.shape[1] % Q8_VALUES:
            raise ValueError(f"{name}: {obj.format} {obj.shape} is not a BF16 matrix of blocks")
        if obj.id in targets:
            raise ValueError(f"{obj.id} is bound twice")
        targets[obj.id] = name
    if not targets:
        raise ValueError("no BF16 hyper-connection matrix to requantize")
    return targets


def spec(obj, targets):
    if isinstance(obj, TensorObject):
        if obj.id in targets:
            return TensorSpec(obj.id, obj.shape, "gguf_q8_0", "gguf_blocks_v1")
        return TensorSpec(obj.id, obj.shape, obj.format, obj.layout, obj.divisors)
    return ResourceSpec(obj.id, obj.bytes, obj.encoding)


def mtp_blocks(mtp: GGUFFile, name: str, words: np.ndarray) -> bytes:
    """The recipe's Q8_0 rows of an MTP matrix, from the GGUF the artifact's BF16 words came from."""
    prefix, role = name.rsplit("/", 1)
    tensor = MTP_TENSORS[prefix] + role + ".weight"
    source = qwen4_exp_gguf.hc_matrix_source(mtp, tensor, words.shape)
    stored = qwen4_exp_gguf.bf16_matrix_source(mtp, tensor, words.shape)[0]
    rounded = stored.rows(0, words.shape[0]).to(torch.bfloat16).view(torch.int16).numpy()
    if not np.array_equal(rounded.view(np.uint16), words):
        raise ValueError(f"{name}: {tensor} of the MTP GGUF is not this artifact's source")
    return source.read_encoded(0, words.shape[0]).codes.numpy().tobytes()


def requantize(source: Path, output: Path, mtp_path: Path | None) -> dict:
    from tools.artifact.writer import ArtifactWriter

    report_path = Path(str(output) + ".conversion.json")
    if output.exists() or report_path.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        base = stack.enter_context(Artifact(source))
        targets = plan(base)
        directory = base.directory
        mtp = None
        if any(name.startswith("mtp/") for name in targets.values()):
            if mtp_path is None:
                raise ValueError("the artifact has an MTP block: name its GGUF with --mtp")
            mtp = stack.enter_context(GGUFFile(mtp_path))
            qwen4_exp_gguf.validate_mtp(mtp, directory.components["mtp"]["config"]["num_experts"])
        error = {"max_abs": 0.0, "rms_sum": 0.0, "values": 0}
        transformation = {
            "type": "hyper_connection_q8_0", "parent_artifact_id": base.artifact_id.hex(),
            "matrices": len(targets),
            "algorithm": "ggml quantize_row_q8_0_ref (d = amax / 127, round half away); MTP "
                         "matrices as stored in their GGUF",
        }
        if mtp is not None:
            transformation["mtp_source"] = {"file": mtp_path.name,
                                            "bytes": mtp_path.stat().st_size}
        quantized = {}
        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(output, [spec(obj, targets) for obj in base.objects],
                            components=directory.components, bindings=directory.bindings,
                            uses=directory.uses, metadata=directory.metadata,
                            provenance={**directory.provenance,
                                        "hyper_connection_q8_0": transformation}) as writer:
            done, total, last = 0, sum(obj.bytes for obj in base.objects), time.monotonic()
            for obj in base.objects:
                if obj.id in targets:
                    name = targets[obj.id]
                    words = bf16_words(base.read_object(obj.id), obj.shape)
                    values = bf16_values(words)
                    blocks = (mtp_blocks(mtp, name, words) if name.startswith("mtp/")
                              else quantize_q8_0(values).tobytes())
                    represented = dequantize_q8_0(
                        np.frombuffer(blocks, np.uint8).reshape(obj.shape[0], -1))
                    difference = represented.astype(np.float64) - values
                    error["max_abs"] = max(error["max_abs"], float(np.abs(difference).max()))
                    error["rms_sum"] += float((difference ** 2).sum())
                    error["values"] += values.size
                    quantized[obj.id] = hashlib.sha256(blocks).hexdigest()
                    writer.write_object(obj.id, iter([blocks]))
                else:
                    writer.write_object(obj.id, base.iter_object(obj.id))
                done += obj.bytes
                if time.monotonic() - last >= 15:
                    print(json.dumps({"bytes": done, "total": total}), flush=True)
                    last = time.monotonic()
            identity = writer.artifact_id.hex()
        # Readback: copied objects equal their source, quantized ones the bytes written.
        with Artifact(output) as result:
            if (result.directory.bindings != directory.bindings or
                    result.directory.components != directory.components):
                raise ValueError("output metadata mismatch")
            for obj in result.objects:
                data = hashlib.sha256()
                for chunk in result.iter_object(obj.id):
                    data.update(chunk)
                if obj.id in targets:
                    if data.hexdigest() != quantized[obj.id] or obj.format != "gguf_q8_0":
                        raise ValueError(f"{obj.id}: quantized payload mismatch")
                    continue
                expected = hashlib.sha256()
                for chunk in base.iter_object(obj.id):
                    expected.update(chunk)
                if data.digest() != expected.digest():
                    raise ValueError(f"{obj.id}: copied payload mismatch")
    report = {"artifact_id": identity, "file": output.name, "bytes": output.stat().st_size,
              "transformation": transformation, "requantized": sorted(targets.values()),
              "max_abs_error": error["max_abs"],
              "rms_error": (error["rms_sum"] / max(error["values"], 1)) ** 0.5,
              "elapsed_seconds": time.monotonic() - started}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mtp", type=Path, help="the MTP GGUF the artifact's MTP block came from")
    args = parser.parse_args()
    print(json.dumps(requantize(args.input, args.output, args.mtp), indent=2))
