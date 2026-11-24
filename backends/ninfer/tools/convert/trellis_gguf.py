"""NariQ GGUF -> NInfer v3 converter for coded-tile weights.

The transferable algorithm of nari-llama.cpp is the coded-linear format itself, not a quantization
pipeline: the exporter's tile words *are* the compressed representation, so this converter copies
them byte for byte and moves only the side planes and the descriptor. It never re-quantizes and
never decodes to FP32, which is the one thing a coded tensor cannot survive (a row's bytes do not
encode that row on their own).

What is written
---------------
* every coded matrix (`GGML_TYPE_TRL1..TRL6`, ids 44..49) as one `trellis_tile16_v1` object, shape
  `[N,K]`, bytes = `N * (K/16) * 8*bits` exactly as stored;
* its side planes as objects of their own: `trellis_s16` (F16 promoted to F32 without loss),
  `trellis_pre`/`trellis_post` (F32), and `trellis_pin`/`trellis_pout` (I32) when present;
* the `trellis.desc.<w>` INT32 array into `metadata["trellis.descriptors"]`, keyed by parameter;
* a banded head/embedding (`<w>.bandNN.weight` + `<w>.vocab_gather`) as one concatenated coded
  matrix in band order plus a composed output permutation (`argsort(vocab_gather)`), because the
  runtime's coded-linear Op carries one `pout` and the bands share `pre`, `s16` and the row
  descriptor. A band set whose sides disagree is rejected rather than silently rewritten.

The artifact is a schema-valid weight pack: the coded tensors are bound by their GGUF names and
their side planes are declared as Use auxiliaries. Model-role binding (`text/layers/...`,
`text/output_head`) and the loader's auxiliary plumbing are the engine-side half of the port; see
`docs/nariq-native-port.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.artifact.formats import TRELLIS_FORMATS_BY_GGML_TYPE
from tools.artifact.layouts import encoded_size
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter

from .sources.gguf import GGUFFile, TYPE_TRL1, TYPE_TRL6

SIDE_SUFFIXES = (
    ("s16", "trellis_s16", np.float32),
    ("pre", "trellis_pre", np.float32),
    ("post", "trellis_post", np.float32),
    ("pin", "trellis_pin", np.int32),
    ("pout", "trellis_pout", np.int32),
)

BAND_PATTERN = re.compile(r"(?P<head>.+)\.band(?P<index>\d+)\.weight$")


@dataclass(frozen=True, slots=True)
class CodedMatrix:
    """One coded matrix, its side planes and its descriptor, all in artifact-ready form."""

    name: str
    n: int
    k: int
    bits: int
    codes: np.ndarray  # uint8 payload, exactly encoded_size()
    s16: np.ndarray  # float32 [K/16]
    pre: np.ndarray  # float32 [K]
    post: np.ndarray  # float32 [N]
    pin: np.ndarray | None  # int32 [K]
    pout_real: np.ndarray | None  # int32 [N] (the GGUF's own pout, before any band composition)
    pout: np.ndarray | None  # int32 [N] after band composition
    desc: list[int]

    @property
    def format(self) -> str:
        return f"trellis_trl{self.bits}"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.k)


def _permutation(values: np.ndarray, size: int, label: str) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype != np.int32 or values.shape != (size,):
        raise ValueError(f"{label}: expected an I32 array of {size} entries")
    if not np.array_equal(np.sort(values), np.arange(size)):
        raise ValueError(f"{label}: not a permutation of [0,{size})")
    return values


def _side(source: GGUFFile, name: str, size: int, label: str) -> np.ndarray | None:
    info = source.tensors.get(name)
    if info is None:
        return None
    if int(np.prod(info.shape)) != size:
        raise ValueError(f"{label}: expected {size} entries, got {info.shape}")
    if info.type_id == 1:  # F16 -> F32 is exact, so the artifact needs one direct format
        return np.ascontiguousarray(source.read_direct(name).reshape(-1))
    if info.type_id == 0:
        return np.ascontiguousarray(source.read_direct(name).reshape(-1))
    if info.type_id == 26:  # I32
        return np.ascontiguousarray(
            np.frombuffer(source.tensor_bytes(name), dtype="<i4").reshape(-1)
        )
    raise ValueError(f"{label}: unsupported side type {info.type_name}")


def _descriptor(source: GGUFFile, base: str) -> list[int]:
    key = "trellis.desc." + base
    if key not in source.kv:
        raise ValueError(f"{base}: missing {key}")
    desc = [int(v) for v in np.asarray(source.kv[key]).reshape(-1)]
    if len(desc) < 7 or desc[6] < 0 or desc[6] > 8 or len(desc) != 7 + desc[6]:
        raise ValueError(f"{base}: malformed descriptor {desc}")
    return desc


def _coded_matrix(source: GGUFFile, name: str, n: int, k: int, codes: np.ndarray, base: str,
                  bits: int) -> CodedMatrix:
    desc = _descriptor(source, base)
    s16 = _side(source, base + ".trellis_s16", k // 16, base + ".trellis_s16")
    pre = _side(source, base + ".trellis_pre", k, base + ".trellis_pre")
    post = _side(source, base + ".trellis_post", n, base + ".trellis_post")
    if s16 is None or pre is None or post is None:
        raise ValueError(f"{base}: coded matrix without s16/pre/post")
    for label, plane in (("s16", s16), ("pre", pre), ("post", post)):
        if not np.isfinite(plane).all():
            raise ValueError(f"{base}: nonfinite {label}")
    pin = _side(source, base + ".trellis_pin", k, base + ".trellis_pin")
    pout = _side(source, base + ".trellis_pout", n, base + ".trellis_pout")
    if pin is not None:
        _permutation(pin, k, base + ".trellis_pin")
    if pout is not None:
        pout = _permutation(pout, n, base + ".trellis_pout")
    return CodedMatrix(name, n, k, bits, codes, s16, pre, post, pin, pout, pout, desc)


def _band_matrix(source: GGUFFile, head: str) -> CodedMatrix:
    """Concatenates `<head>.bandNN.weight` in band order and composes the vocab permutation."""
    bands: list[Any] = []
    for info in source.tensors.values():
        match = BAND_PATTERN.fullmatch(info.name)
        if match and match.group("head") == head:
            bands.append(info)
    if not bands:
        raise ValueError(f"{head}: no bands found")
    bands.sort(key=lambda info: int(BAND_PATTERN.fullmatch(info.name).group("index")))
    indices = [int(BAND_PATTERN.fullmatch(info.name).group("index")) for info in bands]
    if indices != list(range(len(bands))):
        raise ValueError(f"{head}: band indices are not 0..{len(bands) - 1}")
    if head + ".weight" in source.tensors:
        raise ValueError(f"{head}: both bands and a dense matrix")

    gather_info = source.tensors.get(head + ".vocab_gather")
    if gather_info is None:
        raise ValueError(f"{head}: bands without vocab_gather")
    rows = sum(int(np.prod(info.shape[:-1])) for info in bands)
    gather = _permutation(
        np.frombuffer(source.tensor_bytes(head + ".vocab_gather"), dtype="<i4").reshape(-1),
        rows,
        head + ".vocab_gather",
    )

    k = int(bands[0].shape[-1])
    if k % 16:
        raise ValueError(f"{head}: K is not a multiple of 16")
    bits = None
    desc = None
    s16 = pre = None
    posts = []
    payloads = []
    for info in bands:
        if not TYPE_TRL1 <= info.type_id <= TYPE_TRL6:
            raise ValueError(f"{info.name}: not a coded tile tensor")
        band_bits = info.type_id - TYPE_TRL1 + 1
        if bits is None:
            bits = band_bits
        elif band_bits != bits:
            raise ValueError(f"{head}: bands disagree on the code width")
        band_rows, band_k = (int(v) for v in info.shape)
        if band_k != k or band_rows % 16:
            raise ValueError(f"{info.name}: unexpected geometry {info.shape}")
        base = info.name[: -len(".weight")]
        band_desc = _descriptor(source, base)
        band_s16 = _side(source, base + ".trellis_s16", k // 16, base + ".trellis_s16")
        band_pre = _side(source, base + ".trellis_pre", k, base + ".trellis_pre")
        band_post = _side(source, base + ".trellis_post", band_rows, base + ".trellis_post")
        if band_s16 is None or band_pre is None or band_post is None:
            raise ValueError(f"{info.name}: coded band without s16/pre/post")
        if desc is None:
            desc, s16, pre = band_desc, band_s16, band_pre
        elif band_desc != desc or not np.array_equal(band_s16, s16) or not np.array_equal(band_pre, pre):
            raise ValueError(
                f"{info.name}: bands disagree on descriptor/pre/s16, which one coded-linear "
                "cannot carry"
            )
        if _side(source, base + ".trellis_pin", k, base + ".trellis_pin") is not None:
            raise ValueError(f"{info.name}: banded heads with an input gather are not supported")
        posts.append(band_post)
        payloads.append(np.ascontiguousarray(source.tensor_bytes(info.name)))

    codes = np.concatenate(payloads)
    post = np.concatenate(posts)
    # Row r of the concatenated bands is vocabulary entry gather[r]; the Op's output gather is the
    # inverse map, so out[v] = coded[argsort(gather)[v]].
    pout = np.argsort(gather).astype(np.int32)
    if len(post) != rows:
        raise ValueError(f"{head}: band rows and post disagree")
    return CodedMatrix(head + ".weight", rows, k, bits, codes, s16, pre, post, None, None, pout, desc)


def coded_matrices(source: GGUFFile) -> Iterable[CodedMatrix]:
    heads: set[str] = set()
    band_members: set[str] = set()
    for info in source.tensors.values():
        match = BAND_PATTERN.fullmatch(info.name)
        if match:
            heads.add(match.group("head"))
            band_members.add(info.name)
    for info in source.tensors.values():
        if info.name in band_members or ".trellis_" in info.name:
            continue
        if info.name.endswith(".vocab_gather"):
            continue
        if TYPE_TRL1 <= info.type_id <= TYPE_TRL6:
            if len(info.shape) != 2:
                raise ValueError(f"{info.name}: coded tensor must be a matrix")
            n, k = (int(v) for v in info.shape)
            if n % 16 or k % 16:
                raise ValueError(f"{info.name}: coded geometry must be 16-divisible")
            base = info.name[: -len(".weight")] if info.name.endswith(".weight") else info.name
            bits = info.type_id - TYPE_TRL1 + 1
            codes = np.ascontiguousarray(source.tensor_bytes(info.name))
            if codes.nbytes != n * (k // 16) * (2 * bits):
                raise ValueError(f"{info.name}: code payload does not match the geometry")
            yield _coded_matrix(source, info.name, n, k, codes, base, bits)
    for head in sorted(heads):
        yield _band_matrix(source, head)


def _specs(matrices: list[CodedMatrix]) -> tuple[list[Any], dict[str, CodedMatrix]]:
    specs: list[Any] = []
    by_parameter: dict[str, CodedMatrix] = {}
    for matrix in matrices:
        specs.append(TensorSpec(matrix.name, matrix.shape, matrix.format, "trellis_tile16_v1"))
        by_parameter[matrix.name] = matrix
    return specs, by_parameter


def _side_specs(matrix: CodedMatrix) -> list[tuple[str, str, tuple[int, ...], np.ndarray]]:
    out = [
        (matrix.name + "~s16", "fp32", matrix.s16.shape, matrix.s16),
        (matrix.name + "~pre", "fp32", matrix.pre.shape, matrix.pre),
        (matrix.name + "~post", "fp32", matrix.post.shape, matrix.post),
    ]
    if matrix.pin is not None:
        out.append((matrix.name + "~pin", "int32", matrix.pin.shape, matrix.pin))
    if matrix.pout is not None:
        out.append((matrix.name + "~pout", "int32", matrix.pout.shape, matrix.pout))
    return out


AUXILIARY_ROLES = {
    "~s16": "s16",
    "~pre": "pre",
    "~post": "post",
    "~pin": "pin",
    "~pout": "pout",
}


def convert(source_path: str | Path, destination: str | Path, *,
            input_name: str = "text/activation") -> dict[str, Any]:
    """Writes a `.ninfer` weight pack for every coded matrix of a NariQ GGUF."""
    source_path, destination = Path(source_path), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with GGUFFile(source_path) as source:
        matrices = list(coded_matrices(source))
        if not matrices:
            raise ValueError(f"{source_path}: no NariQ coded weights found")
        version = source.kv.get("trellis.version")
        if version != 1:
            raise ValueError(f"{source_path}: trellis.version is {version!r}, expected 1")

        specs, _ = _specs(matrices)
        side_payloads: dict[str, np.ndarray] = {}
        for matrix in matrices:
            for object_id, format_name, shape, values in _side_specs(matrix):
                specs.append(TensorSpec(object_id, shape, format_name, "contiguous_le_v1"))
                side_payloads[object_id] = np.ascontiguousarray(values)

        bindings = {matrix.name: {"object": matrix.name} for matrix in matrices}
        uses = []
        for matrix in matrices:
            auxiliaries = {}
            for suffix, role in AUXILIARY_ROLES.items():
                object_id = matrix.name + suffix
                if object_id in side_payloads:
                    auxiliaries[role] = {"object": object_id}
            uses.append({"parameter": matrix.name, "input": input_name,
                         "auxiliaries": auxiliaries})
        metadata = {
            "name": source_path.stem,
            "trellis.version": 1,
            "trellis.descriptors": {matrix.name: matrix.desc for matrix in matrices},
        }
        components = {"text": {"config": {"architecture": source.kv.get("general.architecture", "qwen35")}}}
        for matrix in matrices:
            if encoded_size("trellis_tile16_v1", matrix.format, matrix.shape) != matrix.codes.nbytes:
                raise ValueError(f"{matrix.name}: encoded size disagrees with the payload")

        with ArtifactWriter(destination, specs, components=components, bindings=bindings,
                            uses=uses, metadata=metadata,
                            provenance={"source": str(source_path),
                                        "source_bytes": source_path.stat().st_size,
                                        "coded_tensors": len(matrices)}) as writer:
            def write_bytes(object_id: str, data: np.ndarray) -> None:
                view = memoryview(np.ascontiguousarray(data)).cast("B")
                chunk = 64 * 1024 * 1024
                for start in range(0, len(view), chunk):
                    writer.write_region(object_id, start, view[start : start + chunk])

            for matrix in matrices:
                write_bytes(matrix.name, matrix.codes)
            for object_id, values in side_payloads.items():
                write_bytes(object_id, values)

    return {
        "destination": str(destination),
        "coded_tensors": len(matrices),
        "bytes": destination.stat().st_size,
    }
