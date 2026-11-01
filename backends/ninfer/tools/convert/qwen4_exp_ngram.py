"""Stream Flash-Next's sharded HF n-gram table into independent FP8/FP16 rows."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import re
from pathlib import Path
import struct

import torch

from tools.artifact.codecs.fp8_row import encode_fp8_row_interleaved
from .methods import fp8_row_maxabs
from .quantization.fp8_row import quantize_bf16_rows
from .sources.logical import LogicalSource
from .sources.safetensors import SafetensorsSource

FORMAT = "fp8_e4m3fn_row_fp16"


def attach_hot_profile(model, path: Path | None) -> None:
    """Store an existing NFNGHOT1 profile beside its table after checking the hash and rows."""
    if path is None:
        return
    import numpy as np

    if "ngram/table" not in model.parameters:
        raise ValueError("ngram.hot requires a stored n-gram table")
    config = model.components["ngram"]["config"]
    data = path.read_bytes()
    if len(data) < 40 or data[:8] != b"NFNGHOT1":
        raise ValueError(f"{path}: not an n-gram hot-row profile")
    table_rows, fingerprint, _tokens, count = struct.unpack_from("<4Q", data, 8)
    if len(data) - 40 != count * 4:
        raise ValueError(f"{path}: invalid hot-row profile row count")
    values = [config["ngram_size"], config["heads_per_ngram"]]
    for name in ("multipliers", "head_vocab", "head_offset"):
        values.extend((len(config[name]), *config[name]))
    values.append(config["rows"])
    expected = 0xCBF29CE484222325
    for byte in struct.pack(f"<{len(values)}Q", *values):
        expected = ((expected ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    if table_rows != config["rows"] or fingerprint != expected:
        raise ValueError(f"{path}: profile was counted for another n-gram hash")
    rows = np.frombuffer(data, dtype="<u4", offset=40)
    if (len(rows) and int(rows.max()) >= table_rows) or len(np.unique(rows)) != count:
        raise ValueError(f"{path}: duplicate or out-of-range profile rows")
    object_id = "resource/ngram/hot_profile"
    model.resources[object_id] = data
    model.components["ngram"].setdefault("resources", {})["hot_profile"] = object_id


def hf_ngram_source(store: SafetensorsSource, descriptor: dict) -> LogicalSource:
    """Concatenate numbered BF16 shards without materializing the complete table."""
    from .qwen4_exp import ngram_config, text_config

    expected = ngram_config(store.config)
    if any(descriptor.get(key) != value for key, value in expected.items()):
        raise ValueError("the HF n-gram source has different table dimensions or hash constants")
    layer = text_config(store.config)["ple_layers"][0]
    prefix = f"model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding."
    pattern = re.compile(re.escape(prefix) + r"shard_(0|[1-9][0-9]*)\.weight$")
    names = {}
    for name in store.weight_map:
        if name.startswith(prefix):
            match = pattern.fullmatch(name)
            if match is None:
                raise ValueError(f"unexpected HF n-gram tensor: {name}")
            names[int(match[1])] = name
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError("HF n-gram shards must be numbered consecutively from zero")
    rows, width = descriptor["rows"], descriptor["row_width"]
    edges = [0]
    ordered = [names[i] for i in range(len(names))]
    for name in ordered:
        info = store.describe(name)
        if info.dtype != "BF16" or len(info.shape) != 2 or info.shape[0] <= 0 or info.shape[1] != width:
            raise ValueError(f"{name}: expected BF16 n-gram rows of width {width}")
        if info.bytes != info.shape[0] * width * 2:
            raise ValueError(f"{name}: n-gram shard byte count differs from its shape")
        edges.append(edges[-1] + info.shape[0] * width)
    if edges[-1] != rows * width:
        raise ValueError(f"HF n-gram shards contain {edges[-1] // width} rows; expected {rows}")

    def read(begin: int, end: int) -> torch.Tensor:
        parts = []
        while begin < end:
            shard = bisect_right(edges, begin) - 1
            stop = min(end, edges[shard + 1])
            parts.append(store.read_flat(ordered[shard], begin - edges[shard], stop - edges[shard]))
            begin = stop
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.bfloat16)

    return LogicalSource((rows, width), f"{store.path}:HF n-gram shards", read)


def fp8_table_digest(source: LogicalSource, *, rows_per_chunk: int = 4096) -> str:
    """Hash exactly the row bytes the writer will produce, with bounded host memory."""
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    digest = hashlib.sha256()
    for begin in range(0, source.shape[0], rows_per_chunk):
        values = source.rows(begin, min(begin + rows_per_chunk, source.shape[0]))
        words = quantize_bf16_rows(values, scale_dtype=torch.float16)
        digest.update(encode_fp8_row_interleaved(words.codes, words.scales, values.shape))
    return digest.hexdigest()


def configure_ngram(model, recipe, table) -> None:
    """Select HF FP8 rows or preserve an IQ4_NL GGUF table for either text recipe."""
    from .qwen4_exp_gguf import configure_ngram_gguf
    from .sources.gguf import GGUFFile

    if isinstance(table, GGUFFile):
        configure_ngram_gguf(model, recipe, table)
        return
    if not isinstance(table, SafetensorsSource):
        raise ValueError("--source ngram must name an HF safetensors table or an IQ4_NL GGUF")
    descriptor = model.components["ngram"]["config"]
    source = hf_ngram_source(table, descriptor)
    size = source.shape[0] * (source.shape[1] + 2)
    print(f"hashing quantized FP8 n-gram rows ({size / 1e9:.1f} GB)", flush=True)
    descriptor["format"] = FORMAT
    # The directory names the digest before payload production. Two bounded passes avoid a
    # table-sized temporary file; per-row quantization is independent of writer chunk boundaries.
    descriptor["table_sha256"] = fp8_table_digest(source)
    if "ngram/table" in model.parameters:
        recipe.assign("ngram/table", format=FORMAT, method=fp8_row_maxabs, source=source)
