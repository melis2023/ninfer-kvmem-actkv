"""Read the llama.cpp DSpine draft GGUF where the official recipes read a directory.

The DSpine release (``Qwen3.8-27B-DSpine-b16``) ships as one ``dflash`` GGUF whose
tensors already use the row order the artifact wants. Its metadata carries the
draft geometry, the block size and the target-layer taps, but not the bridge
geometry, so the selected-prefix sections are derived from the tensors themselves.

Every logical name of the ``dflash`` adapter maps onto the ``fc``/``blk.*``/
``dspine_*`` tensor that holds it. The projections whose consumer walks whole
ggml blocks -- the feature projection, the attention output, the merged gate/up
MLP parent and the MLP down -- are imported byte for byte in the source's own
block type. The fused attention input projection and the two context projections
need a row-split Q8 encoding, so they take the official fallback; the bridge
codebook is read as BF16 because the bridge gathers it row by row.

The GGUF counts the target layers from one. This adapter keeps the GGUF numbering
because the engine reads the *input* of a layer (the residual stream at the top
of that layer's block), exactly like ``t_layer_inp[il]`` in llama.cpp, whereas the
checkpoint numbers layer outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .sources.gguf import (
    GGUFFile,
    TYPE_BF16,
    TYPE_F16,
    TYPE_F32,
    TYPE_NAMES,
    TensorInfo,
)
from .sources.logical import EncodedRows, LogicalSource

_DIRECT_TYPES = (TYPE_F32, TYPE_F16, TYPE_BF16)
_EMPTY_SCALES = torch.empty(0, dtype=torch.float16)

# GGUF tensor name, the adapter's logical parameter and the checkpoint name the adapter builder
# reads; ``{i}`` is the layer index. The three names are one-to-one, so one table serves both the
# builder's store lookups and the recipe's block overrides.
_TOP_TENSORS = (
    ("fc.weight", "dflash/feature_projection", "fc.weight"),
    ("enc.output_norm.weight", "dflash/context_norm", "hidden_norm.weight"),
    ("output_norm.weight", "dflash/final_norm", "norm.weight"),
    (
        "dspine_codebook.weight",
        "dflash/selected_prefix_bridge/codebook",
        "selected_prefix_bridge.codebook",
    ),
    (
        "dspine_gate.weight",
        "dflash/selected_prefix_bridge/gate_weight",
        "selected_prefix_bridge.gate.weight",
    ),
    (
        "dspine_gate.bias",
        "dflash/selected_prefix_bridge/gate_bias",
        "selected_prefix_bridge.gate.bias",
    ),
    (
        "dspine_pred_gate.weight",
        "dflash/selected_prefix_bridge/predecessor_gate_weight",
        "selected_prefix_bridge.predecessor_gate.weight",
    ),
    (
        "dspine_prev.weight",
        "dflash/selected_prefix_bridge/prev_weight",
        "selected_prefix_bridge.prev.weight",
    ),
)

_LAYER_TENSORS = (
    (
        "blk.{i}.attn_norm.weight",
        "dflash/layers/{i}/input_norm",
        "layers.{i}.input_layernorm.weight",
    ),
    (
        "blk.{i}.ffn_norm.weight",
        "dflash/layers/{i}/post_attention_norm",
        "layers.{i}.post_attention_layernorm.weight",
    ),
    (
        "blk.{i}.attn_q.weight",
        "dflash/layers/{i}/attention/query",
        "layers.{i}.self_attn.q_proj.weight",
    ),
    (
        "blk.{i}.attn_k.weight",
        "dflash/layers/{i}/attention/key",
        "layers.{i}.self_attn.k_proj.weight",
    ),
    (
        "blk.{i}.attn_v.weight",
        "dflash/layers/{i}/attention/value",
        "layers.{i}.self_attn.v_proj.weight",
    ),
    (
        "blk.{i}.attn_output.weight",
        "dflash/layers/{i}/attention/output",
        "layers.{i}.self_attn.o_proj.weight",
    ),
    (
        "blk.{i}.attn_q_norm.weight",
        "dflash/layers/{i}/attention/query_norm",
        "layers.{i}.self_attn.q_norm.weight",
    ),
    (
        "blk.{i}.attn_k_norm.weight",
        "dflash/layers/{i}/attention/key_norm",
        "layers.{i}.self_attn.k_norm.weight",
    ),
    (
        "blk.{i}.ffn_gate.weight",
        "dflash/layers/{i}/mlp/gate",
        "layers.{i}.mlp.gate_proj.weight",
    ),
    ("blk.{i}.ffn_up.weight", "dflash/layers/{i}/mlp/up", "layers.{i}.mlp.up_proj.weight"),
    (
        "blk.{i}.ffn_down.weight",
        "dflash/layers/{i}/mlp/down",
        "layers.{i}.mlp.down_proj.weight",
    ),
    (
        "blk.{i}.dspine_read.weight",
        "dflash/layers/{i}/dspine_read",
        "layers.{i}.dspine_read.weight",
    ),
    (
        "blk.{i}.dspine_write.weight",
        "dflash/layers/{i}/dspine_write",
        "layers.{i}.dspine_write.weight",
    ),
)

# The bridge tensors the builder reads through a composite name rather than a GGUF tensor.
_EARLY_READ = "selected_prefix_bridge.early_read_weights"
_FEATURE_WRITE = "selected_prefix_bridge.feature_write_weights"
_READ_WEIGHT = "selected_prefix_bridge.read_weight"
_OUT_WEIGHT = "selected_prefix_bridge.out.weight"

# Bridge names the builder reads as composites (or a zero placeholder) instead of one tensor.
_COMPOSITES = frozenset({_EARLY_READ, _FEATURE_WRITE, _READ_WEIGHT, _OUT_WEIGHT})


def _aliases(layers: int) -> dict[str, str]:
    """Every name a source lookup may use, mapped to the GGUF tensor that holds it."""

    rows = list(_TOP_TENSORS)
    for layer in range(layers):
        rows.extend(tuple(item.format(i=layer) for item in entry) for entry in _LAYER_TENSORS)
    table: dict[str, str] = {}
    for tensor, logical, source in rows:
        for alias in (tensor, logical, source):
            table[alias] = tensor
    # Composites and the terminal-disabled output projection have no single GGUF tensor.
    table[_EARLY_READ] = _EARLY_READ
    table[_FEATURE_WRITE] = _FEATURE_WRITE
    table[_READ_WEIGHT] = _READ_WEIGHT
    table[_OUT_WEIGHT] = _OUT_WEIGHT
    return table


def _integer(kv: dict, key: str, label: str) -> int:
    value = kv.get(key)
    if type(value) is not int:
        raise ValueError(f"{label}: {key} must be an integer, got {value!r}")
    return value


def _real(kv: dict, key: str, label: str) -> float:
    value = kv.get(key)
    if type(value) not in (int, float):
        raise ValueError(f"{label}: {key} must be a real number, got {value!r}")
    return float(value)


def validate(gguf: GGUFFile) -> tuple[dict, dict[str, str]]:
    """Refuse a GGUF that is not a DSpine draft, and return its config and name aliases."""

    label = str(gguf.path)
    architecture = gguf.kv.get("general.architecture")
    if architecture != "dflash":
        raise ValueError(f"{label}: general.architecture = {architecture!r}, expected 'dflash'")
    if gguf.kv.get("dflash.attention.causal") is not True:
        raise ValueError(f"{label}: a DSpine draft is block-causal")
    layers = _integer(gguf.kv, "dflash.block_count", label)
    if layers <= 0:
        raise ValueError(f"{label}: dflash.block_count must be positive")
    tokens = gguf.kv.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError(f"{label}: tokenizer.ggml.tokens is absent")
    pattern = gguf.kv.get("dflash.attention.sliding_window_pattern")
    if (
        not isinstance(pattern, list)
        or len(pattern) != layers
        or any(type(value) is not bool for value in pattern)
    ):
        raise ValueError(f"{label}: dflash.attention.sliding_window_pattern is invalid")
    taps = gguf.kv.get("dflash.target_layers")
    if (
        not isinstance(taps, list)
        or not taps
        or any(type(value) is not int or value <= 0 for value in taps)
        or len(set(taps)) != len(taps)
    ):
        raise ValueError(f"{label}: dflash.target_layers is invalid")
    head_dim = _integer(gguf.kv, "dflash.attention.key_length", label)
    if _integer(gguf.kv, "dflash.attention.value_length", label) != head_dim:
        raise ValueError(f"{label}: the draft key and value widths differ")
    block_size = _integer(gguf.kv, "dflash.block_size", label)
    if block_size <= 0:
        raise ValueError(f"{label}: dflash.block_size must be positive")
    layer_types = [
        "sliding_attention" if sliding else "full_attention" for sliding in pattern
    ]
    names = _aliases(layers)
    codebook = gguf.info("dspine_codebook.weight")
    gate = gguf.info("dspine_gate.weight")
    if len(codebook.shape) != 2 or len(gate.shape) != 2:
        raise ValueError(f"{label}: the bridge tensors must be matrices")
    vocab_size, code_size = codebook.shape
    rank, gate_code_size = gate.shape
    if gate_code_size != code_size:
        raise ValueError(f"{label}: the bridge gate and codebook disagree on code_size")
    read = gguf.info("blk.0.dspine_read.weight")
    if tuple(read.shape) != (code_size, _integer(gguf.kv, "dflash.embedding_length", label)):
        raise ValueError(f"{label}: dspine_read does not project onto the codebook")
    write = gguf.info("blk.0.dspine_write.weight")
    if tuple(write.shape) != (_integer(gguf.kv, "dflash.embedding_length", label), rank):
        raise ValueError(f"{label}: dspine_write does not write the bridge rank")
    # The second pass re-reads the codes the first four exact writes produced.
    first_exact = layers - 1
    if first_exact <= 0:
        raise ValueError(f"{label}: the bridge needs at least one exact sender")
    top_k = _integer(gguf.kv, "dflash.selector_top_k", label)
    config = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "is_causal": True,
        # The GGUF's dflash.target_layers already addresses target layer *inputs*;
        # tell draft_config not to apply the HF checkpoint's output->input +1.
        "target_layer_ids_are_inputs": True,
        "hidden_size": _integer(gguf.kv, "dflash.embedding_length", label),
        "vocab_size": len(tokens),
        "intermediate_size": _integer(gguf.kv, "dflash.feed_forward_length", label),
        "num_attention_heads": _integer(gguf.kv, "dflash.attention.head_count", label),
        "num_key_value_heads": _integer(gguf.kv, "dflash.attention.head_count_kv", label),
        "head_dim": head_dim,
        "num_hidden_layers": layers,
        "max_position_embeddings": _integer(gguf.kv, "dflash.context_length", label),
        "rms_norm_eps": _real(gguf.kv, "dflash.attention.layer_norm_rms_epsilon", label),
        "rope_parameters": {
            "rope_theta": _real(gguf.kv, "dflash.rope.freq_base", label)
        },
        "layer_types": layer_types,
        "block_size": block_size,
        "code_bridge_second_pass": {
            "first_exact": first_exact,
            "head_topk": top_k,
            "mode": "serial_table",
        },
        "dflash_config": {
            # The engine reads layer inputs, so it keeps the GGUF's one-based numbering.
            "target_layer_ids": list(taps),
            "mask_token_id": _integer(gguf.kv, "tokenizer.ggml.mask_token_id", label),
            "block_size": block_size,
            "draft_block_causal": True,
            "selected_prefix_code_bridge": {
                "code_size": code_size,
                "eps": 1e-6,
                "kappa": 10.0,
                "rank": rank,
            },
            "selected_prefix_last_write_refine": {
                "head_topk": top_k,
                "kind": "exact_sender",
                "version": 1,
                "weight": 1.0,
            },
        },
    }
    if "sliding_attention" in layer_types:
        config["sliding_window"] = _integer(
            gguf.kv, "dflash.attention.sliding_window", label
        )
    expected = set(names.values()) - _COMPOSITES
    missing = sorted(expected - set(gguf.tensors))
    extra = sorted(set(gguf.tensors) - expected)
    if missing or extra:
        raise ValueError(
            f"{label}: tensor set mismatch (missing {missing[:5]}, extra {extra[:5]})"
        )
    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{label}: the data section is truncated")
    return config, names


@dataclass(frozen=True, slots=True)
class TensorSignature:
    shape: tuple[int, ...]
    dtype: str


class DraftSource:
    """A DSpine draft GGUF presented as a checkpoint directory.

    ``has``/``describe``/``read_flat`` speak the names the adapter builder reads, so the ordinary
    lazy source resolution works unchanged; ``importable`` returns the same tensor as its own
    ggml blocks where the recipe is allowed to import it.
    """

    def __init__(self, path: str | Path) -> None:
        self.gguf = GGUFFile(path)
        self.path = self.gguf.path
        self.config, self.names = validate(self.gguf)
        self.layers = self.config["num_hidden_layers"]
        self.bridge = self.config["dflash_config"]["selected_prefix_code_bridge"]
        self.first_exact = self.config["code_bridge_second_pass"]["first_exact"]
        self.hidden = self.config["hidden_size"]
        self.rank = self.bridge["rank"]
        self.code_size = self.bridge["code_size"]
        self.vocab_size = self.config["vocab_size"]

    def __enter__(self) -> "DraftSource":
        return self

    def __exit__(self, *exc) -> None:
        self.gguf.__exit__(*exc)

    def _tensor(self, name: str) -> TensorInfo:
        try:
            tensor = self.names[name]
        except KeyError as error:
            raise ValueError(f"{self.path}: missing source tensor {name!r}") from error
        return self.gguf.info(tensor)

    def has(self, name: str) -> bool:
        return name in self.names

    def describe(self, name: str) -> TensorSignature:
        if name == _EARLY_READ:
            return TensorSignature((self.first_exact, self.code_size, self.hidden), "BF16")
        if name == _FEATURE_WRITE:
            return TensorSignature((self.layers, self.hidden, self.rank), "BF16")
        if name == _READ_WEIGHT:
            info = self.gguf.info(f"blk.{self.first_exact}.dspine_read.weight")
            return TensorSignature(info.shape, TYPE_NAMES[info.type_id])
        if name == _OUT_WEIGHT:
            return TensorSignature((self.code_size, self.rank), "BF16")
        info = self._tensor(name)
        return TensorSignature(info.shape, TYPE_NAMES[info.type_id])

    def read_flat(
        self, name: str, begin: int = 0, end: int | None = None
    ) -> torch.Tensor:
        """Float32 values of the named tensor's flat C-order range, as the stores return them."""

        if name == _EARLY_READ:
            parts = [f"blk.{i}.dspine_read.weight" for i in range(self.first_exact)]
            return self._composite(parts, begin, end)
        if name == _FEATURE_WRITE:
            parts = [f"blk.{i}.dspine_write.weight" for i in range(self.layers)]
            return self._composite(parts, begin, end)
        if name == _READ_WEIGHT:
            info = self._tensor(f"blk.{self.first_exact}.dspine_read.weight")
            end = prod(info.shape) if end is None else end
            return self._flat(info, begin, end)
        if name == _OUT_WEIGHT:
            total = self.code_size * self.rank
            end = total if end is None else end
            if not 0 <= begin <= end <= total:
                raise ValueError(
                    f"{name}: source element range [{begin},{end}) exceeds {total}"
                )
            # The release's terminal projection is disabled: the reference
            # bridge requires this weight to be exactly zero.
            return torch.zeros(end - begin, dtype=torch.float32)
        info = self._tensor(name)
        end = prod(info.shape) if end is None else end
        return self._flat(info, begin, end)

    def _composite(self, parts: list[str], begin: int, end: int | None) -> torch.Tensor:
        pieces = [self._tensor(part) for part in parts]
        total = sum(prod(info.shape) for info in pieces)
        end = total if end is None else end
        if not 0 <= begin <= end <= total:
            raise ValueError(f"composite: range [{begin},{end}) exceeds {total}")
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        out = torch.empty(end - begin, dtype=torch.float32)
        offset = 0
        for info in pieces:
            size = prod(info.shape)
            lo, hi = max(begin, offset), min(end, offset + size)
            if lo < hi:
                out[lo - begin : hi - begin] = self._flat(info, lo - offset, hi - offset)
            offset += size
        return out

    def _flat(self, info: TensorInfo, begin: int, end: int) -> torch.Tensor:
        elements = prod(info.shape)
        if not 0 <= begin <= end <= elements:
            raise ValueError(
                f"{info.name}: source element range [{begin},{end}) exceeds {info.shape}"
            )
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        if info.type_id in _DIRECT_TYPES:
            flat = self.gguf.read_direct(info.name).reshape(-1)[begin:end]
            return torch.from_numpy(np.ascontiguousarray(flat))
        if len(info.shape) != 2:
            raise ValueError(f"{info.name}: {info.type_name} is not a matrix")
        columns = info.shape[1]
        first, last = begin // columns, -(-end // columns)
        values = self._dequantize(info, first, last).reshape(-1)
        window = values[begin - first * columns : end - first * columns]
        return torch.from_numpy(np.ascontiguousarray(window))

    def _dequantize(self, info: TensorInfo, first: int, last: int) -> np.ndarray:
        if info.type_id in _DIRECT_TYPES:
            return self.gguf.read_direct(info.name).reshape(last - first, -1)
        try:
            from gguf import GGMLQuantizationType
            from gguf.quants import dequantize
        except ImportError as error:
            raise ValueError(
                "reading the values of a GGUF block matrix needs the `gguf` package; "
                "importing its blocks does not"
            ) from error
        blocks = self.gguf.read_blocks(info.name, first, last)
        kind = GGMLQuantizationType(info.type_id)
        values = dequantize(np.ascontiguousarray(blocks), kind)
        return np.ascontiguousarray(values, dtype=np.float32).reshape(last - first, -1)

    def importable(self, name: str) -> LogicalSource | None:
        """The tensor as its own ggml blocks, or None when the artifact cannot carry it."""

        if name in (_EARLY_READ, _FEATURE_WRITE, _READ_WEIGHT, _OUT_WEIGHT):
            return None
        info = self._tensor(name)
        if info.type_id not in GGUF_FORMATS_BY_TYPE:
            return None
        return self._block_source(info)

    def pair_importable(
        self, first: str, second: str
    ) -> tuple[LogicalSource, LogicalSource] | None:
        """Two tensors of one merged parent, imported only as a single carried block type."""

        left, right = self._tensor(first), self._tensor(second)
        if (
            left.type_id != right.type_id
            or left.shape != right.shape
            or left.type_id not in GGUF_FORMATS_BY_TYPE
        ):
            return None
        return self._block_source(left), self._block_source(right)

    def _block_source(self, info: TensorInfo) -> LogicalSource:
        if len(info.shape) != 2:
            raise ValueError(f"{info.name}: a carried block tensor must be a matrix")
        format = GGUF_FORMATS_BY_TYPE[info.type_id].name

        def encoded(begin: int, end: int) -> EncodedRows:
            blocks = self.gguf.read_blocks(info.name, begin, end)
            return EncodedRows(
                format, torch.from_numpy(np.array(blocks, copy=True, order="C")), _EMPTY_SCALES
            )

        return LogicalSource(
            info.shape,
            f"{info.name}[{format}]{list(info.shape)}",
            lambda begin, end: self.read_flat(info.name, begin, end),
            encoded,
        )


def dspine_blocks(model, source) -> dict[str, LogicalSource]:
    """The DSpine parameters this source can supply as their own ggml blocks."""

    if not isinstance(source, DraftSource):
        return {}
    blocks: dict[str, LogicalSource] = {}
    item = source.importable("dflash/feature_projection")
    if item is not None:
        blocks["dflash/feature_projection"] = item
    layers = model.components["dflash"]["config"]["num_hidden_layers"]
    for layer in range(layers):
        for role in ("attention/output", "mlp/down"):
            name = f"dflash/layers/{layer}/{role}"
            item = source.importable(name)
            if item is not None:
                blocks[name] = item
        prefix = f"dflash/layers/{layer}/mlp/"
        pair = source.pair_importable(prefix + "gate", prefix + "up")
        if pair is not None:
            blocks[prefix + "gate"], blocks[prefix + "up"] = pair
    return blocks


__all__ = ["DraftSource", "TensorSignature", "dspine_blocks", "validate"]
