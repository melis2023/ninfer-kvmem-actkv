"""A DFlash2 draft adapter kept as its own GGUF.

The release ships the adapter both as a safetensors directory and as one ``-DFlash2`` GGUF
whose tensors are already exported in the shapes and row order the artifact wants. This module
reads that GGUF where the official recipes read the directory: the draft geometry is synthesised
from the file's ``dflash.*`` keys and passed through ``draft_config``, every logical name of the
adapter maps onto the ``fc``/``blk.*``/``selector_*`` tensor that holds it, and the projections
whose consumer walks whole ggml blocks -- the feature projection, the merged gate/up MLP parent
and the selector's hidden projection -- are imported byte for byte. Everything else takes the
official fallback, because its consumer asks for a specific encoding there: the fused attention
input projection (``prepare_attn_input_proj_weights``) and the two dynamic-convolution outputs
(``linear_dynamic_grouped_conv_add``) require Q8, the convolution kernels and the codebooks are
read as BF16, and the norms are direct tensors.

The file carries no ``qwen3_5`` metadata, so the gguf-py ``GGUFSource`` cannot describe it; a
small store wrapper presents the tensors under the names the adapter builder reads, which keeps
the ordinary lazy ``matrix_source``/``tensor_source`` resolution working. A row of a carried
block tensor is its own bytes, so only ``read_blocks`` and the ``gguf`` package's dequantiser
are needed, and only values -- never the imported blocks -- pay for the latter.
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
    ("fc.weight", "dflash2/feature_projection", "fc.weight"),
    ("enc.output_norm.weight", "dflash2/context_norm", "hidden_norm.weight"),
    ("output_norm.weight", "dflash2/final_norm", "norm.weight"),
    (
        "selector_hidden.weight",
        "dflash2/candidate_selector/hidden_projection",
        "candidate_selector.hidden_projection.weight",
    ),
    (
        "selector_predecessor.weight",
        "dflash2/candidate_selector/predecessor_codebook",
        "candidate_selector.predecessor_codebook",
    ),
    (
        "selector_successor.weight",
        "dflash2/candidate_selector/successor_codebook",
        "candidate_selector.successor_codebook",
    ),
)

_LAYER_TENSORS = (
    (
        "blk.{i}.attn_norm.weight",
        "dflash2/layers/{i}/input_norm",
        "layers.{i}.input_layernorm.weight",
    ),
    (
        "blk.{i}.ffn_norm.weight",
        "dflash2/layers/{i}/post_attention_norm",
        "layers.{i}.post_attention_layernorm.weight",
    ),
    (
        "blk.{i}.attn_q.weight",
        "dflash2/layers/{i}/attention/query",
        "layers.{i}.self_attn.q_proj.weight",
    ),
    (
        "blk.{i}.attn_k.weight",
        "dflash2/layers/{i}/attention/key",
        "layers.{i}.self_attn.k_proj.weight",
    ),
    (
        "blk.{i}.attn_v.weight",
        "dflash2/layers/{i}/attention/value",
        "layers.{i}.self_attn.v_proj.weight",
    ),
    (
        "blk.{i}.attn_output.weight",
        "dflash2/layers/{i}/attention/output",
        "layers.{i}.self_attn.o_proj.weight",
    ),
    (
        "blk.{i}.attn_q_norm.weight",
        "dflash2/layers/{i}/attention/query_norm",
        "layers.{i}.self_attn.q_norm.weight",
    ),
    (
        "blk.{i}.attn_k_norm.weight",
        "dflash2/layers/{i}/attention/key_norm",
        "layers.{i}.self_attn.k_norm.weight",
    ),
    (
        "blk.{i}.ffn_gate.weight",
        "dflash2/layers/{i}/mlp/gate",
        "layers.{i}.mlp.gate_proj.weight",
    ),
    ("blk.{i}.ffn_up.weight", "dflash2/layers/{i}/mlp/up", "layers.{i}.mlp.up_proj.weight"),
    (
        "blk.{i}.ffn_down.weight",
        "dflash2/layers/{i}/mlp/down",
        "layers.{i}.mlp.down_proj.weight",
    ),
    (
        "blk.{i}.attn_conv_base",
        "dflash2/layers/{i}/attention_conv/base_kernel",
        "layers.{i}.attention_conv.base_kernel",
    ),
    (
        "blk.{i}.attn_conv_proj.weight",
        "dflash2/layers/{i}/attention_conv/kernel_projection",
        "layers.{i}.attention_conv.kernel_projection.weight",
    ),
    (
        "blk.{i}.ffn_conv_base",
        "dflash2/layers/{i}/mlp_conv/base_kernel",
        "layers.{i}.mlp_conv.base_kernel",
    ),
    (
        "blk.{i}.ffn_conv_proj.weight",
        "dflash2/layers/{i}/mlp_conv/kernel_projection",
        "layers.{i}.mlp_conv.kernel_projection.weight",
    ),
)

# Adapter parameters whose consumer is a plain Linear over whole blocks, so a carried ggml type
# is imported unchanged. The gate/up pair is decided per layer, because only one merged parent of
# a single block type reaches gguf_swiglu's [gate; up] form.
_BLOCK_ROLES = ("dflash2/feature_projection", "dflash2/candidate_selector/hidden_projection")


def _aliases(layers: int) -> dict[str, str]:
    """Every name a source lookup may use, mapped to the GGUF tensor that holds it."""

    rows = list(_TOP_TENSORS)
    for layer in range(layers):
        rows.extend(tuple(item.format(i=layer) for item in entry) for entry in _LAYER_TENSORS)
    table: dict[str, str] = {}
    for tensor, logical, source in rows:
        for alias in (tensor, logical, source):
            table[alias] = tensor
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
    """Refuse a GGUF that is not a DFlash2 draft, and return its config and name aliases."""

    label = str(gguf.path)
    architecture = gguf.kv.get("general.architecture")
    if architecture != "dflash":
        raise ValueError(
            f"{label}: general.architecture = {architecture!r}, expected 'dflash'"
        )
    if gguf.kv.get("dflash.attention.causal") is not False:
        raise ValueError(f"{label}: the draft attention must be noncausal")
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
    layer_types = [
        "sliding_attention" if sliding else "full_attention" for sliding in pattern
    ]
    config = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "is_causal": False,
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
        "dflash_config": {
            # The GGUF counts the target's layers from one, the checkpoint from zero.
            "target_layer_ids": [index - 1 for index in taps],
            "mask_token_id": _integer(gguf.kv, "tokenizer.ggml.mask_token_id", label),
            "conv_kernel_size": _integer(gguf.kv, "dflash.conv_kernel_size", label),
            "conv_group_size": _integer(gguf.kv, "dflash.conv_group_size", label),
            "selector_rank": _integer(gguf.kv, "dflash.selector_rank", label),
            "selector_top_k": _integer(gguf.kv, "dflash.selector_top_k", label),
        },
    }
    if "sliding_attention" in layer_types:
        config["sliding_window"] = _integer(
            gguf.kv, "dflash.attention.sliding_window", label
        )
    names = _aliases(layers)
    expected = set(names.values())
    if set(gguf.tensors) != expected:
        missing = sorted(expected - set(gguf.tensors))[:5]
        extra = sorted(set(gguf.tensors) - expected)[:5]
        raise ValueError(
            f"{label}: tensor set mismatch (missing {missing}, extra {extra})"
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
    """A DFlash2 draft GGUF presented as a checkpoint directory.

    ``has``/``describe``/``read_flat`` speak the names the adapter builder reads, so the ordinary
    lazy source resolution works unchanged; ``importable`` returns the same tensor as its own
    ggml blocks where the recipe is allowed to import it.
    """

    def __init__(self, path: str | Path) -> None:
        self.gguf = GGUFFile(path)
        self.path = self.gguf.path
        self.config, self.names = validate(self.gguf)

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
        info = self._tensor(name)
        return TensorSignature(info.shape, TYPE_NAMES[info.type_id])

    def read_flat(
        self, name: str, begin: int = 0, end: int | None = None
    ) -> torch.Tensor:
        """Float32 values of the named tensor's flat C-order range, as the stores return them."""

        info = self._tensor(name)
        elements = prod(info.shape)
        end = elements if end is None else end
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
                format, torch.from_numpy(np.ascontiguousarray(blocks)), _EMPTY_SCALES
            )

        return LogicalSource(
            info.shape,
            f"{info.name}[{format}]{list(info.shape)}",
            lambda begin, end: self.read_flat(info.name, begin, end),
            encoded,
        )


def draft_blocks(model, source) -> dict[str, LogicalSource]:
    """The adapter parameters this source can supply as their own ggml blocks."""

    if not isinstance(source, DraftSource):
        return {}
    blocks: dict[str, LogicalSource] = {}
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name in _BLOCK_ROLES:
            item = source.importable(name)
            if item is not None:
                blocks[name] = item
    layers = model.components["dflash2"]["config"]["num_hidden_layers"]
    for layer in range(layers):
        prefix = f"dflash2/layers/{layer}/mlp/"
        pair = source.pair_importable(prefix + "gate", prefix + "up")
        if pair is not None:
            blocks[prefix + "gate"], blocks[prefix + "up"] = pair
    return blocks


__all__ = ["DraftSource", "TensorSignature", "draft_blocks", "validate"]
