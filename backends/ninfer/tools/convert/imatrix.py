"""Activation importance for grouped quantization, imported from llama.cpp imatrix files.

An importance matrix holds, for every projection, the mean squared value of each input channel
over calibration text. ``grouped_search`` weights each weight's rounding error by its channel's
importance. llama.cpp's ``llama-imatrix`` writes GGUF tensors ``<tensor>.in_sum2`` and
``<tensor>.counts`` keyed by GGUF tensor names; this module maps them onto NInfer logical
parameter names and NInfer's input-channel order, and stores them as one safetensors file:

    python -m tools.convert.imatrix --gguf imatrix.gguf --config <checkpoint>/config.json \
        --out qwen3_8_27b.imatrix.safetensors

Only Dense text layers are mapped. Parameters without a vector (the embedding, the output head,
MTP and draft components) are searched without weights.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .sources.gguf import GGUFFile
from .sources.gguf_source import reorder_linear_attention_v_heads

# GGUF tensor whose input activation each NInfer text-layer role reads.
_GGUF_TENSOR = {
    "attention/query": "attn_q",
    "attention/gate": "attn_q",
    "attention/key": "attn_k",
    "attention/value": "attn_v",
    "attention/output": "attn_output",
    "gdn/query": "attn_qkv",
    "gdn/key": "attn_qkv",
    "gdn/value": "attn_qkv",
    "gdn/z": "attn_gate",
    "gdn/output": "ssm_out",
    "mlp/gate": "ffn_gate",
    "mlp/up": "ffn_up",
    "mlp/down": "ffn_down",
}


def from_gguf(path: Path, config: dict) -> dict[str, torch.Tensor]:
    """Map a llama.cpp imatrix onto ``text/layers/N/<role>`` importance vectors."""

    text = config.get("text_config", config)
    key_heads = int(text["linear_num_key_heads"])
    value_heads = int(text["linear_num_value_heads"])
    value_dim = int(text["linear_value_head_dim"])
    with GGUFFile(path) as gguf:
        sums, counts = {}, {}
        for name in gguf.tensors:
            if name.endswith(".in_sum2"):
                sums[name[: -len(".in_sum2")]] = gguf.read_direct(name).reshape(-1)
            elif name.endswith(".counts"):
                counts[name[: -len(".counts")]] = gguf.read_direct(name).reshape(-1)
        result = {}
        for layer, kind in enumerate(text["layer_types"]):
            prefix = "attention/" if kind == "full_attention" else "gdn/"
            for role, gguf_role in _GGUF_TENSOR.items():
                if not role.startswith((prefix, "mlp/")):
                    continue
                name = f"blk.{layer}.{gguf_role}.weight"
                if name not in sums:
                    continue
                count = counts.get(name)
                if count is None or count.size != 1:
                    raise ValueError(f"{path}: {name} needs one dense activation count")
                importance = torch.from_numpy(sums[name].astype("float32")) / max(
                    float(count[0]), 1.0
                )
                if role == "gdn/output":
                    # ssm_out's input columns are llama.cpp's tiled value heads; the inverse
                    # reorder swaps the two head counts (see reorder_linear_attention_v_heads).
                    importance = reorder_linear_attention_v_heads(
                        importance,
                        tuple(importance.shape),
                        0,
                        value_heads // key_heads,
                        key_heads,
                        value_dim,
                    )
                result[f"text/layers/{layer}/{role}"] = importance.contiguous()
    return result


@lru_cache(maxsize=2)
def load(path: str) -> dict[str, torch.Tensor]:
    """Importance vectors by logical parameter name, from a file written by this module."""

    return load_file(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", type=Path, required=True, help="llama-imatrix output")
    parser.add_argument("--config", type=Path, required=True, help="checkpoint config.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    vectors = from_gguf(args.gguf, json.loads(args.config.read_text()))
    if not vectors:
        raise ValueError(f"{args.gguf}: no text-layer importance vectors matched")
    save_file(vectors, str(args.out), metadata={"source": args.gguf.name})
    print(f"{len(vectors)} parameters -> {args.out}")


if __name__ == "__main__":
    main()
