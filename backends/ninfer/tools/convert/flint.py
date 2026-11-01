"""Flint (FlintForCausalLM) from a llama.cpp GGUF, kept in its blocks.

Flint is a 49-block MoE transformer: grouped-query attention with per-head QK RMS norms and a
learned softmax sink per head, sliding-window blocks with RoPE and full-attention blocks without
it, two leading dense SwiGLU blocks, then routed SwiGLU experts (sigmoid scores, top-k
renormalized) beside an ungated shared SwiGLU expert, and an untied output head. The recipe imports
the GGUF's blocks row for row, as stored: a row of the artifact is byte for byte a row of the GGUF.
llama.cpp stores the experts' gate and up projections as two banks; the artifact interleaves them
per expert ([gate rows; up rows] of expert 0, then expert 1, ...), which moves whole rows only.
The router (BF16 in the GGUF), the norms and the sinks are FP32 vectors.

    python -m tools.convert --model HF_DIR --recipe flint_gguf \\
        --source gguf=flint-preview-UD-Q4_K_XL.gguf --out flint-preview-UD-Q4_K_XL.ninfer

`HF_DIR` holds the release's `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja` and `generation_config.json` (no weights are read from it).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .gguf_blocks import block_source
from .methods import cast_direct, import_encoded
from .resources import token_domain
from .sources.gguf import GGUFFile
from .sources.logical import EncodedRows, LogicalSource, array_source, select_rows
from .ternary import rows

ARCHITECTURES = ("FlintForCausalLM",)
RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
# llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID.
SIGMOID_GATING = 2


def _int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def text_config(source: dict) -> dict:
    """Flint's config, normalised for src/models/flint."""

    if source.get("architectures") != list(ARCHITECTURES) or source.get("model_type") != "flint":
        raise ValueError(f"unsupported Flint architecture {source.get('architectures')!r}")
    if source.get("hidden_act") != "silu":
        raise ValueError("Flint uses SwiGLU (hidden_act silu)")
    if source.get("attention_bias"):
        raise ValueError("attention biases are not implemented")
    if source.get("router_score_function") != "sigmoid" or source.get("norm_topk_prob") is not True:
        raise ValueError("Flint routes by sigmoid scores renormalized over the chosen experts")
    if source.get("softmax_type") != "learnable":
        raise ValueError("Flint's attention has a learned sink per head (softmax_type learnable)")
    if source.get("decoder_sparse_step", 1) != 1:
        raise ValueError("every block after the dense ones must be a MoE block")
    if source.get("rope_scaling") or source.get("rope_parameters", {}).get("rope_type", "default") \
            not in ("default",):
        raise ValueError("Flint's RoPE is unscaled")
    if source.get("tie_word_embeddings") is not False:
        raise ValueError("Flint's output head is untied from its token table")
    out = {"architectures": list(ARCHITECTURES), "model_type": "flint"}
    for key in ("hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings",
                "num_attention_heads", "num_key_value_heads", "head_dim", "sliding_window",
                "intermediate_size", "num_experts", "num_experts_per_tok",
                "moe_intermediate_size", "shared_expert_intermediate_size"):
        out[key] = _int(source.get(key), key)
    layers = source.get("layer_types")
    if (not isinstance(layers, list) or len(layers) != out["num_hidden_layers"]
            or any(k not in ("sliding_attention", "full_attention") for k in layers)):
        raise ValueError("layer_types must name every block")
    out["layer_types"] = list(layers)
    no_rope = source.get("no_rope_layer_indices", [])
    if (not isinstance(no_rope, list) or len(set(no_rope)) != len(no_rope)
            or any(type(i) is not int or not 0 <= i < len(layers) for i in no_rope)):
        raise ValueError("no_rope_layer_indices must be distinct block indices")
    out["rope_layers"] = [i not in no_rope for i in range(len(layers))]
    dense = source.get("mlp_only_layers", [])
    if (not isinstance(dense, list)
            or sorted(dense) != list(range(len(dense))) or len(dense) >= len(layers)):
        raise ValueError("the dense blocks (mlp_only_layers) must be the leading blocks")
    out["dense_layers"] = len(dense)
    if out["num_attention_heads"] % out["num_key_value_heads"]:
        raise ValueError("query heads must be a multiple of the key/value heads")
    if out["num_experts_per_tok"] > out["num_experts"]:
        raise ValueError("more experts per token than experts")
    out["rope_theta"] = float(source.get("rope_theta", 10000.0))
    out["rms_norm_eps"] = float(source.get("rms_norm_eps", 1e-6))
    out["tie_word_embeddings"] = False
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in source:
            out[key] = source[key]
    return out


def validate(gguf: GGUFFile, config: dict) -> None:
    """The GGUF's architecture metadata against the release's config.json."""

    kv = gguf.kv
    if kv.get("general.architecture") != "flint":
        raise ValueError(f"{gguf.path}: expected a llama.cpp flint GGUF")
    f = "flint."
    expected = {
        "block_count": config["num_hidden_layers"],
        "embedding_length": config["hidden_size"],
        "feed_forward_length": config["intermediate_size"],
        "attention.head_count": config["num_attention_heads"],
        "attention.head_count_kv": config["num_key_value_heads"],
        "attention.key_length": config["head_dim"],
        "attention.value_length": config["head_dim"],
        "attention.sliding_window": config["sliding_window"],
        "expert_count": config["num_experts"],
        "expert_used_count": config["num_experts_per_tok"],
        "expert_feed_forward_length": config["moe_intermediate_size"],
        "expert_shared_feed_forward_length": config["shared_expert_intermediate_size"],
        "leading_dense_block_count": config["dense_layers"],
        "expert_gating_func": SIGMOID_GATING,
        "expert_weights_norm": True,
    }
    for key, value in expected.items():
        if kv.get(f + key) != value:
            raise ValueError(f"{gguf.path}: {f + key} is {kv.get(f + key)!r}, the config's "
                             f"{value!r}")
    if float(kv.get(f + "expert_weights_scale", 1.0)) != 1.0:
        raise ValueError(f"{gguf.path}: scaled expert weights are not implemented")
    sliding = [kind == "sliding_attention" for kind in config["layer_types"]]
    if list(kv.get(f + "attention.sliding_window_pattern", [])) != sliding:
        raise ValueError(f"{gguf.path}: sliding_window_pattern differs from layer_types")
    if list(kv.get(f + "attention.rope_pattern", [])) != config["rope_layers"]:
        raise ValueError(f"{gguf.path}: rope_pattern differs from no_rope_layer_indices")
    if float(kv.get(f + "rope.freq_base", 0.0)) != config["rope_theta"]:
        raise ValueError(f"{gguf.path}: rope.freq_base differs from rope_theta")
    eps = float(kv.get(f + "attention.layer_norm_rms_epsilon", 0.0))
    if not np.isclose(eps, config["rms_norm_eps"], rtol=1e-6, atol=0.0):
        raise ValueError(f"{gguf.path}: layer_norm_rms_epsilon differs from rms_norm_eps")
    for i in range(config["dense_layers"], config["num_hidden_layers"]):
        name = f"blk.{i}.ffn_exp_probs_b.bias"
        if name in gguf.tensors:
            raise ValueError(f"{gguf.path}: {name}: an expert selection bias is not implemented")


def _load_resources(root: Path, vocab: int, overrides: dict):
    references, payloads = {}, {}
    for role in RESOURCES:
        path = Path(overrides[role]) if role in overrides else root / role
        data = path.read_bytes()
        if not data:
            raise ValueError(f"{path}: resource is empty")
        object_id = f"resource/text/{role}"
        references[role] = object_id
        payloads[object_id] = data
    tokenizer = json.loads(payloads["resource/text/tokenizer.json"])
    tokenizer_config = json.loads(payloads["resource/text/tokenizer_config.json"])
    count, special = token_domain(tokenizer, tokenizer_config, vocab)
    return references, payloads, count, special


def _direct(gguf: GGUFFile, tensor: str, shape: tuple[int, ...]) -> LogicalSource:
    """F32 as stored, BF16 and F16 widened exactly."""

    info = gguf.info(tensor)
    if info.type_name not in ("F32", "BF16", "F16"):
        raise ValueError(f"{tensor}: expected a float vector or matrix, found {info.type_name}")
    values = gguf.read_direct(tensor).astype(np.float32).reshape(shape)
    return array_source(torch.from_numpy(np.ascontiguousarray(values)), tensor)


def _matrix(gguf: GGUFFile, tensor: str, shape: tuple[int, int]) -> LogicalSource:
    info = gguf.info(tensor)
    if info.type_id not in GGUF_FORMATS_BY_TYPE:
        raise ValueError(f"{tensor}: expected ggml block rows, found {info.type_name}")
    stored = (int(np.prod(info.shape[:-1])), info.shape[-1])
    if stored != shape:
        raise ValueError(f"{tensor}: stored {info.shape}, expected rows x columns {shape}")
    return block_source(gguf, tensor, shape, rows())


def _stacked(parts: list[LogicalSource], label: str) -> LogicalSource:
    """The rows of `parts` one after the other."""

    k = parts[0].shape[1]
    if any(len(p.shape) != 2 or p.shape[1] != k for p in parts):
        raise ValueError(f"{label}: stacked parts must be matrices of one width")
    starts = np.cumsum([0] + [p.shape[0] for p in parts]).tolist()

    def pieces(begin: int, end: int):
        for part, low, high in zip(parts, starts, starts[1:]):
            a, b = max(begin, low), min(end, high)
            if a < b:
                yield part, a - low, b - low

    def values(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        flat = torch.cat([p.values(a * k, b * k) for p, a, b in pieces(first, last)])
        return flat[begin - first * k:end - first * k]

    def encoded(begin: int, end: int) -> EncodedRows:
        blocks = [p.read_encoded(a, b) for p, a, b in pieces(begin, end)]
        if len({b.format for b in blocks}) != 1:
            raise ValueError(f"{label}: stacked parts are stored in different formats")
        if len(blocks) == 1:
            return blocks[0]
        return EncodedRows(blocks[0].format, torch.cat([b.codes for b in blocks]),
                           torch.cat([b.scales for b in blocks]))

    return LogicalSource((starts[-1], k), label, values, encoded)


def expert_gate_up(gguf: GGUFFile, layer: int, experts: int, width: int,
                   hidden: int) -> LogicalSource:
    """`ffn_gate_exps` and `ffn_up_exps` of a block as one bank, each expert's gate rows followed by
    its up rows."""

    g = f"blk.{layer}."
    gate = _matrix(gguf, g + "ffn_gate_exps.weight", (experts * width, hidden))
    up = _matrix(gguf, g + "ffn_up_exps.weight", (experts * width, hidden))
    if gguf.info(g + "ffn_gate_exps.weight").type_id != gguf.info(g + "ffn_up_exps.weight").type_id:
        raise ValueError(f"{g}: expert gate and up banks are stored in different formats")
    both = _stacked([gate, up], f"{g}ffn_gate_exps+ffn_up_exps")
    up_base = experts * width
    spans = []
    for e in range(experts):
        spans += [(e * width, (e + 1) * width), (up_base + e * width, up_base + (e + 1) * width)]
    return select_rows(both, tuple(spans))


def build_model(base, gguf: GGUFFile, *, resource_overrides=None):
    """Flint's text component, every parameter's source the GGUF."""

    from .model import Model, Parameter

    config = text_config(base.config)
    validate(gguf, config)
    h, vocab = config["hidden_size"], config["vocab_size"]
    references, payloads, count, special = _load_resources(
        base.root, vocab, dict(resource_overrides or {}))
    model = Model({"text": {"config": config, "resources": references}}, resources=payloads,
                  token_count=count, special_token_ids=special)

    def matrix(name, tensor, shape, inputs=(), source=None):
        model.add(Parameter(name, shape, source or _matrix(gguf, tensor, shape),
                            inputs=tuple(inputs)))

    def vector(name, tensor, shape):
        model.add(Parameter(name, shape, _direct(gguf, tensor, shape), direct_format="fp32"))

    matrix("text/token_embedding", "token_embd.weight", (vocab, h))
    matrix("text/output_head", "output.weight", (vocab, h), ("text/final_hidden",))
    vector("text/final_norm", "output_norm.weight", (h,))
    heads, kv_heads, d = (config["num_attention_heads"], config["num_key_value_heads"],
                          config["head_dim"])
    for i in range(config["num_hidden_layers"]):
        g, p = f"blk.{i}.", f"text/layers/{i}/"
        a = p + "attention/"
        matrix(a + "query", g + "attn_q.weight", (heads * d, h), (p + "mixer_input",))
        matrix(a + "key", g + "attn_k.weight", (kv_heads * d, h), (p + "mixer_input",))
        matrix(a + "value", g + "attn_v.weight", (kv_heads * d, h), (p + "mixer_input",))
        matrix(a + "output", g + "attn_output.weight", (h, heads * d), (p + "attention_output",))
        vector(a + "query_norm", g + "attn_q_norm.weight", (d,))
        vector(a + "key_norm", g + "attn_k_norm.weight", (d,))
        vector(a + "sinks", g + "attn_sinks.weight", (heads,))
        vector(p + "input_norm", g + "attn_norm.weight", (h,))
        vector(p + "ffn_norm", g + "ffn_norm.weight", (h,))
        if i < config["dense_layers"]:
            width = config["intermediate_size"]
            matrix(p + "mlp/gate", g + "ffn_gate.weight", (width, h), (p + "mlp_input",))
            matrix(p + "mlp/up", g + "ffn_up.weight", (width, h), (p + "mlp_input",))
            matrix(p + "mlp/down", g + "ffn_down.weight", (h, width), (p + "mlp_product",))
            continue
        e, w = config["num_experts"], config["moe_intermediate_size"]
        s = config["shared_expert_intermediate_size"]
        m = p + "moe/"
        vector(m + "router", g + "ffn_gate_inp.weight", (e, h))
        matrix(m + "gate_up", None, (e * 2 * w, h), (m + "input",),
               expert_gate_up(gguf, i, e, w, h))
        matrix(m + "down", g + "ffn_down_exps.weight", (e * h, w), (m + "product",))
        matrix(m + "shared_gate", g + "ffn_gate_shexp.weight", (s, h), (m + "input",))
        matrix(m + "shared_up", g + "ffn_up_shexp.weight", (s, h), (m + "input",))
        matrix(m + "shared_down", g + "ffn_down_shexp.weight", (h, s), (m + "shared_product",))
    return model


def flint_gguf(model, recipe, sources):
    """Every matrix keeps its GGUF blocks; vectors are FP32. Q, K and V of a block share one parent
    when the GGUF stores them in one format, as do the dense MLP's and the shared expert's gate and
    up rows."""

    if model.config.get("architectures") != list(ARCHITECTURES):
        raise ValueError("the flint_gguf recipe requires a Flint model")
    formats = {}
    for name, parameter in model.parameters.items():
        if parameter.direct_format == "fp32" and not parameter.projection:
            recipe.assign(name, format="fp32", method=cast_direct, source=parameter.source)
            continue
        fmt = formats[name] = parameter.source.read_encoded(0, 1).format
        recipe.assign(name, format=fmt, method=import_encoded, source=parameter.source,
                      **({"activation_policy": "AllowA8"} if parameter.projection else {}))
    config = model.config
    for i in range(config["num_hidden_layers"]):
        p = f"text/layers/{i}/"
        a = p + "attention/"
        group = [a + "query", a + "key", a + "value"]
        if len({formats[name] for name in group}) == 1:
            recipe.group(group)
        pair = ([p + "mlp/gate", p + "mlp/up"] if i < config["dense_layers"]
                else [p + "moe/shared_gate", p + "moe/shared_up"])
        if formats[pair[0]] != formats[pair[1]]:
            raise ValueError(f"{p}: gate and up are stored in different formats")
        recipe.group(pair)


RECIPES = {"flint_gguf": flint_gguf}
