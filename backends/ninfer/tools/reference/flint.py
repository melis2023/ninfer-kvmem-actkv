"""FP64 reference forward of Flint (FlintForCausalLM) over a llama.cpp GGUF.

An independent transcription of RecursiveSI's Flint (the release's SGLang model and llama.cpp's
`flint` graph) for golden values: per block the RMS-normed grouped-query attention with per-head
QK RMS norms, NeoX RoPE on the sliding-window blocks only (full-attention blocks have no positional
encoding), a sliding window of `sliding_window` positions including the query's own, and a learned
sink logit per head that joins each softmax denominator without a value; then the RMS-normed
SwiGLU MLP: dense in the leading blocks, otherwise the top-k experts by sigmoid router score with
their scores renormalized to sum to one, plus the ungated shared expert. Last the final norm and
the untied head. Every value is FP64 over the GGUF's exactly decoded blocks.

    python -m tools.reference.flint --gguf flint-preview-UD-Q4_K_XL.gguf --tokens ids.json \\
        [--layers 4] [--out reference.npz]

`ids.json` is the token ids to run (a JSON list). It prints the NLL of every target (token i + 1
given tokens 0..i) and their total, and the largest magnitude of the experts' SwiGLU products (the
down projections' inputs); `--out` keeps every block's output, the routed experts and the logits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.convert.sources.gguf import GGUFFile

from .gemma4 import Weights, rms, target_nll


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, sinks: np.ndarray, mask: np.ndarray,
              scale: float) -> np.ndarray:
    """[n, heads, d] queries over [n, kv_heads, d] keys and values with one sink logit per head."""

    n, heads, d = q.shape
    group = heads // k.shape[1]
    out = np.zeros((n, heads, d))
    for h in range(heads):
        s = (q[:, h, :] @ k[:, h // group, :].T) * scale
        s = np.where(mask, s, -np.inf)
        top = np.maximum(s.max(-1, keepdims=True), sinks[h])
        p = np.exp(s - top)
        denominator = p.sum(-1, keepdims=True) + np.exp(sinks[h] - top)
        out[:, h, :] = (p / denominator) @ v[:, h // group, :]
    return out


def forward(gguf: GGUFFile, ids: list[int], layers: int | None = None) -> dict[str, np.ndarray]:
    kv = gguf.kv
    f = "flint."
    w = Weights(gguf)
    heads = int(kv[f + "attention.head_count"])
    d = int(kv[f + "attention.key_length"])
    sliding = kv[f + "attention.sliding_window_pattern"]
    rope_layers = kv[f + "attention.rope_pattern"]
    window = int(kv[f + "attention.sliding_window"])
    eps = float(kv[f + "attention.layer_norm_rms_epsilon"])
    base = float(kv[f + "rope.freq_base"])
    top_k = int(kv[f + "expert_used_count"])
    width = int(kv[f + "expert_feed_forward_length"])
    dense_blocks = int(kv[f + "leading_dense_block_count"])
    hidden = int(kv[f + "embedding_length"])
    blocks = int(kv[f + "block_count"]) if layers is None else layers
    n = len(ids)
    pos = np.arange(n)
    half = d // 2
    inv = base ** (-2.0 * np.arange(half) / d)
    angle = pos[:, None] * inv[None, :]
    cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]

    def rope(t: np.ndarray) -> np.ndarray:
        t1, t2 = t[..., :half], t[..., half:]
        return np.concatenate([t1 * cos - t2 * sin, t2 * cos + t1 * sin], -1)

    x = np.stack([w.matrix("token_embd.weight", t, t + 1)[0] for t in ids])
    out: dict[str, np.ndarray] = {"embedding": x.copy()}
    causal = np.tril(np.ones((n, n), dtype=bool))
    middle_max = 0.0
    for il in range(blocks):
        p = f"blk.{il}."
        a = rms(x, eps, w.vector(p + "attn_norm.weight"))
        q = (a @ w.matrix(p + "attn_q.weight").T).reshape(n, heads, d)
        k = (a @ w.matrix(p + "attn_k.weight").T).reshape(n, -1, d)
        v = (a @ w.matrix(p + "attn_v.weight").T).reshape(n, -1, d)
        q = rms(q, eps, w.vector(p + "attn_q_norm.weight"))
        k = rms(k, eps, w.vector(p + "attn_k_norm.weight"))
        if rope_layers[il]:
            q, k = rope(q), rope(k)
        out[f"k-{il}"], out[f"v-{il}"] = k.copy(), v.copy()
        mask = causal & ((pos[:, None] - pos[None, :]) < window) if sliding[il] else causal
        attn = attention(q, k, v, w.vector(p + "attn_sinks.weight"), mask, 1.0 / math.sqrt(d))
        x = x + attn.reshape(n, heads * d) @ w.matrix(p + "attn_output.weight").T
        out[f"attn_out-{il}"] = x.copy()

        y = rms(x, eps, w.vector(p + "ffn_norm.weight"))
        if il < dense_blocks:
            middle = silu(y @ w.matrix(p + "ffn_gate.weight").T) * (y @ w.matrix(p + "ffn_up.weight").T)
            x = x + middle @ w.matrix(p + "ffn_down.weight").T
            out[f"l_out-{il}"] = x.copy()
            continue
        logits = y @ w.rows(p + "ffn_gate_inp.weight").T
        score = 1.0 / (1.0 + np.exp(-logits))
        chosen = np.argsort(-score, axis=-1, kind="stable")[:, :top_k]
        moe = np.zeros_like(x)
        cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for t in range(n):
            weights = score[t, chosen[t]]
            weights = weights / (weights.sum() + 1e-20)
            for j, e in enumerate(chosen[t]):
                e = int(e)
                if e not in cache:
                    cache[e] = (
                        w.matrix(p + "ffn_gate_exps.weight", e * width, (e + 1) * width),
                        w.matrix(p + "ffn_up_exps.weight", e * width, (e + 1) * width),
                        w.matrix(p + "ffn_down_exps.weight", e * hidden, (e + 1) * hidden),
                    )
                gate, up, down = cache[e]
                middle = silu(gate @ y[t]) * (up @ y[t])
                middle_max = max(middle_max, float(np.abs(middle).max()))
                moe[t] += weights[j] * (down @ middle)
        shared = silu(y @ w.matrix(p + "ffn_gate_shexp.weight").T) * \
            (y @ w.matrix(p + "ffn_up_shexp.weight").T)
        middle_max = max(middle_max, float(np.abs(shared).max()))
        moe += shared @ w.matrix(p + "ffn_down_shexp.weight").T
        out[f"moe_ids-{il}"] = chosen
        out[f"moe_weights-{il}"] = np.take_along_axis(score, chosen, -1)
        x = x + moe
        out[f"l_out-{il}"] = x.copy()

    out["middle_max"] = np.array(middle_max)
    if blocks == int(kv[f + "block_count"]):
        final = rms(x, eps, w.vector("output_norm.weight"))
        out["final"] = final.copy()
        vocab = gguf.info("output.weight").shape[0]
        logits = np.zeros((n, vocab))
        for first in range(0, vocab, 16384):
            last = min(vocab, first + 16384)
            logits[:, first:last] = final @ w.matrix("output.weight", first, last).T
        out["logits"] = logits
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path, help="JSON list of token ids")
    parser.add_argument("--layers", type=int, default=None, help="run only the first N blocks")
    parser.add_argument("--out", type=Path, default=None, help="write the tensors to this .npz")
    args = parser.parse_args()
    ids = [int(t) for t in json.loads(args.tokens.read_text())]
    result = forward(GGUFFile(args.gguf), ids, args.layers)
    report = {"tokens": len(ids), "middle_max": float(result["middle_max"])}
    if "logits" in result:
        nll = target_nll(result["logits"], ids)
        report |= {"total_nll": float(nll.sum()), "nll": [round(float(v), 4) for v in nll],
                   "argmax": [int(i) for i in result["logits"].argmax(-1)]}
    print(json.dumps(report))
    if args.out is not None:
        np.savez(args.out, **result)


if __name__ == "__main__":
    main()
