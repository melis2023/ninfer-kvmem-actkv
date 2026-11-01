"""FP64 reference forward of Gemma 4 26B-A4B (Gemma4ForCausalLM, text) over a llama.cpp GGUF.

An independent transcription of the text model in transformers' modeling_gemma4.py for golden
values: the embedding scaled by sqrt(H); per block the RMS-normed attention (sliding-window blocks
with their own heads, full-attention blocks whose keys double as values after an unweighted RMS
norm, proportional RoPE), the dense GeGLU MLP beside the routed experts (softmax over all experts,
top-k renormalized, per-expert output scale), the sandwich norms and the layer scalar; then the final
norm, the tied head and the logit soft cap. Every value is FP64 over the GGUF's exactly decoded
blocks, so the only difference from a real engine is the engine's own arithmetic.

    python -m tools.reference.gemma4 --gguf gemma-4-26B_q4_0-it.gguf --tokens ids.json \\
        [--layers 30] [--out reference.npz] [--assistant ASSISTANT.gguf --draft-steps 3]

`ids.json` is the token ids to run (a JSON list, <bos> first). It prints the NLL of every target
(token i + 1 given tokens 0..i) and their total; `--out` keeps every block's output, the routed
experts and the logits. 26 tokens take about three minutes on CPU, mostly decoding expert blocks.

`--assistant` also runs the multi-token-prediction assistant (transformers'
Gemma4AssistantForCausalLM with SinglePositionMultiTokenCandidateGenerator): from the target's
greedy next token (the anchor, at position n) and its final-normed hidden state at position n - 1,
each step projects [scaled target embedding; hidden] to the assistant's width, runs its blocks at
position n with queries only over the target's keys and values of positions 0..n-1 (the
sliding-window blocks the last sliding block's last window, the full-attention block the last full
block's), and takes the argmax of its tied head as the draft; the next step's hidden state is the
post-projection of its final-normed state. It prints the drafts and each step's top-2 margin.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools.convert.sources.gguf import TYPE_BF16, TYPE_F16, TYPE_F32, GGUFFile

DIRECT_TYPES = (TYPE_F32, TYPE_F16, TYPE_BF16)


def _dequantize(gguf: GGUFFile, name: str, row_begin: int = 0, row_end: int | None = None):
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    blocks = gguf.read_blocks(name, row_begin, row_end)
    kind = GGMLQuantizationType(gguf.info(name).type_id)
    values = dequantize(np.ascontiguousarray(blocks), kind)
    return np.asarray(values, dtype=np.float64).reshape(blocks.shape[0], -1)


class Weights:
    def __init__(self, gguf: GGUFFile):
        self.gguf = gguf

    def has(self, name: str) -> bool:
        return name in self.gguf.tensors

    def matrix(self, name: str, row_begin: int = 0, row_end: int | None = None) -> np.ndarray:
        return _dequantize(self.gguf, name, row_begin, row_end)

    def vector(self, name: str) -> np.ndarray:
        return self.gguf.read_direct(name).astype(np.float64).reshape(-1)

    def rows(self, name: str) -> np.ndarray:
        """A whole matrix, stored as float words or as ggml blocks."""
        if self.gguf.info(name).type_id in DIRECT_TYPES:
            info = self.gguf.info(name)
            return self.gguf.read_direct(name).astype(np.float64).reshape(info.shape[0], -1)
        return self.matrix(name)


def rms(x: np.ndarray, eps: float, w: np.ndarray | None = None) -> np.ndarray:
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y * w if w is not None else y


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


def forward(gguf: GGUFFile, ids: list[int], layers: int | None = None) -> dict[str, np.ndarray]:
    kv = gguf.kv
    w = Weights(gguf)
    hidden = int(kv["gemma4.embedding_length"])
    heads = int(kv["gemma4.attention.head_count"])
    kv_heads = kv["gemma4.attention.head_count_kv"]
    sliding = kv["gemma4.attention.sliding_window_pattern"]
    window = int(kv["gemma4.attention.sliding_window"])
    eps = float(kv["gemma4.attention.layer_norm_rms_epsilon"])
    top_k = int(kv["gemma4.expert_used_count"])
    width = int(kv["gemma4.expert_feed_forward_length"])
    cap = float(kv.get("gemma4.final_logit_softcapping", 0.0))
    blocks = int(kv["gemma4.block_count"]) if layers is None else layers
    n = len(ids)
    pos = np.arange(n)

    table = "token_embd.weight"
    x = np.stack([w.matrix(table, t, t + 1)[0] for t in ids]) * math.sqrt(hidden)
    out: dict[str, np.ndarray] = {"inp_scaled": x.copy()}
    for il in range(blocks):
        p = f"blk.{il}."
        full = not bool(sliding[il])
        d = int(kv["gemma4.attention.key_length" if full else "gemma4.attention.key_length_swa"])
        hkv = int(kv_heads[il] if isinstance(kv_heads, list) else kv_heads)
        half = d // 2
        a = rms(x, eps, w.vector(p + "attn_norm.weight"))
        q = (a @ w.matrix(p + "attn_q.weight").T).reshape(n, heads, d)
        k_raw = (a @ w.matrix(p + "attn_k.weight").T).reshape(n, hkv, d)
        v_raw = k_raw if full else (a @ w.matrix(p + "attn_v.weight").T).reshape(n, hkv, d)
        q = rms(q, eps, w.vector(p + "attn_q_norm.weight"))
        k = rms(k_raw, eps, w.vector(p + "attn_k_norm.weight"))
        v = rms(v_raw, eps)
        if full:
            base = float(kv["gemma4.rope.freq_base"])
            inv = base ** (-2.0 * np.arange(half) / d)
            if w.has("rope_freqs.weight"):
                inv = inv / w.vector("rope_freqs.weight")[:half]
            elif w.has(p + "rope_freqs.weight"):
                inv = inv / w.vector(p + "rope_freqs.weight")[:half]
        else:
            base = float(kv["gemma4.rope.freq_base_swa"])
            inv = base ** (-2.0 * np.arange(half) / d)
        angle = pos[:, None] * inv[None, :]
        cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]

        def rope(t: np.ndarray) -> np.ndarray:
            t1, t2 = t[..., :half], t[..., half:]
            return np.concatenate([t1 * cos - t2 * sin, t2 * cos + t1 * sin], -1)

        q, k = rope(q), rope(k)
        out[f"k-{il}"], out[f"v-{il}"] = k.copy(), v.copy()
        group = heads // hkv
        mask = np.tril(np.ones((n, n), dtype=bool))
        if not full:
            mask &= (pos[:, None] - pos[None, :]) < window
        attn = np.zeros((n, heads, d))
        for h in range(heads):
            s = q[:, h, :] @ k[:, h // group, :].T
            s = np.where(mask, s, -np.inf)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            attn[:, h, :] = s @ v[:, h // group, :]
        o = attn.reshape(n, heads * d) @ w.matrix(p + "attn_output.weight").T
        x = x + rms(o, eps, w.vector(p + "post_attention_norm.weight"))
        out[f"attn_out-{il}"] = x.copy()

        m = rms(x, eps, w.vector(p + "ffn_norm.weight"))
        gate = m @ w.matrix(p + "ffn_gate.weight").T
        up = m @ w.matrix(p + "ffn_up.weight").T
        dense = (gelu_tanh(gate) * up) @ w.matrix(p + "ffn_down.weight").T
        dense = rms(dense, eps, w.vector(p + "post_ffw_norm_1.weight"))

        router_in = rms(x, eps) / math.sqrt(hidden) * w.vector(p + "ffn_gate_inp.scale")
        logits = router_in @ w.rows(p + "ffn_gate_inp.weight").T
        prob = np.exp(logits - logits.max(-1, keepdims=True))
        prob /= prob.sum(-1, keepdims=True)
        chosen = np.argsort(-prob, axis=-1, kind="stable")[:, :top_k]
        scale = (
            w.vector(p + "ffn_down_exps.scale")
            if w.has(p + "ffn_down_exps.scale")
            else np.ones(prob.shape[1])
        )
        expert_in = rms(x, eps, w.vector(p + "pre_ffw_norm_2.weight"))
        moe = np.zeros_like(x)
        cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for t in range(n):
            weights = prob[t, chosen[t]]
            weights = weights / weights.sum()
            for j, e in enumerate(chosen[t]):
                e = int(e)
                if e not in cache:
                    cache[e] = (
                        w.matrix(p + "ffn_gate_up_exps.weight", e * 2 * width, (e + 1) * 2 * width),
                        w.matrix(p + "ffn_down_exps.weight", e * hidden, (e + 1) * hidden),
                    )
                gu = cache[e][0] @ expert_in[t]
                middle = gelu_tanh(gu[:width]) * gu[width:]
                moe[t] += weights[j] * (cache[e][1] @ middle) * scale[e]
        out[f"moe_ids-{il}"] = chosen
        out[f"moe_out-{il}"] = moe.copy()
        moe = rms(moe, eps, w.vector(p + "post_ffw_norm_2.weight"))
        layer_scale = (
            w.vector(p + "layer_output_scale.weight")
            if w.has(p + "layer_output_scale.weight")
            else np.ones(1)
        )
        x = (x + rms(dense + moe, eps, w.vector(p + "post_ffw_norm.weight"))) * layer_scale
        out[f"l_out-{il}"] = x.copy()

    if blocks == int(kv["gemma4.block_count"]):
        final = rms(x, eps, w.vector("output_norm.weight"))
        out["final"] = final.copy()
        vocab = gguf.info(table).shape[0]
        logits = np.zeros((n, vocab))
        for first in range(0, vocab, 16384):
            last = min(vocab, first + 16384)
            logits[:, first:last] = final @ w.matrix(table, first, last).T
        if cap > 0:
            logits = cap * np.tanh(logits / cap)
        out["logits"] = logits
    return out


def draft(gguf: GGUFFile, assistant: GGUFFile, target: dict[str, np.ndarray], steps: int):
    """The assistant's drafts after the target's greedy next token; see the module docstring."""

    tk, ak = gguf.kv, assistant.kv
    w, aw = Weights(gguf), Weights(assistant)
    hidden = int(tk["gemma4.embedding_length"])
    sliding_target = tk["gemma4.attention.sliding_window_pattern"]
    last_sliding = max(i for i, s in enumerate(sliding_target) if s)
    last_full = max(i for i, s in enumerate(sliding_target) if not s)
    window = int(tk["gemma4.attention.sliding_window"])
    a = "gemma4-assistant."
    heads = int(ak[a + "attention.head_count"])
    eps = float(ak[a + "attention.layer_norm_rms_epsilon"])
    sliding = ak[a + "attention.sliding_window_pattern"]
    n = target["final"].shape[0]
    anchor = int(np.argmax(target["logits"][n - 1]))
    h_in = target["final"][n - 1]
    drafts, margins = [], []
    for _ in range(steps):
        e = w.matrix("token_embd.weight", anchor, anchor + 1)[0] * math.sqrt(hidden)
        x = aw.matrix("nextn.pre_projection.weight") @ np.concatenate([e, h_in])
        for il in range(len(sliding)):
            p = f"blk.{il}."
            full = not bool(sliding[il])
            d = int(ak[a + ("attention.key_length" if full else "attention.key_length_swa")])
            half = d // 2
            src = last_full if full else last_sliding
            k, v = target[f"k-{src}"], target[f"v-{src}"]
            if not full:
                k, v = k[max(0, n - window):], v[max(0, n - window):]
            if full:
                inv = float(tk["gemma4.rope.freq_base"]) ** (-2.0 * np.arange(half) / d)
                if w.has("rope_freqs.weight"):
                    inv = inv / w.vector("rope_freqs.weight")[:half]
            else:
                inv = float(tk["gemma4.rope.freq_base_swa"]) ** (-2.0 * np.arange(half) / d)
            cos, sin = np.cos(n * inv), np.sin(n * inv)
            r = rms(x, eps, aw.vector(p + "attn_norm.weight"))
            q = (aw.matrix(p + "attn_q.weight") @ r).reshape(heads, d)
            q = rms(q, eps, aw.vector(p + "attn_q_norm.weight"))
            q = np.concatenate([q[:, :half] * cos - q[:, half:] * sin,
                                q[:, half:] * cos + q[:, :half] * sin], -1)
            group = heads // k.shape[1]
            attn = np.zeros((heads, d))
            for hh in range(heads):
                sc = k[:, hh // group, :] @ q[hh]
                sc = np.exp(sc - sc.max())
                attn[hh] = (sc / sc.sum()) @ v[:, hh // group, :]
            o = aw.matrix(p + "attn_output.weight") @ attn.reshape(-1)
            x = x + rms(o, eps, aw.vector(p + "post_attention_norm.weight"))
            m = rms(x, eps, aw.vector(p + "ffn_norm.weight"))
            gate = aw.matrix(p + "ffn_gate.weight") @ m
            up = aw.matrix(p + "ffn_up.weight") @ m
            mlp = aw.matrix(p + "ffn_down.weight") @ (gelu_tanh(gate) * up)
            x = (x + rms(mlp, eps, aw.vector(p + "post_ffw_norm.weight"))) * \
                aw.vector(p + "layer_output_scale.weight")
        final = rms(x, eps, aw.vector("output_norm.weight"))
        logits = aw.matrix("token_embd.weight") @ final
        top = np.argsort(-logits, kind="stable")[:2]
        anchor = int(top[0])
        drafts.append(anchor)
        margins.append(float(logits[top[0]] - logits[top[1]]))
        h_in = aw.matrix("nextn.post_projection.weight") @ final
    return drafts, margins


def target_nll(logits: np.ndarray, ids: list[int]) -> np.ndarray:
    top = logits.max(-1, keepdims=True)
    lse = top[:, 0] + np.log(np.exp(logits - top).sum(-1))
    n = len(ids)
    return lse[: n - 1] - logits[np.arange(n - 1), ids[1:]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path, help="JSON list of token ids")
    parser.add_argument("--layers", type=int, default=None, help="run only the first N blocks")
    parser.add_argument("--out", type=Path, default=None, help="write the tensors to this .npz")
    parser.add_argument("--assistant", type=Path, default=None, help="gemma4-assistant GGUF")
    parser.add_argument("--draft-steps", type=int, default=3)
    args = parser.parse_args()
    ids = [int(t) for t in json.loads(args.tokens.read_text())]
    gguf = GGUFFile(args.gguf)
    result = forward(gguf, ids, args.layers)
    if "logits" in result:
        nll = target_nll(result["logits"], ids)
        print(json.dumps({"tokens": len(ids), "total_nll": float(nll.sum()),
                          "nll": [round(float(v), 4) for v in nll]}))
    if args.assistant is not None:
        drafts, margins = draft(gguf, GGUFFile(args.assistant), result, args.draft_steps)
        print(json.dumps({"anchor": int(np.argmax(result["logits"][-1])), "drafts": drafts,
                          "margins": [round(m, 4) for m in margins]}))
    if args.out is not None:
        np.savez(args.out, **result)


if __name__ == "__main__":
    main()
