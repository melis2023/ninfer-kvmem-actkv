"""FP64 reference forward of Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration), text path.

An independent transcription of transformers' modular_qwen4_exp.py (as summarised in
docs/maintainer/qwen3-8-flash-next-plan.md, section 1) for producing golden tensors: the PLE
hash and block, hyper-connection read/write, Gated DeltaNet, QSA (indexer and sparse attention),
the 512-expert MoE, the final mixer and the head. Tensors load lazily by name from the checkpoint's
safetensors shards, so a layer slice needs only that slice's tensors on disk.

    python -m tools.reference.qwen4_exp --checkpoint DIR --tokens 1,2,3 --layers 4 --out dump.pt

RoPE angles follow the reference's FP32 arithmetic (fl32(position) * fl32 frequency); everything
else is FP64. Single sequence, text only, no MTP.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

F64 = torch.float64
PREFIX = "model.language_model."


class Checkpoint:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.config = json.loads((self.root / "config.json").read_text())["text_config"]
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.files = index["weight_map"]
        self._open = {}

    def tensor(self, name: str, dtype=F64) -> torch.Tensor:
        file = self.files[name]
        if file not in self._open:
            self._open[file] = safe_open(str(self.root / file), framework="pt")
        value = self._open[file].get_tensor(name)
        return value if value.dtype == torch.int64 else value.to(dtype)

    def layer(self, index: int, name: str, dtype=F64) -> torch.Tensor:
        return self.tensor(f"{PREFIX}layers.{index}.{name}", dtype)


def rms0(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Zero-centred RMSNorm over the last dim: x / rms(x) * (1 + w)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight)


def rope(x: torch.Tensor, position: int, rotary: int = 64, theta: float = 1e7) -> torch.Tensor:
    """Rotate-half RoPE of the first `rotary` dims at a text position (all three MRoPE axes equal)."""
    pairs = rotary // 2
    frequency = (1.0 / (theta ** (torch.arange(0, rotary, 2, dtype=F64) / rotary))).to(torch.float32)
    angle = (torch.tensor(float(position), dtype=torch.float32) * frequency).to(F64)
    cos, sin = torch.cos(angle), torch.sin(angle)
    out = x.clone()
    x1, x2 = x[..., :pairs], x[..., pairs:rotary]
    out[..., :pairs] = x1 * cos - x2 * sin
    out[..., pairs:rotary] = x2 * cos + x1 * sin
    return out


# --- PLE ------------------------------------------------------------------------------------------

def ngram_rows(ckpt: Checkpoint, layer: int, context: list[int]) -> list[int]:
    """The 16 table rows of the token context[0] with predecessors context[1:] (EOS-cut already)."""
    multipliers = ckpt.layer(layer, "ple.ple_embedding.layer_multipliers").tolist()
    vocab = ckpt.layer(layer, "ple.ple_embedding.ngram_heads_vocab_sizes").tolist()
    offset = ckpt.layer(layer, "ple.ple_embedding.ngram_heads_offsets").tolist()
    per_order = int(ckpt.config["heads_per_ngram"])
    mixed = context[0] * multipliers[0]
    rows = []
    for order in range(2, int(ckpt.config["ngram_size"]) + 1):
        mixed ^= context[order - 1] * multipliers[order - 1]
        for slot in range(per_order):
            head = (order - 2) * per_order + slot
            rows.append(mixed % vocab[head] + offset[head])
    return rows


def ngram_contexts(tokens: list[int], eos: int, order: int) -> list[list[int]]:
    """Each token with its predecessors, newest first; an EOS predecessor cuts the older ones."""
    previous = [eos] * (order - 1)
    contexts = []
    for token in tokens:
        context, cut = [token], False
        for p in previous:
            context.append(eos if cut else p)
            cut = cut or p == eos
        contexts.append(context)
        previous = [token] + previous[:-1]
    return contexts


class NgramTable:
    """The checkpoint's BF16 n-gram table, read row by row from its 128 shards, or from the
    sparse ngram_rows.npz that tools/reference/fetch_slice.py writes beside a slice."""

    def __init__(self, ckpt: Checkpoint, layer: int):
        self.ckpt, self.layer = ckpt, layer
        self.rows_per_shard = None
        sparse = ckpt.root / "ngram_rows.npz"
        self.sparse = None
        if sparse.exists():
            import numpy as np
            self.sparse = {int(k): v for k, v in np.load(sparse).items()}

    def row(self, row: int) -> torch.Tensor:
        if self.sparse is not None:
            raw = torch.from_numpy(self.sparse[row].copy()).view(torch.bfloat16)
            return raw.to(F64)
        if self.rows_per_shard is None:
            first = f"{PREFIX}layers.{self.layer}.ple.ple_embedding.ngram_embedding.shard_0.weight"
            handle = safe_open(str(self.ckpt.root / self.ckpt.files[first]), framework="pt")
            self.rows_per_shard = handle.get_slice(first).get_shape()[0]
        shard, local = divmod(row, self.rows_per_shard)
        name = f"{PREFIX}layers.{self.layer}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
        handle = safe_open(str(self.ckpt.root / self.ckpt.files[name]), framework="pt")
        return handle.get_slice(name)[local : local + 1].to(F64)[0]


def ple(ckpt, layer, R, tokens, eos, table, eps, state):
    """R (T, HC, H) += G + conv for a sequence from its start; state carries 'history' (9, HC*H)."""
    hc, h = R.shape[1], R.shape[2]
    key = ckpt.layer(layer, "ple.key_proj.weight")
    value = ckpt.layer(layer, "ple.value_proj.weight")
    nk, nq, nc = (ckpt.layer(layer, f"ple.{n}.weight").view(hc, h) for n in ("norm_key", "norm_query", "norm_conv"))
    conv = ckpt.layer(layer, "ple.conv1d.weight").view(hc * h, -1)  # taps oldest first
    taps = conv.shape[1]
    dilation = int(ckpt.config["ngram_size"])
    out = R.clone()
    order = int(ckpt.config["ngram_size"])
    for t, context in enumerate(ngram_contexts(tokens, eos, order)):
        emb = torch.cat([table.row(r) for r in ngram_rows(ckpt, layer, context)])
        k = rms0((key @ emb).view(hc, h), nk, eps)
        v = value @ emb
        q = rms0(R[t], nq, eps)
        s = (k * q).sum(-1) / math.sqrt(h)
        s = torch.sign(s) * torch.sqrt(torch.clamp(s.abs(), min=1e-6))
        g = torch.sigmoid(s)[:, None] * v[None, :]
        n = rms0(g, nc, eps).reshape(-1)
        window = torch.cat([state["history"], n[None]])  # positions t-9 .. t
        x = sum(conv[:, j] * window[window.shape[0] - 1 - dilation * (taps - 1 - j)] for j in range(taps))
        out[t] = R[t] + g + torch.nn.functional.silu(x).view(hc, h)
        state["history"] = window[1:]
    return out


# --- hyper-connections ----------------------------------------------------------------------------

def hc_read(ckpt, prefix, R, eps):
    """Returns (mixed (T, H), inject weights (T, HC) or None)."""
    T, hc, h = R.shape
    norm = ckpt.tensor(prefix + "hc_norm.weight").view(hc, h)
    down = ckpt.tensor(prefix + "input_mix_weight_down.weight")
    up = ckpt.tensor(prefix + "input_mix_weight_up.weight")
    xn = rms0(R, norm, eps).reshape(T, hc * h)
    low = torch.nn.functional.silu((xn @ down.T) / hc)
    gate = torch.sigmoid(low @ up.T).view(T, hc, h)
    mixed = (gate * xn.view(T, hc, h)).sum(1) / hc
    inject_name = prefix + "block_inject_weight.weight"
    inject = None
    if inject_name in ckpt.files:
        inject = 2.0 * torch.sigmoid((xn @ ckpt.tensor(inject_name).T) / hc)
    return mixed, inject


def hc_write(R, y, inject):
    return R + y[:, None, :] * inject[:, :, None]


# --- mixers ---------------------------------------------------------------------------------------

def gdn(ckpt, layer, a, eps, state):
    cfg = ckpt.config
    nk, dk = int(cfg["linear_num_key_heads"]), int(cfg["linear_key_head_dim"])
    nv, dv = int(cfg["linear_num_value_heads"]), int(cfg["linear_value_head_dim"])
    w = lambda n: ckpt.layer(layer, "linear_attn." + n)
    qkv_all = a @ w("in_proj_qkv.weight").T                      # (T, 10240)
    z = (a @ w("in_proj_z.weight").T).view(-1, nv, dv)
    beta = torch.sigmoid(a @ w("in_proj_b.weight").T)
    g = -torch.exp(w("A_log")) * torch.nn.functional.softplus(a @ w("in_proj_a.weight").T + w("dt_bias"))
    conv = w("conv1d.weight")[:, 0, :]                            # (10240, 4)
    taps = conv.shape[1]
    norm = w("norm.weight")
    out = []
    for t in range(a.shape[0]):
        window = torch.cat([state["conv"], qkv_all[t][None]])     # last taps inputs
        state["conv"] = window[1:]
        x = torch.nn.functional.silu((window.T * conv).sum(-1))
        q = x[: nk * dk].view(nk, dk)
        k = x[nk * dk : 2 * nk * dk].view(nk, dk)
        v = x[2 * nk * dk :].view(nv, dv)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q, k = q.repeat_interleave(nv // nk, 0), k.repeat_interleave(nv // nk, 0)
        S = state["recurrent"] * torch.exp(g[t])[:, None, None]  # (nv, dk, dv)
        S = S + beta[t][:, None, None] * k[:, :, None] * (v - torch.einsum("hkv,hk->hv", S, k))[:, None, :]
        state["recurrent"] = S
        o = torch.einsum("hkv,hk->hv", S, q)
        y = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * norm * torch.sigmoid(z[t])
        out.append(y.reshape(-1))
    return torch.stack(out) @ w("out_proj.weight").T


def qsa(ckpt, layer, a, first_position, eps, state):
    cfg = ckpt.config
    heads, kv_heads, dim = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
    w = lambda n: ckpt.layer(layer, "self_attn." + n)
    ratio, top = int(cfg["indexer_compress_ratio"]), int(cfg["indexer_budget"]) // int(cfg["indexer_compress_ratio"])
    ih, idim = int(cfg["indexer_n_heads"]), int(cfg["indexer_head_dim"])
    qg = (a @ w("q_proj.weight").T).view(-1, heads, 2 * dim)
    q_all, gate_all = qg[..., :dim], qg[..., dim:]
    k_all = (a @ w("k_proj.weight").T).view(-1, kv_heads, dim)
    v_all = (a @ w("v_proj.weight").T).view(-1, kv_heads, dim)
    index_all = a @ w("indexer.index_qk_proj.weight").T
    out = []
    for t in range(a.shape[0]):
        p = first_position + t
        state["k"].append(rope(rms0(k_all[t], w("k_norm.weight"), eps), p))
        state["v"].append(v_all[t])
        state["index_keys"].append(index_all[t, ih * idim :])
        blocks = (p + 1) // ratio
        while len(state["pooled"]) < blocks:
            b = len(state["pooled"])
            mean = torch.stack(state["index_keys"][b * ratio : (b + 1) * ratio]).mean(0)
            state["pooled"].append(rope(rms0(mean, w("indexer.k_layernorm.weight"), eps), b * ratio))
        if blocks <= top:
            chosen = list(range(blocks))
        else:
            iq = rope(rms0(index_all[t, : ih * idim].view(ih, idim), w("indexer.q_layernorm.weight"), eps), p)
            pooled = torch.stack(state["pooled"])
            score = torch.relu(iq @ pooled.T).sum(0) / math.sqrt(idim)
            order = sorted(range(blocks), key=lambda b: (-float(score[b]), b))
            chosen = sorted(order[:top])
        positions = [b * ratio + i for b in chosen for i in range(ratio)] + list(range(blocks * ratio, p + 1))
        K = torch.stack([state["k"][j] for j in positions])        # (n, kv, dim)
        V = torch.stack([state["v"][j] for j in positions])
        q = rope(rms0(q_all[t], w("q_norm.weight"), eps), p)       # (heads, dim)
        group = heads // kv_heads
        o = []
        for h in range(heads):
            logits = (K[:, h // group] @ q[h]) / math.sqrt(dim)
            o.append(torch.softmax(logits, 0) @ V[:, h // group])
        o = torch.stack(o) * torch.sigmoid(gate_all[t])
        out.append(o.reshape(-1))
    return torch.stack(out) @ w("o_proj.weight").T


def moe(ckpt, layer, m):
    cfg = ckpt.config
    top, width = int(cfg["num_experts_per_tok"]), int(cfg["moe_intermediate_size"])
    w = lambda n: ckpt.layer(layer, "mlp." + n)
    logits = m @ w("gate.weight").T
    shared_gate = torch.sigmoid(m @ w("shared_expert_gate.weight").T)[:, 0]
    gate_up = w("experts.gate_up_proj")   # (E, 2*width, H)
    down = w("experts.down_proj")         # (E, H, width)
    sg, su, sd = w("shared_expert.gate_proj.weight"), w("shared_expert.up_proj.weight"), w("shared_expert.down_proj.weight")
    out = []
    for t in range(m.shape[0]):
        p = torch.softmax(logits[t], 0)
        order = sorted(range(p.shape[0]), key=lambda e: (-float(p[e]), e))[:top]
        weights = p[order] / p[order].sum()
        y = torch.zeros_like(m[t])
        for weight, e in zip(weights, order):
            gu = gate_up[e] @ m[t]
            y += weight * (down[e] @ (torch.nn.functional.silu(gu[:width]) * gu[width:]))
        y += shared_gate[t] * (sd @ (torch.nn.functional.silu(sg @ m[t]) * (su @ m[t])))
        out.append(y)
    return torch.stack(out)


# --- the stack ------------------------------------------------------------------------------------

def forward(ckpt: Checkpoint, tokens: list[int], layers: range, with_head: bool):
    """Blocks `layers` (a prefix 0 .. N-1) over one sequence from position 0."""
    cfg = ckpt.config
    eps = float(cfg["rms_norm_eps"])
    hc, h = int(cfg["hc_count"]), int(cfg["hidden_size"])
    eos = int(cfg["eos_token_id"])
    embed = ckpt.tensor(PREFIX + "embed_tokens.weight")[tokens]
    R = embed[:, None, :].repeat(1, hc, 1)
    ple_blocks = [i - 1 for i in cfg["ple_layer_ids"]]
    dumps = {"embed": embed, "R": []}
    for layer in layers:
        state = {}
        if layer in ple_blocks:
            table = NgramTable(ckpt, layer)
            pstate = {"history": torch.zeros(9, hc * h, dtype=F64)}
            R = ple(ckpt, layer, R, tokens, eos, table, eps, pstate)
        a, w1 = hc_read(ckpt, f"{PREFIX}layers.{layer}.attn_hyper_connection.", R, eps)
        if cfg["layer_types"][layer] == "linear_attention":
            nv, dk, dv = int(cfg["linear_num_value_heads"]), int(cfg["linear_key_head_dim"]), int(cfg["linear_value_head_dim"])
            conv_channels = 2 * int(cfg["linear_num_key_heads"]) * dk + nv * dv
            state = {"conv": torch.zeros(int(cfg["linear_conv_kernel_dim"]) - 1, conv_channels, dtype=F64),
                     "recurrent": torch.zeros(nv, dk, dv, dtype=F64)}
            y = gdn(ckpt, layer, a, eps, state)
        else:
            state = {"k": [], "v": [], "index_keys": [], "pooled": []}
            y = qsa(ckpt, layer, a, 0, eps, state)
        R = hc_write(R, y, w1)
        m, w2 = hc_read(ckpt, f"{PREFIX}layers.{layer}.mlp_hyper_connection.", R, eps)
        R = hc_write(R, moe(ckpt, layer, m), w2)
        dumps["R"].append(R.clone())
    if with_head:
        final, _ = hc_read(ckpt, PREFIX + "hyper_connection_mixer.", R, eps)
        dumps["logits"] = final @ ckpt.tensor("lm_head.weight").T
    return dumps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokens", required=True, help="comma-separated token ids")
    parser.add_argument("--layers", type=int, default=48, help="run blocks 0 .. N-1")
    parser.add_argument("--head", action="store_true", help="also apply the final mixer and lm_head")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list-ngram-rows", action="store_true",
                        help="print the n-gram table rows the tokens address and exit")
    args = parser.parse_args()
    if args.list_ngram_rows:
        ckpt = Checkpoint(args.checkpoint)
        tokens = [int(t) for t in args.tokens.split(",")]
        layer = int(ckpt.config["ple_layer_ids"][0]) - 1
        contexts = ngram_contexts(tokens, int(ckpt.config["eos_token_id"]), int(ckpt.config["ngram_size"]))
        rows = sorted({r for c in contexts for r in ngram_rows(ckpt, layer, c)})
        print(",".join(str(r) for r in rows))
        return
    if args.out is None:
        parser.error("--out is required")
    ckpt = Checkpoint(args.checkpoint)
    tokens = [int(t) for t in args.tokens.split(",")]
    dumps = forward(ckpt, tokens, range(args.layers), args.head)
    torch.save(dumps, args.out)
    # Raw little-endian FP32 copies for C++ readers: golden.json names them.
    root = args.out.parent
    files = []
    for layer, R in enumerate(dumps["R"]):
        name = f"golden_R{layer}.f32"
        R.to(torch.float32).numpy().tofile(root / name)
        files.append(name)
    meta = {"tokens": tokens, "layers": len(dumps["R"]), "R": files}
    if "logits" in dumps:
        dumps["logits"].to(torch.float32).numpy().tofile(root / "golden_logits.f32")
        meta["logits"] = "golden_logits.f32"
    (root / "golden.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    main()
