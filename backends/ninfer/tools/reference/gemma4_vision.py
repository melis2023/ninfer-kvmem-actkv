"""FP64 reference of Gemma 4's vision tower (Gemma4VisionModel and the projection into the text
model) over a llama.cpp `clip` GGUF with projector `gemma4v`.

An independent transcription of transformers' modeling_gemma4.py for golden values: the patch
projection plus the x and y position rows; per block the sandwich RMS norms (gammas applied as w),
the per-head Q and K norms with their gammas and the unscaled V norm, the axial RoPE (the head's first
half turns by the patch's column, the second by its row, each half its pairs (i, i + D/4) at
theta^(-2i / (D/2))), attention with scale 1 over the unit's own patches, the tanh-GELU GeGLU; then
the average over pooling x pooling patches, times sqrt(hidden), the standardization, an unscaled RMS
norm and the projection. Every value is FP64 over the GGUF's BF16/F32 values.

    python -m tools.reference.gemma4_vision --mmproj MMPROJ.gguf --config config.json \\
        --patches patches.npy --grid W H [--units N] --out soft.npy

`--patches` holds the engine's input, BF16 words [units * H * W, 3 * patch * patch] (a .npy array,
or the raw words of a .bin file such as tests/models/gemma4/test_vision_real.cpp writes), each
patch's values (channel, y, x) with x fastest and the patches row by row; `--out` receives the soft
tokens [units * (H / pooling) * (W / pooling), text hidden] in FP64 (.npy, or raw for .bin).

`--transformers HF_DIR` also runs transformers' own Gemma4VisionModel and embedder in FP32 on the
same pixels with the release's safetensors weights (needs torch and transformers), and prints how
far it lies from the FP64 values: a check of this transcription. The embedder projects with the
GGUF's matrix: the QAT release's mmproj carries its own projection into the text model, which
differs from the BF16 release's (the script prints by how much); the tower's tensors are the same.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.convert.sources.gguf import GGUFFile


def rms(x: np.ndarray, eps: float, w: np.ndarray | None = None) -> np.ndarray:
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def bf16_words(words: np.ndarray) -> np.ndarray:
    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def tower(mmproj: GGUFFile, config: dict, patches: np.ndarray, units: int, grid_w: int,
          grid_h: int) -> np.ndarray:
    """Soft tokens [units * tokens, projection] of the patches [units * grid_h * grid_w, features]."""

    vc = config["vision_config"]
    h = vc["hidden_size"]
    heads, d = vc["num_attention_heads"], vc["head_dim"]
    eps = float(vc.get("rms_norm_eps", 1e-6))
    k = vc["pooling_kernel_size"]
    positions = vc["position_embedding_size"]
    theta = float((vc.get("rope_parameters") or {}).get("rope_theta", 100.0))
    read = lambda name: mmproj.read_direct(name).astype(np.float64)
    per_unit = grid_w * grid_h
    total = units * per_unit
    if patches.shape[0] != total:
        raise ValueError("patch count does not match the grid")

    w_patch = read("v.patch_embd.weight").reshape(h, -1)
    x = patches @ w_patch.T
    table = read("v.position_embd.weight").reshape(2, positions, h)
    local = np.arange(total) % per_unit
    columns, rows = local % grid_w, local // grid_w
    x = x + table[0][columns] + table[1][rows]

    # Axial RoPE angles: [total, D/4] per axis.
    freq = 1.0 / theta ** (np.arange(0, d // 2, 2, dtype=np.float64) / (d // 2))
    angle_x = columns[:, None] * freq[None, :]
    angle_y = rows[:, None] * freq[None, :]

    def rope(v: np.ndarray) -> np.ndarray:  # [total, heads, d]
        out = np.empty_like(v)
        for half, angle in ((0, angle_x), (1, angle_y)):
            part = v[:, :, half * d // 2:(half + 1) * d // 2]
            a, b = part[:, :, : d // 4], part[:, :, d // 4:]
            cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
            out[:, :, half * d // 2: half * d // 2 + d // 4] = a * cos - b * sin
            out[:, :, half * d // 2 + d // 4:(half + 1) * d // 2] = b * cos + a * sin
        return out

    for i in range(vc["num_hidden_layers"]):
        g = f"v.blk.{i}."
        normed = rms(x, eps, read(g + "ln1.weight"))
        q = (normed @ read(g + "attn_q.weight").T).reshape(total, heads, d)
        kk = (normed @ read(g + "attn_k.weight").T).reshape(total, heads, d)
        v = (normed @ read(g + "attn_v.weight").T).reshape(total, heads, d)
        q = rope(rms(q, eps, read(g + "attn_q_norm.weight")))
        kk = rope(rms(kk, eps, read(g + "attn_k_norm.weight")))
        v = rms(v, eps)
        attended = np.empty_like(q)
        for u in range(units):
            s = slice(u * per_unit, (u + 1) * per_unit)
            qh, kh, vh = (t[s].transpose(1, 0, 2) for t in (q, kk, v))  # [heads, patches, d]
            scores = qh @ kh.transpose(0, 2, 1)
            scores -= scores.max(axis=-1, keepdims=True)
            p = np.exp(scores)
            p /= p.sum(axis=-1, keepdims=True)
            attended[s] = (p @ vh).transpose(1, 0, 2)
        o = attended.reshape(total, h) @ read(g + "attn_out.weight").T
        x = x + rms(o, eps, read(g + "attn_post_norm.weight"))
        normed = rms(x, eps, read(g + "ln2.weight"))
        gate = normed @ read(g + "ffn_gate.weight").T
        up = normed @ read(g + "ffn_up.weight").T
        down = (gelu_tanh(gate) * up) @ read(g + "ffn_down.weight").T
        x = x + rms(down, eps, read(g + "ffn_post_norm.weight"))

    cols, rws = grid_w // k, grid_h // k
    pooled = x.reshape(units, rws, k, cols, k, h).mean(axis=(2, 4)).reshape(-1, h)
    pooled = pooled * np.sqrt(h)
    if vc.get("standardize", False):
        pooled = (pooled - read("v.std_bias")) * read("v.std_scale")
    return rms(pooled, eps) @ read("mm.input_projection.weight").T


def transformers_tower(hf_dir: Path, projection: np.ndarray, patches: np.ndarray, units: int,
                       grid_w: int, grid_h: int) -> np.ndarray:
    """transformers' Gemma4VisionModel and embed_vision in FP32 with the release's tower weights and
    the given projection [text hidden, hidden]."""

    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import (Gemma4MultimodalEmbedder,
                                                            Gemma4VisionModel)

    config = AutoConfig.from_pretrained(hf_dir)
    vision = Gemma4VisionModel(config.vision_config).float().eval()
    embedder = Gemma4MultimodalEmbedder(config.vision_config, config.text_config).float().eval()
    # Every safetensors file of the directory: the release's shards, or a file holding only the
    # tower's tensors.
    state_v, state_e = {}, {}
    for shard in sorted(hf_dir.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            for name in f.keys():
                for prefix, state in (("model.vision_tower.", state_v),
                                      ("model.embed_vision.", state_e)):
                    if name.startswith(prefix):
                        state[name[len(prefix):]] = f.get_tensor(name).float()
    missing = vision.load_state_dict(state_v, strict=False)
    if missing.missing_keys:
        raise ValueError(f"vision weights missing: {missing.missing_keys}")
    embedder.load_state_dict(state_e, strict=True)
    release = embedder.embedding_projection.weight.double().numpy()
    print(f"projection: the release's differs from the GGUF's by at most "
          f"{np.abs(release - projection).max():.4g}")
    embedder.embedding_projection.weight.data = torch.from_numpy(projection).float()
    p = 16
    per_unit = grid_w * grid_h
    # The engine's (channel, y, x) patches to transformers' (y, x, channel), pixels as rescaled
    # [0, 1] values (the model maps them back to 2p - 1).
    values = patches.reshape(units * per_unit, 3, p, p).transpose(0, 2, 3, 1)
    values = (values.reshape(units, per_unit, -1) + 1.0) / 2.0
    local = np.arange(per_unit)
    position_ids = np.stack([local % grid_w, local // grid_w], axis=-1)
    position_ids = np.broadcast_to(position_ids, (units, per_unit, 2)).copy()
    with torch.no_grad():
        out = vision(pixel_values=torch.from_numpy(values).float(),
                     pixel_position_ids=torch.from_numpy(position_ids).long())
        soft = embedder(out.last_hidden_state.float())
    return soft.double().numpy().reshape(-1, soft.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mmproj", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="the release's config.json")
    parser.add_argument("--patches", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=int, nargs=2, metavar=("W", "H"))
    parser.add_argument("--units", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--transformers", type=Path, default=None, help="HF release directory")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.patches.suffix == ".bin":
        features = 3 * config["vision_config"]["patch_size"] ** 2
        words = np.fromfile(args.patches, dtype="<u2").reshape(-1, features)
    else:
        words = np.load(args.patches)
    patches = bf16_words(words)
    w, h = args.grid
    with GGUFFile(args.mmproj) as mmproj:
        soft = tower(mmproj, config, patches, args.units, w, h)
        projection = mmproj.read_direct("mm.input_projection.weight").astype(np.float64)
    print(f"soft tokens {soft.shape}, mean |v| {np.abs(soft).mean():.6g}")
    if args.out is not None:
        # A .bin file receives the raw little-endian FP64 values (the C++ test reads them).
        if args.out.suffix == ".bin":
            soft.astype("<f8").tofile(args.out)
        else:
            np.save(args.out, soft)
    if args.transformers is not None:
        hf = transformers_tower(args.transformers, projection.reshape(soft.shape[1], -1), patches,
                                args.units, w, h)
        diff = np.abs(hf - soft)
        scale = np.abs(soft).max()
        print(f"transformers FP32 vs FP64: max |diff| {diff.max():.4g} ({diff.max() / scale:.3g} "
              f"of max |v|), mean |diff| {diff.mean():.4g}, cosine "
              f"{(hf * soft).sum() / np.linalg.norm(hf) / np.linalg.norm(soft):.8f}")


if __name__ == "__main__":
    main()
