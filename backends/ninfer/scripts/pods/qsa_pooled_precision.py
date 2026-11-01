"""Bounded FP64 selection experiment for the proposed BF16 pooled-key store.

Synthetic represented BF16 projections use the model's 4x128 query geometry,
four-token pooling, zero-centred RMSNorm, and partial RoPE. This is an Op-domain
precision decision, not a model-quality estimate.
"""
from __future__ import annotations

import json

import numpy as np


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def norm_rope(values, weight, positions):
    value = np.asarray(values, dtype=np.float64)
    value = value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + 1e-6)
    value *= 1.0 + weight
    frequency = np.exp(-(2.0 * np.arange(32) / 64.0) * np.log(1e7)).astype(np.float32)
    angle = (np.asarray(positions, dtype=np.float32)[..., None] * frequency).astype(np.float64)
    cosine, sine = np.cos(angle), np.sin(angle)
    x, y = value[..., :32].copy(), value[..., 32:64].copy()
    value[..., :32] = x * cosine - y * sine
    value[..., 32:64] = y * cosine + x * sine
    return value


def run(blocks, seed):
    rng = np.random.default_rng(seed)
    key_norm = bf16(rng.uniform(-0.5, 0.5, 128))
    query_norm = bf16(rng.uniform(-0.5, 0.5, 128))
    raw_keys = bf16(rng.uniform(-2, 2, (blocks, 4, 128)))
    pooled = norm_rope(raw_keys.astype(np.float64).mean(axis=1), key_norm,
                       np.arange(blocks) * 4).astype(np.float32)
    reduced = bf16(pooled)
    positions = np.arange(blocks * 4 - 32, blocks * 4)
    queries = norm_rope(bf16(rng.uniform(-2, 2, (32, 4, 128))), query_norm,
                        positions[:, None])
    original = np.maximum(queries @ pooled.astype(np.float64).T, 0).sum(axis=1)
    rounded = np.maximum(queries @ reduced.astype(np.float64).T, 0).sum(axis=1)
    changed = outside = swaps = 0
    worst = 0.0
    for i, position in enumerate(positions):
        visible = (int(position) + 1) // 4
        scores = original[i, :visible]
        expected = np.argsort(-scores, kind="stable")[:512]
        actual = np.argsort(-rounded[i, :visible], kind="stable")[:512]
        difference = np.setxor1d(expected, actual)
        changed += bool(len(difference))
        swaps += len(difference) // 2
        if len(difference):
            edge = scores[expected[-1]]
            distance = float(np.max(np.abs(scores[difference] - edge)) / abs(edge))
            outside += distance > 1e-5
            worst = max(worst, distance)
    return {"context": blocks * 4, "seed": seed, "queries": len(positions),
            "queries_with_changed_selection": changed, "replaced_blocks": swaps,
            "queries_outside_existing_oracle_tolerance": outside,
            "worst_changed_score_distance_relative_to_edge": worst}


if __name__ == "__main__":
    for blocks in (1024, 8192, 32768):
        for seed in (7101, 7102, 7103):
            print(json.dumps(run(blocks, seed)), flush=True)
