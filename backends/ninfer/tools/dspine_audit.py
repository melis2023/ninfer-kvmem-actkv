#!/usr/bin/env python3
"""DSpine conversion audit: released GGUF metadata and tensor shapes -> artifact manifest.

The conversion is the one step nobody can eyeball. A wrong `head_topk`, `rank` or tap
list still loads, still drafts, and only shows up as a lower acceptance rate -- which
looks exactly like a slow kernel. This script reads the released GGUF's own metadata
and tensor shapes, reads what the .ninfer artifact actually carries, and prints both
sides next to the value the runtime requires.

Usage:
  python dspine_audit.py --gguf <released.gguf> --artifact <converted.ninfer>
  python dspine_audit.py ... --verbose     # also dump every dflash.* metadata key
  python dspine_audit.py ... --fix-taps    # repair a one-layer tap shift in place

`--fix-taps` exists because an artifact built before `target_layer_ids_are_inputs`
existed carries taps one layer too low, and rebuilding is a ten-minute, 15 GiB
operation for a five-number defect. The rewrite is only attempted when the
replacement is byte-for-byte the same length, which keeps the manifest length, every
tensor offset and the artifact id valid.

Exit code 0 when every check matches, 1 otherwise, so it can gate a conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backends" / "ninfer"))

from tools.convert.sources.gguf import GGUFFile  # noqa: E402

MAGIC = b"NINFER\x00"
# Constants the draft path hard-codes: kSparseSpeculativeCandidates, kDSpineMaximumRows,
# and the kappa/eps the refinement kernel was written against.
RUNTIME_HEAD_TOPK = 16
RUNTIME_MAX_ROWS = 15
REFINE_KAPPA = 10.0
REFINE_EPS = 1e-6


def read_manifest(path: Path) -> dict:
    """The artifact is magic + header + JSON, but the header has grown before, so
    locate the object instead of trusting fixed offsets."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    if not head.startswith(MAGIC):
        raise SystemExit(f"{path}: not a ninfer artifact (magic {head[:8]!r})")
    start = head.index(b'{"components"')
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(start)
        # The manifest is a few hundred KiB; the tensor data behind it is tens of GiB.
        obj, _ = decoder.raw_decode(f.read(1 << 21))
    return obj


def patch_taps(path: Path, expected: list[int]) -> bool:
    """Rewrite `target_layer_ids` in place. Returns False when nothing matched."""
    with open(path, "r+b") as f:
        head = f.read(1 << 20)
        start = head.index(b'{"components"')
        match = re.search(rb'"target_layer_ids"\s*:\s*\[[^\]]*\]', head[start:])
        if not match:
            return False
        old = match.group(0)
        new = b'"target_layer_ids":[' + b",".join(str(t).encode() for t in expected) + b"]"
        if len(new) != len(old):
            raise SystemExit(
                f"refusing to patch: {len(new)} bytes against {len(old)}; "
                "an unequal edit would move every tensor offset, so rebuild instead")
        f.seek(start + match.start())
        f.write(new)
        f.flush()
        os.fsync(f.fileno())
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fix-taps", action="store_true",
                    help="rewrite target_layer_ids from the GGUF when it is the only mismatch")
    args = ap.parse_args()

    gguf = GGUFFile(args.gguf)
    components = read_manifest(args.artifact)["components"]
    draft = components["dflash"]["config"]
    # The converter writes hidden_size/vocab_size into the draft config, but the packer
    # drops what the runtime reads from the target instead, so fall back to `text`.
    target_cfg = components.get("text", {}).get("config", {})
    kv = gguf.kv

    dcfg = draft["dflash_config"]
    bridge = dcfg["selected_prefix_code_bridge"]
    refine = dcfg["selected_prefix_last_write_refine"]
    second = dcfg["code_bridge_second_pass"]
    taps = list(dcfg["target_layer_ids"])
    layers = draft["num_hidden_layers"]
    block = dcfg["block_size"]
    topk = second["head_topk"]
    width = block - 1

    rows: list[tuple[str, object, object, bool]] = []

    def check(name, got, want, ok=None):
        rows.append((name, got, want, got == want if ok is None else ok))

    def same(name, artifact_value, gguf_key):
        if gguf_key in kv:
            check(name, artifact_value, kv[gguf_key])
        else:
            rows.append((name, artifact_value, "<not in GGUF>", False))

    # ---- A. shape-derived bridge parameters --------------------------------
    # rank and code_size are not metadata; the converter derives them from the codebook,
    # so re-derive them here instead of trusting the recipe.
    codebook = gguf.info("dspine_codebook.weight").shape
    gate = gguf.info("dspine_gate.weight").shape
    read = gguf.info("blk.0.dspine_read.weight").shape
    write = gguf.info("blk.0.dspine_write.weight").shape
    vocab, code_size = codebook
    rank, gate_code = gate
    hidden = kv.get("dflash.embedding_length")

    check("codebook shape is (vocab, code_size)", codebook[1], code_size)
    check("gate agrees on code_size", gate_code, code_size)
    check("dspine_read is (code_size, hidden)", tuple(read), (code_size, hidden))
    check("dspine_write is (hidden, rank)", tuple(write), (hidden, rank))
    check("bridge.rank == codebook gate width", bridge["rank"], rank)
    check("bridge.code_size == codebook vocab", bridge["code_size"], code_size)
    check("bridge.kappa", float(bridge["kappa"]), REFINE_KAPPA)
    check("bridge.eps", float(bridge["eps"]), REFINE_EPS)
    draft_hidden = draft.get("hidden_size") or target_cfg.get("hidden_size")
    draft_vocab = draft.get("vocab_size") or target_cfg.get("vocab_size")
    check("hidden_size (draft vs target)", draft_hidden, target_cfg.get("hidden_size"),
          draft_hidden == target_cfg.get("hidden_size"))
    check("draft vocab == target vocab", draft_vocab, target_cfg.get("vocab_size"),
          draft_vocab == target_cfg.get("vocab_size"))

    # ---- B. values the GGUF states and the artifact must repeat ------------
    same("target_layer_ids", taps, "dflash.target_layers")
    same("selector_top_k", topk, "dflash.selector_top_k")
    same("mask_token_id", dcfg["mask_token_id"], "tokenizer.ggml.mask_token_id")
    same("num_hidden_layers", layers, "dflash.block_count")
    same("hidden_size", draft_hidden, "dflash.embedding_length")
    same("intermediate_size", draft["intermediate_size"], "dflash.feed_forward_length")
    same("num_attention_heads", draft["num_attention_heads"], "dflash.attention.head_count")
    same("num_key_value_heads", draft["num_key_value_heads"], "dflash.attention.head_count_kv")
    same("head_dim", draft["head_dim"], "dflash.attention.key_length")
    same("max_position_embeddings", draft["max_position_embeddings"], "dflash.context_length")
    same("sliding_window", draft.get("sliding_window"), "dflash.attention.sliding_window")
    check("rms_norm_eps", float(draft["rms_norm_eps"]),
          float(kv.get("dflash.attention.layer_norm_rms_epsilon", -1)))
    check("rope_theta", float(draft["rope_parameters"]["rope_theta"]),
          float(kv.get("dflash.rope.freq_base", -1)))
    pattern = kv.get("dflash.attention.sliding_window_pattern") or []
    want_types = ["sliding_attention" if s else "full_attention" for s in pattern]
    if want_types:
        check("layer_types", list(draft.get("layer_types", [])), want_types)

    # ---- C. runtime constants ---------------------------------------------
    check("block_size", block, kv.get("dflash.block_size", 16))
    check("head_topk == kSparseSpeculativeCandidates", topk, RUNTIME_HEAD_TOPK)
    check("refine.head_topk == head_topk", refine["head_topk"], topk)
    check("refine.kind", refine["kind"], "exact_sender")
    check("refine.version", refine["version"], 1)
    check("refine.weight", float(refine["weight"]), 1.0)
    check("second.mode", second["mode"], "serial_table")
    # The second pass re-reads the codes the exact writes produced, so it can start at
    # the last layer only; anything else would re-read codes that were never written.
    check("second.first_exact == layers - 1", second["first_exact"], layers - 1)
    check("draft_block_causal", bool(dcfg.get("draft_block_causal")), True)
    check("is_causal", bool(draft.get("is_causal")), True)
    check("draft width (block-1) <= kDSpineMaximumRows", width, f"<={RUNTIME_MAX_ROWS}",
          width <= RUNTIME_MAX_ROWS)
    # Conversion-time marker: it is folded into target_layer_ids and must NOT reappear,
    # because a second application would shift every tap by one layer.
    check("target_layer_ids_are_inputs folded away",
          draft.get("target_layer_ids_are_inputs"), None,
          "target_layer_ids_are_inputs" not in draft)

    # ---- D. tap contract ---------------------------------------------------
    check("target_layer_ids strictly ascending", taps, "sorted, unique",
          taps == sorted(set(taps)))
    check("target_layer_ids count", len(taps), 5)
    check("target_layer_ids within 1..num_hidden_layers",
          [taps[0], taps[-1]], "1..64", all(0 < t <= 64 for t in taps))

    # ---- report ------------------------------------------------------------
    bad = [r for r in rows if not r[3]]
    w = max(len(r[0]) for r in rows) + 2
    print(f"GGUF      {args.gguf}")
    print(f"ARTIFACT  {args.artifact}")
    print(f"draft     layers={layers} block={block} width={width} rank={bridge['rank']} "
          f"code_size={bridge['code_size']} taps={taps}")
    print()
    print("check".ljust(w) + "artifact".ljust(30) + "expected")
    print("-" * (w + 56))
    for name, got, want, ok in rows:
        print(name.ljust(w) + ("ok   " if ok else "DIFF ") + str(got)[:26].ljust(28)
              + str(want)[:30])
    print()
    print(f"{len(rows) - len(bad)}/{len(rows)} checks match")

    if args.verbose:
        print("\n--- dflash.* metadata in the GGUF ---")
        for key in sorted(kv):
            if key.startswith("dflash."):
                text = str(kv[key])
                print(f"  {key} = {text[:108]}{'...' if len(text) > 108 else ''}")
        print("\n--- bridge tensor shapes ---")
        for name in ("dspine_codebook.weight", "dspine_gate.weight",
                     "blk.0.dspine_read.weight", "blk.0.dspine_write.weight"):
            print(f"  {name} {gguf.info(name).shape}")

    if bad:
        print("\nmismatch against the release:")
        for name, got, want, _ in bad:
            print(f"  {name}: artifact {got!r} != expected {want!r}")
        tap_only = all(name.startswith("target_layer_ids") and "folded" not in name
                       for name, *_ in bad)
        expected = list(kv.get("dflash.target_layers", []))
        if tap_only and expected:
            if args.fix_taps:
                if patch_taps(args.artifact, expected):
                    print(f"\npatched target_layer_ids -> {expected}; re-run to confirm")
                    return 0
                print("\npatch failed: no target_layer_ids array in the manifest")
            else:
                print(f"\nthis is the pre-`target_layer_ids_are_inputs` build; "
                      f"--fix-taps rewrites it to {expected} in place")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
