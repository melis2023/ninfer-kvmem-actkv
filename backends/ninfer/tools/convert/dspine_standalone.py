"""Build a standalone DSpine drafter artifact from the release's GGUF.

Unlike ``dspine_gguf.py``, which embeds the drafter in a target model's ``dflash``
component for the local bridge path, this writes the drafter alone in a single
``dspine`` component — the shape ``models::dspine::read_config`` accepts and
``--draft-model`` loads. The target keeps its own embedding, head and KV; the
engine derives the head-dependent tables when it loads the two together.

The GGUF is llama.cpp's conversion of the release checkpoint, so its names are
``blk.N.*`` rather than a checkpoint's ``layers.N.*``. Two mappings are worth
stating because they are not one-to-one:

- The drafter's context projections are its own key/value projections: the
  upstream converter shares ``context_key``/``context_value`` with ``key``/
  ``value`` (``_share_context``). The GGUF stores those matrices once, so the
  same source feeds both parameters.
- ``bridge/projection`` is the three bridge row blocks stacked
  (``gate``, ``predecessor_gate``, ``prev``), which is ``{3 * rank, code}``.
  ``bridge/head_write`` is the last layer's write transposed, and
  ``bridge/sender_table`` is derived from the codebook and those two bridge
  blocks exactly as the reference's table builder does it.

Usage:
  python -m tools.convert.dspine_standalone \\
      --source Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf \\
      --target-layers 64 --out dspine.ninfer
"""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

from tools.convert.dspine_gguf import DraftSource, validate  # reused GGUF reader/validator
from tools.convert.model import Model, Parameter
from tools.convert.sources.logical import LogicalSource, transpose_source

COMPONENT = "dspine"


def _product(shape) -> int:
    total = 1
    for extent in shape:
        total *= int(extent)
    return total


def _flat(source: DraftSource, gguf_name: str, shape) -> LogicalSource:
    """The named GGUF tensor as a logical source of the given logical shape."""
    if source.describe(gguf_name) is None:
        raise ValueError(f"DSpine GGUF has no tensor {gguf_name!r}")
    return LogicalSource(
        tuple(int(x) for x in shape),
        f"{gguf_name}",
        lambda begin, end: source.read_flat(gguf_name, begin, end),
    )


def _concat_rows(parts: list[LogicalSource], label: str) -> LogicalSource:
    """Matrices of equal width stacked one below the other."""
    width = parts[0].shape[1]
    if any(len(p.shape) != 2 or p.shape[1] != width for p in parts):
        raise ValueError(f"{label}: the rows disagree on width")
    edges = [0]
    for part in parts:
        edges.append(edges[-1] + part.shape[0] * width)

    def read(begin: int, end: int):
        import torch

        chunks = []
        for part, low, high in zip(parts, edges, edges[1:]):
            a, b = max(begin, low), min(end, high)
            if a < b:
                chunks.append(part.values(a - low, b - low))
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    return LogicalSource((edges[-1] // width, width), label, read)


def _sender_table(source: DraftSource, vocab: int, code: int, rank: int,
                  gate_name: str, prev_name: str, codebook_name: str) -> LogicalSource:
    """[V, 2R]: each token's [predecessor gate; value] rows, as the reference builds them."""
    import torch

    codebook = source.read_flat(codebook_name, 0, _product((vocab, code))).reshape(vocab, code)
    sender = torch.cat([
        source.read_flat(gate_name, 0, rank * code).reshape(rank, code),
        source.read_flat(prev_name, 0, rank * code).reshape(rank, code),
    ]).to(torch.float32)
    scale = sqrt(code)
    width = 2 * rank

    def read(begin: int, end: int):
        first, last = begin // width, (end + width - 1) // width
        # The reference rounds the scaled codebook to BF16 before the product, then the result.
        codes = (scale * codebook[first:last].to(torch.float32)).to(torch.bfloat16)
        values = (codes.to(torch.float32) @ sender.T).to(torch.bfloat16).reshape(-1)
        return values[begin - first * width : end - first * width]

    return LogicalSource((vocab, width), f"{codebook_name}:sender table", read)


def component_config(raw: dict, vocab_size: int, num_target_layers: int) -> dict:
    """The `dspine` component's config from the drafter's own."""
    bridge = raw["dflash_config"]["selected_prefix_code_bridge"]
    refine = raw["dflash_config"]["selected_prefix_last_write_refine"]
    config = {
        "architectures": ["DSpineDraftModel"],
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "hidden_size": raw["hidden_size"],
        "intermediate_size": raw["intermediate_size"],
        "num_hidden_layers": raw["num_hidden_layers"],
        "num_attention_heads": raw["num_attention_heads"],
        "num_key_value_heads": raw["num_key_value_heads"],
        "head_dim": raw["head_dim"],
        "max_position_embeddings": raw["max_position_embeddings"],
        "vocab_size": vocab_size,
        "num_target_layers": num_target_layers,
        "block_size": raw["block_size"],
        "layer_types": list(raw["layer_types"]),
        "rms_norm_eps": raw["rms_norm_eps"],
        "rope_parameters": dict(raw.get("rope_parameters") or {}),
        "dspine_config": {
            "target_layer_ids": list(raw["dflash_config"]["target_layer_ids"]),
            "mask_token_id": raw["dflash_config"]["mask_token_id"],
            "code_size": bridge["code_size"],
            "rank": bridge["rank"],
            "bridge_eps": bridge["eps"],
            "head_top_k": refine["head_topk"],
        },
    }
    if "sliding_window" in raw:
        config["sliding_window"] = raw["sliding_window"]
    return config


def build_model(source: DraftSource, num_target_layers: int) -> Model:
    """The standalone artifact's logical parameters, named as models/dspine binds them."""
    raw, aliases = validate(source.gguf)
    vocab = len(source.gguf.kv["tokenizer.ggml.tokens"]) if hasattr(
        source.gguf, "kv") else len(source.gguf.tokens_meta)
    config = component_config(raw, vocab, num_target_layers)
    hidden = config["hidden_size"]
    inner = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    draft = config["dspine_config"]
    code, rank = draft["code_size"], draft["rank"]
    taps = len(draft["target_layer_ids"])
    head_dim = config["head_dim"]
    query_width = config["num_attention_heads"] * head_dim
    kv_width = config["num_key_value_heads"] * head_dim

    model = Model({COMPONENT: {"config": config}})

    def add(name: str, gguf_name: str, shape, inputs: tuple[str, ...] = ()) -> None:
        model.add(Parameter(name, tuple(int(x) for x in shape),
                            _flat(source, gguf_name, shape), None, inputs, "bf16", COMPONENT))

    add(f"{COMPONENT}/feature_projection", "fc.weight", (hidden, taps * hidden),
        (f"{COMPONENT}/target_features",))
    add(f"{COMPONENT}/context_norm", "enc.output_norm.weight", (hidden,))
    add(f"{COMPONENT}/final_norm", "output_norm.weight", (hidden,))

    for i in range(layers):
        p = f"{COMPONENT}/layers/{i}/"
        a = p + "attention/"
        b = f"{COMPONENT}/bridge/layers/{i}/"
        blk = f"blk.{i}."
        add(p + "input_norm", blk + "attn_norm.weight", (hidden,))
        add(p + "post_attention_norm", blk + "ffn_norm.weight", (hidden,))
        add(a + "query", blk + "attn_q.weight", (query_width, hidden), (p + "query_projection_input",))
        add(a + "key", blk + "attn_k.weight", (kv_width, hidden), (p + "query_projection_input",))
        add(a + "value", blk + "attn_v.weight", (kv_width, hidden), (p + "query_projection_input",))
        # The context projections are the same matrices (upstream _share_context).
        add(a + "context_key", blk + "attn_k.weight", (kv_width, hidden),
            (f"{COMPONENT}/context_input",))
        add(a + "context_value", blk + "attn_v.weight", (kv_width, hidden),
            (f"{COMPONENT}/context_input",))
        add(a + "query_norm", blk + "attn_q_norm.weight", (head_dim,))
        add(a + "key_norm", blk + "attn_k_norm.weight", (head_dim,))
        add(a + "output", blk + "attn_output.weight", (hidden, query_width),
            (p + "attention_output",))
        add(p + "mlp/gate", blk + "ffn_gate.weight", (inner, hidden), (p + "mlp_input",))
        add(p + "mlp/up", blk + "ffn_up.weight", (inner, hidden), (p + "mlp_input",))
        add(p + "mlp/down", blk + "ffn_down.weight", (hidden, inner), (p + "mlp_product",))
        add(b + "read", blk + "dspine_read.weight", (code, hidden), (b + "read_input",))
        add(b + "write", blk + "dspine_write.weight", (hidden, rank), (b + "message",))

    bridge = f"{COMPONENT}/bridge/"
    projection = _concat_rows(
        [
            _flat(source, name, (rank, code))
            for name in ("dspine_gate.weight", "dspine_pred_gate.weight", "dspine_prev.weight")
        ],
        "DSpine bridge projection",
    )
    model.add(Parameter(bridge + "projection", tuple(projection.shape), projection, None,
                        (bridge + "code_input",), "bf16", COMPONENT))
    add(bridge + "gate_bias", "dspine_gate.bias", (rank,))

    last = _flat(source, f"blk.{layers - 1}.dspine_write.weight", (hidden, rank))
    model.add(Parameter(bridge + "head_write", (rank, hidden),
                        transpose_source(last, (1, 0), (rank, hidden)), None, (), "bf16", COMPONENT))
    model.add(Parameter(bridge + "sender_table", (vocab, 2 * rank),
                        _sender_table(source, vocab, code, rank, "dspine_pred_gate.weight",
                                      "dspine_prev.weight", "dspine_codebook.weight"),
                        None, (), "bf16", COMPONENT))
    return model


def recipe_bf16(model, recipe, sources) -> None:
    """Every drafter weight in BF16: the bridge's roundings are what the refinement depends on."""
    _share_context(model, recipe)


def recipe_q8(model, recipe, sources) -> None:
    """The layer and feature projections in Q8, the bridge and its tables in BF16.

    The bridge writes into the residual and its tables are what the serial refinement chains, so
    they keep BF16; everything the drafter multiplies by an activation goes to Q8.
    """
    from tools.convert.methods import grouped_absmax

    for name, parameter in model.parameters.items():
        if parameter.projection and not name.startswith(COMPONENT + "/bridge/"):
            recipe.assign(name, format="q8_g32_fp16", method=grouped_absmax)
    _share_context(model, recipe)


def _share_context(model, recipe) -> None:
    # The context projections are the key/value matrices; share them rather than store twice.
    for layer in range(model.components[COMPONENT]["config"]["num_hidden_layers"]):
        prefix = f"{COMPONENT}/layers/{layer}/attention/"
        for role in ("key", "value"):
            recipe.share(prefix + "context_" + role, prefix + role)


RECIPES = {"bf16": recipe_bf16, "q8": recipe_q8}


def main() -> int:
    parser = argparse.ArgumentParser(description="DSpine drafter to a standalone .ninfer")
    parser.add_argument("--source", required=True, help="the DSpine drafter GGUF")
    parser.add_argument("--target-layers", required=True, type=int,
                        help="the target model's layer count (the drafter is trained per target)")
    parser.add_argument("--out", required=True, help="the artifact to write")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument("--recipe", default="q8", choices=sorted(RECIPES),
                        help="q8 (default; projections quantized, bridge BF16) or bf16")
    args = parser.parse_args()

    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe

    source = DraftSource(args.source)
    model = build_model(source, args.target_layers)
    print(f"{len(model.parameters)} parameters, "
          f"{len(model.components[COMPONENT]['config'])} config fields", flush=True)

    recipe = Recipe(model)
    RECIPES[args.recipe](model, recipe, {"dspine": source})
    report = convert(model, recipe, args.out, name="dspine-drafter",
                     provenance={"source": str(args.source)}, device=args.device)
    print(f"wrote {args.out}: {report['payload_bytes'] / 1024 ** 2:.1f} MiB, "
          f"artifact {report['artifact_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
