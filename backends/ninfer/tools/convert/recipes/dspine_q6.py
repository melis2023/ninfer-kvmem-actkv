"""Quantize the DSpine draft body to Q6 and keep its bridge in high precision.

This is the Q6 sibling of ``dspine_q4.py``. The draft MLP dominates the
bridge's delta error (its residual is rescaled by the bridge write), so the
body's projections use ``q6_g64_fp16``; the fused attention projections and
the target-feature projection stay at Q8 because they feed every draft layer's
context, and the 248320x1024 codebook stays BF16 as a lookup table.
"""
from tools.convert.methods import cast_direct, grouped_search
from tools.convert.official_recipes import Q6, Q8


def configure(model, recipe, sources):
    if "dflash" not in model.components:
        raise ValueError("DSpine Q6 recipe requires the dflash component")
    for name, parameter in model.parameters.items():
        if not name.startswith("dflash/"):
            continue
        if "/selected_prefix_bridge/" in name:
            # The bridge codebook is a row lookup, not a GEMM weight; keep it BF16 so
            # the execution path can read it as a contiguous tensor.
            recipe.assign(name, format="bf16", method=cast_direct)
            continue
        # The fused attention input projection admits Q8, not Q6. DFlash2 keeps
        # QKV at Q8 for the same reason; DSpine reuses that route.
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/attention/value",
                "/attention/context_key",
                "/attention/context_value",
            )
        ):
            recipe.assign(name, format=Q8, method=grouped_search)
            continue
        # q4/q8 linear_add does not admit the 4096-column attention output.
        if name.endswith("/attention/output"):
            recipe.assign(name, format="bf16", method=cast_direct)
            continue
        # The target-feature projection feeds every draft layer's context K/V.
        # Q4 there rotates the context by ~34 degrees; keep it wide.
        if name.endswith("/feature_projection"):
            recipe.assign(name, format=Q8, method=grouped_search)
            continue
        if parameter.projection:
            recipe.assign(name, format=Q6, method=grouped_search)
        else:
            recipe.assign(name, format="bf16", method=cast_direct)
