"""Quantize the DSpine draft body to Q4 and keep its bridge in high precision.

DSpine is a DFlashDraftModel with a block-16 causal proposal and a
selected-prefix code bridge. The body (fc, QKV, MLP) uses the native
``q4_g64_fp16`` format; the 248320x1024 codebook uses Q8 because it is a lookup
table rather than a GEMM weight; the remaining bridge tensors stay BF16.
"""
from tools.convert.methods import cast_direct, grouped_search
from tools.convert.official_recipes import Q4, Q8


def configure(model, recipe, sources):
    if "dflash" not in model.components:
        raise ValueError("DSpine Q4 recipe requires the dflash component")
    for name, parameter in model.parameters.items():
        if not name.startswith("dflash/"):
            continue
        if "/selected_prefix_bridge/" in name:
            # The bridge codebook is a row lookup, not a GEMM weight; keep it BF16 so
            # the execution path can read it as a contiguous tensor.
            recipe.assign(name, format="bf16", method=cast_direct)
            continue
        # The fused attention input projection admits Q8, not Q4. DFlash2 keeps
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
            recipe.assign(name, format=Q4, method=grouped_search)
            continue
        # q4/q8 linear_add does not admit the 4096-column attention output.
        if name.endswith("/attention/output"):
            recipe.assign(name, format="bf16", method=cast_direct)
            continue
        if parameter.projection:
            recipe.assign(name, format=Q4, method=grouped_search)
        else:
            recipe.assign(name, format="bf16", method=cast_direct)


