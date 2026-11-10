"""Q4 draft body, but the target-facing projections stay BF16.

The draft MLP is not the acceptance bottleneck (Q6 matched Q4), so it stays Q4.
The feature projection and the context K/V are the two places where target
hidden enters the draft; both move from Q8 to BF16 to test whether their
quantization is the remaining loss.
"""
from tools.convert.methods import cast_direct, grouped_search
from tools.convert.official_recipes import Q4, Q8


def configure(model, recipe, sources):
    if "dflash" not in model.components:
        raise ValueError("DSpine recipe requires the dflash component")
    for name, parameter in model.parameters.items():
        if not name.startswith("dflash/"):
            continue
        if "/selected_prefix_bridge/" in name:
            recipe.assign(name, format="bf16", method=cast_direct)
            continue
        # The loader pins the fused attention input projection to Q8.
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/attention/value",
            )
        ):
            recipe.assign(name, format=Q8, method=grouped_search)
            continue
        # Target hidden enters here; keep the projection exact. The context K/V
        # is a fused linear_pair, which only admits Q8.
        if name.endswith("/feature_projection"):
            recipe.assign(name, format="bf16", method=cast_direct)
            continue
        if name.endswith(("/attention/context_key", "/attention/context_value")):
            recipe.assign(name, format=Q8, method=grouped_search)
            continue
        if name.endswith("/attention/output"):
            recipe.assign(name, format="bf16", method=cast_direct)
            continue
        if parameter.projection:
            recipe.assign(name, format=Q4, method=grouped_search)
        else:
            recipe.assign(name, format="bf16", method=cast_direct)
