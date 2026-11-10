"""Keep the DSpine draft body at BF16 for the accuracy ceiling test.

The loader pins the fused attention input projection to Q8, so only the rest of
the body can go BF16. This is a reference configuration, not deployable: the
body grows from ~0.8 GiB (Q4) to ~2.6 GiB.
"""
from tools.convert.methods import cast_direct, grouped_search
from tools.convert.official_recipes import Q8


def configure(model, recipe, sources):
    if "dflash" not in model.components:
        raise ValueError("DSpine BF16 recipe requires the dflash component")
    for name, parameter in model.parameters.items():
        if not name.startswith("dflash/"):
            continue
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
        recipe.assign(name, format="bf16", method=cast_direct)
