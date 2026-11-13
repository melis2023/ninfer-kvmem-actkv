"""Keep the llama.cpp DSpine draft's Q4_K blocks and put the vision tower at Q4.

Pass ``--source dflash=DSpine-b16-Q4_K_M.gguf`` and
``--override tools/convert/recipes/dspine_gguf_q4.py``.

The draft's feature projection, attention output, merged gate/up parent and MLP
down stay in the source's own ggml block type (``gguf_q4_k``), so the artifact
carries the exact bytes the release shipped with and the draft's Linear,
linear_add and linear_swiglu calls materialise them through the GGUF route. The
fused attention input projection and the context projections keep the official
Q8 row-split encoding their kernels require, and the selected-prefix bridge stays
BF16 because the bridge gathers the codebook row by row.

Vision keeps the official Qwen3.8 tower layout but every projection is Q4; the
norms, biases and position embedding remain direct BF16 tensors.
"""
from tools.convert.methods import grouped_absmax
from tools.convert.official_recipes import Q4


def configure(model, recipe, sources):
    from tools.convert.gguf_blocks import _vision_source

    if "dflash" not in model.components:
        raise ValueError("the DSpine GGUF recipe requires the dflash component")
    vision = _vision_source(model, sources) if "vision" in model.components else None
    for name, parameter in model.parameters.items():
        if name.startswith("vision/"):
            if not parameter.projection:
                continue
            recipe.assign(
                name,
                format=Q4,
                method=grouped_absmax,
                source=model.source(name, vision),
            )
