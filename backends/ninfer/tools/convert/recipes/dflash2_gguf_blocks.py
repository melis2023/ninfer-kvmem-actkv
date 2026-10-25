"""Keep the DFlash2 drafter's GGUF block projections instead of re-quantising them.

Pass ``--source dflash2=<draft.gguf>`` and
``--override tools/convert/recipes/dflash2_gguf_blocks.py``.

``attention/output`` and ``mlp/down`` stay in the source's own ggml block type
(``gguf_q3_k`` / ``gguf_q4_k`` / ...), so the artifact carries the exact bytes the
drafter shipped with and the dynamic grouped conv materialises them through the
GGUF linear route. QKV, the convolution kernels, the feature projection and the
selector hidden projection keep the base recipe's formats.
"""
from tools.convert.methods import import_encoded

_TARGETS = ('/attention/output', '/mlp/down')


def configure(model, recipe, sources):
    if 'dflash2' not in model.components:
        raise ValueError('DFlash2 GGUF-block import requires the DFlash2 component')
    draft = sources['dflash2'] if 'dflash2' in sources else None
    for name, parameter in model.parameters.items():
        if not name.startswith('dflash2/') or not parameter.projection:
            continue
        if not any(name.endswith(suffix) for suffix in _TARGETS):
            continue
        item = draft.importable(name) if draft is not None and draft.has(name) else None
        if item is None:
            continue
        encoded = item.read_encoded(0, 1)
        recipe.assign(name, format=encoded.format, method=import_encoded, source=item)
