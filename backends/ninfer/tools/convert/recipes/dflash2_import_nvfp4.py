"""Import the DFlash2 module's NVFP4 weights verbatim from a ModelOpt source.

Unlike ``dflash2_nvfp4.py`` (which re-quantises a BF16/FP8 companion), this override
keeps the source's NVFP4 codes, block scales and global divisor. Use it when the
companion checkpoint is already ModelOpt NVFP4, e.g.
``maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal``.

The whole drafter is NVFP4: ``attention/output`` and ``mlp/down`` import the source's
encoded words verbatim, while ``feature_projection`` and the selector hidden
projection (BF16 in the checkpoint) are quantised to NVFP4. QKV keeps Q8 because the
fused attention input projection only admits that format, and the convolution kernel
projections stay BF16 because their prepare path has no NVFP4 kernel.
"""
from tools.convert.methods import import_encoded
from tools.convert.quantization.nvfp4 import nvfp4_maxabs

# The engine's native QKV input projection requires Q8_G32_FP16.
_KEPT_Q8 = ('/attention/query', '/attention/key', '/attention/value')

# BF16 in the ModelOpt checkpoint; quantise them so the drafter is one format.
_REQUANTISE = (
    'dflash2/feature_projection',
    'dflash2/candidate_selector/hidden_projection',
)


def configure(model, recipe, sources):
    if 'dflash2' not in model.components:
        raise ValueError('DFlash2 NVFP4 import requires the DFlash2 component')
    for name, parameter in model.parameters.items():
        if not name.startswith('dflash2/') or not parameter.projection:
            continue
        if 'conv' in name or '/attention/context_' in name:
            continue
        if any(name.endswith(suffix) for suffix in _KEPT_Q8):
            continue
        if name in _REQUANTISE:
            recipe.assign(name, format='nvfp4', method=nvfp4_maxabs,
                          activation_policy='A16Only')
            continue
        recipe.assign(name, format='nvfp4', method=import_encoded)
