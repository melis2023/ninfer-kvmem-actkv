"""Quantize supported DFlash2 projections to weight-only NVFP4.

QKV keeps its native Q8 fused route; convolution preparation remains BF16.
"""
from tools.convert.nvfp4 import nvfp4_maxabs

def configure(model, recipe, sources):
    if 'dflash2' not in model.components:
        raise ValueError('DFlash2 NVFP4 override requires the DFlash2 component')
    for name, parameter in model.parameters.items():
        if not name.startswith('dflash2/') or not parameter.projection:
            continue
        if 'conv' in name or '/attention/context_' in name:
            continue
        if name.endswith(('/attention/query', '/attention/key', '/attention/value')):
            continue
        recipe.assign(name, format='nvfp4', method=nvfp4_maxabs, activation_policy='A16Only')
    # Custom methods do not participate in Recipe's automatic packing. SwiGLU needs
    # gate/up in one complete NVFP4 parent, quantized with one shared divisor.
    for name in model.parameters:
        if name.startswith('dflash2/') and name.endswith('/mlp/gate'):
            recipe.group((name, name.removesuffix('gate') + 'up'))
