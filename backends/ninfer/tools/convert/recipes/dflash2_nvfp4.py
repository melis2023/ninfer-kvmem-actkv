"""Shipped fork DFlash2 weight-only NVFP4 override for full/QAT recipes.

Pass --components text,dflash2 --source dflash2=CHECKPOINT and
--override tools/convert/recipes/dflash2_nvfp4.py to tools.convert.
The 31 parents are five layers' QKV/output/gate-up/down and two convolution
control projections, plus the selector hidden projection. Feature projection
remains W8; codebooks, base kernels and norms remain BF16. No A4 divisor is used.
"""
from tools.convert.quantization.nvfp4 import nvfp4_maxabs

# The dynamic grouped conv reads the kernel projection as plain BF16; quantising it makes
# startup fail with "dynamic grouped conv prepare: invalid kernel_projection_weight".
_KEPT_BF16 = ('/attention_conv/kernel_projection', '/mlp_conv/kernel_projection')


def configure(model, recipe, sources):
    if 'dflash2' not in model.components:
        raise ValueError('NVFP4 module override requires the DFlash2 component')
    for name, parameter in model.parameters.items():
        if not name.startswith('dflash2/') or not parameter.projection:
            continue
        if name == 'dflash2/feature_projection' or '/attention/context_' in name:
            continue
        if any(name.endswith(suffix) for suffix in _KEPT_BF16):
            continue
        recipe.assign(name, format='nvfp4', method=nvfp4_maxabs, activation_policy='A16Only')
    for layer in range(model.components['dflash2']['config']['num_hidden_layers']):
        prefix = f'dflash2/layers/{layer}/'
        recipe.group(tuple(prefix + 'attention/' + role for role in ('query', 'key', 'value')))
        recipe.group((prefix + 'mlp/gate', prefix + 'mlp/up'))
        for role in ('key', 'value'):
            recipe.share(prefix + 'attention/context_' + role, prefix + 'attention/' + role)
