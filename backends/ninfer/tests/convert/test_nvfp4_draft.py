"""Independent E2M1 boundary and bounded-draft conversion checks."""
import struct
import numpy as np
import pytest
import torch
from tools.convert.nvfp4 import quantize_rows, nvfp4_maxabs
from tools.convert.model import Model, Parameter
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source
from tools.convert.pipeline import convert
from tools.artifact.reader import Artifact
from tools.artifact.codecs.nvfp4 import decode_nvfp4_words


def unpack(codes,scales,divisor):
    # Scalar format definition: sign + E2M1 magnitude and unsigned E4M3 group scale.
    levels=[0.,.5,1.,1.5,2.,3.,4.,6.]
    out=np.empty((codes.shape[0],codes.shape[1]*2),dtype=np.float64)
    for row in range(len(out)):
        for col in range(out.shape[1]):
            code=(int(codes[row,col//2]) >> (4*(col%2))) & 15
            word=int(scales[row,col//16]); exponent=(word>>3)&15; fraction=word&7
            scale=(fraction*2.**-9 if exponent==0 else (1+fraction/8)*2.**(exponent-7))
            out[row,col]=(-1 if code&8 else 1)*levels[code&7]*scale/divisor
    return out


def test_fp4_midpoints_round_to_even_and_keep_sign():
    midpoints=[.25,.75,1.25,1.75,2.5,3.5,5.]
    row=midpoints+[-x for x in midpoints]+[6.,-6.]
    values=torch.tensor([row*4]*128)
    codes,scales=quantize_rows(values,1.)
    expected=[0.,1.,1.,2.,2.,4.,4.]
    np.testing.assert_array_equal(unpack(codes.numpy(),scales.numpy(),1.)[0,:16],
                                 expected+[-x for x in expected]+[6.,-6.])


def test_weight_only_roundtrip_has_finite_independently_decoded_values(tmp_path):
    values=torch.linspace(-.071,.089,128*64).reshape(128,64)
    model=Model({'text':{'config':{}}})
    model.add(Parameter('dflash2/feature_projection',(128,64),array_source(values,'independent fixture'),inputs=('draft_input',)))
    recipe=Recipe(model)
    recipe.assign('dflash2/feature_projection',format='nvfp4',method=nvfp4_maxabs,activation_policy='A16Only')
    path=tmp_path/'nvfp4.ninfer';convert(model,recipe,path,device='cpu',rows_per_chunk=128)
    with Artifact(path) as artifact:
        obj=artifact.object(artifact.directory.bindings['dflash2/feature_projection']['object'])
        codes,scales,divisor=decode_nvfp4_words(artifact.read_object(obj.id),obj.shape)
        actual=unpack(codes.numpy(),scales.numpy(),float(divisor))
        assert np.isfinite(actual).all()
        assert np.max(np.abs(actual-values.numpy())) < .015
        assert all(u['activation_policy']=='A16Only' for u in artifact.directory.uses)


def test_quantization_rejects_nonfinite_input():
    with pytest.raises(ValueError,match='finite'):
        quantize_rows(torch.full((128,64),float('nan')),1.)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize('chunk', [128, 512])
def test_draft_requantization_keeps_gate_up_in_one_native_parent(tmp_path, dtype, chunk):
    from tools.convert.recipes.dflash2_nvfp4 import configure

    model = Model({'text': {'config': {}}, 'dflash2': {'config': {}}})
    expected = []
    for layer in range(2):
        names = tuple(f'dflash2/layers/{layer}/mlp/{role}' for role in ('gate', 'up'))
        rows = []
        for index, name in enumerate(names):
            values = (torch.linspace(-.07, .09, 256*64).reshape(256, 64)
                      * (1 + index + layer)).to(dtype)
            model.add(Parameter(name, (256, 64), array_source(values, name),
                                inputs=('mlp_input',)))
            rows.append(values.float().numpy())
        model.packing_groups.append(names)
        expected.append((names, np.concatenate(rows)))
    recipe = Recipe(model)
    configure(model, recipe, {})
    path = tmp_path / 'draft.ninfer'
    convert(model, recipe, path, device='cpu', rows_per_chunk=chunk)
    with Artifact(path) as artifact:
        for names, values in expected:
            gate, up = (artifact.directory.bindings[name]['parts'] for name in names)
            assert len(gate) == len(up) == 1
            assert gate[0]['object'] == up[0]['object']
            obj = artifact.object(gate[0]['object'])
            assert obj.shape == (512, 64) and obj.divisors == 1
            assert gate[0]['range'] == [0, 256*64]
            assert up[0]['range'] == [256*64, 512*64]
            codes, scales, divisor = decode_nvfp4_words(artifact.read_object(obj.id), obj.shape)
            actual = unpack(codes.numpy(), scales.numpy(), float(divisor))
            assert np.isfinite(actual).all()
            assert np.max(np.abs(actual-values)) < .04


def test_weight_only_requantization_rejects_activation_quantization():
    model = Model({'text': {'config': {}}})
    model.add(Parameter('draft', (128, 64), array_source(torch.ones(128, 64), 'fixture'),
                        inputs=('input',)))
    recipe = Recipe(model)
    recipe.assign('draft', format='nvfp4', method=nvfp4_maxabs, activation_policy='AllowA4')
    with pytest.raises(ValueError, match='weight-only'):
        recipe.prepare(device='cpu')
