import numpy as np
import pytest
import torch

from tools.artifact.codecs.t5 import pack, unpack
from tools.artifact.codecs.row_split import encode_row_split, decode_row_split_codes
from tools.artifact.layouts import row_split_geometry
from tools.convert.sources.gguf import encode_ptq1_0, decode_ptq1_0


def reference_pack(values):
    out = []
    for unit in values.reshape(-1,64):
        for byte in range(13):
            v=0
            for stage in range(5):
                column=20*(byte//4)+4*stage+byte%4 if byte<12 else 60+stage
                digit=int(unit[column])+1 if column<64 else 0
                v=3*v+digit
            out.append((256*v+242)//243)
    return bytes(out)


def test_all_codewords_and_exact_ptq1_repack():
    rng=np.random.default_rng(7501)
    values=rng.integers(-1,2,(243,2,128),dtype=np.int8)
    for v in range(243):
        for stage in range(5):
            values[v,0,4*stage]=v//3**(4-stage)%3-1
    scales=rng.standard_normal((243,2)).astype(np.float16)
    scales[0]=[np.float16(-0.0),np.nextafter(np.float16(0),np.float16(1))]
    gguf=encode_ptq1_0(values,scales)
    decoded=decode_ptq1_0(gguf)
    packed=pack(torch.from_numpy(decoded.values.reshape(-1,128)))
    assert packed.numpy().tobytes()==reference_pack(values)
    assert torch.equal(unpack(packed),torch.from_numpy(values.reshape(-1,128)))
    payload=encode_row_split(torch.from_numpy(values),torch.from_numpy(scales),'t5_g128_fp16',(243,256))
    actual_scales,actual_codes=decode_row_split_codes(payload,'t5_g128_fp16',(243,256))
    assert torch.equal(actual_codes,torch.from_numpy(values))
    assert torch.equal(actual_scales.view(torch.int16),torch.from_numpy(scales).view(torch.int16))
    g=row_split_geometry('t5_g128_fp16',(243,256))
    assert g.base_bytes_per_group==26 and g.high_bytes_per_group==0


def test_bad_codes_geometry_and_padding():
    with pytest.raises(ValueError):
        pack(torch.full((1,128),2,dtype=torch.int8))
    with pytest.raises(ValueError):
        row_split_geometry('t5_g128_fp16',(3,129))
    valid=pack(torch.zeros((1,128),dtype=torch.int8))
    for byte in (255,1):
        bad=valid.clone(); bad[0,12]=byte
        with pytest.raises(ValueError): unpack(bad)


def test_repack_preserves_bindings_auxiliaries_and_companion(tmp_path):
    from tools.artifact.reader import Artifact
    from tools.artifact.writer import ArtifactWriter
    from tools.artifact.schema import TensorSpec, ResourceSpec
    from tools.convert.repack_t5 import repack

    source, dest = tmp_path/'source.ninfer', tmp_path/'t5.ninfer'
    codes = torch.arange(4*1024, dtype=torch.int32).remainder(3).sub(1).to(torch.int8).reshape(4,8,128)
    scales = torch.arange(32, dtype=torch.float16).reshape(4,8)/128
    payload = encode_row_split(codes, scales, 'ptq1_g128_fp16', (4,1024))
    specs = [TensorSpec('w',(4,1024),'ptq1_g128_fp16','row_split_k128_v1'),
             TensorSpec('mtp',(2,2),'bf16','contiguous_le_v1'), ResourceSpec('template',5)]
    bindings = {'text/weight':{'object':'w'},
                'text/parts':{'parts':[{'object':'w','range':[2048,4096]},
                                      {'object':'w','range':[0,2048]}]},
                'mtp/weight':{'object':'mtp'}}
    uses = [{'parameter':p,'input':p+'/input','activation_policy':'A16Only'} for p in bindings]
    components = {'text':{'config':{},'resources':{'chat_template.jinja':'template'}},'mtp':{'config':{}}}
    with ArtifactWriter(source,specs,components=components,bindings=bindings,uses=uses) as writer:
        writer.write_object('w',payload)
        writer.write_object('mtp',bytes(range(8)))
        writer.write_object('template',b'hello')
    before=source.read_bytes()
    report=repack(source,dest)
    assert report['verified'] and source.read_bytes()==before
    with Artifact(dest) as converted:
        assert converted.directory.bindings==bindings
        assert converted.directory.components==components
        assert [u['activation_policy'] for u in converted.directory.uses]==['AllowA8','AllowA8','A16Only']
        assert converted.read_object('mtp')==bytes(range(8))
        assert converted.read_object('template')==b'hello'
        actual_scales,actual_codes=decode_row_split_codes(converted.read_object('w'),'t5_g128_fp16',(4,1024))
        assert torch.equal(actual_codes,codes)
        assert torch.equal(actual_scales.view(torch.int16),scales.view(torch.int16))
