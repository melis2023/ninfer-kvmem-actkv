"""The draft GGUF has its own metadata and retains supported ggml blocks."""
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.convert.__main__ import _open_named_source
from tools.convert.dflash2_gguf import DraftSource, draft_blocks
from tools.convert.gguf_blocks import _companions
from tools.convert.model import Model
from tools.convert.qwen3_5 import _Builder, draft_config
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf import TYPE_BF16, write_gguf


def fixture(tmp_path: Path, block_projections=None, **overrides):
    kv = {
        'general.architecture':'dflash', 'dflash.attention.causal':False,
        'dflash.block_count':1, 'dflash.embedding_length':256,
        'dflash.feed_forward_length':512, 'dflash.context_length':8192,
        'dflash.attention.head_count':2, 'dflash.attention.head_count_kv':1,
        'dflash.attention.key_length':128, 'dflash.attention.value_length':128,
        'dflash.attention.layer_norm_rms_epsilon':1e-6, 'dflash.rope.freq_base':10000000.,
        'dflash.target_layers':[2,6], 'dflash.attention.sliding_window_pattern':[True],
        'dflash.attention.sliding_window':2048, 'dflash.conv_kernel_size':2,
        'dflash.conv_group_size':16, 'dflash.selector_rank':256, 'dflash.selector_top_k':16,
        'tokenizer.ggml.tokens':[str(i) for i in range(32)], 'tokenizer.ggml.mask_token_id':31,
    } | overrides
    shapes = {
        'fc.weight':(256,512), 'enc.output_norm.weight':(256,), 'output_norm.weight':(256,),
        'selector_hidden.weight':(256,256), 'selector_predecessor.weight':(32,256),
        'selector_successor.weight':(32,256), 'blk.0.attn_norm.weight':(256,),
        'blk.0.ffn_norm.weight':(256,), 'blk.0.attn_q.weight':(256,256),
        'blk.0.attn_k.weight':(128,256), 'blk.0.attn_v.weight':(128,256),
        'blk.0.attn_output.weight':(256,256), 'blk.0.attn_q_norm.weight':(128,),
        'blk.0.attn_k_norm.weight':(128,), 'blk.0.ffn_gate.weight':(512,256),
        'blk.0.ffn_up.weight':(512,256), 'blk.0.ffn_down.weight':(256,512),
        'blk.0.attn_conv_base':(2,2,256), 'blk.0.ffn_conv_base':(2,2,256),
        'blk.0.attn_conv_proj.weight':(64,256), 'blk.0.ffn_conv_proj.weight':(64,256),
    }
    tensors=[]; expected={}
    for index,(name,shape) in enumerate(shapes.items()):
        if name in ('fc.weight','selector_hidden.weight','blk.0.ffn_gate.weight','blk.0.ffn_up.weight') or name in (block_projections or {}):
            # Q4_K: exactly 144 bytes per block of 256 represented values.
            type_id=(block_projections or {}).get(name,12)
            block_bytes={11:110,12:144}[type_id]
            data=(np.arange(np.prod(shape)//256*block_bytes,dtype=np.uint8)+index).tobytes()
            tensors.append((name,shape,type_id,data));expected[name]=data
        else:
            values=(torch.arange(np.prod(shape),dtype=torch.float32).reshape(shape)/257).to(torch.bfloat16)
            tensors.append((name,shape,TYPE_BF16,values.view(torch.int16).numpy().tobytes()))
            expected[name]=values.float().reshape(-1)
    path=tmp_path/'draft.gguf';write_gguf(path,kv,tensors)
    return path,expected


def test_named_source_reads_metadata_and_partial_direct_values(tmp_path):
    path,expected=fixture(tmp_path)
    with _open_named_source('dflash2',path,frozenset({'gguf'})) as source:
        assert isinstance(source,DraftSource)
        target={'hidden_size':256,'vocab_size':32,'num_hidden_layers':8}
        config=draft_config(source.config,target,'dflash2')
        assert config['dflash_config']['target_layer_ids']==[1,5]
        assert config['layer_types']==['sliding_attention'] and config['sliding_window']==2048
        for name in ('selector_predecessor.weight','blk.0.attn_conv_base'):
            torch.testing.assert_close(source.read_flat(name,7,29),expected[name][7:29],rtol=0,atol=0)


def test_recipe_preserves_q4_blocks_and_runtime_fallbacks(tmp_path):
    path,expected=fixture(tmp_path)
    with DraftSource(path) as source:
        target={'hidden_size':256,'vocab_size':32,'num_hidden_layers':8}
        config=draft_config(source.config,target,'dflash2')
        model=Model({'text':{'config':target},'dflash2':{'config':config,'target':'text'}})
        _Builder(model).draft(source,config,target,'dflash2')
        blocks=draft_blocks(model,source)
        names={'dflash2/feature_projection':'fc.weight',
               'dflash2/candidate_selector/hidden_projection':'selector_hidden.weight',
               'dflash2/layers/0/mlp/gate':'blk.0.ffn_gate.weight',
               'dflash2/layers/0/mlp/up':'blk.0.ffn_up.weight'}
        for logical,tensor in names.items():
            encoded=blocks[logical].read_encoded(0,blocks[logical].shape[0])
            assert encoded.format=='gguf_q4_k'
            assert encoded.codes.numpy().tobytes()==expected[tensor]
        recipe=Recipe(model);_companions(model,recipe,{'dflash2':source})
        prepared=recipe.prepare(device='cpu',rows_per_chunk=32)
        formats={name:job.spec.format for job in prepared.weights for name in job.parameters}
        assert formats['dflash2/feature_projection']=='gguf_q4_k'
        assert formats['dflash2/layers/0/mlp/gate']=='gguf_q4_k'
        assert formats['dflash2/layers/0/attention/query']=='q8_g32_fp16'
        assert formats['dflash2/layers/0/mlp/down']=='q8_g32_fp16'
        assert formats['dflash2/layers/0/attention_conv/kernel_projection']=='bf16'


@pytest.mark.parametrize('overrides',[
    {'general.architecture':'qwen35'}, {'dflash.attention.causal':True},
    {'dflash.target_layers':[0,2]}, {'dflash.attention.sliding_window_pattern':[True,False]},
])
def test_refuses_incompatible_draft_metadata(tmp_path,overrides):
    path,_=fixture(tmp_path,**overrides)
    with pytest.raises(ValueError):DraftSource(path)


@pytest.mark.parametrize('type_id,format', [(11,'gguf_q3_k'),(12,'gguf_q4_k')])
def test_output_and_down_override_preserves_source_blocks(tmp_path,type_id,format):
    from tools.convert.recipes.dflash2_gguf_blocks import configure
    from tools.convert.pipeline import convert
    from tools.artifact.reader import Artifact
    tensor_names={'dflash2/layers/0/attention/output':'blk.0.attn_output.weight',
                  'dflash2/layers/0/mlp/down':'blk.0.ffn_down.weight'}
    path,expected=fixture(tmp_path,block_projections={name:type_id for name in tensor_names.values()})
    with DraftSource(path) as source:
        target={'hidden_size':256,'vocab_size':32,'num_hidden_layers':8}
        config=draft_config(source.config,target,'dflash2')
        model=Model({'text':{'config':target},'dflash2':{'config':config,'target':'text'}})
        _Builder(model).draft(source,config,target,'dflash2')
        recipe=Recipe(model);_companions(model,recipe,{'dflash2':source})
        configure(model,recipe,{'dflash2':source})
        out=tmp_path/'draft.ninfer'
        convert(model,recipe,out,device='cpu',rows_per_chunk=128)
        with Artifact(out) as artifact:
            for logical,tensor in tensor_names.items():
                binding=artifact.directory.bindings[logical]
                obj=artifact.object(binding['object'])
                assert obj.format==format
                assert artifact.read_object(obj.id)==expected[tensor]
