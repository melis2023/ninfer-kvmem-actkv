from __future__ import annotations

import json

import numpy as np
import pytest

from tools.convert import flint
from tools.convert.sources.gguf import TYPE_BF16, TYPE_F32, GGUFFile, write_gguf

TYPE_Q8_0 = 8
TYPE_Q5_K = 13

HIDDEN, VOCAB, HEADS, KV_HEADS, HEAD_DIM = 256, 512, 4, 2, 32
WIDTH, EXPERTS, EXPERT_WIDTH, SHARED = 512, 4, 256, 256
LAYER_TYPES = ["full_attention", "sliding_attention", "full_attention"]


def _hf_config(**overrides):
    config = dict(
        architectures=["FlintForCausalLM"], model_type="flint", hidden_act="silu",
        attention_bias=False, hidden_size=HIDDEN, vocab_size=VOCAB, num_hidden_layers=3,
        max_position_embeddings=4096, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM, sliding_window=65, intermediate_size=WIDTH, num_experts=EXPERTS,
        num_experts_per_tok=2, moe_intermediate_size=EXPERT_WIDTH,
        shared_expert_intermediate_size=SHARED, layer_types=LAYER_TYPES,
        no_rope_layer_indices=[0, 2], mlp_only_layers=[0], router_score_function="sigmoid",
        norm_topk_prob=True, softmax_type="learnable", decoder_sparse_step=1, rope_theta=10000.0,
        rms_norm_eps=1e-6, tie_word_embeddings=False, bos_token_id=VOCAB - 2,
        eos_token_id=[VOCAB - 1])
    config.update(overrides)
    return config


def _q8_0(generator, rows, columns):
    blocks = generator.integers(0, 256, size=(rows, columns // 32, 34), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 32)).astype(np.float16)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(rows, columns // 32, 2)
    return blocks.reshape(rows, -1)


def _q5_k(generator, rows, columns):
    blocks = generator.integers(0, 256, size=(rows, columns // 256, 176), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 256, 2)).astype(np.float16)
    blocks[:, :, :4] = scales.view(np.uint8).reshape(rows, columns // 256, 4)
    return blocks.reshape(rows, -1)


def _gguf_kv(**overrides):
    f = "flint."
    kv = {
        "general.architecture": "flint", f + "block_count": 3, f + "embedding_length": HIDDEN,
        f + "feed_forward_length": WIDTH, f + "attention.head_count": HEADS,
        f + "attention.head_count_kv": KV_HEADS, f + "attention.key_length": HEAD_DIM,
        f + "attention.value_length": HEAD_DIM, f + "attention.sliding_window": 65,
        f + "expert_count": EXPERTS, f + "expert_used_count": 2,
        f + "expert_feed_forward_length": EXPERT_WIDTH,
        f + "expert_shared_feed_forward_length": SHARED, f + "leading_dense_block_count": 1,
        f + "expert_gating_func": 2, f + "expert_weights_norm": True,
        f + "expert_weights_scale": 1.0, f + "rope.freq_base": 10000.0,
        f + "attention.layer_norm_rms_epsilon": 1e-6,
        f + "attention.sliding_window_pattern": [False, True, False],
        f + "attention.rope_pattern": [False, True, False],
    }
    kv.update(overrides)
    return kv


def _release(tmp_path, kv=None):
    """A three-block Flint release (dense full, MoE sliding, MoE full): the HF text files and a
    GGUF in llama.cpp's naming. Block 2's experts are Q5_K, everything else Q8_0."""
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps(_hf_config()))
    vocab = {f"t{i}": i for i in range(VOCAB - 2)}
    (hf / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [{"id": VOCAB - 2, "content": "<|begin_of_text|>", "special": True},
                         {"id": VOCAB - 1, "content": "<|end_of_text|>", "special": True}]}))
    (hf / "tokenizer_config.json").write_text("{}")
    (hf / "chat_template.jinja").write_text("{{ messages }}")
    (hf / "generation_config.json").write_text('{"top_p": 0.95, "top_k": 20}')

    generator = np.random.default_rng(7)
    tensors, stored = [], {}

    def block(name, shape, rows, kind=TYPE_Q8_0):
        tensors.append((name, shape, kind, rows.tobytes()))
        stored[name] = rows

    def vector(name, shape, kind=TYPE_F32):
        values = generator.normal(size=shape).astype(np.float32)
        if kind == TYPE_BF16:
            words = (values.view(np.uint32) >> 16).astype(np.uint16)
            values = (words.astype(np.uint32) << 16).view(np.float32)
            tensors.append((name, shape, kind, words.tobytes()))
        else:
            tensors.append((name, shape, kind, values.tobytes()))
        stored[name] = values

    block("token_embd.weight", (VOCAB, HIDDEN), _q8_0(generator, VOCAB, HIDDEN))
    block("output.weight", (VOCAB, HIDDEN), _q8_0(generator, VOCAB, HIDDEN))
    vector("output_norm.weight", (HIDDEN,))
    q, kvr = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    for i in range(3):
        g = f"blk.{i}."
        block(g + "attn_q.weight", (q, HIDDEN), _q8_0(generator, q, HIDDEN))
        block(g + "attn_k.weight", (kvr, HIDDEN), _q8_0(generator, kvr, HIDDEN))
        block(g + "attn_v.weight", (kvr, HIDDEN), _q8_0(generator, kvr, HIDDEN))
        block(g + "attn_output.weight", (HIDDEN, q), _q8_0(generator, HIDDEN, q))
        vector(g + "attn_q_norm.weight", (HEAD_DIM,))
        vector(g + "attn_k_norm.weight", (HEAD_DIM,))
        vector(g + "attn_sinks.weight", (HEADS,))
        vector(g + "attn_norm.weight", (HIDDEN,))
        vector(g + "ffn_norm.weight", (HIDDEN,))
        if i == 0:
            block(g + "ffn_gate.weight", (WIDTH, HIDDEN), _q8_0(generator, WIDTH, HIDDEN))
            block(g + "ffn_up.weight", (WIDTH, HIDDEN), _q8_0(generator, WIDTH, HIDDEN))
            block(g + "ffn_down.weight", (HIDDEN, WIDTH), _q8_0(generator, HIDDEN, WIDTH))
            continue
        experts = _q5_k if i == 2 else _q8_0
        kind = TYPE_Q5_K if i == 2 else TYPE_Q8_0
        vector(g + "ffn_gate_inp.weight", (EXPERTS, HIDDEN), TYPE_BF16)
        block(g + "ffn_gate_exps.weight", (EXPERTS, EXPERT_WIDTH, HIDDEN),
              experts(generator, EXPERTS * EXPERT_WIDTH, HIDDEN), kind)
        block(g + "ffn_up_exps.weight", (EXPERTS, EXPERT_WIDTH, HIDDEN),
              experts(generator, EXPERTS * EXPERT_WIDTH, HIDDEN), kind)
        block(g + "ffn_down_exps.weight", (EXPERTS, HIDDEN, EXPERT_WIDTH),
              experts(generator, EXPERTS * HIDDEN, EXPERT_WIDTH), kind)
        block(g + "ffn_gate_shexp.weight", (SHARED, HIDDEN), _q8_0(generator, SHARED, HIDDEN))
        block(g + "ffn_up_shexp.weight", (SHARED, HIDDEN), _q8_0(generator, SHARED, HIDDEN))
        block(g + "ffn_down_shexp.weight", (HIDDEN, SHARED), _q8_0(generator, HIDDEN, SHARED))
    path = tmp_path / "flint.gguf"
    write_gguf(path, kv or _gguf_kv(), tensors)
    return hf, path, stored


def test_text_config_normalises_the_layer_kinds_and_refuses_other_variants():
    config = flint.text_config(_hf_config())
    assert config["rope_layers"] == [False, True, False]
    assert config["dense_layers"] == 1
    assert config["layer_types"] == LAYER_TYPES
    for bad in (dict(hidden_act="gelu"), dict(router_score_function="softmax"),
                dict(norm_topk_prob=False), dict(softmax_type="vanilla"),
                dict(tie_word_embeddings=True), dict(mlp_only_layers=[1]),
                dict(attention_bias=True), dict(rope_scaling={"rope_type": "yarn"}),
                dict(layer_types=["full_attention"])):
        with pytest.raises(ValueError):
            flint.text_config(_hf_config(**bad))


@pytest.mark.parametrize("bad", [
    {"flint.expert_gating_func": 1},
    {"flint.attention.rope_pattern": [True, True, False]},
    {"flint.attention.sliding_window_pattern": [False, False, False]},
    {"flint.leading_dense_block_count": 2},
    {"flint.expert_weights_scale": 2.5},
])
def test_the_gguf_metadata_must_match_the_config(tmp_path, bad):
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, path, _ = _release(tmp_path, _gguf_kv(**bad))
    with SafetensorsSource(hf) as base, GGUFFile(path) as gguf:
        with pytest.raises(ValueError):
            flint.build_model(base, gguf)


def test_the_artifact_keeps_the_gguf_rows_and_interleaves_each_experts_gate_and_up(tmp_path):
    from tools.artifact.reader import Artifact
    from tools.convert.pipeline import convert
    from tools.convert.recipe import Recipe
    from tools.convert.sources.safetensors import SafetensorsSource

    hf, path, stored = _release(tmp_path)
    output = tmp_path / "flint.ninfer"
    with SafetensorsSource(hf) as base, GGUFFile(path) as gguf:
        model = flint.build_model(base, gguf)
        recipe = Recipe(model)
        flint.flint_gguf(model, recipe, {"base": base, "gguf": gguf})
        convert(model, recipe, output, device="cpu", rows_per_chunk=96)
    with Artifact(output) as artifact:
        directory = artifact.directory
        config = directory.components["text"]["config"]
        assert config["architectures"] == ["FlintForCausalLM"]
        bindings = directory.bindings

        def stored_bytes(name):
            return artifact.read_object(bindings[name]["object"])

        assert stored_bytes("text/token_embedding") == stored["token_embd.weight"].tobytes()
        assert stored_bytes("text/output_head") == stored["output.weight"].tobytes()
        assert "text/layers/0/moe/router" not in bindings
        assert "text/layers/1/mlp/gate" not in bindings
        for layer in (1, 2):
            g = f"blk.{layer}."
            gate, up = stored[g + "ffn_gate_exps.weight"], stored[g + "ffn_up_exps.weight"]
            rows = gate.shape[1]
            expected = np.concatenate([
                np.concatenate([gate[e * EXPERT_WIDTH:(e + 1) * EXPERT_WIDTH],
                                up[e * EXPERT_WIDTH:(e + 1) * EXPERT_WIDTH]])
                for e in range(EXPERTS)])
            gate_up = np.frombuffer(stored_bytes(f"text/layers/{layer}/moe/gate_up"),
                                    dtype=np.uint8).reshape(-1, rows)
            assert np.array_equal(gate_up, expected)
            assert (stored_bytes(f"text/layers/{layer}/moe/down")
                    == stored[g + "ffn_down_exps.weight"].tobytes())
            # The BF16 router widens exactly to FP32.
            router = np.frombuffer(stored_bytes(f"text/layers/{layer}/moe/router"),
                                   dtype=np.float32)
            assert np.array_equal(router, stored[g + "ffn_gate_inp.weight"].reshape(-1))
        sinks = np.frombuffer(stored_bytes("text/layers/1/attention/sinks"), dtype=np.float32)
        assert np.array_equal(sinks, stored["blk.1.attn_sinks.weight"])
