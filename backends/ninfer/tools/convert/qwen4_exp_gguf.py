"""A Qwen3.8-Flash-Next GGUF (llama.cpp ``qwen4exp``) kept in its own ggml block formats.

The GSQ-RCO releases store every matrix in the type its allocation chose, one ggml type per tensor,
so no weight is re-quantised: a row of the artifact is byte for byte a row of the GGUF, and the
expert banks keep the exporter's expert-major layout. BF16 matrices (hyper-connections, router,
indexer, the GDN gates) stay BF16; F32/F16 vectors become the artifact's direct formats.

llama.cpp's exporter conventions are undone where they touch rows or small tensors: Gated DeltaNet
value heads return to the grouped order (``ssm_out`` keeps its tiled input columns, which a row copy
cannot move, and carries them as an ``input_columns`` auxiliary), zero-centred norms drop their
stored ``1 + w`` (every norm but ``ssm_norm``), ``ssm_a`` becomes ``A_log`` again, the head-
interleaved ``attn_q`` splits into query and gate rows, and the indexer's split projections and the
squeezed PLE convolution keep the layout the HF adapter (qwen4_exp.py) declares.

The first shard of a release holds the model; the second holds only ``per_layer_token_embd``, the
n-gram table, which every release shares byte for byte. The recipe imports it row for row as the
``ngram`` component's table, into the model's artifact or into a table artifact of its own, and
records the SHA-256 of its bytes, by which a model stored without the table names the one it reads.
"""

from __future__ import annotations

import hashlib
from math import prod

import numpy as np
import torch

from tools.artifact.formats import GGUF_FORMATS_BY_TYPE

from .gguf_blocks import _vision_source, block_format, block_source
from .methods import AuxiliaryValue, cast_direct, import_encoded
from .sources.gguf import TYPE_BF16, TYPE_F16, TYPE_F32, GGUFFile
from .sources.logical import LogicalSource, array_source
from .ternary import (
    GDN_CHANNELS,
    GDN_HEAD_DIM,
    GDN_KEY_DIM,
    GDN_VALUE_DIM,
    GDN_VALUE_HEADS,
    RowMap,
    attention_rows,
    rows,
    tiled_to_grouped_permutation,
    untile,
    untiled_rows,
)
from .gguf_blocks import tiled_input_columns

HIDDEN = 2560
LAYERS = 48
VOCABULARY = 248320
STREAMS = 4
WIDTH = STREAMS * HIDDEN
LOWRANK = 320
EXPERTS = 512
EXPERT_WIDTH = 640
SHARED_WIDTH = 640
QUERY_ROWS = 24 * 256
KV_ROWS = 2 * 256
HEAD_DIM = 256
INDEXER_HEADS = 4
INDEXER_DIM = 128
NGRAM_WIDTH = 160
NGRAM_HEADS = 16
NGRAM_TENSOR = "per_layer_token_embd.weight"

EXPECTED_HEADER = {
    "general.architecture": "qwen4exp",
    "qwen4exp.block_count": LAYERS,
    "qwen4exp.embedding_length": HIDDEN,
    "qwen4exp.attention.head_count": 24,
    "qwen4exp.attention.head_count_kv": 2,
    "qwen4exp.attention.key_length": HEAD_DIM,
    "qwen4exp.attention.value_length": HEAD_DIM,
    "qwen4exp.expert_used_count": 10,
    "qwen4exp.expert_feed_forward_length": EXPERT_WIDTH,
    "qwen4exp.expert_shared_feed_forward_length": SHARED_WIDTH,
    "qwen4exp.ssm.conv_kernel": 4,
    "qwen4exp.ssm.state_size": GDN_HEAD_DIM,
    "qwen4exp.ssm.group_count": 16,
    "qwen4exp.ssm.time_step_rank": GDN_VALUE_HEADS,
    "qwen4exp.ssm.inner_size": GDN_VALUE_DIM,
    "qwen4exp.full_attention_interval": 4,
    "qwen4exp.hyper_connection.count": STREAMS,
    "qwen4exp.hyper_connection.low_rank": LOWRANK,
    "qwen4exp.attention.indexer.head_count": INDEXER_HEADS,
    "qwen4exp.attention.indexer.key_length": INDEXER_DIM,
    "qwen4exp.attention.indexer.top_k": 2048,
    "qwen4exp.ple.layers": [1],
    "qwen4exp.ple.ngram_size": 3,
    "qwen4exp.ple.heads_per_ngram": 8,
    "qwen4exp.ple.conv_kernel": 4,
    "qwen4exp.embedding_length_per_layer_input": NGRAM_WIDTH,
}

# What a tensor may be stored as: a ggml block matrix or BF16 ("matrix"), or a direct vector.
MATRIX = "matrix"
DIRECT = (TYPE_F32, TYPE_F16, TYPE_BF16)


def untiled_heads() -> RowMap:
    """Grouped value heads of a [heads, k] matrix (one row per head) stored tiled."""

    permutation = tiled_to_grouped_permutation()

    def select(begin: int, end: int) -> np.ndarray:
        return permutation[np.arange(begin, end, dtype=np.int64)]

    return select


def full_attention(layer: int) -> bool:
    return layer % 4 == 3


def _hc(prefix: str) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        prefix + "norm.weight": ((WIDTH,), "direct"),
        prefix + "down.weight": ((LOWRANK, WIDTH), MATRIX),
        prefix + "up.weight": ((WIDTH, LOWRANK), MATRIX),
    }


def expected_tensors(
    ple_layers: tuple[int, ...] = (1,), experts: int = EXPERTS
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Row-major shape and kind of every tensor of the model shard; an expert-pruned release keeps
    `experts` of the 512 (and one router row for each)."""

    out = {
        "token_embd.weight": ((VOCABULARY, HIDDEN), MATRIX),
        "output.weight": ((VOCABULARY, HIDDEN), MATRIX),
        "output_hc_norm.weight": ((WIDTH,), "direct"),
        "output_hc_down.weight": ((LOWRANK, WIDTH), MATRIX),
        "output_hc_up.weight": ((WIDTH, LOWRANK), MATRIX),
    }
    for layer in range(LAYERS):
        p = f"blk.{layer}."
        for half in ("attn", "ffn"):
            out |= {
                p + f"hc_{half}_norm.weight": ((WIDTH,), "direct"),
                p + f"hc_{half}_down.weight": ((LOWRANK, WIDTH), MATRIX),
                p + f"hc_{half}_up.weight": ((WIDTH, LOWRANK), MATRIX),
                p + f"hc_{half}_inject.weight": ((STREAMS, WIDTH), MATRIX),
            }
        out |= {
            p + "ffn_gate_inp.weight": ((experts, HIDDEN), MATRIX),
            p + "ffn_gate_inp_shexp.weight": ((HIDDEN,), "direct"),
            p + "ffn_gate_exps.weight": ((experts, EXPERT_WIDTH, HIDDEN), MATRIX),
            p + "ffn_up_exps.weight": ((experts, EXPERT_WIDTH, HIDDEN), MATRIX),
            p + "ffn_down_exps.weight": ((experts, HIDDEN, EXPERT_WIDTH), MATRIX),
            p + "ffn_gate_shexp.weight": ((SHARED_WIDTH, HIDDEN), MATRIX),
            p + "ffn_up_shexp.weight": ((SHARED_WIDTH, HIDDEN), MATRIX),
            p + "ffn_down_shexp.weight": ((HIDDEN, SHARED_WIDTH), MATRIX),
        }
        if full_attention(layer):
            out |= {
                p + "attn_q.weight": ((2 * QUERY_ROWS, HIDDEN), MATRIX),
                p + "attn_k.weight": ((KV_ROWS, HIDDEN), MATRIX),
                p + "attn_v.weight": ((KV_ROWS, HIDDEN), MATRIX),
                p + "attn_output.weight": ((HIDDEN, QUERY_ROWS), MATRIX),
                p + "attn_q_norm.weight": ((HEAD_DIM,), "direct"),
                p + "attn_k_norm.weight": ((HEAD_DIM,), "direct"),
                p + "indexer.q_proj.weight": ((INDEXER_HEADS * INDEXER_DIM, HIDDEN), MATRIX),
                p + "indexer.k_proj.weight": ((INDEXER_DIM, HIDDEN), MATRIX),
                p + "indexer.q_norm.weight": ((INDEXER_DIM,), "direct"),
                p + "indexer.k_norm.weight": ((INDEXER_DIM,), "direct"),
            }
        else:
            out |= {
                p + "attn_qkv.weight": ((GDN_CHANNELS, HIDDEN), MATRIX),
                p + "attn_gate.weight": ((GDN_VALUE_DIM, HIDDEN), MATRIX),
                p + "ssm_out.weight": ((HIDDEN, GDN_VALUE_DIM), MATRIX),
                p + "ssm_alpha.weight": ((GDN_VALUE_HEADS, HIDDEN), MATRIX),
                p + "ssm_beta.weight": ((GDN_VALUE_HEADS, HIDDEN), MATRIX),
                p + "ssm_a": ((GDN_VALUE_HEADS,), "direct"),
                p + "ssm_dt.bias": ((GDN_VALUE_HEADS,), "direct"),
                p + "ssm_conv1d.weight": ((GDN_CHANNELS, 4), "direct"),
                p + "ssm_norm.weight": ((GDN_HEAD_DIM,), "direct"),
            }
        if layer in ple_layers:
            out |= {
                p + "ple_key.weight": ((WIDTH, HIDDEN), MATRIX),
                p + "ple_value.weight": ((HIDDEN, HIDDEN), MATRIX),
                p + "ple_norm_key.weight": ((WIDTH,), "direct"),
                p + "ple_norm_query.weight": ((WIDTH,), "direct"),
                p + "ple_norm_conv.weight": ((WIDTH,), "direct"),
                p + "ple_conv1d.weight": ((WIDTH, 4), "direct"),
            }
    return out


def _stored(info, kind: str) -> bool:
    if kind == MATRIX:
        return info.type_id in GGUF_FORMATS_BY_TYPE or info.type_id in DIRECT
    return info.type_id in DIRECT


def validate(gguf: GGUFFile, config: dict) -> None:
    """Refuse any GGUF that is not the Flash-Next text model this config describes."""

    for key, expected in EXPECTED_HEADER.items():
        if gguf.kv.get(key) != expected:
            raise ValueError(f"{gguf.path}: {key} = {gguf.kv.get(key)!r}, expected {expected!r}")
    if gguf.kv.get("qwen4exp.expert_count") != config["num_experts"]:
        raise ValueError(
            f"{gguf.path}: {gguf.kv.get('qwen4exp.expert_count')} experts per layer, the model config "
            f"{config['num_experts']} (convert an expert-pruned release with its GGUF as --source gguf)"
        )
    if gguf.kv.get("qwen4exp.ple.eos_token_id") != config["eos_token_id"]:
        raise ValueError(f"{gguf.path}: the PLE EOS differs from the model's")
    if config["ple_layers"] != [1] or config["num_hidden_layers"] != LAYERS:
        raise ValueError("the qwen4exp GGUF recipe implements PLE on block 1 of 48")
    for layer, kind in enumerate(config["layer_types"]):
        if (kind == "full_attention") != full_attention(layer):
            raise ValueError("the qwen4exp GGUF recipe needs QSA on every fourth block")
    expected = expected_tensors(tuple(config["ple_layers"]), config["num_experts"])
    model = {name for name in gguf.tensors if name != NGRAM_TENSOR}
    if model != set(expected):
        missing = sorted(set(expected) - model)[:5]
        extra = sorted(model - set(expected))[:5]
        raise ValueError(f"{gguf.path}: tensor set mismatch (missing {missing}, extra {extra})")
    for name, (shape, kind) in expected.items():
        info = gguf.tensors[name]
        if info.shape != shape or not _stored(info, kind):
            raise ValueError(f"{gguf.path}: {name} is {info.type_name} {info.shape}")
    end = max(info.offset + info.nbytes for info in gguf.tensors.values())
    if gguf.data_bytes_available < end:
        raise ValueError(f"{gguf.path}: the data section is truncated")


def _words(gguf: GGUFFile, tensor: str) -> np.ndarray:
    """A direct tensor's values as float32, row-major."""

    return gguf.read_direct(tensor)


def _direct(values: np.ndarray, dtype: torch.dtype, label: str) -> LogicalSource:
    return array_source(torch.from_numpy(np.array(values, copy=True)).to(dtype), label)


def _norm(gguf: GGUFFile, tensor: str, zero_centred: bool) -> LogicalSource:
    """llama.cpp stores a zero-centred gamma as `1 + w` (exact in FP32 for every BF16 w)."""

    values = _words(gguf, tensor)
    return _direct(values - 1.0 if zero_centred else values, torch.bfloat16, tensor)


def _bf16_rows(gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap) -> LogicalSource:
    """Selected rows of a direct matrix, as BF16 (BF16 words are kept exactly)."""

    info = gguf.info(tensor)
    flat = (prod(info.shape[:-1]), info.shape[-1])
    if info.type_id == TYPE_BF16:
        words = gguf.read_bf16_words(tensor).reshape(flat)
        index = select(0, shape[0])
        values = torch.from_numpy(np.ascontiguousarray(words[index]).view(np.int16)).view(
            torch.bfloat16
        )
    else:
        values = torch.from_numpy(
            np.ascontiguousarray(_words(gguf, tensor).reshape(flat)[select(0, shape[0])])
        ).to(torch.bfloat16)
    return array_source(values.reshape(shape), f"{tensor}[{info.type_name}]{list(shape)}")


def matrix_source(
    gguf: GGUFFile, tensor: str, shape: tuple[int, int], select: RowMap
) -> tuple[LogicalSource, bool]:
    """Rows of a stored matrix: encoded ggml rows (True) or BF16 values (False)."""

    if gguf.info(tensor).type_id in GGUF_FORMATS_BY_TYPE:
        return block_source(gguf, tensor, shape, select), True
    return _bf16_rows(gguf, tensor, shape, select), False


def text_sources(gguf: GGUFFile, config: dict) -> dict[str, tuple[LogicalSource, bool]]:
    """Every text parameter's GGUF source; True marks encoded block rows."""

    out: dict[str, tuple[LogicalSource, bool]] = {}

    def matrix(name: str, tensor: str, shape: tuple[int, int], select: RowMap = rows()):
        out[name] = matrix_source(gguf, tensor, shape, select)

    def direct(name: str, source: LogicalSource):
        out[name] = (source, False)

    def hc(prefix: str, stored: str, inject: bool):
        direct(prefix + "norm", _norm(gguf, stored + "norm.weight", True))
        matrix(prefix + "down", stored + "down.weight", (LOWRANK, WIDTH))
        matrix(prefix + "up", stored + "up.weight", (WIDTH, LOWRANK))
        if inject:
            matrix(prefix + "inject", stored + "inject.weight", (STREAMS, WIDTH))

    matrix("text/token_embedding", "token_embd.weight", (VOCABULARY, HIDDEN))
    matrix("text/output_head", "output.weight", (VOCABULARY, HIDDEN))
    hc("text/final_mixer/", "output_hc_", inject=False)
    for layer in range(LAYERS):
        g, p = f"blk.{layer}.", f"text/layers/{layer}/"
        hc(p + "attn_hc/", g + "hc_attn_", inject=True)
        hc(p + "mlp_hc/", g + "hc_ffn_", inject=True)
        if full_attention(layer):
            a = p + "attention/"
            matrix(a + "query", g + "attn_q.weight", (QUERY_ROWS, HIDDEN), attention_rows(False))
            matrix(a + "gate", g + "attn_q.weight", (QUERY_ROWS, HIDDEN), attention_rows(True))
            matrix(a + "key", g + "attn_k.weight", (KV_ROWS, HIDDEN))
            matrix(a + "value", g + "attn_v.weight", (KV_ROWS, HIDDEN))
            matrix(a + "output", g + "attn_output.weight", (HIDDEN, QUERY_ROWS))
            direct(a + "query_norm", _norm(gguf, g + "attn_q_norm.weight", True))
            direct(a + "key_norm", _norm(gguf, g + "attn_k_norm.weight", True))
            i = p + "indexer/"
            matrix(i + "query", g + "indexer.q_proj.weight", (INDEXER_HEADS * INDEXER_DIM, HIDDEN))
            matrix(i + "key", g + "indexer.k_proj.weight", (INDEXER_DIM, HIDDEN))
            direct(i + "query_norm", _norm(gguf, g + "indexer.q_norm.weight", True))
            direct(i + "key_norm", _norm(gguf, g + "indexer.k_norm.weight", True))
        else:
            n = p + "gdn/"
            qkv = g + "attn_qkv.weight"
            matrix(n + "query", qkv, (GDN_KEY_DIM, HIDDEN))
            matrix(n + "key", qkv, (GDN_KEY_DIM, HIDDEN), rows(GDN_KEY_DIM))
            matrix(n + "value", qkv, (GDN_VALUE_DIM, HIDDEN), untiled_rows(2 * GDN_KEY_DIM))
            matrix(n + "z", g + "attn_gate.weight", (GDN_VALUE_DIM, HIDDEN), untiled_rows(0))
            # Tiled input columns: the Use gathers the grouped activation (tiled_input_columns).
            matrix(n + "output", g + "ssm_out.weight", (HIDDEN, GDN_VALUE_DIM))
            for role, tensor in (("a_projection", "ssm_alpha.weight"),
                                 ("b_projection", "ssm_beta.weight")):
                matrix(n + role, g + tensor, (GDN_VALUE_HEADS, HIDDEN), untiled_heads())
            ssm_a = untile(_words(gguf, g + "ssm_a"), 1).astype(np.float64)
            if not np.all(ssm_a < 0):
                raise ValueError(f"{g}ssm_a must be strictly negative (-exp(A_log))")
            direct(n + "a_log", _direct(np.log(-ssm_a).astype(np.float32), torch.float32,
                                        g + "ssm_a"))
            direct(n + "dt_bias", _direct(untile(_words(gguf, g + "ssm_dt.bias"), 1),
                                          torch.float32, g + "ssm_dt.bias"))
            taps = _words(gguf, g + "ssm_conv1d.weight")
            channels = np.concatenate([taps[: 2 * GDN_KEY_DIM],
                                       untile(taps[2 * GDN_KEY_DIM:], GDN_HEAD_DIM)])
            direct(n + "convolution", _direct(channels.T, torch.bfloat16, g + "ssm_conv1d.weight"))
            direct(n + "norm", _norm(gguf, g + "ssm_norm.weight", False))
        m = p + "moe/"
        experts = config["num_experts"]
        matrix(m + "router", g + "ffn_gate_inp.weight", (experts, HIDDEN))
        direct(m + "shared_score", _direct(_words(gguf, g + "ffn_gate_inp_shexp.weight")
                                           .reshape(1, HIDDEN), torch.bfloat16,
                                           g + "ffn_gate_inp_shexp.weight"))
        for expert in range(experts):
            e = m + f"experts/{expert}/"
            matrix(e + "gate", g + "ffn_gate_exps.weight", (EXPERT_WIDTH, HIDDEN),
                   rows(expert * EXPERT_WIDTH))
            matrix(e + "up", g + "ffn_up_exps.weight", (EXPERT_WIDTH, HIDDEN),
                   rows(expert * EXPERT_WIDTH))
            matrix(e + "down", g + "ffn_down_exps.weight", (HIDDEN, EXPERT_WIDTH),
                   rows(expert * HIDDEN))
        matrix(m + "shared/gate", g + "ffn_gate_shexp.weight", (SHARED_WIDTH, HIDDEN))
        matrix(m + "shared/up", g + "ffn_up_shexp.weight", (SHARED_WIDTH, HIDDEN))
        matrix(m + "shared/down", g + "ffn_down_shexp.weight", (HIDDEN, SHARED_WIDTH))
        if layer in config["ple_layers"]:
            q = p + "ple/"
            matrix(q + "key", g + "ple_key.weight", (WIDTH, HIDDEN))
            matrix(q + "value", g + "ple_value.weight", (HIDDEN, HIDDEN))
            for role in ("norm_key", "norm_query", "norm_conv"):
                direct(q + role, _norm(gguf, g + f"ple_{role}.weight", True))
            # Each channel's taps stay contiguous, oldest first (the HF conv squeezed).
            direct(q + "convolution", _direct(_words(gguf, g + "ple_conv1d.weight"),
                                              torch.bfloat16, g + "ple_conv1d.weight"))
    return out


def _assign(recipe, name: str, source: LogicalSource, encoded: bool, model) -> str:
    if encoded:
        format = source.read_encoded(0, 1).format
        recipe.assign(name, format=format, method=import_encoded, source=source,
                      activation_policy="AllowA8")
        return format
    format = model.parameters[name].direct_format
    recipe.assign(name, format=format, method=cast_direct, source=source)
    return format


def _group_same_format(recipe, names: list[str], formats: dict[str, str]) -> None:
    if len({formats[name] for name in names}) == 1:
        recipe.group(names)


def qwen3_8_flash_next_gguf(model, recipe, sources):
    """A Qwen3.8-Flash-Next GSQ-RCO GGUF release in its own block formats: the model shard as
    `--source gguf=SHARD1.gguf` (for the text component) and the n-gram table shard as
    `--source ngram=SHARD2.gguf`, which is always read, since the model records the digest of the
    table it reads even when the table's rows go into an artifact of their own. The Vision tower
    comes from the release's BF16 `mmproj` GGUF (`--source vision=mmproj.gguf`) and stays BF16."""

    table = sources["ngram"]
    if not isinstance(table, GGUFFile):
        raise ValueError("--source ngram must name the release's n-gram table shard (.gguf)")
    descriptor = model.components["ngram"]["config"]
    source = ngram_source(table, descriptor["rows"])
    descriptor["format"] = source.read_encoded(0, 1).format
    size = table.info(NGRAM_TENSOR).nbytes
    print(f"hashing the n-gram table ({size / 1e9:.1f} GB)", flush=True)
    descriptor["table_sha256"] = table_digest(table)
    if "text" not in model.components:
        recipe.assign("ngram/table", format=descriptor["format"], method=import_encoded,
                      source=source)
        return
    config = model.config
    if config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("the qwen4exp GGUF recipe requires a Qwen3.8-Flash-Next model")
    gguf = sources["gguf"]
    if not isinstance(gguf, GGUFFile):
        raise ValueError("--source gguf must name the model's first .gguf shard")
    validate(gguf, config)
    formats: dict[str, str] = {}
    for name, (text_source, encoded) in text_sources(gguf, config).items():
        formats[name] = _assign(recipe, name, text_source, encoded, model)
    if "ngram/table" in model.parameters:
        formats["ngram/table"] = descriptor["format"]
        recipe.assign("ngram/table", format=descriptor["format"], method=import_encoded,
                      source=source)
    if "vision" in model.components:
        vision = _vision_source(model, sources)
        for name, parameter in model.parameters.items():
            if name.startswith("vision/"):
                formats[name] = parameter.direct_format
                recipe.assign(name, format=parameter.direct_format, method=cast_direct,
                              source=model.source(name, vision))
    if set(formats) != set(model.parameters):
        missing = sorted(set(model.parameters) - set(formats))[:5]
        raise ValueError(f"the GGUF leaves logical parameters without a source: {missing}")
    for layer in range(LAYERS):
        p = f"text/layers/{layer}/"
        if full_attention(layer):
            a = p + "attention/"
            _group_same_format(recipe, [a + "query", a + "gate"], formats)
            _group_same_format(recipe, [a + "key", a + "value"], formats)
            _group_same_format(recipe, [p + "indexer/query", p + "indexer/key"], formats)
        else:
            n = p + "gdn/"
            qkv = [n + "query", n + "key", n + "value"]
            recipe.group(qkv + [n + "z"] if formats[n + "z"] == formats[n + "query"] else qkv)
            recipe.group([n + "a_projection", n + "b_projection"])
        m = p + "moe/"
        experts = config["num_experts"]
        gates = [m + f"experts/{e}/gate" for e in range(experts)]
        ups = [m + f"experts/{e}/up" for e in range(experts)]
        if formats[gates[0]] == formats[ups[0]]:
            # Each expert's gate rows then its up rows, expert-major: one [gate; up] parent per
            # expert, one bank per layer.
            recipe.group([name for pair in zip(gates, ups) for name in pair])
        else:
            recipe.group(gates)
            recipe.group(ups)
        recipe.group([m + f"experts/{e}/down" for e in range(experts)])
        recipe.group([m + "router", m + "shared_score"])
        _group_same_format(recipe, [m + "shared/gate", m + "shared/up"], formats)
        if layer in config["ple_layers"]:
            _group_same_format(recipe, [p + "ple/key", p + "ple/value"], formats)
    columns = AuxiliaryValue(
        "int32", (GDN_VALUE_DIM,), tiled_input_columns().astype("<i4").tobytes()
    )
    for layer in range(LAYERS):
        if full_attention(layer):
            continue
        name = f"text/layers/{layer}/gdn/output"
        for input_name in model.parameters[name].inputs:
            recipe.use(name, input_name, auxiliaries={"input_columns": columns})


def with_gguf_expert_count(config: dict, path) -> dict:
    """The HF config with the expert count of the GGUF at `path`: an expert-pruned release (the
    Coder build keeps 256 of 512 per layer) is that model's config with fewer experts."""

    from copy import deepcopy

    with GGUFFile(path) as gguf:
        count = gguf.kv.get("qwen4exp.expert_count")
    text = config.get("text_config", config)
    if type(count) is not int or count == text.get("num_experts"):
        return config
    if not 10 <= count <= text.get("num_experts", 0):
        raise ValueError(f"{path}: {count} experts per layer cannot be a pruning of this model")
    out = deepcopy(config)
    out.get("text_config", out)["num_experts"] = count
    return out


def ngram_source(gguf: GGUFFile, rows_count: int) -> LogicalSource:
    info = gguf.info(NGRAM_TENSOR)
    if info.shape != (rows_count, NGRAM_WIDTH) or info.type_id not in GGUF_FORMATS_BY_TYPE:
        raise ValueError(f"{gguf.path}: {NGRAM_TENSOR} is {info.type_name} {info.shape}")
    return block_source(gguf, NGRAM_TENSOR, (rows_count, NGRAM_WIDTH), rows())


def table_digest(gguf: GGUFFile) -> str:
    """SHA-256 of the n-gram table's bytes, which the artifact stores unchanged, row for row."""

    size = gguf.info(NGRAM_TENSOR).nbytes
    digest = hashlib.sha256()
    step = 1 << 28
    for begin in range(0, size, step):
        digest.update(gguf.tensor_bytes(NGRAM_TENSOR, begin, min(begin + step, size)))
    return digest.hexdigest()


RECIPES = {
    "qwen3_8_flash_next_gguf": qwen3_8_flash_next_gguf,
}

__all__ = [
    "EXPECTED_HEADER",
    "NGRAM_TENSOR",
    "RECIPES",
    "expected_tensors",
    "with_gguf_expert_count",
    "ngram_source",
    "qwen3_8_flash_next_gguf",
    "table_digest",
    "text_sources",
    "validate",
]
