"""Qwen3.5 dense GGUF geometry and exporter head-order transforms."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DenseGeometry:
    hidden: int
    intermediate: int
    vocabulary: int
    layers: int
    attention_heads: int
    kv_heads: int
    attention_dim: int
    key_heads: int
    value_heads: int
    head_dim: int
    taps: int
    tied: bool

    @classmethod
    def from_config(cls, config):
        if "num_experts" in config:
            raise ValueError("the dense GGUF recipe does not accept MoE weights")
        result = cls(*(config[k] for k in (
            "hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
            "linear_conv_kernel_dim", "tie_word_embeddings")))
        if (any(type(v) is not int or v <= 0 for v in tuple(vars(result).values())[:-1])
                or type(result.tied) is not bool
                or config["linear_value_head_dim"] != result.head_dim
                or result.value_heads % result.key_heads
                or result.attention_heads % result.kv_heads
                or result.attention_dim != 256 or result.head_dim != 128 or result.taps != 4
                or (result.attention_heads, result.kv_heads) not in ((8, 2), (16, 4), (24, 4))
                or result.hidden % 128 or result.key_heads != 16
                or result.value_heads not in (16, 32, 48)
                or config["layer_types"] != ["full_attention" if result.full_attention(i)
                    else "linear_attention" for i in range(result.layers)]):
            raise ValueError("invalid dense GGUF head geometry or layer layout")
        return result

    @property
    def query_rows(self): return self.attention_heads * self.attention_dim

    @property
    def kv_rows(self): return self.kv_heads * self.attention_dim

    @property
    def key_dim(self): return self.key_heads * self.head_dim

    @property
    def value_dim(self): return self.value_heads * self.head_dim

    @property
    def channels(self): return 2 * self.key_dim + self.value_dim

    @staticmethod
    def full_attention(layer): return (layer + 1) % 4 == 0

    def expected_header(self):
        return {"general.architecture": "qwen35", **{"qwen35." + key: value for key, value in {
            "embedding_length": self.hidden, "feed_forward_length": self.intermediate,
            "attention.head_count": self.attention_heads, "attention.head_count_kv": self.kv_heads,
            "attention.key_length": self.attention_dim, "attention.value_length": self.attention_dim,
            "ssm.conv_kernel": self.taps, "ssm.state_size": self.head_dim,
            "ssm.group_count": self.key_heads, "ssm.time_step_rank": self.value_heads,
            "ssm.inner_size": self.value_dim, "full_attention_interval": 4}.items()}}

    def untile(self, array, head_dim):
        if array.shape[0] != self.value_heads * head_dim:
            raise ValueError("GDN value-head row count differs from config")
        per_key = self.value_heads // self.key_heads
        heads = np.arange(self.value_heads)
        indices = (heads % per_key) * self.key_heads + heads // per_key
        return np.ascontiguousarray(array.reshape(self.value_heads, head_dim, *array.shape[1:])
                                   [indices].reshape(array.shape))

    def untiled_rows(self, base):
        per_key = self.value_heads // self.key_heads
        def select(begin, end):
            r = np.arange(begin, end, dtype=np.int64)
            heads, lanes = r // self.head_dim, r % self.head_dim
            return base + ((heads % per_key) * self.key_heads + heads // per_key) * self.head_dim + lanes
        return select

    def attention_rows(self, gate):
        def select(begin, end):
            r = np.arange(begin, end, dtype=np.int64)
            d = self.attention_dim
            return (r // d) * 2 * d + (d if gate else 0) + r % d
        return select
