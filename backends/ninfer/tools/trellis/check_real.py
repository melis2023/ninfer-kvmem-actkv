import hashlib
import numpy as np
from tools.artifact.reader import Artifact
from tools.trellis.verify_trellis import decode_tiles_word_windows, decode_tiles_bitstream, decode_matrix
from tools.convert.sources.gguf import GGUFFile

ART = r"I:\model\Qwen3.8-27B-NariQ-3.0bpw.ninfer"
GGUF = r"I:\model\Qwen3.8-27B-NariQ-3.0bpw.gguf"

A = Artifact.open(ART)
d = A.directory
print("objects:", len(A.objects), "components:", list(d.components), "metadata keys:", list(d.metadata))
heads = d.metadata["trellis.head_bands"]["output"]
print("head bands:", len(heads), heads[0], heads[1])
tensors = [o for o in A.objects if o.format.startswith("trellis_trl")]
print("formats:", sorted({o.format for o in tensors}), "count:", len(tensors),
      "payload bytes:", sum(o.bytes for o in tensors))
g = d.metadata["trellis.head_vocab_gather"]["output"]
print("gather:", g, A.object(g).shape)

def check(name):
    obj = A.object(name)
    bits = int(obj.format[-1])
    n, k = obj.shape
    codes = np.frombuffer(A.read_object(name), dtype=np.uint8)
    with GGUFFile(GGUF) as f:
        src = np.asarray(f.tensor_bytes(name))
    same = hashlib.sha256(codes).hexdigest() == hashlib.sha256(src).hexdigest()
    a = decode_tiles_word_windows(codes, n, k, bits)
    b = decode_tiles_bitstream(codes, n, k, bits)
    s16 = np.frombuffer(A.read_object(name + "~s16"), dtype=np.float32)
    w = decode_matrix(codes, n, k, bits, s16)
    print(f"{name}: {obj.format} {n}x{k} payload==gguf:{same} decoders:{np.array_equal(a, b)} "
          f"grid=[{b.min()},{b.max()}] w_std={w.std():.4f} w_absmax={np.abs(w).max():.4f}")
    return same and np.array_equal(a, b)

ok = all(check(n) for n in ("blk.0.ffn_up.weight", "output.band00.weight", "output.band03.weight",
                            "blk.0.attn_qkv.weight", "token_embd.weight", "blk.3.attn_q.weight"))
print("real-weight check:", "ok" if ok else "FAILED")
A.close()

