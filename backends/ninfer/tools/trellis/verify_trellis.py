"""NariQ coded-tile decoder and self-test, independent of the C++ host reference.

Two formulations of FORMAT.md's tile decode are implemented and compared bit for bit:

* ``decode_tiles_word_windows`` reads the 16-bit window out of the two words that straddle the
  position (the C++ and CUDA kernels' formulation);
* ``decode_tiles_bitstream`` materializes the circular bit stream once and slices it;

then ``decode_matrix`` assembles the 16x16 tiles into ``W_hat`` (s16 applied, side transforms not).

Usage (from `backends/ninfer`):

    python -m tools.trellis.verify_trellis selftest
    python -m tools.trellis.verify_trellis inspect path/to/Qwen3.8-27B-NariQ-3.0bpw.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from tools.artifact.reader import Artifact
from tools.convert.trellis_gguf import convert

GRID_MULTIPLIER = 0x83DCD12D


def tile_positions() -> tuple[np.ndarray, np.ndarray]:
    """ring position p -> (row, column) of the 16x16 tile, in exllamav3's tensor-core order."""
    p = np.arange(256)
    a = p >> 3
    jj = p & 7
    row = (a >> 2) + 8 * ((jj >> 2) & 1)
    col = 2 * (a & 3) + (jj & 1) + 8 * ((jj >> 1) & 1)
    return row, col


def _grid(state: np.ndarray) -> np.ndarray:
    x = (state.astype(np.uint32) * np.uint32(GRID_MULTIPLIER)).astype(np.uint32)
    return (((x >> 2) & 63) + ((x >> 10) & 63) + ((x >> 18) & 63) + ((x >> 26) & 63)).astype(np.int16) - 126


def decode_tiles_word_windows(codes: np.ndarray, n: int, k: int, bits: int) -> np.ndarray:
    words = np.ascontiguousarray(codes).view(np.uint32).reshape(n // 16, k // 16, 8 * bits)
    total = 256 * bits
    values = np.empty((n // 16, k // 16, 16, 16), dtype=np.int16)
    row, col = tile_positions()
    for p in range(256):
        start = ((p + 1) * bits - 16 + total) % total
        wi, offset = start >> 5, start & 31
        pair = (words[..., wi].astype(np.uint64) << np.uint64(32)) | words[
            ..., (wi + 1) % (8 * bits)
        ].astype(np.uint64)
        state = ((pair >> np.uint64(48 - offset)) & np.uint64(0xFFFF)).astype(np.uint32)
        values[:, :, row[p], col[p]] = _grid(state)
    return values


def decode_tiles_bitstream(codes: np.ndarray, n: int, k: int, bits: int) -> np.ndarray:
    """The same window read out of an explicit big-endian bit stream, latest bit first."""
    words = np.ascontiguousarray(codes).view(np.uint32).reshape(n // 16, k // 16, 8 * bits)
    total = 256 * bits
    # stream[q] = bit 31 - q % 32 of word q / 32, i.e. words read most significant bit first.
    shifts = np.arange(32, dtype=np.uint32)
    stream = ((words[..., :, None] >> (31 - shifts)[None, None, :]) & 1).reshape(
        n // 16, k // 16, total
    )
    stream = np.concatenate([stream, stream[..., :16]], axis=2)
    values = np.empty((n // 16, k // 16, 16, 16), dtype=np.int16)
    row, col = tile_positions()
    limits = np.arange(16, dtype=np.uint32)
    for p in range(256):
        start = ((p + 1) * bits - 16 + total) % total
        # The window's last stream bit is the value's bit 0, so the sixteen bits assemble reversed.
        window = stream[..., start : start + 16][..., ::-1].astype(np.uint32)
        state = (window << limits).sum(axis=2, dtype=np.uint32) & np.uint32(0xFFFF)
        values[:, :, row[p], col[p]] = _grid(state)
    return values


def decode_matrix(codes: np.ndarray, n: int, k: int, bits: int, s16: np.ndarray) -> np.ndarray:
    """W_hat [N,K] in FP32: the decoded integer grid times the per-input-tile scale."""
    tiles = decode_tiles_word_windows(codes, n, k, bits)
    out = np.empty((n, k), dtype=np.float32)
    for i in range(k // 16):
        out[:, i * 16 : (i + 1) * 16] = (
            tiles[:, i].reshape(n, 16).astype(np.float32) * np.float32(s16[i])
        )
    return out


def _write_synthetic(path: Path, *, bits: int = 3, k: int = 64, rows: int = 32,
                     band_rows: int = 16, bands: int = 2) -> None:
    """A small GGUF laid out the way the NariQ exporter writes one (flat weight + banded head)."""
    from tools.convert.sources.gguf import TYPE_F16, TYPE_F32, TYPE_I32, TYPE_TRL1, write_gguf

    rng = np.random.default_rng(7)
    tensor_type = TYPE_TRL1 + bits - 1
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
    kv: dict = {"general.architecture": "qwen35", "trellis.version": 1}

    def codes(value_rows: int) -> bytes:
        # one tile per (input tile, output tile): 8*bits words of 256 weights
        words = rng.integers(
            0, 2**32, size=(value_rows // 16) * (k // 16) * (8 * bits), dtype=np.uint32
        )
        return words.astype("<u4").tobytes()

    def sides(base: str, value_rows: int, *, pre: np.ndarray | None = None,
              s16: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        pre = rng.choice([-1.0, 1.0], size=k).astype(np.float32) if pre is None else pre
        s16 = (np.abs(rng.normal(0.02, 0.005, size=k // 16)) + 1e-3).astype(np.float16) \
            if s16 is None else s16
        post = rng.normal(1.0, 0.05, size=value_rows).astype(np.float32)
        tensors.extend(
            [
                (base + ".trellis_s16", (k // 16,), TYPE_F16, s16.tobytes()),
                (base + ".trellis_pre", (k,), TYPE_F32, pre.tobytes()),
                (base + ".trellis_post", (value_rows,), TYPE_F32, post.tobytes()),
            ]
        )
        return pre, s16

    # 1) a plain coded projection: rows side is a block RHT, columns side a whole-slice Hadamard.
    kv["trellis.desc.blk.0.attn_q"] = [1, 16, 1, 2, 0, 1, 0]
    tensors.append(("blk.0.attn_q.weight", (rows, k), tensor_type, codes(rows)))
    _pre, _s16 = sides("blk.0.attn_q", rows)
    tensors.append(("blk.0.attn_q.trellis_pin", (k,), TYPE_I32,
                    rng.permutation(k).astype(np.int32).tobytes()))
    tensors.append(("blk.0.attn_q.trellis_pout", (rows,), TYPE_I32,
                    rng.permutation(rows).astype(np.int32).tobytes()))

    # 2) a banded output head: two bands sharing pre/s16/descriptor, composed through vocab_gather.
    vocab = bands * band_rows
    gather = rng.permutation(vocab).astype(np.int32)
    pre = rng.choice([-1.0, 1.0], size=k).astype(np.float32)
    s16 = (np.abs(rng.normal(0.02, 0.005, size=k // 16)) + 1e-3).astype(np.float16)
    for band in range(bands):
        base = f"output.band{band:02d}"
        kv[f"trellis.desc.{base}"] = [0, 0, 1, 2, 0, 1, 0]
        tensors.append((base + ".weight", (band_rows, k), tensor_type, codes(band_rows)))
        sides(base, band_rows, pre=pre, s16=s16)
    tensors.append(("output.vocab_gather", (vocab,), TYPE_I32, gather.tobytes()))
    kv["trellis.lm_head_bands"] = bands
    write_gguf(path, kv, tensors)


def selftest() -> int:
    # Windows keeps the source memmap open until the reader is collected; the scratch directory is
    # allowed to outlive the process instead of failing the test on cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        source = root / "nariq-synthetic.gguf"
        target = root / "nariq-synthetic.ninfer"
        _write_synthetic(source)
        report = convert(source, target)
        print(f"converted {report['coded_tensors']} coded matrices -> {report['bytes']} bytes")

        with Artifact.open(target) as artifact:
            for obj in artifact.objects:
                if not obj.format.startswith("trellis_trl"):
                    continue
                bits = int(obj.format[-1])
                codes = np.frombuffer(artifact.read_object(obj.id), dtype=np.uint8)
                n, k = obj.shape
                s16 = np.frombuffer(artifact.read_object(obj.id + "~s16"), dtype=np.float32)
                tiles_a = decode_tiles_word_windows(codes, n, k, bits)
                tiles_b = decode_tiles_bitstream(codes, n, k, bits)
                if not np.array_equal(tiles_a, tiles_b):
                    raise SystemExit(f"{obj.id}: the two decoders disagree")
                if np.abs(tiles_b).max() > 126:
                    raise SystemExit(f"{obj.id}: decoded grid leaves [-126,126]")
                weights = decode_matrix(codes, n, k, bits, s16)
                if not np.isfinite(weights).all():
                    raise SystemExit(f"{obj.id}: nonfinite decoded weights")
                print(f"  {obj.id}: {obj.shape} {obj.format} grid=[{tiles_b.min()},{tiles_b.max()}] "
                      f"w_std={weights.std():.4f}")
        print("selftest ok")
    return 0


def inspect(path: str) -> int:
    from tools.convert.sources.gguf import GGUFFile
    from tools.convert.trellis_gguf import coded_matrices

    with GGUFFile(path) as source:
        print(f"trellis.version = {source.kv.get('trellis.version')!r}")
        for matrix in coded_matrices(source):
            print(f"{matrix.name}: N={matrix.n} K={matrix.k} bits={matrix.bits} desc={matrix.desc} "
                  f"pout={'yes' if matrix.pout is not None else 'no'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("selftest")
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("path")
    args = parser.parse_args(argv)
    return selftest() if args.command == "selftest" else inspect(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
