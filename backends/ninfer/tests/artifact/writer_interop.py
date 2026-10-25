"""Exercise the production Python writer through the C++ reader and materializer."""

from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.artifact.formats import GGUF_FORMATS
from tools.artifact.layouts import encoded_size


def gguf_roundtrip(executable: str, directory: Path) -> int:
    specs = [TensorSpec(name, (3, 512), name, "gguf_blocks_v1")
             for name in ("gguf_q2_0", "gguf_q4_0", "gguf_q5_0")]
    path = directory / "gguf-blocks.ninfer"
    with ArtifactWriter(path, specs, components={"text": {"config": {}}},
                        bindings={spec.id: {"object": spec.id} for spec in specs}) as writer:
        for spec in specs:
            size = encoded_size(spec.layout, GGUF_FORMATS[spec.format], spec.shape)
            writer.write_object(spec.id, bytes((index * 37 + 11) % 251 for index in range(size)))
    return subprocess.run([executable, "--writer-fixture", str(path)]).returncode


def main() -> int:
    executable = sys.argv[1]
    data = bytes((index * 37 + 11) % 251 for index in range(130 * 130 * 2))
    with tempfile.TemporaryDirectory(prefix="ninfer-writer-interop-") as temporary:
        for label, limit in (("single", 1_000_000), ("sharded", 12288)):
            path = Path(temporary) / f"{label}.ninfer"
            with ArtifactWriter(
                path,
                [TensorSpec("matrix", (130, 130), "bf16", "contiguous_le_v1")],
                components={"text": {"config": {}}},
                bindings={
                    "whole": {"object": "matrix"},
                    "reordered_rows": {
                        "parts": [
                            {"object": "matrix", "range": [16770, 16900]},
                            {"object": "matrix", "range": [130, 260]},
                        ]
                    },
                },
                max_file_bytes=limit,
            ) as writer:
                writer.write_object("matrix", data)
            result = subprocess.run([executable, "--writer-fixture", str(path)])
            if result.returncode:
                return result.returncode
        return gguf_roundtrip(executable, Path(temporary))


if __name__ == "__main__":
    raise SystemExit(main())
