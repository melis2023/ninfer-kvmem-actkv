"""Check artifact-to-Docker handoff, including split volumes and incorrect sources."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_image", ROOT / ".github/scripts/extract-native-image.py")
IMAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMAGE)
REVISION = "a" * 40
NAME = "ninfer-linux-x64-cuda13.4-sm75-80-86-89-120a-" + REVISION[:12]


class NativeImageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "artifact"
        self.source.mkdir()

    def artifact(self, revision=REVISION, cuda="13.4", metadata=None):
        architectures = {"12.9": ["70", "75", "80", "86", "89"],
                         "13.4": ["75", "80", "86", "89", "120a"]}[cuda]
        archive_name = f"ninfer-linux-x64-cuda{cuda}-sm{'-'.join(architectures)}-{REVISION[:12]}"
        info = dict(source_revision=revision, modified_sources=False, cuda=cuda,
                    platform="linux-x64", architectures=architectures, layout="shared-runtime-v1")
        info.update(metadata or {})
        files = {
            "Dockerfile": b"FROM debian:13-slim\n",
            "ninfer": b"fixture binary",
            "ninfer-serve": b"fixture server",
            f"backends/cuda{cuda.split('.')[0]}/libninfer-runtime.so": b"fixture runtime",
            "packages/multi.txt": b"libc6:amd64\n",
            "BUILD-INFO.json": json.dumps(info).encode(),
        }
        for command in ('ninfer', 'ninfer-serve', 'ninfer_bench', 'ninfer-calibrate',
                        'ninfer-perplexity', 'ninfer-ngram-profile'):
            files.setdefault(command, b'fixture executable')
            files[f'help/{command}.txt'] = b'fixture help'
        files[f"backends/cuda{cuda.split('.')[0]}/architectures.txt"] = ' '.join(architectures).encode()
        files["SHA256SUMS.txt"] = "".join(
            f"{hashlib.sha256(value).hexdigest()}  {name}\n" for name, value in files.items()).encode()
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name, value in files.items():
                info = tarfile.TarInfo(archive_name + "/" + name)
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
        data = output.getvalue()
        manifest = []
        for index, start in enumerate(range(0, len(data), 97), 1):
            name = f"{archive_name}.tar.gz.{index:03d}"
            chunk = data[start:start + 97]
            (self.source / name).write_bytes(chunk)
            manifest.append(f"{hashlib.sha256(chunk).hexdigest()}  {name}\n")
        (self.source / f"SHA256SUMS-{archive_name}.txt").write_text("".join(manifest))

    def test_split_archive_becomes_context_without_recompilation(self):
        self.artifact()
        context = IMAGE.extract(self.source, self.root / "context", REVISION, "13.4")
        self.assertEqual((context / "ninfer").read_bytes(), b"fixture binary")
        self.assertTrue((context / "Dockerfile").is_file())

    def test_corrupt_volume_is_rejected_before_extraction(self):
        self.artifact()
        (self.source / f"{NAME}.tar.gz.001").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            IMAGE.extract(self.source, self.root / "context", REVISION, "13.4")
        self.assertFalse((self.root / "context").exists())

    def test_source_mismatch_is_rejected(self):
        self.artifact(revision="b" * 40)
        with self.assertRaisesRegex(ValueError, "metadata"):
            IMAGE.extract(self.source, self.root / "context", REVISION, "13.4")

    def test_cuda12_context_includes_volta(self):
        self.artifact(cuda="12.9")
        context = IMAGE.extract(self.source, self.root / "context", REVISION, "12.9")
        info = json.loads((context / "BUILD-INFO.json").read_text())
        self.assertEqual(info["architectures"], ["70", "75", "80", "86", "89"])
        self.assertEqual((context / "ninfer").read_bytes(), b"fixture binary")

    def test_wrong_cuda_metadata_is_rejected(self):
        self.artifact(metadata={"cuda": "12.9"})
        with self.assertRaisesRegex(ValueError, "metadata"):
            IMAGE.extract(self.source, self.root / "context", REVISION, "13.4")

    def test_incomplete_architecture_set_is_rejected(self):
        self.artifact(metadata={"architectures": ["75", "86", "120a"]})
        with self.assertRaisesRegex(ValueError, "metadata"):
            IMAGE.extract(self.source, self.root / "context", REVISION, "13.4")

    def test_wrong_variant_archive_is_rejected(self):
        self.artifact(cuda="12.9")
        with self.assertRaises(FileNotFoundError):
            IMAGE.extract(self.source, self.root / "context", REVISION, "13.4")
