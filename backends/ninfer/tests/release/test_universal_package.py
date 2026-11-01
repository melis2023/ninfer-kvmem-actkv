"""Verify backend assembly, provenance rejection, and portable archive contents."""
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('universal', ROOT / 'scripts/assemble-universal-release.py')
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)
REVISION = 'a' * 40


class UniversalPackageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roots = {}
        for cuda, architectures in PACKAGE.VARIANTS.items():
            name = f"ninfer-linux-x64-cuda{cuda}-sm{'-'.join(architectures)}-{REVISION[:12]}"
            root = self.root / 'inputs' / name
            self.roots[cuda] = root
            major = cuda.split('.')[0]
            files = {'BUILD-INFO.json': json.dumps(dict(
                layout='shared-runtime-v1', source_revision=REVISION, modified_sources=False,
                platform='linux-x64', cuda=cuda, architectures=architectures, commands=PACKAGE.COMMANDS)),
                f'backends/cuda{major}/architectures.txt': ' '.join(architectures),
                f'backends/cuda{major}/libninfer-runtime.so': 'runtime ' + cuda,
                f'backends/cuda{major}/NVIDIA-CUDA-EULA.txt': 'license ' + cuda,
                'packages/multi.txt': f'libc6\nfixture-cuda{major}\n',
                'Dockerfile': 'fixture docker', 'LICENSE-ffmpeg.txt': 'fixture notice'}
            for command in PACKAGE.COMMANDS:
                files[command] = 'launcher ' + cuda
                files[f'help/{command}.txt'] = 'help ' + command
            for relative, value in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            self.hashes(root)
        self.destination = self.root / f'ninfer-linux-x64-cuda12.9-13.4-{REVISION[:12]}'

    def hashes(self, root):
        paths = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
        (root / 'SHA256SUMS.txt').write_text(''.join(
            f'{PACKAGE.digest(p)}  {p.relative_to(root).as_posix()}\n' for p in paths))

    def test_both_backends_and_dependency_union_survive_archive(self):
        PACKAGE.assemble(self.roots, self.destination, 'linux-x64', REVISION)
        self.assertEqual((self.destination / 'ninfer').read_text(), 'launcher 13.4')
        for cuda in (12, 13):
            self.assertTrue((self.destination / f'backends/cuda{cuda}/libninfer-runtime.so').is_file())
        self.assertEqual((self.destination / 'packages/multi.txt').read_text(),
                         'fixture-cuda12\nfixture-cuda13\nlibc6\n')
        self.assertFalse((self.destination / 'Dockerfile').exists())
        info = json.loads((self.destination / 'BUILD-INFO.json').read_text())
        self.assertEqual(set(info['backends']), {'12.9', '13.4'})
        PACKAGE.archive(self.destination, 'linux-x64')
        archive = self.destination.with_name(self.destination.name + '.tar.gz.001')
        with tarfile.open(archive) as tar:
            self.assertTrue(tar.getmember(self.destination.name + '/backends/cuda12/libninfer-runtime.so').isfile())
        manifest = self.destination.with_name(f'SHA256SUMS-{self.destination.name}.txt')
        self.assertEqual(manifest.read_text(), f'{PACKAGE.digest(archive)}  {archive.name}\n')

    def test_different_revision_is_rejected_before_assembly(self):
        path = self.roots['12.9'] / 'BUILD-INFO.json'
        info = json.loads(path.read_text())
        info['source_revision'] = 'b' * 40
        path.write_text(json.dumps(info))
        self.hashes(path.parent)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            PACKAGE.assemble(self.roots, self.destination, 'linux-x64', REVISION)
        self.assertFalse(self.destination.exists())

    def test_corrupt_backend_is_rejected_before_assembly(self):
        (self.roots['12.9'] / 'backends/cuda12/libninfer-runtime.so').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            PACKAGE.assemble(self.roots, self.destination, 'linux-x64', REVISION)
        self.assertFalse(self.destination.exists())

    def test_unlisted_dependency_is_rejected(self):
        (self.roots['12.9'] / 'backends/cuda12/unlisted.so').write_text('unlisted')
        with self.assertRaisesRegex(ValueError, 'cover all files'):
            PACKAGE.assemble(self.roots, self.destination, 'linux-x64', REVISION)

    def test_single_variant_archive_extracts_for_assembly(self):
        root = self.roots['12.9']
        PACKAGE.archive(root, 'linux-x64')
        result = PACKAGE.extract(root.parent, self.root / 'extracted', 'linux-x64', '12.9', REVISION)
        self.assertEqual((result / 'backends/cuda12/libninfer-runtime.so').read_text(), 'runtime 12.9')


if __name__ == '__main__':
    unittest.main()
