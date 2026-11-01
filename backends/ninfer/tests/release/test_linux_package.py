"""Check archive contents and failure boundaries with fixture executables, without a GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('linux_package', ROOT / 'scripts/package-linux-release.py')
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


def fatbin(payload):
    return struct.pack('<IHHQ', 0xBA55ED50, 1, 16, len(payload)) + payload


def section_listing(offset, size):
    return f'  [29] .nv_fatbin PROGBITS 0000000001000000 {offset:06x} {size:06x} 00 A 0 0 8\n'


class DeviceImagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.binary = Path(self.tmp.name) / 'runtime.so'
        self.modules = [fatbin(b'first'), fatbin(b'second module')]
        self.section = b''.join(module + bytes((-len(module)) % 8) for module in self.modules)
        self.binary.write_bytes(b'prefix' + self.section + b'unrelated ELF data')
        self.listing = section_listing(6, len(self.section))
        self.seen = []

    def output(self, *args):
        if args[0] == 'readelf':
            return self.listing
        self.assertEqual(args[:2], ('cuobjdump', '--list-elf'))
        self.seen.append(Path(args[2]).read_bytes())
        return f'ELF file: sm_{75 if len(self.seen) == 1 else "120a"}.cubin'

    def inspect(self):
        with patch.object(PACKAGE, 'output', self.output):
            return PACKAGE.device_images(self.binary, Path('cuobjdump'))

    def test_every_module_is_inspected_separately_with_exact_payload(self):
        images = self.inspect()
        self.assertEqual(self.seen, self.modules)
        self.assertIn('sm_75.cubin', images)
        self.assertIn('sm_120a.cubin', images)

    def test_missing_or_duplicate_section_is_rejected(self):
        for listing in ('', self.listing * 2):
            with self.subTest(listing=listing):
                self.listing = listing
                with self.assertRaisesRegex(RuntimeError, 'Expected one'):
                    self.inspect()

    def test_section_outside_file_is_rejected(self):
        self.listing = section_listing(6, self.binary.stat().st_size)
        with self.assertRaisesRegex(RuntimeError, 'section bounds'):
            self.inspect()

    def test_corrupt_headers_payload_and_padding_are_rejected(self):
        cases = [(b'bad', 'Truncated.*header'),
                 (bytes(16), 'Invalid.*header'),
                 (struct.pack('<IHHQ', 0xBA55ED50, 1, 16, 100), 'Truncated.*payload'),
                 (fatbin(b'a') + b'x' * 7, 'padding'),
                 (self.section + b'trailing', 'Truncated.*header')]
        for data, error in cases:
            with self.subTest(error=error):
                self.binary.write_bytes(data)
                self.listing = section_listing(0, len(data))
                with self.assertRaisesRegex(RuntimeError, error):
                    self.inspect()

    def test_cuobjdump_failure_is_not_ignored(self):
        with patch.object(PACKAGE, 'output', side_effect=[self.listing,
                          subprocess.CalledProcessError(1, 'cuobjdump')]):
            with self.assertRaises(subprocess.CalledProcessError):
                PACKAGE.device_images(self.binary, Path('cuobjdump'))

    @unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('cc') and shutil.which('readelf'),
                         'Requires Linux binutils and a C compiler')
    def test_real_elf_section_location(self):
        source = self.binary.with_suffix('.c')
        data = ','.join(str(byte) for byte in self.section)
        source.write_text(f'const unsigned char modules[] __attribute__((section(".nv_fatbin"), aligned(8))) = {{{data}}};')
        subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o', str(self.binary)], check=True)
        original = PACKAGE.output
        def output(*args):
            return original(*args) if args[0] == 'readelf' else self.output(*args)
        with patch.object(PACKAGE, 'output', output):
            PACKAGE.device_images(self.binary, Path('cuobjdump'))
        self.assertEqual(self.seen, self.modules)


class LinuxPackageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'repo'
        self.cuda = self.root / 'cuda'
        self.build = self.repo / 'build'
        for directory in ('scripts', 'docs', 'docker', 'build/portable/backends/cuda13',):
            (self.repo / directory).mkdir(parents=True)
        for directory in ('bin', 'lib64/stubs'):
            (self.cuda / directory).mkdir(parents=True)
        for name in ('bin/nvcc', 'lib64/stubs/libcuda.so', 'EULA.txt'):
            (self.cuda / name).write_text('fixture')
        for name in ('portable/ninfer', 'portable/ninfer-serve', 'portable/ninfer_bench',
                     'portable/ninfer-calibrate', 'portable/ninfer-perplexity', 'portable/ninfer-ngram-profile',
                     'portable/backends/cuda13/libninfer-runtime.so',
                     'portable/backends/cuda13/architectures.txt'):
            (self.build / name).write_text('binary fixture')
        (self.build / 'portable/backends/cuda13/architectures.txt').write_text('75 120a\n')
        self.module = fatbin(b'fixture device images')
        (self.build / 'portable/backends/cuda13/libninfer-runtime.so').write_bytes(self.module)
        (self.build / 'CMakeCache.txt').write_text(
            f'CMAKE_CUDA_COMPILER:FILEPATH={self.cuda}/bin/nvcc\n'
            'CMAKE_CUDA_ARCHITECTURES:STRING=75;120a\nNINFER_SHARED_RUNTIME:BOOL=ON\n')
        for name in ('LICENSE', 'RELEASE_NOTES_0.0.md', 'docs/release-archive-linux.md',
                     'scripts/run.sh', 'scripts/download-model.sh'):
            (self.repo / name).write_text('fixture')
        (self.repo / 'VERSION').write_text('0.0-undefined')
        shutil.copy2(ROOT / 'docker/entrypoint.sh', self.repo / 'docker/entrypoint.sh')
        shutil.copy2(ROOT / 'docker/release.Dockerfile', self.repo / 'docker/release.Dockerfile')
        self.images = 'ELF file: sm_75.cubin\nELF file: sm_120a.cubin'
        self.help_failure = False
        self.original_read = Path.read_text
        def read_text(path, *args, **kwargs):
            if str(path) == '/etc/os-release':
                return 'ID=debian\nVERSION_ID="13"\n'
            return self.original_read(path, *args, **kwargs)
        self.stack = [patch.object(PACKAGE, '__file__', str(self.repo / 'scripts/package-linux-release.py')),
                      patch.dict(os.environ, NINFER_BUILD_ROOT=str(self.build), GITHUB_ACTIONS='true'),
                      patch.object(Path, 'read_text', read_text),
                      patch.object(PACKAGE, 'output', self.output),
                      patch.object(PACKAGE, 'run', self.run_command),
                      patch.object(PACKAGE, 'runtime_packages', return_value={'libc6:amd64'})]
        for item in self.stack:
            item.start()
            self.addCleanup(item.stop)

    def output(self, *args, **kwargs):
        if args[0] == 'readelf':
            return section_listing(0, len(self.module))
        if args[0] == 'git':
            return 'a' * 40 if 'rev-parse' in args else ''
        if args[0] == 'ldd':
            return 'libc.so.6 => /usr/lib/libc.so.6 (0x1)'
        if args[0] == 'getconf':
            return 'glibc 2.41'
        if args[0].endswith('/nvcc'):
            return 'Cuda compilation tools, release 13.4, V13.4.92'
        if args[0].endswith('/cuobjdump'):
            return self.images
        if args[-1] in ('--help', '--version'):
            if self.help_failure:
                raise RuntimeError('fixture startup failure')
            return 'fixture help'
        raise AssertionError(args)

    def run_command(self, *args, **kwargs):
        if args[0] in ('strip', 'patchelf'):
            return
        if args[-1] in ('--help', '--version'):
            if self.help_failure:
                raise RuntimeError('fixture startup failure')
            self.assertTrue(Path(args[0]).is_file())
            return
        raise AssertionError(args)

    def test_archive_roundtrip_and_runtime_docker_context(self):
        PACKAGE.main()
        dist = self.repo / 'dist'
        volume = next(dist.glob('*.tar.gz.001'))
        self.assertNotIn('undefined', volume.name)
        self.assertIn('-aaaaaaaaaaaa.tar.gz.001', volume.name)
        manifest = next(dist.glob('SHA256SUMS-*.txt')).read_text()
        self.assertEqual(manifest, f'{hashlib.sha256(volume.read_bytes()).hexdigest()}  {volume.name}\n')
        with tarfile.open(volume, 'r:gz') as archive:
            names = archive.getnames()
            self.assertFalse(any('libcuda.so' in name for name in names))
            prefix = names[0]
            self.assertTrue(archive.getmember(prefix + '/ninfer').isfile())
            self.assertTrue(archive.getmember(prefix + '/backends/cuda13/libninfer-runtime.so').isfile())
            self.assertIn(prefix + '/Dockerfile', names)
            info = json.load(archive.extractfile(prefix + '/BUILD-INFO.json'))
            self.assertEqual(info['architectures'], ['75', '120a'])

    def test_missing_device_image_cannot_produce_manifest(self):
        self.images = 'ELF file: sm_75.cubin'
        with self.assertRaisesRegex(RuntimeError, 'lacks sm_120a'):
            PACKAGE.main()
        self.assertFalse(list((self.repo / 'dist').glob('SHA256SUMS-*')))

    def test_commit_archive_does_not_require_release_notes(self):
        (self.repo / 'RELEASE_NOTES_0.0.md').unlink()
        PACKAGE.main()
        volume = next((self.repo / 'dist').glob('*.tar.gz.001'))
        with tarfile.open(volume, 'r:gz') as archive:
            names = archive.getnames()
            self.assertFalse(any('RELEASE_NOTES_' in name for name in names))
            self.assertTrue(any(name.endswith('/BUILD-INFO.json') for name in names))

    def test_failed_startup_removes_old_manifest(self):
        PACKAGE.main()
        self.help_failure = True
        with self.assertRaisesRegex(RuntimeError, 'startup failure'):
            PACKAGE.main()
        self.assertFalse(list((self.repo / 'dist').glob('SHA256SUMS-*')))


class DependencyTest(unittest.TestCase):
    def test_unresolved_library_is_rejected(self):
        with patch.object(PACKAGE, 'output', return_value='libavcodec.so.61 => not found'):
            with self.assertRaisesRegex(RuntimeError, 'Unresolved libraries'):
                PACKAGE.runtime_packages(Path('/binary'), {}, Path('/stub'), Path('/cuda'), Path('/bundle'))

    def test_unowned_library_is_rejected(self):
        with patch.object(PACKAGE, 'output', return_value='libavcodec.so.61 => /unowned/libavcodec.so.61 (0x1)'), \
             patch.object(PACKAGE.subprocess, 'run') as run:
            run.return_value.returncode = 1
            with self.assertRaisesRegex(RuntimeError, 'No Debian package owns'):
                PACKAGE.runtime_packages(Path('/binary'), {}, Path('/stub'), Path('/cuda'), Path('/bundle'))


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('patchelf'),
                     'Requires the Linux ELF loader and patchelf')
class BackendIsolationTest(unittest.TestCase):
    def test_installed_toolkit_path_cannot_replace_packaged_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend, ambient = root / 'backend', root / 'toolkit'
            backend.mkdir()
            ambient.mkdir()
            for directory, value in ((backend, 7), (ambient, 9)):
                source = directory / 'dependency.c'
                source.write_text(f'int backend_value(void) {{ return {value}; }}\n')
                subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o',
                                str(directory / 'libninfer_fixture.so')], check=True)
            source = root / 'runtime.c'
            source.write_text('int backend_value(void); int entry(void) { return backend_value(); }\n')
            runtime = backend / 'libninfer-runtime.so'
            subprocess.run(['cc', '-shared', '-fPIC', str(source), '-L' + str(backend),
                            '-lninfer_fixture', '-o', str(runtime)], check=True)
            PACKAGE.set_backend_rpath(runtime)
            source = root / 'launcher.c'
            source.write_text('#include <dlfcn.h>\nint main(int argc, char** argv) {\n'
                              'void* module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);\n'
                              'if (!module) return 100;\n'
                              'int (*entry)(void) = (int (*)(void))dlsym(module, "entry");\n'
                              'return entry ? entry() : 101; }\n')
            launcher = root / 'launcher'
            subprocess.run(['cc', str(source), '-ldl', '-o', str(launcher)], check=True)
            result = subprocess.run([str(launcher), str(runtime)],
                                    env=dict(os.environ, LD_LIBRARY_PATH=str(ambient)), timeout=10)
            self.assertEqual(result.returncode, 7)


if __name__ == '__main__':
    unittest.main()
