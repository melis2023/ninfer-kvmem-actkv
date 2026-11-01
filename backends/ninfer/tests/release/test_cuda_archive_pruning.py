"""Check target preservation and failed-tool handling without requiring a CUDA Toolkit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'cmake/PruneCudaPtx.cmake'


@unittest.skipIf(os.name == 'nt', 'The fake nvprune executable uses a POSIX shebang')
class CudaArchivePruningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'cuda objects.lib'
        self.archive.write_bytes(b'original archive')
        self.tool = self.root / 'nvprune'
        self.tool.write_text(f'#!{sys.executable}\n' + '''
import json, os, pathlib, shutil, sys
args = sys.argv[1:]
pathlib.Path(os.environ['PRUNE_ARGUMENTS']).write_text(json.dumps(args))
output = pathlib.Path(args[args.index('-o') + 1])
mode = os.environ.get('PRUNE_MODE', '')
if mode == 'copy':
    shutil.copyfile(args[-1], output)
    sys.exit(0)
if mode == 'fail':
    output.write_bytes(b'incomplete')
    sys.exit(1)
if mode != 'missing':
    output.write_bytes(b'' if mode == 'empty' else b'pruned archive')
''')
        self.tool.chmod(0o755)

    def prune(self, archs='75;80;86;89;120a', mode=''):
        return subprocess.run(['cmake', f'-DNVPRUNE={self.tool}',
                               f'-DARCHIVE={self.archive}', f'-DARCHITECTURES={archs}',
                               '-P', str(SCRIPT)], capture_output=True, text=True,
                              env=dict(os.environ, PRUNE_ARGUMENTS=str(self.root / 'args.json'),
                                       PRUNE_MODE=mode))

    def test_preserves_every_requested_sass_target(self):
        for archs in ('70;75;80;86;89', '75;80;86;89;120a', '120a'):
            with self.subTest(archs=archs):
                result = self.prune(archs)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = json.loads((self.root / 'args.json').read_text())
                self.assertEqual(args[:-3], [f'--generate-code=arch=compute_{a},code=sm_{a}'
                                             for a in archs.split(';')])
                self.assertEqual(self.archive.read_bytes(), b'pruned archive')

    def test_failed_or_incomplete_output_never_replaces_original(self):
        for mode in ('fail', 'missing', 'empty'):
            with self.subTest(mode=mode):
                result = self.prune(mode=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.archive.read_bytes(), b'original archive')
                self.assertFalse(Path(str(self.archive) + '.pruned').exists())

    def test_invalid_architecture_fails_before_tool_invocation(self):
        result = self.prune('75;invalid')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'args.json').exists())
        self.assertEqual(self.archive.read_bytes(), b'original archive')

    def test_cmake_hook_preserves_linkable_host_code(self):
        module = SCRIPT.parent / 'NinferTargets.cmake'
        (self.root / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.28)
project(pruning_fixture LANGUAGES CXX)
set(NINFER_CUDA_PRUNE_PTX ON)
set(NINFER_NVPRUNE "{self.tool}")
include("{module}")
add_library(kernels STATIC kernels.cpp)
ninfer_cuda_archive(kernels)
set_target_properties(kernels PROPERTIES CUDA_ARCHITECTURES "75;120a")
add_executable(check_host main.cpp)
target_link_libraries(check_host PRIVATE kernels)
''')
        (self.root / 'kernels.cpp').write_text('int answer() { return 42; }\n')
        (self.root / 'main.cpp').write_text('int answer(); int main() { return answer() != 42; }\n')
        env = dict(os.environ, PRUNE_ARGUMENTS=str(self.root / 'args.json'), PRUNE_MODE='copy')
        build = self.root / 'build'
        for command in (['cmake', '-S', str(self.root), '-B', str(build)],
                        ['cmake', '--build', str(build), '-j'], [str(build / 'check_host')]):
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = json.loads((self.root / 'args.json').read_text())
        self.assertEqual(args[:-3], ['--generate-code=arch=compute_75,code=sm_75',
                                    '--generate-code=arch=compute_120a,code=sm_120a'])

    def test_shared_library_is_never_sent_to_archive_pruner(self):
        module = SCRIPT.parent / 'NinferTargets.cmake'
        (self.root / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.28)
project(shared_fixture LANGUAGES CXX)
set(NINFER_CUDA_PRUNE_PTX ON)
set(NINFER_NVPRUNE "{self.tool}")
include("{module}")
add_library(kernels SHARED kernels.cpp)
ninfer_cuda_non_rdc_archive(kernels)
add_executable(check_host main.cpp)
target_link_libraries(check_host PRIVATE kernels)
''')
        (self.root / 'kernels.cpp').write_text('int answer() { return 42; }\n')
        (self.root / 'main.cpp').write_text('int answer(); int main() { return answer() != 42; }\n')
        env = dict(os.environ, PRUNE_ARGUMENTS=str(self.root / 'args.json'), PRUNE_MODE='fail')
        build = self.root / 'build'
        for command in (['cmake', '-S', str(self.root), '-B', str(build)],
                        ['cmake', '--build', str(build), '-j'], [str(build / 'check_host')]):
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.root / 'args.json').exists())


if __name__ == '__main__':
    unittest.main()
