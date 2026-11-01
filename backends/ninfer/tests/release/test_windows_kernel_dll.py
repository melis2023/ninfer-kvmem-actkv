"""Exercise automatic function exports and runtime DLL staging on a real Windows linker."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class KernelCompileCommandTest(unittest.TestCase):
    def test_static_and_dll_object_compile_commands_match(self):
        # Generate MSVC commands on any host. The forced compiler is never executed.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'CMakeLists.txt').write_text('''
cmake_minimum_required(VERSION 3.28)
project(kernel_commands LANGUAGES CXX)
add_library(kernels ${KIND} kernel.cpp)
if(KIND STREQUAL "OBJECT")
  set_target_properties(kernels PROPERTIES COMPILE_PDB_NAME kernels
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/kernels.dir")
endif()
''')
            (root / 'kernel.cpp').write_text('int value() { return 42; }\n')
            commands = []
            for kind in ('STATIC', 'OBJECT'):
                build = root / kind
                result = subprocess.run([
                    'cmake', '-S', str(root), '-B', str(build), '-G', 'Ninja',
                    '-DCMAKE_SYSTEM_NAME=Windows', f'-DCMAKE_CXX_COMPILER={Path(sys.executable).as_posix()}',
                    '-DCMAKE_CXX_COMPILER_FORCED=ON', '-DCMAKE_CXX_COMPILER_ID_RUN=TRUE',
                    '-DCMAKE_CXX_COMPILER_ID=MSVC', '-DCMAKE_CXX_COMPILER_VERSION=19.44',
                    '-DCMAKE_CXX_COMPILER_ARCHITECTURE_ID=x64', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', f'-DKIND={kind}',
                ], text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                commands.append(json.loads((build / 'compile_commands.json').read_text())[0]['command'])
            self.assertEqual(commands[0], commands[1])


@unittest.skipUnless(os.name == 'nt', 'Requires the Windows loader and MSVC')
class WindowsKernelDllTest(unittest.TestCase):
    def test_exported_bridge_runs_only_with_staged_dll(self):
        module = Path(__file__).resolve().parents[2] / 'cmake/NinferTargets.cmake'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.28)
project(kernel_dll_fixture LANGUAGES CXX)
set(NINFER_WINDOWS_KERNEL_DLL ON)
include("{module.as_posix()}")
add_library(ninfer_ggml_quants OBJECT kernel.cpp)
add_library(ninfer_device_tier OBJECT policy.cpp)
add_library(ninfer_ggml_quants_dll SHARED
  $<TARGET_OBJECTS:ninfer_ggml_quants> $<TARGET_OBJECTS:ninfer_device_tier>)
set_target_properties(ninfer_ggml_quants_dll PROPERTIES
  OUTPUT_NAME ninfer_ggml_quants WINDOWS_EXPORT_ALL_SYMBOLS ON
  RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/kernels")
add_executable(consumer main.cpp)
set_target_properties(consumer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/apps")
target_link_libraries(consumer PRIVATE ninfer_ggml_quants_dll)
ninfer_stage_runtime_dlls(consumer)
''')
            (root / 'kernel.cpp').write_text(
                '#ifdef _WINDLL\n#error Object compilation must not depend on DLL linkage\n#endif\n'
                'namespace policy { int get(); }\n'
                'namespace bridge { int value() { return policy::get(); } }\n')
            (root / 'policy.cpp').write_text(
                'namespace { int cached = 0; }\n'
                'namespace policy { void set(int value) { cached = value; } int get() { return cached; } }\n')
            (root / 'main.cpp').write_text(
                'namespace bridge { int value(); } namespace policy { void set(int); }\n'
                'int main() { policy::set(42); return bridge::value() != 42; }\n')
            build = root / 'build'
            for command in (['cmake', '-S', str(root), '-B', str(build), '-G', 'Visual Studio 17 2022', '-A', 'x64'],
                            ['cmake', '--build', str(build), '--config', 'Release', '-j']):
                result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            folder = build / 'apps/Release'
            exe = folder / 'consumer.exe'
            dll = folder / 'ninfer_ggml_quants.dll'
            self.assertTrue(dll.is_file())
            env = dict(os.environ, PATH=os.environ['SystemRoot'] + r'\System32')
            self.assertEqual(subprocess.run([str(exe)], cwd=root, env=env, timeout=10).returncode, 0)
            dll.rename(dll.with_suffix('.hidden'))
            self.assertNotEqual(subprocess.run([str(exe)], cwd=root, env=env, timeout=10).returncode, 0)


if __name__ == '__main__':
    unittest.main()
