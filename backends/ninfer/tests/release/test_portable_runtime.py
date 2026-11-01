"""Exercise the real launcher and private runtime boundary without a CUDA Toolkit."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class PortableRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        (cls.root / 'app.cpp').write_text('''
#include <iostream>
#include <string_view>
#include "product/module_identity.h"
int main(int argc, char** argv) {
  if (argc > 1 && std::string_view(argv[1]) == "--help") {
    std::cout << "fixture command help\\n";
    return 0;
  }
  if (argc > 1 && std::string_view(argv[1]) == "--identity") {
    std::cout << ninfer::product::module_identity(reinterpret_cast<const void*>(&main), "fixture");
    return 0;
  }
  for (int i = 1; i < argc; ++i) std::cout << argv[i] << '\\n';
  return 37;
}
''')
        (cls.root / 'policy.cpp').write_text('''
#include "selection.h"
#include <cassert>
using namespace ninfer::launcher;
int main() {
  assert(select(0, 13040, {70}, true, true) == 12);
  assert(select(0, 13000, {75, 86, 89}, true, true) == 13);
  assert(select(0, 12090, {86}, true, true) == 12);
  assert(select(0, 13040, {120}, true, true) == 13);
  assert(select(12, 13040, {89}, true, true) == 12);
  assert(select(0, 13040, {89}, true, false) == 12);
  int failures = 0;
  for (auto devices : {std::vector<int>{70,120}, std::vector<int>{90}, std::vector<int>{}}) {
    try { select(0, 13040, devices, true, true); } catch (...) { ++failures; }
  }
  try { select(13, 12090, {86}, true, true); } catch (...) { ++failures; }
  try { select(12, 13040, {120}, true, true); } catch (...) { ++failures; }
  try { select(0, 11080, {86}, true, true); } catch (...) { ++failures; }
  try { select(0, 13040, {86}, false, false); } catch (...) { ++failures; }
  assert(failures == 7);
}
''')
        (cls.root / 'driver.cpp').write_text('''
#include <cstdlib>
extern "C" {
int cuInit(unsigned) { return 0; }
int cuDriverGetVersion(int* out) { const char* value = std::getenv("FIXTURE_DRIVER"); *out = value ? std::atoi(value) : 13040; return 0; }
int cuDeviceGetCount(int* out) { *out = 2; return 0; }
int cuDeviceGet(int* out, int index) { *out = index; return 0; }
int cuDeviceGetAttribute(int* out, int attr, int device) {
  const char* value = std::getenv(device ? "FIXTURE_SM1" : "FIXTURE_SM0");
  const int sm = value ? std::atoi(value) : 86;
  *out = attr == 75 ? sm / 10 : sm % 10; return 0;
}
}
''')
        (cls.root / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.28)
project(portable_fixture LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PROJECT_SOURCE_DIR "{ROOT.as_posix()}")
set(CMAKE_CUDA_COMPILER_VERSION 13.4)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;120a")
function(ninfer_stamp_build_id target)
  file(MAKE_DIRECTORY "${{CMAKE_BINARY_DIR}}/generated")
  file(WRITE "${{CMAKE_BINARY_DIR}}/generated/ninfer_build_id.h" "#define NINFER_BUILD_ID \\\"fixture\\\"\\n#define NINFER_VERSION \\\"test\\\"\\n")
  target_include_directories(${{target}} PRIVATE "${{CMAKE_BINARY_DIR}}/generated")
endfunction()
include("{ROOT.as_posix()}/cmake/NinferTargets.cmake")
include("{ROOT.as_posix()}/cmake/NinferRuntime.cmake")
ninfer_runtime_program(fixture fixture_app SOURCES "${{CMAKE_CURRENT_SOURCE_DIR}}/app.cpp")
target_include_directories(fixture_app PRIVATE "{ROOT.as_posix()}/src")
target_include_directories(fixture PRIVATE "{ROOT.as_posix()}/src")
target_compile_definitions(fixture PRIVATE NINFER_ROUTER_LAUNCHER=1)
target_sources(fixture PRIVATE "{ROOT.as_posix()}/src/serve/model_catalog.cpp")
ninfer_finalize_runtime()
add_executable(policy policy.cpp)
target_include_directories(policy PRIVATE "{ROOT.as_posix()}/apps/launcher")
if(MSVC)
  target_compile_options(policy PRIVATE /UNDEBUG)
else()
  target_compile_options(policy PRIVATE -UNDEBUG)
endif()
add_library(driver SHARED driver.cpp)
set_target_properties(driver PROPERTIES PREFIX "" SUFFIX "" OUTPUT_NAME "libcuda.so.1")
''')
        cls.build = cls.root / 'build'
        command = ['cmake', '-S', str(cls.root), '-B', str(cls.build)]
        if os.name == 'nt':
            command += ['-G', 'Visual Studio 17 2022', '-A', 'x64']
        else:
            command += ['-G', 'Ninja']
        cls.run_command(command)
        cls.run_command(['cmake', '--build', str(cls.build), '--config', 'Release', '-j'])
        cls.folder = cls.build / 'portable'
        shutil.copytree(cls.folder / 'backends/cuda13', cls.folder / 'backends/cuda12')
        (cls.folder / 'backends/cuda12/architectures.txt').write_text('70 75 80 86 89\n')
        cls.exe = cls.folder / ('fixture.exe' if os.name == 'nt' else 'fixture')
        cls.env = dict(os.environ)
        cls.env['DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'] = str(cls.build)

    @classmethod
    def run_command(cls, command):
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        return result

    def launch(self, *args, **env):
        return subprocess.run([str(self.exe), *args], env=dict(self.env, **env),
                              capture_output=True, text=True, timeout=10)

    def test_policy_matrix(self):
        path = self.build / ('Release/policy.exe' if os.name == 'nt' else 'policy')
        self.run_command([str(path)])

    def test_version_and_saved_help_need_no_backend(self):
        path = self.folder / 'backends'
        path.rename(self.folder / 'hidden')
        try:
            help_dir = self.folder / 'help'
            help_dir.mkdir(exist_ok=True)
            (help_dir / 'fixture.txt').write_text('saved command help\n')
            self.assertEqual(self.launch('--version').returncode, 0)
            result = self.launch('--help')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('saved command help', result.stdout)
            self.assertIn('--cuda-backend', result.stdout)
            (help_dir / 'fixture.txt').unlink()
        finally:
            (self.folder / 'hidden').rename(path)

    def test_runtime_help_crosses_private_entry_boundary(self):
        result = self.launch('--cuda-backend', '12', '--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('fixture command help', result.stdout)

    @unittest.skipIf(os.name == 'nt', 'Production Windows loads the driver only from System32')
    def test_real_driver_probe_dispatch_and_exit_code(self):
        for driver, sm, selected in [('13040', '86', 13), ('12090', '86', 12),
                                     ('13040', '70', 12), ('13040', '120', 13)]:
            result = self.launch('model.ninfer', '--device', '0', FIXTURE_DRIVER=driver, FIXTURE_SM0=sm)
            self.assertEqual(result.returncode, 37, result.stderr)
            self.assertIn(f'CUDA {selected} backend', result.stderr)
            self.assertIn('model.ninfer', result.stdout)

    @unittest.skipIf(os.name == 'nt', 'Production Windows loads the driver only from System32')
    def test_incompatible_mixed_gpus_and_forced_backend_fail_before_entry(self):
        result = self.launch('model', '--devices', '0,1', FIXTURE_SM0='70', FIXTURE_SM1='120')
        self.assertEqual(result.returncode, 1)
        self.assertIn('No packaged CUDA backend', result.stderr)
        self.assertEqual(result.stdout, '')
        result = self.launch('--cuda-backend=13', 'model', FIXTURE_DRIVER='12090')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, '')

    @unittest.skipIf(os.name == 'nt', 'Production Windows loads the driver only from System32')
    def test_override_is_removed_and_prompt_is_opaque(self):
        result = self.launch('--cuda-backend', '12', 'model', '--prompt', '--cuda-backend=13')
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertIn('CUDA 12 backend', result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['model', '--prompt', '--cuda-backend=13'])

    @unittest.skipIf(os.name == 'nt', 'Production Windows loads the driver only from System32')
    def test_router_presets_use_all_models_and_command_line_precedence(self):
        preset = self.root / 'models.ini'
        preset.write_text('[a]\nmodel = a.ninfer\ndevice = 0\n[b]\nmodel = b.ninfer\ndevice = 1\n')
        result = self.launch('--models-preset', str(preset), FIXTURE_SM0='70', FIXTURE_SM1='120')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('No packaged CUDA backend', result.stderr)
        result = self.launch('--models-preset', str(preset), '--device', '0',
                             FIXTURE_SM0='70', FIXTURE_SM1='120')
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertIn('CUDA 12 backend', result.stderr)

    @unittest.skipIf(os.name == 'nt', 'Production Windows loads the driver only from System32')
    def test_persistent_identity_uses_runtime_module(self):
        result = self.launch('--identity')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('module=cuda13/libninfer-runtime.', result.stdout)
        runtime = next((self.folder / 'backends/cuda13').glob('libninfer-runtime.*'))
        self.assertIn(f'size={runtime.stat().st_size};', result.stdout)

    @unittest.skipIf(os.name == 'nt', 'Production Windows loads the driver only from System32')
    def test_actual_backend_architectures_limit_selection(self):
        path = self.folder / 'backends/cuda13/architectures.txt'
        previous = path.read_text()
        path.write_text('86\n')
        try:
            result = self.launch('model', FIXTURE_SM0='75')
            self.assertEqual(result.returncode, 37, result.stderr)
            self.assertIn('CUDA 12 backend', result.stderr)
            self.assertEqual(self.launch('--cuda-backend=13', 'model', FIXTURE_SM0='75').returncode, 1)
        finally:
            path.write_text(previous)

    def test_invalid_launcher_options(self):
        for args in [('--cuda-backend',), ('--cuda-backend=14',), ('--devices', '0,'),
                     ('--device', '-1'), ('--device', '0', '--devices', '1')]:
            self.assertEqual(self.launch(*args).returncode, 1)


if __name__ == '__main__':
    unittest.main()
