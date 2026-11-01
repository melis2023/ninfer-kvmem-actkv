#!/usr/bin/env python3.11
"""Select build and check scopes from changed repository paths."""
import sys

SCOPES = ('changed', 'native', 'calculator', 'host', 'includes', 'linux_scripts', 'linux_runtime', 'python', 'docker', 'release_tools')
COMPILER_INPUTS = {
    'CMakeLists.txt', 'CMakePresets.json', '.gitmodules', 'VERSION',
    '.github/scripts/configure-linux-build.sh', '.github/scripts/patch-cuda12-glibc.py',
    '.github/scripts/install-linux-release-tools.sh',
    '.github/workflows/native-linux.yml', '.github/workflows/native-windows.yml',
    'scripts/build-native.bat', 'scripts/build.ps1',
}
RUNTIME_INPUTS = {
    'scripts/run.sh', 'scripts/run.bat', 'scripts/download-model.sh',
    'scripts/download-model.bat', 'scripts/package-release.ps1',
    'scripts/package-linux-release.py', 'scripts/assemble-universal-release.py',
    'docker/release.Dockerfile', 'docker/entrypoint.sh',
}


def classify(paths):
    result = dict.fromkeys(SCOPES, False)
    for path in paths:
        if path.endswith('.md'):
            continue
        product = (path.startswith(('src/', 'include/', 'apps/', 'cmake/', 'third_party/'))
                   or path in COMPILER_INPUTS
                   or (path.startswith(('bench/', 'tools/')) and
                       (path.endswith(('.cpp', '.cc', '.c', '.cu', '.cuh', '.h', '.hpp'))
                        or path.endswith('/CMakeLists.txt'))))
        test_source = path.startswith('tests/') and (path.endswith(('.cpp', '.cc', '.cu', '.cuh', '.h', '.hpp'))
                                                     or path.endswith('/CMakeLists.txt'))
        result['native'] |= product or path in RUNTIME_INPUTS
        result['changed'] |= product or test_source
        result['host'] |= product or test_source or path == 'scripts/check-host-compile.py'
        result['includes'] |= path.startswith(('src/', 'include/')) or path == 'scripts/check-include-boundaries.py'
        result['calculator'] |= path.startswith('docs/config-calculator')
        result['linux_scripts'] |= path.endswith('.sh') or path == '.gitattributes'
        result['linux_runtime'] |= path in ('scripts/run.sh', 'scripts/download-model.sh', 'scripts/check-linux-scripts.sh', '.gitattributes')
        result['python'] |= (path.startswith(('tools/', 'tests/artifact/', 'tests/convert/', 'tests/reference/', 'tests/pods/', 'scripts/pods/'))
                             or path.startswith(('src/artifact/', 'include/ninfer/artifact')))
        result['docker'] |= path in ('Dockerfile', '.dockerignore', 'tests/test_docker_build_cache.sh', '.github/scripts/toolchain-image.sh')
        result['release_tools'] |= (path.startswith(('.github/scripts/', 'tests/release/'))
                                   or path in RUNTIME_INPUTS or path.startswith('.github/workflows/'))
    return result


if __name__ == '__main__':
    paths = sys.stdin.buffer.read().decode().split('\0')
    result = dict.fromkeys(SCOPES, True) if '--force' in sys.argv else classify(filter(None, paths))
    for key, value in result.items():
        print(f'{key}={str(value).lower()}')
