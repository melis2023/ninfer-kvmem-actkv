#!/usr/bin/env python3.11
"""Package the configured Linux products and verify runtime dependencies."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile


def output(*args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def run(*args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def set_backend_rpath(binary):
    # A user's Toolkit in LD_LIBRARY_PATH must not replace this backend's libraries.
    run('patchelf', '--force-rpath', '--set-rpath', '$ORIGIN', str(binary))


def device_images(binary, cuobjdump):
    # The linker concatenates CUDA modules in .nv_fatbin. Inspect one module at a
    # time to limit cuobjdump's input on runners with less memory than build hosts.
    sections = output('readelf', '--wide', '--section-headers', str(binary))
    matches = re.findall(r'^\s*\[\s*\d+\]\s+\.nv_fatbin\s+PROGBITS\s+[0-9a-fA-F]+\s+'
                         r'([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s', sections, re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f'Expected one .nv_fatbin section in {binary}')
    offset, size = (int(value, 16) for value in matches[0])
    if size == 0 or offset + size > binary.stat().st_size:
        raise RuntimeError(f'Invalid .nv_fatbin section bounds in {binary}')
    images = []
    count = largest = 0
    with binary.open('rb') as source, tempfile.TemporaryDirectory(prefix='ninfer-fatbin-') as temporary:
        source.seek(offset)
        end = offset + size
        module = Path(temporary) / 'module.fatbin'
        while source.tell() < end:
            position = source.tell()
            if end - position < 16:
                raise RuntimeError(f'Truncated CUDA fatbinary header at {position} in {binary}')
            # NVIDIA fatbinary v1 framing, also emitted by Clang's CUDA driver.
            magic, version, header_size, payload_size = struct.unpack('<IHHQ', source.read(16))
            if magic != 0xBA55ED50 or version != 1 or header_size != 16 or payload_size == 0:
                raise RuntimeError(f'Invalid CUDA fatbinary header at {position} in {binary}')
            length = header_size + payload_size
            if position + length > end:
                raise RuntimeError(f'Truncated CUDA fatbinary payload at {position} in {binary}')
            source.seek(position)
            with module.open('wb') as target:
                remaining = length
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError(f'Truncated CUDA fatbinary in {binary}')
                    target.write(chunk)
                    remaining -= len(chunk)
            images.append(output(str(cuobjdump), '--list-elf', str(module)))
            count += 1
            largest = max(largest, length)
            if count % 50 == 0:
                print(f'Checked {count} CUDA modules ({source.tell() - offset}/{size} bytes)', flush=True)
            # Input sections have 8-byte alignment; padding is not part of a module.
            padding = min((-length) % 8, end - source.tell())
            if source.read(padding) != bytes(padding):
                raise RuntimeError(f'Invalid CUDA fatbinary padding in {binary}')
    print(f'Checked {count} CUDA modules; largest {largest} bytes; total {size} bytes', flush=True)
    return '\n'.join(images)


def runtime_packages(binary, env, driver_stub, cuda_root, bundle):
    dependencies = output('ldd', str(binary), env=env)
    if 'not found' in dependencies:
        raise RuntimeError(f'Unresolved libraries for {binary}:\n{dependencies}')
    packages = set()
    for library in re.findall(r'=> (/\S+)', dependencies):
        path = Path(library)
        if path.resolve() == driver_stub.resolve():
            continue  # The host supplies the real driver; never package the stub.
        if path.resolve().is_relative_to(cuda_root):
            # Keep the SONAME from ldd; copy the real file, not a link into the build Toolkit.
            destination = bundle / path.name
            if not destination.exists():
                shutil.copy2(path, destination)
            continue
        for candidate in dict.fromkeys((str(path.resolve()), library)):
            result = subprocess.run(['dpkg-query', '-S', candidate], text=True, capture_output=True)
            if result.returncode == 0:
                packages.update(line.split(': ', 1)[0] for line in result.stdout.splitlines())
                break
        else:
            raise RuntimeError(f'No Debian package owns runtime library {library}')
    if not packages:
        raise RuntimeError(f'No runtime dependencies found for {binary}')
    return packages


def main():
    repo = Path(__file__).resolve().parents[1]
    build = Path(os.environ.get('NINFER_BUILD_ROOT', repo / 'build-linux')).resolve()
    cache = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        match = re.match(r'([^:#/][^:]*):[^=]+=(.*)', line)
        if match:
            cache[match[1]] = match[2]
    arches = cache['CMAKE_CUDA_ARCHITECTURES'].split(';')
    if not arches or any(a not in ('70', '75', '80', '86', '89', '120a') for a in arches):
        raise RuntimeError(f'Unsupported architectures: {arches}')
    nvcc = Path(cache['CMAKE_CUDA_COMPILER']).resolve()
    cuda_root = nvcc.parent.parent
    compiler_version = output(str(nvcc), '--version')
    cuda = re.search(r'release (\d+\.\d+),', compiler_version)[1]
    revision = output('git', '-C', str(repo), 'rev-parse', 'HEAD')
    modified = bool(output('git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'))
    if os.environ.get('GITHUB_ACTIONS') == 'true' and modified:
        raise RuntimeError('CI source tree is modified')
    name = f"ninfer-linux-x64-cuda{cuda}-sm{'-'.join(arches)}-{revision[:12]}"
    if modified:
        name += '-dirty'
    if cache.get('NINFER_SHARED_RUNTIME') != 'ON':
        raise RuntimeError('Native releases require NINFER_SHARED_RUNTIME=ON')
    commands = ['ninfer', 'ninfer-serve', 'ninfer_bench', 'ninfer-calibrate',
                'ninfer-perplexity', 'ninfer-ngram-profile']
    products = {p: build / 'portable' / p for p in commands}
    backend_relative = Path('backends') / f"cuda{cuda.split('.')[0]}"
    runtime = backend_relative / 'libninfer-runtime.so'
    products[str(runtime)] = build / 'portable' / runtime
    config = backend_relative / 'architectures.txt'
    products[str(config)] = build / 'portable' / config
    for source in products.values():
        if not source.is_file():
            raise RuntimeError(f'Missing release product: {source}')
    if products[str(config)].read_text().split() != arches:
        raise RuntimeError('Backend architecture metadata differs from the build')
    version = (repo / 'VERSION').read_text().strip()
    notes = repo / f"RELEASE_NOTES_{version.split('-')[0]}.md"
    dist = repo / 'dist'
    dist.mkdir(exist_ok=True)
    package = dist / name
    manifest = dist / f'SHA256SUMS-{name}.txt'
    manifest.unlink(missing_ok=True)
    for old in dist.glob(f'{name}.tar.gz.[0-9]*'):
        old.unlink()
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    for product, source in products.items():
        target = package / product
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.suffix != '.txt':
            run('strip', '--strip-unneeded', str(target))
    packages = set()
    libraries = package / backend_relative
    eula = cuda_root / 'EULA.txt'
    if not eula.is_file():
        raise RuntimeError(f'Missing CUDA redistribution licence: {eula}')
    shutil.copy2(eula, libraries / 'NVIDIA-CUDA-EULA.txt')
    # Hosted runners have no GPU. The temporary stub only supports loader/--help checks.
    stub = cuda_root / 'lib64/stubs/libcuda.so'
    if not stub.is_file():
        raise RuntimeError(f'Missing CUDA driver stub: {stub}')
    with tempfile.TemporaryDirectory(prefix='ninfer-driver-check-') as temporary:
        Path(temporary, 'libcuda.so.1').symlink_to(stub)
        env = dict(os.environ, LD_LIBRARY_PATH=f"{temporary}:{cuda_root / 'lib64'}")
        binary = package / runtime
        set_backend_rpath(binary)
        images = device_images(binary, cuda_root / 'bin/cuobjdump')
        for arch in arches:
            if not re.search(rf'sm_{arch}(?:[.\s_]|$)', images):
                raise RuntimeError(f'{binary} lacks sm_{arch}')
        packages.update(runtime_packages(binary, env, stub, cuda_root, libraries))
        for library in libraries.glob('*.so*'):
            set_backend_rpath(library)
        # Check runtime closure without the installed Toolkit.
        env['LD_LIBRARY_PATH'] = str(temporary)
        dependencies = output('ldd', str(binary), env=env)
        if 'not found' in dependencies or str(cuda_root) in dependencies:
            raise RuntimeError(f'Incomplete packaged runtime: {dependencies}')
        (package / 'help').mkdir()
        for command in commands:
            text = output(str(package / command), '--help', env=env, timeout=60)
            (package / 'help' / f'{command}.txt').write_text(text + '\n')
    # Launchers and saved help must work with no driver stub or Toolkit.
    clean_env = dict(os.environ)
    clean_env.pop('LD_LIBRARY_PATH', None)
    for command in commands:
        dependencies = output('ldd', str(package / command), env=clean_env)
        if re.search(r'lib(?:cuda|cublas|ninfer)', dependencies) or 'not found' in dependencies:
            raise RuntimeError(f'Launcher imports a CUDA backend: {dependencies}')
        packages.update(runtime_packages(package / command, clean_env, stub, cuda_root, libraries))
        run(str(package / command), '--help', env=clean_env, timeout=60)
        run(str(package / command), '--version', env=clean_env, timeout=60)
    (package / 'packages').mkdir()
    (package / 'packages/multi.txt').write_text(''.join(f'{p}\n' for p in sorted(packages)))
    for source, destination in (
        (repo / 'LICENSE', 'LICENSE'), (repo / 'VERSION', 'VERSION'),
        (repo / 'docs/release-archive-linux.md', 'README.md'),
        (repo / 'scripts/run.sh', 'run.sh'), (repo / 'scripts/download-model.sh', 'download-model.sh'),
    ):
        shutil.copy2(source, package / destination)
    if notes.is_file():
        shutil.copy2(notes, package / notes.name)
    distro = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines()
                  if '=' in line)
    if distro.get('ID', '').strip('"') == 'debian' and distro.get('VERSION_ID', '').strip('"') == '13':
        shutil.copy2(repo / 'docker/entrypoint.sh', package / 'entrypoint.sh')
        dockerfile = (repo / 'docker/release.Dockerfile').read_text()
        (package / 'Dockerfile').write_text(dockerfile.replace('cuda>=13.0', f"cuda>={cuda.split('.')[0]}.0"))
    patch_note = cuda_root / 'ninfer-toolchain-patches.txt'
    patches = patch_note.read_text() if patch_note.exists() else ''
    info = dict(toolchain_patches=patches, source_revision=revision, modified_sources=modified, version=version,
                cuda=cuda, cuda_compiler=compiler_version, architectures=arches,
                layout='shared-runtime-v1', commands=commands, platform='linux-x64', os_release=distro, glibc=output('getconf', 'GNU_LIBC_VERSION'),
                runtime_packages=sorted(packages),
                validation='Device images and packaged CUDA dependency closure checked; all commands passed --help '
                'without a driver. Runtime help captured with a temporary driver stub. GPU inference was not tested.')
    (package / 'BUILD-INFO.json').write_text(json.dumps(info, indent=2) + '\n')
    files = sorted(p for p in package.rglob('*') if p.is_file())
    hashes = {p.resolve(): digest(p) for p in files if not p.is_symlink()}
    (package / 'SHA256SUMS.txt').write_text(
        ''.join(f'{hashes[p.resolve()]}  {p.relative_to(package)}\n' for p in files))
    # Split while writing; do not keep a second full archive on disk.
    archive = subprocess.Popen(['tar', '-C', str(dist), '-czf', '-', name], stdout=subprocess.PIPE)
    volumes = []
    try:
        index = 1
        chunk = archive.stdout.read(1024 * 1024)
        while chunk:
            volume = dist / f'{name}.tar.gz.{index:03d}'
            with volume.open('wb') as stream:
                size = 0
                while chunk and size < 1900 * 1024**2:
                    stream.write(chunk)
                    size += len(chunk)
                    chunk = archive.stdout.read(1024 * 1024)
            volumes.append(volume)
            index += 1
        if archive.wait() != 0:
            raise RuntimeError('tar archive creation failed')
    finally:
        archive.stdout.close()
        if archive.poll() is None:
            archive.terminate()
            archive.wait()
    check = subprocess.Popen(['tar', '-tzf', '-'], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    try:
        for volume in volumes:
            with volume.open('rb') as stream:
                shutil.copyfileobj(stream, check.stdin)
        check.stdin.close()
        if check.wait() != 0:
            raise RuntimeError('tar archive verification failed')
    finally:
        if check.poll() is None:
            check.terminate()
            check.wait()
    manifest.write_text(''.join(f'{digest(p)}  {p.name}\n' for p in volumes))
    print(f'Verified {len(volumes)} archive volumes: {name}', flush=True)


if __name__ == '__main__':
    main()
