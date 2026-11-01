"""Extract the licence from NVIDIA's versioned, checksum-verified CUDA runtime archive."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'


def fetch(version, output):
    if not re.fullmatch(r'\d+\.\d+\.\d+', version):
        raise ValueError('An exact CUDA Toolkit version is required')
    with urllib.request.urlopen(BASE + f'redistrib_{version}.json', timeout=30) as response:
        entry = json.load(response)['cuda_cudart']['windows-x86_64']
    with urllib.request.urlopen(BASE + entry['relative_path'], timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('CUDA runtime archive checksum mismatch')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = [name for name in archive.namelist() if name.endswith('/LICENSE')]
        if len(names) != 1:
            raise ValueError('Expected one CUDA runtime licence')
        licence = archive.read(names[0])
    if not licence.strip():
        raise ValueError('The CUDA runtime licence is empty')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(licence)
    print(f'CUDA {version} licence: {len(licence)} bytes from verified runtime archive', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fetch(args.version, args.output)
