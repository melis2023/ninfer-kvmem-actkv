#!/usr/bin/env python3.11
"""Align CUDA 12.9 math exception declarations with Debian 13's glibc 2.41."""
from pathlib import Path
import re
import subprocess
import sys


def patch_headers(root):
    header = root / 'targets/x86_64-linux/include/crt/math_functions.h'
    inline = header.with_suffix('.hpp')
    declaration = re.compile(
        r'^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ (?:double|float)\s+'
        r'(?:sinpi|cospi|sinpif|cospif)\((?:double|float) x\));$', re.MULTILINE)
    helper = re.compile(
        r'(__func__\((?:double (?:sinpi|cospi)\(double a\)|'
        r'float (?:sinpif|cospif)\((?:const )?float a\)))\)')
    text, count = declaration.subn(r'\1 noexcept (true);', header.read_text())
    text, prototypes = helper.subn(r'\1 noexcept (true))', text)
    definitions, count_inline = helper.subn(r'\1 noexcept (true))', inline.read_text())
    if (count, prototypes, count_inline) != (4, 4, 4):
        raise RuntimeError(f'Unexpected CUDA 12.9 headers: {(count, prototypes, count_inline)}')
    # Only exception specifications change; function bodies and arithmetic stay as supplied.
    header.write_text(text)
    inline.write_text(definitions)
    (root / 'ninfer-toolchain-patches.txt').write_text(
        'CUDA 12.9: sinpi/cospi/sinpif/cospif noexcept(true) declarations for glibc 2.41.\n'
        'https://forums.developer.nvidia.com/t/323591\n')


if __name__ == '__main__':
    root = Path(sys.argv[1])
    if subprocess.check_output(['getconf', 'GNU_LIBC_VERSION'], text=True).strip() != 'glibc 2.41':
        raise SystemExit('This patch is qualified only for glibc 2.41')
    if 'release 12.9,' not in subprocess.check_output([str(root / 'bin/nvcc'), '--version'], text=True):
        raise SystemExit('This patch requires CUDA 12.9')
    patch_headers(root)
