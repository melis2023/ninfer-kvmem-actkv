#!/usr/bin/env python3
"""Compiles the host side of every translation unit of configured builds, without device code.

The compile database of a CMake build (CMAKE_EXPORT_COMPILE_COMMANDS=ON) lists every .cpp and .cu
the build compiles, with its architecture's definitions. A .cpp is compiled for syntax only
(`/Zs`, `-fsyntax-only`). A .cu goes through nvcc's own steps for host code, taken from
`nvcc --dryrun`: the host preprocessing, cudafe++ and the host compiler over cudafe++'s output
(for syntax only), while cicc, ptxas and fatbinary, which take most of a build's time, do not run.
On Windows this is MSVC and nvcc's MSVC front end, which no other check in CI runs. Device code is
not compiled. Where cicc writes the kernel registration stub (Windows), the host code includes one
without the kernels, and MSVC's C2129 on a static __global__ function, whose host wrapper that stub
defines, is not a failure.

Several build directories are checked together, and a translation unit whose command differs
between them only in the build directory and the architecture numbers (`sm_86`, `compute_86`) is
compiled once: its host code differs only in __CUDA_ARCH_LIST__, which no host code reads.

usage: check-host-compile.py BUILD_DIR... [--jobs N] [--shard I/N] [--filter REGEX]

NINFER_HOST_CHECK_TRACE=REGEX prints the dryrun of the .cu sources it matches.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WINDOWS = os.name == "nt"
# Where cicc writes the module id (CUDA 12), cudafe++ only reads an identifier from it.
MODULE_ID = "_ninfer_host_check_module_id"
DEVICE_TOOLS = ("cicc", "ptxas", "fatbinary", "nvlink", "cuobjdump", "nvdisasm")
ARCH_TOKEN = re.compile(r"\b(compute|sm|lto)_[0-9]+[a-z]?\b")
# The head of the stub cicc writes on Windows, without the kernels' launch stubs and registration
# and the fatbinary: the runtime definitions cudafe++'s output declares and MSVC requires (C2129 on
# __nv_init_managed_rt_with_module otherwise).
HOST_STUB = """#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
"""


def run(command, cwd, env=None):
    """Runs one command line as the build system would: a string for CreateProcess on Windows,
    a shell command elsewhere. Returns (exit status, combined output)."""
    try:
        return run_unchecked(command, cwd, env)
    except OSError as error:
        return 1, f"cannot start ({error}):\n{command[:2000]}"


def run_unchecked(command, cwd, env):
    if WINDOWS:
        # CreateProcess does not expand cmd's %NAME% references.
        if env:
            names = {k.upper(): v for k, v in env.items()}
            command = re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)%",
                             lambda m: names.get(m.group(1).upper(), m.group(0)), command)
        # nvcc's dryrun writes cmd syntax: a preprocessing step redirects its output with
        # `> "file"`, placed right after the program.
        redirect = re.search(r'\s>\s*("[^"]+"|\S+)', command)
        if redirect:
            target = redirect.group(1).strip('"')
            command = command[: redirect.start()] + command[redirect.end():]
            with open(Path(cwd) / target, "wb") as out:
                done = subprocess.run(command, cwd=cwd, env=env, stdout=out,
                                      stderr=subprocess.PIPE)
            return done.returncode, done.stderr.decode(errors="replace")
        done = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    else:
        done = subprocess.run(command, cwd=cwd, env=env, shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    return done.returncode, done.stdout.decode(errors="replace")


def entry_command(entry):
    if "command" in entry:
        return entry["command"]
    arguments = entry["arguments"]
    return subprocess.list2cmdline(arguments) if WINDOWS else shlex.join(arguments)


def host_compiler_is_msvc(command):
    first = re.match(r'\s*"?([^"\s]+)', command).group(1)
    return Path(first).name.lower() in ("cl", "cl.exe")


def syntax_only(command, msvc):
    return command + (" /Zs" if msvc else " -fsyntax-only")


def option_path(command, option):
    found = re.search(re.escape(option) + r'[= ]"?([^"\s]+)"?', command)
    return found.group(1) if found else None


def dryrun_steps(output):
    """The `#$` lines of nvcc --dryrun: environment assignments and commands, in order."""
    env, commands = {}, []
    for line in output.splitlines():
        if not line.startswith("#$ "):
            continue
        body = line[3:].strip()
        # cmd quotes a command whose program is quoted once more: `""%CICC_PATH%\cicc" args"`.
        if WINDOWS and body.startswith('""') and body.endswith('"'):
            body = body[1:-1]
        assignment = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", body)
        if assignment and not re.search(r"\s", assignment.group(1)):
            value = assignment.group(2)
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
            env[assignment.group(1)] = value
            continue
        commands.append(body)
    return env, commands


def tool_name(command):
    first = re.match(r'\s*"?([^"]+?)"?(\s|$)', command)
    return Path(first.group(1)).name.lower() if first else ""


def host_key(entry, build_dir):
    """What decides a translation unit's host code: its source and its command, less the build
    directory and the architecture numbers."""
    def relative(text):
        for spelling in {str(build_dir), build_dir.as_posix()}:
            text = re.sub(re.escape(spelling), "<build>", text,
                          flags=re.IGNORECASE if WINDOWS else 0)
        return text

    return relative(entry["file"]), ARCH_TOKEN.sub(r"\1_*", relative(entry_command(entry)))


def only_static_kernels(output):
    """Whether every error is MSVC's C2129 on a __global__ function: cudafe++ declares a static
    kernel's host wrapper, and the stub cicc writes defines it."""
    errors = [line for line in output.splitlines() if re.search(r"\berror\b", line)]
    for line in errors:
        found = re.match(r"\s*(.+?)\(([0-9]+)\): error C2129:", line)
        if not found:
            return False
        try:
            source = Path(found.group(1)).read_text(errors="replace").splitlines()
        except OSError:
            return False
        # The declaration's line, or `__global__` closing the line before it.
        number = int(found.group(2))
        here = source[number - 1] if 0 < number <= len(source) else ""
        before = source[number - 2] if 1 < number <= len(source) + 1 else ""
        if "__global__" not in here and not before.rstrip().endswith("__global__"):
            return False
    return bool(errors)


def load_entries(build_dirs):
    """The translation units of every build, each once per distinct host key."""
    entries, seen, total = [], set(), 0
    for build_dir in build_dirs:
        # The build id header is generated at build time, not at configure time.
        if not (build_dir / "generated" / "ninfer_build_id.h").exists():
            subprocess.run(["cmake", "--build", str(build_dir), "--target",
                            "ninfer_build_id_stamp"], check=True, stdout=subprocess.DEVNULL)
        for entry in json.loads((build_dir / "compile_commands.json").read_text()):
            if (not entry["file"].endswith((".cpp", ".cu"))
                    or "third_party" in Path(entry["file"]).parts):
                continue
            total += 1
            key = host_key(entry, build_dir)
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    return entries, total


class Checker:
    def __init__(self, work, fatbin_template):
        self.work = work
        self.fatbin_template = fatbin_template

    def check_cpp(self, entry):
        command = entry_command(entry)
        return run(syntax_only(command, host_compiler_is_msvc(command)), entry["directory"])

    def check_cu(self, entry, index):
        keep = self.work / f"tu{index}"
        keep.mkdir(parents=True, exist_ok=True)
        command = entry_command(entry) + f' --dryrun --keep --keep-dir "{keep}"'
        status, output = run(command, entry["directory"])
        if status != 0:
            return status, "nvcc --dryrun failed:\n" + output
        assignments, commands = dryrun_steps(output)
        env = dict(os.environ)
        env.update(assignments)
        trace = os.environ.get("NINFER_HOST_CHECK_TRACE")
        if trace and re.search(trace, entry["file"]):
            print(output, flush=True)
        log, stub_written = [], False
        # Three steps run, in the keep directory, since cudafe++ names its stub relatively and
        # sources of one name in different directories must not share it: the host preprocessing
        # (into .cpp4.ii), cudafe++, and the host compile of cudafe++'s output. The device
        # preprocessing (.cpp1.ii), cicc, ptxas and fatbinary do not; the fatbinary source a stub
        # includes is the probe's.
        for step in commands:
            tool, host_compile = tool_name(step), False
            fatbin = option_path(step, "--embedded-fatbin")
            if fatbin:
                target = keep / fatbin
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.fatbin_template, target)
            if tool.startswith("cudafe++"):
                module = option_path(step, "--module_id_file_name")
                if module:
                    path = keep / module
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(MODULE_ID)
            elif ".cudafe1.cpp" in step and re.search(r"(^|\s)[-/]c(\s|$)", step):
                step, host_compile = syntax_only(step, tool in ("cl", "cl.exe")), True
            elif ".cpp4.ii" not in step or tool.startswith(DEVICE_TOOLS):
                continue
            status, out = run(step, keep, env)
            log.append(out)
            if status != 0 and not (host_compile and stub_written and only_static_kernels(out)):
                return status, f"step failed: {step[:300]}\n" + "".join(log)
            if tool.startswith("cudafe++"):
                stub = option_path(step, "--stub_file_name")
                if stub and not (keep / stub).exists():
                    (keep / stub).write_text(HOST_STUB)
                    stub_written = True
        return 0, ""


def make_fatbin_template(entries, work):
    """A real embedded fatbinary source from a one-kernel .cu compiled with the build's own
    nvcc command, which every translation unit's stub includes in place of its own."""
    cuda = entry_command(next(e for e in entries if e["file"].endswith(".cu")))
    nvcc = re.match(r'\s*("[^"]+"|\S+)', cuda).group(1)
    ccbin = re.search(r'(-ccbin[= ](?:"[^"]+"|\S+))', cuda)
    probe = work / "probe"
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "probe.cu").write_text("__global__ void ninfer_host_check_probe() {}\n")
    command = (f'{nvcc} {ccbin.group(1) if ccbin else ""} -c probe.cu -o probe.o '
               f'--keep --keep-dir "{probe}"')
    status, output = run(command, probe)
    found = sorted(probe.glob("*.fatbin.c"))
    if status != 0 or not found:
        sys.exit("cannot compile the fatbinary probe:\n" + output)
    return found[0]


def shard(text):
    found = re.fullmatch(r"([0-9]+)/([0-9]+)", text)
    if not found or not 1 <= int(found.group(1)) <= int(found.group(2)):
        raise argparse.ArgumentTypeError("expected I/N with 1 <= I <= N")
    return int(found.group(1)), int(found.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dirs", type=Path, nargs="+", metavar="BUILD_DIR")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--shard", type=shard, default=(1, 1),
                        help="check only every N-th source, from the I-th")
    parser.add_argument("--filter", default=None, help="only sources matching this regex")
    args = parser.parse_args()

    entries, total = load_entries([d.resolve() for d in args.build_dirs])
    if args.filter:
        entries = [e for e in entries if re.search(args.filter, e["file"])]
    distinct = len(entries)
    index, count = args.shard
    entries = entries[index - 1::count]
    work = Path(tempfile.mkdtemp(prefix="ninfer-host-check-"))
    started = time.monotonic()
    try:
        has_cu = any(e["file"].endswith(".cu") for e in entries)
        checker = Checker(work, make_fatbin_template(entries, work) if has_cu else None)

        def check(item):
            position, entry = item
            if entry["file"].endswith(".cu"):
                return entry["file"], *checker.check_cu(entry, position)
            return entry["file"], *checker.check_cpp(entry)

        failures = []
        with ThreadPoolExecutor(args.jobs) as pool:
            for name, status, output in pool.map(check, enumerate(entries)):
                if status != 0:
                    failures.append(name)
                    # From the first error: cudafe++'s warnings can fill the excerpt before it.
                    lines = output.strip().splitlines()
                    start = next((i for i, line in enumerate(lines)
                                  if re.search(r"\berror\b", line)), 0)
                    shown = lines[:1] + lines[max(start, 1):start + 40]
                    print(f"FAIL {name}\n" + "\n".join(shown), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    cu = sum(e["file"].endswith(".cu") for e in entries)
    print(f"{len(entries) - len(failures)} of {len(entries)} translation units ({cu} .cu) "
          f"compile on the host in {time.monotonic() - started:.0f} s; shard {index}/{count} of "
          f"{distinct} distinct among {total} in {len(args.build_dirs)} build(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
