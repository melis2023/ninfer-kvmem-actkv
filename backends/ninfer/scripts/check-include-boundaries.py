#!/usr/bin/env python3
"""Include boundaries of the public interface.

- Only include/ninfer/engine.h includes the umbrella ninfer/types.h. Other code includes the
  header that defines the types it uses, so a change to one domain recompiles only its users.
- CUDA translation units under src/ do not reach the Engine-level headers: a change to the Engine
  options, requests or statistics would otherwise recompile the kernels. The kernels take what
  they need from the small headers (ninfer/base.h, ninfer/model_enums.h,
  ninfer/kv_cache_storage.h, ninfer/startup.h).
- ninfer/types_fwd.h declares every struct, class and enum the headers listed in ninfer/types.h
  define, with the same keyword, and nothing else.

The check reads #include lines as text and resolves them against the include directories the build
uses, so an include inside an #if counts.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIRS = ['include', 'src', 'tests', 'bench', 'third_party']
SOURCE_DIRS = ['include', 'src', 'apps', 'tests', 'bench', 'tools']
SOURCE_SUFFIXES = ('.h', '.hpp', '.cuh', '.inl', '.cpp', '.cc', '.cu')
UMBRELLA = 'include/ninfer/types.h'
UMBRELLA_OWNER = 'include/ninfer/engine.h'
DECLARATIONS = 'include/ninfer/types_fwd.h'
ENGINE_LEVEL = {
    'include/ninfer/engine.h',
    'include/ninfer/engine_options.h',
    'include/ninfer/observability/execution.h',
    'include/ninfer/observability/resources.h',
    'include/ninfer/observability/snapshot.h',
    'include/ninfer/request.h',
    'include/ninfer/types.h',
}
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
TYPE = re.compile(r'^(struct|class|enum class)\s+(\w+)\b[^;]*?(\{|;)', re.M)


def resolve(source, name, cache):
    for directory in [os.path.dirname(source)] + INCLUDE_DIRS:
        candidate = os.path.normpath(os.path.join(directory, name))
        if candidate in cache or os.path.isfile(os.path.join(ROOT, candidate)):
            cache.add(candidate)
            return candidate
    return None


def includes(path, memo, cache):
    if path not in memo:
        try:
            with open(os.path.join(ROOT, path), encoding='utf-8', errors='replace') as file:
                text = file.read()
        except OSError:
            text = ''
        memo[path] = [r for r in (resolve(path, n, cache) for n in INCLUDE.findall(text)) if r]
    return memo[path]


def chain_to(start, targets, memo, cache):
    parent = {start: None}
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current in targets:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        for child in includes(current, memo, cache):
            if child not in parent:
                parent[child] = current
                queue.append(child)
    return None


def read(path):
    with open(os.path.join(ROOT, path), encoding='utf-8') as file:
        return file.read()


def declaration_failures(memo, cache):
    defined = {}
    for header in includes(UMBRELLA, memo, cache):
        if header != DECLARATIONS:
            defined.update({m[1]: m[0] for m in TYPE.findall(read(header)) if m[2] == '{'})
    declared = {m[1]: m[0] for m in TYPE.findall(read(DECLARATIONS)) if m[2] == ';'}
    failures = [f'{DECLARATIONS} lacks {kind} {name}' for name, kind in sorted(defined.items())
                if declared.get(name) != kind]
    failures += [f'{DECLARATIONS} declares {name}, which no listed header defines'
                 for name in sorted(set(declared) - set(defined))]
    return failures


def main():
    memo, cache, failures = {}, set(), []
    failures += declaration_failures(memo, cache)
    for top in SOURCE_DIRS:
        for directory, _, names in os.walk(os.path.join(ROOT, top)):
            for name in sorted(names):
                if not name.endswith(SOURCE_SUFFIXES):
                    continue
                source = os.path.relpath(os.path.join(directory, name), ROOT)
                if source != UMBRELLA_OWNER and UMBRELLA in includes(source, memo, cache):
                    failures.append(f'{source} includes {UMBRELLA}; include the defining headers')
    for directory, _, names in os.walk(os.path.join(ROOT, 'src')):
        for name in sorted(names):
            if not name.endswith('.cu'):
                continue
            source = os.path.relpath(os.path.join(directory, name), ROOT)
            chain = chain_to(source, ENGINE_LEVEL, memo, cache)
            if chain:
                failures.append(f'{source} reaches {chain[-1]}: ' + ' -> '.join(chain))
    for failure in failures:
        print(failure)
    if failures:
        print(f'{len(failures)} include boundary violation(s)', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
