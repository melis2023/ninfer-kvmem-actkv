"""Limit compiler processes to the container's CPU and RAM allocation."""
import argparse
import json
import math
import os
from pathlib import Path

GIB = 1024 ** 3


def budget(requested, affinity, cpu_limits, memory_bytes):
    cpus = min([affinity, *cpu_limits])
    # Reserve memory for the runner, linker and cache checkpoint alongside nvcc.
    memory_jobs = max(1, (memory_bytes - 8 * GIB) // (2 * GIB))
    return max(1, min(requested, math.ceil(cpus), memory_jobs))


def allocation(root=Path('/sys/fs/cgroup')):
    cpus = []
    memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    path = root / 'cpu.max'
    if path.exists():
        quota, period = path.read_text().split()
        if quota != 'max':
            cpus.append(int(quota) / int(period))
    for directory in ('cpu', 'cpu,cpuacct', ''):
        quota = root / directory / 'cpu.cfs_quota_us'
        period = root / directory / 'cpu.cfs_period_us'
        if quota.exists() and period.exists() and int(quota.read_text()) > 0:
            cpus.append(int(quota.read_text()) / int(period.read_text()))
    for name in ('memory.max', 'memory/memory.limit_in_bytes', 'memory.limit_in_bytes'):
        path = root / name
        if path.exists() and path.read_text().strip() != 'max':
            memory = min(memory, int(path.read_text()))
    return cpus, memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    limits, memory = allocation()
    affinity = len(os.sched_getaffinity(0))
    jobs = budget(args.jobs, affinity, limits, memory)
    print(json.dumps(dict(requested=args.jobs, affinity=affinity, cpu_quotas=limits,
                          memory_limit_bytes=memory, jobs=jobs)), flush=True)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        output.write(f'jobs={jobs}\n')


if __name__ == '__main__':
    main()
