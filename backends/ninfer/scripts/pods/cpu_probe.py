#!/usr/bin/env python3
"""Extract explicit Q2_0 banks and run the standalone CPU routed-expert experiment."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
sys.path.insert(0, str(Path.cwd()))
from tools.convert.sources.gguf import GGUFFile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("gguf", type=Path)
parser.add_argument("--layer", required=True, type=int)
args = parser.parse_args()
root = Path("/workspace/ninfer-work")
fixture = root / "models" / f"cpu-q2-layer-{args.layer}.bin"
with GGUFFile(args.gguf) as source, fixture.open("wb") as output:
    output.write(struct.pack("<III", 2560, 640, 80))
    for projection, rows, k in (("gate",640,2560), ("up",640,2560), ("down",2560,640)):
        name = f"blk.{args.layer}.ffn_{projection}_exps.weight"
        info = source.info(name)
        if info.type_name != "Q2_0":
            raise ValueError(f"{name}: {info.type_name}, expected Q2_0")
        size = 80 * rows * k // 64 * 18
        output.write(source.tensor_bytes(name, 0, size))
        print(json.dumps({"tensor":name, "format":info.type_name, "bytes":size}), flush=True)
binary = root / "cpu-q2-probe"
probe = Path(__file__).with_name("q2_cpu_probe.cpp")
if not probe.exists():
    probe = Path("bench/ops/q2_cpu_probe.cpp")
subprocess.run(["g++", "-std=c++20", "-O3", "-mavx2", "-mf16c", "-fopenmp",
                str(probe), "-o", str(binary)], check=True)
affinity = Path("/sys/devices/system/node/node0/cpulist").read_text().strip()
print(json.dumps({"cpu_affinity": affinity, "layer": args.layer,
                  "scope": "ten routed experts per token; shared expert excluded"}), flush=True)
environment = dict(os.environ, OMP_PLACES="cores", OMP_PROC_BIND="true", OMP_WAIT_POLICY="PASSIVE")
subprocess.run(["taskset", "-c", affinity, str(binary), str(fixture)],
               check=True, env=environment)
