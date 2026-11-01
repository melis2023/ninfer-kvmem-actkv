#!/usr/bin/env python3
"""Writes src/models/expert_profiles/builtin.inc from routing profiles recorded with
--expert-profile-out.

Each --profile FAMILY=PATH names a model family ("qwen3.8-flash-next", "gemma4", "qwen3.5-moe",
"flint") and a recorded profile of one of its releases. The built-in profile keeps the counts and
the layers' bank widths, not the artifact identity: it serves every artifact of that family and
geometry (models/expert_profile.h). Counts above 2^32 - 1 are scaled down together, which keeps
their order.
"""

import argparse
import json
import pathlib
import sys

HEADER = """// The built-in routing profiles (models/expert_profile.cpp includes this file). Each was recorded
// with --expert-profile-out and converted by tools/expert_profile_inc.py.
"""


def load(path: pathlib.Path) -> list[list[int]]:
    data = json.loads(path.read_text())
    if data.get("version") != 1 or not isinstance(data.get("counts"), list):
        raise ValueError(f"{path}: not a version 1 expert profile")
    counts = data["counts"]
    for layer in counts:
        if not layer or any(not isinstance(v, int) or v < 0 for v in layer):
            raise ValueError(f"{path}: counts must be nonnegative integers")
        if len(layer) > 0xFFFF:
            raise ValueError(f"{path}: a bank wider than 65535 experts")
    peak = max(max(layer) for layer in counts)
    if peak > 0xFFFFFFFF:
        scale = 0xFFFFFFFF / peak
        counts = [[int(v * scale) for v in layer] for layer in counts]
    return counts


def lines(values: list[int], indent: str = "    ", width: int = 100) -> str:
    out, line = [], indent
    for v in values:
        item = f"{v}, "
        if len(line) + len(item) > width:
            out.append(line.rstrip())
            line = indent
        line += item
    if line.strip():
        out.append(line.rstrip())
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", default=[], metavar="FAMILY=PATH")
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    text = [HEADER]
    entries = []
    for index, spec in enumerate(args.profile):
        family, sep, path = spec.partition("=")
        if not sep or not family or not path:
            parser.error(f"--profile {spec!r}: expected FAMILY=PATH")
        counts = load(pathlib.Path(path))
        widths = [len(layer) for layer in counts]
        flat = [v for layer in counts for v in layer]
        text.append(f"constexpr std::uint16_t kWidths{index}[] = {{\n{lines(widths)}\n}};")
        text.append(f"constexpr std::uint32_t kCounts{index}[] = {{\n{lines(flat)}\n}};\n")
        entries.append(f'    {{"{family}", kWidths{index}, kCounts{index}}},')
    if not entries:
        entries.append('    {"", {}, {}},')
    text.append("constexpr Builtin kBuiltinProfiles[] = {\n" + "\n".join(entries) + "\n};\n")
    args.output.write_text("\n".join(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
