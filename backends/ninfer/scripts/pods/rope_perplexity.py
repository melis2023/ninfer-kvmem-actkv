"""Perplexity past the native window, with YaRN and without, for every family.

One text of about a million tokens (the fixed corpus's sixteen streams joined in manifest order)
is scored by ninfer-perplexity in one disjoint window of `--long` tokens (past the 262,144-token
native window), with plain RoPE and with `--rope-yarn`, and in disjoint windows of the native
width as the reference. Each run's report.json is kept beside this script's summary.

usage: rope_perplexity.py --build DIR --models DIR --corpus DIR --out DIR [--long N]
       [--models-run qwen38-27b,gemma4,flash-next] [--runs native,plain,yarn]
       [--expert-residency host|disk] [--text-bytes N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

MODELS = {
    "qwen38-27b": {
        "artifact": "qwen38-27b/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer",
        "options": [],
    },
    "gemma4": {
        "artifact": "gemma4/gemma-4-26B-A4B-it-QAT-Q4_0-MTP-ninfer-v3.ninfer",
        "options": [],
    },
    "flash-next": {
        "artifact": "flash-next/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-ninfer-v3.ninfer",
        "options": ["--expert-residency", "{residency}", "--ngram-table",
                    "flash-next/Qwen3.8-Flash-Next-ngram-table-IQ4_NL-ninfer-v3.ninfer"],
    },
}
NATIVE = 262144


def join_corpus(corpus: pathlib.Path, out: pathlib.Path, max_bytes: int) -> pathlib.Path:
    text = out / ("corpus-joined.txt" if max_bytes == 0 else f"corpus-joined-{max_bytes}.txt")
    if text.exists():
        return text
    manifest = json.loads((corpus / "manifest.json").read_text())
    joined = bytearray()
    for stream in manifest["streams"]:
        joined += (corpus / stream["path"]).read_bytes() + b"\n\n"
    if max_bytes:
        joined = joined[:max_bytes]
    text.write_bytes(joined)
    return text


def run(binary: pathlib.Path, models: pathlib.Path, name: str, text: pathlib.Path,
        context: int, yarn: bool, out: pathlib.Path, extra: list[str], residency: str) -> dict:
    model = MODELS[name]
    label = f"{name}-{'yarn' if yarn else 'plain'}-{context}"
    report_dir = out / label
    if (report_dir / "report.json").exists():
        return json.loads((report_dir / "report.json").read_text())
    if report_dir.exists():
        for stale in report_dir.iterdir():
            stale.unlink()
    options = [option.format(residency=residency) for option in model["options"]]
    for index, option in enumerate(options):
        if option.endswith(".ninfer"):
            options[index] = str(models / option)
    command = [str(binary), str(models / model["artifact"]), "--text", str(text),
               "--context", str(context), "--disjoint", "--output", str(report_dir),
               "--log-level", "warning", *options, *extra]
    if yarn:
        command.append("--rope-yarn")
    print(f"[{label}] {' '.join(command)}", flush=True)
    started = time.time()
    log = out / f"{label}.log"
    with log.open("w") as sink:
        result = subprocess.run(command, stdout=sink, stderr=subprocess.STDOUT, check=False)
    print(f"[{label}] exit {result.returncode} in {time.time() - started:.0f} s", flush=True)
    if result.returncode != 0:
        print((log.read_text())[-3000:], flush=True)
        return {"error": result.returncode}
    return json.loads((report_dir / "report.json").read_text())


def summary(report: dict) -> str:
    if "error" in report:
        return f"failed ({report['error']})"
    overall = report["overall"]
    windows = []
    for stream in report["streams"]:
        for window in stream["windows"]:
            windows.append(window)
    per_window = ", ".join(f"{w['perplexity']:.4f}" for w in windows[:4])
    return (f"PPL {overall['perplexity']:.4f} over {overall['scored_tokens']} tokens, "
            f"{len(windows)} windows [{per_window}], "
            f"{report['timing']['scored_tokens_per_second']:.0f} tok/s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=pathlib.Path)
    parser.add_argument("--models", required=True, type=pathlib.Path)
    parser.add_argument("--corpus", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--long", type=int, default=2 * NATIVE)
    parser.add_argument("--runs", default="native,plain,yarn",
                        help="which of the three runs to make: native, plain, yarn")
    parser.add_argument("--text-bytes", type=int, default=0,
                        help="score only the first N bytes of the joined text (0: the whole corpus)")
    parser.add_argument("--models-run", default=",".join(MODELS))
    parser.add_argument("--extra", default="", help="options passed to every run, space separated")
    parser.add_argument("--expert-residency", default="host", choices=("host", "disk"),
                        help="where Flash-Next's routed experts live (disk when the host RAM is short)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binary = args.build / "apps" / "ninfer-perplexity"
    text = join_corpus(args.corpus, args.out, args.text_bytes)
    extra = args.extra.split() if args.extra else []
    results: dict[str, dict] = {}
    for name in args.models_run.split(","):
        runs = {"native": (NATIVE, False), "plain": (args.long, False), "yarn": (args.long, True)}
        for context, yarn in (runs[r] for r in args.runs.split(",")):
            label = f"{name}-{'yarn' if yarn else 'plain'}-{context}"
            results[label] = run(binary, args.models, name, text, context, yarn, args.out, extra,
                                 args.expert_residency)
            print(f"{label}: {summary(results[label])}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(
        {label: ({"error": r["error"]} if "error" in r else
                 {"perplexity": r["overall"]["perplexity"],
                  "scored_tokens": r["overall"]["scored_tokens"],
                  "windows": [w["perplexity"] for s in r["streams"] for w in s["windows"]],
                  "tokens_per_second": r["timing"]["scored_tokens_per_second"]})
         for label, r in results.items()}, indent=2))
    print("ROPE_PERPLEXITY_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
