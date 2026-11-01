#!/usr/bin/env python3
"""Convert retained pinned inputs to native Q2 experts, reusing the original ngram artifact."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path("/workspace/ninfer-work")
MODELS = ROOT / "models"
OUTPUT = MODELS / "flash-next-native-q2-mtp.ninfer"
GGUF = MODELS / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
TABLE = MODELS / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
MTP = MODELS / "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
COMPANION = MODELS / "flash-next-q2_0-mtp.ninfer"
REPORT = Path(str(OUTPUT) + ".conversion.json")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--refresh", action="store_true", help="replace the verified generated native artifact")
args = parser.parse_args()

for path in (GGUF, TABLE, MTP, COMPANION, ROOT / "jobs/model-inputs.json"):
    if not path.is_file():
        raise RuntimeError(f"retained pinned prerequisite missing: {path}")
print(json.dumps(json.loads((ROOT / "jobs/model-inputs.json").read_text())), flush=True)
if args.refresh and OUTPUT.exists():
    from tools.artifact.reader import Artifact

    previous = json.loads(REPORT.read_text())
    with Artifact(OUTPUT) as artifact:
        if (previous["artifact_id"] != artifact.artifact_id.hex() or
                artifact.directory.provenance.get("recipe") != "qwen3_8_flash_next_gsq_q2" or
                previous["provenance"].get("recipe") != "qwen3_8_flash_next_gsq_q2" or
                len(artifact.directory.files) != 1):
            raise RuntimeError("refusing refresh: output is not the generated single-file native artifact")
    # Preserve the old report; the original artifact and its pinned inputs remain available.
    shutil.copyfile(REPORT, Path(os.environ["NINFER_JOB_DIR"]) / "previous-conversion.json")
    OUTPUT.unlink()
    REPORT.unlink()
    print("REFRESH_NATIVE_ARTIFACT previous report retained", flush=True)
if not OUTPUT.exists():
    # Same trained grids and exact plane byte counts at H=2560/I=640; exclude table payload.
    from tools.artifact.reader import Artifact

    with Artifact(COMPANION) as artifact:
        binding = artifact.directory.bindings["ngram/table"]
        table_object = binding["object"]
        table_bytes = artifact.by_id[table_object].bytes
    estimate = COMPANION.stat().st_size - table_bytes
    if shutil.disk_usage(ROOT).free < estimate + 2 * 1024**3:
        raise RuntimeError(f"need {estimate} output bytes plus 2 GiB reserve")
    temporary = OUTPUT.with_name("flash-next-native-q2-mtp.partial.ninfer")
    if temporary.exists():
        raise RuntimeError(f"unfinished output retained; inspect before retry: {temporary}")
    subprocess.run([sys.executable, "-m", "tools.convert", "--model", str(MODELS / "config"),
                    "--recipe", "qwen3_8_flash_next_gsq_q2", "--components", "text,mtp",
                    "--source", f"gguf={GGUF}", "--source", f"ngram={TABLE}",
                    "--source", f"mtp={MTP}", "--device", "cpu", "--rows-per-chunk", "65536",
                    "--name", "qwen3.8-flash-next", "--out", str(temporary)], check=True)
    temporary.rename(OUTPUT)
    report = Path(str(temporary) + ".conversion.json")
    if report.exists():
        report.rename(Path(str(OUTPUT) + ".conversion.json"))
print(f"NATIVE_ARTIFACT_READY {OUTPUT} bytes={OUTPUT.stat().st_size} companion={COMPANION}", flush=True)
