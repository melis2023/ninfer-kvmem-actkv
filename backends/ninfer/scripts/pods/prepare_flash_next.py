#!/usr/bin/env python3
"""Download pinned public inputs and convert the Q2_0 + shared-Q8_0 MTP test artifact."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT = Path("/workspace/ninfer-work")
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def metadata(repo, revision=None):
    suffix = f"/revision/{revision}" if revision else ""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}{suffix}", timeout=30) as response:
        return json.load(response)


def fetch(repo, info, name, directory):
    destination = directory / Path(name).name
    directory.mkdir(exist_ok=True)
    if not destination.exists() or Path(str(destination) + ".aria2").exists():
        url = f"https://huggingface.co/{repo}/resolve/{info['sha']}/{urllib.parse.quote(name)}"
        subprocess.run(["aria2c", "-x16", "-s16", "-c", "--file-allocation=none",
                        "--timeout=60", "--lowest-speed-limit=1M", "--retry-wait=3", "--max-tries=20",
                        "--summary-interval=30", "--console-log-level=warn", "-d", str(directory),
                        "-o", destination.name, url], check=True)
    return destination


def unique_file(info, suffix):
    files = [s["rfilename"] for s in info["siblings"] if s["rfilename"].endswith(suffix)]
    if len(files) != 1:
        raise ValueError(f"expected one {suffix}, found {files}")
    return files[0]


config_repo = "Qwen/Qwen3.8-Flash-Next"
gguf_repo = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
mtp_repo = "unsloth/Qwen3.8-Flash-Next-GGUF"
manifest_path = ROOT / "jobs/model-inputs.json"
pinned = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
config_info, gguf_info, mtp_info = (metadata(repo, pinned.get(repo))
                                  for repo in (config_repo, gguf_repo, mtp_repo))
manifest = {repo: info["sha"] for repo, info in ((config_repo, config_info),
                                               (gguf_repo, gguf_info), (mtp_repo, mtp_info))}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest), flush=True)
config_dir = MODELS / "config"
for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
             "chat_template.jinja"):
    fetch(config_repo, config_info, name, config_dir)
model_name = unique_file(gguf_info, "Q2_0-00001-of-00002.gguf")
table_name = unique_file(gguf_info, "Q2_0-00002-of-00002.gguf")
mtp_name = unique_file(mtp_info, "shared-Q8_0.gguf")
output = MODELS / "flash-next-q2_0-mtp.ninfer"
if not output.exists():
    model = fetch(gguf_repo, gguf_info, model_name, MODELS)
    table = fetch(gguf_repo, gguf_info, table_name, MODELS)
    mtp = fetch(mtp_repo, mtp_info, mtp_name, MODELS)
    assert shutil.disk_usage(ROOT).free > 75 * 1024**3, "need 75 GiB free for converted artifact"
    temporary = output.with_name("flash-next-q2_0-mtp.partial.ninfer")
    subprocess.run([sys.executable, "-m", "tools.convert", "--model", str(config_dir),
                    "--recipe", "qwen3_8_flash_next_gguf", "--components", "text,ngram,mtp",
                    "--source", f"gguf={model}", "--source", f"ngram={table}",
                    "--source", f"mtp={mtp}", "--device", "cpu", "--rows-per-chunk", "65536",
                    "--name", "qwen3.8-flash-next", "--out", str(temporary)], check=True)
    temporary.rename(output)
    report = Path(str(temporary) + ".conversion.json")
    if report.exists():
        report.rename(Path(str(output) + ".conversion.json"))
print(f"ARTIFACT_READY {output} bytes={output.stat().st_size}", flush=True)
