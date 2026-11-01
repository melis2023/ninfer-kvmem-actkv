"""Install the pinned ccache binary used by native and container CI builds."""
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile

VERSION = "4.14.2"
ASSETS = {
    "Windows": ("windows-x86_64.zip", "6993ce645e53192cec592ee23186200b574ac4c59097ecde8d9cfd681b47ae84"),
    "Linux": ("linux-x86_64-musl-static.tar.gz", "d540ba01f160addbd3dd5e18eda9bd34005591b095c09ac8ae9d0a1b655e7ef5"),
}


def main():
    suffix, digest = ASSETS[platform.system()]
    root = Path(".cache/ccache-tool").resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / suffix
    url = f"https://github.com/ccache/ccache/releases/download/v{VERSION}/ccache-{VERSION}-{suffix}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
            with archive.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Compiler cache download checksum mismatch")
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5)
    if suffix.endswith(".zip"):
        with zipfile.ZipFile(archive) as contents:
            contents.extractall(root)
    else:
        with tarfile.open(archive) as contents:
            contents.extractall(root, filter="data")
    binaries = list(root.rglob("ccache.exe" if os.name == "nt" else "ccache"))
    binary, = [path for path in binaries if path.is_file()]
    subprocess.run([str(binary), "--version"], check=True)
    with open(os.environ["GITHUB_PATH"], "a") as output:
        output.write(str(binary.parent) + "\n")
    # CCACHE_PATH is the compiler search path, not the ccache executable. Leave it
    # unset so CUDA cache keys include the host compiler found on PATH.


if __name__ == "__main__":
    main()
