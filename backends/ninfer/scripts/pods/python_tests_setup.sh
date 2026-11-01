#!/usr/bin/env bash
set -Eeuo pipefail
python=/workspace/ninfer-work/py311/bin/python
/root/.local/bin/uv pip install --python "$python" --quiet \
    --index-url https://pypi.org/simple pytest \
    >/dev/null 2>&1
"$python" -c 'import pytest, torch; print("pytest="+pytest.__version__, "torch="+torch.__version__)'
