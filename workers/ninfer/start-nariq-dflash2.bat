@echo off
setlocal EnableExtensions

title NariQ server (ninfer backend, DFlash2 + KVMem B=32000) - Qwen3.8-27B NariQ 3.0bpw MTP + Vision

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM DFlash2 drafter adds ~1 GiB on top of the 9.74 GiB weights, so the KVMem
REM device budget drops from 65536 to 32000 (KV 34,048 tokens with the reserve,
REM free 2.07 GiB). CUDA Graphs stay off for DFlash2 by design.
set "MODEL=I:\model\MTP-splice-vision.ninfer"
set "DRAFT=I:\model\dflash2\Qwen3.8-27B-DFlash2-Q4_K_M.gguf"

if not exist "%EXE%" (
    echo ERROR: worker missing: "%EXE%"
    pause
    exit /b 2
)
if not exist "%MODEL%" (
    echo ERROR: model missing: "%MODEL%"
    pause
    exit /b 2
)
if not exist "%DRAFT%" (
    echo ERROR: draft model missing: "%DRAFT%"
    pause
    exit /b 2
)

cd /d "%BIN%"

"%EXE%" ^
  "%MODEL%" ^
  --model-id ninfer-nariq-27b --host 127.0.0.1 --port 18200 ^
  --draft-model "%DRAFT%" ^
  --spec dflash2 --draft-tokens 4 --adaptive-mtp ^
  --max-context 262144 --max-concurrency 1 ^
  --prefill-chunk 256 ^
  --kvmem-budget 32000 --kvmem-gen-reserve 2048 --kvmem-host-mib 12288 ^
  --kvmem-lrfu --kvmem-keep ^
  --ngram-draft-tokens 0 ^
  --kv-dtype int8 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --default-reasoning-effort xhigh

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
