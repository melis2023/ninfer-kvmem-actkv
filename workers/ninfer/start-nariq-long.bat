@echo off
setlocal EnableExtensions

title NariQ server (ninfer backend, KVMem long context) - Qwen3.8-27B NariQ 3.0bpw MTP + Vision

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM Long-context profile: KVMem B=65536 covers 220K strict recall (validated:
REM ORCHID=493817; COLOR=COBALT at 220K, 128K decode 57.3 tok/s). KVMem's paging
REM window disables CUDA Graphs (correctness: illegal state-slot access during
REM replay), so decode is ~38-44 tok/s - use start-nariq.bat for daily work.
set "MODEL=I:\model\MTP-splice-vision.ninfer"

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

cd /d "%BIN%"

"%EXE%" ^
  "%MODEL%" ^
  --port 18200 --model-id ninfer-nariq-27b --host 127.0.0.1 ^
  --max-context 262144 --max-concurrency 1 ^
  --prefill-chunk 256 ^
  --kvmem-budget 65536 --kvmem-gen-reserve 2048 --kvmem-host-mib 12288 ^
  --kvmem-lrfu --kvmem-keep ^
  --spec mtp --draft-tokens 4 ^
  --ngram-draft-tokens 0 ^
  --kv-dtype int8 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --default-reasoning-effort xhigh

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
