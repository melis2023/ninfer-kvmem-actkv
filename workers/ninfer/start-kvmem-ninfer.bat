@echo off
setlocal EnableExtensions

title KVMem server (ninfer backend) - Qwen3.8-27B NariQ 3.0bpw (native Trellis)

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM model: converted from I:\model\Qwen3.8-27B-NariQ-3.0bpw.gguf
REM .ninfer is the raw conversion (incomplete config); engine10 is the loadable one.
REM For coded Trellis weights the engine auto-avoids the CUDA-graph/state-slot fault:
REM with KVMem it turns graphs off, otherwise it drops the device/host context cache.
REM No manual --no-cuda-graph / --no-prefix-reuse needed.
set "MODEL=I:\model\Qwen3.8-27B-NariQ-3.0bpw.engine10.ninfer"

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
  --spec none --ngram-draft-tokens 0 ^
  --max-context 262144 ^
  --prefill-chunk 256 ^
  --kvmem-budget 28672 --kvmem-gen-reserve 2048 --kvmem-host-mib 12288 ^
  --kvmem-lrfu --kvmem-keep ^
  --kv-dtype int8 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --default-reasoning-effort xhigh

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
