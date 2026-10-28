@echo off
setlocal EnableExtensions

title KVMem server (ninfer backend) - Qwen3.8-27B q3blocks + DFlash2

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-kvmem-server.exe"

REM model
set "MODEL=I:\model\ninfer\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-q3km-mtp-dflash2.ninfer"


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
  --port 18200 --model-id ninfer-27b --host 0.0.0.0 ^
    --max-context 262144 ^
    --prefill-chunk 256  --kvmem-keep --kvmem-keep-ratio 35 ^
    --kvmem-budget 40320 --kvmem-gen-reserve 4096 --kvmem-host-mib 12288 ^
  --kvmem-lrfu ^
  --kv-dtype rk8v4  --temperature 0.7 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --spec dflash2 --draft-tokens 4 --ngram-draft-tokens 31 ^
  --vision --vision-residency cpu --vision-max-merged 256 ^
  --default-reasoning-effort xhigh --default-thinking-budget 16384


set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
