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
    --prefill-chunk 256  --kvmem-keep --kvmem-recent-tokens 12288 ^
    --kvmem-budget 39040 --kvmem-gen-reserve 4096 --kvmem-host-mib 12288 ^
  --kvmem-lrfu ^
  --kv-dtype int8  --temperature 1 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --spec dflash2 --draft-tokens 4 --ngram-draft-tokens 31 ^
  --vision --vision-residency cpu --vision-max-merged 256 ^
  --default-reasoning-effort xhigh --default-thinking-budget 32768


set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%



