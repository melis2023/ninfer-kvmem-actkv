@echo off
setlocal EnableExtensions

title KVMem server (ninfer backend) - Qwen3.8-27B q3blocks + DFlash2

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-kvmem-server.exe"

REM model
set "MODEL=I:\model\gguf\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp-q3blocks.ninfer"


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
  --port 18200 --model-id ninfer-27b --host 127.0.0.1 ^
    --max-context 262144 ^
    --prefill-chunk 256 --full-draft-head ^
    --kvmem-budget 36480 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288 ^
  --kvmem-lrfu ^
  --kv-dtype nvfp4  --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --spec dflash2 --draft-tokens 5 --ngram-draft-tokens 31 ^
  --default-reasoning-effort medium --default-thinking-budget 16384


set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
