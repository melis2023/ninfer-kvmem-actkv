@echo off
setlocal EnableExtensions

title KVMem server (ninfer backend) - Qwen3.8-27B-IQ3_XXS + DFlash2

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-kvmem-server.exe"

REM model
set "MODEL=I:\model\ninfer\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer"

REM GPU: an existing CUDA_VISIBLE_DEVICES wins; otherwise this box's RTX 5060 Ti
if defined CUDA_VISIBLE_DEVICES (set "GPU=%CUDA_VISIBLE_DEVICES%") else (set "GPU=GPU-dba26dd9-930f-1b01-2b05-22c1fca6cdce")
set "CUDA_VISIBLE_DEVICES=%GPU%"

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
  --port 18200 --model-id ninfer-27b ^
    --max-context 262144   --kvmem-sink-tokens 1024 ^
    --prefill-chunk 256 --no-cuda-graph --full-draft-head ^
    --kvmem-budget 36864 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288 ^
  --kvmem-lrfu ^
  --kv-dtype nvfp4  --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --spec dflash2 --draft-tokens 4 --ngram-draft-tokens 15 ^
  --vision --vision-residency cpu --vision-max-merged 256 ^
  --default-reasoning-effort xhigh  --default-thinking-budget 16384

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
