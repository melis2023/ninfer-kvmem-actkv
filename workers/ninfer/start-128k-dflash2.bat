@echo off
setlocal EnableExtensions

title KVMem server (ninfer backend) - Qwen3.8-27B IQ3_XS + DFlash2 Q4_K_M

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM model
set "MODEL=I:\model\ninfer\Qwen3.8-27B-GSQ-RCO-IQ3_S-q4km-mtp-dflash2.ninfer"
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
   --model-id ninfer-27b ^
  --draft-model "%DRAFT%" ^
  --port 18200 --host 127.0.0.1 ^
  --max-context 262144 ^
  --prefill-chunk 256 ^
  --kvmem-budget 36834 --kvmem-gen-reserve 2048 --kvmem-host-mib 12288 ^
  --kvmem-keep ^
  --ngram-draft-tokens 12 ^
  --kv-dtype rk8v4 ^
  --spec dflash2 --draft-tokens 4 --adaptive-mtp ^
--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --vision --vision-residency cpu --vision-max-merged 256 ^
  --default-reasoning-effort xhigh

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
