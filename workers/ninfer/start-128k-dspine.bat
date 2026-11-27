@echo off
setlocal EnableExtensions

title KVMem server (ninfer backend) - Qwen3.8-27B IQ3_XS + DSpine Q4_K_M

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM model
set "MODEL=I:\model\NariQ-MTP-q4vis.ninfer"
set "DRAFT=I:\model\dflash2\Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf"

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
  --kvmem-budget 30720 --kvmem-gen-reserve 2048 --kvmem-host-mib 12288 ^
  --kvmem-keep ^
  --ngram-draft-tokens 15 ^
  --kv-dtype nvfp4 --kv-dspine-f16 8192 ^
--temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --spec dspine --draft-tokens 15 ^
  --default-reasoning-effort xhigh

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
