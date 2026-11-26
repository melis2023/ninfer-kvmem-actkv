@echo off
setlocal EnableExtensions

title NariQ server (ninfer backend, DSpine 64K) - Qwen3.8-27B NariQ 3.0bpw + Vision

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM 64K-capacity DSpine profile: 16-token verify window (draft 15) over a 65,536-token
REM KV pool. The FP16 hot ring still covers the most recent 8192 tokens, so recall
REM beyond that falls back to the int8 Backend pool.
REM
REM Budget on a 16 GB card: weights 9.74 GiB + DSpine drafter 1.06 GiB + 64K int8 KV
REM 3.44 GiB + ring 0.30 GiB sits within a few hundred MiB of the device limit. If the
REM decode collapses to single digits with host near 100 percent, drop --draft-tokens
REM to 8 (halves the verify workspace) or --kv-capacity to 32768.
set "MODEL=I:\model\MTP-sl-p131k.ninfer"
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
  --model-id ninfer-nariq-27b --host 127.0.0.1 --port 18200 ^
  --draft-model "%DRAFT%" ^
  --spec dspine --draft-tokens 15 ^
  --kv-dspine-f16 8192 ^
  --max-context 65536 --kv-capacity 65536 --max-concurrency 1 ^
  --host-kv-mib 0 --host-state-slots 0 ^
  --ngram-draft-tokens 15 --ngram-min-match 12 ^
  --kv-dtype int8 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --default-reasoning-effort xhigh

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
