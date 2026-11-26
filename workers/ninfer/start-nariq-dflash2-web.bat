@echo off
setlocal EnableExtensions

title NariQ server (ninfer backend, DFlash2 plain) - Qwen3.8-27B NariQ 3.0bpw + Vision

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM Plain-profile DFlash2: 12288-token KV, no KVMem paging. Measured fastest for
REM short-prompt long-output work (the pelican HTML bench: 65.9 tok/s decode 66.3,
REM against 64.4 for the KVMem profile). Use start-nariq-dflash2.bat (KVMem B=32000)
REM for long prompts instead - the plain profile needs the KVMem paging for 8K+
REM and falls to ~6 tok/s without it (GPU runtime pathology, not a code path).
REM
REM --draft-tokens 6 (seven verify rows) lands on the TT=8/CPS=4 coded tile and
REM measured faster than 8 (nine rows, TT=16) and 5 (62.1): the 50-52 percent
REM acceptance covers the shorter window.
set "MODEL=I:\model\MTP-sl-p131k.ninfer"
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
  --spec dflash2 --draft-tokens 6 ^
  --max-context 12288 --kv-capacity 12288 --max-concurrency 1 ^
  --host-kv-mib 0 --host-state-slots 0 ^
  --ngram-draft-tokens 15 --ngram-min-match 12 ^
  --kv-dtype int8 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --default-reasoning-effort xhigh

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
