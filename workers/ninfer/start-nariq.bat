@echo off
setlocal EnableExtensions

title NariQ server (ninfer backend, fast decode) - Qwen3.8-27B NariQ 3.0bpw MTP + Vision

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM Fast daily profile: full device KV (no KVMem tier), CUDA Graphs on.
REM The engine keeps graphs by dropping the context cache (root-state MTP verify),
REM which is worth ~40%: 53.6 tok/s vs 38 tok/s eager. Only --ngram-draft-tokens 0
REM is set explicitly because the serve default (15) widens the verify window in
REM repetitive/code generations and loses more on verify than it gains on drafts.
REM Capacity: 65,536 KV tokens int8 (3.33 GiB). For 128K+ contexts use
REM start-nariq-long.bat (KVMem, CUDA Graphs off, ~38-44 tok/s).
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
  --max-context 65536 --max-concurrency 1 ^
  --spec mtp --draft-tokens 4 ^
  --ngram-draft-tokens 0 ^
  --kv-dtype int8 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --default-reasoning-effort xhigh

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
