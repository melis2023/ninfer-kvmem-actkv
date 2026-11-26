@echo off
setlocal EnableExtensions

title NariQ server (ninfer backend, DFlash2 + KVMem B=32000 web) - Qwen3.8-27B NariQ 3.0bpw + Vision

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM Web-profile DFlash2 on KVMem: 32000-token budget, 12288 MiB host tier, LRU eviction
REM with the keep hint. Same paging profile as start-nariq-dflash2.bat, but with the
REM ngram copy rounds on for chat-style output (15 drafts, 12-token minimum match).
REM
REM --draft-tokens 6 (seven verify rows) lands on the TT=8/CPS=4 coded tile:
REM 65.6 tok/s on the 2048-token HTML bench and 64.4 at 8K, against 49 for the
REM old TT=16 window. Draft 5 loses (62.1): the acceptance gain does not cover
REM the shorter window.
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
  --max-context 262144 --max-concurrency 1 ^
  --prefill-chunk 256 ^
  --kvmem-budget 32000 --kvmem-gen-reserve 2048 --kvmem-host-mib 12288 ^
  --kvmem-lrfu --kvmem-keep ^
  --ngram-draft-tokens 15 --ngram-min-match 12 ^
  --kv-dtype int8 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --default-reasoning-effort xhigh

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
