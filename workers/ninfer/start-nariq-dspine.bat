@echo off
setlocal EnableExtensions

title NariQ server (ninfer backend, DSpine + FP16 hot ring 8192) - Qwen3.8-27B NariQ 3.0bpw + Vision

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM DSpine all-sliding profile: the FP16 hot ring keeps the most recent 8192 target
REM KV tokens resident at FP16 (nine attention layers, ~302 MiB), which is every KV
REM column the sliding verify reads - no int8 dequant on the hot path. Contexts
REM beyond 8192 are dropped (sliding); for full-history recall use start-nariq-long.bat.
REM The context cache's default 8 GiB pinned host arena is dead weight here and pages
REM the weights to shared memory, so it is switched off explicitly.
REM
REM --draft-tokens 15 (verify window 16): the trellis verify runs a 16-row MMA tile,
REM so a 5-token window idles 69 percent of it. A full tile lifts the HTML bench from
REM 53 to 84 tok/s and the 8K recall decode from 54 to 93 tok/s; the acceptance rate
REM falls (63 to 42 percent on HTML) but the amortised weight decode wins. MTP is the
REM opposite - its serial drafter pays for every extra token - so it stays at 4.
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
