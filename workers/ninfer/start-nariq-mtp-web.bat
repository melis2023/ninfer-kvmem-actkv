@echo off
setlocal EnableExtensions

title NariQ server (ninfer backend, MTP + KVMem B=32000 web) - Qwen3.8-27B NariQ 3.0bpw + Vision

REM launcher directory
set "BIN=%~dp0"
set "EXE=%BIN%ninfer-serve.exe"

REM Web-profile MTP on KVMem: 36834-token budget, 12288 MiB host tier with the keep hint.
REM The MTP head ships inside the artifact, so there is no --draft-model and no drafter gguf.
REM ngram copy rounds stay on for chat-style output (15 drafts).
REM
REM --draft-tokens 4 is the measured optimum: the adaptive controller loses a few percent
REM on the HTML bench and wider fixed windows lose more (every extra draft is one serial
REM drafter pass). The five-row verify now runs the TT=8/CPS=5 coded tile: 64.9 tok/s on
REM the 2048-token HTML bench, ahead of the GSQ artifact's 58.6.
set "MODEL=I:\model\NariQ-MTP-q4vis.ninfer"

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
  --model-id ninfer-27b ^
  --host 127.0.0.1 --port 18200 ^
  --spec mtp --draft-tokens 4 ^
  --max-context 262144 ^
  --prefill-chunk 256 ^
  --kvmem-budget 36834 --kvmem-gen-reserve 2048 --kvmem-host-mib 12288 ^
  --kvmem-keep ^
  --ngram-draft-tokens 15 ^
  --kv-dtype rk8v4 --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --default-reasoning-effort xhigh ^
  --vision --vision-residency cpu --vision-max-merged 256

set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
