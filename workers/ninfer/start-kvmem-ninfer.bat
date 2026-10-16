@echo off
setlocal EnableExtensions

REM ====================================================================
REM  KVMem multi-backend launcher - ninfer backend
REM
REM  Same shape as start-kvmem-server.bat (llama.cpp backend): ONE command
REM  line, every parameter written out inline. Edit a value, save, run it.
REM  Ctrl+C stops the server.
REM
REM  EDITING RULE: every parameter line except the LAST one must end with " ^".
REM  A trailing "^" followed by a blank line sends an empty argument, and the
REM  server then exits with 'unknown argument: ' before it loads anything.
REM  The last parameter line must have no "^" at all.
REM
REM      start-kvmem-server.bat           llama.cpp backend, ActKV ON
REM      start-kvmem-server-noactkv.bat   same line with the ActKV flags removed
REM      start-kvmem-ninfer.bat           THIS FILE - ninfer backend, ActKV ON
REM
REM  The worker is taken from workers\ninfer\ of the repository this script
REM  sits in (where the ninfer build is staged); CUDA_VISIBLE_DEVICES pins the
REM  card by UUID.
REM
REM  Alternative launcher: scripts\windows\start-backend.ps1 -Backend ninfer
REM  wraps the same worker and adds device/stage/alignment validation. This
REM  .bat trades those checks for the plain inline form, like the llama one.
REM
REM --------------------------------------------------------------------
REM  ActKV retrieval block (arXiv:2609.31395) - the flags below the model
REM  line. Ported into the shared kvmem core, so the llama.cpp and ninfer
REM  backends take the SAME names and defaults:
REM
REM  (1) Action-oriented eviction  --kvmem-lrfu
REM      Blocks are kept by a temporal LRFU keep score instead of a one-shot
REM      attention score:  KS_t = lambda*KS_{t-1} + (top-p hit ? score : 0)
REM      lambda 0.50, top-p 0.90 are the paper's tuned values.
REM          lambda -> 0  pure recency (chases the newest query)
REM          lambda -> 1  pure frequency (working set freezes)
REM      The keep score is clamped at 0 and two scoring passes inside one round
REM      no longer decay the pool twice. Remove all three flags to revert.
REM
REM  (2) GQA head aggregation      --kvmem-score-agg   [REMOVED]
REM      ActKV Alg. 3 step 8 says MAX across a GQA group's query heads. It was
REM      implemented faithfully and then MEASURED on the 40-needle / 256k test:
REM          mean  39/40 same-turn, 39/40 final, 40 tool calls, 81 total
REM          max   28/40 same-turn, 32/40 final,  1 recovery,   68 total
REM      The max run decays progressively from round 6 (3/3 -> 2/3 -> 1/3)
REM      instead of failing randomly, so the mechanism is cumulative: a per-head
REM      max is dominated by whichever head happens to peak, which makes the
REM      LRFU keep score noisier and lets action-critical blocks lose their
REM      slots over successive rounds.
REM      The option is therefore GONE, not merely defaulted: passing
REM      "--kvmem-score-agg" at all is now a startup error, and the flag is no
REM      longer accepted as a compatibility stub either. GQA aggregation is
REM      always the folded mean (which is also g x cheaper).
REM      See docs/actkv-section56-wiring-record.md section 5.
REM
REM  (3) Confidence-driven budget  --kvmem-conf-*   [REMOVED]
REM      ActKV Sec. 6 Alg. 2 was implemented and then MEASURED on the 40-needle
REM      / 256k test: starting B at 8192 tokens took recall from 40/40 to 17/40
REM      while costing about 15% of round wall time. The static B+R page pool is
REM      allocated at startup and cannot bank the memory a smaller round frees,
REM      so the mechanism has nothing to gain here. The family is gone: passing
REM      any --kvmem-conf-* flag is a startup error.
REM  (4) Semantic protection    --kvmem-protect-semantics      [WIRED, ON below]
REM      ActKV Sec. 8.1. The leading system/developer block and the last real
REM      user message are the pages the round tries to keep. Protection is ON
REM      DEMAND, not an up-front reservation: retrieval picks the window first,
REM      then the protected pages it dropped are rescued best-score-first, at
REM      most 0.25 x window pages (kProtectRescueFraction, hard-coded: no flag
REM      and no environment variable). A protected page the retrieval already
REM      keeps costs nothing, so the instruction prefix does not push a higher-
REM      scoring page out. Charging the whole span up front instead let one user
REM      turn longer than --kvmem-budget pin the entire window (measured:
REM      mandatory == selected == 576 pages, nothing left to score) and cost
REM      about 9% of decode throughput; the quarter bound measured the same
REM      33/33 recall on a 262k NIAH at no cost over switching the protection
REM      off. Without it only the positional sink window and the recent window
REM      hold those pages, so the tail of a system prompt longer than the sink
REM      is dropped as soon as LRFU scoring starts losing interest in it. It
REM      needs no other flag - it is honoured whether --kvmem-lrfu is on
REM      or off.
REM          set "KVMEM_PROTECT_TRACE=1"
REM      prints one line per compression round with protected / dropped_by_
REM      retrieval / rescued and the rescued page list, changing no decision.
REM
REM  (5) AAV access-pattern classification   KVMEM_AAV_TRACE=1   [DIAGNOSTIC ONLY]
REM      ActKV Sec. 5.1. The store keeps a 16-round access ring per block (one
REM      bit per compression round: was this block kept?) and classifies it:
REM          Silent        never kept since it appeared
REM          Cold          kept fewer than two observed rounds
REM          Continuous    kept in every observed round
REM          Intermittent  kept some rounds, missed others
REM      The paper uses the classification (its Fig. 6) to argue for ONE global
REM      lambda; it defines no per-class decay. A per-class decay was implemented
REM      here and then REMOVED: an intermittent block decaying at 0.85 instead of
REM      0.50 pushes lambda toward 1, exactly the frequency-dominated regime the
REM      paper warns "may retain obsolete continuous entries", and it measurably
REM      cost recall. The classification survives as data only.
REM          set "KVMEM_AAV_TRACE=1"
REM      before the command line and the accounting prints one row per compression
REM      round to stderr WITHOUT changing any retention decision. The row carries
REM      the class counts and intermittent_dropped - how many intermittent blocks
REM      that round's selection dropped - which is the measurement that decides
REM      whether --kvmem-lruf-lambda (the paper's own knob) is set right.
REM
REM  Chinese notes for this block: docs/actkv-retention-flags.md
REM
REM  An invalid ActKV value is rejected at startup, e.g.
REM      --kvmem-lrfu-lambda 1.5      -> "must be in (0, 1)"
REM      --kvmem-score-agg mean       -> "was removed: GQA aggregation is always mean"
REM      --kvmem-aav-protect          -> "was removed: AAV is a diagnostic"
REM      --kvmem-action-region        -> "was removed: the action probe never activated"
REM      --kvmem-query-union          -> "was removed: the action probe never activated"
REM      --kvmem-conf-budget 8192     -> "was removed: ActKV Sec. 6 adaptive budget"
REM      --kvmem-lruf-lambda 0.6      -> "requires --kvmem-lrfu"
REM --------------------------------------------------------------------
REM  ninfer differs from the llama.cpp backend (docs/multi-backend-differences.md):
REM    * page/block size is FIXED at 64 tokens (llama.cpp defaults to 128), so B,
REM      R, sink and recent round down to whole 64-token pages;
REM    * there is NO --kvmem-method: with a complete query ninfer always scores
REM      by mean-K, so ActKV's LRFU is the only retention-policy knob here;
REM    * KV dtype names are int8/bf16/fp8/nvfp4/k8v4/rk8v4/... - not q8_0;
REM    * cold disk snapshots (--kvmem-disk-path / --kvmem-disk-mib) are refused.
REM ====================================================================

title KVMem server (ninfer backend) - Qwen3.8-27B-IQ3_S-mtp + ActKV

REM ---- locate the worker, and the repository root that owns it ---------
REM This script may sit in the repository root (normal) or next to the worker
REM in workers\ninfer (copied there, the way the llama launcher lives in its
REM bin\ directory). ROOT must still name the repository root either way,
REM because model.ninfer and the error text below refer to it.
REM
REM Plain sequential "if" tests, no "else if": cmd parses "else if" as an else
REM whose command happens to be another if, and the set inside a skipped
REM branch can be lost depending on how the script was invoked. That left
REM %EXE% empty, and an empty %EXE% makes cmd fail with 9009 while the
REM continuation line's remaining tokens are run as commands.
set "EXE="
set "BIN="
set "ROOT="
if exist "%~dp0workers\ninfer\infer-kvmem-server.exe" if not defined BIN (
    set "ROOT=%~dp0"
    set "BIN=%~dp0workers\ninfer"
)
if exist "%~dp0infer-kvmem-server.exe" if not defined BIN (
    set "ROOT=%~dp0..\.."
    set "BIN=%~dp0"
)
if exist "%~dp0..\..\workers\ninfer\infer-kvmem-server.exe" if not defined BIN (
    set "ROOT=%~dp0..\.."
    set "BIN=%~dp0..\..\workers\ninfer"
)
if exist "%~dp0..\workers\ninfer\infer-kvmem-server.exe" if not defined BIN (
    set "ROOT=%~dp0.."
    set "BIN=%~dp0..\workers\ninfer"
)
if defined BIN set "EXE=%BIN%\infer-kvmem-server.exe"
if not defined ROOT set "ROOT=%~dp0"
if not defined EXE (
    echo ERROR: ninfer worker not found. Looked in:
    echo   %~dp0workers\ninfer
    echo   %~dp0
    echo   %~dp0..\..\workers\ninfer
    echo   %~dp0..\workers\ninfer
    echo.
    echo Build and stage it first:
    echo   git submodule update --init --depth 1
    echo   scripts\windows\build-backends.ps1 -Backend ninfer -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" -Jobs 10
    echo   then copy build-backends\ninfer\apps\ninfer-serve.exe to workers\ninfer\infer-kvmem-server.exe
    echo   ^(docs\ninfer-server-build-record.md has the full recipe^)
    echo.
    pause
    exit /b 2
)

REM ---- model: repo-root model.ninfer, else the artifact this box keeps
set "MODEL=%ROOT%\model.ninfer"
if not exist "%MODEL%" set "MODEL=I:\model\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.ninfer"
if not exist "%MODEL%" (
    echo ERROR: ninfer model not found:
    echo   %MODEL%
    echo Put a .ninfer artifact in the repository root as model.ninfer,
    echo or edit MODEL above.
    pause
    exit /b 2
)

REM ---- GPU: ninfer wants UUIDs, comma separated in stage order --------
REM An existing CUDA_VISIBLE_DEVICES wins; otherwise this box's RTX 5060 Ti.
if defined CUDA_VISIBLE_DEVICES (set "GPU=%CUDA_VISIBLE_DEVICES%") else (set "GPU=GPU-dba26dd9-930f-1b01-2b05-22c1fca6cdce")
set "CUDA_VISIBLE_DEVICES=%GPU%"

REM ---- last guard before the command line -----------------------------
REM An empty %EXE% would make cmd run "" and then treat the continuation
REM line's tokens as commands, reporting "not recognized ... 9009" for a
REM parameter instead of naming the real problem.
if not defined EXE (
    echo ERROR: worker path resolved empty; cannot start.
    echo   BIN=%BIN%
    echo   ROOT=%ROOT%
    pause
    exit /b 2
)
if not exist "%EXE%" (
    echo ERROR: worker missing at resolved path:
    echo   %EXE%
    pause
    exit /b 2
)
if not defined BIN set "BIN=%~dp0"

cd /d "%BIN%"


"%EXE%" ^
  "%MODEL%" ^
  --port 18200 --model-id ninfer-27b ^
  --max-context 262144 ^
  --prefill-chunk 256 ^
  --kvmem-budget 36864 --kvmem-gen-reserve 16384 --kvmem-host-mib 12288 ^
  --kvmem-lrfu --kvmem-protect-semantics ^
  --kv-dtype int8  --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --spec mtp --draft-tokens 4 --ngram-draft-tokens 31 --adaptive-mtp ^
  --vision --vision-residency cpu --vision-max-merged 256 ^
  --default-reasoning-effort xhigh  --default-thinking-budget 16384
 
set "RC=%ERRORLEVEL%"
echo.
echo Server exited with code %RC%.
pause
endlocal & exit /b %RC%
