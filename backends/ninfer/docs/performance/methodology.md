# Serving corpus methodology

[Performance](../performance.md) · [Runner usage](../../tools/bench/README.md#serving-corpus-benchmark)

The workloads, measurement boundaries and metrics of the serving corpus runners
(`tools/bench/run_serve_corpus.py`, `tools/bench/run_serve_concurrency.py`). Engine and Op
benchmarks have their own contracts in [bench/README.md](../../bench/README.md); their rates do not
establish HTTP serving performance.

## Server profile

Every run serves one persistent `ninfer-serve` over loopback OpenAI Chat Completions with
`stream=false`, CUDA Graphs on, prefix reuse off (`--no-prefix-reuse`), 1,024-token prefill chunks
and the KV format the runner is given (`int8` by default). Startup and warmup lie outside the
measured requests. Stochastic runs pin their whole sampling profile (temperature 0.6, top-p 0.95,
top-k 20, min-p 0, presence penalty 1.0, frequency penalty 0), so a change of model defaults does
not change the method; `--sampling greedy` selects exact argmax.

| Report label | Runner mode | Server options |
|---|---|---|
| MTP0 (no speculation) | `mtp0` | No `--spec` |
| MTP3 | `mtp3` | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| DFlash, block=8 | `dflash7` | `--spec dflash --draft-tokens 7 --lm-head-draft` |
| DFlash2, block=8 | `dflash2_7` | `--spec dflash2 --draft-tokens 7 --lm-head-draft` |

`K` denotes draft tokens; block size is `K+1`. Weight format, backend, draft count, and proposal
head are separate experimental dimensions.

## Workloads and measurement boundaries

### Single-request phases

The MTP0 context-length profile uses `long_niah_8k`, `long_niah_64k`, `long_niah_128k`, and
`long_niah_256k`: thinking disabled, 128 output tokens requested, five seeds per fixture.
Tables retain actual prompt lengths. This suite measures prefill, server TTFT, and baseline decode.

The speculative corpus uses three reasoning fixtures (`long_decode_aime26_01`,
`long_decode_aime26_15`, `long_decode_aime26_30`) with thinking enabled and a 65,536-token output
budget. Twelve other fixtures cover Code, Story, Translation, and Structured output, with three
fixtures per category, thinking disabled, and a 4,096-token output budget. Each fixture has five
seeds, giving 75 requests per corpus. Actual completion lengths are measurements, not budgets.

Fixture content is in [examples/cli/manifest.json](../../examples/cli/manifest.json); fixture
selection and the five seeds are in [run_serve_corpus.py](../../tools/bench/run_serve_corpus.py).
Keep their identity when comparing runs. Serial runs submit one request at a time to one
persistent server. A corpus-makespan C=1 point can also supply these phase statistics.

### Corpus makespan

Each concurrency point starts a fresh server. The runner shuffles the fixed request set once with
seed `20260811` and preserves its ordered HTTP send sequence at every C. Exactly C persistent
workers submit their next request after reading their current response. Makespan starts at worker
release and ends when the last complete HTTP response has been read. It includes prefill, decode,
workload transitions, admission waits, and drain. Average decode batch covers the entire run.

Prompts, seeds, and send order are fixed; stochastic continuations and token totals can vary with
weights, backends, and concurrency. Report token totals alongside makespan and rate.

### Decode saturation

The MTP3 waves use `long_decode_aime26_15`, thinking enabled, and an 8,192-token output budget per
request; the rendered prompt length is model-specific. At each C, a fresh server releases C requests with distinct fixed seeds and waits for every
response. Each request has a 16,384-token context ceiling; the measured automatic KV capacity is
`C * 16,384` tokens.

Steady throughput includes only complete one-second server intervals with zero computed prefill
tokens, `running=C`, `prefilling=0`, `decode_ready=C`, at least one decode round, and exactly C rows
in every round. Ramp-up, prefill, and drain are excluded. Wave makespan still spans client release
through the last complete response. Acceptance is aggregated over the full wave, not just steady
intervals. Each point is one wave, not a repeated-sample mean.

## Metrics and statistics

| Metric | Definition |
|---|---|
| Prefill phase (tok/s) | `prompt_tokens / prefill_seconds` for each serial request |
| Server TTFT (ms) | `1000 * (prepare_seconds + vision_seconds + prefill_seconds)` |
| Decode phase (tok/s) | `(completion_tokens - 1) / decode_seconds` for each serial request |
| Corpus prefill (tok/s) | Total computed prefill tokens / full corpus makespan |
| Corpus decode (tok/s) | Total decode tokens / full corpus makespan |
| Steady decode (tok/s) | Sum of committed decode tokens / sum of selected interval seconds |
| Requests/s | Number of requests / full corpus makespan |
| Spec acceptance | Accepted draft tokens / drafted tokens |
| Spec tokens/round | `1 + accepted_tokens / speculative_rounds` |

Server TTFT is an internal phase sum. It does not include ingress queueing or HTTP transport and
is not the external streaming TTFT measured by the [Serve TTFT client](../../tools/bench/ttft/README.md).
The three decode rates have different time denominators and must not share an unlabeled column.
Committed decode tokens exclude the first completion token produced by prefill.

Single-request tables report arithmetic mean ± sample standard deviation of per-request values.
Category rows pool 3 fixtures × 5 seeds: their deviation includes differences between fixtures.
For greedy repetitions, within-fixture timing variation remains, while deterministic output and
acceptance values may have zero deviation. A single wave or corpus point has no sample deviation.

Acceptance in a phase table is the mean of per-request ratios. Acceptance in a corpus or wave
table is the ratio of summed accepted and drafted tokens. Do not interchange these aggregations.
Calculate new statistics and comparisons from unrounded reports. Display phase rates and their
deviations to one decimal, seconds to two decimals, acceptance to one percentage decimal, and
tokens/round and speedups to two decimals. Keep token totals and sample counts exact.

## Output checks and comparisons

Performance runs check execution failures, token accounting, termination reasons, and obvious
repetition. Pathological samples stay in fixed-corpus statistics, labeled next to the affected
results; statistics that exclude samples name the exclusions. A natural stop or high throughput
alone does not establish successful task completion.

A comparison names both runs, the metric, and the differences in configuration or workload. A
throughput speedup is `new / baseline`; a makespan speedup is `baseline / new`; relative change is
`(new / baseline - 1) * 100%`. C=1 must be the matching suite/profile baseline. Fixed-request
workloads with varying output lengths are not fixed-token measurements, and MTP0 context profiles
and speculative scenario suites use different workloads, so they cannot establish per-scenario
speculative speedups. Calculate comparisons from unrounded reports; raw reports stay local under
`profiles/bench/`.
