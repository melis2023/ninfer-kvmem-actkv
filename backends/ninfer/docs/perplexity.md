# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_nvfp4.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype int8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are the server's nine: `bf16`, `int8`, `fp8`, `rk8v4`, `rk4v4`, `rk4v4-e8`,
`rk2v4-e8`, `nvfp4`, and `k8v4`.

Seven of them were measured on this corpus with Qwen3.8-27B on an RTX 3090; the results, alongside
each format's size and decode speed, are in [`docs/config-calculator.html`](config-calculator.html).
`--fast-prefill-kernel` scores an `int8` or `rk*` cache with the fast prompt-attention kernel (as
`ninfer-serve --fast-prefill-kernel` prefills); `report.json` records it as `fast_prefill_kernel`.
The evaluator also takes the server's `--devices` (one pipeline stage per GPU), its
Qwen3.8-Flash-Next options (`--expert-residency`, `--expert-cache-mib`, `--ngram-table`,
`--ngram-ram`, `--no-ngram-table`) and its precision options (`--prefill-cublas`,
`--lm-head-q4/q6`, `--embedding-q4/q6`, `--gdn-state-fp16`, `--mlp-a8-decode`, `--no-prefill-a8`).

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

## Held-out corpus

`ninfer-ppl-1m-v1` is built from public datasets the model has almost certainly trained on, which
can understate the cost of a weight or KV representation change. The repository also includes
`ninfer-ppl-heldout-2026-09-v1`: 12 streams written after Qwen3.8's release, covering new English
and Chinese Wikipedia articles, arXiv abstracts, new GitHub code in four languages, recent code of
the fork it was built in, and chat-template conversations. Use it to judge representation quality,
and keep `ninfer-ppl-1m-v1` for comparisons with numbers already published against it. Its
selection rules and limitations are in its
[README](../eval/corpora/perplexity-heldout-2026-09/README.md).

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --corpus eval/corpora/perplexity-heldout-2026-09/manifest.json \
  --quick \
  --kv-dtype int8
```

Perplexity alone compares two artifacts only through the reference text. `tools/eval/gguf_eval.py`
measures a weight encoding's KL divergence from a reference model on the same streams with
llama.cpp's `llama-perplexity`. Every grouped NInfer format maps exactly onto llama.cpp Q8_0
blocks (a 64-weight group is two blocks sharing its FP16 scale), so `export` writes any recipe's
encoding of the BF16 checkpoint, including role/format plans the engine cannot run yet, into a
copy of a Q8_0 template GGUF whose weights dequantize to exactly the engine's values. `reference`
saves the reference logits (`--kl-divergence-base`) of each stream's first chunks, `score` reports
PPL, mean and 99th-percentile KLD and top-token agreement per stream, `probe` measures the KLD that
one band of layers adds at a lower format, and `table` summarizes the scored variants:

```bash
python3 tools/eval/gguf_eval.py reference --model ref-q8_0.gguf --base-dir kld-base \
  --corpus eval/corpora/perplexity-heldout-2026-09 --llama-perplexity <llama-perplexity>
python3 tools/eval/gguf_eval.py export --source /path/to/Qwen3.8-27B --template q8_0.gguf \
  --encoder search-neg --imatrix imatrix.gguf --plan 'mlp/down=q4' --device cuda --out search.gguf
python3 tools/eval/gguf_eval.py score --model search.gguf --name search --base-dir kld-base \
  --out kld-results --llama-perplexity <llama-perplexity>
```

The template must be a Qwen3.8-27B Q8_0 GGUF with llama.cpp's tensor names and Q8_0 layout (the
GDN `a`/`b` projections and the MTP layer keep its values). `--encoder` is `rtn`
(`grouped_absmax`), `search` or `search-neg` (`grouped_search` without or with signed scales).

`--disjoint` replaces the sliding windows with back-to-back windows of `--context` tokens, each
scored from its second token on, and drops a partial window at the end. That is the protocol of
the WikiText-2 perplexities quantization papers and model cards quote (GPTQ lineage: the test rows
joined by blank lines, 2,048-token windows). With the test split written to a file:

```bash
./build/apps/ninfer-perplexity models/qwen3_8_27b_gsq_rco_iq3_s.ninfer \
  --text wiki.test.joined.txt --context 2048 --disjoint --kv-dtype bf16
```

Past the model's native window `--rope-yarn` applies YaRN at factor `--context` / native, and
`--rope-yarn-factor F` at a fixed factor in `[1,4]` whatever the window is (default `1`, native
RoPE). `--rope-scaling-factor F` instead interpolates positions past
`--rope-scaling-original-context` (default: the native window) linearly by `F` in `[1,32]`, keeping
the angles of every position up to it. None changes the artifact or the default 4,096-token window;
long-context extrapolation is not a quality guarantee, so keep the factor fixed while comparing other
numerical settings. The report records all four settings.

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v4 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results; its execution configuration
records `rope_yarn`, `rope_yarn_factor`, `rope_scaling_factor` and `rope_scaling_original_context`.
