# HTTP serving

`build/apps/ninfer-serve` loads one v3 `.ninfer` artifact and exposes OpenAI-, Anthropic- and
llama.cpp-compatible HTTP endpoints over one resident NInfer Engine, or serves a catalog of
artifacts as a [router](#several-models-router), one Engine per loaded model.

## Start the server

See [CUDA synchronization](cli.md#cuda-synchronization) for the shared `NINFER_CUDA_SYNC` setting.

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype int8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

The command uses Qwen3.8-27B NVFP4. Each request has a 240,000-token logical ceiling. A shared
240,000-token Main Text KV pool serves admitted requests; either request may use the full capacity
when running alone, and two requests run concurrently when their complete reservations fit.

With `C=2` and two extra Device checkpoint slots, the process owns two active StateImage guarantees
plus a global pool of two Device-resident checkpoints. Eight pinned Host State slots and 8 GiB of
pinned Host KV retain inactive continuations under Device pressure. Active request capacity is two.

`--host-state-slots` and `--host-kv-mib` are independent allocations, so their sum — not
`--host-kv-mib` alone — is the Host RAM the process pins: a Host StateImage costs one
`memory.host_state_image_bytes` regardless of the prefix depth it resumes. `--host-cache-mib N`
replaces both with a single ceiling: the engine sizes the Host state pool from the checkpoint
inventory the capture path creates, `(2 + max-long-anchors-per-continuation) * private
continuations + shared prefixes` images, then, with `--auto-long-anchors` and a nonzero anchor
count, spends the remaining state headroom under the half-budget cap on **more long anchors per
continuation** — up to the count whose re-prefill gap still outweighs one StateImage — re-sizes the
pool for the grown count, gives Host KV the remaining bytes, and refuses to start if that state
footprint would exceed half the budget. A budget therefore never lowers the configured anchor
count, and the count it resolves is the one the frontend grid, the ResourceManager and the Program
all use; without automatic anchoring only client markers and the engine's endpoint and branch
anchors create anchors, so the count stays as configured. Both unit costs and the derived split are reported in the `server_start` memory
ledger.

Other artifacts use the same command shape with their own path. For 35B-A3B DFlash, replace the MTP
selection with `--spec dflash --draft-tokens 7 --lm-head-draft`. Qwen3.8-27B
artifacts with DFlash2 companion weights also support `--spec dflash2 --draft-tokens 7`, with
`--lm-head-draft` optional. DFlash2 accepts draft counts 1..15 and supports the same sampling,
concurrency, prefix reuse, and image/video request surfaces. It may remain combined with
`--vision`.

When `--model-id` is omitted, the server advertises and accepts the artifact's `metadata.name`,
falling back to its architecture name when no name is stored. An explicit `--model-id` is a public
HTTP alias override and does not select or alter model execution.

Vision is disabled by default: its weights and Vision-specific unified-workspace extent are not
allocated, and media requests and token-count requests fail with HTTP 400 `vision_disabled`. Add
`--vision` when the server must accept image or video input. Speculative residency is likewise
frozen by `--spec mtp|dflash|dflash2` and `--draft-tokens`; omitting `--spec` loads no speculative backend.
`--lm-head-draft` additionally loads the optimized proposal head. DFlash on 35B-A3B and DFlash2 on Qwen3.8-27B can be combined
with `--vision`; each accelerates generated-text decode after multimodal prefill, while Vision encode
and prefill remain outside speculative acceleration. A later request cannot enable a capability
omitted at startup. The artifact need only contain the Text backbone and the optional components
selected for this process.

### Adaptive MTP

With `--adaptive-mtp`, `--draft-tokens K` is the most drafts an MTP round verifies; every round
still proposes `K` for the next one. Each request tracks how far its drafts survive, position by
position, with a fast and a slow moving average; the Engine prices each verification width from
the round times it has measured at that width and batch size (a width not yet run is priced from
a relative cost shape scaled by the widths that have) and picks the width with the most expected
committed tokens per second over this round and the next. A width changes only after it has won
several rounds, and a width is probed upward when every row's drafts have kept surviving. While
three drafts are ready the width stays at three or more, so the Engine captures CUDA Graphs for
widths `min(3, K)..K`: the graph memory, and the startup capture time, grow with the number of
widths. Committed tokens are still the target's greedy choices, but attention splits its
reduction by the verify width, so a near-tie can resolve differently at another width, as it does
between two fixed `--draft-tokens` settings. The per-request log reports the rounds run at each
width and the width changes.

### MTP attention window

Each MTP draft step attends over the whole history of its cache, so at a long context the draft
reads as many keys as one target attention layer, once per draft. With `--mtp-attention-window N`
the draft head attends to the first 64 keys and the newest `N` (at least `--draft-tokens + 1`,
rounded out to whole 64-key pages) before its first query, which keeps its read constant as the
context grows. Only the drafts see the window: target verification attends to everything and
alone decides which tokens are committed, so greedy output is unchanged and only the acceptance
rate can move. The Engine rebinds the draft's page table to those pages and shifts its positions,
so the attention kernels are the unwindowed ones. `server_start` records the window.

### Vision residency

`--vision` keeps the Vision tower, its encode workspace and the item handoff resident for the
process lifetime. `--vision-residency overlay` removes that cost on memory-tight cards: the tower
lives in pinned host memory, the sequence plan reserves nothing for Vision, and each image is
encoded inside a bounded window whose device memory is borrowed for the duration of the encode.

A window is funded from free KV pages when they cover it. Those pages hold nothing, so nothing is
copied, the text weights stay mapped, and the encode runs on its own stream while other lanes keep
decoding: the lane that owns the image simply yields its prefill units until the encode completes.
The pages are out of circulation while the loan is open, so the admission capacity shrinks with it
and no request is ever admitted into memory that has been lent away.

When free pages cannot cover the window, it falls back to the evict-ranked tail of the text weights
(lm_head, token embedding, draft head, MTP head), restored from a pinned mirror before the prefill
unit ends. That window is exclusive: the borrowed weights are unmapped, so nothing else runs until
it closes. The fallback is always available, so a vision request never fails for lack of memory,
and a KV cache smaller than one window simply uses it for every image.

Either way the window streams the tower through two layer slots, lands the embeddings in pinned
memory, and the prefill then uploads only each chunk's embedding columns. A follow-up turn over the
same image reuses the prefix and opens no window. Embeddings are produced by the same kernels on the
same bytes, so completions are identical between residencies. `--vision-max-merged N` bounds one
item's merged tokens (media above it is downscaled at preprocessing) and sizes the window. The
request log line reports `overlay=<windows>x<conc|excl|mixed> <ms> (evict <MiB> <ms>, restore <ms>,
staged <MiB>)` and the JSON record carries `vision_overlay`, including `exclusive_windows`.

`--vision-residency cpu` (or `--vision-cpu`, which also sets `--vision`) keeps no Vision memory on
the device at all: the tower is decoded to FP32 host memory at load (about 1.7 GiB for the Qwen3.6
tower) and each image is encoded on CPU threads, then uploaded a chunk's columns at a time like an
overlay result. The encode of the next image runs beside other lanes' decode, and the lane that
owns it yields its prefill units until it completes, but it is slow: on a 12-core desktop CPU a
512x512 image takes seconds, and the cost grows with the square of the patches, so this residency
caps `--vision-max-merged` at 256 unless it is given. One encode runs at a time across the process.
The CPU encoder computes in FP32 where the device encoder rounds activations to BF16, so embeddings
agree closely but not bit for bit. It reads the tower's weights in the official artifacts' grouped
formats as well as BF16, FP8 and NVFP4; a projection stored with a Hadamard rotation or an input
gather is refused at load.

## Several models (router)

Started with `--models-dir` or `--models-preset` and no artifact path, the server is a router in
llama.cpp's sense: it serves every model of a catalog at one address, loads each on demand, and
keeps at most `--models-max` loaded at once (default 1, 0 for no limit). Where llama.cpp's router
starts one server process per model, every model here is an Engine in the one process.

```bash
./build/apps/ninfer-serve --models-dir models --models-max 1 --model-suspend \
  --max-context 65536 --sleep-idle-seconds 600
```

- `--models-dir DIR` serves every `.ninfer` directly in `DIR` under its file stem, and every
  subdirectory holding exactly one `.ninfer` under the subdirectory's name.
- `--models-preset FILE` is an INI file in llama.cpp's preset format. `[*]` holds options for every
  model; any other section is a model: `model = PATH` (or `artifact`), `alias = a, b` for further
  names, `load-on-startup = true`, and any serve option without its leading dashes
  (`kv-capacity = 65536`; a flag is `model-suspend = true`). A section named after a model of
  `--models-dir` configures that model and needs no path. Comments start with `;` or `#`.
- Serve options given on the command line apply to every model and win over the preset; a model's
  own section wins over `[*]`. Each model is configured exactly as a single-model server started
  with those options would be, except that the host, port and API key are the router's.

A request names its model in the body's `model` field (POST) or in the `model` query parameter
(GET); a request that names none gets the most recently used loaded model. A model that is not
loaded is loaded first, or woken when it sleeps, unless `--no-models-autoload` is set (per request,
`?autoload=0|1` overrides it; `--models-autoload` names the default); the request then fails with
503 `model_not_loaded`. An unknown name
is a 404 `model_not_found`, and a failed load a 503 `model_load_failed`.

To make room, the least recently used model without requests in flight goes to sleep when it was
started with [`--model-suspend`](#model-suspend), and is unloaded otherwise. A sleeping model keeps
its Engine, host caches and retained conversations and gives back its device memory; it wakes in
about two seconds where a cold load takes over ten (27B, RTX 3090: 1.8 s against 11.6 s), and its
retained conversations reuse their prefixes as before. A model with requests in flight is never put
to sleep or unloaded under them: the request that needs the room waits for them to finish (up to
ten minutes, then 503). `--sleep-idle-seconds N` puts a model idle for N seconds to sleep (it needs
`--model-suspend`) and `--unload-idle-seconds N` unloads one; both apply to a single-model server
too. Only requests count as use: polling a model's state (`/props`, `/metrics`, `/slots`) does not
keep it awake. A model whose Engine fails Engine-wide is unloaded and loads again on its next request
(see [Stop the server](#stop-the-server)).

| Method and path | Behavior |
|---|---|
| `GET /models` | every catalog model: `status.value` (`unloaded`, `loading`, `loaded`, `sleeping`, `unloading`), `status.args` (its serve arguments), `status.failed` and `status.error` after a failed load, `aliases`, `path`; a loaded or sleeping model adds the fields of [`GET /v1/models`](#models) |
| `POST /models/load` | `{"model": "<id>"}`: load the model, or wake it when it sleeps |
| `POST /models/unload` | `{"model": "<id>"}`: unload it once its requests have finished |
| `POST /models/sleep` | `{"model": "<id>"}`: NInfer's addition, put an idle model to sleep; `409 model_busy` while it has requests |
| `GET /models/sse` | one event per status change: `{"model": "<id>", "event": "model_status", "data": {"status": "<state>"}}`, with `failed` and `error` after a failed load |

Read-only endpoints (`/props`, `/slots`, `/v1/load`, `/metrics`, `/v1/models/{id}/residency`) report
a sleeping model as it is and neither load nor wake it. Neither do `/tokenize`, `/detokenize` and
`/apply-template`, which run on the host. The `/models` routes exist on a single-model server too,
whose catalog is its one model.

## Structured output

NInfer uses the vendored XGrammar v0.2.7 C++ compiler and token matcher to constrain final
response content. This requires no model conversion or additional weights. Ordinary decoding,
MTP, DFlash, and DFlash2 share the same target sampling contract, including CUDA Graphs,
streaming, prefix reuse, and mixed constrained/unconstrained concurrent requests.

It is opt-in: start the server with `--structured-output` (the CLI's `--json` and `--json-schema`
enable it for their run). That reserves one grammar mask plane per draft position and lane, and a
DFlash or DFlash2 round then carries a host grammar stage even when no request is constrained.
Without it the server keeps the default path unchanged and refuses a constrained request with HTTP
400 `response_format_not_supported`; with `--unconstrained-response-format` it generates such a
request unconstrained instead, for clients that send a JSON format on every request and parse the
answer themselves.

Chat Completions accepts:

```json
{
  "model": "your-model-id",
  "messages": [{"role": "user", "content": "Give the city and temperature as JSON."}],
  "max_tokens": 128,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "weather",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "temperature": {"type": "number"}
        },
        "required": ["city", "temperature"],
        "additionalProperties": false
      }
    }
  }
}
```

Use `response_format: {"type":"json_object"}` for any JSON object. JSON Schema mode can also
constrain other root types. Responses uses the flat form
`text.format: {"type":"json_schema","name":"weather","strict":true,"schema":{...}}` and
reports that format in its Response object. Anthropic Messages accepts
`output_config.format: {"type":"json_schema","schema":{...}}`.

Supported schema constraints are `type`, `properties`, `required`, `additionalProperties`,
`items`, `prefixItems`, `minItems`, `maxItems`, `minLength`, `maxLength`, `enum`, `const`,
`anyOf`, `pattern`, asserted `format` (`date`, `date-time`, `time`), `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `$defs`, `definitions`,
and local fragment `$ref` (including recursive schemas).
Annotations `$schema`, `title`, `description`, `default`, `examples`, and `$comment`
do not impose generation constraints. Other keywords are rejected with HTTP 400: this includes
`multipleOf`, other formats, `oneOf`, `allOf`, and conditionals.
Unsupported `uniqueItems` item domains and intersections are also rejected.
Object required lists contain unique string names. A required name forbidden by
additionalProperties is rejected; required additional names in open objects inherit the
additional-property value schema. Schema cache identities preserve complete serialized
values and declaration order, including annotation-like property names and literal data.
A const/enum intersection is supported when the constant is an enum member;
incompatible intersections are rejected. Value equivalence preserves object-key
semantics, array order, numeric equality and boolean/number distinctions.
Common `type` and `required` assertions beside `anyOf` are distributed to every
branch as intersections; branch-specific assertions remain intact. Conflicting
branches are removed only when impossibility is proved. Other unimplemented
applicator intersections still fail compilation.
`maxProperties` is an asserted object upper bound. A `$id` is accepted as harmless
resource metadata only if its resource subtree contains no schema references;
scoped reference resolution through resource identifiers remains unsupported.
`uniqueItems: true` uses request-owned semantic token masks for unrestricted
string arrays and finite string enum/const arrays. Equivalent JSON escape spellings
are compared as decoded strings; finite domains cannot repeat a value or continue
after exhaustion. Arrays limited to zero/one items are inherently unique. Item
pattern/format/length intersections and ambiguous union contexts that cannot be
proved safe are rejected; arbitrary unique object/number arrays are not advertised.
Semantic state follows accepted tokens, independent forks and speculative draft
columns, including reasoning-to-JSON boundaries. Ordinary schemas have no semantic
vocabulary-filter pass.
Pattern/format require explicit string type. Patterns use JSON Schema search semantics and
JSON-aware escaping; unsupported regex syntax fails compilation. Anchors must occur at the
pattern boundaries; anchored top-level alternation requires an explicit group.
Dot and whitespace classes follow ECMA262 Unicode semantics. Backreferences, boundary
assertions, inline flags, lookarounds and unknown alphabetic escapes are rejected.
Whitespace class escapes adjacent to a range hyphen, spaced repetition counts and
stacked/lazy quantifiers are unsupported. Unicode escapes require scalar values or
valid paired surrogate spellings; an unsupported branch fails the whole schema.
Pattern/format cannot be combined with each other or length constraints. Temporal formats
validate Gregorian month lengths and leap years, positive four-digit years, and RFC 3339
timezones; leap-second spellings are outside the generated subset.
Numeric bounds require an explicit `integer` or `number` type and finite values within
`+/- (2^53-1)`. Integer bounds must be whole numbers. Bounded numbers use ordinary decimal
notation with at most six fractional digits; a range with no representable value is rejected.
Reference-bearing `$id` scopes and external references are rejected. An explicit `$schema` must be JSON Schema 2020-12
or draft-07. Local references use `#` or literal object paths such as `#/$defs/node`;
escaped or empty path segments are rejected. Strings accept Unicode scalar values via raw UTF-8, short JSON escapes or Unicode
escapes (including valid surrogate pairs); lone surrogates and malformed UTF-8 are
rejected. Length bounds count decoded Unicode scalars, independent of escape spelling. Nonnegative length/item bounds
must fit a signed 32-bit integer. `$ref` cannot have sibling constraints; `anyOf` permits
the proven common `type`/`required` intersections described above. `const` and `enum`
allow a matching single `type` declaration and compatible const/enum intersections,
but no other sibling constraints. Move other assertions into the referenced schema
or each union branch. Missing `additionalProperties`
and `items` retain JSON Schema defaults unless the schema is strict (see below). Properties are emitted in schema declaration order;
this is a valid subset of the requested schema. Whitespace between JSON elements is bounded to
eight characters per run to prevent whitespace-only generation loops. Additional-property key spellings are restricted
where necessary to prevent escaped aliases from overwriting declared typed properties.
`strict:false` does not disable enforcement. With `strict:true` (Chat `json_schema.strict`,
Responses `text.format.strict`), an object schema without `additionalProperties` admits only its
declared properties, as OpenAI's strict mode requires; Anthropic `output_config.format` schemas are
always strict.

Structured responses default to thinking disabled when no reasoning mode or budget is requested.
Explicit thinking and thinking budgets are supported: reasoning is returned in its normal channel,
and the first reasoning-close delimiter transitions to constrained final content. Active tools are
also supported. After reasoning, a response can emit native Qwen tool calls or the final JSON value.
The schema applies to final content, not reasoning or tool arguments. Tool responses have their normal
tool-call finish reason and need not contain a JSON content value. Tool definitions and thinking
settings may remain enabled on later requests after tool results are supplied.

Serving also exposes the requested format and schema to the model as a leading instruction,
asking for raw JSON without Markdown fences and retaining tools for information gathering.
This helps the model choose the intended final-answer branch when tools remain enabled; the
token grammar enforces validity independently. Tool selection and termination remain model
decisions under `tool_choice:"auto"`; use `tool_choice:"none"` when no further tools are needed.
Requests without an explicit response format receive no JSON format instruction or JSON body grammar.

Tool requests use constrained decoding by default, independently of `--structured-output`.
`tool_constraints:"basic"` restricts native Qwen calls to declared/allowed tool names and complete
terminal call sequences. Non-strict parameter values retain best-effort normalization.
`tool_constraints:"auto"` opts out of this automatic basic grammar only when the request has no
strict tools, required/named/allowed selection, or parallel-call restriction.

`strict:true` enforces the supported JSON Schema subset for tool arguments. Its root must resolve
to one `type:"object"` with `additionalProperties:false`. Required fields, nested JSON objects and
arrays, enums, raw string patterns/lengths, and numeric bounds are constrained during sampling.
Top-level parameters mixing string and non-string types are rejected because Qwen renders strings
as raw text and other values as JSON. The pinned compiler rejects unsupported assertions instead
of weakening them; this includes `uniqueItems` in strict tool arguments, inexact numeric schema
literals, and general schema intersections. Singleton root `allOf` and local root references are
supported. Final JSON responses retain the existing unique-string validator.

OpenAI `required` supports multiple declarations; a named OpenAI choice produces exactly one call.
Anthropic `any` requires at least one call, and a named Anthropic choice allows repeated calls to
that tool unless parallel use is disabled. `none` blocks tool calls. Allowed selections and `none`
retain the original prompt declarations and their order so KV reuse is independent of selection.
Thinking remains available with required/named choices. With a JSON response format, `auto` permits
either the JSON final body or tool calls; `required` permits calls, and `none` permits the JSON body.

Active constraints require default EOS stops and decoded output. Assistant continuation must be
a legal grammar prefix. Cancellation and truncation publish only completed tool calls; partial
calls are discarded without repair or a synthetic error-tool call. Public C++ callers use
`RequestOptions::tool_choice`; JSON formats also use `ExecutionOptions::structured_output`.

Only normal completed final-content responses guarantee a complete JSON document satisfying the supported
schema. Token/context limits, cancellation, transport failure, or generation errors can leave a
partial document; inspect the finish reason or Responses status before parsing it as complete.
SSE content deltas are ordinary partial JSON bytes; concatenate them before parsing. Constraints
guarantee format, not factual accuracy or semantic task success.

Masks are applied before target top-k/top-p/min-p filtering at **every** speculative position,
including correction and bonus tokens. Unconstrained draft distributions remain unchanged;
accept/reject and residual sampling use the constrained target distribution. Per-request grammar
state advances only with Engine output commit, and is never restored from a prompt/KV cache.
DFlash/DFlash2 CUDA Graphs contain a host matcher node between draft generation and verification;
this adds a CPU synchronization point and mask transfers per round. No speedup claim is implied.

## Stop the server

Press Ctrl+C twice to stop the server. The first press only shows a prompt on the console's bottom
line, beneath the statistics panel: `Press Ctrl+C again within 5 s to save the prefix cache and
close` (`… to close` without `--prefix-cache-file`). Without an interactive console, the prompt is
logged as a warning. A second press within those 5 seconds stops the server; otherwise the prompt
disappears and serving continues. Ctrl+Break (also what a parent process sends with
`GenerateConsoleCtrlEvent`), closing the console window and `SIGTERM` stop at once.

Stopping refuses new connections, and every running and queued request ends with an HTTP 503
`service_unavailable` error, or an error event on a stream that has already started. The Engine
then saves the prefix cache when `--prefix-cache-file` is set, while the bottom line shows
`Closing: saving the prefix cache | Press Ctrl+C again to exit without saving`. The process exits
after logging `server stopped` and the save result. One more Ctrl+C during the stop exits at once
(exit code 130) without saving: an unfinished save stops, its temporary file is deleted, and the
previous file is kept. Ctrl+C with console text selected only copies the text. Windows ends the
process about 5 seconds after its console window is closed, so stop with Ctrl+C when a large Host
tier must be saved. Before the server is ready, one Ctrl+C ends startup at once.

An Engine-wide failure (an error the Engine's worker cannot attribute to one request) fails every
queued request and refuses every new one, and the Engine never recovers from it. A single-model
server then stops on its own and exits with status 2, distinct from startup failures (1), so a
supervisor such as a container restart policy reloads the model instead of keeping a dead endpoint
up. A router instead unloads the failed model once its requests are gone and reports it as failed in
`GET /models`; its next request loads it again.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | process health |
| `GET /v1/load` | serving capacity, current load, and monotonic token counters (see [Load](#load)) |
| `GET /metrics` | Prometheus text with llama.cpp's `--metrics` series plus NInfer's (see [Metrics](#metrics)) |
| `GET /stats` | the `/v1/load` snapshot plus the ingress peak and every Engine counter since startup (see [Stats](#stats)) |
| `GET /slots` | per-cell occupancy of the private context cache (see [Slots](#slots)) |
| `POST /slots/{id}?action=save\|restore\|erase` | save a retained session to a file, restore one, or evict one (see [Slots](#slots)) |
| `GET /props` | llama.cpp-shaped server properties: default sampling, context, lanes, modalities (see [Props](#props)) |
| `GET /v1` | endpoint index for the announced API base: the model alias and this table |
| `GET /v1/models` | configured OpenAI model alias, effective context limit (`max_model_len`, also as `context_window` and `context_length`), input modalities (`modalities.vision` and an OpenRouter-style `architecture`), and a llama.cpp-compatible `meta` object (see [Models](#models)) |
| `GET /v1/models/{id}` | lookup of the configured alias with the same fields |
| `GET /v1/models/{id}/residency` | whether the model is resident or suspended, what a suspend releases and holds (see [Model suspend](#model-suspend)) |
| `POST /v1/models/{id}/suspend`, `POST /v1/models/{id}/resume` | give the model's device memory back while idle, and take it again (`--model-suspend`) |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/completions` | OpenAI legacy completion of a raw prompt (see [Raw-prompt completion](#raw-prompt-completion-and-the-tokenizer)) |
| `POST /completion`, `POST /completions` | llama.cpp's native completion of a raw prompt |
| `POST /tokenize`, `POST /detokenize` | the model's tokenizer, in llama.cpp's shape |
| `POST /apply-template` | the prompt a chat request renders to, as text, without generation |
| `POST /v1/rerank`, `POST /rerank` (and `/reranking`) | documents ranked by relevance to a query, scored by the model (see [Rerank](#rerank)) |
| `GET /models`, `POST /models/load\|unload\|sleep`, `GET /models/sse` | llama.cpp's router API: the catalog and its lifecycle (see [Several models](#several-models-router)) |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

Anthropic SDKs append `/v1/messages` to their base URL, so their base URL is
`http://127.0.0.1:8080`, while OpenAI SDKs take `http://127.0.0.1:8080/v1`. A client given the
OpenAI-style base URL requests `/v1/v1/...`; every `/v1/...` path above also answers under that
doubled prefix, with the same behavior, errors, request IDs, and log endpoint names.

Every OpenAI-compatible response carries a unique `x-request-id` header, including streaming and
error responses. Anthropic endpoints use their separate `request-id` contract.

All three generation SSE endpoints emit the standard `: keep-alive` comment after five seconds
without a protocol event. The comment is transport-only: SSE clients ignore it, and it does not
change generated text, event ordering, usage, stored Responses, or request logs. Anthropic Messages
follows each comment with the protocol's `ping` event, which clients skip: a client that abandons a
request whose first event is late does not count comments, so a queued or long-prefilling request
would otherwise be dropped by it. Accepted connections also use TCP keepalive and a 15-second
retransmission limit (`TCP_USER_TIMEOUT` on Linux, `TCP_MAXRTMS` on Windows); together with the
heartbeat,
a dead or unacknowledging peer is normally cancelled within about 20 seconds, including while the
request is waiting or prefilling. A peer whose TCP stack remains connected and acknowledges data
cannot be distinguished from a reading application; proxies must close their upstream NInfer
connection when the downstream client disappears.

The HTTP transport is thread-per-connection, and a worker stays with a connection for its whole
life, including the idle window between keep-alive requests. The pool therefore reserves one base
worker per admissible request (`max_concurrency + max_pending_requests + 1`) and grows on demand up
to 64 further workers for connections that are merely open; the extra workers retire once idle.
Without that headroom a client-side connection pool of otherwise idle sockets occupies every worker
and the server accepts real requests strictly one at a time.

### Props

`GET /props` serves clients that discover a llama.cpp server, such as llama.cpp's WebUI. It is
read-only and fills only the llama.cpp fields NInfer can state truthfully:

```json
{"default_generation_settings": {"n_ctx": 65536, "speculative": false,
   "params": {"n_predict": -1, "max_tokens": -1, "temperature": 1.0, "top_k": 20, "top_p": 0.95,
              "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "seed": -1}},
 "total_slots": 1, "model_alias": "qwen3.8-27b", "model_path": "models/qwen3_8_27b.ninfer",
 "modalities": {"vision": false, "audio": false},
 "endpoint_slots": true, "endpoint_props": true, "endpoint_metrics": true,
 "cors_proxy_enabled": false}
```

`n_ctx` is `--max-context` and `total_slots` is `--max-concurrency`. `n_predict` and its alias
`max_tokens` are the [default output limit](#default-output-limit): `-1`, llama.cpp's "no fixed
cap", when it is derived per request from the prompt and lane share, or the `--default-max-tokens`
cap. The sampler is the loaded model's preset for the default thinking mode (thinking unless
`--no-thinking`) under the process sampling flags and `--greedy`; request fields still override it
per request. `seed` is `--seed`, or `-1` when requests draw a fresh random seed. `model_alias` is the
public model id and `model_path` the artifact path the server was started with.
`cors_proxy_enabled` reports `--webui-mcp-proxy` (see [WebUI](#webui)). There is no `build_info`,
`chat_template`, or writable `POST /props`.

### Startup readiness

The server binds its port and starts accepting connections before the Engine has loaded weights
and finished warmup, so a port clash is reported in milliseconds rather than after loading. On
Linux the listener sets `SO_REUSEADDR` beside cpp-httplib's `SO_REUSEPORT`, so connections left in
`TIME_WAIT` by a stopped NInfer, or by another server that sets only `SO_REUSEADDR` (llama.cpp,
uvicorn), do not block the next bind of the port for about a minute. Every
route -- including `/health`, `OPTIONS`, and requests that would otherwise be unauthenticated --
answers `503` with `Retry-After: 2` until warmup completes, in the target's own error shape:

```json
{"error":{"message":"The model is still loading. Retry shortly.","type":"service_unavailable","code":"model_loading"}}
```

`POST /v1/messages` and `/v1/messages/count_tokens` receive the Anthropic envelope instead, with a
`request_id` field and a `request-id` header, matching every other error response on those
endpoints:

```json
{"type":"error","error":{"type":"api_error","message":"The model is still loading. Retry shortly."},"request_id":"req_..."}
```

A readiness probe should poll `GET /health` (or any endpoint) and expect `503` until the model is
ready rather than treating an accepted TCP connection as a signal of readiness.

After startup, a worker out-of-memory failure (and, with `--recover-invariant-failures`, a broken
internal invariant) fails only the requests that were running or being admitted, waits for the
device, releases and if necessary rebuilds the context cache, and keeps serving the queue. Each
recovery is counted as a top-level `engine_recoveries` counter on the request log's `throughput`
event and as `ninfer:engine_recoveries_total` on [Metrics](#metrics). `GET /health` turns `503` for
good after any other worker failure, when a recovery cannot complete, or after eight consecutive
recoveries, and then the server needs a restart.

### Load

`GET /v1/load` is a cheap, pollable snapshot for load balancers and gateways that schedule across
several servers. It requires the API key when one is configured, answers `503 model_loading` until
warmup completes like every other route, and reads only the Engine's already-published runtime
counters and the ingress count, so polling it does not wait on or delay the GPU executor.

```bash
curl http://127.0.0.1:8080/v1/load -H 'Authorization: Bearer local-secret'
```

```json
{
  "object": "ninfer.load",
  "model": "qwen3.8-27b",
  "uptime_seconds": 812.4,
  "capacity": {"max_concurrency": 4, "max_pending_requests": 32, "max_admitted_requests": 36,
               "max_context": 65536, "kv_capacity_tokens": 131072, "kv_capacity_pages": 2048,
               "kv_page_tokens": 64, "device_state_slots": 8, "host_state_slots": 24,
               "host_kv_bytes": 17179869184},
  "requests": {"admitted": 6, "running": 4, "prefilling": 1, "decode_ready": 3, "waiting": 2,
               "materializing": 0},
  "occupancy": {"device_main_kv_pages": 1500, "device_main_kv_tokens": 96000,
                "device_state_slots": 5, "host_state_slots": 7, "host_kv_bytes": 2147483648},
  "counters": {"computed_prefill_tokens": 48200113, "committed_decode_tokens": 6120452,
               "reused_prompt_tokens": 30911840, "decode_rounds": 861307,
               "decode_row_rounds": 2448180}
}
```

- `uptime_seconds` counts from the moment the server became ready, not from process start.
- `capacity` is fixed once the Engine is ready. `max_admitted_requests` is
  `max_concurrency + max_pending_requests`, the ingress bound described in
  [Execution behavior](#execution-behavior); `kv_capacity_tokens` is the resolved page-aligned Main
  KV pool and `kv_page_tokens` its page size.
- `requests.admitted` counts requests holding ingress capacity, from preparation until the response
  is released; a new generation request is rejected with `server_overloaded` (HTTP 429, or 529 on
  Anthropic endpoints) when it would exceed `max_admitted_requests`. `running` counts occupied execution lanes (at most
  `max_concurrency`), of which `prefilling` is the lane that owns the staged prefill and
  `decode_ready` the lanes in the decode batch. `waiting` counts requests submitted to the Engine
  FIFO that have not been admitted to a lane, including those held back until their KV entitlement
  fits.
- `occupancy` reports current Main KV pages (and tokens), Device and Host StateImage slots, and Host
  KV bytes in use. Main KV occupancy includes retained reusable prefixes, which the resource planner
  may evict under pressure, so a full pool does not by itself mean new requests will wait.
- `counters` are monotonic since startup; derive rates by differencing two polls.
  `computed_prefill_tokens` excludes prefix-reused prompt tokens (reported separately in
  `reused_prompt_tokens`). `committed_decode_tokens` counts tokens committed by decode rounds and
  excludes each request's first token, which prefill emits; with speculative decoding a round commits
  several tokens per row. `decode_row_rounds` is the sum of decode batch sizes over `decode_rounds`.
- Gauges and counters come from the snapshot the Engine publishes at execution boundaries, so they
  can trail the instant of the poll by up to one boundary.

### Stats

`GET /stats` is the `/v1/load` report (`object` `ninfer.stats`) with `requests.peak_admitted`, the
most requests admitted at once since startup, and the Engine's counters since startup in the shape of
the request log's `throughput` record (see [Structured request log](#structured-request-log)):
`tokens`, `throughput_tokens_per_second` averaged over the uptime, `scheduler`, `decode_batch`,
`host_work` and the `context_cache` counters and gauges. `queue` holds `depth`, the number of
requests waiting for admission, and `entries`: the first 16 of them in submission order, each with
its Engine `request_id`, `position` and `wait_seconds`, refreshed at least once a second while
requests wait. It needs the API key like `/v1/load` and, like it, reads only published snapshots.
Dashboards poll it, or the same route on `--stats-port`.

### Model suspend

With `--model-suspend`, an idle server can give its device memory back to the GPU without exiting
and take it again later, for instance to let another program use the card for a while. Every device
allocation the model owns -- weights, KV cache and recurrent state, workspace, on every device of a
`--devices` split -- sits at a fixed address whose physical memory a suspend releases; the CUDA
Graphs, block tables and Host caches stay as they are. A resume maps fresh memory at the same
addresses, uploads the weights and copies the retained state back, so conversations retained before
the suspend reuse their prefixes afterwards exactly as before. It needs CUDA virtual memory
management on every device and cannot be combined with `--wddm-evictable-budget`.

```bash
curl -X POST http://127.0.0.1:8080/v1/models/qwen3.8-27b/suspend -d '{}'
curl -X POST http://127.0.0.1:8080/v1/models/qwen3.8-27b/suspend -d '{"auto_resume": false}'
curl http://127.0.0.1:8080/v1/models/qwen3.8-27b/residency
curl -X POST http://127.0.0.1:8080/v1/models/qwen3.8-27b/resume -d '{}'
```

A suspend runs only while nothing is queued, admitted or running and answers
`409 model_busy` otherwise; it never cancels work. While suspended nothing executes. With
`auto_resume` (the default, `--no-auto-resume` changes it, a suspend body may override it), the next
generation request resumes the model and then runs, its wait counting toward the queue timeout;
without it requests fail with `503` and `GET /health` reports the server unavailable. A failed
resume (the device no longer has the memory, say) fails the requests waiting on it, leaves the
model suspended with its state intact and reports `last_error`; the next request or `resume`
retries. Session save, restore and erase resume the model first, under the same rule. Stopping a
suspended server resumes it to save the prefix cache and the disk tier when `auto_resume` is set;
otherwise it exits without saving what only the device held.

| Option | Effect |
|---|---|
| `--suspend-snapshot pageable` | default: the retained state goes to ordinary host memory allocated at suspend, sized to what is live (free KV pages are skipped), and freed after the resume |
| `--suspend-snapshot pinned` | a page-locked block sized for the whole persistent state is reserved at startup and kept: transfers at full bus speed, that much host memory for the life of the server |
| `--suspend-weights artifact` | default: a resume reads the weights again from the artifact, whose files the server keeps open |
| `--suspend-weights host` | a suspend copies the weights to host memory (in the snapshot memory): as much host memory as the weights while suspended, resume at bus speed |

`GET /v1/models/{id}/residency` returns `state` (`resident` or `suspended`), `auto_resume`,
`releasable_device_bytes`, `host_snapshot_bytes` and `host_reserved_bytes`, the suspend and resume
counts, the last of each with its duration and the bytes it moved (`state_bytes`, `weight_bytes`,
`artifact_read_bytes`), and `last_error`. Suspend and resume answer with the same object.

### Slots

A slot is one private context-cache cell: the place a finished conversation is retained so its next
turn reuses the cached prefix. There are `--max-private-continuations` slots. `GET /slots` lists them
like llama.cpp's endpoint and reads only published state, so it never waits on the GPU:

```json
[{"id": 0, "is_processing": false, "retained": true, "session_digest": "8c3f1e0a7b2d4c19",
  "checkpoints": [{"frontier": 1812, "session_digest": "51d0..."},
                  {"frontier": 2409, "session_digest": "e27a90c4d15b3f68"}],
  "n_ctx": 131072, "n_prompt_tokens": 2410, "n_prompt_tokens_cache": 2410, "speculative": true}]
```

A retained slot reports the session depth as both token counts, its session digest (FNV-1a 64 of
the token ids, 16 hex characters) and the checkpoints a later request can resume from, each with the
digest of its prefix. The endpoint checkpoint sits at the executed frontier, which is usually one
token short of the session: the last sampled token is recorded but not yet in the KV or the
recurrent state, so the next turn executes it first. A slot an
active request will publish into reports `is_processing` with that request's prompt and reused
tokens. Chat Completions responses carry the slot and digest a finished session was retained under
as top-level `id_slot` and `session_digest`, on the aggregate response and on the final streamed
chunk that carries timings.

With `--slot-save-path DIR`, `POST /slots/{id}?action=...` persists sessions across restarts and
evictions. Without it the route answers `501 slot_persistence_disabled`.

| Action | Body | Response |
|---|---|---|
| `save` | `{"filename": NAME, "if_digest": DIGEST?}` | `id_slot`, `filename`, `n_saved` tokens, `n_written` bytes, `session_digest`, `timings.save_ms` |
| `restore` | `{"filename": NAME}` | `id_slot`, `filename`, `n_restored` tokens, `n_read` bytes, `session_digest`, `timings.restore_ms` |
| `erase` | `{"if_digest": DIGEST?}` | `id_slot`, `n_erased` tokens (0 for an empty slot) |

`NAME` is 1-128 characters of `[A-Za-z0-9._-]`, may not start or end with a dot, and may not be a
Windows device name; files live directly in `DIR`. Names are case-insensitive: the server stores and
reports them lowercase, so one file never has two names. `if_digest` makes save or erase conditional on the slot
still holding that session, checked atomically with the operation. Restore replaces whatever the
slot held and makes the restored session an ordinary cache entry that any request with a matching
prefix reuses, including from its checkpoints. A snapshot restores only on a server with the same
model artifact, weight formats, KV dtype, speculative backend, draft tokens and draft head; DFlash
servers do not support persistence. Files are written to a temporary name and renamed, and end
with a checksum that restore verifies before it allocates anything. Slots belong to the
checkpoint catalog, so `--slot-save-path` and `--auto-save-evicted` are refused with
`--use-alt-prefix-caching` and `--no-prefix-reuse`. They are independent of the disk tier
(`--disk-kv-path`), which spills evicted conversations on its own, keyed by prompt digest.

Errors: `409 slot_busy` while the slot or any context-cache transaction is in use (retry),
`409 slot_session_mismatch` for a failed `if_digest`, `400 invalid_slot`, `invalid_action`,
`invalid_filename`, and `400 slot_save_failed`/`slot_restore_failed` for a missing, corrupt or
incompatible file or a slot with nothing to save. A failed restore leaves the slot empty.

With `--auto-save-evicted`, a session last saved to or restored from a file is written back to that
file, on a background thread, before an involuntary eviction destroys it. Continuing the
conversation keeps the binding, so the file tracks its newest turn. An explicit erase never writes.
A spill never replaces a file with a shallower copy of the session than the last save or restore
recorded, and at most two spills wait for the writer. An explicit save, restore or erase of a file
supersedes every spill of it still waiting (a restore first writes the waiting spills of the file it
reads, so it reads the newest state). The operational log reports each spill, skip or failure. Snapshots are uncompressed. Besides its KV pages, a session stores one recurrent-state
image per checkpoint it retains (endpoint, rewrite checkpoint, long anchors), about 150 MB each on
the 27B, so even a short session is a few hundred MB: a 39-token Qwen3.8-27B session saved as
295 MiB, with save and restore at about 0.3 s each on an RTX 3090.

### Metrics

`GET /metrics` serves the Prometheus text format, also on `--stats-port`. Like `/v1/load` it
requires the API key when one is configured and reads only already-published counters. The
`llamacpp:` series keep the names and meanings of llama.cpp's `--metrics`, so dashboards and
autoscalers built for it read this server unchanged. Token and time counters come from the Engine's
per-unit totals (every executed prefill unit and decode round, whichever request it served), so
they advance during a long request rather than at its completion, and their ratio is a real rate.

| Series | Type | Meaning |
|---|---|---|
| `llamacpp:prompt_tokens_total` | counter | prompt tokens computed by prefill; prefix-cache hits excluded |
| `llamacpp:prompt_seconds_total` | counter | prefill execution time |
| `llamacpp:tokens_predicted_total` | counter | tokens committed by decode rounds |
| `llamacpp:tokens_predicted_seconds_total` | counter | decode execution time |
| `llamacpp:prompt_tokens_seconds` | gauge | prompt tokens over prefill seconds |
| `llamacpp:predicted_tokens_seconds` | gauge | decoded tokens over decode seconds |
| `llamacpp:n_decode_total` | counter | decode rounds executed |
| `llamacpp:n_busy_slots_per_decode` | gauge | average requests per decode round |
| `llamacpp:kv_cache_usage_ratio` | gauge | occupied share of the device KV cache |
| `llamacpp:kv_cache_tokens` | gauge | tokens held in the device KV cache |
| `llamacpp:requests_processing` | gauge | requests holding an execution lane |
| `llamacpp:requests_deferred` | gauge | requests waiting for admission |
| `ninfer:requests_total` | counter | requests completed with an outcome |
| `ninfer:requests_failed_total` | counter | accepted requests that ended in an error |
| `ninfer:requests_rejected_total` | counter | generation requests rejected during preparation, one per `request_rejected` request-log event: overload, invalid or oversized prompt or media. Unparseable and oversized (413) HTTP bodies are not counted; failures after acceptance, including a queue timeout after submission, count in `requests_failed_total` |
| `ninfer:requests_admitted` | gauge | requests holding server ingress capacity, in any phase |
| `ninfer:prefix_cache_hit_tokens_total` | counter | prompt tokens of completed requests served from a cached prefix |
| `ninfer:reused_prompt_tokens_total` | counter | prompt tokens restored from the context cache instead of prefilled |
| `ninfer:draft_tokens_total` | counter | speculative draft tokens proposed for completed requests |
| `ninfer:draft_accepted_tokens_total` | counter | speculative draft tokens accepted for completed requests |
| `ninfer:context_cache_exhausted_requests_total` | counter | requests failed because the context cache had no placement for them |
| `ninfer:engine_recoveries_total` | counter | host-side worker failures the Engine survived instead of latching unavailable |
| `ninfer:uptime_seconds` | gauge | seconds since the Engine became ready |

### WebUI

A build configured with `-DNINFER_WEBUI_DIR=<dir>` compiles the files of a built WebUI into
`ninfer-serve` and serves them on `GET /`; any other `GET` path outside the API routes returns the
entry page, for the UI's client-side router. The static files need no API key, the API routes it
calls still do. `--no-webui` turns it off at startup. The build never downloads anything:
`scripts/fetch-webui.sh [tag]` unpacks one of llama.cpp's released WebUIs, which speaks this
server's `/props`, `/slots`, `/v1/models` and streamed Chat Completions, and prints the directory to
configure with.

```bash
./scripts/fetch-webui.sh
cmake -S . -B build -DNINFER_WEBUI_DIR=<the directory it printed>
```

The WebUI's MCP client cannot reach an MCP server from the browser: the Streamable HTTP transports
answer POST from a raw writer without CORS headers. `--webui-mcp-proxy` serves the `/cors-proxy`
relay that the WebUI's "Use llama-server proxy" option targets (the target in the `url` query
parameter, its headers prefixed with `x-llama-server-proxy-header-`) and reports
`cors_proxy_enabled` in `/props`. The relay streams, forwards to any `http://` host it is given
(`https://` is refused: there is no TLS client), and carries no API key, because the WebUI prefixes
its own `Authorization` along with every other header; enable it only on a trusted bind address.

### Models

`GET /v1/models` and `GET /v1/models/{id}` return the configured public OpenAI model alias
(defaults to the artifact `metadata.name`, overridable with `--model-id`) together with the
effective `max_model_len` (the `--max-context` ceiling). No client ecosystem agrees on one name for
the context limit, so the same value is also reported as `context_window` (Anthropic's Models API)
and `context_length` (OpenRouter, Ollama). Input modalities appear twice: `modalities.vision`
(whether `--vision` accepts images) and an OpenRouter-style `architecture` object,
`{"input_modalities": ["text", "image", "video"], "output_modalities": ["text"]}`, whose image and
video entries appear only with `--vision`. A `meta` object in the shape exposed by `llama.cpp`
completes it. The `meta` facts describe the registered artifact behind the alias:

| Field | Meaning |
|---|---|
| `n_vocab` | tokenizer token domain |
| `n_ctx` | configured per-request context ceiling (equal to `max_model_len`) |
| `n_ctx_train` | model native/training context |
| `n_embd` | model embedding width |
| `n_params` | total logical weight elements across the registered artifact tensors |
| `size` | encoded weight payload bytes of the registered artifact |
| `ftype` | encoded tensor formats of the artifact, joined with `+` when mixed |

`GET /v1/models/{id}` returns the same object for the single configured alias and a `404` for any
other id.

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history, plus legacy `function` history;
- string content and ordered text/refusal parts; adjacent parts are preserved without inserted
  separators, and empty wire content remains an empty turn;
- User `image_url` parts, tool-result `image_url` parts used by compatible clients, and the User
  `video_url` extension using HTTP(S) or data URIs; image `detail` may be omitted, null,
  `auto`, `low`, `high`, or `original`. All recognized levels use the same fixed `auto`
  preprocessing and startup `--vision-max-merged` budget; explicit levels are compatibility
  hints, not separate resolution or token-budget guarantees. Unknown strings and non-string
  values other than null are rejected;
- nonnegative `max_completion_tokens` and the legacy `max_tokens` spelling; zero performs prompt
  processing without generation; llama.cpp's `-1` ("no limit", the WebUI's default) and omitting
  both apply the [default output limit](#default-output-limit);
- `temperature`, `top_p`, presence/frequency penalties, and signed integer `seed`;
- the compatible `top_k` and `min_p` (`0..1`) sampler extensions; `top_k` is a non-negative
  integer, and a value above the sampler's 20-candidate domain is clamped to 20 rather than
  rejected (llama.cpp and Ollama default to 40), on every endpoint and in `post_thinking`; the
  request log records the effective value;
- the `post_thinking` extension object (see [post-thinking sampling](#post-thinking-sampling));
- up to four non-empty stop strings, applied to both reasoning and answer output;
- the `ignore_eos` benchmarking extension shared with vLLM, SGLang and llama.cpp: a boolean,
  default `false`, that drops the checkpoint's own stop tokens so generation runs to the output
  budget (`finish_reason:"length"`) unless a caller-supplied stop string or context capacity ends
  it first. Caller stop strings still apply; any other value type is rejected with 400. Output past
  the model's natural end is not meaningful text. `/v1/responses` and `/v1/messages` do not
  honor it;
- `n:1` and text-only `modalities`;
- `response_format` of type `text`, `json_object`, or `json_schema` (`name`, optional
  `description`, `schema` and `strict`); JSON formats need `--structured-output` and are enforced
  token by token, see [Structured output](#structured-output);
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`, optionally shaped by `--usage-chunk-choice` for strict client
  parsers;
- function tools and free-form `custom` tools, the latter served to the model as a
  single-string-input function under the caller's own tool name so callers that dispatch by name
  keep working;
- `tool_choice` `auto`, `none`, named functions/custom tools, `required` across one or more
  callable tools, and `allowed_tools`; selections are enforced during generation;
- `strict:true` for the supported tool-argument schemas, and `parallel_tool_calls:false` enforced
  by sampling at most one call;
- assistant tool-call history, tool-result messages, and legacy function-call history;
- the top-level `reasoning_effort` field, where the `default` and `auto` aliases resolve to the
  server-configured level;
- `enable_thinking` and `preserve_thinking`, either at top level or in
  `chat_template_kwargs`;
- the `graft` extension selecting a [prompt graft](#prompt-grafts);
- the `thinking_budget` extension, a positive per-request [thinking cap](#openai-chat-completions);
- Assistant `reasoning_content` and `reasoning` history aliases.

Options whose observable behavior the Engine cannot provide are rejected when they request that
behavior. This includes JSON constrained output on a server without `--structured-output` (unless
it runs with `--unconstrained-response-format`), nonzero `logit_bias`, audio/file input or audio
output, web search, moderation, low/high verbosity, stored Chat Completions, and non-empty legacy
`functions`.
Each capability rejection identifies the affected field and the guarantee NInfer cannot provide.
The vLLM/llama.cpp constrained-decoding extensions (`grammar`, `structured_outputs`, `guided_json`,
`guided_regex`, `guided_choice`, and `guided_grammar`) are rejected explicitly instead of being
treated as unknown hints; structured JSON is requested through `response_format`.

Semantically neutral fields do not make an otherwise executable request fail. All-zero
`logit_bias`, `logprobs:false`, `verbosity:"medium"`, empty legacy tool controls,
text-only `audio` configuration, and `prediction` are accepted without changing Engine execution.
Metadata, user/safety identifiers, service-tier and prompt-cache hints are likewise advisory.
Unknown top-level fields are ignored.

A string `name` on a `tool` message is accepted as an ignored, output-neutral compatibility
extension for clients that mirror the function name onto tool results. It does not participate in
tool identity, prompt rendering, or output. Non-string values are malformed; non-empty names on
other message roles remain unsupported because they carry participant identity that the loaded chat
template cannot represent.

For commonly generated OpenAI-compatible payloads, `repetition_penalty` is accepted only at its
neutral value `1`, and `mm_processor_kwargs` when empty or containing only null values. String-form
image/video URLs are also accepted.

Malformed protocol values return field-specific HTTP 400 errors. Invalid media sources, bytes, or
decoded content use `invalid_media`; remote fetch and timeout failures retain their dedicated
server-error codes. Failures in the normalized prompt contract use `invalid_prompt`; typed capacity
and availability failures retain their dedicated codes. Internal invariant failures are not
relabeled as client input errors.

The request `model` must equal the public model ID: the artifact `identity.model_id` by default, or
the explicit `--model-id` override. A Chat Completions request that omits `model` or sends it null
or empty, as the llama.cpp WebUI does, is served by the loaded model. Reasoning is returned separately as `reasoning_content`; answer
text remains in `content`.

Across Chat Completions, Responses, and Anthropic Messages, an explicit top-level tool-parameter
type controls conversion of Qwen's untyped parameter text. String-admitting values remain strings;
other explicitly typed values are decoded as JSON without coercion. NInfer does not validate
generated arguments against the full JSON Schema.

Tool calls are read in the Qwen form (`<tool_call>` around `<function=NAME>` with
`<parameter=NAME>` values) and in the forms agent harnesses write: `<function name="NAME">`,
`<invoke name="NAME">` and `<param name="NAME">`, bare or inside a `<function_calls>` container.
Every opening tag must be closed by its own kind.

String values keep tool-call markup and balanced nested parameter tags as text. The format has no
escape, so a closing parameter tag inside a value is read by what follows it: ordinary text makes it
value text, while `<` or the end of the output, after optional whitespace, makes it the end of the
value. A value therefore cannot contain a closing parameter tag followed by markup, nor an unmatched
nested parameter opener; either makes the call malformed.

Generated prose can quote tool-call markup before the real call. Unless the first marker's region
opens with a complete call, each marker is tried in order, and the first complete region that
consumes the rest of the response becomes the structured turn; quoted markup before it stays
ordinary content. Generated reasoning closes only at a `</think>` followed by format whitespace or
the end of the turn, so a marker the model quotes while reasoning (followed by punctuation or an
escaped newline) stays in the reasoning channel. At most sixteen markers are tried, the first one
included, so an output that keeps repeating the marker costs a bounded amount of parsing. A region that opens a function but does not parse is recovered: complete calls
before the failure are kept, a call that cannot be read is returned as a call to the reserved
`malformed_tool_call` tool, whose arguments tell the model what went wrong so that it retries, and
text after it is dropped.

Messages enter the selected template in their input order. The maintained Qwen templates keep
system/developer messages at their original positions. With `--assistant-prefill` a final assistant
message is an assistant prefill, as on `/v1/messages`: generation continues that turn in place
instead of opening a new assistant turn after it. The prefill must be text only, and a prefill whose
thinking is not explicitly disabled (top-level or `chat_template_kwargs` `enable_thinking`, or
`reasoning_effort: "none"`) is refused with `assistant_prefill_not_supported`.

Prompt-bearing JSON objects retain their received member order through request parsing and prompt
rendering, including tool schemas and historical tool inputs. Canonical model-origin tool arguments
retain that member order in aggregate and streaming responses, so an unmodified replay reconstructs
the same ordered tool call. NInfer does not canonicalize semantically equivalent JSON: if a client
reorders members, inserts defaults, or otherwise rewrites a tool object, the changed rendered input
does not match the model-held endpoint and can reuse only an earlier exact checkpoint.

`--chat-template FILE` selects a local Jinja template; by default, the server uses the template
stored in the artifact. See the [CLI guide](cli.md#text-input) for an example.

Clients send a wider effort vocabulary than the maintained Qwen templates accept (`low`, `medium`,
`xhigh`). This fork collapses `minimal` onto `low` and `high`/`max` onto `xhigh` before rendering,
so OpenAI and Claude Code requests that send `high` do not fail inside the template.

Control-token spellings quoted in message content, tool data or ordinary template kwargs are
encoded as text. Media placeholders come from the template and bind to actual image/video inputs.

`chat_template_kwargs` passes a JSON object to the template in Chat Completions, Responses and
Anthropic Messages. Values duplicated in typed request fields must agree. Null standard options
mean unspecified; other null values remain `none`. Messages, tools, generation mode and tokenizer
special tokens cannot be overridden through kwargs.

`--default-thinking-budget N` sets a positive default thinking-token cap for requests that start
in thinking mode. Non-thinking requests receive no cap. It may coexist with `--no-thinking`
because requests can explicitly enable thinking. A request overrides this default with a positive
integer: `thinking_budget` on Chat Completions and Responses (a NInfer extension, accepted at top
level), or `thinking:{"type":"enabled","budget_tokens":N}` on Anthropic Messages (a value of at
least 1024 and below `max_tokens`, per that protocol). A request that does not think receives no
cap, whatever it sends.

Add `--default-thinking-budget 512` to the startup command to cap model-origin thinking at 512
tokens for every thinking-enabled request.

At the cap boundary, Engine first honors a natural `</think>`, stop condition, cancellation, or
total output/context limit. If thinking remains open, it commits Qwen's canonical early-close
guidance and close marker to the same model sequence without sampling, streams the guidance as a
reasoning delta, and continues normal content or tool-call generation. Inserted tokens count in
completion usage and the request's `max_tokens`/`max_output_tokens` budget.

Near the end of the output window the early-close guidance may not fit. Let B be the requested
thinking budget, C the output the request can still produce (the smaller of its output limit and
the context left after the prompt, counting the final writable position), and R the number of tokens
that early close needs: the tokenizer-derived guidance and close marker plus one post-close model
token. R depends on the model's tokenizer and is not a fixed number; with
`--thinking-budget-message` the guidance is that message, so R grows with its token count. A request is never rejected
because C falls in B < C < B + R; Engine resolves it as follows:

- C <= B, or C - B >= R: the budget is B and early close is available, as above.
- C > R (and C - B < R): the effective budget is C - R, which is always below B. Early-close
  guidance and at least one post-close model token still fit.
- C <= R (and C > B): the budget cannot be enforced, so thinking runs to the output limit with no
  guidance inserted. The response can end inside the reasoning with empty content.

Logs report the requested budget, and the effective budget when it differs. A request that cannot
enforce its budget reports the requested value only. The server does not promise that the model will
emit nonempty content or a tool call after the marker.

For Chat Completions, `reasoning_effort: "none"` requests disabled thinking. The other standard
values (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`) reach the template unchanged.

Templates accept different subsets of those values: the official Qwen3.8 template raises for
anything but `low`, `medium` and `xhigh`. NInfer renders each value once when the template loads,
and a request for a value the template rejects renders with the nearest accepted value, a tie
rounding up: on Qwen3.8, `high` and `max` render as `xhigh` and `minimal` as `low`, so clients such
as Claude Code that send `high` work against it, while a template that accepts `high` receives it.
This applies to every endpoint, `chat_template_kwargs.reasoning_effort`,
`--default-reasoning-effort` and the CLI's `--reasoning-effort`. The values are tried on a
one-message chat; a template that rejects every value, or cannot render that chat at all (one that
requires a system message, for example), keeps the request's value and its error. Request logs
record the effort the client asked for.

`--default-reasoning-effort` sets the effort for requests that name none. It yields to everything
the request decides: a request effort replaces it, and a request that disables thinking, forces a
tool call or prefills the assistant turn is served as if it were unset. `none` turns thinking off
by default; any other value conflicts with `--no-thinking` and fails startup. Request logs record
only the effort a request named itself: one the server default filled in is logged as `null`.

`preserve_thinking` controls reasoning retention according to the selected template. Request
options override server defaults set with `--no-thinking` and `--preserve-thinking`. Unspecified
thinking, effort and preservation options use the template's defaults.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage. Aggregate and streamed usage include cached prompt tokens
and reasoning-token details; choices carry `logprobs: null` when log probabilities were not
requested, and aggregate assistant messages carry `refusal: null` because refusal output is not
supported.

#### Token log probabilities

`logprobs: true` asks a Chat Completions request for each content token's log probability, and
`top_logprobs` (`0..20`, which needs `logprobs: true`) for that many of the most likely tokens at
its position. A Responses request asks with `include: ["message.output_text.logprobs"]` or a nonzero
`top_logprobs`. Both routes answer with and without streaming, for every model family, speculative
decoding included.

The values describe the distribution the token was drawn from, before truncation: the logits after
the structured-output mask and the presence and frequency penalties, divided by the temperature (1
for greedy decoding) and normalized over the whole vocabulary. `top_k`, `top_p` and `min_p` do not
change them. A token the mask excludes never appears among the alternatives, so a constrained
position can report fewer than asked, and a sampled token outside its position's 20 most likely
tokens reports OpenAI's `-9999.0`. Under speculative decoding a verified position's penalties count
the drafts before it, as its sampling does.

Records cover the content channel: every generated token after the reasoning block and the
whitespace that closes it, the markup of a tool call included; reasoning tokens and the stop token
have none. A record travels with its text: a stream sends it with the chunk that publishes the
token's first byte (whitespace held back while a tool call may follow arrives after its record), and
a stop string drops the records of the tokens it cuts. Each record has `token` (the token's bytes as
text, with `U+FFFD` for a sequence that is not valid UTF-8 on its own), `logprob`, `bytes` and
`top_logprobs`. A Chat Completions choice carries them as `logprobs.content` with `refusal: null`,
streamed chunks as their choice's `logprobs`; a Responses `output_text` part carries them as
`logprobs`, and `response.output_text.delta` and `.done` carry them without `bytes`.

A request that does not ask gathers nothing.

### llama.cpp-compatible request observations

Every successful Chat Completions response includes a top-level `timings` object. This is a
llama.cpp-compatible response extension, not an OpenAI field. In a stream it is attached to the
last JSON chunk before `[DONE]`: the empty `choices` usage chunk when
`stream_options.include_usage` is true, otherwise the finish-reason chunk.

```json
{
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 83.0,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 129,
    "predicted_ms": 1140.0,
    "predicted_per_token_ms": 8.90625,
    "predicted_per_second": 112.28070175438596
  }
}
```

`cache_n` is the exact Engine-proven reused prompt prefix and `prompt_n` is the remaining prompt
suffix, so `cache_n + prompt_n` equals `usage.prompt_tokens`. Prompt time starts when admission
commits that exact reuse choice and ends when the first output token is committed. Generation time
starts at that first token and ends at the last committed output token. Accordingly, generation
speed uses `max(predicted_n - 1, 0)` token intervals; the first token belongs to prompt latency and
is not counted again as a decode interval. Zero-token, one-token, zero-duration, and exact-cache-hit
cases report finite zero rates rather than `NaN` or infinity. Speculative requests additionally
include terminal `draft_n` and `draft_n_accepted` when draft work occurred.

Set top-level `timings_per_token: true` on a streaming request to attach the latest cumulative
timing snapshot to each visible reasoning or content chunk and, when `return_progress` is also
enabled, to each prompt-processing chunk. During prompt processing that snapshot is prompt-only:
`predicted_n`, `predicted_ms`, `predicted_per_token_ms`, and `predicted_per_second` remain zero
until the first output commit, `prompt_ms` is the elapsed wall time since committed admission, and
the prompt rates span the processed suffix (`processed - cache`) tokens rather than the complete
prompt. This does not enable terminal timings, which are always present. A model commit that is
temporarily hidden by UTF-8, stop-string, reasoning, or tool-call buffering still advances the
cumulative token count; the next visible chunk observes that committed frontier. The option
increases response serialization and transport volume and is off by default.

Set top-level `return_progress: true` together with `stream: true` to receive prompt-processing
chunks:

```json
{
  "prompt_progress": {
    "total": 8192,
    "cache": 4096,
    "processed": 6144,
    "time_ms": 41
  },
  "timings": {
    "cache_n": 4096,
    "prompt_n": 4096,
    "prompt_ms": 41.5,
    "prompt_per_token_ms": 0.020263671875,
    "prompt_per_second": 49349.39759036145,
    "predicted_n": 0,
    "predicted_ms": 0.0,
    "predicted_per_token_ms": 0.0,
    "predicted_per_second": 0.0
  }
}
```

The `timings` object appears only when `timings_per_token: true` is also set. It reports the same
prompt-only live snapshot described above: `time_ms` is integral milliseconds while
`timings.prompt_ms` keeps sub-millisecond precision, and the prompt rates cover the 2,048
processed suffix tokens so far.

The initial event has `processed == cache`. Later cumulative events are published only after the
corresponding prefill unit commits, may be coalesced when the consumer is slower than prefill, and
never move backwards. The final event has `processed == total` and precedes the first output delta.
For an exact full-prefix hit, the initial event already has `cache == processed == total` and no
synthetic prompt work is reported. `time_ms` is elapsed wall time since committed admission;
clients may calculate actual suffix progress as `(processed-cache)/(total-cache)` when the
denominator is nonzero.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media requests use one complete-prompt context contract. After chat-template rendering and
media-token expansion, the result must fit Engine `--max-context`. The current Vision runtime also
has a 32,768 merged-token envelope (131,072 raw patches); the effective Vision limit is therefore
`min(--max-context, 32768)`. There is no fixed image/video item-count limit: item count is admitted
through aggregate source-byte, decoded-pixel, raw-patch, Vision-token, and live-memory budgets.

Media cache misses run as independent decode → resize → BF16-pack tasks on a bounded host worker
pool. Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality, so repeated media
in later requests reuses the exact immutable BF16 patch input; concurrent identical misses use one
single-flight build. `--media-cache-mib` bounds LRU-retained payloads, while
`--media-live-mib` bounds every cache-, request-, or runtime-referenced payload. Cache eviction does
not invalidate a request reference, and live bytes are returned only when the final reference is
released. A request-level preparation gate derived from the live limit prevents concurrent partial
builds from deadlocking the memory account.

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including
the prepared token count and configured context ceiling. A media preprocessing resource rejection
returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request
body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media
resource errors.

## Raw-prompt completion and the tokenizer

llama.cpp's native `POST /completion` (also `/completions`) and OpenAI's legacy
`POST /v1/completions` continue a raw prompt: no chat template, no reasoning block, no tool
parsing. The text the model generates is returned as it is, `<think>` tags included, and its
special tokens are not. The prompt is a string, an array of token ids, or token ids and strings
mixed (`[248045, "user\nHi"]`); special-token text in a string encodes as that token. llama.cpp's
array of several prompts is refused (`multiple_prompts_not_supported`): one request produces one
completion. A one-element array is that one prompt.

```bash
curl http://127.0.0.1:8080/completion -d '{"prompt": "The capital of France is", "n_predict": 16,
  "temperature": 0, "stop": ["\n"]}'
curl http://127.0.0.1:8080/v1/completions -d '{"prompt": "The capital of France is",
  "max_tokens": 16, "stream": true, "stream_options": {"include_usage": true}}'
```

Both accept `temperature` (below zero samples greedily), `top_k` (zero or below is the widest
candidate set, 20), `top_p`, `min_p`, `presence_penalty`, `frequency_penalty`, `seed` (negative
for a random one), `stop` (a string or up to 32 strings), `ignore_eos`, `stream`,
`timings_per_token`, `response_format`, and llama.cpp's `cache_prompt` (`false` keeps the request
out of the context cache). The output limit is llama.cpp's `n_predict` (then `max_tokens`) on
`/completion`, where any negative value is no limit, and `max_tokens` on `/v1/completions`, where
`-1` is; omitted, it is the [default output limit](#default-output-limit). llama.cpp sampler
controls NInfer does not have are accepted at their neutral values and refused otherwise, each
with `<field>_not_supported`: `repeat_penalty` (1), `typical_p` (1), `tfs_z` (1), `dynatemp_range`
(0), `mirostat` (0), `xtc_probability` (0), `dry_multiplier` (0), `top_n_sigma` (0 or below),
`n_probs` (0), `n_indent` (0) and `t_max_predict_ms` (0 or below); a non-empty `grammar`, `lora` or
nonzero `logit_bias` is refused too. `samplers`, `n_keep`, `min_keep`, `id_slot` and the parameters
of the controls above are accepted without effect.

`/completion` adds llama.cpp's `json_schema` (the bare schema; `{}` admits any JSON object; needs
`--structured-output`), `return_tokens`, `return_progress` and `response_fields` (the response
keeps only the named fields; `generation_settings/n_predict` names a nested one and is reported
under that path). Its response is llama.cpp's object: `content`, `tokens` (with `return_tokens`),
`stop`, `stop_type` (`eos`, `word` with `stopping_word`, `limit`, or `none`), `tokens_predicted`
(the stop token included), `tokens_evaluated` (the prompt), `tokens_cached` (prompt tokens the
context cache supplied), `prompt` (as text), `generation_settings` (the settings the request
resolved to), `has_new_line`, `truncated` (always false: NInfer never shifts the context),
`id_slot`, `model` and `timings`. Streamed, each event carries the next `content` with the running
`tokens_predicted`, and the last is the full object with an empty `content`; the generated token
ids, which the chunks do not carry, come in that last object's `tokens`. There is no `[DONE]`.

`/v1/completions` adds OpenAI's `echo` (the prompt text precedes the completion, and opens the
stream) and `stream_options.include_usage`, accepts `n`, `best_of` and `logprobs` only at 1, 1 and
0, and refuses a non-empty `suffix` (fill-in-the-middle, `suffix_not_supported`). Its response is a
`text_completion` object with `choices[0].text` and `finish_reason` (`stop` or `length`), `usage`,
and llama.cpp's `timings`; a stream ends with the finish chunk, the usage chunk when asked for, and
`[DONE]`.

The context cache retains a raw request's state at its end, so a prompt that continues an earlier
prompt and its completion (a client resending the conversation with the next turn) resumes from
it. A raw prompt has no message structure to checkpoint inside it, so a prompt of at least 512
tokens also keeps a private checkpoint one token short of its end, from which the same prompt sent
again (a regenerated completion) resumes: measured on an RTX 3090 with a 27B model, a repeated
1,602-token prompt reused 1,601 tokens and its time to first token fell from 615 ms to 26 ms. The
checkpoint costs one more prefill pass, about 20 ms there, which is why shorter prompts go without.

The tokenizer endpoints answer from the host, so they do not wake a sleeping model:

| Method and path | Request | Response |
|---|---|---|
| `POST /tokenize` | `content`; `parse_special` (default true: special-token text encodes as that token); `with_pieces`; `add_special` (accepted; Qwen checkpoints add no BOS) | `{"tokens": [ids]}`, or `[{"id", "piece"}]` with `with_pieces`, a piece that is not whole UTF-8 given as its bytes |
| `POST /detokenize` | `tokens` | `{"content": text}`, special tokens included |
| `POST /apply-template` | a Chat Completions body | `{"prompt": text}`: the prompt the request renders to, the assistant opener included |

## Rerank

`POST /v1/rerank` (also `/rerank`, `/v1/reranking`, `/reranking`) ranks documents by their
relevance to a query, in the shape of Jina's and llama.cpp's rerank API, with the served chat model
as the judge. Each document becomes one prompt in Qwen3-Reranker's formulation -- a system turn
asking whether the document meets the requirements of the instruction and the query, answered only
"yes" or "no", thinking off -- and its `relevance_score` is P(yes) / (P(yes) + P(no)) over the
first answer token's exact log probabilities ("Yes" and "No" counted with them). A general chat
model is not trained as a reranker, so its scores order documents well but are not calibrated
across queries. No startup option is needed.

```bash
curl http://127.0.0.1:8080/v1/rerank -d '{"query": "What is a panda?", "top_n": 2,
  "documents": ["The giant panda is a bear species endemic to China.", "Paris is in France."]}'
```

The request takes `query`, `documents` (strings or `{"text": ...}` objects, at most 1000), `top_n`
(every document by default), `return_documents` (default true) and NInfer's `instruction`, which
says what relevance means (Qwen3-Reranker's default: "Given a web search query, retrieve relevant
passages that answer the query"). The response is a list of `results`, best first, each with its
`index`, `relevance_score` and `document.text`, and `usage.prompt_tokens` over all judgements. A
request with `texts` instead of `documents` follows TEI, as llama.cpp's `/rerank` does: a bare
array of `{"index", "score"}`, with `text` when `return_text` is set. The judgements run a lane's
worth (`--max-concurrency`) at a time and share the cached judging prefix; on an RTX 3090 with a 27B
model and two lanes, four short documents took 0.4 s.

## OpenAI Responses Core

NInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). All
supported model instances use this same adapter and Engine route. It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | NInfer Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the artifact-derived public model ID or explicit `--model-id` override |
| `input` | string or typed Item array; it may be omitted or empty only when `previous_response_id` already supplies a user query |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | non-negative integer; omission executes with the [default output limit](#default-output-limit) but remains `null` in the Response object |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `client_metadata` | Codex client extension; an object or `null`, accepted as opaque tracing metadata with no generation effect |
| `reasoning.effort` | `none` requests disabled thinking; other standard effort values pass to the selected template |
| `reasoning.summary`, `reasoning.generate_summary` | `auto`, `concise` or `detailed` (other values are rejected, and the two must agree when both are sent); accepted and echoed without changing model execution. Summaries are not generated: reasoning Items retain `summary: []` and complete raw `reasoning_text`. `reasoning.context` and `reasoning.mode` are rejected with `reasoning_option_not_supported` |
| `chat_template_kwargs` | template parameters as a JSON object; standard options merge with typed fields |
| `preserve_thinking` | alias for `chat_template_kwargs.preserve_thinking`; conflicting values are rejected |
| `graft` | NInfer extension: name of a [prompt graft](#prompt-grafts), or `null`; also accepted by input token count |
| `thinking_budget` | NInfer extension: positive per-request [thinking cap](#openai-chat-completions), or `null`; rejected by input token count, which does not generate |
| `text.format` | `text`, `json_object`, or flat `json_schema`; see [structured output](#structured-output) |
| `tools` | direct function definitions or namespace groups containing function definitions; see below |
| `tool_choice` | `auto`, `none`, a named function, `required` over one or more callable tools, or function-only `allowed_tools` in either mode; a namespaced selection carries both `namespace` and `name` |
| `parallel_tool_calls` | `true` by default; `false` constrains generation to at most one tool call |
| `max_tool_calls` | non-negative integer accepted as a hosted-tool no-op; NInfer does not execute hosted tools |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | integer in `[0,20]`; a nonzero value also asks for [token log probabilities](#token-log-probabilities) |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted, empty, or any of `reasoning.encrypted_content`, which requests the local raw-reasoning mirror described below, and `message.output_text.logprobs`, which requests [token log probabilities](#token-log-probabilities) |
| `stream_options.include_obfuscation` | optional boolean; accepted as a transport hint, but this local server emits no padding |
| cache and client hints | valid `prompt_cache_key`, `prompt_cache_options`, `prompt_cache_retention`, `safety_identifier`, and `user` values are accepted without being mapped to Engine session identity |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored. When a rejection is caused by a
specific value, the error names that value (ASCII-escaped and truncated), its size where relevant,
and its location in the request, both in the error `param` (for example `tools[0].function.name`
or `messages[2].tool_calls[0].function.name`) and in the message text.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `refusal` | assistant-message replay part; its text enters assistant history |
| `input_image` | user- or assistant-message part with HTTP(S) or data-URI `image_url`; detail omitted, null, `auto`, `low`, `high`, or `original`; all recognized levels fall back to fixed `auto` preprocessing; requires server `--vision` |
| `input_video` | NInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with `reasoning_text` content; summary/encrypted metadata may accompany raw text but cannot replace it |
| `function_call` | completed assistant call with optional `id` and namespace, plus required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed result with required `call_id` and optional matching name/namespace assertion; `output` may be a string or a non-empty array of `input_text`/`input_image` parts |

Contiguous assistant-owned Items form one assistant history turn in the representable order
`reasoning` -> assistant message content -> `function_call`. Multiple message Items append their
content parts, multiple calls retain declaration order, and a reasoning-only turn is retained. A
user, system, developer, or `function_call_output` Item ends the group; an order that would require
rearranging assistant content fails with `invalid_assistant_history`. Results are validated by
`call_id` and reordered to call declaration order before prompt rendering; unknown, duplicate, or
unrepresentable partial result sets fail with `invalid_tool_history`. Canonical input Items retain
client order. Input Item IDs are preserved when supplied and generated otherwise; duplicate IDs
fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend.

An `input_text`, `input_image`, or tool-result part may carry
`prompt_cache_breakpoint:{"mode":"explicit"}`. Up to four such values become shared stable-prefix
boundaries; they affect reuse opportunities, not prompt identity or output semantics. String
message status/phase metadata is accepted but has no Qwen prompt representation.

Image detail levels do not select separate preprocessing profiles: the server's startup
`--vision-max-merged` budget applies to every recognized level, and canonical image Items stored
for response-chain replay use `detail:"auto"`. Unknown strings and non-string values other than
null remain HTTP 400 errors.

`input_file`, `input_audio`, image `file_id`, reasoning metadata without raw
reasoning text, partial tool Items, and other Item/content types are not supported. HTTP media URLs
stored in a response chain are fetched again when that chain is continued; use data URIs when the
historical media bytes must be immutable.

### Function tools

Responses function definitions may be declared directly rather than inside Chat Completions'
nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

They may also be grouped in a Responses namespace:

```json
{
  "type": "namespace",
  "name": "mcp__weather",
  "description": "Weather service",
  "tools": [{"type": "function", "name": "get_current"}]
}
```

NInfer gives each namespace/function pair a distinct internal Engine identity and restores the
separate `namespace` and `name` fields in aggregate output, SSE events, and replayed Items. The same
function name may therefore appear in different namespaces. Namespace members remain ordinary
client-executed functions; this does not add a remote MCP executor.

NInfer renders a function definition in the Qwen prompt, whichever protocol declared it, as
`{"type":"function","function":{"name":…,"description":…,"parameters":…}}` in declaration order
and without `strict`, which only selects the server's output contract, and parses model output
into separate `function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. Only functions in the current effective tool set can become structured calls;
undeclared names are blocked during constrained decoding. `allowed_tools` filters that set without changing
declaration order, while `tool_choice:"none"` disables structured tool output even when the history
contains earlier calls.

Required, named and allowed tool selections use the common [tool constraints](#structured-output)
contract, retain declarations in the prompt, and support thinking. `strict:true` enforces supported
tool-argument schemas. Hosted tools, remote MCP tools, custom Responses free-form tools, deferred
loading, output schemas and caller restrictions excluding direct invocation remain unsupported.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. NInfer may emit:

- a `reasoning` item containing complete raw `reasoning_text` and an empty `summary` array,
  including when `reasoning.summary` is requested;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

When `include:["reasoning.encrypted_content"]` is requested, reasoning Items also
carries en `encrypted_content` equal to its raw `reasoning_text`. This field is **not**
**encrypted** and provides no confidentiality.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`; validation and preparation errors remain
normal HTTP error responses. `completed_at` is populated only for completed Responses. A
reasoning-only incomplete result contains no invented empty assistant message.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
checkpoint-proven prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

Reasoning Items carry `summary: []` in `response.output_item.added`, `.done`, and the terminal
Response, regardless of the requested summary style. The stream emits the raw `reasoning_text`
content lifecycle and no `response.reasoning_summary_*` events. Unsupported summaries must not
be replaced with explanatory placeholder text: clients that prefer a non-empty summary over raw
content would otherwise lose the full reasoning body used for display and context accounting.

For `include:["reasoning.encrypted_content"]`, the in-progress
`response.output_item.added` Item omits `encrypted_content` because the complete reasoning text is
not available yet. `response.output_item.done` and the terminal Response output contain the same
complete raw mirror. A response with no reasoning Item emits no encrypted placeholder.

Function arguments use `response.function_call_arguments.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only an ambiguous tool-call marker prefix or the structured tool region is
held. Tool markup that never opens a function is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so compatible
checkpoint reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact
checkpoint already proved compatible by the model runtime.

For Engine-local reuse, a stored root Response receives one bounded session key derived from its
response ID, and every `previous_response_id` child inherits that key. `store:false` roots remain
anonymous; a `store:false` child may read its inherited session checkpoint but does not replace the
stored chain's latest endpoint. Response-store eviction or deletion removes the HTTP object, not an
independently retained Engine checkpoint; the latter remains bounded by the Engine's own retention
and pressure policy. No session key or cache marker is added to the HTTP schema.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found`; stream recovery and non-empty `include` are rejected rather than ignored |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`); image URLs are redacted unless `include=message.input_image.image_url` |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

### Responses input token count

`POST /v1/responses/input_tokens` uses the same prompt path as Create and does not run generation.
It accepts `model`, `input`, `instructions`, `previous_response_id`, reasoning, function tools and
tool choice, supported text/truncation values, and the `preserve_thinking` extension. Parent lookup,
call-ID normalization, template rendering, and media expansion are therefore identical to the
corresponding Create request:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, `include` values other than
`reasoning.encrypted_content`, background execution, compaction, files/audio, and
OpenAI-hosted/MCP/custom tools. Documented accepted hints and the local raw-reasoning mirror
are exceptions; other unsupported capabilities remain explicit compatibility boundaries.


## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint accepts top-level System text, ordered User/Assistant/System history, text and image
blocks, Thinking history, tool-use history, tool results, user-defined tools, aggregate responses,
and Anthropic SSE. Consecutive User or Assistant messages are joined without adding separators.
System messages in `messages` may appear anywhere, including first, between two User messages,
directly after an Assistant message, or last; each renders as its own System turn at that position
after the top-level `system` text, so appending one keeps the earlier rendered prompt reusable. The
one excluded position is between an Assistant `tool_use` and the User message carrying its
`tool_result`, which is rejected as invalid tool history. A final text-only Assistant message
is an Assistant prefill: generation continues its existing text instead of opening another turn.
Assistant prefill cannot contain media, Thinking, or tool calls and cannot start with Thinking
enabled; Thinking left to the server default counts as enabled, so a prefill needs it disabled
explicitly.

`max_tokens` is optional for local clients and otherwise uses the
[default output limit](#default-output-limit); a positive
value is the complete output budget. `max_tokens:0` is rejected because NInfer does not expose a
completed zero-output cache-prewarm lifecycle. `temperature`, `top_p`, `top_k` (clamped to 20
like the OpenAI endpoints), and `stop_sequences` enter Engine execution. A matched custom stop is returned as
`stop_reason:"stop_sequence"` together with the actual `stop_sequence`; context exhaustion returns
`model_context_window_exceeded`.

Thinking supports `disabled`, `adaptive`, and `enabled`. Enabled Thinking requires
`budget_tokens >= 1024`, and that budget is passed to Engine. Unlike the Anthropic API, a budget at
or above `max_tokens` is accepted: the output limit ends thinking before the budget can, so the
budget takes no effect. Clients such as Qwen Code send a fixed budget while shrinking `max_tokens`
to the context left. Visible
Thinking is returned with an opaque compatibility signature; SSE emits its `signature_delta`
before closing the block. Request lowering reconstructs the local prompt from the visible
`thinking` text and treats `signature` as non-semantic transport metadata, so retained history
remains usable across serve restarts.

`thinking.display:"omitted"`, which current Claude Code sends on every request, returns each
Thinking block with an empty `thinking` string and no `thinking_delta` events. Its `signature`
(`signature_delta` when streaming) is `ninfer-reasoning.v1:` followed by the Base64 reasoning text,
so the reasoning stays out of the visible transcript but comes back with the block the client
echoes. When a replayed Thinking block has empty `thinking` and such a signature, request lowering
restores the reasoning from it, so the prompt (and its prefix-cache identity) is the one
`summarized` display would give; non-empty `thinking` text takes precedence, any other signature
stays non-semantic metadata, and a malformed NInfer signature fails with
`invalid_thinking_signature`. The signature is not encrypted or authenticated: this is a local
trusted server, and the value is the same reasoning text `summarized` display shows. Count Tokens
ignores `display`. `preserve_thinking` remains a NInfer extension for closed-turn reasoning
history, and `graft` selects a [prompt graft](#prompt-grafts). `output_config.effort` passes its
protocol-validated value to the selected template, substituting the nearest value the template
accepts as described above. `output_config.format` accepts `{"type":"json_schema","schema":{...}}`,
enforced strictly (see [Structured output](#structured-output)); Count Tokens ignores it.

User-defined tools support `name`, `description`, object `input_schema`, `strict` and
`input_examples`. `auto`, `none`, a named `tool`, and `any` use the common constrained selection
rules described above; `any` without tools is rejected. `disable_parallel_tool_use:true` constrains
generation to one call. Deferred tools, tools excluding direct model calls, Anthropic-provided
server tools, toolsets, MCP and containers remain unsupported. `tool_result` preserves text/image order and marks
`is_error:true` explicitly in the model prompt. For a visible Assistant tool-use turn, the next
User turn must provide exactly one leading result for every declared ID; valid results are matched
by ID and normalized to call order. A history that begins with results remains valid as a truncated
or imported conversation.

Ephemeral `cache_control` on the request, tools, System blocks, and User text/image frontiers is a
best-effort retention hint. NInfer maps representable breakpoints to exact prompt frontiers, keeps
the latest markers allowed by the Engine configuration, and ignores TTL and unrepresentable cache
hints rather than rejecting generation. The Engine's own discovery of repeated prefixes (published
after their second sighting) stays on beside the request-level automatic `cache_control` boundary,
as on the OpenAI endpoints. Reuse still requires exact rendered-token compatibility;
aggregate usage reports verified reused tokens in `cache_read_input_tokens` and leaves cache
creation unknown. Streaming emits `message_start` after Engine admission commits the prefix
selection and before transfer/prefill output, so its uncached/cache-read split is already exact;
terminal cumulative usage matches the aggregate response.

Documents, Search Results, Files, server-tool results, container uploads, and
other execution-dependent blocks are rejected with the missing capability identified. Metadata,
service tier, inference geography, protocol-version/beta headers, cache TTL, and unknown advisory
fields do not block an otherwise executable request. The request `model` is any non-empty local
proxy label and is echoed in the response; it does not select the resident artifact.

Every Messages response carries a `request-id` header; error bodies also carry `request_id` and use
Anthropic error categories. Local admission overload maps to HTTP 529 and queue/media timeouts to
HTTP 504. Streaming owns the full Anthropic block lifecycle for Thinking, text, and tool use.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without generation. It shares the same prompt normalization, tools, Thinking mode, Assistant
prefill, media processing, and cache-marker interpretation as Messages; output-only sampling and
streaming fields do not affect the count:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Prompt grafts

A prompt graft is a hidden conversation prefix, for example a system turn plus an assistant
acknowledgement, which the operator loads at startup and a request selects by name. Grafts are
[phantom-kv](https://github.com/lordx64/phantom-kv) containers (format_version 1): a safetensors
file plus a `.json` sidecar beside it. A `prefill_kv` graft carries token ids that NInfer replays;
a trained `softprompt_kv` or `direct_kv` graft carries cache state that NInfer writes directly
(both described below).

```bash
ninfer-serve model.ninfer --graft v1=C:/grafts/v1_q38_nf4.bin --graft red=C:/grafts/red.bin
```

A request selects one with the top-level `"graft": "NAME"` field on OpenAI Chat Completions,
Responses (create and input token count) and Anthropic Messages. A name the server did not load
fails with `400 unknown_graft`, and a non-string value fails as a malformed `graft` field.

`--default-graft NAME` makes one loaded graft the default for requests that state none. It must
name a `--graft`, or the server refuses to start.

Grafts are local files; the repository and release archives ship none. `scripts/run.sh` and
`scripts/run.bat` pass `--graft NAME=<file>` for every `NAME.bin` with its `NAME.json` sidecar in
`grafts/<model key>/` beside `models/` (or `NINFER_GRAFT_DIR`), serve without grafts when there are
none, skip them under `NINFER_GRAFTS=off`, and pass `--default-graft` from `NINFER_DEFAULT_GRAFT`.

| request `graft` | without `--default-graft` | with `--default-graft D` |
|---|---|---|
| absent or `null` | none | `D` |
| `""` | none | none (explicit opt-out) |
| `"X"` | `X` | `X` |

A grafted request runs exactly as if the graft's hidden turns preceded its own messages. The
graft's tokens occupy positions `[0, n)` and the request's own rendered prompt starts at `n`. With
greedy decoding, output is identical to sending those turns as literal messages. The rendered
prompt should carry no system turn of its own, because the graft already holds one. Graft tokens
count toward `--max-context` and the reported prompt/input tokens.

A `prefill_kv` container also stores the cache state that phantom-kv computed for the graft
(attention K/V and Gated DeltaNet conv/recurrent state). NInfer does not inject that state for this
kind. It replays the graft's own token ids through its own prefill, which is exact for the loaded weights and KV
storage and covers every layer, including the MTP draft layer. The end of the graft is offered to
the shared-prefix cache, so conversations after the first reuse the whole graft instead of
re-prefilling it. When the shared catalog is already full of other prefixes, the engine's
admission rule for structural candidates admits the graft on its second use.

Startup validates each graft against the loaded model and refuses to start on any mismatch:
- the per-layer attention/linear-attention layout;
- the KV-head, conv and recurrent-state geometry;
- the sidecar's payload sha256;
- the replay ids against the vocabulary.

Trained `softprompt_kv` and `direct_kv` grafts carry no replayable token ids. Their stored K/V and
Gated DeltaNet state are instead written at startup into a pinned shared-prefix slot, and grafted
requests start from it. Because nothing is replayed, these grafts cover the text layers only: the
graft carries no draft-backend state, so under `--spec` the MTP or DFlash cache over the graft's
positions is zero-filled and the draft proposes without graft context there. Output is unchanged,
because the target verifies every proposal against the injected state; only the acceptance rate
can fall. Each one holds a Device StateImage and a shared-prefix slot for the life of the server;
startup adds them on top of `--device-state-slots` and `--max-shared-prefixes`, and a disabled
context cache refuses them. Grafted requests use the context cache like any other: the pinned slot
is the root, and later turns and shared prefixes are captured after it. In the prompt the graft's
positions are held by ids derived from the container's sha256, so a cached prefix is only ever
matched by requests using the same graft. With `--devices`, each layer's K/V and state are written on the device
of the stage that holds that layer.

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated; `GET /v1/load`,
`GET /stats` and `GET /metrics` require the key, on `--stats-port` too.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default. A preflight's
`Access-Control-Request-Headers` are allowed on top of the fixed list, so browser clients that send
their own headers (a WebUI's `x-conversation-id`) pass the preflight.

After the listening line, startup prints the URLs to open: the API base (`/v1`) and the WebUI when
one is served. A wildcard bind address (`0.0.0.0`, `::`) is announced through loopback, since a
browser does not accept it as a destination.

## Server options

The table lists executable defaults. The startup example selects a long-context INT8/MTP3 profile.

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--stats-port N` | also serve `GET /health`, `/stats`, `/v1/load` and `/metrics` on port `N` of the same address, with one worker of their own, so a dashboard or watchdog is never queued behind the connections the request pool serves; the readiness and API-key rules are the main listener's | off |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | override the public OpenAI model alias | artifact `identity.model_id` |
| `--max-context N` | logical context ceiling of each sequence | `8192` |
| `--rope-yarn` | past the model's native window, YaRN at factor `--max-context` / native instead of plain RoPE for the text and MTP layers | off |
| `--rope-yarn-factor F` | YaRN at a fixed factor in `[1,4]` for every position whatever `--max-context` is, as Qwen documents one factor per deployment; it grows neither the default context nor the KV pool, and long-context extrapolation is not a quality guarantee | `1` (as `--rope-yarn` decides) |
| `--rope-scaling-factor F` | instead of YaRN, linear position interpolation for the text and MTP layers: a position past `--rope-scaling-original-context` rotates at original + (position - original) / F, so positions up to it keep their exact angles; `[1,32]`, excludes `--rope-yarn` and `--rope-yarn-factor`, and raises neither the four-times-native window cap nor the KV pool. The DFlash adapter keeps its plain RoPE | `1` (off) |
| `--rope-scaling-original-context N` | the interpolation threshold | the native window |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context`, or `auto` with `--use-alt-prefix-caching` | `8192` |
| `--kv-headroom-mib N` | device memory in MiB that `--kv-capacity auto` leaves free after sizing the KV pool; requires `auto`. `--vram-headroom-mib` is accepted as an alias | `1024` |
| `--max-concurrency N` | maximum admitted requests; valid range `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `600000` |
| `--recover-invariant-failures` | a broken internal invariant in the Engine worker fails the active and materializing requests and leaves the waiting ones queued, as recovery from out of memory does, instead of failing the Engine; eight consecutive recoveries without a completed unit still fail it | off |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--fast-prefill-kernel` | prefill an `int8` or `rk*` KV cache with the fast prompt-attention kernel (FP16 PV accumulation per 64-key tile) and round `--prefill-chunk` down to whole attention waves; on Blackwell, prefill an `nvfp4` KV cache past 2048 visible keys with its fast kernel (QK on block-scaled FP4 Tensor Cores); a small perplexity cost (see [perplexity](perplexity.md)). Without the flag the [device profile](device-profiles.md)'s `attn_prompt_fast` decides, and the built-in profiles turn the kernel on where it measured faster | the device profile (KVMem defaults off and admits INT8/RK profiles only) |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--log-colours on\|off` | `on` colours the console log's levels and gives every statistic of the operational lines a stable colour; `off` keeps the log plain; a redirected stderr is always plain | levels coloured on a console |
| `--log-stats-panel on\|off` | pin the session statistics panel beneath the console log on an interactive terminal | off |
| `--log-level trace\|debug\|info\|warning\|error\|critical\|off` | pretty stderr verbosity | `info` |
| `--device N` | CUDA device index | `0` |
| `--devices A,B,...` | one pipeline stage per listed CUDA device (2 to 8, Linux; see [pipeline stages](maintainer/pipeline-parallel-plan.md)); on the Qwen3.5 family MTP, DFlash and DFlash2 run across the stages and Vision is refused; overrides `--device` | none |
| `--stage-layers A,B,...` | layers per stage, in `--devices` order; omitted means a split chosen from each device's free memory | memory-balanced |
| `--expert-residency device\|host\|disk` | Qwen3.8-Flash-Next: routed expert banks in GPU memory, in pinned host memory read across the bus, or left in the artifact's files and streamed into the device expert cache (see [Qwen3.8-Flash-Next](qwen3-8-flash-next.md#run)) | `device` |
| `--expert-cache-mib N\|auto` | with host or disk experts, device memory for the most used experts; `0` turns the host-mode cache off, and disk mode needs one | `auto` (what is free after startup) |
| `--ngram-table PATH` | Qwen3.8-Flash-Next: the n-gram table artifact of a model published without its table | the model's own table |
| `--ngram-ram` | Qwen3.8-Flash-Next: load the n-gram table into RAM instead of reading its rows from the file | off |
| `--no-ngram-table` | Qwen3.8-Flash-Next: run without the n-gram table, a non-standard experimental mode (the Q2_0 release's WikiText-2 perplexity rises from 2.66 to 5.01) | off |
| `--device-profile auto\|off\|calibrate` | the per-GPU [route profile](device-profiles.md): `auto` uses the stored or built-in one and calibrates a device that has none, `off` keeps the compiled routes, `calibrate` measures anew | `auto` |
| `--device-profile-path FILE` | profile file instead of the user cache | the user cache |
| `--context-cost-presets FILE` | optional runtime context-cost preset registry | generic + compiled defaults |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | bounded media preprocessing workers; `0` selects at most 16 from host concurrency | `0` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--request-log-max-mib N` | rotate the request log once it reaches N MiB: the file is renamed to `FILE.1`, older copies move up to `FILE.<keep>` and the oldest is dropped, and the new file starts with the `server_start` record; requires `--request-log-jsonl` | `0` (one unbounded file) |
| `--request-log-keep N` | rotated request logs kept; `0` keeps none | `4` |
| `--response-store-max-records N` | maximum locally retained Responses objects | `1024` |
| `--response-store-max-mib N` | total local Response envelope/Item/context budget | `256` |
| `--kv-dtype bf16\|int8\|fp8\|rk8v4\|rk4v4\|rk4v4-e8\|rk2v4-e8\|nvfp4\|k8v4` | KV-cache storage. `rk8v4` is opt-in RotorQuant, `rk4v4` opt-in Lloyd-Max 4-bit keys, `rk4v4-e8` opt-in E8-lattice INT4 keys and `rk2v4-e8` opt-in E8 root-code keys; all nine are accepted on every build target (see [Context and memory](cli.md#context-and-memory)); for Qwen3.8-Flash-Next it is the storage of the sparse-attention layers' KV | `bf16` |
| `--spec mtp\|dflash\|dflash2` | speculative backend | off |
| `--draft-tokens N` | `1..15` for MTP, DFlash and DFlash2 | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--adaptive-mtp` | MTP only: each round verifies 3..`--draft-tokens` drafts, the width favored by the drafts' measured survival and the measured round cost; see [Adaptive MTP](#adaptive-mtp) | off |
| `--mtp-attention-window N` | MTP only: the draft head attends to the first 64 keys and the newest `N` before its query; verification keeps full attention; see [MTP attention window](#mtp-attention-window) | `0` (whole history) |
| `--lookup-ngram N` | legacy context-lookup drafting alongside `--spec`; disabled with KVMem, which uses `--ngram-draft-tokens` and `--ngram-min-match` | `0` (off) |
| `--ngram-draft-tokens N` | copy drafting alongside `--spec`: up to `N` tokens (1..63; widths 16..63 automatically reduce to 15 at startup with `--max-concurrency > 1`, with a warning) copied from earlier prompt, tool-result or output text that the last `--ngram-min-match` tokens match, verified by the target; `0` disables it; see [Ngram copy proposals](ngram.md) | `15` with `--spec`, else `0` |
| `--ngram-min-match N` | shortest match a copy is drawn from, `4..64` | `12` |
| `--ngram-archive-mib N` | RAM archive that keeps finished requests' copy sources for later requests naming the same `X-NInfer-Draft-Session` | `0` (off) |
| `--ngram-session-mib N` | per-session share of that archive, 1 MiB up to the archive size | `128` |
| `--ngram-native-sessions` | with the archive, also recognize the session identities Kilo, Codex and Claude send | off |
| `--prefill-cublas` | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost, and it wants a larger `--prefill-chunk` to pay (see [performance](performance.md)) | off |
| `--no-prefill-cublas-projections` | with `--prefill-cublas`, keep the attention and GDN input projections off that route | projections on |
| `--no-prefill-a8` | prefill every projection on the A16 routes instead of the integer-activation ones | integer routes |
| `--mlp-a8-decode` | integer-activation MLP gate_up at decode and verify widths | off |
| `--lm-head-q4`, `--lm-head-q6` | store the output head as Q4 or Q6 while loading (Q4 costs +0.69% perplexity, Q6 +0.01%) | off |
| `--embedding-q4`, `--embedding-q6` | store the token embedding as Q4 or Q6 while loading | off |
| `--mtp-experts-q4` | Qwen3.6-35B-A3B: store the MTP layer's routed experts in the text layers' formats | off |
| `--gdn-state-fp16` | keep the GDN recurrent state in FP16, which halves each Host StateImage ([quality trades](maintainer/quality-trade-experiments.md)) | FP32 |
| `--default-max-tokens N` | output limit when omitted by a request; see [default output limit](#default-output-limit) | largest budget that keeps every lane admissible |
| `--default-thinking-budget N` | positive thinking cap inherited by thinking-enabled requests | unset |
| `--no-webui` | stop serving a WebUI built in with `NINFER_WEBUI_DIR` | served when built in |
| `--webui-mcp-proxy` | relay the WebUI's MCP traffic at `/cors-proxy` (http targets, no API key) | off |
| `--structured-output` | accept JSON and JSON Schema response formats (see [Structured output](#structured-output)) | off |
| `--unconstrained-response-format` | without `--structured-output`, generate a JSON or JSON Schema request unconstrained instead of refusing it | off |
| `--assistant-prefill` | continue a Chat Completions request's trailing assistant message in place, as `/v1/messages` does | off |
| `--lenient-assistant-history` | accept Responses input whose assistant message content or reasoning follows `function_call` Items in one run: it joins that run's assistant turn, as Messages flattens content blocks, and the template renders it before the calls. Without it such input fails with `invalid_assistant_history` rather than being silently reordered | off |
| `--thinking-budget-message TEXT` | message a thinking-enabled request receives at its thinking budget instead of the built-in notice; the canonical `</think>` close is appended when missing | built-in |
| `--default-reasoning-effort E` | effort for requests that name none: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` or `max` | unset |
| `--vision` | enable media input and load Vision GPU allocations | off |
| `--vision-residency resident\|overlay\|cpu` | `overlay` keeps the Vision tower in pinned host memory and encodes each image inside a window borrowed from the evict-ranked text weight tail, so `--vision` no longer reserves device memory and `--kv-capacity auto` resolves the no-vision capacity; requires `--vision` and CUDA virtual memory management. `--vision-offload on\|off` is accepted as an alias for `overlay\|resident`. `cpu` decodes the tower to host FP32 and encodes on CPU threads with no device Vision memory (see [Vision residency](#vision-residency)); it caps `--vision-max-merged` at 256 unless given | `resident` |
| `--vision-cpu` | `--vision` with `--vision-residency cpu` | off |
| `--vision-max-merged N` | merged-token budget of one media item, `[64, 16384]`; larger images and video frame pairs are downscaled at preprocessing instead of being rejected, and the overlay window is sized for it | 16384 |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--cuda-graph-allowance-mib N` | total CUDA Graph driver-state allowance in MiB, subtracted from the KV sizing budget | computed |
| `--wddm-evictable-budget` | Windows builds with `-DNINFER_D3D12_RESIDENCY=ON`: budget against dedicated memory, holding the device arenas resident | off |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--use-alt-prefix-caching` | select the hybrid prefix cache ([spec](maintainer/hybrid-prefix-cache-spec.md)): content-addressed 64-token KV blocks shared across requests plus sparse state snapshots. It configures itself: `--kv-capacity` defaults to `auto` (free VRAM becomes Device block cache) and `--host-cache-mib` sizes the one pinned Host pool that blocks and snapshots share. The checkpoint-catalog flags are rejected with it: its capacities, `--auto-long-anchors`, `--auto-prefix-grid`, `--derive-session-keys`, `--context-cache-policy`, `--release-diverged-checkpoints`, `--thorough-admission-search`, `--recency-eviction`, `--value-aware-demote` and the disk tier, as are pipeline `--devices`. Requests that share a new prefix with one still prefilling wait for its snapshot instead of prefilling the prefix again, which needs `--concurrent-prefill` to admit them meanwhile. | off |
| `--use-original-prefix-caching` | select the checkpoint catalog explicitly; it is already the default, and the flag is accepted for command lines written for builds where the hybrid cache is. Rejected together with `--use-alt-prefix-caching` | default |
| `--device-snapshot-slots N` | hybrid: Device state snapshot slots (`1..64`) | `max-concurrency + 1`; `+ 2` without a Host tier |
| `--cache-taps-per-request N` | hybrid: new prefill state snapshots per request (`0..64`) | `8`; `2` without a Host tier |
| `--cache-tap-ladder N` | hybrid: history-snapshot ladder base G; flexible taps at `prompt − G·2^k` | `max(4096, 2 * prefill-chunk)` |
| `--cache-tap-min-gap N` | hybrid: minimum tokens between ladder snapshots | `max(1024, prefill-chunk)` |
| `--prefix-cache-file PATH` | hybrid: at startup, restore the Host tier from `PATH` if the file exists; when the server stops ([Stop the server](#stop-the-server)), save it there once running and queued requests are cancelled (every Host-backed snapshot and the block path it resumes through). A save cut short is abandoned and the previous file kept: one more Ctrl+C during the stop also deletes the unfinished `PATH.tmp`, while Windows ending a closed console's process about 5 s after the close leaves it until the next save, so stop large caches with Ctrl+C. `PATH` may be relative (resolved against the launch directory) or absolute, e.g. `--prefix-cache-file "e:\NInfer-Deploy-V3\file.cache"`. Its directory must exist, and the flag needs a Host tier (not `--host-cache-mib 0`). A file written for another artifact, KV format, speculative backend, RoPE scaling or `ninfer-serve` binary is ignored and replaced at shutdown. The startup log reports what was restored. Saving writes up to `--host-cache-mib` of data. | off: nothing is saved or restored |
| `--device-state-slots N` | extra Device checkpoint StateImages beyond the active-lane guarantee | `max-concurrency` |
| `--host-state-slots N` | pinned Host StateImage capacity | `8` |
| `--host-kv-mib N` | shared pinned Host Main/Backend KV byte capacity in MiB | `8192` |
| `--host-cache-mib N` | single pinned Host RAM ceiling for the whole retention tier in MiB. Hybrid mode: the slab pool KV blocks and state snapshots share, split at run time by eviction value; `0` keeps the cache on the Device only, and a nonzero budget below one snapshot is rejected. Default cache: the engine derives the Host StateImage slot count from the checkpoint inventory the capture path creates, spends the remaining state headroom on more long anchors per continuation when `--auto-long-anchors` is on, and gives Host KV the remainder. Replaces `--host-state-slots` and `--host-kv-mib`, which are rejected alongside it. | hybrid `8192`; otherwise unset (component flags used) |
| `--max-private-continuations N` | private continuation descriptor capacity | `2 * max-concurrency` |
| `--max-shared-prefixes N` | Engine-wide shared stable-prefix descriptor capacity | `max(max-concurrency, 7)` |
| `--max-long-anchors-per-continuation N` | private long-anchor limit per continuation; with `--auto-long-anchors`, `--host-cache-mib` raises it within the state inventory it funds and never lowers it | `2`, `4` with `--auto-long-anchors` |
| `--derive-session-keys` | a request that names no session -- Chat Completions and Messages carry none -- gets a key derived from its instruction messages and first user message, which stay fixed for the conversation's life. The conversation then keeps a session lineage and live-session retention for its checkpoints, as a Responses conversation does. A client compaction rewrites the first user message and starts a new key | off |
| `--auto-long-anchors` | the engine anchors up to that many message boundaries of every request itself, so a later request that rewrites earlier history -- a compacted or edited transcript -- resumes from the nearest retained anchor instead of root; each anchor costs a prefill split and a StateImage whether or not the client ever rewrites | off |
| `--branch-anchors` | before a request is planned, the engine finds the deepest point its prompt matches a retained conversation or shared prefix token for token; when that owner's checkpoints all end at least 1024 tokens below it, the request captures a private long anchor there, so the next request that diverges at the same point (an agent branching off a long shared history) resumes from it instead of re-prefilling the matched tail. Costs a catalog scan per planned request and, where it fires, a prefill split and a StateImage; it uses the long-anchor capacity. Not with `--use-alt-prefix-caching`. | off |
| `--no-endpoint-anchors` | do not keep endpoint anchors. By default a request that resumes a retained conversation at its endpoint and extends it keeps that endpoint as a private long anchor, taken from the resumed state before the new turn is prefilled (no prefill split), so a later request that diverges right there -- another reply to the same answer, an edited last message -- resumes from it instead of re-prefilling the conversation from root. Such a request keeps the conversation where the cache holds the branch beside it, and otherwise rewrites the conversation from the anchor, as a regenerated reply does from its turn checkpoint. On an RTX 4090 with Qwen3.8-27B (IQ3_S) and the default cache, the first token of such a branch of a 30.7K-token conversation came after 108-110 ms instead of 9.4-9.5 s; the continued turn that takes the anchor paid 7-8 ms more (110-111 ms instead of 102-104 ms), a Host copy of the 147 MiB state because both Device slots were taken. It fires where the endpoint lies at least 1024 tokens above the conversation's deeper long anchors, costs a StateImage per firing and uses the long-anchor capacity. Not with `--use-alt-prefix-caching`. | kept |
| `--long-anchor-spacing N` | with `--auto-long-anchors`, the minimum token gap between automatic anchors, doubling per anchor walking back from the prompt end (anchor k sits at least `N * 2^k` tokens below the previous grid point), so short tool-loop turns do not each cost an anchor and deep history stays covered; `0` anchors every one of the last boundaries | `1024` |
| `--max-cache-markers-per-request N` | caller marker input-complexity bound | `4` |
| `--slot-save-path DIR` | enable `POST /slots/{id}` save/restore/erase with files in `DIR` (created at startup); see [Slots](#slots) | disabled |
| `--auto-save-evicted` | write an evicted session back to its bound slot file; requires `--slot-save-path` | off |
| `--disk-kv-path DIR` | disk tier under the Host tier; see [Disk tier](#disk-tier) | off |
| `--disk-kv-gib N` | disk tier budget in GiB | `64` |
| `--disk-kv-restore` | seed a new request's matching prefix from the disk tier | write-only |
| `--disk-kv-directstorage` | read restores through Microsoft DirectStorage; Windows builds with `-DNINFER_DIRECTSTORAGE=ON` only, untested | mapped reads |
| `--context-cache-policy default\|rolling` | `rolling`: within one cache session (a Responses `prompt_cache_key`), a capture that extends a resident checkpoint the request matched exactly inherits that resident's demand, so a conversation whose prompt only grows keeps rolling its frontier forward; with conversations sharing a prefix, one conversation's extension can evict the prefix the others use | `default` |
| `--concurrent-prefill` | admit waiting requests to free lanes while other requests prefill, instead of holding admission until the staged prefill finishes | off |
| `--kv-lease-growth` | admission reserves the prompt plus a 4096-token output window (or `--prefill-chunk` when larger) instead of the whole `max_tokens` budget and extends it at decode-round boundaries, releasing idle retained cache owners least recently used first when the pool is short; an answer whose smallest step still does not fit ends with `finish_reason=length` before `max_tokens` | off |
| `--recency-eviction` | pressure ranks private conversations and shared prefixes in one least-recently-used order (latest hit or publication): a request that does not fit gives up the fewest oldest owners that make it fit and spares those it does not need, instead of clearing every owner it cannot reuse; kept owners, shared prefixes included, are demoted to Host where it has room, and the search may fully evict only inside that tail and only an owner Host cannot take. A private checkpoint that finds the state pools full reclaims the idle conversation that is oldest in this order, not the oldest publication | off |
| `--value-aware-demote` | pressure search charges evicting a private conversation its rank by rebuild cost among the request's victims, so among otherwise equal plans it keeps the conversations most expensive to rebuild, demoted to Host, and evicts the cheapest; it only reorders a bounded search, which can then settle on a different plan | off |
| `--thorough-admission-search` | a new request's admission searches up to 250 ms for its reuse plan at every boundary, even while other requests decode (otherwise 50 ms idle, 10 ms busy), and a request whose plan saves more earns a longer base grant; every eligible option with an uncertain forecast gets its own short discovery grant (otherwise one per search), an option already generated for a candidate is always assessed, and the gain a search may spend against is the request's own re-prefill rather than a share of it per runnable request; a running decode can then pause up to that long when a request arrives | off |
| `--release-diverged-checkpoints` | a private checkpoint of one cache session whose next request diverges from it at the checkpoint's frontier keeps no retention value, so it is the first to go when the cache needs room; a checkpoint that merely cannot serve the request, another session's or a shared prefix, keeps its value. A client that switches back to an earlier branch of the conversation loses that branch's cache | off |
| `--no-thinking` | disable thinking by default | thinking on |
| `--preserve-thinking` | preserve closed-turn assistant reasoning by default | off |
| `--graft NAME=PATH` | load a [prompt graft](#prompt-grafts) a request may select by name; repeatable | none |
| `--default-graft NAME` | apply a loaded graft to requests that state none; `"graft": ""` opts out | none |
| `--cors` | permissive browser CORS headers | off |
| `--usage-chunk-choice` | give the streamed usage chunk a zero-delta choice, for strict client parsers that reject the OpenAI-conformant empty `choices` array | off |
| `--temperature F` | process-level temperature override | unset |
| `--top-p F` | process-level top-p override | unset |
| `--top-k N` | process-level top-k override (`0..20`; zero selects the top-20 cap) | unset |
| `--min-p F` | process-level min-p override | unset |
| `--presence-penalty F` | process-level presence-penalty override | unset |
| `--frequency-penalty F` | process-level frequency-penalty override | unset |
| `--seed N` | fixed seed when a request omits one | fresh random seed per request |
| `--greedy` | force exact argmax for all requests | off |
| `--post-thinking` | sample the answer of every thinking request with the post-thinking preset once its reasoning closes | off |
| `--post-thinking-temperature F`, `--post-thinking-top-p F`, `--post-thinking-top-k N` | post-thinking overrides; each implies `--post-thinking` | preset |
| `--post-thinking-sampler temp=F,top_p=F,top_k=N[,min_p=F,presence=F,frequency=F]` | the same overrides in one flag | preset |

Context-cost coefficients resolve once at startup from generic defaults, matching compiled values,
and optional transfer or prefill entries from `--context-cost-presets FILE`. Prefill entries match
the hardware and a signature derived from the actual Text/Vision configuration, bindings and Uses.
A new representation without a matching measurement uses generic prefill coefficients. A malformed
file aborts startup; the operational context-cost record and JSONL `server_start` identify the
selected source.

Engine selects sampling defaults from the loaded architecture and the request's resolved thinking mode.
Qwen3.6-27B, Qwen3.8-27B and the artifacts built from it, and Qwen3.8-Flash-Next use
`1.0/0.95/20/0/0` for temperature/top-p/top-k/min-p/presence penalty in thinking mode and
`0.7/0.80/20/0/1.5` in non-thinking mode. Qwen3.6-35B-A3B differs only in its thinking presence
penalty, which is `1.5`.
Frequency penalty is `0` for all registered presets. Process flags override registered values,
request fields override process flags, and `--greedy` finally forces temperature `0`.
A request that offers tools (`tools` with a `tool_choice` other than `none`, on every protocol)
samples without the presence penalty unless the request or `--presence-penalty` sets one: a second
tool call repeats the first call's markup and function name, and under `1.5` the model drops calls
it would otherwise make. The request log records the resolved value.

### Post-thinking sampling

A thinking request samples its reasoning and its answer with one preset. Post-thinking sampling
switches the device sampler once, at the first decode round after the model closes its reasoning
block, to a second preset: the thinking preset with temperature `0.2`. Tokens that a speculative
round accepts together with the close keep the reasoning sampler. It is off by default. The server enables
it for every thinking request with `--post-thinking` or any `--post-thinking-*` override; a single
request enables it with a `post_thinking` object on Chat Completions, Responses or Messages:

```json
{"post_thinking": {"temperature": 0.1, "top_k": 10}}
```

The object accepts `temperature` (`0..2`; `0..1` on Messages), `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty` and `seed`; an empty object selects the preset. Request
fields override the server's post-thinking flags, omitted fields come from the preset, an omitted
seed continues the request's seed, and `--greedy` forces temperature `0` in both phases. A request
without thinking, or one whose reasoning never closes, keeps its original sampler. The JSONL
`request_done` record reports `post_thinking_sampling` when the switch happened.

For `C=--max-concurrency` and `H=--device-state-slots`, total Device StateImage capacity is `C+H`:
`C` slots guarantee active requests and `H` is a global checkpoint pool. Host State and Host KV are
independent startup-fixed pinned-memory capacities; Host KV is shared by Main and the selected
Backend pool and is consumed in physical page extents. `--no-prefix-reuse` selects root-only Engine
mode and cannot be combined with any of the seven explicit context-cache capacity flags, including
zero-valued flags.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

Serve writes human-readable operational records to stderr using
`YYYY-MM-DD HH:MM:SS.mmm  LEVEL  message`. Normal output covers material startup milestones,
readiness, request lifecycle, fixed-interval throughput, and shutdown; `--log-level debug` exposes
internal startup and resource-planning detail. FFmpeg's media-decoding messages are records
prefixed `media |`, so they never write inside the statistics panel: FFmpeg errors are warnings,
and its warnings and notices, such as swscaler's `deprecated pixel format` notice for each JPEG,
are `debug`. A terminal may use one transient line during startup, but Serve throughput is always a
persistent record. Redirected stderr contains no terminal control
sequences.

With `--log-stats-panel on`, on an interactive terminal that accepts VT cursor control and at `info`
verbosity or more, Serve pins a session statistics panel beneath the scrolling records.
Records scroll above it and remain in the scrollback; the panel is redrawn after each record and
left on screen as ordinary output at exit. It has one row over every completed request and, once
more than ten have completed, one over the last ten:

| Column | Aggregate |
|---|---|
| TTFT | mean time to first token |
| cached | prefix-cache hit tokens / prompt tokens |
| prefill | computed (non-cached) prompt tokens / prefill seconds, in tok/s |
| decode | output tokens after the first / decode seconds, in tok/s (per-request rate, not batch throughput) |
| `<DRAFTER>`, acc/rnd | model-drafter (MTP or DFlash) accepted / drafted tokens, and accepted tokens per model-drafted round; n-gram rounds are excluded |
| ngram, ng rnds | n-gram accepted / drafted tokens and verification rounds |
| archive | n-gram archive accepted / drafted tokens, shown once the archive has drafted |

The table is 77 columns wide, 86 with the archive column, so it fits a console window snapped to
half of a 1920-pixel screen; a narrower window cuts the rows at its edge.

Ratios and rates divide summed tokens by summed seconds, so each request weighs by its size. The
title counts completed, failed, cancelled, and rejected requests and, while throughput reporting is
enabled, the current running and waiting requests. The panel only reads the same outcomes as the
`req#N done` records; it changes no request behavior.

Pretty values use readable units and rounded rates; use the independent request JSONL for complete
fields and full precision. Operational records never contain prompts, generated text,
request bodies, credentials, or arbitrary client error messages.
If a tool marker is returned to text because its structure or tool identity cannot be represented,
Serve emits one warning with only the failure classification, never the generated markup.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

Add `--request-log-jsonl profiles/bench/run/server.requests.jsonl` to the startup command to write
the log at that path.

Every line is one `ninfer_serve_request_log` schema-v28 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | artifact path, architecture, public name, actual formats and prefill signature; resolved Engine and context-cache capacities and switches, n-gram drafting options, thinking/non-thinking sampler defaults plus process overrides, thinking-history and thinking-budget defaults, Device arenas, the optional non-additive Vision layout inside the unified workspace, Host State/KV capacity and occupancy, KV sizing ledger, CUDA Graph allowance and measured cost, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, requested reasoning effort, actual initial thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, requested reasoning effort, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, tool-call parse diagnostics, request-owned materialization cost/search diagnostics, thinking-budget application counters, unrounded request-stage seconds, per-request Engine Host exposure, and complete speculative-decoding counters including n-gram and draft-archive counters |
| `request_error` | the resolved request configuration and the generation, cancellation, or pre-outcome transport terminal message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges, and decode-round batch statistics |

`requested_reasoning_effort` and `preserve_thinking` record the explicit options, or `null` when
unspecified. `enable_thinking` records whether the response starts in thinking mode.

`request_done.materialization` is the context-cache decision committed for that request: predicted
immediate, future-loss and total nanoseconds (`predicted_now_ns`, `predicted_future_loss_ns`,
`predicted_total_ns`, and `initial_predicted_total_ns` before search); `targets_evaluated`,
`projection_work`, `planning_elapsed_ns` and `search_elapsed_ns`; `stop_reason`; `budget_exhausted`;
`selected_degradation_units` and `selected_maximal_fallback`; `best_reuse_prompt_tokens`, the most
prompt reuse any admission candidate offered regardless of the plan chosen (beside a `root` plan, 0
means no reusable prefix was found and a large value means the planner priced reuse out); and the
optional-search accounting
`first_improvement_ns` (or `null`), `incumbent_improvements`, `search_work`, `search_granted_ns`,
`search_renewals`, `search_discovery_used`, `search_overshoot_ns`, `search_stop_phase` and
`search_boundary_limited`. Stop reasons are `no_pressure`, `queue_exhausted`, `target_budget`,
`expansion_capacity`, `time_budget`, `insufficient_expected_gain` and `work_budget`; `time_budget`
means the wall or control allowance ran out, while a request too cheap to justify optional search
reports `insufficient_expected_gain`. Search phases are `none`, `setup`, `construction`,
`assessment`, `expansion` and `refinement`. Search is bounded and heuristic; these diagnostics do not
claim model or global optimality, and aborted planning attempts are not published. With the
environment variable `NINFER_MAT_DEBUG` set, admission also writes its raw inputs to stderr: a
`[candgen]` line for each catalogued checkpoint it did or did not turn into a candidate, and
`[mat-debug]` lines for each identity cost, fast selection, assessed target and denied search step,
the last with the gain and economic bound behind the denial.
`cached_prefix_tokens` and `restored_host_bytes` describe a hybrid prefix-cache admission and are `0`
with the checkpoint catalog. `cached_prefix_tokens` is the longest prompt prefix held as cached KV
blocks, whether or not it was reusable: reuse also needs a state snapshot inside it, so a gap to
`prefix_cache_hit_tokens` is prefix lost to snapshot placement. `restored_host_bytes` is what the
admission copied back from the Host tier; those copies overlap the request's first prefill pass, so
their time is part of its prefill.

`request_done.result.tool_call_parse` records whether a complete marker was seen, the structured
call count, empty non-string arguments omitted during normalization, schema-mismatched arguments
preserved for consumer validation, `duplicate_parameters_repaired`, and a stable fallback reason. A
parameter named more than once in one call keeps its last value, as in JSON object syntax, and
counts once in `duplicate_parameters_repaired` for each repeat instead of demoting the call to text.
`forced_call_closed` reports a forced call whose closing tag the decoder supplied. When the strict
pass rejects a region, `recovered` reports that a recovery pass still produced calls from it (with
`recovered_call_count` the calls only that pass kept), `malformed_call_reported` that an unreadable
call became a call to the reserved error tool, and `trailing_content_dropped` that text after the
last call was dropped. Fallback reasons are `none`, `malformed_structure`, `invalid_tool_name`,
`undeclared_tool`, and `trailing_content`. These counters contain no tool arguments or generated
text.

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`, and under
`--adaptive-mtp` also `window_transitions` and `rounds_per_window`. Rates can be derived downstream
from raw token counts and seconds instead of rounded stderr strings. `ngram_rounds`,
`ngram_drafted_tokens`, and `ngram_accepted_tokens` are the part of `rounds`, `drafted_tokens`, and
`accepted_tokens` whose proposal came from n-gram copy drafting, and `ngram_archive_rounds`,
`ngram_archive_drafted_tokens`, and `ngram_archive_accepted_tokens` the part of those whose copy
source was the retained draft archive (see [n-gram drafting](ngram.md)). The
`speculative.ngram_archive` object reports that archive at the end of the request: `enabled` when
the server has one (`--ngram-archive-mib`), `bound` when the request was bound to a draft session,
`published` when its input and output were published into it, `generation` as the session's latest
completed generation, `sources` and `session_bytes` as the bound session's retained source count and
bytes, `total_bytes` as the whole archive's bytes, and `sampling_seed` as the effective seed after
request-domain separation for a request that named a session, otherwise `null`.

For `server_start.memory`, `workspace.capacity_bytes` is the only physical workspace allocation.
When Vision is enabled, `vision_workspace` reports the `residency` (`resident`, `overlay` or `cpu`),
the aggregate prompt and maximum-item token bounds plus encode peak and handoff layout/usage within
that same allocation; these bytes must not be added to `workspace.capacity_bytes`. The field is
`null` when Vision is disabled.
`cuda_graph_allowance_bytes` is the CUDA Graph memory the KV sizing reserved, and
`cuda_graph_measured_bytes` the Device memory graph preparation actually took at startup (`0`
without CUDA Graphs); the startup log warns when the second exceeds the first.

`server_start.engine` records `ngram_draft_window` and `ngram_min_match` (`--ngram-draft-tokens`,
`0` when n-gram drafting is off, and `--ngram-min-match`), the draft-archive budgets
`ngram_archive_bytes` and `ngram_session_bytes` (`--ngram-archive-mib`, `0` keeps drafting
request-local, and `--ngram-session-mib`), and `ngram_native_sessions`. Its `context_cache` object
records the resolved capacities and the `rolling_retention`, `release_diverged_checkpoints`,
`thorough_admission_search`, `recency_eviction`, `value_aware_demote`, `kv_lease_growth`,
`automatic_long_anchors`, `branch_anchors`, `endpoint_anchors` and `long_anchor_min_spacing_tokens` settings, the
cache `mode` (`legacy` or `hybrid`), the
`host_cache_budget_bytes`, and under `--use-alt-prefix-caching` a `hybrid` object with the resolved
snapshot slots, taps per request, tap ladder and minimum gap and whether the Host tier persists.

`request_done.engine_timing` separates FIFO `queue_wait_seconds`, blocking
`device_wait_exposed_seconds`, and five mutually exclusive Host-active exposure phases under
`host_exposed_seconds`: `engine_boundary`, `program_submit`, `program_post`,
`engine_commit_output`, and `engine_maintenance`. `total` is exactly their sum and excludes Device
wait. The nested `decode` object reports the request's decode-class Host exposure, Device wait, and
round count; `units` reports its prefill/control unit counts. In a compact batch every participating
request is delayed by the full round, so these values explain request latency but **must not be
summed across concurrent requests**.

The JSONL file contains no generated response text and never records an API-key value; `argv`
replaces that value with `<redacted>`. The existing stderr summaries remain available for operators
but are rounded and are not the aggregation source. Console lines use local
`[YYYY-MM-DD HH:MM:SS.mmm] [level]` timestamps. Engine runtime diagnostics are ordinary records
prefixed `engine |` that `--log-level` filters: the device route profile the Engine installs and a
calibration it runs are `info`; an ignored or unsaved profile, recovery from out of memory or,
with `--recover-invariant-failures`, from a broken invariant, and context stores rebuilt after it
are warnings; a worker crash or repeated recoveries that fail every pending request are errors. OpenAI Responses, OpenAI Chat, and Anthropic
generation requests receive a request ID when they enter synchronous preparation. Successful
preparation produces `request_start`; a preparation failure produces `request_rejected` without a
matching start. Later generation failures produce `request_error`. Schema/model validation
rejections before preparation and token-count-only calls are not measurement requests and do not
receive request IDs.

By default the server persistently reports aggregate activity every five seconds. `prefill` counts
prompt suffix tokens actually computed during the interval, excluding prefix-cache hits; `decode`
counts tokens finally committed by decode rounds, excluding the first token produced by prefill.
For MTP, DFlash and DFlash2 this is the accepted committed output, not draft or rejected tokens.
Pretty `batch` and JSONL `average_size` are decode row-rounds divided by decode rounds during the
same interval. The
`running`, `prefilling`, `decode_ready`, `waiting`, `materializing`, `capture_pending`, and
`terminal_pending` fields are the Engine scheduler snapshot at the end of the interval. The JSONL
`context_cache` object reports selection, capture, transfer, COW, pressure spill, private/shared
owner degradation and eviction, private owner demotion to Host (`pressure.private_owners_demoted`, a
subset of `private_owners_degraded`), checkpoint drop, pressure search, budget exhaustion, maximal fallback, and historical-fork
counters as interval deltas; `occupancy` and `last_selection` are end-of-interval gauges. Materialization predictions are
request-owned and appear only on the corresponding `request_done` event. The top-level
`engine_recoveries` field is the interval delta of the worker-recovery counter described above; it
is not part of `context_cache` because a recovery is an engine-wide event, not a context-cache
operation.
`pressure.searches` counts plans accepted into Program resource transactions, including a transaction that later ends in
request-local abort; committed victim counters likewise report the resulting stable cache changes.
`salvage.published` counts cancelled requests whose live state was published as a continuation
endpoint at its last committed frontier.
`captures.skipped` counts capture offers the Program declined because they were not physically
feasible: unlike `captures.aborted` it has no other trace, so it is the counter for silent retention
loss. `occupancy.device_main_kv_lease_pages` and `device_backend_kv_lease_pages` are the part of
`device_main_kv_pages` and `device_backend_kv_pages` that active requests hold as reservation but
have not yet written.

With `--use-alt-prefix-caching`, `context_cache.hybrid` is present once the cache has inserted a
block or holds a snapshot. `device_blocks` (Device-resident 64-token KV blocks), `evictable_blocks`
(those Device eviction may drop now), `tree_blocks` (blocks on the Device or Host), `snapshots`,
`host_capacity_bytes` and `host_used_bytes` are end-of-interval gauges; the rest are interval
deltas. `snapshot_hits` counts admissions that resumed from a snapshot and `reused_tokens` the
prompt tokens they reused. `blocks_inserted` counts new tree blocks, `blocks_reattached` blocks whose
existing tree entry took a request's Device pages, and `blocks_duplicate` committed blocks the tree
already held on the Device, whose pages were released. `taps_created` and `taps_skipped` count
planned prefill snapshots published and dropped, and `endpoints_created` end-of-answer snapshots.
`host_image_writes`, `host_block_writes`, `host_image_restores`, `host_block_restores`,
`host_write_bytes` and `host_restore_bytes` count Host-tier write-through and restores.
`evicted_blocks` counts Device block evictions, `host_snapshot_evictions` snapshots evicted from the
Host tier, `host_dead_reclaims` Host slabs reclaimed from KV that no snapshot can reach, and
`unbacked_node_losses` Device evictions of blocks with no Host copy, which remove them and the
blocks after them from the cache.

The JSONL `throughput.host_work` object is the aggregation authority: the Engine worker counts each
wall-time segment once, independent of batch size. `elapsed_seconds` contains the same five
mutually exclusive Host phases and their `total`; `device_wait_seconds` is separate.
`work_class_seconds` splits Host and Device-wait time into decode, prefill, and control classes.
`detail_subset_seconds` and `detail_invocations` expose admission, context-transaction, replica, and
stats-publication slow paths; these detail values are already contained in a top-level Host phase
and must not be added to `total`. Per-round, per-row-round, and per-invocation normalized values are
`null` when their denominator is zero. The stderr interval line shows only total Host milliseconds,
decode Host/device-wait microseconds per round, boundary, and maintenance; use JSONL for analysis.
Intervals with context materialization or retention activity are retained even when they contain no
token execution; only fully idle intervals are omitted. Downstream measurement should prefer the
raw counters and seconds over rounded stderr rates.

### Disk tier

`--disk-kv-path DIR` adds a disk tier under the Host tier. When a private continuation is evicted,
by pressure or by the catalog limit, it is written there before its memory is released: its KV pages
as a prefix chain, the StateImage at its endpoint, and the tail page and StateImage of its rewrite
seam and of its three earliest long anchors. The continuations still resident at shutdown are
written the same way. Pages are keyed by the prefix digest at their last token and by the execution
profile, so an identical prefix is stored once, and the files survive restarts. Each artifact gets
its own subdirectory and each KV format, stride and YaRN setting its own files, so one model or
profile never restores another's KV. `--disk-kv-gib` (default 64) is the total budget, split
65/25/10 across Main KV, Backend KV and StateImages (85/15 without a backend); the least recently
used page goes first. The files are sparse until written, and a torn page fails its CRC and reads as
a miss.

With `--disk-kv-restore`, a request with no resident prefix looks for the longest frontier `E` on
disk whose StateImage and KV chain match its own prompt, is admitted as reusing `E`, and has `[0, E)`
copied into its sequence before prefill starts at `E`; under MTP the StateImage's hidden state
extends the draft KV to `E`. A restore is abandoned after 8 seconds, and any miss or failure falls
back to recomputing the prompt, so a result never depends on the disk contents. Restores read with
eight threads and run on the engine thread, so other lanes pause during one; media prompts are never
restored. Without the flag the tier only writes. Writes of an evicted owner during admission proceed
a batch at a time between decode rounds; an owner released elsewhere is written within 20 seconds,
and shutdown writes for at most 60.

On Windows, a build configured with `-DNINFER_DIRECTSTORAGE=ON` fetches the DirectStorage 1.3
runtime and offers `--disk-kv-directstorage`: a restore then reads its pages into host memory
through a DirectStorage queue, every page of a staging batch in flight at once, instead of copying
from the mapped files. Each page is still checked against its CRC, and a batch that fails or does
not complete within 10 seconds falls back to the mapped reads. This path has been checked to compile
against Windows headers but has not been run.

## Execution behavior

The server owns one resident Engine with a startup-fixed capacity of `1..8` active generation
requests. At each decode boundary, every decode-ready request is compacted into one batch and
processed by one model traversal and, when graphs are enabled, one exact-batch CUDA Graph replay.
A request joins that batch only after its staged prefill finishes; when it completes or is
cancelled, the next boundary rebuilds the batch without an empty row. A staged prefill holds
admission back until it finishes. With `--concurrent-prefill` waiting requests are still admitted
to free lanes while others prefill, so a new request's admission and prefill overlap the prefill
and decode of the others (each prefill unit advances one chunk of the lowest staged lane per
worker boundary).

`--max-pending-requests` bounds the requests waiting behind the active set. The total generation
request lifetime capacity is `max_concurrency + max_pending_requests`, including requests still in
CPU/media preparation and completed model results whose response has not yet been released. A full
capacity returns HTTP 429 with code `server_overloaded`. Token-count requests (Anthropic
`count_tokens`, Responses `input_tokens`) run the same preparation without generating; a separate
capacity of the same size bounds how many run at once, and beyond it they are rejected as
overloaded, as generation requests are. The absolute
`--pending-timeout-ms` deadline starts before preparation, covers media acquisition and Engine FIFO
waiting, and returns HTTP 503 with code `request_queue_timeout` if admission does not occur in time.
There is no admission ETA or unbounded overflow queue. Because there is no preemption, a queued
request waits out the generations ahead of it, so the deadline has to be scaled to the longest
response the deployment allows rather than to a connection timeout: at C1 on an RTX 3090 a single
6,500-token response occupies the engine for about 106 seconds. The 600,000 ms default admits a
queued caller behind roughly ten such responses; lower it only to fail fast on purpose.

One request owns the staged prefill at a time, and the executor alternates a single prefill chunk
with a single decode round, so `--prefill-chunk` sets the worst-case pause every active stream sees
while a new prompt is ingested. On an RTX 3090 ingesting a 4,900-token prompt behind four active
streams, the largest inter-token gap measured 1,043 ms at chunk 1024, 515 ms at 512, and 312 ms at
256, against an 82 ms median decode interval; the ingesting request's own prefill rate fell only
from 1,135 to 1,130 to 1,110 tok/s. Prefill is not batched across requests at any chunk size, so a
smaller chunk trades almost no ingestion throughput for a proportionally smaller stall. The shipped
concurrent launcher uses 512.

Input memory is bounded by the outstanding-request count and the per-request
`--max-request-mib` limit. Media requests additionally share one preparation permit, so a waiting
media request retains the same cancellation and timeout deadline. Model output is bounded by the
same finite request count and each request's effective output-token limit; output callbacks and
network serialization run outside the GPU executor and do not delay formation of the next batch.

`--max-context` is each sequence's logical ceiling. `--kv-capacity` fixes the shared Main Text KV
pool used by active requests and retained prefixes. `auto` accounts for the complete enabled runtime
and leaves 1 GiB of sizing headroom; omitting the option makes it follow `--max-context`. The
CUDA Graph driver-state allowance reserved against that budget is computed from the enabled
graph profiles and concurrency unless `--cuda-graph-allowance-mib` supplies an explicit total; a
too-small value risks CUDA out-of-memory at graph capture and a too-large one shrinks the KV pool.
Capacity resolves once at startup.

Admission reserves the full prompt-plus-effective-output page entitlement through request
completion. A request remains queued until a legal resource plan can satisfy that entitlement.
With `--kv-lease-growth` admission instead reserves the prompt plus a bounded Device KV window over
the remaining output (4096 tokens, or `--prefill-chunk` when larger) and extends it at each
decode-round boundary, so a client that asks for far more output than it generates does not hold
the prefix cache out of the pool. When the pool cannot extend the window, idle retained cache
owners are released least recently used first; if the smallest step still does not fit, the
request completes at the frontier its window covers and reports `finish_reason=length` before
reaching `max_tokens`.

### Default output limit

A request that omits its output limit (`max_completion_tokens`/`max_tokens` on Chat Completions,
`max_output_tokens` on Responses, `max_tokens` on Messages) receives the largest budget that still
lets every configured lane be admitted at the same time. Once its prompt is prepared, the Engine
finds the largest output whose admission entitlement -- Main KV pages for the prompt and output,
plus the MTP draft-window or DFlash backend KV pages when speculation is on -- fits one lane's share
(`1/--max-concurrency`) of each KV pool, and clamps it to the remaining context
(`--max-context` minus the prompt). With one lane, or a pool of at least `--max-concurrency` times
`--max-context`, that is the whole remaining context, so long reasoning runs are not cut at an
arbitrary count; with several lanes over a smaller pool, each limitless request stays inside its
share and they all run concurrently. A prompt that alone overruns one lane's share can never run
beside full-share lanes, so it keeps the whole remaining context.

A run that exhausts the budget finishes with `finish_reason:"length"`, Responses `incomplete` with
reason `max_output_tokens`, or Anthropic `stop_reason:"max_tokens"`, or
`stop_reason:"model_context_window_exceeded"` when the budget was the remaining context. An explicit
request limit always wins, and `--default-max-tokens N` replaces the derived default with a fixed cap
(still bounded by the remaining context). Chat Completions `max_tokens: -1`, llama.cpp's "no limit"
that its WebUI sends by default, always receives the derived budget, even with
`--default-max-tokens`. The JSONL request record reports the budget actually
submitted as `requested_output_tokens`.

Each reusable checkpoint contains KV and complete continuation state. At admission, capture, and
finish boundaries, resource pressure may keep it on Device, move its StateImage and/or KV replicas
to pinned Host memory, or evict it. The planner compares incoming-request work with the later
recovery cost imposed on retained checkpoints. Active requests retain their state and completion
reservations, and placement choices preserve model semantics. The full policy and invariants are
defined in [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md).

Compatible prefixes are reused for both text and multimodal histories unless the server starts with
`--no-prefix-reuse`. A multimodal hit additionally requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans. Media wholly inside a matched prefix
skips Vision execution, while new suffix media is encoded normally. The completion log reports the
reused token count as `cache=`.

The completion log reports one of six reuse paths: `root`, `private_endpoint`,
`private_turn_closure`, `private_response_replay`, `private_long_anchor`, or
`shared_stable_prefix`. Reuse validation covers KV, recurrent state, hidden state, selected-backend
state, and the exact prompt frontier. With stable `preserve_thinking=true`, the auxiliary checkpoint
rolls to the message frontier immediately before the current response's deterministic generation
prologue. A normalized response, compact-summary instruction, or replacement user suffix therefore
replays the small generation prologue and only the changed suffix while retaining the complete
stable conversation prefix. Stable `false` places the turn-closure checkpoint before the first
assistant opener in the open turn, so closing that turn can recompute its opener and omit its
reasoning without discarding the preceding conversation.

`preserve_thinking` selects the capture frontier for newly created checkpoints. Existing exact
checkpoints remain reusable across a mode change. If the desired boundary is behind the selected
reuse frontier and has no snapshot, the Engine keeps the valid hit and defers the new checkpoint. A
later request that diverges before every retained checkpoint starts from root. The JSONL completion
record exposes the restored checkpoint as `prefix_reuse_path`. Reasoning-effort changes participate
in rendered-token identity and exact-prefix selection.

An appended mid-conversation system message is an ordinary prompt suffix, so an unchanged prior
history remains eligible for `private_endpoint`. If the client modifies, removes, or moves a
historical system message, the token prefix genuinely differs and a miss/reset is correct.

Speculative backends preserve protocol output shapes, stop behavior, and usage accounting. If a stop
truncates a multi-token MTP, DFlash or DFlash2 round, the Engine commits the exact accepted target prefix so
a following compatible turn can reuse it. Output-limit and context-capacity finishes map to
`length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools or constrain their arguments. Final JSON responses can
use [structured output](#structured-output).

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
