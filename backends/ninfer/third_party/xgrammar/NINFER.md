Vendored XGrammar v0.2.7, commit 82505d0d987c36a4209fb3d8571cf6b0f28b5acd.
Source: https://github.com/mlc-ai/xgrammar
Apache-2.0; see LICENSE. Native C++ sources only; no Python/TVM dependency.
DLPack commit bbd2f4d32427e548797929af08cfe2a9cbb3cf12, see its LICENSE.
PicoJSON and its license are included from the pinned XGrammar tree.
NInfer supplies a target-scoped CMake build without downloads. Local correctness patch in cpp/json_schema_converter.cc: bounded and unbounded JSON
strings share a Unicode scalar FSM; raw control bytes are excluded while equivalent
legal JSON escapes remain available.
Regression: tests/text/test_structured_output.cpp. The additional-property exclusion trie uses Unicode codepoints and excludes escaped aliases of
declared properties before divergence, preventing duplicate-key overwrites from bypassing a
property schema. This restricts some otherwise valid key spellings.
The local NInfer changes below also modify the FSM string path.

Numeric-range correctness patch: GenerateNumber rejects the empty-range regex sentinel instead of
interpreting it as an empty JSON number. This matters when the six-decimal generation grid contains
no value inside an otherwise nonempty real interval. Covered by the structured-output grammar tests.

NInfer supplements: JSON-string regex compilation never falls back to plain EBNF; pattern
search semantics are preserved with JSON-escaped prefix/suffix matching. Temporal regexes
validate Gregorian month lengths, leap years and RFC 3339 offsets. Native entry validation
rejects unsupported conjunctions instead of dropping constraints. Regression coverage is
in tests/text/test_structured_output.cpp.

NInfer scalar and escape supplement: Unicode FSM transitions accept only legal scalar
values and canonical UTF-8. JSON-string regexes match decoded logical characters through
raw UTF-8, short JSON escapes and Unicode escape spellings. ECMA262 dot and whitespace
sets are normalized by the schema converter; unsupported regex assertions fail closed.
CPU coverage includes test_unicode_scalar_output.cpp and the native_schema_probe.

Schema compatibility supplement: the text compiler distributes proven
anyOf type/required intersections before XGrammar compilation. String-array
uniqueness is enforced by request-owned incremental semantic masks (src/text/string_constraints.cpp), not by dropping
the assertion from public JSON or by rewriting completed arrays. Object upper
bounds use the existing vendored object generator. CPU equivalence, prefix
reachability and committed/speculative-state regressions guard these contracts.

Counted-repetition supplement (from upstream XGrammar after v0.2.7, generalized in Neroued/ninfer
0a8530d44): a repetition whose unit is a deterministic, prefix-free byte language, followed by a
byte that cannot start a unit, stays one counted repeat edge in a rule of its own
(`RepetitionRangeExpanderImpl::TryCountedRepeat`), and the unit's FSM is normalized to a DFA. For
every state of such a unit, `PopulateRepeatMasks` classifies the tokens by the repetitions they
consume, including partial UTF-8 and escape states: tokens that stay inside go into cumulative
bitsets (or sorted sparse lists) by count, tokens that leave on their last byte into an exit list,
and both leave the mask's ordinary classes. The matcher reads the count from the repetition parent
and accepts the fitting tokens without replaying them; tokens that cross into the parent's
continuation keep the ordinary replay. The data is rebuilt on compilation and deserialization, not
serialized; the serialization version is `v16-ninfer-repeat-masks`. A JSON string with
`minLength`/`maxLength` compiles to `"\"" scalar{min,max} "\""` over one shared JSON-scalar unit.
With Qwen3.5's 248,070-token vocabulary (Apple M-series, one thread) a schema with an `anyOf`
string of `maxLength: 200` and one of `maxLength: 2800` compiles in 81 ms and 9.8 MB instead of
24.8 s and 3.1 GB, and its masks inside the strings take 0.04 ms instead of 70-96 ms. Regression:
tests/text/test_structured_output.cpp compares every mask inside bounded strings with token
acceptance over a vocabulary of multi-scalar, partial and closing tokens.


Qwen XML parameter supplement (cpp/json_schema_converter_ext.cc): the `qwen_xml_parameter` value
wrapper carries the newlines of Qwen's chat template, `<parameter=NAME>\nVALUE\n</parameter>\n`,
which the model writes and NInfer's tool-call parser strips, so a raw string value is exactly the
argument. A whitespace-only wrapper prefix is not preceded by another whitespace run, which would
otherwise reach a raw string's value. Strict tool arguments depend on it; coverage is in
tests/test_tool_call_parser.cpp.
