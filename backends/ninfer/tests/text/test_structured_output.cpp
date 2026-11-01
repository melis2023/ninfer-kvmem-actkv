#include "ninfer/request.h"
#include "text/structured_output.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace ninfer;
using namespace ninfer::text;

void require(bool value, const char* what) {
    if (!value) { throw std::runtime_error(what); }
}

int main() {
    try {
        std::vector<std::string> vocab(257);
        for (int i = 0; i < 256; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
        StructuredCompiler compiler(vocab, {256});
        const auto schema =
            R"({"type":"object","properties":{"a":{"type":"string"},"b":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2}},"required":["a","b"],"additionalProperties":false})";
        auto state          = compiler.compile({StructuredOutputKind::JsonSchema, schema});
        constexpr int words = 9;
        std::vector<std::uint32_t> before(words), after(words);
        state->fill_masks(before, {});
        require((before['{' / 32] & (1U << ('{' % 32))) != 0, "object start masked");
        require((before['[' / 32] & (1U << ('[' % 32))) == 0, "schema permits wrong root");
        require((before[8] & 1U) == 0, "premature EOS permitted");
        auto preview           = state->fork();
        const std::string json = "{\"a\":\"é😀\\n\\\"\",\"b\":[-1,23]}";
        std::vector<TokenId> ids;
        for (unsigned char ch : json) { ids.push_back(ch); }
        preview->accept(ids);
        state->fill_masks(after, {});
        require(before == after, "preview advanced committed grammar");
        *state = std::move(*preview);
        state->fill_masks(after, {});
        require((after[8] & 1U) != 0, "completed JSON cannot stop");
        state->accept(std::vector<TokenId>{256});
        require(nlohmann::json::parse(json).at("b").size() == 2, "invalid JSON fixture");

        const auto open_schema =
            R"({"type":"object","properties":{"a":{"type":"integer"},"東京":{"type":"boolean"},"quote\"":{"type":"boolean"}},"required":["a","東京"],"additionalProperties":{"type":"string"}})";
        const auto accepts = [&](const std::string& value) {
            auto grammar = compiler.compile({StructuredOutputKind::JsonSchema, open_schema});
            std::vector<TokenId> tokens;
            for (unsigned char ch : value) { tokens.push_back(ch); }
            tokens.push_back(256);
            try {
                grammar->accept(tokens);
                return true;
            } catch (const std::logic_error&) { return false; }
        };
        require(accepts(R"({"a":1,"東京":true,"other":"x"})"), "open object lost valid extra key");
        require(!accepts(R"({"a":1,"東京":true,"\u0061":"bad"})"),
                "escaped key bypasses property type");
        require(!accepts(R"({"a":1,"東京":true,"東京":"bad"})"),
                "Unicode key bypasses property type");
        require(!accepts(R"({"a":1,"東京":true,"quote"":"bad"})"), "invalid key escape accepted");

        auto recursive = compiler.compile(
            {StructuredOutputKind::JsonSchema, R"({"type":"array","items":{"$ref":"#"}})"});
        std::vector<TokenId> recursive_tokens;
        for (unsigned char ch : std::string("[[],[[]]]")) { recursive_tokens.push_back(ch); }
        recursive_tokens.push_back(256);
        recursive->accept(recursive_tokens);

        auto bounded = compiler.compile(
            {StructuredOutputKind::JsonSchema, R"({"type":"string","minLength":2,"maxLength":2})"});
        bounded->accept(std::vector<TokenId>{'"'});
        bounded->fill_masks(after, {});
        for (int control = 0; control < 32; ++control) {
            require((after[control / 32] & (1U << (control % 32))) == 0,
                    "bounded string permits unescaped control character");
        }
        std::vector<TokenId> unicode;
        for (unsigned char ch : std::string("é😀\"")) { unicode.push_back(ch); }
        bounded->accept(unicode);
        bounded->fill_masks(after, {});
        require((after[8] & 1U) != 0, "bounded string did not count Unicode code points");

        // A root string's bounds leave the grammar like those inside an object; under anyOf they
        // stay and compile to a chain of rules (one per count still allowed). Either way every
        // length from minLength to maxLength closes, escapes and multi-byte scalars count once,
        // and neither a shorter nor a longer string does.
        const auto root_bounded = R"({"type":"string","minLength":2,"maxLength":40})";
        const auto chained =
            R"({"anyOf":[{"type":"string","minLength":2,"maxLength":40},{"type":"integer"}]})";
        for (const char* schema : {root_bounded, chained}) {
            const auto within = [&](const std::string& body) {
                auto grammar = compiler.compile({StructuredOutputKind::JsonSchema, schema});
                std::vector<TokenId> tokens{'"'};
                for (unsigned char ch : body) { tokens.push_back(ch); }
                tokens.push_back('"');
                tokens.push_back(256);
                try {
                    grammar->accept(tokens);
                    return true;
                } catch (const std::logic_error&) { return false; }
            };
            require(!within("a"), "string under minLength accepted");
            require(within("ab"), "string at minLength rejected");
            require(within(std::string(40, 'x')), "string at maxLength rejected");
            require(!within(std::string(41, 'x')), "string over maxLength accepted");
            require(within(std::string(38, 'x') + "\\n\\u00e9"), "escapes do not count as one scalar");
            require(!within(std::string(39, 'x') + "\\n\\u00e9"), "escaped scalars exceed maxLength");
            require(within(std::string(37, 'x') + "é😀\\\""), "multi-byte scalars do not count once");
        }
        {
            auto grammar = compiler.compile({StructuredOutputKind::JsonSchema, root_bounded});
            grammar->accept(std::vector<TokenId>{'"', 'x'});
            grammar->fill_masks(after, {});
            require((after['"' / 32] & (1U << ('"' % 32))) == 0, "mask closes a root string under minLength");
            grammar->accept(std::vector<TokenId>(39, 'x'));
            grammar->fill_masks(after, {});
            require((after['"' / 32] & (1U << ('"' % 32))) != 0, "mask cannot close a root string at maxLength");
            require((after['x' / 32] & (1U << ('x' % 32))) == 0, "mask extends a root string past maxLength");
        }

        // Inside an object the bounds leave the grammar and the string constraint state enforces
        // them: the same lengths close, and the masks at the bounds say so.
        const auto object_bounded =
            R"({"type":"object","properties":{"s":{"type":"string","minLength":2,"maxLength":40}},"required":["s"]})";
        const auto in_object = [&](const std::string& body) {
            auto grammar = compiler.compile({StructuredOutputKind::JsonSchema, object_bounded});
            std::vector<TokenId> tokens;
            for (unsigned char ch : "{\"s\":\"" + body + "\"}") { tokens.push_back(ch); }
            tokens.push_back(256);
            try {
                grammar->accept(tokens);
                return true;
            } catch (const std::logic_error&) { return false; }
        };
        require(!in_object("a"), "object string under minLength accepted");
        require(in_object("ab"), "object string at minLength rejected");
        require(in_object(std::string(40, 'x')), "object string at maxLength rejected");
        require(!in_object(std::string(41, 'x')), "object string over maxLength accepted");
        require(in_object(std::string(38, 'x') + "\\n\\u00e9"), "object escapes do not count once");
        require(!in_object(std::string(39, 'x') + "\\n\\u00e9"), "object escapes exceed maxLength");
        require(in_object(std::string(37, 'x') + "é😀\\\""), "object multi-byte scalars do not count once");
        {
            auto grammar = compiler.compile({StructuredOutputKind::JsonSchema, object_bounded});
            std::vector<TokenId> prefix;
            for (unsigned char ch : std::string("{\"s\":\"x")) { prefix.push_back(ch); }
            grammar->accept(prefix);
            grammar->fill_masks(after, {});
            require((after['"' / 32] & (1U << ('"' % 32))) == 0, "mask closes a string under minLength");
            std::vector<TokenId> fill(39, 'x');
            grammar->accept(fill);
            grammar->fill_masks(after, {});
            require((after['"' / 32] & (1U << ('"' % 32))) != 0, "mask cannot close a string at maxLength");
            require((after['x' / 32] & (1U << ('x' % 32))) == 0, "mask extends a string past maxLength");
            require((after['\\' / 32] & (1U << ('\\' % 32))) == 0, "mask starts an escape past maxLength");
            // A draft column conditioned on drafts follows the same bound.
            auto drafted = compiler.compile({StructuredOutputKind::JsonSchema, object_bounded});
            drafted->accept(prefix);
            std::vector<TokenId> drafts(38, 'x');
            std::vector<std::uint32_t> columns(words * (drafts.size() + 1));
            drafted->fill_masks(columns, drafts);
            const auto last = std::span(columns).subspan(drafts.size() * words, words);
            require((last['x' / 32] & (1U << ('x' % 32))) != 0, "draft column at 39 scalars cannot extend");
            require((last['"' / 32] & (1U << ('"' % 32))) != 0, "draft column at 39 scalars cannot close");
            // The columns, filled in parallel, equal the masks of the grammar after each draft
            // prefix one at a time; after a rejected draft they are unrestricted.
            for (const std::string draft : {std::string(45, 'x'), std::string("xx\"}") + std::string(3, 'x')}) {
                auto parallel = compiler.compile({StructuredOutputKind::JsonSchema, object_bounded});
                parallel->accept(prefix);
                const std::vector<TokenId> ids(draft.begin(), draft.end());
                std::vector<std::uint32_t> all(words * (ids.size() + 1));
                parallel->fill_masks(all, ids);
                bool reachable = true;
                for (std::size_t j = 0; j <= ids.size(); ++j) {
                    std::vector<std::uint32_t> single(words, ~std::uint32_t{0});
                    if (reachable) {
                        auto one = compiler.compile({StructuredOutputKind::JsonSchema, object_bounded});
                        one->accept(prefix);
                        try {
                            one->accept(std::span(ids.data(), j));
                            one->fill_masks(single, {});
                        } catch (const std::logic_error&) {
                            reachable = false;
                            std::fill(single.begin(), single.end(), ~std::uint32_t{0});
                        }
                    }
                    require(std::equal(single.begin(), single.end(), all.begin() + j * words),
                            "a parallel draft column differs from its sequential mask");
                }
            }
        }

        auto whitespace          = compiler.compile({StructuredOutputKind::JsonObject, {}});
        bool too_much_whitespace = false;
        try {
            whitespace->accept(std::vector<TokenId>(9, ' '));
        } catch (const std::logic_error&) { too_much_whitespace = true; }
        require(too_much_whitespace, "unbounded whitespace run accepted");

        auto object = compiler.compile({StructuredOutputKind::JsonObject, {}});
        std::vector<TokenId> drafts{'{', '"', 'x', '"', ':', '[', '1', ',', '2', ']', '}'};
        std::vector<std::uint32_t> masks(words * (drafts.size() + 1));
        object->fill_masks(masks, drafts);
        for (std::size_t i = 0; i < drafts.size(); ++i) {
            require(masks[i * words + drafts[i] / 32] & (1U << (drafts[i] % 32)),
                    "valid speculative draft masked");
        }
        require(masks[drafts.size() * words + 8] & 1U, "bonus mask missing EOS");
        object->fill_masks(before, {});
        require(before[0] == masks[0], "draft traversal advanced committed grammar");
        const auto tokens_for = [](std::string_view value) {
            std::vector<TokenId> tokens;
            for (unsigned char ch : value) { tokens.push_back(ch); }
            return tokens;
        };
        const auto accepts_value = [&](const std::shared_ptr<GrammarState>& grammar,
                                       const std::string& value) {
            auto trial  = grammar->fork();
            auto tokens = tokens_for(value);
            tokens.push_back(256);
            try {
                trial->accept(tokens);
                return true;
            } catch (const std::logic_error&) { return false; }
        };
        const auto accepts_schema_value = [&](const std::string& schema_value, const std::string& json_value) {
            return accepts_value(compiler.compile({StructuredOutputKind::JsonSchema, schema_value}), json_value);
        };
        const std::string anchored = R"({"type":"string","pattern":"^code-[AB][0-9]+$"})";
        require(accepts_schema_value(anchored, R"("code-A12")"), "anchored pattern lost valid value");
        require(!accepts_schema_value(anchored, R"("bad-code-A12")"), "start anchor ignored");
        require(!accepts_schema_value(anchored, R"("code-A12bad")"), "end anchor ignored");
        const std::string searched = R"({"type":"string","pattern":"code"})";
        require(accepts_schema_value(searched, R"("prefix-code-suffix")"), "pattern search became full match");
        require(accepts_schema_value(searched, R"("\ncode\n")"), "JSON escaping weakened regex search");
        require(!accepts_schema_value(searched, R"("prefix-only")"), "search pattern ignored");
        const std::string single = R"({"type":"string","pattern":"^.$"})";
        require(accepts_schema_value(single, R"("\"")"), "escaped quote does not match logical character");
        require(!accepts_schema_value(single, std::string("\"") + char(1) + "\""), "pattern allowed raw JSON control byte");
        const std::string calendar = R"({"type":"string","format":"date"})";
        for (const auto* value : {"2000-02-29", "2024-02-29", "2026-04-30", "0001-01-01"}) {
            require(accepts_schema_value(calendar, nlohmann::json(value).dump()), "calendar lost valid date");
        }
        for (const auto* value : {"1900-02-29", "2025-02-29", "2026-04-31", "2026-13-01", "0000-01-01"}) {
            require(!accepts_schema_value(calendar, nlohmann::json(value).dump()), "invalid calendar date accepted");
        }
        const std::string timestamp = R"({"type":"string","format":"date-time"})";
        require(accepts_schema_value(timestamp, R"("2024-02-29T23:59:59.125+08:00")"), "timestamp lost fractional timezone value");
        require(accepts_schema_value(timestamp, R"("2026-10-03t12:30:00z")"), "timestamp lost lowercase separators");
        require(!accepts_schema_value(timestamp, R"("2026-02-30T12:00:00Z")"), "timestamp ignored calendar date");
        require(!accepts_schema_value(timestamp, R"("2026-10-03T24:00:00Z")"), "timestamp ignored time bounds");
        require(!accepts_schema_value(timestamp, R"("2026-10-03T12:00:00")"), "timestamp lost required timezone");

        auto object_union = compiler.compile({StructuredOutputKind::JsonSchema,
            R"({"anyOf":[{}],"type":"object"})"});
        require(accepts_value(object_union, "{}") && !accepts_value(object_union, "1"),
                "anyOf common type was dropped");
        auto rating = compiler.compile(
            {StructuredOutputKind::JsonSchema, R"({"type":"number","minimum":0,"maximum":10})"});
        require(accepts_value(rating, "0") && accepts_value(rating, "10") &&
                    accepts_value(rating, "9.5"),
                "inclusive number bounds lost valid values");
        for (int i = -80; i <= 240; ++i) {
            const double value  = i / 16.0;
            const auto spelling = nlohmann::json(value).dump();
            if (accepts_value(rating, spelling)) {
                const double decoded = nlohmann::json::parse(spelling).get<double>();
                require(decoded >= 0 && decoded <= 10, "number escaped independent range oracle");
            }
        }
        auto exclusive = compiler.compile(
            {StructuredOutputKind::JsonSchema,
             R"({"type":"number","exclusiveMinimum":-0.25,"exclusiveMaximum":0.25})"});
        require(accepts_value(exclusive, "0") && accepts_value(exclusive, "0.249999") &&
                    !accepts_value(exclusive, "0.25") && !accepts_value(exclusive, "-0.25") &&
                    !accepts_value(exclusive, "1e2"),
                "exclusive number bounds weakened");
        auto integer =
            compiler.compile({StructuredOutputKind::JsonSchema,
                              R"({"type":"integer","minimum":-2,"exclusiveMaximum":3})"});
        for (int i = -10; i <= 10; ++i) {
            require(accepts_value(integer, std::to_string(i)) == (i >= -2 && i < 3),
                    "integer range disagrees with independent oracle");
        }
        require(!accepts_value(integer, "1.5"), "integer accepted a fraction");
        auto tiny =
            compiler.compile({StructuredOutputKind::JsonSchema,
                              R"({"type":"number","minimum":0.000001,"maximum":0.000002})"});
        require(accepts_value(tiny, "0.000001") && accepts_value(tiny, "0.000002") &&
                    !accepts_value(tiny, "0") && !accepts_value(tiny, "0.000003"),
                "number precision boundary escaped bounds");

        auto reasoning = compiler.compile({StructuredOutputKind::JsonObject, {}},
                                          {.reasoning_close = "</think>"});
        reasoning->accept(tokens_for("reasoning with << overlap </thi"));
        const auto crossing = tokens_for("nk>{\"ok\":true}");
        std::vector<std::uint32_t> crossing_masks(words * (crossing.size() + 1));
        reasoning->fill_masks(crossing_masks, crossing);
        for (std::size_t i = 0; i < crossing.size(); ++i) {
            require(crossing_masks[i * words + crossing[i] / 32] & (1U << (crossing[i] % 32)),
                    "reasoning-to-JSON speculative transition masked a valid token");
        }
        require(!(crossing_masks[3 * words + 'p' / 32] & (1U << ('p' % 32))),
                "reasoning grammar escaped into final prose");
        require(crossing_masks[crossing.size() * words + 8] & 1U, "final bonus EOS missing");
        require(accepts_value(reasoning, "nk>{}"), "draft masks advanced committed phase");
        require(accepts_value(reasoning, "nk>\n\n{}"), "canonical close whitespace rejected");
        require(!accepts_value(reasoning, "nk>plain text</think>{}"),
                "a second reasoning close bypassed the first boundary");
        auto mixed_vocab = vocab;
        mixed_vocab.push_back("</think>{");
        mixed_vocab.push_back("</think>prose");
        StructuredCompiler mixed_compiler(mixed_vocab, {256});
        auto mixed = mixed_compiler.compile({StructuredOutputKind::JsonObject, {}},
                                            {.reasoning_close = "</think>"});
        mixed->fill_masks(before, {});
        require((before[8] & 2U) && !(before[8] & 4U),
                "single token spanning reasoning and content escaped grammar");
        mixed->accept(std::vector<TokenId>{257, '}', 256});
        for (const char* bad :
             {R"({"type":"array","uniqueItems":true})",
              R"({"$ref":"#/$defs/a~1b","$defs":{"a/b":{"const":1},"a~1b":{"const":2}}})",
              R"({"oneOf":[{},{}]})", R"({"$ref":"https://example.org/schema"})",
              R"({"const":1,"type":"string"})",
              R"({"type":"integer","minimum":3,"maximum":2})", R"({"type":"number","minimum":"0"})",
              R"({"type":"number","minimum":0.0000001,"maximum":0.0000002})", R"({"minimum":0})",
              R"({"type":"number","minimum":1e30})"}) {
            bool failed = false;
            try {
                compiler.compile({StructuredOutputKind::JsonSchema, bad});
            } catch (const std::invalid_argument&) { failed = true; }
            require(failed, "unsupported constraint silently accepted");
        }
        // A regex, a choice and a GBNF grammar match the whole response, with or without the
        // reasoning envelope, and a malformed one fails before any vocabulary is involved.
        const auto with = [&](StructuredOutputKind kind, std::string grammar,
                              std::vector<std::string> choices = {}) {
            StructuredOutputOptions options{kind, {}};
            options.grammar = std::move(grammar);
            options.choices = std::move(choices);
            return options;
        };
        auto regex = compiler.compile(with(StructuredOutputKind::Regex, "[a-c]{2}-[0-9]+"));
        require(accepts_value(regex, "ab-12") && !accepts_value(regex, "ab-") &&
                    !accepts_value(regex, "xab-1") && !accepts_value(regex, "ab-1 "),
                "regex does not match the whole response");
        auto choice = compiler.compile(
            with(StructuredOutputKind::Choice, {}, {"yes", "no", "say \"maybe\"\n\\"}));
        require(accepts_value(choice, "yes") && accepts_value(choice, "no") &&
                    accepts_value(choice, "say \"maybe\"\n\\") && !accepts_value(choice, "ye") &&
                    !accepts_value(choice, "yesno") && !accepts_value(choice, "No"),
                "choice admits other than exactly one option");
        auto gbnf = compiler.compile(with(StructuredOutputKind::Grammar,
                                          "# a GBNF list\nroot ::= item (\",\" item)*\n"
                                          "item ::= \"-\"? [0-9]{1,3} | \"x\"\n"));
        require(accepts_value(gbnf, "1,x,-23") && !accepts_value(gbnf, "1,") &&
                    !accepts_value(gbnf, "y") && !accepts_value(gbnf, "1234"),
                "grammar does not match from its root rule");
        auto thought = compiler.compile(with(StructuredOutputKind::Choice, {}, {"red", "blue"}),
                                        {.reasoning_close = "</think>"});
        require(accepts_value(thought, "mull it over</think>\n\nblue") &&
                    !accepts_value(thought, "mull</think>green") && !accepts_value(thought, "blue"),
                "choice after reasoning lost the reasoning close or the options");
        for (const auto& bad :
             {with(StructuredOutputKind::Regex, "[a-"), with(StructuredOutputKind::Regex, ""),
              with(StructuredOutputKind::Grammar, "root ::= missing"),
              with(StructuredOutputKind::Grammar, "start ::= \"a\""),
              with(StructuredOutputKind::Choice, {}, {}),
              with(StructuredOutputKind::Choice, {}, {"a", ""}),
              with(StructuredOutputKind::JsonObject, "[a-z]")}) {
            bool failed = false;
            try {
                validate_structured_output(bad);
            } catch (const std::invalid_argument&) { failed = true; }
            require(failed, "malformed regex, grammar or choice accepted");
        }
        // Inside a length-bounded string the masks come from the counted repetition's budget
        // (multi-scalar tokens, partial UTF-8 and escapes, closing tokens at the bounds). Every
        // mask must equal what the grammar's own token acceptance admits.
        {
            std::vector<std::string> wide(256);
            for (int i = 0; i < 256; ++i) { wide[i] = std::string(1, static_cast<char>(i)); }
            for (const char* token : {"ab",
                                      "abc",
                                      "abcd",
                                      "abcdef",
                                      "\\n",
                                      "\\u00",
                                      "\\u00e9",
                                      "\\ud83d",
                                      "\\ude00",
                                      "\xc3\xa9",
                                      "\xf0\x9f\x98\x80",
                                      "\xf0\x9f",
                                      "\x98\x80",
                                      "\"",
                                      "x\"",
                                      "ab\"",
                                      "\"}",
                                      "\",",
                                      "x\",\"",
                                      "\\\"",
                                      "1\"",
                                      "\"]",
                                      "a\xc3"}) {
                wide.emplace_back(token);
            }
            const int eos = static_cast<int>(wide.size());
            wide.emplace_back();
            StructuredCompiler wide_compiler(wide, {eos});
            const std::size_t wide_words = (wide.size() + 31) / 32;
            const std::vector<std::pair<const char*, std::vector<std::string>>> walks = {
                {R"({"anyOf":[{"type":"string","minLength":2,"maxLength":5},{"type":"integer"}]})",
                 {"\"ab\\u00e9xy\"", "\"\xc3\xa9\xf0\x9f\x98\x80x\""}},
                {R"({"type":"object","properties":{"s":{"type":"string","maxLength":4}},"required":["s"]})",
                 {"{\"s\":\"\\n\\\"ab\"}", "{\"s\":\"\\ud83d\\ude00abc\"}"}},
                {R"({"type":"array","items":{"type":"string","minLength":1,"maxLength":3},"maxItems":2})",
                 {"[\"a\\u00e9b\",\"\xc3\xa9\"]"}},
            };
            for (const auto& [schema, texts] : walks) {
                for (const auto& text : texts) {
                    auto grammar =
                        wide_compiler.compile({StructuredOutputKind::JsonSchema, schema});
                    for (std::size_t at = 0; at <= text.size(); ++at) {
                        std::vector<std::uint32_t> masks(wide_words);
                        grammar->fill_masks(masks, {});
                        for (int token = 0; token < static_cast<int>(wide.size()); ++token) {
                            auto trial       = grammar->fork();
                            bool admits      = true;
                            const TokenId id = token;
                            try {
                                trial->accept(std::span<const TokenId>(&id, 1));
                            } catch (const std::logic_error&) { admits = false; }
                            const bool masked = (masks[token / 32] >> (token % 32)) & 1U;
                            if (masked != admits) {
                                throw std::runtime_error(
                                    "bounded-string mask disagrees with acceptance at byte " +
                                    std::to_string(at) + " of " + text + " for token " +
                                    std::to_string(token));
                            }
                        }
                        if (at < text.size()) {
                            grammar->accept(
                                std::vector<TokenId>{static_cast<unsigned char>(text[at])});
                        }
                    }
                }
            }
        }
        std::cout << "OK structured grammar: masks, transaction, UTF-8, schema, EOS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
