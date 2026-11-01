#pragma once
#include "ninfer/base.h"
#include "ninfer/types_fwd.h"
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::text {
// A token the request's grammar does not admit. The masks the request sampled under should have
// excluded it, so it is an execution fault, but one confined to that request.
class GrammarViolation final : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

// Reject unsupported constraints instead of letting a compiler silently weaken a schema.
void validate_structured_output(const StructuredOutputOptions& options);

// Model-owned framing around the final response. The alternate format is an XGrammar
// structural-tag descriptor (e.g. a model's native tool-call serialization), not a user API.
struct StructuredOutputEnvelope {
    std::string reasoning_close;
    std::string alternative_format;
};

class GrammarState {
public:
    ~GrammarState();
    GrammarState(GrammarState&&) noexcept;
    GrammarState& operator=(GrammarState&&) noexcept;
    [[nodiscard]] std::unique_ptr<GrammarState> fork() const;
    void accept(std::span<const TokenId> tokens);
    // Column i is conditioned on drafts[0..i). Unreachable suffix columns are unrestricted.
    void fill_masks(std::span<std::uint32_t> masks, std::span<const TokenId> drafts) const;
private:
    struct Impl;
    explicit GrammarState(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class StructuredCompiler;
};

class StructuredCompiler {
public:
    StructuredCompiler(std::vector<std::string> decoded_vocab, std::vector<int> stop_tokens);
    ~StructuredCompiler();
    std::shared_ptr<GrammarState> compile(const StructuredOutputOptions& options,
                                          const StructuredOutputEnvelope& envelope = {});
    // A response with no final format of its own: `format`, an XGrammar structural-tag format
    // (free text with a model's tool calls), after the reasoning close when the envelope has one.
    std::shared_ptr<GrammarState> compile_format(const std::string& format,
                                                 const StructuredOutputEnvelope& envelope = {});
    std::shared_ptr<GrammarState> compile_model(const std::string& grammar_source,
                                               const StructuredOutputOptions& body,
                                               const std::string& reasoning_close,
                                               const std::string& continuation = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::text
