#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace ninfer::text {
// Request-owned incremental string constraints the grammar does not carry: uniqueness for JSON
// string arrays. The immutable schema plan is shared; a copy owns independent parser, branch and
// seen-value state.
class StringConstraintState {
public:
    static void AnalyzeSchema(const std::string& normalized_schema);
    explicit StringConstraintState(const std::string& normalized_schema,
                                   const std::string& reasoning_close = {});
    ~StringConstraintState();
    StringConstraintState(const StringConstraintState&);
    StringConstraintState& operator=(const StringConstraintState&);
    StringConstraintState(StringConstraintState&&) noexcept;
    StringConstraintState& operator=(StringConstraintState&&) noexcept;
    [[nodiscard]] bool accepts(std::string_view decoded_token) const;
    // False never changes the committed state.
    bool accept(std::string_view decoded_token);
    [[nodiscard]] bool has_constraints() const noexcept;
    [[nodiscard]] bool needs_check(std::string_view decoded_token) const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::text
