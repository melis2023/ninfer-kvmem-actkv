// The branch memo without an Engine: which reads share a key, the record a repeat answers with,
// and which records the memo keeps once it is full.

#include "serve/branch_memo.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;
using ninfer::TokenId;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

BranchKey key(const std::vector<TokenId>& prompt, const std::vector<TokenId>& readout,
              std::optional<int> top_logprobs = std::nullopt) {
    return branch_key(prompt, readout, top_logprobs);
}

ninfer::BranchResult record(std::uint32_t prompt_tokens, TokenId token, float readout) {
    ninfer::BranchResult result;
    result.prompt_tokens      = prompt_tokens;
    result.first.id           = token;
    result.first.logprob      = -0.25F;
    result.first.bytes        = "yes";
    result.first.top_ids[0]   = token;
    result.first.top_bytes[0] = "yes";
    result.first.readout      = {readout, -3.5F};
    return result;
}

void keys() {
    const std::vector<TokenId> prompt{11, 12, 13, 14};
    const std::vector<TokenId> readout{7, 8};
    expect(key(prompt, readout) == key(prompt, readout), "the same read has one key");
    expect(!(key(prompt, readout) == key({11, 12, 13, 15}, readout)),
           "the last prompt token separates reads");
    expect(!(key(prompt, readout) == key({10, 12, 13, 14}, readout)),
           "the first prompt token separates reads");
    expect(!(key(prompt, readout) == key({11, 12, 13, 14, 0}, readout)),
           "a longer prompt separates reads");
    expect(!(key(prompt, readout) == key(prompt, {8, 7})), "the readout order separates reads");
    expect(!(key({1, 2}, {3}) == key({1}, {2, 3})), "the prompt's end separates reads");
    expect(!(key(prompt, readout) == key(prompt, readout, 0)),
           "asking for logprobs separates reads");
    expect(!(key(prompt, readout, 0) == key(prompt, readout, 5)),
           "the alternatives asked for separate reads");
}

void repeats() {
    BranchMemo memo(4);
    const BranchKey read = key({1, 2, 3}, {4});
    expect(!memo.find(read), "an empty memo holds nothing");
    memo.keep(read, record(3, 9, -0.5F));
    const std::optional<ninfer::BranchResult> held = memo.find(read);
    expect(held.has_value(), "a kept record answers its read");
    if (held) {
        expect(held->prompt_tokens == 3 && held->first.id == 9 && held->first.logprob == -0.25F &&
                   held->first.bytes == "yes" && held->first.top_ids[0] == 9 &&
                   held->first.top_bytes[0] == "yes" &&
                   held->first.readout == std::vector<float>{-0.5F, -3.5F},
               "the held record is the kept one");
    }
    expect(!memo.find(key({1, 2, 3}, {5})), "another read is not held");
    memo.keep(read, record(3, 10, -0.75F));
    expect(memo.size() == 1, "keeping a read again replaces its record");
    expect(memo.find(read)->first.id == 10, "the newer record answers");
}

void capacity() {
    BranchMemo memo(2);
    const BranchKey a = key({1}, {});
    const BranchKey b = key({2}, {});
    const BranchKey c = key({3}, {});
    memo.keep(a, record(1, 1, 0.0F));
    memo.keep(b, record(1, 2, 0.0F));
    expect(memo.find(a).has_value(), "the first record is held");
    memo.keep(c, record(1, 3, 0.0F));
    expect(memo.size() == 2, "a full memo keeps its capacity");
    expect(!memo.find(b), "the least recently used record goes first");
    expect(memo.find(a).has_value() && memo.find(c).has_value(), "the recent records stay");
    BranchMemo none(0);
    none.keep(a, record(1, 1, 0.0F));
    expect(!none.find(a), "a memo of no records holds none");
}

} // namespace

int main() {
    keys();
    repeats();
    capacity();
    if (failures != 0) {
        std::cerr << failures << " expectation(s) failed\n";
        return 1;
    }
    std::cout << "branch memo: ok\n";
    return 0;
}
