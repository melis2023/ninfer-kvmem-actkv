#include "ops/linear/gguf/mmq_limits.h"

#include <climits>
#include <iostream>
#include <stdexcept>

int main() {
    using ninfer::ops::gguf::detail::matrix_work_blocks;
    constexpr int limit = 1 << 30;
    struct Case { int rows; int columns; int k; int expected; };
    const Case cases[] = {
        {1, 1, 1, 1}, {40, 1, 68, 2720},
        {1, 1, limit - 1, limit - 1}, {3, 1, 357913941, limit - 1},
        {limit - 1, 1, 1, limit - 1},
        {1, 1, limit, 0}, {1, 1, limit + 1, 0},
        {32768, 32768, 1, 0}, {3, 1, 357913942, 0},
        {INT_MAX, INT_MAX, INT_MAX, 0}, {1, 1, INT_MAX, 0},
        {0, 1, 1, 0}, {1, 0, 1, 0}, {1, 1, 0, 0},
        {-1, 1, 1, 0}, {1, -1, 1, 0}, {1, 1, -1, 0},
        {INT_MIN, INT_MAX, INT_MAX, 0},
    };
    try {
        for (const auto& c : cases) {
            int actual = 0;
            bool rejected = false;
            try {
                actual = matrix_work_blocks(c.rows, c.columns, c.k);
            } catch (const std::invalid_argument&) {
                rejected = true;
                if (c.expected != 0) { throw; }
            }
            if (actual != c.expected || rejected != (c.expected == 0)) {
                throw std::runtime_error("MMQ work bound mismatch");
            }
        }
        std::cout << "OK GGUF MMQ index limits\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
