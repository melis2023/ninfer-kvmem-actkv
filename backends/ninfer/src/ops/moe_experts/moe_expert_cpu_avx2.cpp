#include "ops/moe_experts/cpu_projection.h"
#include <immintrin.h>

namespace ninfer::ops::cpu_expert_detail {
namespace {
struct Avx2Dot {
    static int integer(const std::int8_t* weights, const std::int8_t* input) {
        const auto w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights));
        const auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
        // Activations are [-127,127]. Even Q8's magnitude 128 keeps each adjacent pair
        // within int16, so maddubs' saturation cannot alter a product.
        const auto pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
        const auto sums = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
        auto sum = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
        sum = _mm_hadd_epi32(sum, sum);
        return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
    }
    static float floating(const std::int8_t* weights, const float* input) {
        auto sum = _mm256_setzero_ps();
        for (int start = 0; start < 32; start += 8) {
            const auto codes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(weights + start));
            const auto w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(w, _mm256_loadu_ps(input + start)));
        }
        auto halves = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        halves = _mm_hadd_ps(halves, halves);
        return _mm_cvtss_f32(_mm_hadd_ps(halves, halves));
    }
};
} // namespace
void project_avx2(const Weight& w, const Activations& x, bool a8, float* y) {
    project<Avx2Dot>(w, x, a8, y);
}
} // namespace ninfer::ops::cpu_expert_detail
