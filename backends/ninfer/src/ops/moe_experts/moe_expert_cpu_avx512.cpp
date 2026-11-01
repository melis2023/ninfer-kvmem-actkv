#include "ops/moe_experts/cpu_projection.h"
#include <immintrin.h>

namespace ninfer::ops::cpu_expert_detail {
namespace {
struct VnniDot {
    static int integer(const std::int8_t* weights, const std::int8_t* input) {
        const auto w = _mm512_maskz_loadu_epi8(0xffffffffULL, weights);
        const auto x = _mm512_maskz_loadu_epi8(0xffffffffULL, input);
        const auto negative = _mm512_cmp_epi8_mask(w, _mm512_setzero_si512(), _MM_CMPINT_LT);
        const auto signed_x = _mm512_mask_sub_epi8(x, negative, _mm512_setzero_si512(), x);
        return _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(
            _mm512_setzero_si512(), _mm512_abs_epi8(w), signed_x));
    }
    static float floating(const std::int8_t* weights, const float* input) {
        auto sum = _mm512_setzero_ps();
        for (int start = 0; start < 32; start += 16) {
            const auto codes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + start));
            const auto w = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(codes));
            sum = _mm512_add_ps(sum, _mm512_mul_ps(w, _mm512_loadu_ps(input + start)));
        }
        return _mm512_reduce_add_ps(sum);
    }
};
} // namespace
void project_avx512(const Weight& w, const Activations& x, bool a8, float* y) {
    project<VnniDot>(w, x, a8, y);
}
} // namespace ninfer::ops::cpu_expert_detail
