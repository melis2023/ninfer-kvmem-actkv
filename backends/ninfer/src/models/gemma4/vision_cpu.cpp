#include "models/gemma4/vision_cpu.h"

#include "core/weight_view.h"
#include "models/gemma4/vision_cpu_kernels.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace ninfer::models::gemma4 {
namespace {

float bf16_to_f32(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16U);
}

std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0) {
        return static_cast<std::uint16_t>((bits >> 16U) | 0x40U);
    }
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

// A host weight's values as FP32 (rows x columns, row-major; a vector is one row): BF16 words
// decoded, FP32 copied.
std::vector<float> host_values(const Model& model, WeightId id, std::size_t columns,
                               std::size_t rows = 1) {
    const auto& bound = model.weight(id);
    const Tensor t    = rows == 1 ? weight_tensor(bound.view, {static_cast<std::int32_t>(columns)})
                                  : weight_tensor(bound.view, {static_cast<std::int32_t>(columns),
                                                               static_cast<std::int32_t>(rows)});
    const std::size_t count = columns * rows;
    std::vector<float> out(count);
    if (t.dtype == DType::FP32) {
        std::memcpy(out.data(), t.data, count * sizeof(float));
    } else if (t.dtype == DType::BF16) {
        const auto* words = static_cast<const std::uint16_t*>(t.data);
        for (std::size_t i = 0; i < count; ++i) { out[i] = bf16_to_f32(words[i]); }
    } else {
        throw std::invalid_argument(bound.name + ": the CPU vision tower reads BF16 or FP32");
    }
    return out;
}

// A fork-join team kept for one encode, so its few hundred parallel sections do not each create
// threads.
class Team {
public:
    explicit Team(unsigned threads) : size_(std::max(1U, threads)) {
        workers_.reserve(size_ - 1);
        for (unsigned index = 1; index < size_; ++index) {
            workers_.emplace_back([this, index] { serve(index); });
        }
    }

    ~Team() {
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (std::thread& worker : workers_) { worker.join(); }
    }

    Team(const Team&)            = delete;
    Team& operator=(const Team&) = delete;

    // Runs fn(begin, end) over [0, count) in contiguous slices of at least `grain`, one per member.
    template <class Fn>
    void run(std::size_t count, std::size_t grain, Fn&& fn) {
        if (count == 0) { return; }
        const std::size_t slices =
            std::max<std::size_t>(1, std::min<std::size_t>(size_, (count + grain - 1) / grain));
        const std::size_t step = (count + slices - 1) / slices;
        const auto slice       = [&](unsigned index) {
            const std::size_t begin = std::min(count, index * step);
            const std::size_t end   = std::min(count, begin + step);
            if (begin < end) { fn(begin, end); }
        };
        if (slices == 1) {
            slice(0);
            return;
        }
        {
            const std::lock_guard lock(mutex_);
            job_       = slice;
            active_    = static_cast<unsigned>(slices);
            remaining_ = static_cast<unsigned>(slices - 1);
            ++generation_;
        }
        wake_.notify_all();
        slice(0);
        std::unique_lock lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
    }

private:
    void serve(unsigned index) {
        std::uint64_t seen = 0;
        for (;;) {
            std::function<void(unsigned)> job;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                if (stopping_) { return; }
                seen = generation_;
                if (index >= active_) { continue; }
                job = job_;
            }
            job(index);
            {
                const std::lock_guard lock(mutex_);
                --remaining_;
            }
            done_.notify_one();
        }
    }

    unsigned size_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::function<void(unsigned)> job_;
    std::uint64_t generation_ = 0;
    unsigned active_          = 0;
    unsigned remaining_       = 0;
    bool stopping_            = false;
};

// Activations are feature-major: row r holds feature r across every patch, so each projection
// streams contiguous patch runs.
struct Matrix {
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    std::vector<float> data;

    Matrix() = default;
    Matrix(std::int32_t r, std::int32_t c)
        : rows(r), cols(c), data(static_cast<std::size_t>(r) * static_cast<std::size_t>(c)) {}

    [[nodiscard]] float* row(std::int32_t r) {
        return data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    }
    [[nodiscard]] const float* row(std::int32_t r) const {
        return data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    }
};

// y = W x, W row-major [n, k], x feature-major [k, T], y feature-major [n, T]: the rows of W split
// over the team, four at a time.
void linear(const std::vector<float>& w, std::int32_t n, const Matrix& x, Matrix& y, Team& team) {
    const std::int32_t k = x.rows;
    if (w.size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(k) || y.rows != n ||
        y.cols != x.cols) {
        throw std::logic_error("CPU vision: projection extents do not match");
    }
    const auto& kernels = cpu_vision::kernels();
    const std::int32_t groups = (n + 3) / 4;
    team.run(static_cast<std::size_t>(groups), 4, [&](std::size_t begin, std::size_t end) {
        const auto row  = static_cast<std::int32_t>(begin) * 4;
        const auto rows = std::min(n, static_cast<std::int32_t>(end) * 4) - row;
        kernels.gemm(w.data() + std::size_t(row) * k, std::size_t(k), rows, k, x.row(0),
                     std::size_t(x.cols), x.cols, y.row(row), std::size_t(y.cols));
    });
}

// Per-patch RMS statistics over the feature rows: 1 / sqrt(mean x^2 + eps).
std::vector<float> rms_scales(const Matrix& x, float eps, Team& team) {
    std::vector<float> out(static_cast<std::size_t>(x.cols));
    team.run(static_cast<std::size_t>(x.cols), 64, [&](std::size_t begin, std::size_t end) {
        std::vector<double> sums(end - begin, 0.0);
        for (std::int32_t r = 0; r < x.rows; ++r) {
            const float* values = x.row(r);
            for (std::size_t t = begin; t < end; ++t) {
                sums[t - begin] += double(values[t]) * values[t];
            }
        }
        for (std::size_t t = begin; t < end; ++t) {
            out[t] = 1.0F / std::sqrt(float(sums[t - begin] / x.rows) + eps);
        }
    });
    return out;
}

// y[r][t] = x[r][t] * scale[t] * gamma[r] (gamma empty: 1).
void apply_norm(const Matrix& x, const std::vector<float>& scale, const std::vector<float>& gamma,
                Matrix& y, Team& team) {
    team.run(static_cast<std::size_t>(x.rows), 8, [&](std::size_t begin, std::size_t end) {
        for (auto r = static_cast<std::int32_t>(begin); r < static_cast<std::int32_t>(end); ++r) {
            const float g = gamma.empty() ? 1.0F : gamma[static_cast<std::size_t>(r)];
            const float* in = x.row(r);
            float* out      = y.row(r);
            for (std::int32_t t = 0; t < x.cols; ++t) { out[t] = in[t] * scale[t] * g; }
        }
    });
}

} // namespace

CpuVisionTower::CpuVisionTower(const Model& model, unsigned threads)
    : config_(*model.vision_config()),
      threads_(threads != 0 ? threads : std::max(1U, std::thread::hardware_concurrency())) {
    if (!model.vision() || model.vision_placement() != VisionPlacement::Host) {
        throw std::invalid_argument("gemma4 CPU vision: the model holds no host vision tower");
    }
    const VisionWeights& w = *model.vision();
    const std::size_t h = config_.hidden_size, width = config_.intermediate_size;
    const std::size_t d = config_.head_dim;
    patch_embedding_    = host_values(model, w.patch_embedding, config_.patch_features(), h);
    projection_         = host_values(model, w.projection, h, config_.projection_dim);
    position_table_ =
        host_values(model, w.position_table, h, 2 * std::size_t(config_.position_embedding_size));
    if (config_.standardize) {
        std_bias_  = host_values(model, *w.std_bias, h);
        std_scale_ = host_values(model, *w.std_scale, h);
    }
    for (const VisionLayerWeights& lw : w.layers) {
        Layer layer;
        for (const WeightId id : {lw.query, lw.key, lw.value}) {
            const auto part = host_values(model, id, h, h);
            layer.qkv.insert(layer.qkv.end(), part.begin(), part.end());
        }
        layer.output = host_values(model, lw.output, h, h);
        for (const WeightId id : {lw.mlp_gate, lw.mlp_up}) {
            const auto part = host_values(model, id, h, width);
            layer.gate_up.insert(layer.gate_up.end(), part.begin(), part.end());
        }
        layer.down                = host_values(model, lw.mlp_down, width, h);
        layer.input_norm          = host_values(model, lw.input_norm, h);
        layer.post_attention_norm = host_values(model, lw.post_attention_norm, h);
        layer.ffn_norm            = host_values(model, lw.ffn_norm, h);
        layer.post_ffn_norm       = host_values(model, lw.post_ffn_norm, h);
        layer.q_norm              = host_values(model, lw.query_norm, d);
        layer.k_norm              = host_values(model, lw.key_norm, d);
        layers_.push_back(std::move(layer));
    }
    inv_freq_.resize(d / 4);
    for (std::size_t i = 0; i < inv_freq_.size(); ++i) {
        inv_freq_[i] = float(1.0 / std::pow(double(config_.rope_theta),
                                            double(2 * i) / double(d / 2)));
    }
}

std::vector<std::uint16_t> CpuVisionTower::encode(std::span<const std::uint16_t> patches,
                                                  int units, int grid_w, int grid_h) const {
    static std::mutex turn; // one encode at a time across the process
    const std::lock_guard lock(turn);
    const int k        = int(config_.pooling_kernel_size);
    const int per_unit = grid_w * grid_h;
    const int total    = units * per_unit;
    const int h = int(config_.hidden_size), width = int(config_.intermediate_size);
    const int heads = int(config_.num_attention_heads), d = int(config_.head_dim);
    const int features = int(config_.patch_features());
    if (units <= 0 || grid_w % k != 0 || grid_h % k != 0 ||
        patches.size() != std::size_t(total) * features) {
        throw std::invalid_argument("gemma4 CPU vision: invalid encode batch");
    }
    const float eps = config_.rms_norm_eps;
    Team team(threads_);

    Matrix x(features, total);
    for (int p = 0; p < total; ++p) {
        for (int f = 0; f < features; ++f) {
            x.row(f)[p] = bf16_to_f32(patches[std::size_t(p) * features + f]);
        }
    }
    Matrix residual(h, total);
    linear(patch_embedding_, h, x, residual, team);
    for (int p = 0; p < total; ++p) {
        const int local = p % per_unit, column = local % grid_w, row = local / grid_w;
        const float* xs = position_table_.data() + std::size_t(column) * h;
        const float* ys =
            position_table_.data() + std::size_t(config_.position_embedding_size + row) * h;
        for (int c = 0; c < h; ++c) { residual.row(c)[p] += xs[c] + ys[c]; }
    }
    // The axial RoPE's cos and sin per angle index and patch, feature-major: the head's first half
    // turns by the patch's column, its second half by its row.
    const int quarter = d / 4;
    Matrix cos_x(quarter, total), sin_x(quarter, total), cos_y(quarter, total), sin_y(quarter, total);
    for (int j = 0; j < quarter; ++j) {
        for (int p = 0; p < total; ++p) {
            const int local = p % per_unit;
            const float ax = float(local % grid_w) * inv_freq_[j];
            const float ay = float(local / grid_w) * inv_freq_[j];
            cos_x.row(j)[p] = std::cos(ax);
            sin_x.row(j)[p] = std::sin(ax);
            cos_y.row(j)[p] = std::cos(ay);
            sin_y.row(j)[p] = std::sin(ay);
        }
    }
    // Attention runs per (unit, head, block of queries): the block's scores against every key of
    // the unit, their column softmax, and the values weighted by it.
    constexpr int kQueryBlock = 128;
    const int query_blocks    = (per_unit + kQueryBlock - 1) / kQueryBlock;
    const auto& kernels       = cpu_vision::kernels();
    Matrix normed(h, total), qkv(3 * h, total), q(h, total), kk(h, total), v(h, total),
        keys(heads * total, d), attended(h, total), o(h, total), gu(2 * width, total),
        mid(width, total);
    for (const Layer& layer : layers_) {
        apply_norm(residual, rms_scales(residual, eps, team), layer.input_norm, normed, team);
        linear(layer.qkv, 3 * h, normed, qkv, team);
        // Per-head norms of Q, K (with their gammas) and V, and the RoPE of Q and K: pairs
        // (i, i + D/4) within each half.
        team.run(std::size_t(3) * heads, 1, [&](std::size_t begin, std::size_t end) {
            std::vector<float> scale(total);
            for (std::size_t job = begin; job < end; ++job) {
                const int part = int(job) / heads, hd = int(job) % heads;
                const int from = part * h + hd * d;
                std::fill(scale.begin(), scale.end(), 0.0F);
                for (int i = 0; i < d; ++i) {
                    const float* values = qkv.row(from + i);
                    for (int p = 0; p < total; ++p) { scale[p] += values[p] * values[p]; }
                }
                for (int p = 0; p < total; ++p) { scale[p] = 1.0F / std::sqrt(scale[p] / d + eps); }
                Matrix& out = part == 0 ? q : part == 1 ? kk : v;
                const std::vector<float>* gamma =
                    part == 0 ? &layer.q_norm : part == 1 ? &layer.k_norm : nullptr;
                for (int i = 0; i < d; ++i) {
                    const float g     = gamma != nullptr ? (*gamma)[i] : 1.0F;
                    const float* from_row = qkv.row(from + i);
                    float* to_row     = out.row(hd * d + i);
                    for (int p = 0; p < total; ++p) { to_row[p] = from_row[p] * scale[p] * g; }
                }
                if (part == 2) { continue; }
                for (int half = 0; half < 2; ++half) {
                    for (int j = 0; j < quarter; ++j) {
                        float* a        = out.row(hd * d + half * 2 * quarter + j);
                        float* b        = out.row(hd * d + half * 2 * quarter + quarter + j);
                        const float* cs = (half == 0 ? cos_x : cos_y).row(j);
                        const float* sn = (half == 0 ? sin_x : sin_y).row(j);
                        for (int p = 0; p < total; ++p) {
                            const float av = a[p], bv = b[p];
                            a[p]           = av * cs[p] - bv * sn[p];
                            b[p]           = bv * cs[p] + av * sn[p];
                        }
                    }
                }
            }
        });
        // Each head's keys patch-major, the scores' left operand.
        team.run(std::size_t(heads), 1, [&](std::size_t begin, std::size_t end) {
            for (auto hd = int(begin); hd < int(end); ++hd) {
                for (int i = 0; i < d; ++i) {
                    const float* from_row = kk.row(hd * d + i);
                    for (int p = 0; p < total; ++p) { keys.row(hd * total + p)[i] = from_row[p]; }
                }
            }
        });
        team.run(std::size_t(units) * heads * query_blocks, 1,
                 [&](std::size_t begin, std::size_t end) {
            std::vector<float> scores(std::size_t(per_unit) * kQueryBlock), sums(kQueryBlock);
            for (std::size_t job = begin; job < end; ++job) {
                const int unit  = int(job / (std::size_t(heads) * query_blocks));
                const int rest  = int(job % (std::size_t(heads) * query_blocks));
                const int hd    = rest / query_blocks;
                const int first = rest % query_blocks * kQueryBlock;
                const int count = std::min(kQueryBlock, per_unit - first);
                const int base  = unit * per_unit;
                // Unscaled scores [key][query] = K Q^T over the unit's patches.
                kernels.gemm(keys.row(hd * total + base), std::size_t(d), per_unit, d,
                             q.row(hd * d) + base + first, std::size_t(total), count,
                             scores.data(), std::size_t(count));
                kernels.softmax_columns(scores.data(), std::size_t(count), per_unit, count,
                                        sums.data());
                // [feature][query] = V^T P, then each query's column over its sum.
                float* out = attended.row(hd * d) + base + first;
                kernels.gemm(v.row(hd * d) + base, std::size_t(total), d, per_unit, scores.data(),
                             std::size_t(count), count, out, std::size_t(total));
                for (int i = 0; i < d; ++i) {
                    float* row = out + std::size_t(i) * total;
                    for (int t = 0; t < count; ++t) { row[t] /= sums[t]; }
                }
            }
        });
        linear(layer.output, h, attended, o, team);
        const auto attention_scale = rms_scales(o, eps, team);
        team.run(std::size_t(h), 8, [&](std::size_t begin, std::size_t end) {
            for (auto c = int(begin); c < int(end); ++c) {
                for (int p = 0; p < total; ++p) {
                    residual.row(c)[p] += o.row(c)[p] * attention_scale[p] *
                                          layer.post_attention_norm[c];
                }
            }
        });
        apply_norm(residual, rms_scales(residual, eps, team), layer.ffn_norm, normed, team);
        linear(layer.gate_up, 2 * width, normed, gu, team);
        team.run(std::size_t(width), 8, [&](std::size_t begin, std::size_t end) {
            for (auto c = int(begin); c < int(end); ++c) {
                kernels.geglu(gu.row(c), gu.row(width + c), mid.row(c), std::size_t(total));
            }
        });
        linear(layer.down, h, mid, o, team);
        const auto mlp_scale = rms_scales(o, eps, team);
        team.run(std::size_t(h), 8, [&](std::size_t begin, std::size_t end) {
            for (auto c = int(begin); c < int(end); ++c) {
                for (int p = 0; p < total; ++p) {
                    residual.row(c)[p] += o.row(c)[p] * mlp_scale[p] * layer.post_ffn_norm[c];
                }
            }
        });
    }
    // Pool, scale, standardize and normalize each soft token, then project it.
    const int columns = grid_w / k, rows = grid_h / k, tokens_per_unit = columns * rows;
    const int tokens  = units * tokens_per_unit;
    Matrix pooled(h, tokens);
    const float scale = std::sqrt(float(h)) / float(k * k);
    team.run(std::size_t(tokens), 8, [&](std::size_t begin, std::size_t end) {
        std::vector<float> value(h);
        for (std::size_t t = begin; t < end; ++t) {
            const int unit = int(t) / tokens_per_unit, local = int(t) % tokens_per_unit;
            const int ty = local / columns, tx = local % columns;
            std::fill(value.begin(), value.end(), 0.0F);
            for (int dy = 0; dy < k; ++dy) {
                for (int dx = 0; dx < k; ++dx) {
                    const int p = unit * per_unit + (ty * k + dy) * grid_w + tx * k + dx;
                    for (int c = 0; c < h; ++c) { value[c] += residual.row(c)[p]; }
                }
            }
            double ss = 0.0;
            for (int c = 0; c < h; ++c) {
                value[c] *= scale;
                if (!std_bias_.empty()) { value[c] = (value[c] - std_bias_[c]) * std_scale_[c]; }
                ss += double(value[c]) * value[c];
            }
            const float rs = 1.0F / std::sqrt(float(ss / h) + eps);
            for (int c = 0; c < h; ++c) { pooled.row(c)[t] = value[c] * rs; }
        }
    });
    const int out_width = int(config_.projection_dim);
    Matrix embedded(out_width, tokens);
    linear(projection_, out_width, pooled, embedded, team);
    std::vector<std::uint16_t> out(std::size_t(tokens) * out_width);
    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < out_width; ++c) {
            out[std::size_t(t) * out_width + c] = f32_to_bf16(embedded.row(c)[t]);
        }
    }
    return out;
}

} // namespace ninfer::models::gemma4
