#include "mythos/eval/evaluator.hpp"

#include "../../evaluate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iterator>
#include <limits>
#include <numeric>
#include <string_view>

namespace mythos::eval {

namespace {

template <typename T>
[[nodiscard]] bool read_exact(std::ifstream& input, T& out) {
    input.read(reinterpret_cast<char*>(&out), sizeof(T));
    return static_cast<bool>(input);
}

[[nodiscard]] bool read_vector(std::ifstream& input, std::vector<float>& out, std::size_t count) {
    out.resize(count);
    input.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(sizeof(float) * count));
    return static_cast<bool>(input);
}

[[nodiscard]] float clipped_relu(float value) noexcept {
    if (value <= 0.0F) {
        return 0.0F;
    }
    return value >= 1.0F ? 1.0F : value;
}

inline void accumulate_sparse_scalar(
    float* accumulator,
    const float* rows,
    const std::uint16_t* active,
    std::size_t active_count,
    std::size_t stride) noexcept {
    for (std::size_t i = 0; i < active_count; ++i) {
        const float* row = rows + static_cast<std::size_t>(active[i]) * stride;
        for (std::size_t j = 0; j < stride; ++j) {
            accumulator[j] += row[j];
        }
    }
}

#if defined(__AVX512F__)
inline void accumulate_sparse_avx512(
    float* accumulator,
    const float* rows,
    const std::uint16_t* active,
    std::size_t active_count,
    std::size_t stride) noexcept {
    constexpr std::size_t kStep = 16;
    for (std::size_t i = 0; i < active_count; ++i) {
        const float* row = rows + static_cast<std::size_t>(active[i]) * stride;
        std::size_t j = 0;
        for (; j + kStep <= stride; j += kStep) {
            const auto lhs = _mm512_loadu_ps(accumulator + j);
            const auto rhs = _mm512_loadu_ps(row + j);
            _mm512_storeu_ps(accumulator + j, _mm512_add_ps(lhs, rhs));
        }
        for (; j < stride; ++j) {
            accumulator[j] += row[j];
        }
    }
}
#endif

#if defined(__AVX2__)
inline void accumulate_sparse_avx2(
    float* accumulator,
    const float* rows,
    const std::uint16_t* active,
    std::size_t active_count,
    std::size_t stride) noexcept {
    constexpr std::size_t kStep = 8;
    for (std::size_t i = 0; i < active_count; ++i) {
        const float* row = rows + static_cast<std::size_t>(active[i]) * stride;
        std::size_t j = 0;
        for (; j + kStep <= stride; j += kStep) {
            const auto lhs = _mm256_loadu_ps(accumulator + j);
            const auto rhs = _mm256_loadu_ps(row + j);
            _mm256_storeu_ps(accumulator + j, _mm256_add_ps(lhs, rhs));
        }
        for (; j < stride; ++j) {
            accumulator[j] += row[j];
        }
    }
}
#endif

inline void accumulate_sparse(
    float* accumulator,
    const float* rows,
    const std::uint16_t* active,
    std::size_t active_count,
    std::size_t stride) noexcept {
#if defined(__AVX512F__)
    accumulate_sparse_avx512(accumulator, rows, active, active_count, stride);
#elif defined(__AVX2__)
    accumulate_sparse_avx2(accumulator, rows, active, active_count, stride);
#else
    accumulate_sparse_scalar(accumulator, rows, active, active_count, stride);
#endif
}

[[nodiscard]] float dot_scalar(const float* lhs, const float* rhs, std::size_t count) noexcept {
    float result = 0.0F;
    for (std::size_t i = 0; i < count; ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
}

#if defined(__AVX512F__)
[[nodiscard]] float dot_avx512(const float* lhs, const float* rhs, std::size_t count) noexcept {
    constexpr std::size_t kStep = 16;
    std::size_t i = 0;
    auto acc = _mm512_setzero_ps();
    for (; i + kStep <= count; i += kStep) {
        acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_loadu_ps(lhs + i), _mm512_loadu_ps(rhs + i)));
    }
    alignas(64) std::array<float, kStep> lane{};
    _mm512_store_ps(lane.data(), acc);
    float result = std::accumulate(lane.begin(), lane.end(), 0.0F);
    for (; i < count; ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
}
#endif

#if defined(__AVX2__)
[[nodiscard]] float dot_avx2(const float* lhs, const float* rhs, std::size_t count) noexcept {
    constexpr std::size_t kStep = 8;
    std::size_t i = 0;
    auto acc = _mm256_setzero_ps();
    for (; i + kStep <= count; i += kStep) {
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(lhs + i), _mm256_loadu_ps(rhs + i)));
    }
    alignas(32) std::array<float, kStep> lane{};
    _mm256_store_ps(lane.data(), acc);
    float result = std::accumulate(lane.begin(), lane.end(), 0.0F);
    for (; i < count; ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
}
#endif

[[nodiscard]] float dot_product(const float* lhs, const float* rhs, std::size_t count) noexcept {
#if defined(__AVX512F__)
    return dot_avx512(lhs, rhs, count);
#elif defined(__AVX2__)
    return dot_avx2(lhs, rhs, count);
#else
    return dot_scalar(lhs, rhs, count);
#endif
}

template <std::size_t In, std::size_t Out>
void dense_forward(
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    const std::array<float, In>& input,
    std::array<float, Out>& output,
    bool clip) noexcept {
    for (std::size_t out_idx = 0; out_idx < Out; ++out_idx) {
        const float* row = weights.data() + out_idx * In;
        float value = bias[out_idx] + dot_product(row, input.data(), In);
        output[out_idx] = clip ? clipped_relu(value) : value;
    }
}

[[nodiscard]] int piece_plane(Piece piece) noexcept {
    switch (piece) {
        case W_PAWN: return 0;
        case W_KNIGHT: return 1;
        case W_BISHOP: return 2;
        case W_ROOK: return 3;
        case W_QUEEN: return 4;
        case W_KING: return 5;
        case B_PAWN: return 6;
        case B_KNIGHT: return 7;
        case B_BISHOP: return 8;
        case B_ROOK: return 9;
        case B_QUEEN: return 10;
        case B_KING: return 11;
        default: return -1;
    }
}

}  // namespace

bool DenseNnue::loaded() const noexcept {
    std::scoped_lock lock(model_mutex_);
    return static_cast<bool>(model_);
}

std::string DenseNnue::source_path() const {
    std::scoped_lock lock(model_mutex_);
    return model_ ? model_->source : std::string{};
}

void DenseNnue::clear() noexcept {
    std::scoped_lock lock(model_mutex_);
    model_.reset();
}

bool DenseNnue::load(const std::filesystem::path& path, std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        if (error != nullptr) {
            *error = "failed to open weights file";
        }
        return false;
    }

    std::array<char, 4> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input || std::string_view(magic.data(), magic.size()) != "NNUE") {
        if (error != nullptr) {
            *error = "invalid weights magic";
        }
        return false;
    }

    std::uint32_t version = 0;
    std::uint32_t layers = 0;
    if (!read_exact(input, version) || !read_exact(input, layers)) {
        if (error != nullptr) {
            *error = "truncated weights header";
        }
        return false;
    }

    if (version != kExpectedVersion || layers != 4U) {
        if (error != nullptr) {
            *error = "unsupported weights layout";
        }
        return false;
    }

    auto model = std::make_shared<ModelData>();
    model->source = path.string();

    struct LayerSpec final {
        std::size_t in;
        std::size_t out;
        std::vector<float>* weights;
        std::vector<float>* bias;
    };

    const std::array<LayerSpec, 4> specs{{
        {kInput, kHidden1, &model->l1_rows, &model->l1_bias},
        {kHidden1, kHidden2, &model->l2_weights, &model->l2_bias},
        {kHidden2, kHidden3, &model->l3_weights, &model->l3_bias},
        {kHidden3, kOutput, &model->l4_weights, &model->l4_bias},
    }};

    for (const auto& spec : specs) {
        std::uint32_t in = 0;
        std::uint32_t out = 0;
        if (!read_exact(input, in) || !read_exact(input, out)) {
            if (error != nullptr) {
                *error = "truncated layer header";
            }
            return false;
        }
        if (in != spec.in || out != spec.out) {
            if (error != nullptr) {
                *error = "unexpected layer dimensions";
            }
            return false;
        }
        if (!read_vector(input, *spec.weights, spec.in * spec.out) ||
            !read_vector(input, *spec.bias, spec.out)) {
            if (error != nullptr) {
                *error = "truncated weights payload";
            }
            return false;
        }
    }

    {
        std::scoped_lock lock(model_mutex_);
        model_ = std::move(model);
    }
    return true;
}

std::vector<std::uint16_t> DenseNnue::extract_active_features(const Position& pos) noexcept {
    std::vector<std::uint16_t> features;
    features.reserve(32);

    for (int square = SQ_A1; square <= SQ_H8; ++square) {
        const auto piece = pos.board[square];
        const int plane = piece_plane(piece);
        if (plane >= 0) {
            features.push_back(static_cast<std::uint16_t>(plane * 64 + square));
        }
    }
    return features;
}

int DenseNnue::infer(const ModelData& model, const Position& pos) noexcept {
    const auto features = extract_active_features(pos);

    std::array<float, kHidden1> hidden1{};
    std::copy(model.l1_bias.begin(), model.l1_bias.end(), hidden1.begin());
    if (!features.empty()) {
        accumulate_sparse(hidden1.data(), model.l1_rows.data(), features.data(), features.size(), kHidden1);
    }
    std::ranges::transform(hidden1, hidden1.begin(), clipped_relu);

    std::array<float, kHidden2> hidden2{};
    dense_forward<kHidden1, kHidden2>(model.l2_weights, model.l2_bias, hidden1, hidden2, true);

    std::array<float, kHidden3> hidden3{};
    dense_forward<kHidden2, kHidden3>(model.l3_weights, model.l3_bias, hidden2, hidden3, true);

    std::array<float, kOutput> output{};
    dense_forward<kHidden3, kOutput>(model.l4_weights, model.l4_bias, hidden3, output, false);

    const float white_pov = output[0] * 400.0F;
    const float stm_pov = pos.side_to_move() == WHITE ? white_pov : -white_pov;
    return static_cast<int>(std::lrint(stm_pov));
}

int DenseNnue::evaluate(const Position& pos) const noexcept {
    std::shared_ptr<ModelData> model_snapshot;
    {
        std::scoped_lock lock(model_mutex_);
        model_snapshot = model_;
    }
    if (!model_snapshot) {
        return 0;
    }
    return infer(*model_snapshot, pos);
}

bool Evaluator::load_weights(const std::filesystem::path& path, std::string* error) {
    return nnue_.load(path, error);
}

void Evaluator::clear_weights() noexcept {
    nnue_.clear();
}

bool Evaluator::has_nnue() const noexcept {
    return nnue_.loaded();
}

int Evaluator::classical(const Position& pos) noexcept {
    return ::Eval::evaluate(pos);
}

int Evaluator::evaluate(const Position& pos) const noexcept {
    const int classical_score = classical(pos);
    if (!nnue_.loaded()) {
        return classical_score;
    }

    const int nnue_score = nnue_.evaluate(pos);
    const int blended = (nnue_score * 3 + classical_score) / 4;
    return std::clamp(blended, -VALUE_MATE + 1, VALUE_MATE - 1);
}

}  // namespace mythos::eval
