#pragma once

#include "../board/position.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace mythos::eval {

class DenseNnue final {
public:
    static constexpr std::uint32_t kExpectedVersion = 1U;
    static constexpr std::size_t kInput = 768;
    static constexpr std::size_t kHidden1 = 256;
    static constexpr std::size_t kHidden2 = 32;
    static constexpr std::size_t kHidden3 = 32;
    static constexpr std::size_t kOutput = 1;

    DenseNnue() = default;

    [[nodiscard]] bool loaded() const noexcept;
    [[nodiscard]] std::string source_path() const;

    void clear() noexcept;
    [[nodiscard]] bool load(const std::filesystem::path& path, std::string* error = nullptr);
    [[nodiscard]] int evaluate(const Position& pos) const noexcept;

private:
    struct ModelData {
        std::string source;
        std::vector<float> l1_rows;
        std::vector<float> l1_bias;
        std::vector<float> l2_weights;
        std::vector<float> l2_bias;
        std::vector<float> l3_weights;
        std::vector<float> l3_bias;
        std::vector<float> l4_weights;
        std::vector<float> l4_bias;
    };

    [[nodiscard]] static std::vector<std::uint16_t> extract_active_features(const Position& pos) noexcept;
    [[nodiscard]] static int infer(const ModelData& model, const Position& pos) noexcept;

    std::shared_ptr<ModelData> model_;
    mutable std::mutex model_mutex_;
};

class Evaluator final {
public:
    Evaluator() = default;

    [[nodiscard]] bool load_weights(const std::filesystem::path& path, std::string* error = nullptr);
    void clear_weights() noexcept;

    [[nodiscard]] bool has_nnue() const noexcept;
    [[nodiscard]] int evaluate(const Position& pos) const noexcept;

private:
    [[nodiscard]] static int classical(const Position& pos) noexcept;

    DenseNnue nnue_;
};

}  // namespace mythos::eval
