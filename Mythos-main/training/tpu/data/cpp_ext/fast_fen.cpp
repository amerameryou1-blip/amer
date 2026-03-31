#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::size_t kFeatureCount = 768;

struct DecodedBatch final {
    std::vector<float> features;
    std::vector<float> targets;
    std::size_t count = 0;
};

[[nodiscard]] int plane_for_piece(char piece) noexcept {
    switch (piece) {
        case 'P': return 0;
        case 'N': return 1;
        case 'B': return 2;
        case 'R': return 3;
        case 'Q': return 4;
        case 'K': return 5;
        case 'p': return 6;
        case 'n': return 7;
        case 'b': return 8;
        case 'r': return 9;
        case 'q': return 10;
        case 'k': return 11;
        default: return -1;
    }
}

[[nodiscard]] bool decode_single_fen(
    const char* data,
    std::size_t size,
    float* out_features,
    bool* black_to_move) noexcept {
    std::memset(out_features, 0, sizeof(float) * kFeatureCount);

    int square = 56;
    std::size_t index = 0;
    bool saw_white_king = false;
    bool saw_black_king = false;

    while (index < size && data[index] != ' ') {
        const unsigned char ch = static_cast<unsigned char>(data[index]);
        if (ch == '/') {
            square -= 16;
        } else if (ch >= '1' && ch <= '8') {
            square += static_cast<int>(ch - '0');
        } else {
            const int plane = plane_for_piece(static_cast<char>(ch));
            if (plane < 0 || square < 0 || square >= 64) {
                return false;
            }
            out_features[plane * 64 + square] = 1.0F;
            saw_white_king = saw_white_king || ch == 'K';
            saw_black_king = saw_black_king || ch == 'k';
            ++square;
        }
        ++index;
    }

    if (!saw_white_king || !saw_black_king) {
        return false;
    }
    if (square < 0 || square > 64) {
        return false;
    }
    if (index >= size) {
        *black_to_move = false;
        return true;
    }

    while (index < size && data[index] == ' ') {
        ++index;
    }
    if (index >= size) {
        *black_to_move = false;
        return true;
    }

    *black_to_move = data[index] == 'b';
    return true;
}

[[nodiscard]] DecodedBatch decode_binary_records_impl(py::buffer binary_records, bool flip_to_stm) {
    const py::buffer_info info = binary_records.request();
    if (info.ndim != 1) {
        throw std::runtime_error("binary record blob must be a 1D byte buffer");
    }
    if (info.itemsize != 1) {
        throw std::runtime_error("binary record blob itemsize must be 1");
    }

    const auto* bytes = static_cast<const std::uint8_t*>(info.ptr);
    const std::size_t size = static_cast<std::size_t>(info.size);

    std::size_t offset = 0;
    std::size_t record_count = 0;
    while (offset + 3 <= size) {
        const std::uint16_t fen_length =
            static_cast<std::uint16_t>(bytes[offset]) |
            static_cast<std::uint16_t>(bytes[offset + 1] << 8U);
        offset += 2;
        if (offset + fen_length + 1 > size) {
            break;
        }
        offset += fen_length + 1;
        ++record_count;
    }

    DecodedBatch batch;
    batch.features.assign(record_count * kFeatureCount, 0.0F);
    batch.targets.assign(record_count, 0.0F);

    offset = 0;
    std::size_t out_index = 0;
    while (offset + 3 <= size) {
        const std::uint16_t fen_length =
            static_cast<std::uint16_t>(bytes[offset]) |
            static_cast<std::uint16_t>(bytes[offset + 1] << 8U);
        offset += 2;
        if (offset + fen_length + 1 > size) {
            break;
        }

        const char* fen_ptr = reinterpret_cast<const char*>(bytes + offset);
        offset += fen_length;
        const std::int8_t result = static_cast<std::int8_t>(bytes[offset]);
        ++offset;

        bool black_to_move = false;
        float* feature_row = batch.features.data() + out_index * kFeatureCount;
        if (!decode_single_fen(fen_ptr, fen_length, feature_row, &black_to_move)) {
            continue;
        }

        float target = static_cast<float>(result);
        if (flip_to_stm && black_to_move) {
            target = -target;
        }
        batch.targets[out_index] = target;
        ++out_index;
    }

    batch.count = out_index;
    batch.features.resize(out_index * kFeatureCount);
    batch.targets.resize(out_index);
    return batch;
}

py::tuple decode_binary_records(py::buffer binary_records, bool flip_to_stm) {
    DecodedBatch batch = decode_binary_records_impl(binary_records, flip_to_stm);

    py::array_t<float> feature_array({static_cast<py::ssize_t>(batch.count), static_cast<py::ssize_t>(kFeatureCount)});
    py::array_t<float> target_array({static_cast<py::ssize_t>(batch.count), static_cast<py::ssize_t>(1)});

    if (batch.count > 0) {
        std::memcpy(feature_array.mutable_data(), batch.features.data(), batch.features.size() * sizeof(float));
        float* target_ptr = target_array.mutable_data();
        for (std::size_t i = 0; i < batch.count; ++i) {
            target_ptr[i] = batch.targets[i];
        }
    }

    return py::make_tuple(feature_array, target_array, static_cast<py::int_>(batch.count));
}

}  // namespace

PYBIND11_MODULE(_fast_fen, module) {
    module.doc() = "Fast binary-record FEN decoder for Mythos TPU training";
    module.def(
        "decode_binary_records",
        &decode_binary_records,
        py::arg("binary_records"),
        py::arg("flip_to_stm") = true);
}
