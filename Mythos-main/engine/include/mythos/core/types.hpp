#pragma once

#include "../../../types.h"

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace mythos::core {

using ::Bitboard;
using ::Bound;
using ::CastlingRights;
using ::Color;
using ::Direction;
using ::File;
using ::Key;
using ::Move;
using ::MoveType;
using ::Piece;
using ::PieceType;
using ::Rank;
using ::ScoredMove;
using ::Score;
using ::Square;
using ::Value;

inline constexpr auto kMaxMoves = MAX_MOVES;
inline constexpr auto kMaxPly = MAX_PLY;
inline constexpr auto kMaxGameLength = MAX_GAME_LENGTH;

template <typename Enum>
concept EnumType = std::is_enum_v<Enum>;

template <EnumType Enum>
constexpr auto to_underlying(Enum value) noexcept {
    return static_cast<std::underlying_type_t<Enum>>(value);
}

template <typename T>
concept SignedIntegral = std::signed_integral<T>;

template <SignedIntegral T>
constexpr T clamp_eval(T value, T lo, T hi) noexcept {
    return value < lo ? lo : (value > hi ? hi : value);
}

template <typename T>
constexpr bool within(T value, T lo, T hi) noexcept {
    return value >= lo && value <= hi;
}

constexpr std::array<Color, 2> kColors{WHITE, BLACK};

}  // namespace mythos::core
