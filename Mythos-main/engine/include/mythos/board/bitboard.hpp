#pragma once

#include "../core/types.hpp"
#include "../../../bitboard.h"

namespace mythos::board {

using ::BishopMagics;
using ::BishopTable;
using ::BetweenBB;
using ::Bitboards::init;
using ::Bitboards::pretty;
using ::KingAttacks;
using ::KnightAttacks;
using ::LineBB;
using ::Magic;
using ::PawnAttacks;
using ::PseudoAttacks;
using ::RookMagics;
using ::RookTable;
using ::SquareDistance;
using ::aligned;
using ::attacks_bb;
using ::between_bb;
using ::distance;
using ::file_bb;
using ::forward_file_bb;
using ::forward_ranks_bb;
using ::line_bb;
using ::lsb;
using ::more_than_one;
using ::msb;
using ::passed_pawn_span;
using ::pawn_attack_span;
using ::pawn_attacks_bb;
using ::pop_lsb;
using ::popcount;
using ::rank_bb;
using ::shift;
using ::square_bb;

}  // namespace mythos::board
