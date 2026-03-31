#ifndef BITBOARD_H
#define BITBOARD_H

#include "types.h"
#include <string>

// =============================================================================
// BITBOARD MANIPULATION
// =============================================================================

inline int popcount(Bitboard b) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(b);
#elif defined(_MSC_VER)
    return (int)__popcnt64(b);
#else
    int count = 0;
    while (b) { count++; b &= b - 1; }
    return count;
#endif
}

inline Square lsb(Bitboard b) {
    assert(b);
#if defined(__GNUC__) || defined(__clang__)
    return Square(__builtin_ctzll(b));
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return Square(idx);
#else
    // De Bruijn bitscan
    constexpr uint64_t debruijn = 0x03f79d71b4cb0a89ULL;
    constexpr int index64[64] = {
        0,  1, 48,  2, 57, 49, 28,  3, 61, 58, 50, 42, 38, 29, 17,  4,
        62, 55, 59, 36, 53, 51, 43, 22, 45, 39, 33, 30, 24, 18, 12,  5,
        63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11,
        46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19,  9, 13,  8,  7,  6
    };
    return Square(index64[((b & -b) * debruijn) >> 58]);
#endif
}

inline Square msb(Bitboard b) {
    assert(b);
#if defined(__GNUC__) || defined(__clang__)
    return Square(63 ^ __builtin_clzll(b));
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse64(&idx, b);
    return Square(idx);
#else
    constexpr uint64_t debruijn = 0x03f79d71b4cb0a89ULL;
    constexpr int index64[64] = {
        0, 47,  1, 56, 48, 27,  2, 60, 57, 49, 41, 37, 28, 16,  3, 61,
        54, 58, 35, 52, 50, 42, 21, 44, 38, 32, 29, 23, 17, 11,  4, 62,
        46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45,
        25, 39, 14, 33, 19, 30,  9, 24, 13, 18,  8, 12,  7,  6,  5, 63
    };
    b |= b >> 1; b |= b >> 2; b |= b >> 4; b |= b >> 8; b |= b >> 16; b |= b >> 32;
    return Square(index64[(b * debruijn) >> 58]);
#endif
}

inline Square pop_lsb(Bitboard& b) {
    assert(b);
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

inline Bitboard square_bb(Square s) {
    assert(is_ok(s));
    return 1ULL << s;
}

inline bool more_than_one(Bitboard b) {
    return b & (b - 1);
}

// =============================================================================
// BITBOARD OPERATORS
// =============================================================================

constexpr Bitboard operator&(Bitboard b, Square s) { return b & (1ULL << s); }
constexpr Bitboard operator|(Bitboard b, Square s) { return b | (1ULL << s); }
constexpr Bitboard operator^(Bitboard b, Square s) { return b ^ (1ULL << s); }
inline Bitboard& operator|=(Bitboard& b, Square s) { return b |= 1ULL << s; }
inline Bitboard& operator^=(Bitboard& b, Square s) { return b ^= 1ULL << s; }

constexpr Bitboard operator|(Square s1, Square s2) { return (1ULL << s1) | (1ULL << s2); }

// =============================================================================
// FILE/RANK BITBOARDS
// =============================================================================

constexpr Bitboard file_bb(File f) { return FileABB << f; }
constexpr Bitboard file_bb(Square s) { return file_bb(file_of(s)); }

constexpr Bitboard rank_bb(Rank r) { return Rank1BB << (8 * r); }
constexpr Bitboard rank_bb(Square s) { return rank_bb(rank_of(s)); }

// =============================================================================
// SHIFT OPERATIONS
// =============================================================================

template<Direction D>
constexpr Bitboard shift(Bitboard b) {
    return D == NORTH      ? b << 8
         : D == SOUTH      ? b >> 8
         : D == NORTH+NORTH? b << 16
         : D == SOUTH+SOUTH? b >> 16
         : D == EAST       ? (b & ~FileHBB) << 1
         : D == WEST       ? (b & ~FileABB) >> 1
         : D == NORTH_EAST ? (b & ~FileHBB) << 9
         : D == NORTH_WEST ? (b & ~FileABB) << 7
         : D == SOUTH_EAST ? (b & ~FileHBB) >> 7
         : D == SOUTH_WEST ? (b & ~FileABB) >> 9
         : 0;
}

// =============================================================================
// PAWN ATTACK SPANS
// =============================================================================

template<Color C>
constexpr Bitboard pawn_attacks_bb(Bitboard b) {
    return C == WHITE ? shift<NORTH_WEST>(b) | shift<NORTH_EAST>(b)
                      : shift<SOUTH_WEST>(b) | shift<SOUTH_EAST>(b);
}

// =============================================================================
// MAGIC BITBOARDS
// =============================================================================

struct Magic {
    Bitboard  mask;
    Bitboard  magic;
    Bitboard* attacks;
    unsigned  shift;

    unsigned index(Bitboard occupied) const {
        return unsigned(((occupied & mask) * magic) >> shift);
    }
};

extern Magic RookMagics[SQUARE_NB];
extern Magic BishopMagics[SQUARE_NB];

extern Bitboard RookTable[0x19000];
extern Bitboard BishopTable[0x1480];

// =============================================================================
// ATTACK TABLES
// =============================================================================

extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
extern Bitboard KnightAttacks[SQUARE_NB];
extern Bitboard KingAttacks[SQUARE_NB];
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];
extern Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];

// =============================================================================
// ATTACK LOOKUP FUNCTIONS
// =============================================================================

inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occupied) {
    assert(pt != PAWN && pt != NO_PIECE_TYPE);
    switch (pt) {
        case BISHOP: return BishopMagics[s].attacks[BishopMagics[s].index(occupied)];
        case ROOK:   return RookMagics[s].attacks[RookMagics[s].index(occupied)];
        case QUEEN:  return attacks_bb(BISHOP, s, occupied) | attacks_bb(ROOK, s, occupied);
        default:     return PseudoAttacks[pt][s];
    }
}

template<PieceType Pt>
inline Bitboard attacks_bb(Square s, Bitboard occupied) {
    static_assert(Pt != PAWN, "Pawn attacks need color");
    
    if constexpr (Pt == BISHOP)
        return BishopMagics[s].attacks[BishopMagics[s].index(occupied)];
    else if constexpr (Pt == ROOK)
        return RookMagics[s].attacks[RookMagics[s].index(occupied)];
    else if constexpr (Pt == QUEEN)
        return attacks_bb<BISHOP>(s, occupied) | attacks_bb<ROOK>(s, occupied);
    else
        return PseudoAttacks[Pt][s];
}

inline Bitboard pawn_attacks_bb(Color c, Square s) {
    assert(is_ok(s));
    return PawnAttacks[c][s];
}

// =============================================================================
// LINE / BETWEEN
// =============================================================================

inline Bitboard between_bb(Square s1, Square s2) {
    return BetweenBB[s1][s2];
}

inline Bitboard line_bb(Square s1, Square s2) {
    return LineBB[s1][s2];
}

inline bool aligned(Square s1, Square s2, Square s3) {
    return line_bb(s1, s2) & s3;
}

// =============================================================================
// DISTANCE
// =============================================================================

extern int SquareDistance[SQUARE_NB][SQUARE_NB];

inline int distance(Square s1, Square s2) { return SquareDistance[s1][s2]; }
inline int distance(File f1, File f2) { return std::abs(f1 - f2); }
inline int distance(Rank r1, Rank r2) { return std::abs(r1 - r2); }

template<typename T1, typename T2>
inline int distance(T1 x, T2 y);

template<>
inline int distance<Square, Square>(Square x, Square y) {
    return SquareDistance[x][y];
}

template<>
inline int distance<File, File>(File x, File y) {
    return std::abs(x - y);
}

template<>
inline int distance<Rank, Rank>(Rank x, Rank y) {
    return std::abs(x - y);
}

// =============================================================================
// PASSED PAWN / PAWN STRUCTURE
// =============================================================================

inline Bitboard forward_ranks_bb(Color c, Square s) {
    return c == WHITE ? ~Rank1BB << 8 * relative_rank(WHITE, s)
                      : ~Rank8BB >> 8 * relative_rank(BLACK, s);
}

inline Bitboard forward_file_bb(Color c, Square s) {
    return forward_ranks_bb(c, s) & file_bb(s);
}

inline Bitboard pawn_attack_span(Color c, Square s) {
    return forward_ranks_bb(c, s) & (shift<WEST>(file_bb(s)) | shift<EAST>(file_bb(s)));
}

inline Bitboard passed_pawn_span(Color c, Square s) {
    return pawn_attack_span(c, s) | forward_file_bb(c, s);
}

// =============================================================================
// INITIALIZATION
// =============================================================================

namespace Bitboards {
    void init();
    std::string pretty(Bitboard b);
}

#endif // BITBOARD_H
