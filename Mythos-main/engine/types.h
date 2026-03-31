#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <string>
#include <cassert>
#include <algorithm>

// =============================================================================
// BASIC TYPES
// =============================================================================

using Bitboard = uint64_t;
using Key = uint64_t;
using Score = int;
typedef int Value;

constexpr int MAX_MOVES = 256;
constexpr int MAX_PLY = 128;
constexpr int MAX_GAME_LENGTH = 1024;

// =============================================================================
// PIECE TYPES AND COLORS
// =============================================================================

enum Color : int {
    WHITE, BLACK, COLOR_NB = 2
};

enum PieceType : int {
    NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    PIECE_TYPE_NB = 8
};

enum Piece : int {
    NO_PIECE,
    W_PAWN = PAWN,     W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = PAWN + 8, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_NB = 16
};

// =============================================================================
// SQUARES, FILES, RANKS
// =============================================================================

enum Square : int {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE,
    SQUARE_NB = 64
};

enum File : int {
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB
};

enum Rank : int {
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB
};

// =============================================================================
// DIRECTIONS
// =============================================================================

enum Direction : int {
    NORTH =  8,
    EAST  =  1,
    SOUTH = -8,
    WEST  = -1,

    NORTH_EAST = NORTH + EAST,
    NORTH_WEST = NORTH + WEST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST
};

// =============================================================================
// CASTLING RIGHTS
// =============================================================================

enum CastlingRights : int {
    NO_CASTLING,
    WHITE_OO  = 1,
    WHITE_OOO = 2,
    BLACK_OO  = 4,
    BLACK_OOO = 8,
    
    KING_SIDE      = WHITE_OO  | BLACK_OO,
    QUEEN_SIDE     = WHITE_OOO | BLACK_OOO,
    WHITE_CASTLING = WHITE_OO  | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO  | BLACK_OOO,
    ANY_CASTLING   = WHITE_CASTLING | BLACK_CASTLING,
    
    CASTLING_RIGHT_NB = 16
};

// =============================================================================
// MOVE TYPES
// =============================================================================

enum MoveType : int {
    NORMAL,
    PROMOTION    = 1 << 14,
    EN_PASSANT   = 2 << 14,
    CASTLING     = 3 << 14
};

// =============================================================================
// VALUES
// =============================================================================

enum : int {
    VALUE_ZERO      = 0,
    VALUE_DRAW      = 0,
    VALUE_KNOWN_WIN = 10000,
    VALUE_MATE      = 32000,
    VALUE_INFINITE  = 32001,
    VALUE_NONE      = 32002,

    VALUE_TB_WIN_IN_MAX_PLY  =  VALUE_MATE - 2 * MAX_PLY,
    VALUE_TB_LOSS_IN_MAX_PLY = -VALUE_TB_WIN_IN_MAX_PLY,
    VALUE_MATE_IN_MAX_PLY    =  VALUE_MATE - MAX_PLY,
    VALUE_MATED_IN_MAX_PLY   = -VALUE_MATE_IN_MAX_PLY,

    // Piece values (centipawns)
    PawnValueMg   = 126,   PawnValueEg   = 208,
    KnightValueMg = 781,   KnightValueEg = 854,
    BishopValueMg = 825,   BishopValueEg = 915,
    RookValueMg   = 1276,  RookValueEg   = 1380,
    QueenValueMg  = 2538,  QueenValueEg  = 2682,

    MidgameLimit  = 15258,
    EndgameLimit  = 3915
};

// =============================================================================
// BOUND TYPE FOR TT
// =============================================================================

enum Bound : int {
    BOUND_NONE,
    BOUND_UPPER,
    BOUND_LOWER,
    BOUND_EXACT = BOUND_UPPER | BOUND_LOWER
};

// =============================================================================
// BITBOARD CONSTANTS
// =============================================================================

constexpr Bitboard FileABB = 0x0101010101010101ULL;
constexpr Bitboard FileBBB = FileABB << 1;
constexpr Bitboard FileCBB = FileABB << 2;
constexpr Bitboard FileDBB = FileABB << 3;
constexpr Bitboard FileEBB = FileABB << 4;
constexpr Bitboard FileFBB = FileABB << 5;
constexpr Bitboard FileGBB = FileABB << 6;
constexpr Bitboard FileHBB = FileABB << 7;

constexpr Bitboard Rank1BB = 0xFFULL;
constexpr Bitboard Rank2BB = Rank1BB << (8 * 1);
constexpr Bitboard Rank3BB = Rank1BB << (8 * 2);
constexpr Bitboard Rank4BB = Rank1BB << (8 * 3);
constexpr Bitboard Rank5BB = Rank1BB << (8 * 4);
constexpr Bitboard Rank6BB = Rank1BB << (8 * 5);
constexpr Bitboard Rank7BB = Rank1BB << (8 * 6);
constexpr Bitboard Rank8BB = Rank1BB << (8 * 7);

constexpr Bitboard DarkSquares  = 0xAA55AA55AA55AA55ULL;
constexpr Bitboard LightSquares = ~DarkSquares;

constexpr Bitboard AllSquares = ~Bitboard(0);
constexpr Bitboard Center     = (FileDBB | FileEBB) & (Rank4BB | Rank5BB);

// =============================================================================
// OPERATOR OVERLOADS
// =============================================================================

#define ENABLE_INCR_OPERATORS_ON(T)                                 \
inline T& operator++(T& d) { return d = T(int(d) + 1); }            \
inline T& operator--(T& d) { return d = T(int(d) - 1); }

ENABLE_INCR_OPERATORS_ON(Square)
ENABLE_INCR_OPERATORS_ON(File)
ENABLE_INCR_OPERATORS_ON(Rank)
ENABLE_INCR_OPERATORS_ON(PieceType)
ENABLE_INCR_OPERATORS_ON(Piece)

#undef ENABLE_INCR_OPERATORS_ON

constexpr Color operator~(Color c) { return Color(c ^ BLACK); }

constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
inline Square& operator+=(Square& s, Direction d) { return s = s + d; }
inline Square& operator-=(Square& s, Direction d) { return s = s - d; }

constexpr Direction operator+(Direction d1, Direction d2) { return Direction(int(d1) + int(d2)); }
constexpr Direction operator-(Direction d1, Direction d2) { return Direction(int(d1) - int(d2)); }
constexpr Direction operator*(int i, Direction d) { return Direction(i * int(d)); }
constexpr Direction operator-(Direction d) { return Direction(-int(d)); }

constexpr CastlingRights operator|(CastlingRights cr1, CastlingRights cr2) {
    return CastlingRights(int(cr1) | int(cr2));
}
constexpr CastlingRights operator&(CastlingRights cr1, CastlingRights cr2) {
    return CastlingRights(int(cr1) & int(cr2));
}
inline CastlingRights& operator|=(CastlingRights& cr1, CastlingRights cr2) {
    return cr1 = cr1 | cr2;
}
inline CastlingRights& operator&=(CastlingRights& cr1, CastlingRights cr2) {
    return cr1 = cr1 & cr2;
}
constexpr CastlingRights operator~(CastlingRights cr) {
    return CastlingRights(~int(cr) & 15);
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

constexpr bool is_ok(Square s) { return s >= SQ_A1 && s <= SQ_H8; }

constexpr File file_of(Square s) { return File(s & 7); }
constexpr Rank rank_of(Square s) { return Rank(s >> 3); }

constexpr Square make_square(File f, Rank r) { return Square((r << 3) + f); }

constexpr Piece make_piece(Color c, PieceType pt) { return Piece((c << 3) + pt); }
constexpr PieceType type_of(Piece pc) { return PieceType(pc & 7); }
constexpr Color color_of(Piece pc) { 
    assert(pc != NO_PIECE);
    return Color(pc >> 3); 
}

constexpr bool opposite_colors(Square s1, Square s2) {
    return (s1 + rank_of(s1) + s2 + rank_of(s2)) & 1;
}

constexpr Square relative_square(Color c, Square s) {
    return Square(s ^ (c * 56));
}

constexpr Rank relative_rank(Color c, Rank r) {
    return Rank(r ^ (c * 7));
}

constexpr Rank relative_rank(Color c, Square s) {
    return relative_rank(c, rank_of(s));
}

constexpr Direction pawn_push(Color c) {
    return c == WHITE ? NORTH : SOUTH;
}

constexpr CastlingRights castling_rights(Color c, bool kingside) {
    return kingside ? (c == WHITE ? WHITE_OO : BLACK_OO)
                    : (c == WHITE ? WHITE_OOO : BLACK_OOO);
}

inline int edge_distance(File f) { return std::min(f, File(FILE_H - f)); }
inline int edge_distance(Rank r) { return std::min(r, Rank(RANK_8 - r)); }

// =============================================================================
// MOVE CLASS
// =============================================================================

class Move {
public:
    Move() = default;
    constexpr explicit Move(uint16_t d) : data(d) {}
    
    constexpr Move(Square from, Square to) : data((from << 6) + to) {}
    
    constexpr Move(Square from, Square to, MoveType mt, PieceType pt = KNIGHT)
        : data((from << 6) + to + mt + ((pt - KNIGHT) << 12)) {}

    constexpr Square from() const { return Square((data >> 6) & 0x3F); }
    constexpr Square to() const { return Square(data & 0x3F); }
    constexpr int from_to() const { return data & 0xFFF; }
    
    constexpr MoveType type() const { return MoveType(data & (3 << 14)); }
    constexpr PieceType promotion_type() const { return PieceType(((data >> 12) & 3) + KNIGHT); }

    constexpr bool operator==(const Move& m) const { return data == m.data; }
    constexpr bool operator!=(const Move& m) const { return data != m.data; }
    
    constexpr explicit operator bool() const { return data != 0; }
    
    constexpr uint16_t raw() const { return data; }
    
    static constexpr Move none() { return Move(0); }
    static constexpr Move null() { return Move(65); }
    
    std::string to_uci() const {
        if (!*this) return "0000";
        
        std::string uci;
        uci += char('a' + file_of(from()));
        uci += char('1' + rank_of(from()));
        uci += char('a' + file_of(to()));
        uci += char('1' + rank_of(to()));
        
        if (type() == PROMOTION) {
            uci += " nbrq"[promotion_type() - KNIGHT + 1];
        }
        
        return uci;
    }

private:
    uint16_t data = 0;
};

constexpr Move MOVE_NONE = Move::none();
constexpr Move MOVE_NULL = Move::null();

// =============================================================================
// SCORED MOVE FOR MOVE ORDERING
// =============================================================================

struct ScoredMove {
    Move move;
    int score;
    
    ScoredMove() : move(MOVE_NONE), score(0) {}
    ScoredMove(Move m, int s = 0) : move(m), score(s) {}
    
    bool operator<(const ScoredMove& other) const { return score > other.score; }
};

// =============================================================================
// MATE VALUE HELPERS
// =============================================================================

constexpr Value mate_in(int ply) { return Value(VALUE_MATE - ply); }
constexpr Value mated_in(int ply) { return Value(-VALUE_MATE + ply); }

// =============================================================================
// PIECE VALUE ARRAY
// =============================================================================

constexpr Value PieceValue[PIECE_TYPE_NB] = {
    VALUE_ZERO, Value(PawnValueMg), Value(KnightValueMg), Value(BishopValueMg),
    Value(RookValueMg), Value(QueenValueMg), VALUE_ZERO, VALUE_ZERO
};

// =============================================================================
// START POSITION
// =============================================================================

constexpr const char* StartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#endif // TYPES_H
