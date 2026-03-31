#include "evaluate.h"
#include "bitboard.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iomanip>

namespace Eval {

namespace {

// =============================================================================
// PIECE-SQUARE TABLES
// =============================================================================

// Values are from White's perspective, for squares A1..H8
// Access as PST[piece][square]

constexpr int PawnMgPST[SQUARE_NB] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     -1,  -7, -11, -35, -13,   5,   3,  -5,
     -3,  -5,   2, -25,   1,  10,   7,   4,
     -2,  10,  -2,  -2,   8,   6,  10,   5,
     -5,   1,  13,  29,  38,  21,   7,  -7,
     -7,   7,  -3,  -13, -27, -22,  24,   3,
    -19, -16,  19,  17,  13,  37,   7, -17,
      0,   0,   0,   0,   0,   0,   0,   0
};

constexpr int PawnEgPST[SQUARE_NB] = {
      0,   0,   0,   0,   0,   0,   0,   0,
     -4,  -6,   1,   1,   1,   0,  -3, -11,
     -7,  -2,   4,  -2,   4,   5,   5,  -8,
    -10,   4,   4,  12,   8,   1,   6, -16,
     -4,   9,   5,  -7,  -8,  -4,  12,  -9,
     10,   3, -12, -14, -17, -12,   5,  -1,
    -11,  21,   2,  -1,  -4, -10,  15,   0,
      0,   0,   0,   0,   0,   0,   0,   0
};

constexpr int KnightMgPST[SQUARE_NB] = {
   -175, -92, -74, -73, -73, -74, -92,-175,
    -77, -41, -27, -15, -15, -27, -41, -77,
    -61, -17,   6,  12,  12,   6, -17, -61,
    -35,   8,  40,  49,  49,  40,   8, -35,
    -34,  13,  44,  51,  51,  44,  13, -34,
    -11,  28,  63,  55,  55,  63,  28, -11,
    -67, -21,   6,  37,  37,   6, -21, -67,
   -201, -83, -56, -26, -26, -56, -83,-201
};

constexpr int KnightEgPST[SQUARE_NB] = {
    -96, -65, -49, -21, -21, -49, -65, -96,
    -67, -54, -18,   8,   8, -18, -54, -67,
    -40, -27,  -8,  29,  29,  -8, -27, -40,
    -35,  -2,  13,  28,  28,  13,  -2, -35,
    -45, -16,   9,  39,  39,   9, -16, -45,
    -51, -44,  -5,  17,  17,  -5, -44, -51,
    -69, -50, -51,  12,  12, -51, -50, -69,
   -100, -88, -56, -17, -17, -56, -88,-100
};

constexpr int BishopMgPST[SQUARE_NB] = {
    -37,  -4, -19, -38, -38, -19,  -4, -37,
    -22,  -3,  -1, -16, -16,  -1,  -3, -22,
    -27,  -7,  -3,  21,  21,  -3,  -7, -27,
     -1,  11,  13,  26,  26,  13,  11,  -1,
     -9,  18,  11,  13,  13,  11,  18,  -9,
    -13,  19,  11,  -3,  -3,  11,  19, -13,
    -16,  12,   2,   6,   6,   2,  12, -16,
    -26,  -8, -25,  -2,  -2, -25,  -8, -26
};

constexpr int BishopEgPST[SQUARE_NB] = {
    -40, -21, -26, -30, -30, -26, -21, -40,
    -23, -12, -19,  -6,  -6, -19, -12, -23,
    -12,  -9,   1,   0,   0,   1,  -9, -12,
     -9,   5,  -1,  17,  17,  -1,   5,  -9,
    -14,   1,  -1,  15,  15,  -1,   1, -14,
    -21,  -7,   2,   5,   5,   2,  -7, -21,
    -27, -15, -10,   1,   1, -10, -15, -27,
    -32, -22, -18, -19, -19, -18, -22, -32
};

constexpr int RookMgPST[SQUARE_NB] = {
    -31, -20, -14,  -5,  -5, -14, -20, -31,
    -21, -13,  -8,   6,   6,  -8, -13, -21,
    -25, -11,  -1,   3,   3,  -1, -11, -25,
    -13,  -5,  -4,  -6,  -6,  -4,  -5, -13,
    -27, -15,  -4,   3,   3,  -4, -15, -27,
    -22,  -2,   6,  12,  12,   6,  -2, -22,
     -2,  12,  16,  18,  18,  16,  12,  -2,
    -17, -19,  -1,   9,   9,  -1, -19, -17
};

constexpr int RookEgPST[SQUARE_NB] = {
      3,   0,   3,   7,   7,   3,   0,   3,
      5,   8,   8,  -3,  -3,   8,   8,   5,
     -6,   1,  -2,   6,   6,  -2,   1,  -6,
     -5,   5,   8,  11,  11,   8,   5,  -5,
    -10,   7,   7,   4,   4,   7,   7, -10,
     -8,   2,   5,   2,   2,   5,   2,  -8,
    -16,   5,   6,  -6,  -6,   6,   5, -16,
     -4,   2,   2,  -1,  -1,   2,   2,  -4
};

constexpr int QueenMgPST[SQUARE_NB] = {
      1,  -3,  -3,   0,   0,  -3,  -3,   1,
     -4,  -1,   4,   5,   5,   4,  -1,  -4,
     -7,   4,   9,   8,   8,   9,   4,  -7,
    -10,   9,  15,  17,  17,  15,   9, -10,
     -4,   8,  15,  17,  17,  15,   8,  -4,
     -5,  14,  13,  19,  19,  13,  14,  -5,
     -5,   8,  15,   6,   6,  15,   8,  -5,
    -10,  -8,  -1,  -7,  -7,  -1,  -8, -10
};

constexpr int QueenEgPST[SQUARE_NB] = {
    -53, -36, -24, -11, -11, -24, -36, -53,
    -30, -27, -10,   7,   7, -10, -27, -30,
    -26,  -9,  16,  10,  10,  16,  -9, -26,
    -18,  28,  19,  47,  47,  19,  28, -18,
    -16,  31,  45,  54,  54,  45,  31, -16,
    -10,  24,  31,  35,  35,  31,  24, -10,
    -34, -14, -10,  19,  19, -10, -14, -34,
    -27, -25, -19, -12, -12, -19, -25, -27
};

constexpr int KingMgPST[SQUARE_NB] = {
    271, 327, 271, 198, 198, 271, 327, 271,
    278, 303, 234, 179, 179, 234, 303, 278,
    195, 258, 169, 120, 120, 169, 258, 195,
    164, 190, 138,  98,  98, 138, 190, 164,
    154, 179, 105,  70,  70, 105, 179, 154,
    123, 145,  81,  31,  31,  81, 145, 123,
     88, 120,  65,  33,  33,  65, 120,  88,
     59,  89,  45,  -1,  -1,  45,  89,  59
};

constexpr int KingEgPST[SQUARE_NB] = {
      1,  45,  85,  76,  76,  85,  45,   1,
     53,  95, 102, 101, 101, 102,  95,  53,
     88, 108, 132, 128, 128, 132, 108,  88,
     73, 102, 138, 148, 148, 138, 102,  73,
     65,  95, 125, 134, 134, 125,  95,  65,
     47,  69, 105, 110, 110, 105,  69,  47,
     21,  44,  62,  80,  80,  62,  44,  21,
      0,  24,  42,  46,  46,  42,  24,   0
};

// Piece values for phase calculation
constexpr int PhaseValue[PIECE_TYPE_NB] = { 0, 0, 1, 1, 2, 4, 0, 0 };
constexpr int TotalPhase = 24;  // 2*1 + 2*1 + 2*2 + 1*4 = 4 knights + 4 bishops + 4 rooks + 2 queens

// =============================================================================
// EVALUATION TERMS
// =============================================================================

// Mobility bonuses
constexpr int KnightMobility[9] = { -62, -53, -12, -4, 3, 13, 22, 28, 33 };
constexpr int BishopMobility[14] = { -48, -20, 16, 26, 38, 51, 55, 63, 63, 68, 81, 81, 91, 98 };
constexpr int RookMobility[15] = { -60, -20, 2, 3, 3, 11, 22, 31, 40, 40, 41, 48, 57, 57, 62 };
constexpr int QueenMobility[28] = { -30, -12, -8, -9, 20, 23, 23, 35, 38, 53, 64, 65, 65, 
                                     66, 67, 67, 72, 72, 77, 79, 93, 108, 108, 108, 110, 114, 114, 116 };

// Pawn structure
constexpr int DoubledPawnPenalty = -11;
constexpr int IsolatedPawnPenalty = -5;
constexpr int BackwardPawnPenalty = -9;
constexpr int PassedPawnBonus[RANK_NB] = { 0, 10, 17, 15, 62, 168, 276, 0 };

// Piece bonuses
constexpr int BishopPairBonus = 30;
constexpr int RookOnOpenFile = 43;
constexpr int RookOnSemiOpenFile = 19;
constexpr int RookOn7th = 11;

// King safety
constexpr int KingShieldBonus = 9;
constexpr int KingAttackWeight[PIECE_TYPE_NB] = { 0, 0, 2, 2, 3, 5, 0, 0 };

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

inline Square flip_rank(Square s) {
    return Square(s ^ 56);
}

inline int pst_value(PieceType pt, Square s, bool mg) {
    switch (pt) {
        case PAWN:   return mg ? PawnMgPST[s]   : PawnEgPST[s];
        case KNIGHT: return mg ? KnightMgPST[s] : KnightEgPST[s];
        case BISHOP: return mg ? BishopMgPST[s] : BishopEgPST[s];
        case ROOK:   return mg ? RookMgPST[s]   : RookEgPST[s];
        case QUEEN:  return mg ? QueenMgPST[s]  : QueenEgPST[s];
        case KING:   return mg ? KingMgPST[s]   : KingEgPST[s];
        default:     return 0;
    }
}

// =============================================================================
// MAIN EVALUATION
// =============================================================================

template<Color Us>
int evaluate_side(const Position& pos, int& mgScore, int& egScore) {
    constexpr Color Them = ~Us;
    
    Bitboard ourPawns = pos.pieces(Us, PAWN);
    Bitboard theirPawns = pos.pieces(Them, PAWN);
    Bitboard occupied = pos.pieces();
    
    Square ksq = pos.king_square(Us);
    Square theirKsq = pos.king_square(Them);
    
    int attackUnits = 0;
    int attackers = 0;
    Bitboard kingZone = KingAttacks[theirKsq] | theirKsq;
    
    // Pawns
    Bitboard b = ourPawns;
    while (b) {
        Square s = pop_lsb(b);
        Square relSq = Us == WHITE ? s : flip_rank(s);
        mgScore += PawnMgPST[relSq] + PawnValueMg;
        egScore += PawnEgPST[relSq] + PawnValueEg;
        
        // Doubled pawn
        if (ourPawns & forward_file_bb(Us, s)) {
            mgScore += DoubledPawnPenalty;
            egScore += DoubledPawnPenalty;
        }
        
        // Isolated pawn
        Bitboard adjacent = shift<EAST>(file_bb(s)) | shift<WEST>(file_bb(s));
        if (!(ourPawns & adjacent)) {
            mgScore += IsolatedPawnPenalty;
            egScore += IsolatedPawnPenalty;
        }
        
        // Passed pawn
        if (!(passed_pawn_span(Us, s) & theirPawns)) {
            Rank r = relative_rank(Us, s);
            mgScore += PassedPawnBonus[r];
            egScore += PassedPawnBonus[r] * 2;  // More valuable in endgame
        }
    }
    
    // Knights
    b = pos.pieces(Us, KNIGHT);
    while (b) {
        Square s = pop_lsb(b);
        Square relSq = Us == WHITE ? s : flip_rank(s);
        mgScore += KnightMgPST[relSq] + KnightValueMg;
        egScore += KnightEgPST[relSq] + KnightValueEg;
        
        // Mobility
        Bitboard attacks = attacks_bb<KNIGHT>(s, occupied) & ~pos.pieces(Us);
        int mobility = popcount(attacks);
        mgScore += KnightMobility[std::min(mobility, 8)];
        egScore += KnightMobility[std::min(mobility, 8)];
        
        // King attacks
        if (attacks & kingZone) {
            attackUnits += KingAttackWeight[KNIGHT] * popcount(attacks & kingZone);
            attackers++;
        }
    }
    
    // Bishops
    b = pos.pieces(Us, BISHOP);
    int bishopCount = 0;
    while (b) {
        Square s = pop_lsb(b);
        Square relSq = Us == WHITE ? s : flip_rank(s);
        mgScore += BishopMgPST[relSq] + BishopValueMg;
        egScore += BishopEgPST[relSq] + BishopValueEg;
        bishopCount++;
        
        // Mobility (x-ray through queens)
        Bitboard attacks = attacks_bb<BISHOP>(s, occupied ^ pos.pieces(Us, QUEEN));
        attacks &= ~pos.pieces(Us);
        int mobility = popcount(attacks);
        mgScore += BishopMobility[std::min(mobility, 13)];
        egScore += BishopMobility[std::min(mobility, 13)];
        
        // King attacks
        if (attacks & kingZone) {
            attackUnits += KingAttackWeight[BISHOP] * popcount(attacks & kingZone);
            attackers++;
        }
    }
    if (bishopCount >= 2) {
        mgScore += BishopPairBonus;
        egScore += BishopPairBonus;
    }
    
    // Rooks
    b = pos.pieces(Us, ROOK);
    while (b) {
        Square s = pop_lsb(b);
        Square relSq = Us == WHITE ? s : flip_rank(s);
        mgScore += RookMgPST[relSq] + RookValueMg;
        egScore += RookEgPST[relSq] + RookValueEg;
        
        // Open/semi-open files
        File f = file_of(s);
        if (!(ourPawns & file_bb(f))) {
            if (!(theirPawns & file_bb(f))) {
                mgScore += RookOnOpenFile;
                egScore += RookOnOpenFile;
            } else {
                mgScore += RookOnSemiOpenFile;
                egScore += RookOnSemiOpenFile;
            }
        }
        
        // 7th rank
        if (relative_rank(Us, s) == RANK_7) {
            mgScore += RookOn7th;
            egScore += RookOn7th;
        }
        
        // Mobility (x-ray through queens and rooks)
        Bitboard attacks = attacks_bb<ROOK>(s, occupied ^ pos.pieces(Us, QUEEN, ROOK));
        attacks &= ~pos.pieces(Us);
        int mobility = popcount(attacks);
        mgScore += RookMobility[std::min(mobility, 14)];
        egScore += RookMobility[std::min(mobility, 14)];
        
        // King attacks
        if (attacks & kingZone) {
            attackUnits += KingAttackWeight[ROOK] * popcount(attacks & kingZone);
            attackers++;
        }
    }
    
    // Queens
    b = pos.pieces(Us, QUEEN);
    while (b) {
        Square s = pop_lsb(b);
        Square relSq = Us == WHITE ? s : flip_rank(s);
        mgScore += QueenMgPST[relSq] + QueenValueMg;
        egScore += QueenEgPST[relSq] + QueenValueEg;
        
        // Mobility
        Bitboard attacks = attacks_bb<QUEEN>(s, occupied);
        attacks &= ~pos.pieces(Us);
        int mobility = popcount(attacks);
        mgScore += QueenMobility[std::min(mobility, 27)];
        egScore += QueenMobility[std::min(mobility, 27)];
        
        // King attacks
        if (attacks & kingZone) {
            attackUnits += KingAttackWeight[QUEEN] * popcount(attacks & kingZone);
            attackers++;
        }
    }
    
    // King PST
    {
        Square relSq = Us == WHITE ? ksq : flip_rank(ksq);
        mgScore += KingMgPST[relSq];
        egScore += KingEgPST[relSq];
        
        // Pawn shield
        Bitboard shield = pawn_attacks_bb(Them, ksq) | shift<pawn_push(Us)>(square_bb(ksq));
        int shieldPawns = popcount(shield & ourPawns);
        mgScore += shieldPawns * KingShieldBonus;
    }
    
    // King safety penalty (only if enough attackers)
    if (attackers >= 2) {
        mgScore -= attackUnits * attackUnits / 100;
    }
    
    return 0;  // scores accumulated in mgScore/egScore
}

} // anonymous namespace

// =============================================================================
// PUBLIC API
// =============================================================================

int game_phase(const Position& pos) {
    int phase = 0;
    phase += popcount(pos.pieces(KNIGHT)) * PhaseValue[KNIGHT];
    phase += popcount(pos.pieces(BISHOP)) * PhaseValue[BISHOP];
    phase += popcount(pos.pieces(ROOK)) * PhaseValue[ROOK];
    phase += popcount(pos.pieces(QUEEN)) * PhaseValue[QUEEN];
    return std::min(phase, TotalPhase);
}

Value material(const Position& pos) {
    int score = 0;
    score += pos.count(WHITE, PAWN) * PawnValueMg;
    score += pos.count(WHITE, KNIGHT) * KnightValueMg;
    score += pos.count(WHITE, BISHOP) * BishopValueMg;
    score += pos.count(WHITE, ROOK) * RookValueMg;
    score += pos.count(WHITE, QUEEN) * QueenValueMg;
    score -= pos.count(BLACK, PAWN) * PawnValueMg;
    score -= pos.count(BLACK, KNIGHT) * KnightValueMg;
    score -= pos.count(BLACK, BISHOP) * BishopValueMg;
    score -= pos.count(BLACK, ROOK) * RookValueMg;
    score -= pos.count(BLACK, QUEEN) * QueenValueMg;
    
    return pos.side_to_move() == WHITE ? Value(score) : Value(-score);
}

Value evaluate(const Position& pos) {
    // Check for insufficient material draw
    if (pos.is_insufficient_material()) {
        return VALUE_DRAW;
    }
    
    int mgWhite = 0, egWhite = 0;
    int mgBlack = 0, egBlack = 0;
    
    evaluate_side<WHITE>(pos, mgWhite, egWhite);
    evaluate_side<BLACK>(pos, mgBlack, egBlack);
    
    int mgScore = mgWhite - mgBlack;
    int egScore = egWhite - egBlack;
    
    // Tapered evaluation
    int phase = game_phase(pos);
    int score = (mgScore * phase + egScore * (TotalPhase - phase)) / TotalPhase;
    
    // Tempo bonus
    score += 28;
    
    // Return from side to move's perspective
    return pos.side_to_move() == WHITE ? Value(score) : Value(-score);
}

void trace(const Position& pos) {
    std::cout << "\n" << pos.to_string() << "\n";
    
    int mgWhite = 0, egWhite = 0;
    int mgBlack = 0, egBlack = 0;
    
    evaluate_side<WHITE>(pos, mgWhite, egWhite);
    evaluate_side<BLACK>(pos, mgBlack, egBlack);
    
    std::cout << std::setw(20) << "Term" << " | " 
              << std::setw(10) << "White MG" << " | "
              << std::setw(10) << "White EG" << " | "
              << std::setw(10) << "Black MG" << " | "
              << std::setw(10) << "Black EG" << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::setw(20) << "Material+PST" << " | "
              << std::setw(10) << mgWhite << " | "
              << std::setw(10) << egWhite << " | "
              << std::setw(10) << mgBlack << " | "
              << std::setw(10) << egBlack << "\n";
    
    int mgScore = mgWhite - mgBlack;
    int egScore = egWhite - egBlack;
    int phase = game_phase(pos);
    int score = (mgScore * phase + egScore * (TotalPhase - phase)) / TotalPhase;
    
    std::cout << "\nPhase: " << phase << "/" << TotalPhase << "\n";
    std::cout << "MG Score: " << mgScore << "\n";
    std::cout << "EG Score: " << egScore << "\n";
    std::cout << "Final (white POV): " << score << "\n";
    std::cout << "Final (STM POV): " << (pos.side_to_move() == WHITE ? score : -score) << "\n";
}

} // namespace Eval
