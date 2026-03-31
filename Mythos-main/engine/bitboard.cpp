#include "bitboard.h"
#include <iostream>
#include <sstream>
#include <cstring>

// =============================================================================
// GLOBAL TABLES
// =============================================================================

Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];
Bitboard KnightAttacks[SQUARE_NB];
Bitboard KingAttacks[SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
int SquareDistance[SQUARE_NB][SQUARE_NB];

Magic RookMagics[SQUARE_NB];
Magic BishopMagics[SQUARE_NB];
Bitboard RookTable[0x19000];
Bitboard BishopTable[0x1480];

// =============================================================================
// MAGIC NUMBERS (pre-computed)
// =============================================================================

namespace {

constexpr Bitboard RookMagicNumbers[SQUARE_NB] = {
    0x8a80104000800020ULL, 0x140002000100040ULL,  0x2801880a0017001ULL,  0x100081001000420ULL,
    0x200020010080420ULL,  0x3001c0002010008ULL,  0x8480008002000100ULL, 0x2080088004402900ULL,
    0x800098204000ULL,     0x2024401000200040ULL, 0x100802000801000ULL,  0x120800800801000ULL,
    0x208808088000400ULL,  0x2802200800400ULL,    0x2200800100020080ULL, 0x801000060821100ULL,
    0x80044006422000ULL,   0x100808020004000ULL,  0x12108a0010204200ULL, 0x140848010000802ULL,
    0x481828014002800ULL,  0x8094004002004100ULL, 0x4010040010010802ULL, 0x20008806104ULL,
    0x100400080208000ULL,  0x2040002120081000ULL, 0x21200680100081ULL,   0x20100080080080ULL,
    0x2000a00200410ULL,    0x20080800400ULL,      0x80088400100102ULL,   0x80004600042881ULL,
    0x4040008040800020ULL, 0x440003000200801ULL,  0x4200011004500ULL,    0x188020010100100ULL,
    0x14800401802800ULL,   0x2080040080800200ULL, 0x124080204001001ULL,  0x200046502000484ULL,
    0x480400080088020ULL,  0x1000422010034000ULL, 0x30200100110040ULL,   0x100021010009ULL,
    0x2002080100110004ULL, 0x202008004008002ULL,  0x20020004010100ULL,   0x2048440040820001ULL,
    0x101002200408200ULL,  0x40802000401080ULL,   0x4008142004410100ULL, 0x2060820c0120200ULL,
    0x1001004080100ULL,    0x20c020080040080ULL,  0x2935610830022400ULL, 0x44440041009200ULL,
    0x280001040802101ULL,  0x2100190040002085ULL, 0x80c0084100102001ULL, 0x4024081001000421ULL,
    0x20030a0244872ULL,    0x12001008414402ULL,   0x2006104900a0804ULL,  0x1004081002402ULL
};

constexpr Bitboard BishopMagicNumbers[SQUARE_NB] = {
    0x40040844404084ULL,   0x2004208a004208ULL,   0x10190041080202ULL,   0x108060845042010ULL,
    0x581104180800210ULL,  0x2112080446200010ULL, 0x1080820820060210ULL, 0x3c0808410220200ULL,
    0x4050404440404ULL,    0x21001420088ULL,      0x24d0080801082102ULL, 0x1020a0a020400ULL,
    0x40308200402ULL,      0x4011002100800ULL,    0x401484104104005ULL,  0x801010402020200ULL,
    0x400210c3880100ULL,   0x404022024108200ULL,  0x810018200204102ULL,  0x4002801a02003ULL,
    0x85040820080400ULL,   0x810102c808880400ULL, 0xe900410884800ULL,    0x8002020480840102ULL,
    0x220200865090201ULL,  0x2010100a02021202ULL, 0x152048408022401ULL,  0x20080002081110ULL,
    0x4001001021004000ULL, 0x800040400a011002ULL, 0xe4004081011002ULL,   0x1c004001012080ULL,
    0x8004200962a00220ULL, 0x8422100208500202ULL, 0x2000402200300c08ULL, 0x8646020080080080ULL,
    0x80020a0200100808ULL, 0x2010004880111000ULL, 0x623000a080011400ULL, 0x42008c0340209202ULL,
    0x209188240001000ULL,  0x400408a884001800ULL, 0x110400a6080400ULL,   0x1840060a44020800ULL,
    0x90080104000041ULL,   0x201011000808101ULL,  0x1a2208080504f080ULL, 0x8012020600211212ULL,
    0x500861011240000ULL,  0x180806108200800ULL,  0x4000020e01040044ULL, 0x300000261044000aULL,
    0x802241102020002ULL,  0x20906061210001ULL,   0x5a84841004010310ULL, 0x4010801011c04ULL,
    0xa010109502200ULL,    0x4a02012000ULL,       0x500201010098b028ULL, 0x8040002811040900ULL,
    0x28000010020204ULL,   0x6000020202d0240ULL,  0x8918844842082200ULL, 0x4010011029020020ULL
};

// Sliding attacks without blockers
Bitboard sliding_attack(PieceType pt, Square sq, Bitboard occupied) {
    Bitboard attacks = 0;
    Direction directions[4];
    
    if (pt == ROOK) {
        directions[0] = NORTH; directions[1] = SOUTH;
        directions[2] = EAST;  directions[3] = WEST;
    } else {
        directions[0] = NORTH_EAST; directions[1] = NORTH_WEST;
        directions[2] = SOUTH_EAST; directions[3] = SOUTH_WEST;
    }
    
    for (Direction d : directions) {
        Square s = sq;
        while (true) {
            s = s + d;
            if (!is_ok(s)) break;
            
            // Check wrap-around
            int fileDiff = std::abs(file_of(s) - file_of(Square(s - d)));
            int rankDiff = std::abs(rank_of(s) - rank_of(Square(s - d)));
            if (fileDiff > 1 || rankDiff > 1) break;
            
            attacks |= s;
            if (occupied & s) break;
        }
    }
    return attacks;
}

// Create blocker mask for rook (excludes edges unless on edge)
Bitboard rook_mask(Square s) {
    Bitboard result = 0;
    int rk = rank_of(s), fl = file_of(s);
    
    for (int r = rk + 1; r <= 6; r++) result |= 1ULL << (fl + r * 8);
    for (int r = rk - 1; r >= 1; r--) result |= 1ULL << (fl + r * 8);
    for (int f = fl + 1; f <= 6; f++) result |= 1ULL << (f + rk * 8);
    for (int f = fl - 1; f >= 1; f--) result |= 1ULL << (f + rk * 8);
    
    return result;
}

// Create blocker mask for bishop (excludes edges)
Bitboard bishop_mask(Square s) {
    Bitboard result = 0;
    int rk = rank_of(s), fl = file_of(s);
    
    for (int r = rk + 1, f = fl + 1; r <= 6 && f <= 6; r++, f++) result |= 1ULL << (f + r * 8);
    for (int r = rk + 1, f = fl - 1; r <= 6 && f >= 1; r++, f--) result |= 1ULL << (f + r * 8);
    for (int r = rk - 1, f = fl + 1; r >= 1 && f <= 6; r--, f++) result |= 1ULL << (f + r * 8);
    for (int r = rk - 1, f = fl - 1; r >= 1 && f >= 1; r--, f--) result |= 1ULL << (f + r * 8);
    
    return result;
}

// Initialize magic bitboard for a single square
Bitboard* init_magics(Square s, Bitboard* table, Magic& magic, Bitboard magicNum, PieceType pt) {
    Bitboard mask = pt == ROOK ? rook_mask(s) : bishop_mask(s);
    int bits = popcount(mask);
    int size = 1 << bits;
    
    magic.mask = mask;
    magic.magic = magicNum;
    magic.attacks = table;
    magic.shift = 64 - bits;
    
    // Enumerate all occupancy permutations
    Bitboard occupied = 0;
    for (int i = 0; i < size; i++) {
        unsigned idx = magic.index(occupied);
        magic.attacks[idx] = sliding_attack(pt, s, occupied);
        
        // Carry-Rippler trick to enumerate subsets
        occupied = (occupied - mask) & mask;
    }
    
    return table + size;
}

} // anonymous namespace

// =============================================================================
// INITIALIZATION
// =============================================================================

namespace Bitboards {

void init() {
    // Initialize distance table
    for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1) {
        for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2) {
            SquareDistance[s1][s2] = std::max(distance(file_of(s1), file_of(s2)),
                                               distance(rank_of(s1), rank_of(s2)));
        }
    }
    
    // Initialize pawn attacks
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        Bitboard b = square_bb(s);
        PawnAttacks[WHITE][s] = shift<NORTH_WEST>(b) | shift<NORTH_EAST>(b);
        PawnAttacks[BLACK][s] = shift<SOUTH_WEST>(b) | shift<SOUTH_EAST>(b);
    }
    
    // Initialize knight attacks
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        Bitboard b = square_bb(s);
        KnightAttacks[s] = 0;
        
        // 8 knight moves
        if (file_of(s) >= FILE_B && rank_of(s) <= RANK_6) KnightAttacks[s] |= b << 15;  // NNW
        if (file_of(s) <= FILE_G && rank_of(s) <= RANK_6) KnightAttacks[s] |= b << 17;  // NNE
        if (file_of(s) <= FILE_F && rank_of(s) <= RANK_7) KnightAttacks[s] |= b << 10;  // NEE
        if (file_of(s) <= FILE_F && rank_of(s) >= RANK_2) KnightAttacks[s] |= b >> 6;   // SEE
        if (file_of(s) <= FILE_G && rank_of(s) >= RANK_3) KnightAttacks[s] |= b >> 15;  // SSE
        if (file_of(s) >= FILE_B && rank_of(s) >= RANK_3) KnightAttacks[s] |= b >> 17;  // SSW
        if (file_of(s) >= FILE_C && rank_of(s) >= RANK_2) KnightAttacks[s] |= b >> 10;  // SWW
        if (file_of(s) >= FILE_C && rank_of(s) <= RANK_7) KnightAttacks[s] |= b << 6;   // NWW
        
        PseudoAttacks[KNIGHT][s] = KnightAttacks[s];
    }
    
    // Initialize king attacks
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        Bitboard b = square_bb(s);
        KingAttacks[s] = 0;
        
        KingAttacks[s] |= shift<NORTH>(b);
        KingAttacks[s] |= shift<SOUTH>(b);
        KingAttacks[s] |= shift<EAST>(b);
        KingAttacks[s] |= shift<WEST>(b);
        KingAttacks[s] |= shift<NORTH_EAST>(b);
        KingAttacks[s] |= shift<NORTH_WEST>(b);
        KingAttacks[s] |= shift<SOUTH_EAST>(b);
        KingAttacks[s] |= shift<SOUTH_WEST>(b);
        
        PseudoAttacks[KING][s] = KingAttacks[s];
    }
    
    // Initialize magic bitboards
    Bitboard* rookTable = RookTable;
    Bitboard* bishopTable = BishopTable;
    
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        rookTable = init_magics(s, rookTable, RookMagics[s], RookMagicNumbers[s], ROOK);
    }
    
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        bishopTable = init_magics(s, bishopTable, BishopMagics[s], BishopMagicNumbers[s], BISHOP);
    }
    
    // Initialize pseudo-attacks for sliders (without blockers)
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        PseudoAttacks[BISHOP][s] = attacks_bb<BISHOP>(s, 0);
        PseudoAttacks[ROOK][s]   = attacks_bb<ROOK>(s, 0);
        PseudoAttacks[QUEEN][s]  = PseudoAttacks[BISHOP][s] | PseudoAttacks[ROOK][s];
    }
    
    // Initialize between and line bitboards
    for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1) {
        for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2) {
            BetweenBB[s1][s2] = 0;
            LineBB[s1][s2] = 0;
            
            if (s1 == s2) continue;
            
            // Check if on same line (rook or bishop)
            if (PseudoAttacks[ROOK][s1] & s2) {
                LineBB[s1][s2] = (attacks_bb<ROOK>(s1, 0) & attacks_bb<ROOK>(s2, 0)) | s1 | s2;
                BetweenBB[s1][s2] = attacks_bb<ROOK>(s1, square_bb(s2)) & 
                                   attacks_bb<ROOK>(s2, square_bb(s1));
            } else if (PseudoAttacks[BISHOP][s1] & s2) {
                LineBB[s1][s2] = (attacks_bb<BISHOP>(s1, 0) & attacks_bb<BISHOP>(s2, 0)) | s1 | s2;
                BetweenBB[s1][s2] = attacks_bb<BISHOP>(s1, square_bb(s2)) & 
                                    attacks_bb<BISHOP>(s2, square_bb(s1));
            }
        }
    }
}

std::string pretty(Bitboard b) {
    std::ostringstream ss;
    ss << "+---+---+---+---+---+---+---+---+\n";
    for (Rank r = RANK_8; r >= RANK_1; --r) {
        for (File f = FILE_A; f <= FILE_H; ++f) {
            ss << "| " << ((b & make_square(f, r)) ? "X " : ". ");
        }
        ss << "| " << (1 + r) << "\n+---+---+---+---+---+---+---+---+\n";
    }
    ss << "  a   b   c   d   e   f   g   h\n";
    return ss.str();
}

} // namespace Bitboards
