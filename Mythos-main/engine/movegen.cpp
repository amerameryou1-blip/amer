#include "movegen.h"
#include "bitboard.h"
#include <iostream>

namespace {

// =============================================================================
// PAWN MOVE GENERATION
// =============================================================================

template<Color Us, GenType GT>
void generate_pawn_moves(const Position& pos, MoveList& moves, Bitboard target) {
    constexpr Color Them = ~Us;
    constexpr Bitboard TRank7BB = Us == WHITE ? Rank7BB : Rank2BB;
    constexpr Bitboard TRank3BB = Us == WHITE ? Rank3BB : Rank6BB;
    constexpr Direction Up = Us == WHITE ? NORTH : SOUTH;
    constexpr Direction UpRight = Us == WHITE ? NORTH_EAST : SOUTH_WEST;
    constexpr Direction UpLeft = Us == WHITE ? NORTH_WEST : SOUTH_EAST;
    
    Bitboard pawnsOn7 = pos.pieces(Us, PAWN) & TRank7BB;
    Bitboard pawnsNotOn7 = pos.pieces(Us, PAWN) & ~TRank7BB;
    
    Bitboard enemies = GT == EVASIONS ? pos.checkers() : pos.pieces(Them);
    Bitboard emptySquares = ~pos.pieces();
    
    // Single and double pawn pushes (non-capture)
    if constexpr (GT != CAPTURES) {
        Bitboard b1 = shift<Up>(pawnsNotOn7) & emptySquares;
        Bitboard b2 = shift<Up>(b1 & TRank3BB) & emptySquares;
        
        if constexpr (GT == EVASIONS) {
            b1 &= target;
            b2 &= target;
        }
        
        while (b1) {
            Square to = pop_lsb(b1);
            moves.add(Move(to - Up, to));
        }
        
        while (b2) {
            Square to = pop_lsb(b2);
            moves.add(Move(to - Up - Up, to));
        }
    }
    
    // Promotions
    if (pawnsOn7) {
        Bitboard b1 = shift<UpRight>(pawnsOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsOn7) & enemies;
        Bitboard b3 = shift<Up>(pawnsOn7) & emptySquares;
        
        if constexpr (GT == EVASIONS) {
            b3 &= target;
        }
        
        while (b1) {
            Square to = pop_lsb(b1);
            Square from = to - UpRight;
            moves.add(Move(from, to, PROMOTION, QUEEN));
            moves.add(Move(from, to, PROMOTION, ROOK));
            moves.add(Move(from, to, PROMOTION, BISHOP));
            moves.add(Move(from, to, PROMOTION, KNIGHT));
        }
        
        while (b2) {
            Square to = pop_lsb(b2);
            Square from = to - UpLeft;
            moves.add(Move(from, to, PROMOTION, QUEEN));
            moves.add(Move(from, to, PROMOTION, ROOK));
            moves.add(Move(from, to, PROMOTION, BISHOP));
            moves.add(Move(from, to, PROMOTION, KNIGHT));
        }
        
        while (b3) {
            Square to = pop_lsb(b3);
            Square from = to - Up;
            moves.add(Move(from, to, PROMOTION, QUEEN));
            moves.add(Move(from, to, PROMOTION, ROOK));
            moves.add(Move(from, to, PROMOTION, BISHOP));
            moves.add(Move(from, to, PROMOTION, KNIGHT));
        }
    }
    
    // Captures (non-promotion)
    if constexpr (GT == CAPTURES || GT == EVASIONS || GT == NON_EVASIONS || GT == LEGAL) {
        Bitboard b1 = shift<UpRight>(pawnsNotOn7) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsNotOn7) & enemies;
        
        while (b1) {
            Square to = pop_lsb(b1);
            moves.add(Move(to - UpRight, to));
        }
        
        while (b2) {
            Square to = pop_lsb(b2);
            moves.add(Move(to - UpLeft, to));
        }
    }
    
    // En passant
    if (pos.ep_square() != SQ_NONE) {
        Square epSq = pos.ep_square();
        
        if constexpr (GT == EVASIONS) {
            // Only consider ep if it can resolve the check
            if (!(target & (epSq - Up))) return;
        }
        
        Bitboard b = pawn_attacks_bb(Them, epSq) & pawnsNotOn7;
        
        while (b) {
            Square from = pop_lsb(b);
            moves.add(Move(from, epSq, EN_PASSANT));
        }
    }
}

// =============================================================================
// PIECE MOVE GENERATION
// =============================================================================

template<PieceType Pt>
void generate_piece_moves(const Position& pos, MoveList& moves, Color us, Bitboard target) {
    static_assert(Pt != PAWN && Pt != KING, "Use specialized generators");
    
    Bitboard bb = pos.pieces(us, Pt);
    
    while (bb) {
        Square from = pop_lsb(bb);
        Bitboard attacks = attacks_bb<Pt>(from, pos.pieces()) & target;
        
        while (attacks) {
            Square to = pop_lsb(attacks);
            moves.add(Move(from, to));
        }
    }
}

// =============================================================================
// CASTLING GENERATION
// =============================================================================

template<Color Us, CastlingRights Cr>
void generate_castling(const Position& pos, MoveList& moves) {
    constexpr bool KingSide = (Cr == WHITE_OO || Cr == BLACK_OO);
    constexpr Square kFrom = Us == WHITE ? SQ_E1 : SQ_E8;
    constexpr Square kTo = KingSide ? (Us == WHITE ? SQ_G1 : SQ_G8)
                                    : (Us == WHITE ? SQ_C1 : SQ_C8);
    
    if (!pos.can_castle(Cr)) return;
    if (pos.castling_impeded(Cr)) return;
    
    // Check that king doesn't pass through or land in check
    // Note: "in check" is already handled before calling this
    Square rook = pos.castling_rook_square(Cr);
    
    // Check squares between king and target
    Direction step = kTo > kFrom ? EAST : WEST;
    for (Square s = kFrom + step; s != kTo + step; s += step) {
        if (pos.attackers_to(s) & pos.pieces(~Us)) return;
    }
    
    moves.add(Move(kFrom, kTo, CASTLING));
}

template<Color Us>
void generate_all_castling(const Position& pos, MoveList& moves) {
    if (pos.is_check()) return;
    
    if constexpr (Us == WHITE) {
        generate_castling<WHITE, WHITE_OO>(pos, moves);
        generate_castling<WHITE, WHITE_OOO>(pos, moves);
    } else {
        generate_castling<BLACK, BLACK_OO>(pos, moves);
        generate_castling<BLACK, BLACK_OOO>(pos, moves);
    }
}

// =============================================================================
// KING MOVE GENERATION  
// =============================================================================

template<Color Us>
void generate_king_moves(const Position& pos, MoveList& moves, Bitboard target) {
    Square ksq = pos.king_square(Us);
    Bitboard attacks = attacks_bb<KING>(ksq, 0) & target;
    
    // For king moves, we need to check that the destination isn't attacked
    while (attacks) {
        Square to = pop_lsb(attacks);
        // Check if square is safe
        if (!(pos.attackers_to(to, pos.pieces() ^ ksq) & pos.pieces(~Us))) {
            moves.add(Move(ksq, to));
        }
    }
}

// =============================================================================
// EVASION GENERATION
// =============================================================================

template<Color Us>
void generate_evasions(const Position& pos, MoveList& moves) {
    Square ksq = pos.king_square(Us);
    Bitboard checkers = pos.checkers();
    
    // King moves are always possible
    generate_king_moves<Us>(pos, moves, ~pos.pieces(Us));
    
    // If double check, only king moves are legal
    if (more_than_one(checkers)) return;
    
    // Single checker - can block or capture
    Square checker = lsb(checkers);
    Bitboard target = between_bb(ksq, checker) | checker;
    
    // Generate non-king moves that block or capture
    generate_pawn_moves<Us, EVASIONS>(pos, moves, target);
    generate_piece_moves<KNIGHT>(pos, moves, Us, target);
    generate_piece_moves<BISHOP>(pos, moves, Us, target);
    generate_piece_moves<ROOK>(pos, moves, Us, target);
    generate_piece_moves<QUEEN>(pos, moves, Us, target);
}

// =============================================================================
// LEGALITY FILTERING
// =============================================================================

void filter_legal(const Position& pos, MoveList& moves) {
    Color us = pos.side_to_move();
    Square ksq = pos.king_square(us);
    Bitboard pinned = pos.blockers_for_king(us);
    
    int writeIdx = 0;
    for (int readIdx = 0; readIdx < moves.size(); readIdx++) {
        Move m = moves[readIdx].move;
        
        // King moves already filtered
        if (m.from() == ksq) {
            moves[writeIdx++] = moves[readIdx];
            continue;
        }
        
        // En passant needs special handling
        if (m.type() == EN_PASSANT) {
            if (pos.legal(m)) {
                moves[writeIdx++] = moves[readIdx];
            }
            continue;
        }
        
        // Non-pinned pieces are always legal
        if (!(pinned & m.from())) {
            moves[writeIdx++] = moves[readIdx];
            continue;
        }
        
        // Pinned pieces can only move along the pin line
        if (aligned(m.from(), m.to(), ksq)) {
            moves[writeIdx++] = moves[readIdx];
        }
    }
    moves.count = writeIdx;
}

} // anonymous namespace

// =============================================================================
// MAIN GENERATION FUNCTIONS
// =============================================================================

template<GenType GT>
void generate(const Position& pos, MoveList& moves) {
    static_assert(GT == CAPTURES || GT == QUIETS || GT == NON_EVASIONS || 
                  GT == EVASIONS || GT == LEGAL, "Unsupported GenType");
    
    moves.clear();
    
    Color us = pos.side_to_move();
    
    // Evasions
    if constexpr (GT == EVASIONS) {
        if (us == WHITE)
            generate_evasions<WHITE>(pos, moves);
        else
            generate_evasions<BLACK>(pos, moves);
        return;
    }
    
    // If in check, generate evasions
    if (pos.is_check()) {
        if (us == WHITE)
            generate_evasions<WHITE>(pos, moves);
        else
            generate_evasions<BLACK>(pos, moves);
        
        if constexpr (GT == LEGAL) {
            filter_legal(pos, moves);
        }
        return;
    }
    
    Bitboard target;
    
    if constexpr (GT == CAPTURES) {
        target = pos.pieces(~us);
    } else if constexpr (GT == QUIETS) {
        target = ~pos.pieces();
    } else {
        target = ~pos.pieces(us);  // NON_EVASIONS or LEGAL
    }
    
    // Pawn moves
    if (us == WHITE) {
        generate_pawn_moves<WHITE, GT>(pos, moves, target);
    } else {
        generate_pawn_moves<BLACK, GT>(pos, moves, target);
    }
    
    // Piece moves
    generate_piece_moves<KNIGHT>(pos, moves, us, target);
    generate_piece_moves<BISHOP>(pos, moves, us, target);
    generate_piece_moves<ROOK>(pos, moves, us, target);
    generate_piece_moves<QUEEN>(pos, moves, us, target);
    
    // King moves (not already handled in evasions)
    if (us == WHITE) {
        generate_king_moves<WHITE>(pos, moves, target);
    } else {
        generate_king_moves<BLACK>(pos, moves, target);
    }
    
    // Castling (only for quiets or non-evasions/legal)
    if constexpr (GT != CAPTURES) {
        if (us == WHITE) {
            generate_all_castling<WHITE>(pos, moves);
        } else {
            generate_all_castling<BLACK>(pos, moves);
        }
    }
    
    // Filter for full legality if needed
    if constexpr (GT == LEGAL) {
        filter_legal(pos, moves);
    }
}

// Explicit instantiations
template void generate<CAPTURES>(const Position& pos, MoveList& moves);
template void generate<QUIETS>(const Position& pos, MoveList& moves);
template void generate<NON_EVASIONS>(const Position& pos, MoveList& moves);
template void generate<EVASIONS>(const Position& pos, MoveList& moves);
template void generate<LEGAL>(const Position& pos, MoveList& moves);

// =============================================================================
// PERFT
// =============================================================================

uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;
    
    MoveList moves;
    generate<LEGAL>(pos, moves);
    
    if (depth == 1) return moves.size();
    
    uint64_t nodes = 0;
    StateInfo st;
    
    for (const auto& sm : moves) {
        pos.do_move(sm.move, st);
        nodes += perft(pos, depth - 1);
        pos.undo_move(sm.move);
    }
    
    return nodes;
}

void perft_divide(Position& pos, int depth) {
    MoveList moves;
    generate<LEGAL>(pos, moves);
    
    uint64_t totalNodes = 0;
    StateInfo st;
    
    for (const auto& sm : moves) {
        pos.do_move(sm.move, st);
        uint64_t nodes = perft(pos, depth - 1);
        pos.undo_move(sm.move);
        
        std::cout << sm.move.to_uci() << ": " << nodes << std::endl;
        totalNodes += nodes;
    }
    
    std::cout << "\nTotal: " << totalNodes << std::endl;
}
