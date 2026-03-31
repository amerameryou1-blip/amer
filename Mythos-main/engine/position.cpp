#include "position.h"
#include <sstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <iomanip>

// =============================================================================
// ZOBRIST KEYS
// =============================================================================

namespace Zobrist {
    Key psq[PIECE_NB][SQUARE_NB];
    Key enpassant[FILE_NB];
    Key castling[CASTLING_RIGHT_NB];
    Key side;
    Key noPawns;

    void init() {
        // Use a deterministic PRNG for reproducible keys
        uint64_t seed = 1070372;
        auto rand64 = [&seed]() {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            return seed * 0x2545F4914F6CDD1DULL;
        };

        for (Piece pc = W_PAWN; pc <= B_KING; ++pc) {
            for (Square s = SQ_A1; s <= SQ_H8; ++s) {
                psq[pc][s] = rand64();
            }
        }

        for (File f = FILE_A; f <= FILE_H; ++f) {
            enpassant[f] = rand64();
        }

        for (int cr = 0; cr < CASTLING_RIGHT_NB; ++cr) {
            castling[cr] = 0;
            if (cr & WHITE_OO)  castling[cr] ^= rand64();
            if (cr & WHITE_OOO) castling[cr] ^= rand64();
            if (cr & BLACK_OO)  castling[cr] ^= rand64();
            if (cr & BLACK_OOO) castling[cr] ^= rand64();
        }

        side = rand64();
        noPawns = rand64();
    }
}

// =============================================================================
// PIECE CHARACTERS
// =============================================================================

namespace {
    constexpr const char* PieceToChar = " PNBRQK  pnbrqk";
    
    Piece char_to_piece(char c) {
        const char* ptr = strchr(PieceToChar, c);
        return ptr ? Piece(ptr - PieceToChar) : NO_PIECE;
    }
}

// =============================================================================
// INITIALIZATION
// =============================================================================

void Position::init() {
    Zobrist::init();
}

// =============================================================================
// FEN PARSING
// =============================================================================

Position& Position::set(const std::string& fenStr, StateInfo* si) {
    // Clear everything
    std::memset(this, 0, sizeof(Position));
    std::fill_n(&pieceCount[0], PIECE_NB, 0);
    st = si;
    
    std::istringstream ss(fenStr);
    std::string token;
    
    ss >> std::noskipws;
    
    // 1. Piece placement
    Square sq = SQ_A8;
    char c;
    while ((ss >> c) && !isspace(c)) {
        if (isdigit(c)) {
            sq += Direction(c - '0') * EAST;
        } else if (c == '/') {
            sq += 2 * SOUTH;
        } else {
            Piece pc = char_to_piece(c);
            if (pc != NO_PIECE) {
                put_piece(pc, sq);
                ++sq;
            }
        }
    }
    
    // 2. Active color
    ss >> c;
    sideToMove = (c == 'w') ? WHITE : BLACK;
    ss >> c;
    
    // 3. Castling rights
    st->castlingRights = NO_CASTLING;
    while ((ss >> c) && !isspace(c)) {
        if (c == '-') continue;
        
        Color color = isupper(c) ? WHITE : BLACK;
        Rank backRank = color == WHITE ? RANK_1 : RANK_8;
        c = char(toupper(c));
        
        Square kingSq = king_square(color);
        Square rookSq = SQ_NONE;
        
        if (c == 'K') {
            // King-side castling: find rook on H-file side of king
            for (File f = FILE_H; f >= file_of(kingSq); --f) {
                Square s = make_square(f, backRank);
                if (piece_on(s) == make_piece(color, ROOK)) {
                    rookSq = s;
                    break;
                }
            }
        } else if (c == 'Q') {
            // Queen-side castling: find rook on A-file side of king
            for (File f = FILE_A; f <= file_of(kingSq); ++f) {
                Square s = make_square(f, backRank);
                if (piece_on(s) == make_piece(color, ROOK)) {
                    rookSq = s;
                    break;
                }
            }
        } else if (c >= 'A' && c <= 'H') {
            // Chess960: file letter
            rookSq = make_square(File(c - 'A'), backRank);
        }
        
        if (rookSq != SQ_NONE) {
            set_castling_right(color, rookSq);
        }
    }
    
    // 4. En passant square
    st->epSquare = SQ_NONE;
    char col, row;
    if ((ss >> col) && col != '-') {
        if ((ss >> row) && col >= 'a' && col <= 'h' && (row == '3' || row == '6')) {
            Square epSq = make_square(File(col - 'a'), Rank(row - '1'));
            
            // Only set if there's a pawn that can capture
            Color them = ~sideToMove;
            Bitboard potentialCapturers = pawn_attacks_bb(them, epSq) & pieces(sideToMove, PAWN);
            
            if (potentialCapturers) {
                st->epSquare = epSq;
            }
        }
    }
    ss >> c;
    
    // 5. Half-move clock
    ss >> std::skipws >> st->rule50;
    
    // 6. Full-move number
    int fullMoveNum;
    ss >> fullMoveNum;
    gamePly = std::max(2 * (fullMoveNum - 1), 0) + (sideToMove == BLACK);
    
    // Set remaining state
    st->pliesFromNull = 0;
    st->previous = nullptr;
    st->capturedPiece = NO_PIECE;
    st->repetition = 0;
    
    set_state(st);
    
    return *this;
}

Position& Position::set(const Position& pos, StateInfo* si) {
    std::memcpy(this, &pos, sizeof(Position));
    st = si;
    *st = *pos.st;
    st->previous = nullptr;
    return *this;
}

// =============================================================================
// SET CASTLING RIGHT
// =============================================================================

void Position::set_castling_right(Color c, Square rfrom) {
    Square kfrom = king_square(c);
    CastlingRights cr = ::castling_rights(c, rfrom > kfrom);
    
    st->castlingRights |= cr;
    castlingRightsMask[kfrom] |= cr;
    castlingRightsMask[rfrom] |= cr;
    castlingRookSquare[cr] = rfrom;
    
    Square kto = make_square(rfrom > kfrom ? FILE_G : FILE_C, relative_rank(c, RANK_1));
    Square rto = make_square(rfrom > kfrom ? FILE_F : FILE_D, relative_rank(c, RANK_1));
    
    castlingPath[cr] = (between_bb(rfrom, rto) | between_bb(kfrom, kto) | rto | kto)
                     & ~(square_bb(kfrom) | rfrom);
}

// =============================================================================
// SET STATE (key, checkers, etc)
// =============================================================================

void Position::set_state(StateInfo* si) const {
    si->key = 0;
    si->pawnKey = Zobrist::noPawns;
    si->materialKey = 0;
    si->nonPawnMaterial[WHITE] = VALUE_ZERO;
    si->nonPawnMaterial[BLACK] = VALUE_ZERO;
    
    si->checkersBB = attackers_to(king_square(sideToMove)) & pieces(~sideToMove);
    
    set_check_info(si);
    
    for (Bitboard b = pieces(); b; ) {
        Square s = pop_lsb(b);
        Piece pc = piece_on(s);
        si->key ^= Zobrist::psq[pc][s];
        
        if (type_of(pc) == PAWN) {
            si->pawnKey ^= Zobrist::psq[pc][s];
        } else if (type_of(pc) != KING) {
            si->nonPawnMaterial[color_of(pc)] += PieceValue[type_of(pc)];
        }
    }
    
    if (si->epSquare != SQ_NONE) {
        si->key ^= Zobrist::enpassant[file_of(si->epSquare)];
    }
    
    if (sideToMove == BLACK) {
        si->key ^= Zobrist::side;
    }
    
    si->key ^= Zobrist::castling[si->castlingRights];
}

// =============================================================================
// SET CHECK INFO
// =============================================================================

void Position::set_check_info(StateInfo* si) const {
    si->blockersForKing[WHITE] = slider_blockers(pieces(BLACK), king_square(WHITE), si->pinners[BLACK]);
    si->blockersForKing[BLACK] = slider_blockers(pieces(WHITE), king_square(BLACK), si->pinners[WHITE]);
    
    Square ksq = king_square(~sideToMove);
    
    si->checkSquares[PAWN]   = pawn_attacks_bb(~sideToMove, ksq);
    si->checkSquares[KNIGHT] = attacks_bb<KNIGHT>(ksq, 0);
    si->checkSquares[BISHOP] = attacks_bb<BISHOP>(ksq, pieces());
    si->checkSquares[ROOK]   = attacks_bb<ROOK>(ksq, pieces());
    si->checkSquares[QUEEN]  = si->checkSquares[BISHOP] | si->checkSquares[ROOK];
    si->checkSquares[KING]   = 0;
}

// =============================================================================
// FEN OUTPUT
// =============================================================================

std::string Position::fen() const {
    std::ostringstream ss;
    
    // Piece placement
    for (Rank r = RANK_8; r >= RANK_1; --r) {
        int emptyCount = 0;
        for (File f = FILE_A; f <= FILE_H; ++f) {
            Piece pc = piece_on(make_square(f, r));
            if (pc == NO_PIECE) {
                emptyCount++;
            } else {
                if (emptyCount > 0) {
                    ss << emptyCount;
                    emptyCount = 0;
                }
                ss << PieceToChar[pc];
            }
        }
        if (emptyCount > 0) ss << emptyCount;
        if (r > RANK_1) ss << '/';
    }
    
    // Active color
    ss << (sideToMove == WHITE ? " w " : " b ");
    
    // Castling rights
    if (can_castle(ANY_CASTLING)) {
        if (can_castle(WHITE_OO))  ss << 'K';
        if (can_castle(WHITE_OOO)) ss << 'Q';
        if (can_castle(BLACK_OO))  ss << 'k';
        if (can_castle(BLACK_OOO)) ss << 'q';
    } else {
        ss << '-';
    }
    
    // En passant
    ss << ' ';
    if (st->epSquare != SQ_NONE) {
        ss << char('a' + file_of(st->epSquare)) << char('1' + rank_of(st->epSquare));
    } else {
        ss << '-';
    }
    
    // Half-move clock and full-move number
    ss << ' ' << st->rule50 << ' ' << 1 + (gamePly - (sideToMove == BLACK)) / 2;
    
    return ss.str();
}

// =============================================================================
// ATTACKERS
// =============================================================================

Bitboard Position::attackers_to(Square s) const {
    return attackers_to(s, pieces());
}

Bitboard Position::attackers_to(Square s, Bitboard occupied) const {
    return (pawn_attacks_bb(BLACK, s) & pieces(WHITE, PAWN))
         | (pawn_attacks_bb(WHITE, s) & pieces(BLACK, PAWN))
         | (attacks_bb<KNIGHT>(s, occupied) & pieces(KNIGHT))
         | (attacks_bb<ROOK>(s, occupied) & pieces(ROOK, QUEEN))
         | (attacks_bb<BISHOP>(s, occupied) & pieces(BISHOP, QUEEN))
         | (attacks_bb<KING>(s, occupied) & pieces(KING));
}

// =============================================================================
// SLIDER BLOCKERS (for pins)
// =============================================================================

Bitboard Position::slider_blockers(Bitboard sliders, Square s, Bitboard& pinners) const {
    Bitboard blockers = 0;
    pinners = 0;
    
    Bitboard snipers = ((attacks_bb<ROOK>(s, 0) & pieces(ROOK, QUEEN))
                      | (attacks_bb<BISHOP>(s, 0) & pieces(BISHOP, QUEEN))) & sliders;
    Bitboard occupied = pieces() ^ snipers;
    
    while (snipers) {
        Square sniperSq = pop_lsb(snipers);
        Bitboard b = between_bb(s, sniperSq) & occupied;
        
        if (b && !more_than_one(b)) {
            blockers |= b;
            if (b & pieces(color_of(piece_on(s)))) {
                pinners |= sniperSq;
            }
        }
    }
    return blockers;
}

// =============================================================================
// GIVES CHECK
// =============================================================================

bool Position::gives_check(Move m) const {
    assert(m);
    
    Square from = m.from();
    Square to = m.to();
    
    // Direct check
    if (st->checkSquares[type_of(piece_on(from))] & to) {
        return true;
    }
    
    // Discovered check
    if ((blockers_for_king(~sideToMove) & from) &&
        !aligned(from, to, king_square(~sideToMove))) {
        return true;
    }
    
    switch (m.type()) {
        case NORMAL:
            return false;
            
        case PROMOTION:
            return attacks_bb(m.promotion_type(), to, pieces() ^ from) & king_square(~sideToMove);
            
        case EN_PASSANT: {
            Square capsq = make_square(file_of(to), rank_of(from));
            Bitboard occupied = (pieces() ^ from ^ capsq) | to;
            return (attacks_bb<ROOK>(king_square(~sideToMove), occupied) & pieces(sideToMove, ROOK, QUEEN))
                 | (attacks_bb<BISHOP>(king_square(~sideToMove), occupied) & pieces(sideToMove, BISHOP, QUEEN));
        }
        
        case CASTLING: {
            Square rto = to > from ? to + WEST : to + 2 * EAST;
            return attacks_bb<ROOK>(rto, pieces() ^ from ^ to) & king_square(~sideToMove);
        }
        
        default:
            return false;
    }
}

// =============================================================================
// PSEUDO-LEGAL MOVE CHECKING
// =============================================================================

bool Position::pseudo_legal(const Move m) const {
    Color us = sideToMove;
    Square from = m.from();
    Square to = m.to();
    Piece pc = moved_piece(m);
    
    // Basic validity
    if (!m || pc == NO_PIECE || color_of(pc) != us) {
        return false;
    }
    
    // Destination check
    if (pieces(us) & to) {
        return false;
    }
    
    if (m.type() == CASTLING) {
        // Handled specially
        if (pc != make_piece(us, KING)) return false;
        // Direction check
        Square rto = to > from ? to + WEST : to + 2 * EAST;
        return castling_rook_square(::castling_rights(us, to > from)) == (to > from ? to + EAST : to + 2 * WEST)
            || (to == (us == WHITE ? SQ_G1 : SQ_G8) || to == (us == WHITE ? SQ_C1 : SQ_C8));
    }
    
    if (type_of(pc) == PAWN) {
        // Pawn moves are complex
        if (m.type() == PROMOTION) {
            if (relative_rank(us, to) != RANK_8) return false;
        }
        
        if (m.type() == EN_PASSANT) {
            return to == st->epSquare && (pawn_attacks_bb(us, from) & to);
        }
        
        Bitboard b = pawn_attacks_bb(us, from);
        if ((b & to) && !empty(to)) return true;
        
        Direction push = pawn_push(us);
        if (to == from + push && empty(to)) return true;
        if (to == from + 2 * push && relative_rank(us, from) == RANK_2 && 
            empty(to) && empty(from + push)) return true;
        
        return false;
    }
    
    // Normal piece moves
    return attacks_bb(type_of(pc), from, pieces()) & to;
}

// =============================================================================
// LEGAL MOVE CHECKING
// =============================================================================

bool Position::legal(const Move m) const {
    assert(m);
    
    Color us = sideToMove;
    Square from = m.from();
    Square to = m.to();
    
    // En passant is complex due to discovered check possibility
    if (m.type() == EN_PASSANT) {
        Square ksq = king_square(us);
        Square capsq = make_square(file_of(to), rank_of(from));
        Bitboard occupied = (pieces() ^ from ^ capsq) | to;
        
        return !(attacks_bb<ROOK>(ksq, occupied) & pieces(~us, ROOK, QUEEN))
            && !(attacks_bb<BISHOP>(ksq, occupied) & pieces(~us, BISHOP, QUEEN));
    }
    
    // Castling legality
    if (m.type() == CASTLING) {
        // King must not be in check
        if (checkers()) return false;
        
        to = to > from ? SQ_G1 : SQ_C1;
        if (us == BLACK) to = Square(to + 56);
        
        Direction d = to > from ? EAST : WEST;
        
        // Check that no square king passes through is attacked
        for (Square s = from + d; s != to + d; s += d) {
            if (attackers_to(s) & pieces(~us)) {
                return false;
            }
        }
        return true;
    }
    
    // King moves
    if (type_of(piece_on(from)) == KING) {
        return !(attackers_to(to, pieces() ^ from) & pieces(~us));
    }
    
    // Pinned pieces
    return !(blockers_for_king(us) & from) || aligned(from, to, king_square(us));
}

// =============================================================================
// DO MOVE
// =============================================================================

void Position::do_move(Move m, StateInfo& newSt) {
    do_move(m, newSt, gives_check(m));
}

void Position::do_move(Move m, StateInfo& newSt, bool givesCheck) {
    assert(m);
    
    Key k = st->key ^ Zobrist::side;
    
    // Copy state
    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st = &newSt;
    
    ++gamePly;
    ++st->rule50;
    ++st->pliesFromNull;
    
    Color us = sideToMove;
    Color them = ~us;
    Square from = m.from();
    Square to = m.to();
    Piece pc = piece_on(from);
    Piece captured = m.type() == EN_PASSANT ? make_piece(them, PAWN) : piece_on(to);
    
    assert(color_of(pc) == us);
    assert(captured == NO_PIECE || color_of(captured) == them || m.type() == CASTLING);
    
    // Castling
    if (m.type() == CASTLING) {
        assert(pc == make_piece(us, KING));
        assert(captured == NO_PIECE || type_of(captured) == ROOK);
        
        Square rfrom, rto;
        bool kingSide = to > from;
        
        // Standard king target squares
        Square kTo = make_square(kingSide ? FILE_G : FILE_C, rank_of(from));
        rfrom = castling_rook_square(::castling_rights(us, kingSide));
        rto = make_square(kingSide ? FILE_F : FILE_D, rank_of(from));
        
        // Remove pieces
        remove_piece(from);
        remove_piece(rfrom);
        
        // Put pieces in new positions
        board[from] = board[rfrom] = NO_PIECE;
        put_piece(make_piece(us, KING), kTo);
        put_piece(make_piece(us, ROOK), rto);
        
        k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][kTo];
        k ^= Zobrist::psq[make_piece(us, ROOK)][rfrom] ^ Zobrist::psq[make_piece(us, ROOK)][rto];
        
        captured = NO_PIECE;
        st->capturedPiece = NO_PIECE;
        
        // Fix the "to" for later castling rights update
        to = kTo;
    }
    
    // Handle captures
    if (captured != NO_PIECE) {
        Square capsq = to;
        
        if (m.type() == EN_PASSANT) {
            capsq = to - pawn_push(us);
            assert(pc == make_piece(us, PAWN));
            assert(relative_rank(us, to) == RANK_6);
            assert(piece_on(capsq) == make_piece(them, PAWN));
        }
        
        remove_piece(capsq);
        
        if (m.type() == EN_PASSANT) {
            board[capsq] = NO_PIECE;
        }
        
        k ^= Zobrist::psq[captured][capsq];
        
        if (type_of(captured) == PAWN) {
            st->pawnKey ^= Zobrist::psq[captured][capsq];
        } else {
            st->nonPawnMaterial[them] -= PieceValue[type_of(captured)];
        }
        
        st->rule50 = 0;
    }
    
    st->capturedPiece = captured;
    
    // Update castling rights
    if (st->castlingRights && (castlingRightsMask[from] | castlingRightsMask[to])) {
        k ^= Zobrist::castling[st->castlingRights];
        st->castlingRights &= ~(castlingRightsMask[from] | castlingRightsMask[to]);
        k ^= Zobrist::castling[st->castlingRights];
    }
    
    // Update en passant
    if (st->epSquare != SQ_NONE) {
        k ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }
    
    // Move the piece (if not castling, already handled)
    if (m.type() != CASTLING) {
        move_piece(from, to);
        k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    }
    
    // Pawn specifics
    if (type_of(pc) == PAWN) {
        // Double pawn push
        if ((int(to) ^ int(from)) == 16) {
            Square epSq = to - pawn_push(us);
            
            // Only set ep square if enemy pawn can capture
            if (pawn_attacks_bb(them, epSq) & pieces(them, PAWN)) {
                st->epSquare = epSq;
                k ^= Zobrist::enpassant[file_of(epSq)];
            }
        }
        // Promotion
        else if (m.type() == PROMOTION) {
            Piece promotion = make_piece(us, m.promotion_type());
            
            assert(relative_rank(us, to) == RANK_8);
            
            remove_piece(to);
            put_piece(promotion, to);
            
            k ^= Zobrist::psq[pc][to] ^ Zobrist::psq[promotion][to];
            st->pawnKey ^= Zobrist::psq[pc][to];
            st->nonPawnMaterial[us] += PieceValue[m.promotion_type()];
        }
        
        st->pawnKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
        st->rule50 = 0;
    }
    
    // Update state
    st->key = k;
    st->checkersBB = givesCheck ? attackers_to(king_square(them)) & pieces(us) : 0;
    
    sideToMove = them;
    
    set_check_info(st);
    
    // Repetition detection
    st->repetition = 0;
    int end = std::min(st->rule50, st->pliesFromNull);
    if (end >= 4) {
        StateInfo* stp = st->previous->previous;
        for (int i = 4; i <= end; i += 2) {
            stp = stp->previous->previous;
            if (stp->key == st->key) {
                st->repetition = stp->repetition ? -i : i;
                break;
            }
        }
    }
}

// =============================================================================
// UNDO MOVE
// =============================================================================

void Position::undo_move(Move m) {
    assert(m);
    
    sideToMove = ~sideToMove;
    
    Color us = sideToMove;
    Square from = m.from();
    Square to = m.to();
    Piece pc = piece_on(to);
    
    if (m.type() == CASTLING) {
        bool kingSide = to > from;
        Square kTo = make_square(kingSide ? FILE_G : FILE_C, rank_of(from));
        Square rfrom = castling_rook_square(::castling_rights(us, kingSide));
        Square rto = make_square(kingSide ? FILE_F : FILE_D, rank_of(from));
        
        // Remove pieces from castled positions
        remove_piece(kTo);
        remove_piece(rto);
        board[kTo] = board[rto] = NO_PIECE;
        
        // Put pieces back
        put_piece(make_piece(us, KING), from);
        put_piece(make_piece(us, ROOK), rfrom);
    } else {
        if (m.type() == PROMOTION) {
            remove_piece(to);
            pc = make_piece(us, PAWN);
            put_piece(pc, to);
        }
        
        move_piece(to, from);
        
        if (st->capturedPiece != NO_PIECE) {
            Square capsq = to;
            
            if (m.type() == EN_PASSANT) {
                capsq -= pawn_push(us);
            }
            
            put_piece(st->capturedPiece, capsq);
        }
    }
    
    st = st->previous;
    --gamePly;
}

// =============================================================================
// NULL MOVE
// =============================================================================

void Position::do_null_move(StateInfo& newSt) {
    assert(!checkers());
    
    std::memcpy(&newSt, st, sizeof(StateInfo));
    newSt.previous = st;
    st = &newSt;
    
    if (st->epSquare != SQ_NONE) {
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }
    
    st->key ^= Zobrist::side;
    ++st->rule50;
    st->pliesFromNull = 0;
    
    sideToMove = ~sideToMove;
    
    set_check_info(st);
    
    st->repetition = 0;
}

void Position::undo_null_move() {
    assert(!checkers());
    st = st->previous;
    sideToMove = ~sideToMove;
}

// =============================================================================
// STATIC EXCHANGE EVALUATION (SEE)
// =============================================================================

bool Position::see_ge(Move m, int threshold) const {
    if (m.type() != NORMAL) {
        return VALUE_ZERO >= threshold;
    }
    
    Square from = m.from();
    Square to = m.to();
    
    int swap = int(PieceValue[type_of(piece_on(to))]) - threshold;
    if (swap < 0) return false;
    
    swap = int(PieceValue[type_of(piece_on(from))]) - swap;
    if (swap <= 0) return true;
    
    Bitboard occupied = pieces() ^ from ^ to;
    Color stm = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);
    Bitboard stmAttackers, bb;
    int res = 1;
    
    while (true) {
        stm = ~stm;
        attackers &= occupied;
        
        stmAttackers = attackers & pieces(stm);
        if (!stmAttackers) break;
        
        if (pinners(~stm) & occupied) {
            stmAttackers &= ~blockers_for_king(stm);
            if (!stmAttackers) break;
        }
        
        res ^= 1;
        
        // Find least valuable attacker
        if ((bb = stmAttackers & pieces(PAWN))) {
            if ((swap = PawnValueMg - swap) < res) break;
            occupied ^= lsb(bb);
            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        } else if ((bb = stmAttackers & pieces(KNIGHT))) {
            if ((swap = KnightValueMg - swap) < res) break;
            occupied ^= lsb(bb);
        } else if ((bb = stmAttackers & pieces(BISHOP))) {
            if ((swap = BishopValueMg - swap) < res) break;
            occupied ^= lsb(bb);
            attackers |= attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN);
        } else if ((bb = stmAttackers & pieces(ROOK))) {
            if ((swap = RookValueMg - swap) < res) break;
            occupied ^= lsb(bb);
            attackers |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN);
        } else if ((bb = stmAttackers & pieces(QUEEN))) {
            if ((swap = QueenValueMg - swap) < res) break;
            occupied ^= lsb(bb);
            attackers |= (attacks_bb<BISHOP>(to, occupied) & pieces(BISHOP, QUEEN))
                       | (attacks_bb<ROOK>(to, occupied) & pieces(ROOK, QUEEN));
        } else {
            // King: if we're here and opponent still has attackers, we lose
            if (attackers & ~pieces(stm)) {
                res ^= 1;
            }
            break;
        }
    }
    
    return bool(res);
}

// =============================================================================
// DRAW DETECTION
// =============================================================================

bool Position::is_draw(int ply) const {
    if (st->rule50 > 99 && (!checkers() || count(sideToMove, PAWN) || count(sideToMove, ROOK) 
                           || count(sideToMove, QUEEN) || count(sideToMove, KNIGHT)
                           || count(sideToMove, BISHOP))) {
        return true;
    }
    
    return has_game_cycle(ply) || is_insufficient_material();
}

bool Position::has_repeated() const {
    StateInfo* stp = st;
    int end = std::min(st->rule50, st->pliesFromNull);
    
    while (end-- >= 4) {
        if (stp->repetition) {
            return true;
        }
        stp = stp->previous;
    }
    return false;
}

bool Position::has_game_cycle(int ply) const {
    int j;
    int end = std::min(st->rule50, st->pliesFromNull);
    
    if (end < 3) return false;
    
    Key originalKey = st->key;
    StateInfo* stp = st->previous;
    
    for (int i = 3; i <= end; i += 2) {
        stp = stp->previous->previous;
        
        if (stp->key == originalKey) {
            // Found repetition
            if (ply > i) return true;  // Repetition in search tree
            
            // Check if it's a draw by threefold
            if (stp->repetition) return true;
            
            StateInfo* stp2 = stp;
            j = i;
            while (j <= end && (stp2 = stp2->previous->previous)) {
                j += 2;
                if (stp2->key == originalKey) return true;
            }
        }
    }
    return false;
}

bool Position::is_insufficient_material() const {
    int npm_w = popcount(pieces(WHITE) & ~pieces(PAWN) & ~pieces(KING));
    int npm_b = popcount(pieces(BLACK) & ~pieces(PAWN) & ~pieces(KING));
    
    // If any pawns exist, not insufficient
    if (pieces(PAWN)) return false;
    
    // If any queens or rooks exist, not insufficient
    if (pieces(ROOK) || pieces(QUEEN)) return false;
    
    // K vs K
    if (npm_w == 0 && npm_b == 0) return true;
    
    // K+minor vs K
    if (npm_w + npm_b == 1) return true;
    
    // K+B vs K+B with same-colored bishops
    if (npm_w == 1 && npm_b == 1) {
        if (count(WHITE, BISHOP) == 1 && count(BLACK, BISHOP) == 1) {
            Square wb = lsb(pieces(WHITE, BISHOP));
            Square bb = lsb(pieces(BLACK, BISHOP));
            return !opposite_colors(wb, bb);
        }
    }
    
    // K+N+N vs K is technically winnable but extremely difficult
    // We'll consider it sufficient material
    
    return false;
}

// =============================================================================
// KEY AFTER MOVE (without making it)
// =============================================================================

Key Position::key_after(Move m) const {
    Square from = m.from();
    Square to = m.to();
    Piece pc = piece_on(from);
    Piece captured = piece_on(to);
    Key k = st->key ^ Zobrist::side;
    
    if (captured != NO_PIECE) {
        k ^= Zobrist::psq[captured][to];
    }
    
    // En passant
    if (m.type() == EN_PASSANT) {
        Square capsq = make_square(file_of(to), rank_of(from));
        k ^= Zobrist::psq[make_piece(~sideToMove, PAWN)][capsq];
    }
    
    // Castling
    if (m.type() == CASTLING) {
        // This is simplified - full key computation would be more complex
        k ^= Zobrist::psq[pc][from];
        Square kto = to > from ? SQ_G1 : SQ_C1;
        if (sideToMove == BLACK) kto = Square(kto + 56);
        k ^= Zobrist::psq[pc][kto];
    } else if (m.type() == PROMOTION) {
        k ^= Zobrist::psq[pc][from];
        k ^= Zobrist::psq[make_piece(sideToMove, m.promotion_type())][to];
    } else {
        k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    }
    
    // EP square changes
    if (st->epSquare != SQ_NONE) {
        k ^= Zobrist::enpassant[file_of(st->epSquare)];
    }
    
    // New EP square for double pawn push
    if (type_of(pc) == PAWN && ((int(to) ^ int(from)) == 16)) {
        k ^= Zobrist::enpassant[file_of(to)];
    }
    
    // Castling rights changes
    if (st->castlingRights && (castlingRightsMask[from] | castlingRightsMask[to])) {
        k ^= Zobrist::castling[st->castlingRights];
        k ^= Zobrist::castling[st->castlingRights & ~(castlingRightsMask[from] | castlingRightsMask[to])];
    }
    
    return k;
}

// =============================================================================
// CASTLING IMPEDED
// =============================================================================

bool Position::castling_impeded(CastlingRights cr) const {
    return pieces() & castlingPath[cr];
}

// =============================================================================
// DEBUG / STRING OUTPUT
// =============================================================================

std::string Position::to_string() const {
    std::ostringstream ss;
    
    ss << "\n +---+---+---+---+---+---+---+---+\n";
    
    for (Rank r = RANK_8; r >= RANK_1; --r) {
        for (File f = FILE_A; f <= FILE_H; ++f) {
            ss << " | " << PieceToChar[piece_on(make_square(f, r))];
        }
        ss << " | " << (1 + r) << "\n +---+---+---+---+---+---+---+---+\n";
    }
    
    ss << "   a   b   c   d   e   f   g   h\n\n";
    ss << "Fen: " << fen() << "\n";
    ss << "Key: " << std::hex << st->key << std::dec << "\n";
    ss << "Checkers: " << Bitboards::pretty(checkers());
    
    return ss.str();
}

bool Position::pos_is_ok() const {
    // Basic validation
    if (count(WHITE, KING) != 1 || count(BLACK, KING) != 1) return false;
    if (pieces(WHITE) & pieces(BLACK)) return false;
    if ((pieces(WHITE) | pieces(BLACK)) != pieces()) return false;
    
    // Piece counts
    for (Color c : {WHITE, BLACK}) {
        if (count(c, PAWN) > 8) return false;
        if (popcount(pieces(c)) > 16) return false;
    }
    
    // King not in check by side to move
    if (attackers_to(king_square(~sideToMove)) & pieces(sideToMove)) {
        return false;
    }
    
    return true;
}

void Position::flip() {
    // Mirror position vertically
    std::string f = fen();
    // This is a simplified flip - full implementation would be more complex
    set(f, st);
}
