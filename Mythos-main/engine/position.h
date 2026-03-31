#ifndef POSITION_H
#define POSITION_H

#include "bitboard.h"
#include "types.h"
#include <string>
#include <deque>

// =============================================================================
// ZOBRIST HASHING
// =============================================================================

namespace Zobrist {
    extern Key psq[PIECE_NB][SQUARE_NB];
    extern Key enpassant[FILE_NB];
    extern Key castling[CASTLING_RIGHT_NB];
    extern Key side;
    extern Key noPawns;
    
    void init();
}

// =============================================================================
// STATE INFO (for unmake)
// =============================================================================

struct StateInfo {
    // Copied when making a move
    Key    pawnKey;
    Key    materialKey;
    Value  nonPawnMaterial[COLOR_NB];
    int    castlingRights;
    int    rule50;
    int    pliesFromNull;
    Square epSquare;
    
    // Not copied
    Key        key;
    Bitboard   checkersBB;
    StateInfo* previous;
    Bitboard   blockersForKing[COLOR_NB];
    Bitboard   pinners[COLOR_NB];
    Bitboard   checkSquares[PIECE_TYPE_NB];
    Piece      capturedPiece;
    int        repetition;
};

// =============================================================================
// POSITION CLASS
// =============================================================================

class Position {
public:
    static void init();
    
    // Setup
    Position& set(const std::string& fenStr, StateInfo* si);
    Position& set(const Position& pos, StateInfo* si);
    std::string fen() const;
    
    // Piece access
    Bitboard pieces(PieceType pt = NO_PIECE_TYPE) const;
    Bitboard pieces(PieceType pt1, PieceType pt2) const;
    Bitboard pieces(Color c) const;
    Bitboard pieces(Color c, PieceType pt) const;
    Bitboard pieces(Color c, PieceType pt1, PieceType pt2) const;
    Piece piece_on(Square s) const;
    Square ep_square() const;
    bool empty(Square s) const;
    int count(Piece pc) const;
    int count(Color c, PieceType pt) const;
    Square square(PieceType pt, Color c) const;
    
    // Castling
    CastlingRights castling_rights(Color c) const;
    bool can_castle(CastlingRights cr) const;
    bool castling_impeded(CastlingRights cr) const;
    Square castling_rook_square(CastlingRights cr) const;
    
    // Checking
    Bitboard checkers() const;
    Bitboard blockers_for_king(Color c) const;
    Bitboard pinners(Color c) const;
    Bitboard check_squares(PieceType pt) const;
    bool is_check() const;
    
    // Attacks
    Bitboard attackers_to(Square s) const;
    Bitboard attackers_to(Square s, Bitboard occupied) const;
    Bitboard slider_blockers(Bitboard sliders, Square s, Bitboard& pinners) const;
    bool gives_check(Move m) const;
    
    // Properties
    Color side_to_move() const;
    int game_ply() const;
    int rule50_count() const;
    Key key() const;
    Key pawn_key() const;
    Key material_key() const;
    Value non_pawn_material(Color c) const;
    Value non_pawn_material() const;
    int piece_count() const;
    
    // King
    Square king_square(Color c) const;
    
    // Move validity
    bool pseudo_legal(const Move m) const;
    bool legal(const Move m) const;
    bool capture(Move m) const;
    bool capture_or_promotion(Move m) const;
    Piece moved_piece(Move m) const;
    
    // Move execution
    void do_move(Move m, StateInfo& newSt);
    void do_move(Move m, StateInfo& newSt, bool givesCheck);
    void undo_move(Move m);
    void do_null_move(StateInfo& newSt);
    void undo_null_move();
    
    // Static exchange evaluation
    bool see_ge(Move m, int threshold = 0) const;
    
    // Draw detection
    bool is_draw(int ply) const;
    bool has_repeated() const;
    bool has_game_cycle(int ply) const;
    bool is_insufficient_material() const;
    
    // Key after move
    Key key_after(Move m) const;
    
    // Debugging
    bool pos_is_ok() const;
    void flip();
    std::string to_string() const;
    
public:
    // Internal helpers
    void set_castling_right(Color c, Square rfrom);
    void set_check_info(StateInfo* si) const;
    void set_state(StateInfo* si) const;
    void put_piece(Piece pc, Square s);
    void remove_piece(Square s);
    void move_piece(Square from, Square to);
    
    // Board representation
    Piece board[SQUARE_NB];
    Bitboard byTypeBB[PIECE_TYPE_NB];
    Bitboard byColorBB[COLOR_NB];
    int pieceCount[PIECE_NB];
    int castlingRightsMask[SQUARE_NB];
    Square castlingRookSquare[CASTLING_RIGHT_NB];
    Bitboard castlingPath[CASTLING_RIGHT_NB];
    
    // Game state
    Color sideToMove;
    int gamePly;
    StateInfo* st;
    
    // State history for repetition
    StateInfo startState;
    std::deque<StateInfo> stateHistory;
};

// =============================================================================
// INLINE IMPLEMENTATIONS
// =============================================================================

inline Bitboard Position::pieces(PieceType pt) const {
    return byTypeBB[pt];
}

inline Bitboard Position::pieces(PieceType pt1, PieceType pt2) const {
    return byTypeBB[pt1] | byTypeBB[pt2];
}

inline Bitboard Position::pieces(Color c) const {
    return byColorBB[c];
}

inline Bitboard Position::pieces(Color c, PieceType pt) const {
    return byColorBB[c] & byTypeBB[pt];
}

inline Bitboard Position::pieces(Color c, PieceType pt1, PieceType pt2) const {
    return byColorBB[c] & (byTypeBB[pt1] | byTypeBB[pt2]);
}

inline Piece Position::piece_on(Square s) const {
    assert(is_ok(s));
    return board[s];
}

inline bool Position::empty(Square s) const {
    return piece_on(s) == NO_PIECE;
}

inline Color Position::side_to_move() const {
    return sideToMove;
}

inline int Position::game_ply() const {
    return gamePly;
}

inline int Position::rule50_count() const {
    return st->rule50;
}

inline Square Position::ep_square() const {
    return st->epSquare;
}

inline Key Position::key() const {
    return st->key;
}

inline Key Position::pawn_key() const {
    return st->pawnKey;
}

inline Key Position::material_key() const {
    return st->materialKey;
}

inline Value Position::non_pawn_material(Color c) const {
    return st->nonPawnMaterial[c];
}

inline Value Position::non_pawn_material() const {
    return non_pawn_material(WHITE) + non_pawn_material(BLACK);
}

inline int Position::count(Piece pc) const {
    return pieceCount[pc];
}

inline int Position::count(Color c, PieceType pt) const {
    return pieceCount[make_piece(c, pt)];
}

inline Square Position::square(PieceType pt, Color c) const {
    assert(count(c, pt) == 1);
    return lsb(pieces(c, pt));
}

inline Square Position::king_square(Color c) const {
    return square(KING, c);
}

inline CastlingRights Position::castling_rights(Color c) const {
    return CastlingRights(st->castlingRights & (c == WHITE ? WHITE_CASTLING : BLACK_CASTLING));
}

inline bool Position::can_castle(CastlingRights cr) const {
    return st->castlingRights & cr;
}

inline Square Position::castling_rook_square(CastlingRights cr) const {
    return castlingRookSquare[cr];
}

inline Bitboard Position::checkers() const {
    return st->checkersBB;
}

inline Bitboard Position::blockers_for_king(Color c) const {
    return st->blockersForKing[c];
}

inline Bitboard Position::pinners(Color c) const {
    return st->pinners[c];
}

inline Bitboard Position::check_squares(PieceType pt) const {
    return st->checkSquares[pt];
}

inline bool Position::is_check() const {
    return checkers();
}

inline int Position::piece_count() const {
    return popcount(pieces());
}

inline bool Position::capture(Move m) const {
    assert(m);
    return (!empty(m.to()) && m.type() != CASTLING) || m.type() == EN_PASSANT;
}

inline bool Position::capture_or_promotion(Move m) const {
    assert(m);
    return capture(m) || m.type() == PROMOTION;
}

inline Piece Position::moved_piece(Move m) const {
    return piece_on(m.from());
}

inline void Position::put_piece(Piece pc, Square s) {
    board[s] = pc;
    byTypeBB[NO_PIECE_TYPE] |= s;
    byTypeBB[type_of(pc)] |= s;
    byColorBB[color_of(pc)] |= s;
    pieceCount[pc]++;
    pieceCount[make_piece(color_of(pc), NO_PIECE_TYPE)]++;
}

inline void Position::remove_piece(Square s) {
    Piece pc = board[s];
    byTypeBB[NO_PIECE_TYPE] ^= s;
    byTypeBB[type_of(pc)] ^= s;
    byColorBB[color_of(pc)] ^= s;
    board[s] = NO_PIECE;
    pieceCount[pc]--;
    pieceCount[make_piece(color_of(pc), NO_PIECE_TYPE)]--;
}

inline void Position::move_piece(Square from, Square to) {
    Piece pc = board[from];
    Bitboard fromTo = from | to;
    byTypeBB[NO_PIECE_TYPE] ^= fromTo;
    byTypeBB[type_of(pc)] ^= fromTo;
    byColorBB[color_of(pc)] ^= fromTo;
    board[from] = NO_PIECE;
    board[to] = pc;
}

#endif // POSITION_H
