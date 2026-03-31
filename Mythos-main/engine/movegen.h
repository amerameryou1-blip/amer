#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "position.h"
#include "types.h"

// =============================================================================
// MOVE GENERATION TYPES
// =============================================================================

enum GenType {
    CAPTURES,
    QUIETS,
    QUIET_CHECKS,
    EVASIONS,
    NON_EVASIONS,
    LEGAL
};

// =============================================================================
// MOVE LIST
// =============================================================================

struct MoveList {
    ScoredMove moves[MAX_MOVES];
    int count = 0;
    
    void add(Move m, int score = 0) {
        assert(count < MAX_MOVES);
        moves[count++] = ScoredMove(m, score);
    }
    
    ScoredMove* begin() { return moves; }
    ScoredMove* end() { return moves + count; }
    const ScoredMove* begin() const { return moves; }
    const ScoredMove* end() const { return moves + count; }
    
    int size() const { return count; }
    bool empty() const { return count == 0; }
    
    ScoredMove& operator[](int i) { return moves[i]; }
    const ScoredMove& operator[](int i) const { return moves[i]; }
    
    void clear() { count = 0; }
    
    bool contains(Move m) const {
        for (int i = 0; i < count; i++) {
            if (moves[i].move == m) return true;
        }
        return false;
    }
};

// =============================================================================
// MOVE GENERATION
// =============================================================================

template<GenType GT>
void generate(const Position& pos, MoveList& moves);

// Convenience function
inline void generate_legal(const Position& pos, MoveList& moves) {
    generate<LEGAL>(pos, moves);
}

// =============================================================================
// PERFT
// =============================================================================

uint64_t perft(Position& pos, int depth);
void perft_divide(Position& pos, int depth);

#endif // MOVEGEN_H
