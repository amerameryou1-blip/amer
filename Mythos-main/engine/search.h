#ifndef SEARCH_H
#define SEARCH_H

#include "position.h"
#include "movegen.h"
#include "types.h"
#include <atomic>
#include <chrono>
#include <vector>
#include <memory>

// =============================================================================
// TRANSPOSITION TABLE
// =============================================================================

struct TTEntry {
    Key key;
    Move move;
    int16_t score;
    int16_t staticEval;
    uint8_t depth;
    uint8_t bound;
    uint8_t generation;
    
    void save(Key k, Value v, Bound b, int d, Move m, Value eval, uint8_t gen);
    Value value(int ply) const;
};

class TranspositionTable {
public:
    TranspositionTable(size_t mbSize = 16);
    ~TranspositionTable();
    
    void resize(size_t mbSize);
    void clear();
    void new_search();
    
    TTEntry* probe(Key key, bool& found) const;
    int hashfull() const;
    uint8_t current_generation() const { return generation; }
    
private:
    TTEntry* table;
    size_t clusterCount;
    size_t clusterMask;
    uint8_t generation;
};

// =============================================================================
// SEARCH LIMITS
// =============================================================================

struct SearchLimits {
    int depth = MAX_PLY;
    int64_t nodes = 0;
    int movetime = 0;
    int time[COLOR_NB] = {0, 0};
    int inc[COLOR_NB] = {0, 0};
    int movestogo = 0;
    bool infinite = false;
    bool ponder = false;
    std::vector<Move> searchmoves;
    
    void clear() {
        depth = MAX_PLY;
        nodes = 0;
        movetime = 0;
        time[WHITE] = time[BLACK] = 0;
        inc[WHITE] = inc[BLACK] = 0;
        movestogo = 0;
        infinite = false;
        ponder = false;
        searchmoves.clear();
    }
};

// =============================================================================
// SEARCH INFO
// =============================================================================

struct SearchInfo {
    int depth;
    int seldepth;
    int64_t nodes;
    int64_t tbHits;
    int score;
    bool isMate;
    int mateIn;
    int hashfull;
    int64_t nps;
    int time;
    Move pv[MAX_PLY];
    int pvLength;
};

// =============================================================================
// SEARCH CLASS
// =============================================================================

class Search {
public:
    Search();
    ~Search();
    
    // Main interface
    void go(Position& pos, const SearchLimits& limits);
    void stop();
    void wait();
    
    // Settings
    void set_tt_size(size_t mbSize);
    void clear();
    void new_game();
    
    // Info
    Move best_move() const { return bestMove; }
    Move ponder_move() const { return ponderMove; }
    
    // Accessors
    TranspositionTable& tt() { return TT; }
    bool is_stopped() const { return stopped.load(std::memory_order_relaxed); }
    
private:
    // Search functions
    void iterative_deepening(Position& pos);
    Value search(Position& pos, Value alpha, Value beta, int depth, int ply, bool cutNode);
    Value qsearch(Position& pos, Value alpha, Value beta, int ply);
    
    // Time management
    void allocate_time(Color us);
    void check_time();
    
    // Move ordering
    void score_moves(Position& pos, MoveList& moves, Move ttMove, int ply);
    void pick_move(MoveList& moves, int startIdx);
    
    // State
    TranspositionTable TT;
    SearchLimits limits;
    
    std::atomic<bool> stopped;
    bool searching;
    
    Move bestMove;
    Move ponderMove;
    
    // Time management
    std::chrono::time_point<std::chrono::steady_clock> startTime;
    int64_t allocatedTime;
    int64_t maxTime;
    
    // Statistics
    int64_t nodes;
    int selDepth;
    
    // Killer moves
    Move killers[MAX_PLY][2];
    
    // History heuristic
    int history[COLOR_NB][SQUARE_NB][SQUARE_NB];
    
    // Counter moves
    Move counterMoves[PIECE_NB][SQUARE_NB];
    
    // Static eval cache per ply
    Value staticEval[MAX_PLY];
    
    // PV table
    Move pvTable[MAX_PLY][MAX_PLY];
    int pvLength[MAX_PLY];
    
    // State stack for search
    StateInfo stateStack[MAX_PLY + 10];
    int rootPly;
};

// Global search instance
extern Search Threads;

#endif // SEARCH_H
