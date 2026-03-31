#include "search.h"
#include "evaluate.h"
#include "movegen.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <thread>

// Global search instance
Search Threads;

// =============================================================================
// LATE MOVE REDUCTION TABLE
// =============================================================================

namespace {

int LMRTable[MAX_PLY][MAX_MOVES];

void init_lmr() {
    for (int d = 1; d < MAX_PLY; d++) {
        for (int m = 1; m < MAX_MOVES; m++) {
            LMRTable[d][m] = int(1.10 + std::log(d) * std::log(m) / 1.85);
        }
    }
}

struct LMRInit {
    LMRInit() { init_lmr(); }
} lmrInit;

// Move ordering scores
constexpr int SCORE_TT_MOVE     = 10000000;
constexpr int SCORE_CAPTURE     = 8000000;
constexpr int SCORE_KILLER1     = 6000000;
constexpr int SCORE_KILLER2     = 5000000;
constexpr int SCORE_COUNTER     = 4000000;

// Futility margins (more aggressive for faster self-play)
constexpr int FutilityMargin[4] = { 0, 300, 500, 800 };

// Razoring margin
constexpr int RazoringMargin = 400;

// Null move reduction
constexpr int NullMoveDepth = 2;

// Delta pruning margin for qsearch
constexpr int DeltaMargin = 200;

} // anonymous namespace

// =============================================================================
// TRANSPOSITION TABLE
// =============================================================================

void TTEntry::save(Key k, Value v, Bound b, int d, Move m, Value eval, uint8_t gen) {
    // Preserve move if none provided
    if (m || k != key) {
        move = m;
    }
    
    // Overwrite if new entry is from current search with better depth or different position
    if (b == BOUND_EXACT || k != key || d + 4 > depth || generation != gen) {
        key = k;
        score = int16_t(v);
        staticEval = int16_t(eval);
        depth = uint8_t(d);
        bound = uint8_t(b);
        generation = gen;
    }
}

Value TTEntry::value(int ply) const {
    if (score >= VALUE_TB_WIN_IN_MAX_PLY) {
        return Value(score - ply);
    } else if (score <= VALUE_TB_LOSS_IN_MAX_PLY) {
        return Value(score + ply);
    }
    return Value(score);
}

TranspositionTable::TranspositionTable(size_t mbSize) : table(nullptr), clusterCount(0), clusterMask(0), generation(0) {
    resize(mbSize);
}

TranspositionTable::~TranspositionTable() {
    delete[] table;
}

void TranspositionTable::resize(size_t mbSize) {
    delete[] table;

    size_t requested = mbSize * 1024 * 1024 / sizeof(TTEntry);
    requested = std::max(requested, size_t(1024));

    clusterCount = 1;
    while ((clusterCount << 1) <= requested) {
        clusterCount <<= 1;
    }
    clusterMask = clusterCount - 1;

    table = new TTEntry[clusterCount]();
    clear();
}

void TranspositionTable::clear() {
    std::memset(table, 0, clusterCount * sizeof(TTEntry));
    generation = 0;
}

void TranspositionTable::new_search() {
    generation++;
}

TTEntry* TranspositionTable::probe(Key key, bool& found) const {
    TTEntry* entry = &table[key & clusterMask];
    found = (entry->key == key);
    return entry;
}

int TranspositionTable::hashfull() const {
    int count = 0;
    const size_t sample = std::min(clusterCount, size_t(1000));
    for (size_t i = 0; i < sample; i++) {
        if (table[i].generation == generation && table[i].key != 0) {
            count++;
        }
    }
    return count;
}

// =============================================================================
// SEARCH CONSTRUCTOR/DESTRUCTOR
// =============================================================================

Search::Search() : TT(16), stopped(false), searching(false), nodes(0), selDepth(0), rootPly(0) {
    std::memset(killers, 0, sizeof(killers));
    std::memset(history, 0, sizeof(history));
    std::memset(counterMoves, 0, sizeof(counterMoves));
    std::memset(staticEval, 0, sizeof(staticEval));
    std::memset(pvTable, 0, sizeof(pvTable));
    std::memset(pvLength, 0, sizeof(pvLength));
}

Search::~Search() {
    stop();
    wait();
}

// =============================================================================
// INTERFACE
// =============================================================================

void Search::set_tt_size(size_t mbSize) {
    TT.resize(mbSize);
}

void Search::clear() {
    TT.clear();
    std::memset(killers, 0, sizeof(killers));
    std::memset(history, 0, sizeof(history));
    std::memset(counterMoves, 0, sizeof(counterMoves));
}

void Search::new_game() {
    clear();
}

void Search::stop() {
    stopped.store(true, std::memory_order_relaxed);
}

void Search::wait() {
    while (searching) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Search::go(Position& pos, const SearchLimits& lim) {
    stop();
    wait();
    
    limits = lim;
    stopped.store(false, std::memory_order_relaxed);
    searching = true;
    
    iterative_deepening(pos);
    
    searching = false;
}

// =============================================================================
// TIME MANAGEMENT
// =============================================================================

void Search::allocate_time(Color us) {
    startTime = std::chrono::steady_clock::now();
    
    if (limits.movetime > 0) {
        allocatedTime = limits.movetime - 50;  // Small overhead buffer
        maxTime = limits.movetime - 10;
    } else if (limits.time[us] > 0) {
        int time = limits.time[us];
        int inc = limits.inc[us];
        int moves = limits.movestogo > 0 ? limits.movestogo : 40;
        
        // Simple time allocation: time/moves + increment/2
        allocatedTime = time / moves + inc / 2;
        
        // Don't use more than 1/3 of remaining time
        allocatedTime = std::min(allocatedTime, int64_t(time / 3));
        
        // Max time is double allocated
        maxTime = std::min(allocatedTime * 2, int64_t(time - 100));
        
        // Minimum thinking time
        allocatedTime = std::max(allocatedTime, int64_t(50));
        maxTime = std::max(maxTime, int64_t(100));
    } else {
        allocatedTime = INT64_MAX;
        maxTime = INT64_MAX;
    }
}

void Search::check_time() {
    if (limits.nodes > 0 && nodes >= limits.nodes) {
        stopped.store(true, std::memory_order_relaxed);
        return;
    }

    if (limits.infinite) return;

    if ((nodes & 1023) == 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();

        if (elapsed >= maxTime) {
            stopped.store(true, std::memory_order_relaxed);
        }
    }
}

// =============================================================================
// MOVE ORDERING
// =============================================================================

void Search::score_moves(Position& pos, MoveList& moves, Move ttMove, int ply) {
    Color us = pos.side_to_move();
    
    for (int i = 0; i < moves.size(); i++) {
        Move m = moves[i].move;
        
        if (m == ttMove) {
            moves[i].score = SCORE_TT_MOVE;
        } else if (pos.capture(m)) {
            // MVV-LVA
            Piece captured = pos.piece_on(m.to());
            Piece moving = pos.moved_piece(m);
            
            if (m.type() == EN_PASSANT) {
                captured = make_piece(~us, PAWN);
            }
            
            int victimValue = PieceValue[type_of(captured)];
            int attackerValue = PieceValue[type_of(moving)];
            
            moves[i].score = SCORE_CAPTURE + victimValue * 10 - attackerValue;
            
            // Boost winning captures, reduce losing ones
            if (!pos.see_ge(m, 0)) {
                moves[i].score -= SCORE_CAPTURE;
            }
        } else if (m == killers[ply][0]) {
            moves[i].score = SCORE_KILLER1;
        } else if (m == killers[ply][1]) {
            moves[i].score = SCORE_KILLER2;
        } else {
            // History heuristic
            moves[i].score = history[us][m.from()][m.to()];
        }
    }
}

void Search::pick_move(MoveList& moves, int startIdx) {
    int bestIdx = startIdx;
    int bestScore = moves[startIdx].score;
    
    for (int i = startIdx + 1; i < moves.size(); i++) {
        if (moves[i].score > bestScore) {
            bestScore = moves[i].score;
            bestIdx = i;
        }
    }
    
    if (bestIdx != startIdx) {
        std::swap(moves[startIdx], moves[bestIdx]);
    }
}

// =============================================================================
// ITERATIVE DEEPENING
// =============================================================================

void Search::iterative_deepening(Position& pos) {
    allocate_time(pos.side_to_move());
    TT.new_search();
    
    nodes = 0;
    selDepth = 0;
    rootPly = pos.game_ply();
    bestMove = MOVE_NONE;
    ponderMove = MOVE_NONE;
    
    std::memset(pvLength, 0, sizeof(pvLength));
    
    // Generate root moves
    MoveList rootMoves;
    generate<LEGAL>(pos, rootMoves);
    
    if (rootMoves.empty()) {
        std::cout << "bestmove 0000" << std::endl;
        return;
    }
    
    if (rootMoves.size() == 1) {
        bestMove = rootMoves[0].move;
        std::cout << "bestmove " << bestMove.to_uci() << std::endl;
        return;
    }
    
    Value alpha, beta, score;
    Value prevScore = VALUE_ZERO;
    int aspirationWindow = 25;
    
    for (int depth = 1; depth <= limits.depth; depth++) {
        selDepth = 0;
        
        // Aspiration windows
        if (depth >= 5) {
            alpha = Value(std::max<int>(prevScore - aspirationWindow, -VALUE_INFINITE));
            beta = Value(std::min<int>(prevScore + aspirationWindow, VALUE_INFINITE));
        } else {
            alpha = -VALUE_INFINITE;
            beta = VALUE_INFINITE;
        }
        
        // Search with aspiration window
        while (true) {
            score = search(pos, alpha, beta, depth, 0, false);
            
            if (stopped.load(std::memory_order_relaxed)) break;
            
            if (score <= alpha) {
                alpha = Value(std::max(int(alpha - aspirationWindow), -int(VALUE_INFINITE)));
                aspirationWindow *= 2;
            } else if (score >= beta) {
                beta = Value(std::min(int(beta + aspirationWindow), int(VALUE_INFINITE)));
                aspirationWindow *= 2;
            } else {
                break;
            }
        }
        
        if (stopped.load(std::memory_order_relaxed)) break;
        
        prevScore = score;
        aspirationWindow = 25;
        
        // Update best move
        if (pvLength[0] > 0) {
            bestMove = pvTable[0][0];
            if (pvLength[0] > 1) {
                ponderMove = pvTable[0][1];
            }
        }
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();

        if (depth <= 3 || elapsed >= allocatedTime / 4 || limits.movetime > 0 || !limits.searchmoves.empty()) {
            int64_t nps = elapsed > 0 ? nodes * 1000 / elapsed : nodes;

            std::cout << "info depth " << depth
                      << " seldepth " << selDepth
                      << " score ";

            if (score >= VALUE_MATE_IN_MAX_PLY) {
                std::cout << "mate " << (VALUE_MATE - score + 1) / 2;
            } else if (score <= VALUE_MATED_IN_MAX_PLY) {
                std::cout << "mate " << -(VALUE_MATE + score) / 2;
            } else {
                std::cout << "cp " << score;
            }

            std::cout << " nodes " << nodes
                      << " nps " << nps
                      << " time " << elapsed
                      << " hashfull " << TT.hashfull()
                      << " pv";

            for (int i = 0; i < pvLength[0]; i++) {
                std::cout << " " << pvTable[0][i].to_uci();
            }
            std::cout << std::endl;
        }

        if ((limits.nodes > 0 && nodes >= limits.nodes) || (elapsed >= allocatedTime && !limits.infinite)) {
            break;
        }
    }
    
    std::cout << "bestmove " << bestMove.to_uci();
    if (ponderMove) {
        std::cout << " ponder " << ponderMove.to_uci();
    }
    std::cout << std::endl;
}

// =============================================================================
// ALPHA-BETA SEARCH
// =============================================================================

Value Search::search(Position& pos, Value alpha, Value beta, int depth, int ply, bool cutNode) {
    // Check for stop
    check_time();
    if (stopped.load(std::memory_order_relaxed)) return VALUE_ZERO;
    
    // Initialize PV length
    pvLength[ply] = ply;
    
    // Check extension
    bool inCheck = pos.is_check();
    if (inCheck) depth++;
    
    // Quiescence search at depth 0
    if (depth <= 0) {
        return qsearch(pos, alpha, beta, ply);
    }
    
    nodes++;
    selDepth = std::max(selDepth, ply);
    
    // Draw detection
    if (ply > 0) {
        if (pos.is_draw(ply)) {
            return VALUE_DRAW;
        }
        
        // Max ply check
        if (ply >= MAX_PLY - 1) {
            return inCheck ? VALUE_DRAW : Eval::evaluate(pos);
        }
    }
    
    // Mate distance pruning
    alpha = std::max(alpha, mated_in(ply));
    beta = std::min(beta, mate_in(ply + 1));
    if (alpha >= beta) return alpha;
    
    // TT probe
    bool ttHit;
    TTEntry* tte = TT.probe(pos.key(), ttHit);
    Move ttMove = ttHit ? tte->move : MOVE_NONE;
    Value ttValue = ttHit ? tte->value(ply) : VALUE_NONE;
    
    // TT cutoff
    if (ttHit && ply > 0 && tte->depth >= depth) {
        if (tte->bound == BOUND_EXACT) {
            return ttValue;
        } else if (tte->bound == BOUND_LOWER && ttValue >= beta) {
            return ttValue;
        } else if (tte->bound == BOUND_UPPER && ttValue <= alpha) {
            return ttValue;
        }
    }
    
    // Static evaluation
    Value eval;
    if (inCheck) {
        eval = staticEval[ply] = VALUE_NONE;
    } else if (ttHit && tte->staticEval != VALUE_NONE) {
        eval = staticEval[ply] = Value(tte->staticEval);
    } else {
        eval = staticEval[ply] = Eval::evaluate(pos);
    }
    
    bool improving = !inCheck && ply >= 2 && staticEval[ply] > staticEval[ply - 2];
    
    // Razoring
    if (!inCheck && depth <= 3 && eval + RazoringMargin < alpha) {
        Value razorScore = qsearch(pos, alpha, beta, ply);
        if (razorScore <= alpha) {
            return razorScore;
        }
    }
    
    // Futility pruning (child node)
    bool canPrune = !inCheck && depth <= 3 && eval >= beta + FutilityMargin[depth];
    if (canPrune && pos.non_pawn_material(pos.side_to_move()) > 0) {
        return eval;
    }
    
    // Null move pruning
    if (!inCheck && ply > 0 && depth >= NullMoveDepth &&
        eval >= beta && pos.non_pawn_material(pos.side_to_move()) > 0 &&
        staticEval[ply - 1] != VALUE_NONE) {
        
        int R = 3 + depth / 3 + std::min((eval - beta) / 200, 3);
        
        pos.do_null_move(stateStack[ply]);
        Value nullValue = -search(pos, -beta, -beta + 1, depth - R - 1, ply + 1, !cutNode);
        pos.undo_null_move();
        
        if (stopped.load(std::memory_order_relaxed)) return VALUE_ZERO;
        
        if (nullValue >= beta) {
            // Don't return unproven mate scores
            if (nullValue >= VALUE_TB_WIN_IN_MAX_PLY) {
                nullValue = beta;
            }
            return nullValue;
        }
    }
    
    // Generate moves
    MoveList moves;
    generate<LEGAL>(pos, moves);
    
    if (moves.empty()) {
        return inCheck ? mated_in(ply) : VALUE_DRAW;
    }
    
    // Score moves
    score_moves(pos, moves, ttMove, ply);

    bool restrictRootMoves = (ply == 0 && !limits.searchmoves.empty());

    Move bestMoveLocal = MOVE_NONE;
    Value bestValue = -VALUE_INFINITE;
    int moveCount = 0;
    Value origAlpha = alpha;

    for (int i = 0; i < moves.size(); i++) {
        pick_move(moves, i);
        Move m = moves[i].move;

        if (restrictRootMoves) {
            bool found = false;
            for (Move sm : limits.searchmoves) {
                if (sm == m) { found = true; break; }
            }
            if (!found) continue;
        }
        
        moveCount++;
        
        bool isCapture = pos.capture_or_promotion(m);
        bool isQuiet = !isCapture && m.type() != PROMOTION;
        bool givesCheck = pos.gives_check(m);
        
        // Late move pruning (more aggressive for self-play throughput)
        if (!inCheck && isQuiet && depth <= 4 && moveCount > (improving ? 5 : 2) + depth * 2) {
            continue;
        }

        // Futility pruning for quiet moves
        if (!inCheck && isQuiet && depth <= 7 && moveCount > 1 &&
            eval + FutilityMargin[std::min(depth, 3)] + 260 * depth <= alpha) {
            continue;
        }
        
        // SEE pruning for quiet moves
        if (!inCheck && isQuiet && depth <= 4 && !pos.see_ge(m, -20 * depth * depth)) {
            continue;
        }
        
        // Make move
        pos.do_move(m, stateStack[ply], givesCheck);
        
        Value value;
        int newDepth = depth - 1;
        
        // Late Move Reductions
        if (depth >= 3 && moveCount > 1 && isQuiet) {
            int R = LMRTable[std::min(depth, MAX_PLY - 1)][std::min(moveCount, MAX_MOVES - 1)];

            // Reduce less for killers
            if (m == killers[ply][0] || m == killers[ply][1]) R--;

            // Reduce more if not improving
            if (!improving) R += 1;

            // Reduce more for cut nodes
            if (cutNode) R += 1;

            // Extra reduction for very late quiets
            if (moveCount > 8 && depth >= 4) R += 1;

            R = std::max(0, std::min(R, newDepth - 1));
            
            // Reduced depth search
            value = -search(pos, -alpha - 1, -alpha, newDepth - R, ply + 1, true);
            
            // Re-search if failed high
            if (value > alpha && R > 0) {
                value = -search(pos, -alpha - 1, -alpha, newDepth, ply + 1, !cutNode);
            }
        } else if (moveCount > 1) {
            // Zero-window search
            value = -search(pos, -alpha - 1, -alpha, newDepth, ply + 1, !cutNode);
        } else {
            value = alpha + 1;  // Force full search
        }
        
        // Full window search
        if (value > alpha) {
            value = -search(pos, -beta, -alpha, newDepth, ply + 1, false);
        }
        
        pos.undo_move(m);
        
        if (stopped.load(std::memory_order_relaxed)) return VALUE_ZERO;
        
        // Update best value
        if (value > bestValue) {
            bestValue = value;
            
            if (value > alpha) {
                bestMoveLocal = m;
                
                // Update PV
                pvTable[ply][ply] = m;
                for (int j = ply + 1; j < pvLength[ply + 1]; j++) {
                    pvTable[ply][j] = pvTable[ply + 1][j];
                }
                pvLength[ply] = pvLength[ply + 1];
                
                if (value >= beta) {
                    // Fail high
                    
                    // Update killers
                    if (!isCapture) {
                        if (killers[ply][0] != m) {
                            killers[ply][1] = killers[ply][0];
                            killers[ply][0] = m;
                        }
                        
                        // Update history
                        int bonus = std::min(depth * depth, 400);
                        Color us = pos.side_to_move();
                        history[us][m.from()][m.to()] += bonus - history[us][m.from()][m.to()] * bonus / 16384;
                        
                        // Penalty for earlier quiet moves
                        for (int j = 0; j < i; j++) {
                            Move prev = moves[j].move;
                            if (!pos.capture(prev)) {
                                history[us][prev.from()][prev.to()] -= bonus - history[us][prev.from()][prev.to()] * bonus / 16384;
                            }
                        }
                    }
                    
                    break;
                }
                
                alpha = value;
            }
        }
    }
    
    // Store in TT
    Bound bound = bestValue >= beta ? BOUND_LOWER :
                  bestValue > origAlpha ? BOUND_EXACT : BOUND_UPPER;
    
    tte->save(pos.key(), bestValue, bound, depth, bestMoveLocal, staticEval[ply], TT.current_generation());
    
    return bestValue;
}

// =============================================================================
// QUIESCENCE SEARCH
// =============================================================================

Value Search::qsearch(Position& pos, Value alpha, Value beta, int ply) {
    nodes++;
    selDepth = std::max(selDepth, ply);
    
    // Check for stop
    if (stopped.load(std::memory_order_relaxed)) return VALUE_ZERO;
    
    // Max ply check
    if (ply >= MAX_PLY - 1) {
        return Eval::evaluate(pos);
    }
    
    // Draw detection
    if (pos.is_draw(ply)) {
        return VALUE_DRAW;
    }
    
    bool inCheck = pos.is_check();
    Value bestValue;
    
    if (inCheck) {
        bestValue = -VALUE_INFINITE;
    } else {
        bestValue = Eval::evaluate(pos);
        
        if (bestValue >= beta) {
            return bestValue;
        }
        
        // Delta pruning
        if (bestValue + QueenValueMg + DeltaMargin < alpha) {
            return alpha;
        }
        
        if (bestValue > alpha) {
            alpha = bestValue;
        }
    }
    
    // Generate moves
    MoveList moves;
    if (inCheck) {
        generate<EVASIONS>(pos, moves);
    } else {
        generate<CAPTURES>(pos, moves);
    }
    
    if (moves.empty()) {
        return inCheck ? mated_in(ply) : bestValue;
    }
    
    // Score captures by MVV-LVA
    for (int i = 0; i < moves.size(); i++) {
        Move m = moves[i].move;
        Piece captured = pos.piece_on(m.to());
        if (m.type() == EN_PASSANT) {
            captured = make_piece(~pos.side_to_move(), PAWN);
        }
        Piece moving = pos.moved_piece(m);
        moves[i].score = PieceValue[type_of(captured)] * 10 - PieceValue[type_of(moving)];
    }
    
    StateInfo st;
    
    for (int i = 0; i < moves.size(); i++) {
        pick_move(moves, i);
        Move m = moves[i].move;
        
        // SEE pruning (skip bad captures when not in check)
        if (!inCheck && !pos.see_ge(m, 0)) {
            continue;
        }
        
        // Delta pruning for individual captures
        if (!inCheck && m.type() != PROMOTION) {
            Piece captured = pos.piece_on(m.to());
            if (m.type() == EN_PASSANT) {
                captured = make_piece(~pos.side_to_move(), PAWN);
            }
            if (bestValue + PieceValue[type_of(captured)] + DeltaMargin < alpha) {
                continue;
            }
        }
        
        pos.do_move(m, st);
        Value value = -qsearch(pos, -beta, -alpha, ply + 1);
        pos.undo_move(m);
        
        if (stopped.load(std::memory_order_relaxed)) return VALUE_ZERO;
        
        if (value > bestValue) {
            bestValue = value;
            
            if (value > alpha) {
                if (value >= beta) {
                    return value;
                }
                alpha = value;
            }
        }
    }
    
    return bestValue;
}
