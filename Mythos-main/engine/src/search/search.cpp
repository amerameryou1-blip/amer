#include "mythos/search/search.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <ranges>

namespace mythos::search {

namespace {

constexpr int kScoreTtMove = 10'000'000;
constexpr int kScoreCapture = 8'000'000;
constexpr int kScoreKiller1 = 6'000'000;
constexpr int kScoreKiller2 = 5'000'000;
constexpr std::array<int, 4> kFutilityMargin{0, 300, 500, 800};
constexpr int kRazoringMargin = 400;
constexpr int kNullMoveBaseReduction = 2;
constexpr int kDeltaMargin = 200;

using LmrTable = std::array<std::array<int, MAX_MOVES>, MAX_PLY>;

[[nodiscard]] const LmrTable& lmr_table() {
    static const LmrTable table = [] {
        LmrTable value{};
        for (int depth = 1; depth < MAX_PLY; ++depth) {
            for (int move = 1; move < MAX_MOVES; ++move) {
                value[depth][move] = static_cast<int>(1.15 + std::log(static_cast<double>(depth)) *
                                                                 std::log(static_cast<double>(move)) / 1.90);
            }
        }
        return value;
    }();
    return table;
}

[[nodiscard]] int elapsed_ms_since(const std::chrono::steady_clock::time_point& start) {
    return static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
}

class SearchWorker final {
public:
    SearchWorker(
        int worker_id,
        TranspositionTable& tt,
        const eval::Evaluator& evaluator,
        std::atomic<bool>& stop,
        std::atomic<std::uint64_t>& global_nodes,
        const InfoCallback* info_callback) noexcept
        : worker_id_(worker_id),
          tt_(tt),
          evaluator_(evaluator),
          stop_(stop),
          global_nodes_(global_nodes),
          info_callback_(info_callback) {
        std::memset(killers_, 0, sizeof(killers_));
        std::memset(history_, 0, sizeof(history_));
        std::memset(static_eval_, 0, sizeof(static_eval_));
        std::memset(pv_table_, 0, sizeof(pv_table_));
        std::memset(pv_length_, 0, sizeof(pv_length_));
    }

    [[nodiscard]] SearchInfo run(Position& root, const SearchLimits& limits) {
        limits_ = limits;
        start_time_ = std::chrono::steady_clock::now();
        nodes_ = 0;
        sel_depth_ = 0;
        soft_time_ms_ = allocate_soft_time(root.side_to_move());
        hard_time_ms_ = allocate_hard_time(root.side_to_move(), soft_time_ms_);

        MoveList root_moves;
        generate<LEGAL>(root, root_moves);
        if (root_moves.empty()) {
            SearchInfo info;
            info.stopped = true;
            return info;
        }

        const int rotation = worker_id_ > 0 ? worker_id_ % std::max(root_moves.size(), 1) : 0;
        if (rotation > 0) {
            std::rotate(root_moves.begin(), root_moves.begin() + rotation, root_moves.end());
        }

        SearchInfo best_info;
        best_info.best_move = root_moves[0].move;
        best_info.depth = 0;

        Value previous_score = VALUE_ZERO;
        for (int depth = 1; depth <= limits_.depth && !stop_.load(std::memory_order_relaxed); ++depth) {
            pv_length_[0] = 0;
            sel_depth_ = 0;

            int aspiration = depth >= 5 ? 24 + worker_id_ * 8 : VALUE_INFINITE;
            Value alpha = depth >= 5 ? std::max(previous_score - aspiration, -VALUE_INFINITE) : -VALUE_INFINITE;
            Value beta = depth >= 5 ? std::min(previous_score + aspiration, VALUE_INFINITE) : VALUE_INFINITE;

            Value score = VALUE_ZERO;
            while (!stop_.load(std::memory_order_relaxed)) {
                score = search(root, alpha, beta, depth, 0, false);
                if (stop_.load(std::memory_order_relaxed)) {
                    break;
                }
                if (score <= alpha) {
                    alpha = std::max(score - aspiration, -VALUE_INFINITE);
                    aspiration *= 2;
                    continue;
                }
                if (score >= beta) {
                    beta = std::min(score + aspiration, VALUE_INFINITE);
                    aspiration *= 2;
                    continue;
                }
                break;
            }

            if (stop_.load(std::memory_order_relaxed)) {
                break;
            }

            previous_score = score;
            best_info = make_info(depth, score);
            publish(best_info);

            if (should_stop_after_depth(best_info.elapsed_ms)) {
                break;
            }
        }

        best_info.stopped = stop_.load(std::memory_order_relaxed);
        return best_info;
    }

private:
    [[nodiscard]] int allocate_soft_time(Color us) const noexcept {
        if (limits_.movetime > 0) {
            return std::max(1, limits_.movetime - 20);
        }
        if (limits_.time[us] <= 0 || limits_.infinite) {
            return std::numeric_limits<int>::max();
        }
        const int moves = limits_.movestogo > 0 ? limits_.movestogo : 40;
        const int base = limits_.time[us] / std::max(moves, 1);
        const int increment = limits_.inc[us] / 2;
        return std::max(30, std::min(base + increment, limits_.time[us] / 3));
    }

    [[nodiscard]] int allocate_hard_time(Color us, int soft_time) const noexcept {
        if (limits_.movetime > 0) {
            return std::max(1, limits_.movetime - 5);
        }
        if (limits_.time[us] <= 0 || limits_.infinite) {
            return std::numeric_limits<int>::max();
        }
        return std::max(soft_time, std::min(soft_time * 2, limits_.time[us] - 25));
    }

    [[nodiscard]] bool should_stop_after_depth(int elapsed_ms) const noexcept {
        if (limits_.nodes > 0 &&
            static_cast<std::int64_t>(global_nodes_.load(std::memory_order_relaxed)) >= limits_.nodes) {
            return true;
        }
        if (limits_.infinite) {
            return false;
        }
        return elapsed_ms >= soft_time_ms_;
    }

    void publish(const SearchInfo& info) const {
        if (worker_id_ == 0 && info_callback_ != nullptr && *info_callback_) {
            (*info_callback_)(info);
        }
    }

    [[nodiscard]] SearchInfo make_info(int depth, Value score) const noexcept {
        SearchInfo info;
        info.depth = depth;
        info.seldepth = sel_depth_;
        info.score = score;
        info.nodes = nodes_;
        info.elapsed_ms = elapsed_ms_since(start_time_);
        info.nps = info.elapsed_ms > 0
                       ? static_cast<std::int64_t>(nodes_) * 1000 / info.elapsed_ms
                       : static_cast<std::int64_t>(nodes_);
        info.hashfull = tt_.hashfull();
        info.pv_length = pv_length_[0];
        for (int i = 0; i < pv_length_[0]; ++i) {
            info.pv[i] = pv_table_[0][i];
        }
        info.best_move = pv_length_[0] > 0 ? pv_table_[0][0] : MOVE_NONE;
        info.ponder_move = pv_length_[0] > 1 ? pv_table_[0][1] : MOVE_NONE;
        return info;
    }

    void check_time() noexcept {
        if ((nodes_ & 1023ULL) != 0ULL) {
            return;
        }
        if (limits_.nodes > 0 &&
            static_cast<std::int64_t>(global_nodes_.load(std::memory_order_relaxed)) >= limits_.nodes) {
            stop_.store(true, std::memory_order_relaxed);
            return;
        }
        if (limits_.infinite) {
            return;
        }
        if (elapsed_ms_since(start_time_) >= hard_time_ms_) {
            stop_.store(true, std::memory_order_relaxed);
        }
    }

    void count_node() noexcept {
        ++nodes_;
        global_nodes_.fetch_add(1, std::memory_order_relaxed);
        if ((nodes_ & 127ULL) == 0ULL) {
            check_time();
        }
    }

    void score_moves(Position& pos, MoveList& moves, Move tt_move, int ply) const noexcept {
        const Color us = pos.side_to_move();
        for (int i = 0; i < moves.size(); ++i) {
            const Move move = moves[i].move;
            if (move == tt_move) {
                moves[i].score = kScoreTtMove;
                continue;
            }
            if (pos.capture(move)) {
                Piece captured = pos.piece_on(move.to());
                if (move.type() == EN_PASSANT) {
                    captured = make_piece(~us, PAWN);
                }
                const Piece mover = pos.moved_piece(move);
                moves[i].score = kScoreCapture + PieceValue[type_of(captured)] * 16 - PieceValue[type_of(mover)];
                if (!pos.see_ge(move, 0)) {
                    moves[i].score -= kScoreCapture;
                }
                continue;
            }
            if (move == killers_[ply][0]) {
                moves[i].score = kScoreKiller1;
                continue;
            }
            if (move == killers_[ply][1]) {
                moves[i].score = kScoreKiller2;
                continue;
            }
            moves[i].score = history_[us][move.from()][move.to()];
        }
    }

    static void pick_move(MoveList& moves, int index) noexcept {
        int best_index = index;
        int best_score = moves[index].score;
        for (int i = index + 1; i < moves.size(); ++i) {
            if (moves[i].score > best_score) {
                best_score = moves[i].score;
                best_index = i;
            }
        }
        if (best_index != index) {
            std::swap(moves[index], moves[best_index]);
        }
    }

    [[nodiscard]] Value search(Position& pos, Value alpha, Value beta, int depth, int ply, bool cut_node) {
        if (stop_.load(std::memory_order_relaxed)) {
            return VALUE_ZERO;
        }

        check_time();
        if (stop_.load(std::memory_order_relaxed)) {
            return VALUE_ZERO;
        }

        pv_length_[ply] = ply;
        count_node();
        sel_depth_ = std::max(sel_depth_, ply);

        const bool in_check = pos.is_check();
        if (in_check) {
            ++depth;
        }

        if (depth <= 0) {
            return qsearch(pos, alpha, beta, ply);
        }

        if (ply > 0) {
            if (pos.is_draw(ply)) {
                return VALUE_DRAW;
            }
            if (ply >= MAX_PLY - 1) {
                return evaluator_.evaluate(pos);
            }
        }

        alpha = std::max(alpha, mated_in(ply));
        beta = std::min(beta, mate_in(ply + 1));
        if (alpha >= beta) {
            return alpha;
        }

        const auto tt_entry = tt_.probe(pos.key());
        const Move tt_move = tt_entry.hit ? tt_entry.move : MOVE_NONE;
        const Value tt_value = tt_entry.hit ? tt_entry.score : VALUE_NONE;

        if (tt_entry.hit && ply > 0 && tt_entry.depth >= depth) {
            if (tt_entry.bound == BOUND_EXACT) {
                return tt_value;
            }
            if (tt_entry.bound == BOUND_LOWER && tt_value >= beta) {
                return tt_value;
            }
            if (tt_entry.bound == BOUND_UPPER && tt_value <= alpha) {
                return tt_value;
            }
        }

        Value eval = VALUE_NONE;
        if (in_check) {
            static_eval_[ply] = VALUE_NONE;
        } else if (tt_entry.hit && tt_entry.static_eval != VALUE_NONE) {
            eval = static_eval_[ply] = tt_entry.static_eval;
        } else {
            eval = static_eval_[ply] = evaluator_.evaluate(pos);
        }

        const bool improving = !in_check && ply >= 2 && static_eval_[ply - 2] != VALUE_NONE &&
                               static_eval_[ply] != VALUE_NONE && static_eval_[ply] > static_eval_[ply - 2];

        if (!in_check && depth <= 3 && eval != VALUE_NONE && eval + kRazoringMargin < alpha) {
            const Value razor = qsearch(pos, alpha, beta, ply);
            if (razor <= alpha) {
                return razor;
            }
        }

        if (!in_check && depth <= 3 && eval != VALUE_NONE &&
            eval >= beta + kFutilityMargin[std::min(depth, 3)] &&
            pos.non_pawn_material(pos.side_to_move()) > 0) {
            return eval;
        }

        if (!in_check && ply > 0 && depth >= kNullMoveBaseReduction &&
            eval != VALUE_NONE && eval >= beta &&
            pos.non_pawn_material(pos.side_to_move()) > 0 &&
            static_eval_[std::max(0, ply - 1)] != VALUE_NONE) {
            const int reduction = 3 + depth / 3 + std::min((eval - beta) / 200, 3);
            pos.do_null_move(state_stack_[ply]);
            const Value value = -search(pos, -beta, -beta + 1, depth - reduction - 1, ply + 1, !cut_node);
            pos.undo_null_move();
            if (stop_.load(std::memory_order_relaxed)) {
                return VALUE_ZERO;
            }
            if (value >= beta) {
                return value >= VALUE_TB_WIN_IN_MAX_PLY ? beta : value;
            }
        }

        MoveList moves;
        generate<LEGAL>(pos, moves);
        if (moves.empty()) {
            return in_check ? mated_in(ply) : VALUE_DRAW;
        }
        if (ply == 0 && worker_id_ > 0 && moves.size() > 1) {
            const int rotation = worker_id_ % moves.size();
            std::rotate(moves.begin(), moves.begin() + rotation, moves.end());
        }

        score_moves(pos, moves, tt_move, ply);
        const bool restrict_root_moves = (ply == 0 && !limits_.searchmoves.empty());

        Move best_move = MOVE_NONE;
        Value best_value = -VALUE_INFINITE;
        const Value original_alpha = alpha;
        int move_count = 0;

        for (int i = 0; i < moves.size(); ++i) {
            pick_move(moves, i);
            const Move move = moves[i].move;

            if (restrict_root_moves && std::ranges::find(limits_.searchmoves, move) == limits_.searchmoves.end()) {
                continue;
            }

            ++move_count;

            const bool is_capture = pos.capture_or_promotion(move);
            const bool is_quiet = !is_capture && move.type() != PROMOTION;
            const bool gives_check = pos.gives_check(move);

            if (!in_check && is_quiet && depth <= 4 && move_count > (improving ? 5 : 2) + depth * 2) {
                continue;
            }
            if (!in_check && is_quiet && depth <= 7 && move_count > 1 &&
                eval != VALUE_NONE &&
                eval + kFutilityMargin[std::min(depth, 3)] + 260 * depth <= alpha) {
                continue;
            }
            if (!in_check && is_quiet && depth <= 4 && !pos.see_ge(move, -20 * depth * depth)) {
                continue;
            }

            pos.do_move(move, state_stack_[ply], gives_check);
            Value value = VALUE_ZERO;
            const int new_depth = depth - 1;

            if (depth >= 3 && move_count > 1 && is_quiet) {
                int reduction = lmr_table()[std::min(depth, MAX_PLY - 1)][std::min(move_count, MAX_MOVES - 1)];
                if (move == killers_[ply][0] || move == killers_[ply][1]) {
                    --reduction;
                }
                if (!improving) {
                    ++reduction;
                }
                if (cut_node) {
                    ++reduction;
                }
                if (move_count > 8 && depth >= 4) {
                    ++reduction;
                }
                reduction = std::max(0, std::min(reduction, new_depth - 1));
                value = -search(pos, -alpha - 1, -alpha, new_depth - reduction, ply + 1, true);
                if (value > alpha && reduction > 0) {
                    value = -search(pos, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
                }
            } else if (move_count > 1) {
                value = -search(pos, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
            } else {
                value = alpha + 1;
            }

            if (value > alpha) {
                value = -search(pos, -beta, -alpha, new_depth, ply + 1, false);
            }

            pos.undo_move(move);

            if (stop_.load(std::memory_order_relaxed)) {
                return VALUE_ZERO;
            }

            if (value > best_value) {
                best_value = value;
                if (value > alpha) {
                    best_move = move;
                    pv_table_[ply][ply] = move;
                    for (int j = ply + 1; j < pv_length_[ply + 1]; ++j) {
                        pv_table_[ply][j] = pv_table_[ply + 1][j];
                    }
                    pv_length_[ply] = pv_length_[ply + 1];

                    if (value >= beta) {
                        if (!is_capture) {
                            if (killers_[ply][0] != move) {
                                killers_[ply][1] = killers_[ply][0];
                                killers_[ply][0] = move;
                            }
                            const int bonus = std::min(depth * depth, 512);
                            const Color us = pos.side_to_move();
                            auto& table = history_[us][move.from()][move.to()];
                            table += bonus - table * bonus / 16384;
                            for (int j = 0; j < i; ++j) {
                                const Move previous = moves[j].move;
                                if (!pos.capture(previous)) {
                                    auto& penalty = history_[us][previous.from()][previous.to()];
                                    penalty -= bonus - penalty * bonus / 16384;
                                }
                            }
                        }
                        break;
                    }
                    alpha = value;
                }
            }
        }

        const Bound bound = best_value >= beta ? BOUND_LOWER :
                            best_value > original_alpha ? BOUND_EXACT : BOUND_UPPER;
        tt_.store(pos.key(), best_move, best_value, static_eval_[ply], depth, bound);
        return best_value;
    }

    [[nodiscard]] Value qsearch(Position& pos, Value alpha, Value beta, int ply) {
        if (stop_.load(std::memory_order_relaxed)) {
            return VALUE_ZERO;
        }

        count_node();
        sel_depth_ = std::max(sel_depth_, ply);

        if (ply >= MAX_PLY - 1) {
            return evaluator_.evaluate(pos);
        }
        if (pos.is_draw(ply)) {
            return VALUE_DRAW;
        }

        const bool in_check = pos.is_check();
        Value best_value = -VALUE_INFINITE;
        if (!in_check) {
            best_value = evaluator_.evaluate(pos);
            if (best_value >= beta) {
                return best_value;
            }
            if (best_value + QueenValueMg + kDeltaMargin < alpha) {
                return alpha;
            }
            alpha = std::max(alpha, best_value);
        }

        MoveList moves;
        if (in_check) {
            generate<EVASIONS>(pos, moves);
        } else {
            generate<CAPTURES>(pos, moves);
        }

        if (moves.empty()) {
            return in_check ? mated_in(ply) : best_value;
        }

        for (int i = 0; i < moves.size(); ++i) {
            Piece captured = pos.piece_on(moves[i].move.to());
            if (moves[i].move.type() == EN_PASSANT) {
                captured = make_piece(~pos.side_to_move(), PAWN);
            }
            const Piece moving = pos.moved_piece(moves[i].move);
            moves[i].score = PieceValue[type_of(captured)] * 16 - PieceValue[type_of(moving)];
        }

        for (int i = 0; i < moves.size(); ++i) {
            pick_move(moves, i);
            const Move move = moves[i].move;
            if (!in_check && !pos.see_ge(move, 0)) {
                continue;
            }
            if (!in_check && move.type() != PROMOTION) {
                Piece captured = pos.piece_on(move.to());
                if (move.type() == EN_PASSANT) {
                    captured = make_piece(~pos.side_to_move(), PAWN);
                }
                if (best_value + PieceValue[type_of(captured)] + kDeltaMargin < alpha) {
                    continue;
                }
            }

            pos.do_move(move, state_stack_[ply]);
            const Value value = -qsearch(pos, -beta, -alpha, ply + 1);
            pos.undo_move(move);

            if (stop_.load(std::memory_order_relaxed)) {
                return VALUE_ZERO;
            }
            if (value > best_value) {
                best_value = value;
                if (value > alpha) {
                    if (value >= beta) {
                        return value;
                    }
                    alpha = value;
                }
            }
        }

        return best_value;
    }

    int worker_id_;
    TranspositionTable& tt_;
    const eval::Evaluator& evaluator_;
    std::atomic<bool>& stop_;
    std::atomic<std::uint64_t>& global_nodes_;
    const InfoCallback* info_callback_;
    SearchLimits limits_{};
    std::chrono::steady_clock::time_point start_time_{};
    std::uint64_t nodes_ = 0;
    int sel_depth_ = 0;
    int soft_time_ms_ = std::numeric_limits<int>::max();
    int hard_time_ms_ = std::numeric_limits<int>::max();
    StateInfo state_stack_[MAX_PLY + 16]{};
    Move killers_[MAX_PLY][2]{};
    int history_[COLOR_NB][SQUARE_NB][SQUARE_NB]{};
    Value static_eval_[MAX_PLY]{};
    Move pv_table_[MAX_PLY][MAX_PLY]{};
    int pv_length_[MAX_PLY]{};
};

}  // namespace

void SearchLimits::clear() noexcept {
    depth = MAX_PLY;
    nodes = 0;
    movetime = 0;
    time = {0, 0};
    inc = {0, 0};
    movestogo = 0;
    infinite = false;
    ponder = false;
    searchmoves.clear();
}

TranspositionTable::TranspositionTable(std::size_t megabytes) {
    resize(megabytes);
}

void TranspositionTable::resize(std::size_t megabytes) {
    const std::size_t bytes = std::max<std::size_t>(megabytes, 1U) * 1024ULL * 1024ULL;
    cluster_count_ = std::bit_floor(std::max<std::size_t>(bytes / sizeof(Cluster), 1024ULL));
    clusters_ = std::make_unique<Cluster[]>(cluster_count_);
    cluster_mask_ = cluster_count_ - 1;
    clear();
}

void TranspositionTable::clear() {
    for (std::size_t i = 0; i < cluster_count_; ++i) {
        for (auto& slot : clusters_[i].slots) {
            slot.key.store(0ULL, std::memory_order_relaxed);
            slot.data.store(0ULL, std::memory_order_relaxed);
        }
    }
    generation_.store(0, std::memory_order_relaxed);
}

void TranspositionTable::new_search() noexcept {
    generation_.store(static_cast<std::uint8_t>((generation_.load(std::memory_order_relaxed) + 1U) & 0x3FU),
                      std::memory_order_relaxed);
}

int TranspositionTable::hashfull() const {
    const std::size_t sample = std::min<std::size_t>(cluster_count_, 1000ULL);
    int used = 0;
    const std::uint8_t age = generation_.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < sample; ++i) {
        for (const auto& slot : clusters_[i].slots) {
            const auto key = slot.key.load(std::memory_order_relaxed);
            if (key == 0ULL) {
                continue;
            }
            const auto entry = unpack(key, slot.data.load(std::memory_order_relaxed));
            if (entry.age == age) {
                ++used;
            }
        }
    }
    return used;
}

std::uint64_t TranspositionTable::pack(
    Move move,
    Value score,
    Value static_eval,
    int depth,
    Bound bound,
    std::uint8_t age) noexcept {
    const auto move_bits = static_cast<std::uint64_t>(move.raw());
    const auto score_bits = static_cast<std::uint64_t>(static_cast<std::uint16_t>(score));
    const auto eval_bits = static_cast<std::uint64_t>(static_cast<std::uint16_t>(static_eval));
    const auto depth_bits = static_cast<std::uint64_t>(static_cast<std::uint8_t>(std::clamp(depth, 0, 255)));
    const auto bound_bits = static_cast<std::uint64_t>(static_cast<std::uint8_t>(bound) & 0x3U);
    const auto age_bits = static_cast<std::uint64_t>(age & 0x3FU);
    return move_bits |
           (score_bits << 16U) |
           (eval_bits << 32U) |
           (depth_bits << 48U) |
           (bound_bits << 56U) |
           (age_bits << 58U);
}

TranspositionTable::Entry TranspositionTable::unpack(Key key, std::uint64_t data) noexcept {
    Entry entry;
    entry.key = key;
    entry.move = Move(static_cast<std::uint16_t>(data & 0xFFFFU));
    entry.score = static_cast<std::int16_t>((data >> 16U) & 0xFFFFU);
    entry.static_eval = static_cast<std::int16_t>((data >> 32U) & 0xFFFFU);
    entry.depth = static_cast<std::uint8_t>((data >> 48U) & 0xFFU);
    entry.bound = static_cast<Bound>((data >> 56U) & 0x3U);
    entry.age = static_cast<std::uint8_t>((data >> 58U) & 0x3FU);
    entry.hit = key != 0ULL;
    return entry;
}

Value TranspositionTable::score_from_tt(Value value, int ply) noexcept {
    if (value >= VALUE_TB_WIN_IN_MAX_PLY) {
        return value - ply;
    }
    if (value <= VALUE_TB_LOSS_IN_MAX_PLY) {
        return value + ply;
    }
    return value;
}

Value TranspositionTable::score_to_tt(Value value, int ply) noexcept {
    if (value >= VALUE_TB_WIN_IN_MAX_PLY) {
        return value + ply;
    }
    if (value <= VALUE_TB_LOSS_IN_MAX_PLY) {
        return value - ply;
    }
    return value;
}

TranspositionTable::Cluster& TranspositionTable::cluster_for(Key key) noexcept {
    return clusters_[key & cluster_mask_];
}

const TranspositionTable::Cluster& TranspositionTable::cluster_for(Key key) const noexcept {
    return clusters_[key & cluster_mask_];
}

std::size_t TranspositionTable::replacement_index(const Cluster& cluster, Key key) const noexcept {
    const std::uint8_t current_age = generation_.load(std::memory_order_relaxed);
    std::size_t best_index = 0;
    int best_score = std::numeric_limits<int>::max();

    for (std::size_t i = 0; i < cluster.slots.size(); ++i) {
        const auto stored_key = cluster.slots[i].key.load(std::memory_order_relaxed);
        if (stored_key == 0ULL || stored_key == key) {
            return i;
        }
        const auto entry = unpack(stored_key, cluster.slots[i].data.load(std::memory_order_relaxed));
        const int age_penalty = (current_age - entry.age) & 0x3F;
        const int score = entry.depth - age_penalty * 4;
        if (score < best_score) {
            best_score = score;
            best_index = i;
        }
    }
    return best_index;
}

TranspositionTable::Entry TranspositionTable::probe(Key key) const noexcept {
    const auto& cluster = cluster_for(key);
    for (const auto& slot : cluster.slots) {
        const auto stored_key = slot.key.load(std::memory_order_acquire);
        if (stored_key == key) {
            return unpack(stored_key, slot.data.load(std::memory_order_acquire));
        }
    }
    return {};
}

void TranspositionTable::store(Key key, Move move, Value score, Value static_eval, int depth, Bound bound) noexcept {
    auto& cluster = cluster_for(key);
    const auto index = replacement_index(cluster, key);
    const auto packed = pack(move, score_to_tt(score, 0), static_eval, depth, bound,
                             generation_.load(std::memory_order_relaxed));
    cluster.slots[index].data.store(packed, std::memory_order_release);
    cluster.slots[index].key.store(key, std::memory_order_release);
}

SearchController::SearchController() : tt_(32U) {
    const unsigned hw = std::thread::hardware_concurrency();
    thread_count_ = hw == 0U ? 1 : static_cast<int>(hw);
}

SearchController::~SearchController() {
    stop();
    wait();
}

void SearchController::set_threads(int threads) noexcept {
    thread_count_ = std::max(1, threads);
}

void SearchController::set_hash_mb(std::size_t hash_mb) {
    hash_mb_ = std::max<std::size_t>(1U, hash_mb);
    tt_.resize(hash_mb_);
}

int SearchController::thread_count() const noexcept {
    return thread_count_;
}

std::size_t SearchController::hash_mb() const noexcept {
    return hash_mb_;
}

bool SearchController::load_weights(const std::filesystem::path& path, std::string* error) {
    return evaluator_.load_weights(path, error);
}

void SearchController::clear_weights() noexcept {
    evaluator_.clear_weights();
}

int SearchController::evaluate(const Position& pos) const noexcept {
    return evaluator_.evaluate(pos);
}

void SearchController::clear() {
    tt_.clear();
}

void SearchController::new_game() {
    clear();
}

void SearchController::go(
    const Position& root,
    const SearchLimits& limits,
    InfoCallback info_callback,
    BestMoveCallback best_callback) {
    stop();
    wait();

    const std::string root_fen = root.fen();

    std::scoped_lock lock(launch_mutex_);
    stop_.store(false, std::memory_order_relaxed);
    searching_.store(true, std::memory_order_relaxed);
    global_nodes_.store(0ULL, std::memory_order_relaxed);

    coordinator_ = std::thread([this, root_fen, limits, info_callback = std::move(info_callback),
                                best_callback = std::move(best_callback)]() mutable {
        tt_.new_search();

        std::vector<SearchInfo> results(static_cast<std::size_t>(thread_count_));
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(thread_count_));

        for (int worker_id = 0; worker_id < thread_count_; ++worker_id) {
            workers.emplace_back([&, worker_id] {
                results[static_cast<std::size_t>(worker_id)] =
                    run_worker(worker_id, root_fen, limits, worker_id == 0 ? info_callback : InfoCallback{});
            });
        }

        for (auto& thread : workers) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        Position fallback_position;
        StateInfo fallback_state{};
        fallback_position.set(root_fen, &fallback_state);
        MoveList legal_moves;
        generate<LEGAL>(fallback_position, legal_moves);

        SearchInfo best{};
        bool have_best = false;
        for (const auto& info : results) {
            if (!info.best_move && legal_moves.empty()) {
                continue;
            }
            if (!have_best || info.depth > best.depth ||
                (info.depth == best.depth && info.score > best.score)) {
                best = info;
                have_best = true;
            }
        }

        if (!best.best_move && !legal_moves.empty()) {
            best.best_move = legal_moves[0].move;
            if (legal_moves.size() > 1) {
                best.ponder_move = legal_moves[1].move;
            }
        }

        if (best_callback) {
            best_callback(best.best_move, best.ponder_move);
        }

        searching_.store(false, std::memory_order_relaxed);
    });
}

void SearchController::stop() noexcept {
    stop_.store(true, std::memory_order_relaxed);
}

void SearchController::wait() {
    std::scoped_lock lock(launch_mutex_);
    if (coordinator_.joinable()) {
        coordinator_.join();
    }
    searching_.store(false, std::memory_order_relaxed);
}

bool SearchController::searching() const noexcept {
    return searching_.load(std::memory_order_relaxed);
}

SearchInfo SearchController::run_worker(
    int worker_id,
    const std::string& root_fen,
    const SearchLimits& limits,
    const InfoCallback& info_callback) {
    Position local_position;
    StateInfo local_state{};
    local_position.set(root_fen, &local_state);
    SearchWorker worker(worker_id, tt_, evaluator_, stop_, global_nodes_, &info_callback);
    return worker.run(local_position, limits);
}

}  // namespace mythos::search
