#pragma once

#include "../eval/evaluator.hpp"
#include "../movegen/movegen.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mythos::search {

struct SearchLimits final {
    int depth = MAX_PLY;
    std::int64_t nodes = 0;
    int movetime = 0;
    std::array<int, COLOR_NB> time{0, 0};
    std::array<int, COLOR_NB> inc{0, 0};
    int movestogo = 0;
    bool infinite = false;
    bool ponder = false;
    std::vector<Move> searchmoves;

    void clear() noexcept;
};

struct SearchInfo final {
    Move best_move = MOVE_NONE;
    Move ponder_move = MOVE_NONE;
    std::array<Move, MAX_PLY> pv{};
    int pv_length = 0;
    int depth = 0;
    int seldepth = 0;
    Value score = VALUE_ZERO;
    std::uint64_t nodes = 0;
    std::int64_t nps = 0;
    int elapsed_ms = 0;
    int hashfull = 0;
    bool stopped = false;
};

class TranspositionTable final {
public:
    explicit TranspositionTable(std::size_t megabytes = 32U);

    void resize(std::size_t megabytes);
    void clear();
    void new_search() noexcept;
    [[nodiscard]] int hashfull() const;

    struct Entry final {
        Key key = 0;
        Move move = MOVE_NONE;
        Value score = VALUE_NONE;
        Value static_eval = VALUE_NONE;
        int depth = 0;
        Bound bound = BOUND_NONE;
        std::uint8_t age = 0;
        bool hit = false;
    };

    [[nodiscard]] Entry probe(Key key) const noexcept;
    void store(Key key, Move move, Value score, Value static_eval, int depth, Bound bound) noexcept;

private:
    struct Slot final {
        std::atomic<std::uint64_t> key{0};
        std::atomic<std::uint64_t> data{0};
    };

    struct Cluster final {
        std::array<Slot, 4> slots{};
    };

    [[nodiscard]] static std::uint64_t pack(Move move, Value score, Value static_eval, int depth, Bound bound, std::uint8_t age) noexcept;
    [[nodiscard]] static Entry unpack(Key key, std::uint64_t data) noexcept;
    [[nodiscard]] static Value score_from_tt(Value value, int ply) noexcept;
    [[nodiscard]] static Value score_to_tt(Value value, int ply) noexcept;

    [[nodiscard]] Cluster& cluster_for(Key key) noexcept;
    [[nodiscard]] const Cluster& cluster_for(Key key) const noexcept;
    [[nodiscard]] std::size_t replacement_index(const Cluster& cluster, Key key) const noexcept;

    std::unique_ptr<Cluster[]> clusters_;
    std::size_t cluster_count_ = 0;
    std::size_t cluster_mask_ = 0;
    std::atomic<std::uint8_t> generation_{0};
};

using InfoCallback = std::function<void(const SearchInfo&)>;
using BestMoveCallback = std::function<void(Move, Move)>;

class SearchController final {
public:
    SearchController();
    ~SearchController();

    void set_threads(int threads) noexcept;
    void set_hash_mb(std::size_t hash_mb);
    [[nodiscard]] int thread_count() const noexcept;
    [[nodiscard]] std::size_t hash_mb() const noexcept;

    [[nodiscard]] bool load_weights(const std::filesystem::path& path, std::string* error = nullptr);
    void clear_weights() noexcept;
    [[nodiscard]] int evaluate(const Position& pos) const noexcept;

    void clear();
    void new_game();

    void go(const Position& root, const SearchLimits& limits, InfoCallback info_callback, BestMoveCallback best_callback);
    void stop() noexcept;
    void wait();
    [[nodiscard]] bool searching() const noexcept;

private:
    SearchInfo run_worker(int worker_id, const std::string& root_fen, const SearchLimits& limits, const InfoCallback& info_callback);

    mutable std::mutex launch_mutex_;
    std::thread coordinator_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> searching_{false};
    std::atomic<std::uint64_t> global_nodes_{0};
    TranspositionTable tt_;
    eval::Evaluator evaluator_;
    std::size_t hash_mb_ = 32U;
    int thread_count_ = 1;
};

}  // namespace mythos::search
