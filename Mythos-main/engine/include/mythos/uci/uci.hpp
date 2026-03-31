#pragma once

#include "../search/search.hpp"

#include <array>
#include <mutex>
#include <string>
#include <string_view>

namespace mythos::uci {

class UciLoop final {
public:
    UciLoop();

    void run();

private:
    [[nodiscard]] Move parse_move(const Position& pos, std::string_view token) const;
    void print_line(const std::string& line);
    void print_bestmove(Move best, Move ponder);
    void print_info(const search::SearchInfo& info);

    void cmd_uci();
    void cmd_isready();
    void cmd_setoption(const std::string& line);
    void cmd_position(const std::string& line);
    void cmd_go(const std::string& line);
    void cmd_stop();
    void cmd_quit();
    void cmd_ucinewgame();
    void cmd_d();
    void cmd_eval();
    void cmd_perft(const std::string& line);
    void cmd_divide(const std::string& line);

    std::mutex io_mutex_;
    Position position_;
    std::array<StateInfo, MAX_GAME_LENGTH> state_pool_{};
    int state_index_ = 0;
    bool quitting_ = false;
    search::SearchController search_;
};

}  // namespace mythos::uci
