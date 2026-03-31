#include "mythos/uci/uci.hpp"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

namespace mythos::uci {

namespace {

[[nodiscard]] std::vector<std::string> tokenize(const std::string& line) {
    std::istringstream stream(line);
    std::vector<std::string> tokens;
    for (std::string token; stream >> token;) {
        tokens.push_back(std::move(token));
    }
    return tokens;
}

[[nodiscard]] std::string score_to_string(Value score) {
    std::ostringstream out;
    if (score >= VALUE_MATE_IN_MAX_PLY) {
        out << "mate " << (VALUE_MATE - score + 1) / 2;
    } else if (score <= VALUE_MATED_IN_MAX_PLY) {
        out << "mate " << -(VALUE_MATE + score) / 2;
    } else {
        out << "cp " << score;
    }
    return out.str();
}

}  // namespace

UciLoop::UciLoop() {
    position_.set(StartFEN, &state_pool_[state_index_++]);
}

Move UciLoop::parse_move(const Position& pos, std::string_view token) const {
    if (token.size() < 4) {
        return MOVE_NONE;
    }

    const File from_file = File(token[0] - 'a');
    const Rank from_rank = Rank(token[1] - '1');
    const File to_file = File(token[2] - 'a');
    const Rank to_rank = Rank(token[3] - '1');

    if (from_file > FILE_H || from_rank > RANK_8 || to_file > FILE_H || to_rank > RANK_8) {
        return MOVE_NONE;
    }

    const Square from = make_square(from_file, from_rank);
    const Square to = make_square(to_file, to_rank);

    PieceType promotion = NO_PIECE_TYPE;
    if (token.size() > 4) {
        switch (token[4]) {
            case 'n': promotion = KNIGHT; break;
            case 'b': promotion = BISHOP; break;
            case 'r': promotion = ROOK; break;
            case 'q': promotion = QUEEN; break;
            default: break;
        }
    }

    MoveList legal_moves;
    generate<LEGAL>(pos, legal_moves);
    for (const auto& scored : legal_moves) {
        const Move move = scored.move;
        if (move.from() == from && move.to() == to) {
            if (move.type() == PROMOTION) {
                if (move.promotion_type() == promotion) {
                    return move;
                }
            } else if (promotion == NO_PIECE_TYPE) {
                return move;
            }
        }
        if (move.type() == CASTLING && move.from() == from) {
            const Square king_target = to > from ? make_square(FILE_G, rank_of(from))
                                                 : make_square(FILE_C, rank_of(from));
            if (to == king_target) {
                return move;
            }
        }
    }
    return MOVE_NONE;
}

void UciLoop::print_line(const std::string& line) {
    std::scoped_lock lock(io_mutex_);
    std::cout << line << std::endl;
}

void UciLoop::print_bestmove(Move best, Move ponder) {
    std::ostringstream out;
    out << "bestmove " << (best ? best.to_uci() : "0000");
    if (ponder) {
        out << " ponder " << ponder.to_uci();
    }
    print_line(out.str());
}

void UciLoop::print_info(const search::SearchInfo& info) {
    std::ostringstream out;
    out << "info depth " << info.depth
        << " seldepth " << info.seldepth
        << " score " << score_to_string(info.score)
        << " nodes " << info.nodes
        << " nps " << info.nps
        << " time " << info.elapsed_ms
        << " hashfull " << info.hashfull
        << " pv";
    for (int i = 0; i < info.pv_length; ++i) {
        out << ' ' << info.pv[i].to_uci();
    }
    print_line(out.str());
}

void UciLoop::cmd_uci() {
    print_line("id name Mythos 10");
    print_line("id author OpenAI Codex");
    print_line("option name Hash type spin default 32 min 1 max 65536");
    print_line("option name Threads type spin default 1 min 1 max 256");
    print_line("option name WeightsFile type string default");
    print_line("option name Clear Hash type button");
    print_line("uciok");
}

void UciLoop::cmd_isready() {
    print_line("readyok");
}

void UciLoop::cmd_setoption(const std::string& line) {
    search_.stop();
    search_.wait();

    const auto tokens = tokenize(line);
    std::string name;
    std::string value;

    bool in_name = false;
    bool in_value = false;
    for (std::size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == "name") {
            in_name = true;
            in_value = false;
            continue;
        }
        if (tokens[i] == "value") {
            in_name = false;
            in_value = true;
            continue;
        }
        if (in_name) {
            if (!name.empty()) {
                name += ' ';
            }
            name += tokens[i];
        } else if (in_value) {
            if (!value.empty()) {
                value += ' ';
            }
            value += tokens[i];
        }
    }

    if (name == "Hash") {
        search_.set_hash_mb(static_cast<std::size_t>(std::max(1, std::stoi(value))));
    } else if (name == "Threads") {
        search_.set_threads(std::max(1, std::stoi(value)));
    } else if (name == "WeightsFile") {
        if (value.empty()) {
            search_.clear_weights();
            return;
        }
        std::string error;
        if (!search_.load_weights(value, &error)) {
            print_line("info string failed to load weights: " + error);
        }
    } else if (name == "Clear Hash") {
        search_.clear();
    }
}

void UciLoop::cmd_position(const std::string& line) {
    search_.stop();
    search_.wait();

    const auto tokens = tokenize(line);
    if (tokens.size() < 2) {
        return;
    }

    state_index_ = 0;

    std::size_t index = 1;
    if (tokens[index] == "startpos") {
        position_.set(StartFEN, &state_pool_[state_index_++]);
        ++index;
    } else if (tokens[index] == "fen") {
        ++index;
        std::ostringstream fen;
        while (index < tokens.size() && tokens[index] != "moves") {
            if (fen.tellp() > 0) {
                fen << ' ';
            }
            fen << tokens[index++];
        }
        position_.set(fen.str(), &state_pool_[state_index_++]);
    }

    if (index < tokens.size() && tokens[index] == "moves") {
        ++index;
        for (; index < tokens.size(); ++index) {
            const Move move = parse_move(position_, tokens[index]);
            if (move != MOVE_NONE) {
                position_.do_move(move, state_pool_[state_index_++]);
            }
        }
    }
}

void UciLoop::cmd_go(const std::string& line) {
    const auto tokens = tokenize(line);
    search::SearchLimits limits;
    limits.clear();

    std::size_t index = 1;
    while (index < tokens.size()) {
        const auto& token = tokens[index++];
        if (token == "searchmoves") {
            while (index < tokens.size()) {
                const Move move = parse_move(position_, tokens[index]);
                if (move == MOVE_NONE) {
                    break;
                }
                limits.searchmoves.push_back(move);
                ++index;
            }
        } else if (token == "wtime" && index < tokens.size()) {
            limits.time[WHITE] = std::stoi(tokens[index++]);
        } else if (token == "btime" && index < tokens.size()) {
            limits.time[BLACK] = std::stoi(tokens[index++]);
        } else if (token == "winc" && index < tokens.size()) {
            limits.inc[WHITE] = std::stoi(tokens[index++]);
        } else if (token == "binc" && index < tokens.size()) {
            limits.inc[BLACK] = std::stoi(tokens[index++]);
        } else if (token == "movestogo" && index < tokens.size()) {
            limits.movestogo = std::stoi(tokens[index++]);
        } else if (token == "depth" && index < tokens.size()) {
            limits.depth = std::stoi(tokens[index++]);
        } else if (token == "nodes" && index < tokens.size()) {
            limits.nodes = std::stoll(tokens[index++]);
        } else if (token == "movetime" && index < tokens.size()) {
            limits.movetime = std::stoi(tokens[index++]);
        } else if (token == "infinite") {
            limits.infinite = true;
        } else if (token == "ponder") {
            limits.ponder = true;
        }
    }

    search_.go(position_, limits,
               [this](const search::SearchInfo& info) { print_info(info); },
               [this](Move best, Move ponder) { print_bestmove(best, ponder); });
}

void UciLoop::cmd_stop() {
    search_.stop();
}

void UciLoop::cmd_quit() {
    search_.stop();
    search_.wait();
    quitting_ = true;
}

void UciLoop::cmd_ucinewgame() {
    search_.stop();
    search_.wait();
    search_.new_game();
    state_index_ = 0;
    position_.set(StartFEN, &state_pool_[state_index_++]);
}

void UciLoop::cmd_d() {
    print_line(position_.to_string());
}

void UciLoop::cmd_eval() {
    print_line("info string eval " + std::to_string(search_.evaluate(position_)) + " cp");
}

void UciLoop::cmd_perft(const std::string& line) {
    search_.stop();
    search_.wait();
    std::istringstream input(line);
    std::string token;
    int depth = 0;
    input >> token >> depth;
    std::ostringstream out;
    out << "info string perft nodes " << perft(position_, depth);
    print_line(out.str());
}

void UciLoop::cmd_divide(const std::string& line) {
    search_.stop();
    search_.wait();
    std::istringstream input(line);
    std::string token;
    int depth = 0;
    input >> token >> depth;
    perft_divide(position_, depth);
}

void UciLoop::run() {
    for (std::string line; !quitting_ && std::getline(std::cin, line);) {
        if (line.empty()) {
            continue;
        }

        std::istringstream input(line);
        std::string command;
        input >> command;

        if (command == "uci") {
            cmd_uci();
        } else if (command == "isready") {
            cmd_isready();
        } else if (command == "setoption") {
            cmd_setoption(line);
        } else if (command == "position") {
            cmd_position(line);
        } else if (command == "go") {
            cmd_go(line);
        } else if (command == "stop") {
            cmd_stop();
        } else if (command == "quit") {
            cmd_quit();
        } else if (command == "ucinewgame") {
            cmd_ucinewgame();
        } else if (command == "d") {
            cmd_d();
        } else if (command == "eval") {
            cmd_eval();
        } else if (command == "perft") {
            cmd_perft(line);
        } else if (command == "divide") {
            cmd_divide(line);
        }
    }
}

}  // namespace mythos::uci
