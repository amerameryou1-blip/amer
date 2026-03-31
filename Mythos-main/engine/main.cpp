#include "bitboard.h"
#include "position.h"
#include "movegen.h"
#include "search.h"
#include "evaluate.h"
#include "types.h"

#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>

namespace {

Position pos;
StateInfo stateInfoPool[MAX_GAME_LENGTH];
int stateInfoIdx = 0;

// Parse a move from UCI string
Move parse_move(const Position& pos, const std::string& str) {
    if (str.length() < 4) return MOVE_NONE;
    
    File fromFile = File(str[0] - 'a');
    Rank fromRank = Rank(str[1] - '1');
    File toFile = File(str[2] - 'a');
    Rank toRank = Rank(str[3] - '1');
    
    if (fromFile > FILE_H || fromRank > RANK_8 || toFile > FILE_H || toRank > RANK_8) {
        return MOVE_NONE;
    }
    
    Square from = make_square(fromFile, fromRank);
    Square to = make_square(toFile, toRank);
    
    PieceType promotion = NO_PIECE_TYPE;
    if (str.length() > 4) {
        switch (str[4]) {
            case 'n': promotion = KNIGHT; break;
            case 'b': promotion = BISHOP; break;
            case 'r': promotion = ROOK; break;
            case 'q': promotion = QUEEN; break;
            default: break;
        }
    }
    
    // Generate legal moves and find the matching one
    MoveList moves;
    generate<LEGAL>(pos, moves);
    
    for (const auto& sm : moves) {
        Move m = sm.move;
        if (m.from() == from && m.to() == to) {
            // Check promotion matches
            if (m.type() == PROMOTION) {
                if (m.promotion_type() == promotion) return m;
            } else if (promotion == NO_PIECE_TYPE) {
                return m;
            }
        }
        
        // Handle castling: UCI uses king's from/to, but we also accept rook's square
        if (m.type() == CASTLING && m.from() == from) {
            // King target squares for standard chess
            Square kTo = (to > from) ? make_square(FILE_G, rank_of(from)) 
                                     : make_square(FILE_C, rank_of(from));
            if (to == kTo) return m;
        }
    }
    
    return MOVE_NONE;
}

void cmd_uci() {
    std::cout << "id name ChessEngine 1.0" << std::endl;
    std::cout << "id author ChessEngine Team" << std::endl;
    std::cout << "option name Hash type spin default 16 min 1 max 1024" << std::endl;
    std::cout << "option name Threads type spin default 1 min 1 max 1" << std::endl;
    std::cout << "uciok" << std::endl;
}

void cmd_setoption(std::istringstream& is) {
    std::string token, name, value;
    
    is >> token;  // "name"
    
    while (is >> token && token != "value") {
        name += (name.empty() ? "" : " ") + token;
    }
    
    while (is >> token) {
        value += (value.empty() ? "" : " ") + token;
    }
    
    if (name == "Hash") {
        int mb = std::stoi(value);
        Threads.set_tt_size(mb);
    }
}

void cmd_position(std::istringstream& is) {
    std::string token;
    is >> token;
    
    stateInfoIdx = 0;
    
    if (token == "startpos") {
        pos.set(StartFEN, &stateInfoPool[stateInfoIdx++]);
        is >> token;  // Consume "moves" if present
    } else if (token == "fen") {
        std::string fen;
        while (is >> token && token != "moves") {
            fen += token + " ";
        }
        pos.set(fen, &stateInfoPool[stateInfoIdx++]);
    }
    
    // Parse moves
    while (is >> token) {
        Move m = parse_move(pos, token);
        if (m != MOVE_NONE) {
            pos.do_move(m, stateInfoPool[stateInfoIdx++]);
        } else {
            std::cerr << "Invalid move: " << token << std::endl;
        }
    }
}

void cmd_go(std::istringstream& is) {
    SearchLimits limits;
    limits.clear();
    
    std::string token;
    
    while (is >> token) {
        if (token == "searchmoves") {
            while (is >> token) {
                Move m = parse_move(pos, token);
                if (m != MOVE_NONE) {
                    limits.searchmoves.push_back(m);
                } else {
                    // Not a move, put it back somehow... simplified: just break
                    break;
                }
            }
        } else if (token == "wtime") {
            is >> limits.time[WHITE];
        } else if (token == "btime") {
            is >> limits.time[BLACK];
        } else if (token == "winc") {
            is >> limits.inc[WHITE];
        } else if (token == "binc") {
            is >> limits.inc[BLACK];
        } else if (token == "movestogo") {
            is >> limits.movestogo;
        } else if (token == "depth") {
            is >> limits.depth;
        } else if (token == "nodes") {
            is >> limits.nodes;
        } else if (token == "movetime") {
            is >> limits.movetime;
        } else if (token == "infinite") {
            limits.infinite = true;
        } else if (token == "ponder") {
            limits.ponder = true;
        }
    }
    
    // Start search in separate thread so we can receive "stop"
    std::thread([&limits]() {
        Threads.go(pos, limits);
    }).detach();
}

void cmd_stop() {
    Threads.stop();
}

void cmd_quit() {
    Threads.stop();
    Threads.wait();
    exit(0);
}

void cmd_isready() {
    std::cout << "readyok" << std::endl;
}

void cmd_ucinewgame() {
    Threads.new_game();
    stateInfoIdx = 0;
    pos.set(StartFEN, &stateInfoPool[stateInfoIdx++]);
}

void cmd_d() {
    std::cout << pos.to_string() << std::endl;
}

void cmd_eval() {
    Eval::trace(pos);
}

void cmd_perft(std::istringstream& is) {
    int depth;
    is >> depth;
    
    auto start = std::chrono::steady_clock::now();
    uint64_t nodes = perft(pos, depth);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "\nNodes: " << nodes << std::endl;
    std::cout << "Time: " << elapsed << " ms" << std::endl;
    if (elapsed > 0) {
        std::cout << "NPS: " << nodes * 1000 / elapsed << std::endl;
    }
}

void cmd_divide(std::istringstream& is) {
    int depth;
    is >> depth;
    perft_divide(pos, depth);
}

void uci_loop() {
    std::string line, token;
    
    // Set up initial position
    pos.set(StartFEN, &stateInfoPool[stateInfoIdx++]);
    
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        is >> std::skipws >> token;
        
        if (token == "uci") cmd_uci();
        else if (token == "setoption") cmd_setoption(is);
        else if (token == "isready") cmd_isready();
        else if (token == "ucinewgame") cmd_ucinewgame();
        else if (token == "position") cmd_position(is);
        else if (token == "go") cmd_go(is);
        else if (token == "stop") cmd_stop();
        else if (token == "quit") cmd_quit();
        else if (token == "d") cmd_d();
        else if (token == "eval") cmd_eval();
        else if (token == "perft") cmd_perft(is);
        else if (token == "divide") cmd_divide(is);
        else if (token == "bench") {
            // Simple benchmark
            SearchLimits limits;
            limits.depth = 10;
            Threads.go(pos, limits);
            Threads.wait();
        }
    }
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    // Initialize engine
    Bitboards::init();
    Position::init();
    
    // Check for command-line args
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "bench") {
            pos.set(StartFEN, &stateInfoPool[0]);
            SearchLimits limits;
            limits.depth = 12;
            Threads.go(pos, limits);
            Threads.wait();
            return 0;
        }
    }
    
    // Start UCI loop
    uci_loop();
    
    return 0;
}
