#include "mythos/board/bitboard.hpp"
#include "mythos/board/position.hpp"
#include "mythos/uci/uci.hpp"

int main() {
    Bitboards::init();
    Position::init();

    mythos::uci::UciLoop loop;
    loop.run();
    return 0;
}
