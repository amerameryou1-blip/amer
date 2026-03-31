#ifndef EVALUATE_H
#define EVALUATE_H

#include "position.h"
#include "types.h"

namespace Eval {

// Main evaluation function - returns score in centipawns from side-to-move's perspective
Value evaluate(const Position& pos);

// Trace evaluation (for debugging)
void trace(const Position& pos);

// Simple material-only evaluation
Value material(const Position& pos);

// Phase calculation (0 = endgame, 256 = midgame)
int game_phase(const Position& pos);

} // namespace Eval

#endif // EVALUATE_H
