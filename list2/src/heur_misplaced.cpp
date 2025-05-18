#include "heur_misplaced.hpp"

int heuristic_misplaced_tiles(const std::vector<int>& board, const std::vector<int>& goal_state) {
    int misplaced = 0;

    for (size_t i = 0; i < board.size(); ++i) {
        if (board[i] != 0 && board[i] != goal_state[i]) misplaced++;
    }
    return misplaced;
}
