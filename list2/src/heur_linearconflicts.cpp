#include "heur_linearconflicts.hpp"

int heuristic_linear_conflicts(const std::vector<int>& board, const std::vector<int>& goal_state, int grid_size, const std::vector<int>& tile_to_goal) {
    int manhattan = 0, conflicts = 0, N = grid_size * grid_size;

    for (int i = 0; i < N; ++i) {
        if (board[i] == 0) continue;
        int goal_idx = tile_to_goal[board[i]];
        manhattan += abs((i / grid_size) - (goal_idx / grid_size)) +
                     abs((i % grid_size) - (goal_idx % grid_size));
    }

    // Row conflicts
    for (int r = 0; r < grid_size; ++r) {
        for (int c1 = 0; c1 < grid_size; ++c1) {
            int idxA = r * grid_size + c1, tileA = board[idxA];
            if (tileA == 0) continue;
            int goalA = tile_to_goal[tileA];

            if ((goalA / grid_size) == r) {
                for (int c2 = c1 + 1; c2 < grid_size; ++c2) {
                    int idxB = r * grid_size + c2, tileB = board[idxB];
                    if (tileB == 0) continue;
                    int goalB = tile_to_goal[tileB];
                    if ((goalB / grid_size) == r && (goalA % grid_size) > (goalB % grid_size)) conflicts++;
                }
            }
        }
    }
    // Col conflicts
    for (int c = 0; c < grid_size; ++c) {
        for (int r1 = 0; r1 < grid_size; ++r1) {
            int idxA = r1 * grid_size + c, tileA = board[idxA];
            if (tileA == 0) continue;
            int goalA = tile_to_goal[tileA];

            if ((goalA % grid_size) == c) {
                for (int r2 = r1 + 1; r2 < grid_size; ++r2) {
                    int idxB = r2 * grid_size + c, tileB = board[idxB];
                    if (tileB == 0) continue;
                    int goalB = tile_to_goal[tileB];
                    if ((goalB % grid_size) == c && (goalA / grid_size) > (goalB / grid_size)) conflicts++;
                }
            }
        }
    }
    return manhattan + 2 * conflicts;
}
