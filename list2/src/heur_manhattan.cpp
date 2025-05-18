#include "heur_manhattan.hpp"

void precompute_tile_goal_positions(const std::vector<int>& goal_state, std::vector<int>& tile_to_goal) {
    int N = goal_state.size();
    tile_to_goal.assign(N + 1, -1);
    for (int i = 0; i < N; ++i) {
        if (goal_state[i] != 0) tile_to_goal[goal_state[i]] = i;
    }
}

int heuristic_manhattan_distance(const std::vector<int>& board, const std::vector<int>& goal_state, int grid_size, const std::vector<int>& tile_to_goal) {
    int sum = 0;

    for (size_t i = 0; i < board.size(); ++i) {
        if (board[i] == 0) continue;
        int goal_idx = tile_to_goal[board[i]];
        sum += abs((int(i) / grid_size) - (goal_idx / grid_size))
             + abs((int(i) % grid_size) - (goal_idx % grid_size));
    }
    return sum;
}
