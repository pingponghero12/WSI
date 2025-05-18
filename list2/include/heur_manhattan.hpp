#pragma once
#include <vector>

int heuristic_manhattan_distance(const std::vector<int>& board, const std::vector<int>& goal_state, int grid_size, const std::vector<int>& tile_to_goal);
void precompute_tile_goal_positions(const std::vector<int>& goal_state, std::vector<int>& tile_to_goal);
