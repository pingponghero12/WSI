#pragma once
#include <vector>
#include <map>
#include "common.hpp"

AStarResult solve_A_star(
    const std::vector<int>& initial_board,
    const std::vector<int>& goal_state,
    int grid_size,
    int heuristic_choice_id,
    const std::vector<int>& tile_to_goal,
    const std::vector<int>& pdb_pattern_tiles,
    const std::map<std::vector<int>, unsigned char>& pdb
);
