#pragma once
#include <vector>
#include <map>

int heuristic_pdb_plus_manhattan_remaining(
    const std::vector<int>& board,
    const std::vector<int>& goal_state,
    int grid_size,
    const std::vector<int>& tile_to_goal,
    const std::vector<int>& pdb_pattern_tiles,
    const std::map<std::vector<int>, unsigned char>& pdb
);
void precompute_pdb(
    std::map<std::vector<int>, unsigned char>& pdb,
    int grid_size,
    const std::vector<int>& goal_state,
    const std::vector<int>& pdb_pattern_tiles
);
std::vector<int> get_pattern_positions(
    const std::vector<int>& board,
    const std::vector<int>& pdb_pattern_tiles,
    int grid_size
);
