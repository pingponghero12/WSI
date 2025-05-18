#pragma once
#include <vector>
#include <iostream>

void print_board(const std::vector<int>& board, int grid_size);
int find_empty_idx(const std::vector<int>& board);
std::vector<int> get_moved_tiles_from_path(const std::vector<std::vector<int>>& path, int grid_size);
int count_inversions(const std::vector<int>& board);
bool is_solvable(const std::vector<int>& board, int grid_size, const std::vector<int>& goal_state);
std::vector<int> generate_initial_state(int grid_size, const std::vector<int>& goal_state);
std::vector<std::vector<int>> get_neighbors(const std::vector<int>& current_board, int empty_idx, int grid_size);
std::vector<int> make_goal_state(int grid_size);
