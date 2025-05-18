#include "utils.hpp"
#include <algorithm>
#include <random>
#include <chrono>

void print_board(const std::vector<int>& board, int grid_size) {
    for (size_t i = 0; i < board.size(); ++i) {
        if (board[i] == 0) std::cout << "   ";
        else std::cout << (board[i] < 10 ? " " : "") << board[i] << " ";
        if ((i + 1) % grid_size == 0) std::cout << std::endl;
    }
}

int find_empty_idx(const std::vector<int>& board) {
    for (size_t i = 0; i < board.size(); ++i) {
        if (board[i] == 0) return i;
    }
    return -1;
}

std::vector<int> get_moved_tiles_from_path(const std::vector<std::vector<int>>& path, int grid_size) {
    std::vector<int> moved;
    if (path.size() < 2) return moved;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        int prev_empty_idx = find_empty_idx(path[i]);
        moved.push_back(path[i + 1][prev_empty_idx]);
    }
    return moved;
}

int count_inversions(const std::vector<int>& board) {
    int inv = 0;
    for (size_t i = 0; i < board.size(); ++i) {
        if (board[i] == 0) continue;
        for (size_t j = i + 1; j < board.size(); ++j) {
            if (board[j] == 0) continue;
            if (board[i] > board[j]) ++inv;
        }
    }
    return inv;
}

bool is_solvable(const std::vector<int>& board, int grid_size, const std::vector<int>& goal_state) {
    int inv = count_inversions(board);
    int num_tiles = grid_size * grid_size;
    int empty_idx = find_empty_idx(board);

    if (grid_size % 2 == 1) {
        return inv % 2 == 0;
    } 
    else {
        int empty_row_from_bottom_1idx = grid_size - (empty_idx / grid_size);
        if (empty_row_from_bottom_1idx % 2 == 0) {
            return inv % 2 == 1;
        } 
        else {
            return inv % 2 == 0;
        }
    }
}

std::vector<int> make_goal_state(int grid_size) {
    int n = grid_size * grid_size;
    std::vector<int> goal(n);
    for (int i = 0; i < n - 1; ++i) goal[i] = i + 1;
    goal[n - 1] = 0;
    return goal;
}

std::vector<int> generate_initial_state(int grid_size, const std::vector<int>& goal_state) {
    int num_tiles = grid_size * grid_size;
    std::vector<int> board(num_tiles), tiles;

    for (int i = 1; i < num_tiles; ++i) tiles.push_back(i);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g(seed);
    do {
        std::shuffle(tiles.begin(), tiles.end(), g);
        for (int i = 0; i < num_tiles - 1; ++i) board[i] = tiles[i];
        board[num_tiles - 1] = 0;
    } while (!is_solvable(board, grid_size, goal_state));
    return board;
}

std::vector<std::vector<int>> get_neighbors(const std::vector<int>& current_board, int empty_idx, int grid_size) {
    std::vector<std::vector<int>> neighbors;

    int r = empty_idx / grid_size, c = empty_idx % grid_size;
    int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};

    for (int i = 0; i < 4; ++i) {
        int nr = r + dr[i], nc = c + dc[i];
        if (nr >= 0 && nr < grid_size && nc >= 0 && nc < grid_size) {
            std::vector<int> nb = current_board;
            std::swap(nb[empty_idx], nb[nr * grid_size + nc]);
            neighbors.push_back(nb);
        }
    }
    return neighbors;
}
