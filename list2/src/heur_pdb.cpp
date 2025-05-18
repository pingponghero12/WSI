#include "heur_pdb.hpp"
#include "heur_manhattan.hpp"
#include <queue>
#include <set>
#include <algorithm>

std::vector<int> get_pattern_positions(const std::vector<int>& board, const std::vector<int>& pdb_pattern_tiles, int grid_size) {
    std::vector<int> positions(pdb_pattern_tiles.size() + 1);

    for (size_t i = 0; i < pdb_pattern_tiles.size(); ++i) {
        auto it = std::find(board.begin(), board.end(), pdb_pattern_tiles[i]);
        positions[i] = it - board.begin();
    }
    positions.back() = std::find(board.begin(), board.end(), 0) - board.begin();
    return positions;
}

void precompute_pdb(
    std::map<std::vector<int>, unsigned char>& pdb,
    int grid_size,
    const std::vector<int>& goal_state,
    const std::vector<int>& pdb_pattern_tiles
) {
    pdb.clear();

    int N = grid_size * grid_size;
    std::vector<int> pattern_goal(N, -1);

    for (size_t i = 0; i < pdb_pattern_tiles.size(); ++i) {
        pattern_goal[i] = pdb_pattern_tiles[i];
    }

    pattern_goal[N - 1] = 0;
    auto pdb_goal_pattern_positions = get_pattern_positions(pattern_goal, pdb_pattern_tiles, grid_size);

    std::queue<std::pair<std::vector<int>, unsigned char>> q;
    q.push({pdb_goal_pattern_positions, 0});
    pdb[pdb_goal_pattern_positions] = 0;

    int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};
    while (!q.empty()) {
        auto [cur, cost] = q.front(); q.pop();
        int blank_pos = cur.back();
        int r = blank_pos / grid_size, c = blank_pos % grid_size;

        for (int i = 0; i < 4; ++i) {
            int nr = r + dr[i], nc = c + dc[i];

            if (nr >= 0 && nr < grid_size && nc >= 0 && nc < grid_size) {
                int new_blank = nr * grid_size + nc;
                std::vector<int> next = cur;
                auto it = std::find(cur.begin(), cur.end() - 1, new_blank);
                next.back() = new_blank;

                if (it != cur.end() - 1) *it = blank_pos;
                if (pdb.find(next) == pdb.end() && cost + 1 < 255) {
                    pdb[next] = cost + 1;
                    q.push({next, (unsigned char)(cost + 1)});
                }
            }
        }
    }
}

int heuristic_pdb_plus_manhattan_remaining(
    const std::vector<int>& board,
    const std::vector<int>& goal_state,
    int grid_size,
    const std::vector<int>& tile_to_goal,
    const std::vector<int>& pdb_pattern_tiles,
    const std::map<std::vector<int>, unsigned char>& pdb
) {
    auto key = get_pattern_positions(board, pdb_pattern_tiles, grid_size);
    auto it = pdb.find(key);
    int pdb_cost = (it != pdb.end()) ? it->second : 0;
    int N = grid_size * grid_size;
    int manhattan = 0;
    for (int i = 0; i < N; ++i) {
        int v = board[i];
        if (v == 0) continue;
        if (std::find(pdb_pattern_tiles.begin(), pdb_pattern_tiles.end(), v) != pdb_pattern_tiles.end()) continue;
        int goal_idx = tile_to_goal[v];
        manhattan += abs((i / grid_size) - (goal_idx / grid_size))
                   + abs((i % grid_size) - (goal_idx % grid_size));
    }
    return pdb_cost + manhattan;
}
