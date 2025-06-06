#include "solver.hpp"
#include "heur_misplaced.hpp"
#include "heur_manhattan.hpp"
#include "heur_linearconflicts.hpp"
#include "heur_pdb.hpp"
#include "utils.hpp"
#include <set>
#include <queue>
#include <map>

AStarResult solve_A_star(
    const std::vector<int>& initial_board,
    const std::vector<int>& goal_state,
    int grid_size,
    int heuristic_choice_id,
    const std::vector<int>& tile_to_goal,
    const std::vector<int>& pdb_pattern_tiles,
    const std::map<std::vector<int>, unsigned char>& pdb
) {
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_list;
    std::set<std::vector<int>> closed_set;
    std::map<std::vector<int>, std::vector<int>> came_from;
    std::map<std::vector<int>, int> g_costs;

    int initial_empty_idx = find_empty_idx(initial_board);
    int initial_h_cost = 0;

    if (heuristic_choice_id == 1) initial_h_cost = heuristic_misplaced_tiles(initial_board, goal_state);
    else if (heuristic_choice_id == 2) initial_h_cost = heuristic_manhattan_distance(initial_board, goal_state, grid_size, tile_to_goal);
    else if (heuristic_choice_id == 3) initial_h_cost = heuristic_linear_conflicts(initial_board, goal_state, grid_size, tile_to_goal);
    else if (heuristic_choice_id == 4) initial_h_cost = heuristic_pdb_plus_manhattan_remaining(initial_board, goal_state, grid_size, tile_to_goal, pdb_pattern_tiles, pdb);

    Node start_node = {initial_board, 0, initial_h_cost, initial_empty_idx};
    open_list.push(start_node);
    g_costs[initial_board] = 0;
    int visited_states_count = 0;

    while (!open_list.empty()) {
        Node current_node = open_list.top(); open_list.pop();

        if (closed_set.count(current_node.board)) continue;
        closed_set.insert(current_node.board);
        visited_states_count++;

        if (current_node.board == goal_state) {
            std::vector<std::vector<int>> path;
            std::vector<int> temp_board = current_node.board;

            while (temp_board != initial_board) {
                path.push_back(temp_board);
                if (came_from.find(temp_board) == came_from.end()) break;
                temp_board = came_from[temp_board];
            }

            path.push_back(initial_board);
            std::reverse(path.begin(), path.end());
            return {path, visited_states_count, true};
        }
        auto neighbors = get_neighbors(current_node.board, current_node.empty_idx, grid_size);

        for (const auto& neighbor_board : neighbors) {
            if (closed_set.count(neighbor_board)) continue;
            int tentative_g_cost = current_node.g_cost + 1;
            if (g_costs.find(neighbor_board) == g_costs.end() || tentative_g_cost < g_costs[neighbor_board]) {
                g_costs[neighbor_board] = tentative_g_cost;
                came_from[neighbor_board] = current_node.board;
                int h_val = 0;

                if (heuristic_choice_id == 1) {
                    h_val = heuristic_misplaced_tiles(neighbor_board, goal_state);
                }
                else if (heuristic_choice_id == 2) {
                    h_val = heuristic_manhattan_distance(neighbor_board, goal_state, grid_size, tile_to_goal);
                }
                else if (heuristic_choice_id == 3) {
                    h_val = heuristic_linear_conflicts(neighbor_board, goal_state, grid_size, tile_to_goal);
                }
                else if (heuristic_choice_id == 4) {
                    h_val = heuristic_pdb_plus_manhattan_remaining(neighbor_board, goal_state, grid_size, tile_to_goal, pdb_pattern_tiles, pdb);
                }

                Node neighbor_node = {neighbor_board, tentative_g_cost, h_val, find_empty_idx(neighbor_board)};
                open_list.push(neighbor_node);
            }
        }
    }
    return {{}, visited_states_count, false};
}
