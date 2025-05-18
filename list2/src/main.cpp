#include <iostream>
#include <vector>
#include <string>
#include "solver.hpp"
#include "utils.hpp"
#include "heur_manhattan.hpp"
#include "heur_pdb.hpp"
#include "pdb_loader.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <grid_size:int> <heuristic_type: -1, -2, -3, -4> [--print]" << std::endl;
        return 1;
    }

    int grid_size = std::stoi(argv[1]);
    int heuristic_arg = std::stoi(argv[2]);

    if (grid_size < 2 ||
        (heuristic_arg != -1 && heuristic_arg != -2 && heuristic_arg != -3 && heuristic_arg != -4)) {
        std::cerr << "Invalid arguments." << std::endl;
        return 1;
    }
    int heuristic_id = -heuristic_arg;

    bool print_details = (argc > 3 && std::string(argv[3]) == "--print");
    std::vector<int> goal_state = make_goal_state(grid_size);
    std::vector<int> tile_to_goal;
    precompute_tile_goal_positions(goal_state, tile_to_goal);

    std::vector<int> pdb_pattern_tiles;
    if (grid_size == 4) pdb_pattern_tiles = {1,2,3,4,5,6};
    else if (grid_size == 3) pdb_pattern_tiles = {1,2,3,4};
    else for (int i = 1; i <= std::min(grid_size * grid_size - 2, 6); ++i) pdb_pattern_tiles.push_back(i);

    std::map<std::vector<int>, unsigned char> pdb;
    std::string pdb_filename = "pdb_" + std::to_string(pdb_pattern_tiles.size()) + "tile_" + std::to_string(grid_size) + "x" + std::to_string(grid_size) + ".dat";

    if (heuristic_id == 4) {
        if (!load_pdb_from_file(pdb_filename, pdb, pdb_pattern_tiles.size() + 1)) {
            if (print_details) std::cout << "PDB file not found: Precomputing." << std::endl;
            precompute_pdb(pdb, grid_size, goal_state, pdb_pattern_tiles);
            save_pdb_to_file(pdb_filename, pdb);
        }
    }

    std::vector<int> initial_state = generate_initial_state(grid_size, goal_state);
    if (print_details) {
        std::cout << "Initial State:" << std::endl;
        print_board(initial_state, grid_size);
        std::cout << std::endl;
    }

    AStarResult result = solve_A_star(
        initial_state, goal_state, grid_size, heuristic_id, tile_to_goal, pdb_pattern_tiles, pdb
    );

    if (result.success) {
        int solution_length = result.path.empty() ? 0 : static_cast<int>(result.path.size()) - 1;

        if (print_details) {
            std::cout << "Solution Path (tile numbers moved):" << std::endl;

            std::vector<int> moves = get_moved_tiles_from_path(result.path, grid_size);

            for (size_t i = 0; i < moves.size(); ++i) {
                std::cout << moves[i] << (i == moves.size() - 1 ? "" : " -> ");
            }
            std::cout << std::endl << std::endl << "Solution as board states (" << solution_length << " moves):" << std::endl;

            for (const auto& b : result.path) {
                print_board(b, grid_size);
                if (&b != &result.path.back()) std::cout << "---" << std::endl;
            }

            std::cout << std::endl;
            std::cout << "Number of visited states: " << result.visited_states << std::endl;
            std::cout << "Length of solution path (moves): " << solution_length << std::endl;
        } else {
            std::cout << result.visited_states << "," << solution_length << std::endl;
        }
    } 
    else {
        if (print_details) {
            std::cout << "No solution found." << std::endl;
            std::cout << "Visited states before failure: " << result.visited_states << std::endl;
        }
        else {
            std::cout << result.visited_states << ",-1" << std::endl;
        }
    }
    return 0;
}
