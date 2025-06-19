#pragma once
#include <climits>
#include <vector>
#include <algorithm>
#include <random>

// Win/Loss evaluation values
#define WIN_SCORE 1000
#define LOSS_SCORE -1000

// Threat evaluation values
#define THREE_IN_ROW_SCORE 100
#define OPP_THREE_IN_ROW_SCORE -150
#define TWO_IN_ROW_SCORE 10
#define OPP_TWO_IN_ROW_SCORE -15

// Pattern recognition values
#define LONG_SPACING_BONUS 80  // X _ _ X pattern
#define SHORT_SPACING_BONUS 40 // X _ X pattern
#define CLUSTERING_PENALTY -30 // Adjacent piece penalty

// Position evaluation values
#define CORNER_BONUS 50      // Positions 11, 15, 51, 55
#define NEAR_CENTER_BONUS 30 // Positions 22, 24, 42, 44
#define SECONDARY_BONUS 15   // Positions 23, 32, 34, 43
#define EDGE_BONUS 10         // Any edge position

/*
 * Heurisic Function:
 *
 * 1. ROW PATTERNS:
 *    - 4-in-a-row: +-1000
 *    - 3-in-a-row: +100 -150
 *    - 2-in-a-row: +10 -15
 *
 * 2. BOARD POSITIONS:
 *    - Corners (11,15,51,55): +50
 *    - Near-center (22,24,42,44): +30
 *    - Secondary positions: +15
 *    - Edge positions: +10
 *
 * 3. SPECIAL PATTERNS:
 *    - X _ _ X pattern: +80
 *    - X _ X pattern: +40
 *    - Adjacent pieces: -30
 */

class Minimax {
private:
    int max_depth;
    int player; 
    int opp;
    long long nodes;
    std::mt19937 rng;
    
public:
    Minimax(int depth, int p) : max_depth(depth), player(p) {
        opp = (p == 1) ? 2 : 1;
        nodes = 0;
        
        std::random_device rd;
        rng = std::mt19937(rd());
    }
    
    int get_best_move(int board[5][5]) {
        nodes = 0;
        
        std::vector<int> all_moves;
        std::vector<int> safe_moves;
        
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (board[i][j] == 0) {
                    int move = (i + 1) * 10 + (j + 1);
                    all_moves.push_back(move);
                    
                    board[i][j] = player; // Temporarily make the move
                    if (!would_create_losing_three(board, i, j)) { // Check for the AI player
                        safe_moves.push_back(move);
                    }
                    board[i][j] = 0; // Undo the move
                }
            }
        }
        
        if (safe_moves.empty()) {
            if (!all_moves.empty()) return all_moves[0];
            else return -1;
        }


        //Check if we can win immediately (in one move)
        for (int move : safe_moves) {
            int i = (move / 10) - 1;
            int j = (move % 10) - 1;
            
            // Temporarily make the move to check for win
            board[i][j] = player;
            bool wins = has_four_in_a_row(board, i, j, player); // Check if this move creates a 4-in-a-row for us
            board[i][j] = 0; // Undo
            
            if (wins) {
                return move;
            }
        }

        // Block opponent's winning moves (in one move)
        for (int move : safe_moves) { // Iterate over our safe moves
            int r = (move / 10) - 1;
            int c = (move % 10) - 1;
            
            board[r][c] = opp; // Simulate opponent making a move at this empty spot
            bool opp_wins = has_four_in_a_row(board, r, c, opp);
            board[r][c] = 0; // Undo
            
            if (opp_wins) {
                return move; 
            }
        }

        // Use minimax with alpha-beta pruning (only on safe moves)
        int bestVal = INT_MIN;
        int best_move = -1;
        
        // If there are safe moves, pick one using minimax
        if (!safe_moves.empty()) {
            best_move = safe_moves[0]; // Default to first safe move if minimax doesn't find better
        } else if (!all_moves.empty()) {
            best_move = all_moves[0]; // If no safe moves, default to first available move (likely losing)
        } else {
            return -1; // No moves available
        }


        for (int move : safe_moves) {
            int i = (move / 10) - 1;
            int j = (move % 10) - 1;
            
            board[i][j] = player;
            int moveVal = minimax(board, max_depth-1, false, INT_MIN, INT_MAX);
            board[i][j] = 0;
            
            if (moveVal > bestVal) {
                bestVal = moveVal;
                best_move = move;
            }
        }
        
        return best_move;
    }
    
private:
    bool has_four_in_a_row(int board[5][5], int row, int col, int p) {
        int directions[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
        
        for (int d = 0; d < 4; d++) {
            int dx = directions[d][0];
            int dy = directions[d][1];
            int count = 1; 
            
            for (int i = 1; i <= 3; i++) {
                int nr = row + i * dx;
                int nc = col + i * dy;
                if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5 && board[nr][nc] == p) {
                    count++;
                } else break;
            }
            
            for (int i = 1; i <= 3; i++) {
                int nr = row - i * dx;
                int nc = col - i * dy;
                if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5 && board[nr][nc] == p) {
                    count++;
                } else break;
            }
            
            if (count >= 4) {
                return true;
            }
        }
        return false;
    }

    bool would_create_losing_three(int board[5][5], int row, int col) {
        if (has_four_in_a_row(board, row, col, player)) {
            return false; 
        }
        
        int directions[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
        for (int d = 0; d < 4; d++) {
            int dx = directions[d][0];
            int dy = directions[d][1];
            
            int count = 1;
            
            for (int i = 1; i < 5; i++) { // Check up to board boundaries
                int nr = row + i * dx;
                int nc = col + i * dy;
                if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5 && board[nr][nc] == player) {
                    count++;
                } else {
                    break;
                }
            }
            
            for (int i = 1; i < 5; i++) {
                int nr = row - i * dx;
                int nc = col - i * dy;
                if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5 && board[nr][nc] == player) {
                    count++;
                } else {
                    break;
                }
            }
            
            // If the total line length is exactly 3
            if (count == 3) {
                return true; 
            }
        }
        
        return false;
    }
    
    bool check_winning_move(int current_board[5][5], int r, int c, int p_id) {
        current_board[r][c] = p_id; 
        bool is_win = has_four_in_a_row(current_board, r, c, p_id);
        current_board[r][c] = 0; 
        return is_win;
    }

    // Check for spacing patterns (X _ _ X and X _ X)
    int get_spaceing_bonus(int board[5][5], int i, int j, int p) {
        int bonus = 0;
        int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}}; 
        
        for (int d = 0; d < 4; d++) {
            int dx = directions[d][0];
            int dy = directions[d][1];
            
            // Check X _ _ X pattern (long spacing)
            int x1 = i + 3*dx, y1 = j + 3*dy;
            if (x1 >= 0 && x1 < 5 && y1 >= 0 && y1 < 5 && board[x1][y1] == p) {
                if (board[i + dx][j + dy] == 0 && board[i + 2*dx][j + 2*dy] == 0) {
                    bonus += LONG_SPACING_BONUS;
                }
            }
            
            int x2 = i - 3*dx, y2 = j - 3*dy; // Check other direction for symmetry
            if (x2 >= 0 && x2 < 5 && y2 >= 0 && y2 < 5 && board[x2][y2] == p) {
                 if (i - dx >= 0 && i - dx < 5 && j - dy >= 0 && j - dy < 5 && board[i - dx][j - dy] == 0 &&
                    i - 2*dx >= 0 && i - 2*dx < 5 && j - 2*dy >= 0 && j - 2*dy < 5 && board[i - 2*dx][j - 2*dy] == 0) {
                    bonus += LONG_SPACING_BONUS;
                }
            }
            
            // Check X _ X pattern (short spacing)
            int x3 = i + 2*dx, y3 = j + 2*dy;
            if (x3 >= 0 && x3 < 5 && y3 >= 0 && y3 < 5 && board[x3][y3] == p) {
                if (board[i + dx][j + dy] == 0) {
                    bonus += SHORT_SPACING_BONUS;
                }
            }
            
            int x4 = i - 2*dx, y4 = j - 2*dy; // Check other direction
             if (x4 >= 0 && x4 < 5 && y4 >= 0 && y4 < 5 && board[x4][y4] == p) {
                if (i - dx >= 0 && i - dx < 5 && j - dy >= 0 && j - dy < 5 && board[i - dx][j - dy] == 0) {
                    bonus += SHORT_SPACING_BONUS;
                }
            }
        }
        
        return bonus;
    }
    
    // Calculate adjacency penalty for clustering
    int get_adjacent_penalty(int board[5][5], int i, int j, int p) {
        int penalty = 0;
        int adjacent[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
        
        for (int a = 0; a < 8; a++) {
            int ni = i + adjacent[a][0];
            int nj = j + adjacent[a][1];
            
            if (ni >= 0 && ni < 5 && nj >= 0 && nj < 5 && board[ni][nj] == p) {
                penalty += CLUSTERING_PENALTY;
            }
        }
        
        return penalty;
    }
    
    // Calculate position-based bonuses
    int get_position_bonus(int board[5][5], int i, int j, int p) {
        int move = (i + 1) * 10 + (j + 1);
        int bonus = 0;
        
        if (move == 11 || move == 15 || move == 51 || move == 55) 
            bonus += CORNER_BONUS;
        else if (move == 22 || move == 24 || move == 42 || move == 44) 
            bonus += NEAR_CENTER_BONUS;
        else if (move == 23 || move == 32 || move == 34 || move == 43) 
            bonus += SECONDARY_BONUS;
        else if (i == 0 || i == 4 || j == 0 || j == 4) 
            bonus += EDGE_BONUS;
        
        bonus += get_spaceing_bonus(board, i, j, p);
        bonus += get_adjacent_penalty(board, i, j, p);
        
        return bonus;
    }
    
    int minimax(int board[5][5], int depth, bool is_max, int alpha, int beta) {
        nodes++;
        
        
        int static_eval = eval_pos(board);
        if (static_eval == WIN_SCORE || static_eval == LOSS_SCORE) {
            return static_eval;
        }

        if (depth == 0 || !is_moves_left(board)) {
            return static_eval;
        }
        
        int current_eval_player = is_max ? player : opp;
        
        if (is_max) {
            int maxEval = INT_MIN;
            for (int r = 0; r < 5; r++) {
                for (int c = 0; c < 5; c++) {
                    if (board[r][c] == 0) {
                        board[r][c] = current_eval_player;
                        if (would_create_losing_three(board, r, c)) {
                                board[r][c] = 0;
                                maxEval = std::max(maxEval, LOSS_SCORE);
                                alpha = std::max(alpha, LOSS_SCORE);

                                maxEval = std::max(maxEval, LOSS_SCORE);
                                alpha = std::max(alpha, LOSS_SCORE);
                                if (beta <= alpha) {
                                    board[r][c] = 0;
                                    return maxEval; // Beta cutoff
                                }
                             board[r][c] = 0; // backtrack before continue
                             continue; // Skip this losing move for the AI
                        }

                        int eval = minimax(board, depth - 1, false, alpha, beta);
                        board[r][c] = 0;
                        
                        maxEval = std::max(maxEval, eval);
                        alpha = std::max(alpha, eval);
                        
                        if (beta <= alpha) {
                            return maxEval; 
                        }
                    }
                }
            }
            return maxEval == INT_MIN ? (is_moves_left(board) ? 0 : static_eval) : maxEval;
        } else { 
            int minEval = INT_MAX;
            for (int r = 0; r < 5; r++) {
                for (int c = 0; c < 5; c++) {
                    if (board[r][c] == 0) {
                        board[r][c] = current_eval_player;

                        int eval = minimax(board, depth - 1, true, alpha, beta);
                        board[r][c] = 0; // Backtrack
                        
                        minEval = std::min(minEval, eval);
                        beta = std::min(beta, eval);
                        
                        if (beta <= alpha) {
                            return minEval; 
                        }
                    }
                }
            }
            return minEval == INT_MAX ? (is_moves_left(board) ? 0 : static_eval) : minEval;
        }
    }
    
    bool is_moves_left(int board[5][5]) {
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                if (board[i][j] == 0) return true;
        return false;
    }
    
    int eval_pos(int board[5][5]) {
        if (has_four_in_a_row_on_board(board, player)) return WIN_SCORE;
        if (has_four_in_a_row_on_board(board, opp)) return LOSS_SCORE;

        if (has_three_in_a_row_on_board_strict(board, player) && !has_four_in_a_row_on_board(board, player)) return LOSS_SCORE;
        if (has_three_in_a_row_on_board_strict(board, opp) && !has_four_in_a_row_on_board(board, opp)) return WIN_SCORE; // Opponent lost


        int score = 0;
        for (int i = 0; i < 5; i++) { // Horizontal
            for (int j = 0; j <= 5 - 4; j++) {
                int ai = 0, op_count = 0;
                for (int k = 0; k < 4; k++) {
                    if (board[i][j + k] == player) ai++;
                    else if (board[i][j + k] == opp) op_count++;
                }
                if (ai == 3 && op_count == 0) score += THREE_IN_ROW_SCORE;
                if (op_count == 3 && ai == 0) score += OPP_THREE_IN_ROW_SCORE;
                if (ai == 2 && op_count == 0) score += TWO_IN_ROW_SCORE;
                if (op_count == 2 && ai == 0) score += OPP_TWO_IN_ROW_SCORE;
            }
        }
        // Vertical
        for (int j = 0; j < 5; j++) {
            for (int i = 0; i <= 5 - 4; i++) {
                int ai = 0, op_count = 0;
                for (int k = 0; k < 4; k++) {
                    if (board[i + k][j] == player) ai++;
                    else if (board[i + k][j] == opp) op_count++;
                }
                if (ai == 3 && op_count == 0) score += THREE_IN_ROW_SCORE;
                if (op_count == 3 && ai == 0) score += OPP_THREE_IN_ROW_SCORE;
                if (ai == 2 && op_count == 0) score += TWO_IN_ROW_SCORE;
                if (op_count == 2 && ai == 0) score += OPP_TWO_IN_ROW_SCORE;
            }
        }
        // Diagonal (top-left to bottom-right)
        for (int i = 0; i <= 5 - 4; i++) {
            for (int j = 0; j <= 5 - 4; j++) {
                int ai = 0, op_count = 0;
                for (int k = 0; k < 4; k++) {
                    if (board[i + k][j + k] == player) ai++;
                    else if (board[i + k][j + k] == opp) op_count++;
                }
                if (ai == 3 && op_count == 0) score += THREE_IN_ROW_SCORE;
                if (op_count == 3 && ai == 0) score += OPP_THREE_IN_ROW_SCORE;
                if (ai == 2 && op_count == 0) score += TWO_IN_ROW_SCORE;
                if (op_count == 2 && ai == 0) score += OPP_TWO_IN_ROW_SCORE;
            }
        }
        // Diagonal (top-right to bottom-left)
        for (int i = 0; i <= 5 - 4; i++) {
            for (int j = 3; j < 5; j++) { // Start j from 3 (0-indexed) for a 4-length diagonal
                int ai = 0, op_count = 0;
                for (int k = 0; k < 4; k++) {
                    if (board[i + k][j - k] == player) ai++;
                    else if (board[i + k][j - k] == opp) op_count++;
                }
                if (ai == 3 && op_count == 0) score += THREE_IN_ROW_SCORE;
                if (op_count == 3 && ai == 0) score += OPP_THREE_IN_ROW_SCORE;
                if (ai == 2 && op_count == 0) score += TWO_IN_ROW_SCORE;
                if (op_count == 2 && ai == 0) score += OPP_TWO_IN_ROW_SCORE;
            }
        }

        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 5; c++) {
                if (board[r][c] == player) {
                    score += get_position_bonus(board, r, c, player);
                } else if (board[r][c] == opp) {
                    score -= get_position_bonus(board, r, c, opp) / 2;
                }
            }
        }
        return score;
    }

    bool has_four_in_a_row_on_board(int board[5][5], int p_id) {
        for (int r = 0; r < 5; ++r) {
            for (int c = 0; c < 5; ++c) {
                if (board[r][c] == p_id) {
                    if (has_four_in_a_row(board, r, c, p_id)) return true;
                }
            }
        }
        return false;
    }
    
    bool has_three_in_a_row_on_board_strict(int board[5][5], int p_id) {
        for (int r = 0; r < 5; ++r) {
            for (int c = 0; c < 5; ++c) {
                if (board[r][c] == p_id) { // Check lines starting/passing through this piece
                    int directions[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
                    for (int d = 0; d < 4; d++) {
                        int dx = directions[d][0];
                        int dy = directions[d][1];
                        int count = 1;
                        for (int k = 1; k < 5; k++) {
                            int nr = r + k * dx; int nc = c + k * dy;
                            if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5 && board[nr][nc] == p_id) count++; else break;
                        }
                        for (int k = 1; k < 5; k++) {
                            int nr = r - k * dx; int nc = c - k * dy;
                            if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5 && board[nr][nc] == p_id) count++; else break;
                        }
                        if (count == 3) return true; // Found a line of exactly 3
                    }
                }
            }
        }
        return false;
    }
    
    std::vector<int> gen_moves(int board[5][5]) {
        std::vector<int> moves;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (board[i][j] == 0) {
                    moves.push_back((i + 1) * 10 + (j + 1));
                }
            }
        }
        return moves;
    }
    
    void make_move(int board[5][5], int move, int p) {
        board[(move / 10) - 1][(move % 10) - 1] = p;
    }
    
    void undo_move(int board[5][5], int move) {
        board[(move / 10) - 1][(move % 10) - 1] = 0;
    }
    
    long long get_nodes() const { return nodes; }
};
