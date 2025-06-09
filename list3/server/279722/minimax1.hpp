#pragma once
#include <climits>
#include <vector>
#include <algorithm>
#include <random>

// Win/Loss evaluation values
#define WIN_SCORE 100000
#define LOSS_SCORE -100000

// Threat evaluation values
#define THREE_IN_ROW_SCORE 10000
#define OPP_THREE_IN_ROW_SCORE -15000
#define TWO_IN_ROW_SCORE 1000
#define OPP_TWO_IN_ROW_SCORE -1500

// Pattern recognition values
#define LONG_SPACING_BONUS 8000  // X _ _ X pattern
#define SHORT_SPACING_BONUS 3000 // X _ X pattern
#define CLUSTERING_PENALTY -2000 // Adjacent piece penalty

// Position evaluation values
#define CORNER_BONUS 5000      // Positions 11, 15, 51, 55
#define NEAR_CENTER_BONUS 3000 // Positions 22, 24, 42, 44
#define SECONDARY_BONUS 1500   // Positions 23, 32, 34, 43
#define EDGE_BONUS 800         // Any edge position

/*
 * EVALUATION VALUES
 * =====================================
 *
 * 1. ROW PATTERNS:
 *    - 4-in-a-row: +-100,000 (win/loss)
 *    - 3-in-a-row: +10,000/-15,000
 *    - 2-in-a-row: +1,000/-1,500
 *
 * 2. BOARD POSITIONS:
 *    - Corners (11,15,51,55): +5,000
 *    - Near-center (22,24,42,44): +3,000
 *    - Secondary positions: +1,500
 *    - Edge positions: +800
 *
 * 3. SPECIAL PATTERNS:
 *    - X _ _ X pattern: +8,000
 *    - X _ X pattern: +3,000
 *    - Adjacent pieces: -2,000 per piece
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
        
        // Collect all valid moves
        std::vector<int> all_moves;
        std::vector<int> safe_moves;
        
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (board[i][j] == 0) {
                    int move = (i + 1) * 10 + (j + 1);
                    all_moves.push_back(move);
                    
                    // Check if this move is safe (doesn't create a losing 3-in-a-row)
                    board[i][j] = player;
                    if (!would_create_losing_three(board, i, j)) {
                        safe_moves.push_back(move);
                    }
                    board[i][j] = 0;
                }
            }
        }
        
        // If no safe moves, use any legal move
        if (safe_moves.empty() && !all_moves.empty()) {
            return all_moves[0];
        }
        
        // FIRST PRIORITY: Check if we can win immediately (in one move)
        for (int move : safe_moves) {
            int i = (move / 10) - 1;
            int j = (move % 10) - 1;
            
            if (check_winning_move(board, i, j, player)) {
                return move;
            }
        }

        // SECOND PRIORITY: Block opponent's winning moves (in one move)
        for (int move : safe_moves) {
            int i = (move / 10) - 1;
            int j = (move % 10) - 1;
            
            if (check_winning_move(board, i, j, opp)) {
                return move;
            }
        }

        // THIRD PRIORITY: Use minimax with alpha-beta pruning (only on safe moves)
        int bestVal = INT_MIN;
        int best_move = -1;
        
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
        
        // If best_move is -1 but we have safe moves, pick the first safe move
        if (best_move == -1 && !safe_moves.empty()) {
            return safe_moves[0];
        }
        
        return best_move;
    }
    
private:
    // Check if placing a piece at (row, col) creates a winning line
    bool check_winning_move(int board[5][5], int row, int col, int p) {
        board[row][col] = p; 
        
        int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}};
        
        for (int d = 0; d < 4; d++) {
            int dx = directions[d][0];
            int dy = directions[d][1];
            int count = 1;         
          
            for (int i = 1; i < 4; i++) {
                int nr = row + i * dx;
                int nc = col + i * dy;
                if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5 && board[nr][nc] == p) {
                    count++;
                } else break;
            }
            
            for (int i = 1; i < 4; i++) {
                int nr = row - i * dx;
                int nc = col - i * dy;
                if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5 && board[nr][nc] == p) {
                    count++;
                } else break;
            }
            
            if (count >= 4) {
                board[row][col] = 0; 
                return true;
            }
        }
        
        board[row][col] = 0; 
        return false;
    }
    
    // Check if a move would create a 3-in-a-row with no possibility to extend
    bool would_create_losing_three(int board[5][5], int row, int col) {
        // Assume our piece is already at (row, col)
        
        // Check if this move creates a 4-in-a-row (which would override any 3-in-a-row loss)
        if (has_four_in_a_row(board, row, col, player)) {
            return false; // Not a losing move if it creates a 4-in-a-row
        }
        
        // Check all 4 directions
        int directions[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
        
        for (int d = 0; d < 4; d++) {
            int dx = directions[d][0];
            int dy = directions[d][1];
            
            // Count consecutive pieces in both directions
            int count = 1; // Start with 1 for the current piece
            
            // Count forward
            for (int i = 1; i <= 2; i++) {
                int nx = row + i * dx;
                int ny = col + i * dy;
                if (nx >= 0 && nx < 5 && ny >= 0 && ny < 5 && board[nx][ny] == player) {
                    count++;
                } else {
                    break;
                }
            }
            
            // Count backward
            for (int i = 1; i <= 2; i++) {
                int nx = row - i * dx;
                int ny = col - i * dy;
                if (nx >= 0 && nx < 5 && ny >= 0 && ny < 5 && board[nx][ny] == player) {
                    count++;
                } else {
                    break;
                }
            }
            
            // Check if we have exactly 3 in a row
            if (count == 3) {
                // Now check if this 3-in-a-row can be extended to a 4-in-a-row
                bool can_extend = false;
                
                // Find the endpoints of the 3-in-a-row
                int start_x = row, start_y = col;
                int end_x = row, end_y = col;
                
                // Find the start point (backward)
                for (int i = 1; i <= 2; i++) {
                    int nx = row - i * dx;
                    int ny = col - i * dy;
                    if (nx >= 0 && nx < 5 && ny >= 0 && ny < 5 && board[nx][ny] == player) {
                        start_x = nx;
                        start_y = ny;
                    } else {
                        break;
                    }
                }
                
                // Find the end point (forward)
                for (int i = 1; i <= 2; i++) {
                    int nx = row + i * dx;
                    int ny = col + i * dy;
                    if (nx >= 0 && nx < 5 && ny >= 0 && ny < 5 && board[nx][ny] == player) {
                        end_x = nx;
                        end_y = ny;
                    } else {
                        break;
                    }
                }
                
                // Check position before the 3-in-a-row
                int before_x = start_x - dx;
                int before_y = start_y - dy;
                if (before_x >= 0 && before_x < 5 && before_y >= 0 && before_y < 5 && board[before_x][before_y] == 0) {
                    can_extend = true;
                }
                
                // Check position after the 3-in-a-row
                int after_x = end_x + dx;
                int after_y = end_y + dy;
                if (after_x >= 0 && after_x < 5 && after_y >= 0 && after_y < 5 && board[after_x][after_y] == 0) {
                    can_extend = true;
                }
                
                if (!can_extend) {
                    return true; // This is a losing 3-in-a-row
                }
            }
        }
        
        return false; // No losing 3-in-a-row found
    }
    
    // Check if a position has a 4-in-a-row
    bool has_four_in_a_row(int board[5][5], int row, int col, int p) {
        int directions[4][2] = {{1,0}, {0,1}, {1,1}, {1,-1}};
        
        for (int d = 0; d < 4; d++) {
            int dx = directions[d][0];
            int dy = directions[d][1];
            
            // Count consecutive pieces in both directions
            int count = 1; // Start with 1 for the current piece
            
            // Count forward
            for (int i = 1; i <= 3; i++) {
                int nx = row + i * dx;
                int ny = col + i * dy;
                if (nx >= 0 && nx < 5 && ny >= 0 && ny < 5 && board[nx][ny] == p) {
                    count++;
                } else {
                    break;
                }
            }
            
            // Count backward
            for (int i = 1; i <= 3; i++) {
                int nx = row - i * dx;
                int ny = col - i * dy;
                if (nx >= 0 && nx < 5 && ny >= 0 && ny < 5 && board[nx][ny] == p) {
                    count++;
                } else {
                    break;
                }
            }
            
            if (count >= 4) {
                return true;
            }
        }
        
        return false;
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
            
            int x2 = i - 3*dx, y2 = j - 3*dy;
            if (x2 >= 0 && x2 < 5 && y2 >= 0 && y2 < 5 && board[x2][y2] == p) {
                if (board[i - dx][j - dy] == 0 && board[i - 2*dx][j - 2*dy] == 0) {
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
            
            int x4 = i - 2*dx, y4 = j - 2*dy;
            if (x4 >= 0 && x4 < 5 && y4 >= 0 && y4 < 5 && board[x4][y4] == p) {
                if (board[i - dx][j - dy] == 0) {
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
        
        // Prioritize corners
        if (move == 11 || move == 15 || move == 51 || move == 55) 
            bonus += CORNER_BONUS;
        
        // Favor tiles near center
        else if (move == 22 || move == 24 || move == 42 || move == 44) 
            bonus += NEAR_CENTER_BONUS;
        
        // Secondary good positions
        else if (move == 23 || move == 32 || move == 34 || move == 43) 
            bonus += SECONDARY_BONUS;
        
        // Edge positions
        else if (i == 0 || i == 4 || j == 0 || j == 4) 
            bonus += EDGE_BONUS;
        
        // Add spacing bonus patterns
        bonus += get_spaceing_bonus(board, i, j, p);
        
        // Add clustering penalty
        bonus += get_adjacent_penalty(board, i, j, p);
        
        return bonus;
    }
    
    int minimax(int board[5][5], int depth, bool is_max, int alpha, int beta) {
        nodes++;
        
        // Check for terminal states (wins, losses, or depth limit)
        if (depth == 0 || !is_moves_left(board)) {
            return eval_pos(board);
        }
        
        int curr_player = is_max ? player : opp;
        
        if (is_max) {
            int maxEval = INT_MIN;
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    if (board[i][j] == 0) {
                        // Skip moves that create a losing 3-in-a-row
                        board[i][j] = curr_player;
                        if (would_create_losing_three(board, i, j)) {
                            board[i][j] = 0;
                            continue;
                        }
                        
                        int eval = minimax(board, depth - 1, false, alpha, beta);
                        board[i][j] = 0;
                        
                        maxEval = std::max(maxEval, eval);
                        alpha = std::max(alpha, eval);
                        
                        if (beta <= alpha) {
                            return maxEval; // Beta cutoff
                        }
                    }
                }
            }
            return maxEval == INT_MIN ? 0 : maxEval; // Return 0 if no valid moves
        } else {
            int minEval = INT_MAX;
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    if (board[i][j] == 0) {
                        board[i][j] = curr_player;
                        int eval = minimax(board, depth - 1, true, alpha, beta);
                        board[i][j] = 0;
                        
                        minEval = std::min(minEval, eval);
                        beta = std::min(beta, eval);
                        
                        if (beta <= alpha) {
                            return minEval; // Alpha cutoff
                        }
                    }
                }
            }
            return minEval == INT_MAX ? 0 : minEval; // Return 0 if no valid moves
        }
    }
    
    bool is_moves_left(int board[5][5]) {
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                if (board[i][j] == 0) return true;
        return false;
    }
    
    int eval_pos(int board[5][5]) {
        int score = 0;
        
        // Check horizontal lines
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j <= 5 - 4; j++) {
                int ai = 0, op = 0, empty = 0;
                for (int k = 0; k < 4; k++) {
                    if (board[i][j + k] == player) ai++;
                    else if (board[i][j + k] == opp) op++;
                    else empty++;
                }
                
                if (ai == 4) score += WIN_SCORE;
                if (op == 4) score += LOSS_SCORE;
                if (ai == 3 && op == 0) score += THREE_IN_ROW_SCORE;
                if (op == 3 && ai == 0) score += OPP_THREE_IN_ROW_SCORE;
                if (ai == 2 && op == 0) score += TWO_IN_ROW_SCORE;
                if (op == 2 && ai == 0) score += OPP_TWO_IN_ROW_SCORE;
            }
        }
        
        // Check vertical lines
        for (int i = 0; i <= 5 - 4; i++) {
            for (int j = 0; j < 5; j++) {
                int ai = 0, op = 0, empty = 0;
                for (int k = 0; k < 4; k++) {
                    if (board[i + k][j] == player) ai++;
                    else if (board[i + k][j] == opp) op++;
                    else empty++;
                }
                
                if (ai == 4) score += WIN_SCORE;
                if (op == 4) score += LOSS_SCORE;
                if (ai == 3 && op == 0) score += THREE_IN_ROW_SCORE;
                if (op == 3 && ai == 0) score += OPP_THREE_IN_ROW_SCORE;
                if (ai == 2 && op == 0) score += TWO_IN_ROW_SCORE;
                if (op == 2 && ai == 0) score += OPP_TWO_IN_ROW_SCORE;
            }
        }
        
        // Check diagonal lines (top-left to bottom-right)
        for (int i = 0; i <= 5 - 4; i++) {
            for (int j = 0; j <= 5 - 4; j++) {
                int ai = 0, op = 0, empty = 0;
                for (int k = 0; k < 4; k++) {
                    if (board[i + k][j + k] == player) ai++;
                    else if (board[i + k][j + k] == opp) op++;
                    else empty++;
                }
                
                if (ai == 4) score += WIN_SCORE;
                if (op == 4) score += LOSS_SCORE;
                if (ai == 3 && op == 0) score += THREE_IN_ROW_SCORE;
                if (op == 3 && ai == 0) score += OPP_THREE_IN_ROW_SCORE;
                if (ai == 2 && op == 0) score += TWO_IN_ROW_SCORE;
                if (op == 2 && ai == 0) score += OPP_TWO_IN_ROW_SCORE;
            }
        }
        
        // Check diagonal lines (top-right to bottom-left)
        for (int i = 0; i <= 5 - 4; i++) {
            for (int j = 3; j < 5; j++) {
                int ai = 0, op = 0, empty = 0;
                for (int k = 0; k < 4; k++) {
                    if (board[i + k][j - k] == player) ai++;
                    else if (board[i + k][j - k] == opp) op++;
                    else empty++;
                }
                
                if (ai == 4) score += WIN_SCORE;
                if (op == 4) score += LOSS_SCORE;
                if (ai == 3 && op == 0) score += THREE_IN_ROW_SCORE;
                if (op == 3 && ai == 0) score += OPP_THREE_IN_ROW_SCORE;
                if (ai == 2 && op == 0) score += TWO_IN_ROW_SCORE;
                if (op == 2 && ai == 0) score += OPP_TWO_IN_ROW_SCORE;
            }
        }

        // Add position-based evaluation
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (board[i][j] == player) {
                    score += get_position_bonus(board, i, j, player);
                } else if (board[i][j] == opp) {
                    score -= get_position_bonus(board, i, j, opp) / 3;
                }
            }
        }
        
        return score;
    }
    
    // Helper functions to maintain the same interface as the original class
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
        int row = (move / 10) - 1;
        int col = (move % 10) - 1;
        board[row][col] = p;
    }
    
    void undo_move(int board[5][5], int move) {
        int row = (move / 10) - 1;
        int col = (move % 10) - 1;
        board[row][col] = 0;
    }
    
    long long get_nodes() const { return nodes; }
};
