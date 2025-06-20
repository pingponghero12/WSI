#pragma once
#include "board.h"
#include <climits>
#include <vector>
#include <algorithm>
#include <random>

/*
 * HEURISTIC EVALUATION FUNCTION
 * =====================================
 *
 * 1. WINNING/THREAT DETECTION (Primary):
 *    - 4-in-a-row: ±100,000 (win/loss)
 *    - 3-in-a-row: +10,000/-15,000 (prioritizes blocking opponent threats)
 *    - 2-in-a-row: +1,000/-1,500
 *
 * 2. POSITION EVALUATION (Strategic):
 *    - Center tile (33): -50,000 (MASSIVE penalty - center is "dead" in 5x5)
 *    - Corners (11,15,51,55): +5,000 (maximum tactical flexibility)
 *    - Near-center (22,24,42,44): +3,000
 *    - Secondary positions: +1,500
 *    - Edge positions: +800
 *
 * 3. PATTERN RECOGNITION (Advanced):
 *    - X _ _ X spacing: +8,000 (creates multiple winning threats)
 *    - Adjacent clustering: -5,000 per piece (prevents flexibility loss)
 *
 * PENALTIES TARGET:
 * - Center occupation (-50,000): Strategic dead zone
 * - Piece clustering (-5,000 each): Reduces tactical options
 * - Opponent threats (-15,000): Defense prioritized over offense
 * - Poor positioning: Non-strategic square placement
 */

class Minimax {
private:
    int max_depth;
    int player;
    int opp;
    long long nodes;
    std::mt19937 rng;  // Using mt19937 for randomness
    
public:
    Minimax(int depth, int p) : max_depth(depth), player(p) {
        opp = (p == 1) ? 2 : 1;
        nodes = 0;
        
        // Initialize random number generator with time-based seed
        std::random_device rd;
        rng = std::mt19937(rd());
    }
    
    int get_best_move(int board[5][5]) {
        nodes = 0;
        int bestVal = INT_MIN;
        int move = -1;

        // FIRST PRIORITY: Check if we can win immediately (in one move)
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (board[i][j] == 0) {
                    if (checkWinningMove(board, i, j, player)) {
                        return (i + 1) * 10 + (j + 1);
                    }
                }
            }
        }

        // SECOND PRIORITY: Block opponent's winning moves (in one move)
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (board[i][j] == 0) {
                    if (checkWinningMove(board, i, j, opp)) {
                        return (i + 1) * 10 + (j + 1);
                    }
                }
            }
        }

        // THIRD PRIORITY: Use minimax with alpha-beta pruning
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (board[i][j] == 0) {
                    board[i][j] = player;
                    int moveVal = minimax(board, max_depth-1, false, INT_MIN, INT_MAX);
                    board[i][j] = 0;
                    
                    if (moveVal > bestVal) {
                        bestVal = moveVal;
                        move = (i + 1) * 10 + (j + 1);
                    }
                }
            }
        }
        
        return move;
    }
    
private:
    // Check if placing a piece at (row, col) creates a winning line
    bool checkWinningMove(int board[5][5], int row, int col, int p) {
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
    bool creates_losing_three(int board[5][5], int row, int col, int p) {
        board[row][col] = p;
        
        int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}};
        
        for (int d = 0; d < 4; d++) {
            int dx = directions[d][0];
            int dy = directions[d][1];
            
            // Check each possible starting position for a 3-in-a-row
            for (int start = -2; start <= 0; start++) {
                int count = 0;
                bool can_extend = false;
                
                // Count pieces in this potential 3-in-a-row
                for (int i = 0; i < 3; i++) {
                    int nr = row + (start + i) * dx;
                    int nc = col + (start + i) * dy;
                    
                    if (nr >= 0 && nr < 5 && nc >= 0 && nc < 5 && board[nr][nc] == p) {
                        count++;
                    } else {
                        count = -1; // Not a 3-in-a-row
                        break;
                    }
                }
                
                if (count == 3) {
                    // Check if this 3-in-a-row can be extended to 4
                    
                    // Check one position before
                    int nr_before = row + (start - 1) * dx;
                    int nc_before = col + (start - 1) * dy;
                    if (nr_before >= 0 && nr_before < 5 && nc_before >= 0 && nc_before < 5 && 
                        board[nr_before][nc_before] == 0) {
                        can_extend = true;
                    }
                    
                    // Check one position after
                    int nr_after = row + (start + 3) * dx;
                    int nc_after = col + (start + 3) * dy;
                    if (nr_after >= 0 && nr_after < 5 && nc_after >= 0 && nc_after < 5 && 
                        board[nr_after][nc_after] == 0) {
                        can_extend = true;
                    }
                    
                    if (!can_extend) {
                        board[row][col] = 0;
                        return true; // Found a losing 3-in-a-row
                    }
                }
            }
        }
        
        board[row][col] = 0;
        return false;
    }
    
    // Check for spacing patterns (X _ _ X)
    int getSpacingBonus(int board[5][5], int i, int j, int p) {
        int bonus = 0;
        int directions[4][2] = {{0,1}, {1,0}, {1,1}, {1,-1}}; 
        
        for (int d = 0; d < 4; d++) {
            int dx = directions[d][0];
            int dy = directions[d][1];
            
            // Check forward pattern X _ _ X
            int x1 = i + 3*dx, y1 = j + 3*dy;
            if (x1 >= 0 && x1 < 5 && y1 >= 0 && y1 < 5 && board[x1][y1] == p) {
                if (board[i + dx][j + dy] == 0 && board[i + 2*dx][j + 2*dy] == 0) {
                    bonus += 8000; 
                }
            }
            
            // Check backward pattern X _ _ X
            int x2 = i - 3*dx, y2 = j - 3*dy;
            if (x2 >= 0 && x2 < 5 && y2 >= 0 && y2 < 5 && board[x2][y2] == p) {
                if (board[i - dx][j - dy] == 0 && board[i - 2*dx][j - 2*dy] == 0) {
                    bonus += 8000; 
                }
            }
        }
        
        return bonus;
    }
    
    // Calculate adjacency penalty for clustering
    int getAdjacentPenalty(int board[5][5], int i, int j, int p) {
        int penalty = 0;
        int adjacent[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
        
        for (int a = 0; a < 8; a++) {
            int ni = i + adjacent[a][0];
            int nj = j + adjacent[a][1];
            
            if (ni >= 0 && ni < 5 && nj >= 0 && nj < 5 && board[ni][nj] == p) {
                penalty -= 5000; 
            }
        }
        
        return penalty;
    }
    
    // Calculate position-based bonuses
    int getPositionBonus(int board[5][5], int i, int j, int p) {
        int move = (i + 1) * 10 + (j + 1);
        int bonus = 0;
        
        // HEAVILY penalize center tile
        if (move == 33) return -50000;
        
        // Prioritize corners
        if (move == 11 || move == 15 || move == 51 || move == 55) bonus += 5000;
        
        // Favor tiles near center (but not dead center)
        else if (move == 22 || move == 24 || move == 42 || move == 44) bonus += 3000;
        
        // Secondary good positions
        else if (move == 23 || move == 32 || move == 34 || move == 43) bonus += 1500;
        
        // Edge positions
        else if (i == 0 || i == 4 || j == 0 || j == 4) bonus += 800;
        
        // Add spacing bonus (X _ _ X pattern)
        bonus += getSpacingBonus(board, i, j, p);
        
        // Subtract clustering penalty
        bonus += getAdjacentPenalty(board, i, j, p);
        
        return bonus;
    }
    
    int minimax(int board[5][5], int depth, bool is_max, int alpha, int beta) {
        nodes++;
        
        // Check for terminal states (wins, losses, or depth limit)
        if (depth == 0 || !isMovesLeft(board)) {
            return eval_pos(board);
        }
        
        int curr_player = is_max ? player : opp;
        int opponent = 3 - curr_player;
        
        if (is_max) {
            int maxEval = INT_MIN;
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    if (board[i][j] == 0) {
                        // Skip moves that create a losing 3-in-a-row
                        if (creates_losing_three(board, i, j, curr_player)) {
                            continue;
                        }
                        
                        board[i][j] = curr_player;
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
            return maxEval;
        } else {
            int minEval = INT_MAX;
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    if (board[i][j] == 0) {
                        // For opponent, we don't skip losing moves
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
            return minEval;
        }
    }
    
    bool isMovesLeft(int board[5][5]) {
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
                
                if (ai == 4) score += 100000;
                if (op == 4) score -= 100000;
                if (ai == 3 && op == 0) score += 10000; 
                if (op == 3 && ai == 0) score -= 15000;
                if (ai == 2 && op == 0) score += 1000;
                if (op == 2 && ai == 0) score -= 1500;
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
                
                if (ai == 4) score += 100000;
                if (op == 4) score -= 100000;
                if (ai == 3 && op == 0) score += 10000;
                if (op == 3 && ai == 0) score -= 15000;
                if (ai == 2 && op == 0) score += 1000;
                if (op == 2 && ai == 0) score -= 1500;
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
                
                if (ai == 4) score += 100000;
                if (op == 4) score -= 100000;
                if (ai == 3 && op == 0) score += 10000;
                if (op == 3 && ai == 0) score -= 15000;
                if (ai == 2 && op == 0) score += 1000;
                if (op == 2 && ai == 0) score -= 1500;
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
                
                if (ai == 4) score += 100000;
                if (op == 4) score -= 100000;
                if (ai == 3 && op == 0) score += 10000;
                if (op == 3 && ai == 0) score -= 15000;
                if (ai == 2 && op == 0) score += 1000;
                if (op == 2 && ai == 0) score -= 1500;
            }
        }

        // Add position-based evaluation
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (board[i][j] == player) {
                    score += getPositionBonus(board, i, j, player);
                } else if (board[i][j] == opp) {
                    score -= getPositionBonus(board, i, j, opp) / 3;
                }
            }
        }
        
        return score;
    }
    
    // Helper functions to maintain the same interface as your original class
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
    
    int check_game_over(int board[5][5]) {
        // This is kept for compatibility but not used in the new implementation
        return 0;
    }
    
    long long get_nodes() const { return nodes; }
};
