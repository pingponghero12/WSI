#pragma once
#include "board.h"
#include <climits>
#include <vector>
#include <algorithm>
/*
* Heuristic evaluation function for non-terminal positions
* 
*Operational win/loss patterns
* +100000: AI has 4-in-a-row (WIN)
* -100000: Opponent has 4-in-a-row (LOSE)
* -99999:  AI has 3-in-a-row without 4th possibility (LOSE due to special rule)
* +99999:  Opponent forced into losing 3-in-a-row
* 
*  Tactical:
* +5000:   AI has open 3-in-a-row (can extend to 4)
* -5000:   Opponent has open 3-in-a-row (must block)
* +1000:   AI has open 2-in-a-row (good development)
* -1000:   Opponent has open 2-in-a-row
* 
* Strategic bonuses:
* +50:     Center control (33 position)
* +10:      Connected pieces (general connectivity)
*/

class Minimax {
private:
    int max_depth;
    int player;
    int opp;
    long long nodes;
    
public:
    Minimax(int depth, int p) : max_depth(depth), player(p) {
        opp = (p == 1) ? 2 : 1;
        nodes = 0;
    }
    
    int get_best_move(int board[5][5]) {
        nodes = 0;
        int best_move = -1;
        int best_score = INT_MIN;
        
        std::vector<int> moves = gen_moves(board);
        
        for (int move : moves) {
            make_move(board, move, player);
            int score = minimax(board, max_depth - 1, false, INT_MIN, INT_MAX);
            undo_move(board, move);
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
        }
        
        return best_move;
    }
    
private:
    int minimax(int board[5][5], int depth, bool is_max, int alpha, int beta) {
        nodes++;
        
        if (depth == 0) {
            return eval_pos(board);
        }
        
        int game_result = check_game_over(board);
        if (game_result != 0) {
            return game_result;
        }
        
        int curr_player = is_max ? player : opp;
        std::vector<int> moves = gen_moves(board);
        
        if (is_max) {
            int max_eval = INT_MIN;
            
            for (int move : moves) {
                make_move(board, move, curr_player);
                int eval = minimax(board, depth - 1, false, alpha, beta);
                undo_move(board, move);
                
                max_eval = std::max(max_eval, eval);
                alpha = std::max(alpha, eval);
                
                if (beta <= alpha) {
                    break;
                }
            }
            return max_eval;
            
        } else {
            int min_eval = INT_MAX;
            
            for (int move : moves) {
                make_move(board, move, curr_player);
                int eval = minimax(board, depth - 1, true, alpha, beta);
                undo_move(board, move);
                
                min_eval = std::min(min_eval, eval);
                beta = std::min(beta, eval);
                
                if (beta <= alpha) {
                    break;
                }
            }
            return min_eval;
        }
    }
    
    int eval_pos(int board[5][5]) {
        int score = 0;
        
        for (int i = 0; i < 28; i++) {
            score += eval_win_pattern(board, i);
        }
        
        for (int i = 0; i < 48; i++) {
            score += eval_lose_pattern(board, i);
        }
        
        score += eval_positional(board);
        
        return score;
    }
    
    int eval_win_pattern(int board[5][5], int idx) {
        int p_count = 0, o_count = 0, empty = 0;
        
        for (int j = 0; j < 4; j++) {
            int row = win[idx][j][0];
            int col = win[idx][j][1];
            
            if (board[row][col] == player) p_count++;
            else if (board[row][col] == opp) o_count++;
            else empty++;
        }
        
        if (p_count > 0 && o_count > 0) return 0;
        
        if (p_count == 4) return 100000;
        if (o_count == 4) return -100000;
        if (p_count == 3) return 5000;
        if (o_count == 3) return -5000;
        if (p_count == 2) return 1000;
        if (o_count == 2) return -1000;
        
        return 0;
    }
    
    int eval_lose_pattern(int board[5][5], int idx) {
        int p_count = 0, o_count = 0;
        
        for (int j = 0; j < 3; j++) {
            int row = lose[idx][j][0];
            int col = lose[idx][j][1];
            
            if (board[row][col] == player) p_count++;
            else if (board[row][col] == opp) o_count++;
        }
        
        bool can_ext = can_extend_to_four(board, idx);
        
        if (p_count == 3 && !can_ext) return -99999;
        if (o_count == 3 && !can_ext) return +99999;
        
        return 0;
    }
    
    bool can_extend_to_four(int board[5][5], int lose_idx) {
        for (int win_idx = 0; win_idx < 28; win_idx++) {
            int match = 0;
            int empty_win = 0;
            
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    if (lose[lose_idx][i][0] == win[win_idx][j][0] && 
                        lose[lose_idx][i][1] == win[win_idx][j][1]) {
                        match++;
                        break;
                    }
                }
            }
            
            if (match == 3) {
                for (int j = 0; j < 4; j++) {
                    int row = win[win_idx][j][0];
                    int col = win[win_idx][j][1];
                    if (board[row][col] == 0) empty_win++;
                }
                if (empty_win > 0) return true;
            }
        }
        return false;
    }
    
    int eval_positional(int board[5][5]) {
        int score = 0;
        
        if (board[2][2] == player) score += 50;
        else if (board[2][2] == opp) score -= 50;
        
        // Connected pieces bonus
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (board[i][j] == player) {
                    // Check all 8 directions for adjacent pieces
                    int directions[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};
                    
                    for (int d = 0; d < 8; d++) {
                        int ni = i + directions[d][0];
                        int nj = j + directions[d][1];
                        
                        if (ni >= 0 && ni < 5 && nj >= 0 && nj < 5 && board[ni][nj] == player) {
                            score += 5; // +5 per connection (so each pair gets +10 total)
                        }
                    }
                } else if (board[i][j] == opp) {
                    // Same for opponent
                    int directions[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};
                    
                    for (int d = 0; d < 8; d++) {
                        int ni = i + directions[d][0];
                        int nj = j + directions[d][1];
                        
                        if (ni >= 0 && ni < 5 && nj >= 0 && nj < 5 && board[ni][nj] == opp) {
                            score -= 5;
                        }
                    }
                }
            }
        }
        return score;
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
    
    int check_game_over(int board[5][5]) {
        if (winCheck(player)) return 100000;
        if (winCheck(opp)) return -100000;
        
        if (loseCheck(player)) return -100000;
        if (loseCheck(opp)) return 100000;
        
        bool full = true;
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                if (board[i][j] == 0) {
                    full = false;
                    break;
                }
            }
            if (!full) break;
        }
        
        return full ? 1 : 0;
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
