#pragma once
#include <vector>
#include <array>

struct Node {
    std::vector<int> board;
    int g_cost;
    int h_cost;
    int empty_idx;
    int f_cost() const { return g_cost + h_cost; }
    bool operator>(const Node& other) const {
        if (f_cost() != other.f_cost()) return f_cost() > other.f_cost();
        return h_cost > other.h_cost;
    }
};

struct AStarResult {
    std::vector<std::vector<int>> path;
    int visited_states;
    bool success;
};
