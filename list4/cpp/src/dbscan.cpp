#include "dbscan.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <queue>

DBSCAN::DBSCAN(double eps_param, int min_samples_param) : 
    eps(eps_param), 
    min_samples(min_samples_param),
    n_clusters(0),
    n_noise(0) {}

double DBSCAN::euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::vector<int> DBSCAN::find_neighbors(const std::vector<std::vector<double>>& X, int point_idx) {
    std::vector<int> neighbors;
    
    for (size_t i = 0; i < X.size(); ++i) {
        if (int(i) != point_idx && euclidean_distance(X[point_idx], X[i]) <= eps) {
            neighbors.push_back(i);
        }
    }
    
    return neighbors;
}

void DBSCAN::expand_cluster(const std::vector<std::vector<double>>& X, int point_idx, 
                          int cluster_id, std::vector<bool>& visited,
                          std::vector<std::vector<int>>& neighbors_cache) {
    // Mark the point as part of the current cluster
    labels[point_idx] = cluster_id;
    
    // Get all points in the eps neighborhood
    std::vector<int>& neighbors = neighbors_cache[point_idx];
    if (neighbors.empty()) {
        neighbors = find_neighbors(X, point_idx);
    }
    
    // If not a core point, just assign to cluster but don't expand
    if (neighbors.size() < static_cast<size_t>(min_samples)) {
        return;
    }
    
    // Process all neighbors
    std::queue<int> seeds;
    for (int neighbor : neighbors) {
        if (!visited[neighbor]) {
            visited[neighbor] = true;
            
            // Get neighbors of this neighbor
            std::vector<int>& neighbor_neighbors = neighbors_cache[neighbor];
            if (neighbor_neighbors.empty()) {
                neighbor_neighbors = find_neighbors(X, neighbor);
            }
            
            // If this is a core point, add its neighbors to the seeds queue
            if (neighbor_neighbors.size() >= static_cast<size_t>(min_samples)) {
                seeds.push(neighbor);
            }
        }
        
        // If neighbor has no cluster yet, add it to current cluster
        if (labels[neighbor] == -1) {
            labels[neighbor] = cluster_id;
        }
    }
    
    // Expand the cluster using the seeds
    while (!seeds.empty()) {
        int current_point = seeds.front();
        seeds.pop();
        
        // Get neighbors of current point
        std::vector<int>& current_neighbors = neighbors_cache[current_point];
        
        // Process all neighbors of current point
        for (int neighbor : current_neighbors) {
            // If neighbor hasn't been visited yet
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                
                // Get neighbors of this neighbor
                std::vector<int>& neighbor_neighbors = neighbors_cache[neighbor];
                if (neighbor_neighbors.empty()) {
                    neighbor_neighbors = find_neighbors(X, neighbor);
                }
                
                // If this is a core point, add its neighbors to the seeds queue
                if (neighbor_neighbors.size() >= static_cast<size_t>(min_samples)) {
                    seeds.push(neighbor);
                }
            }
            
            // If neighbor has no cluster yet, add it to current cluster
            if (labels[neighbor] == -1) {
                labels[neighbor] = cluster_id;
            }
        }
    }
}

std::vector<int> DBSCAN::fit_predict(const std::vector<std::vector<double>>& X) {
    if (X.empty() || X[0].empty()) {
        throw std::invalid_argument("Input data cannot be empty");
    }
    
    const size_t n_samples = X.size();
    
    // Initialize labels as noise (-1)
    labels.assign(n_samples, -1);
    
    // Track visited points
    std::vector<bool> visited(n_samples, false);
    
    // Cache for neighbors to avoid recalculating
    std::vector<std::vector<int>> neighbors_cache(n_samples);
    
    // Reset cluster count
    n_clusters = 0;
    n_noise = 0;
    
    // Process all points
    for (size_t i = 0; i < n_samples; ++i) {
        // Skip if already visited
        if (visited[i]) {
            continue;
        }
        
        // Mark as visited
        visited[i] = true;
        
        // Get neighbors
        std::vector<int>& neighbors = neighbors_cache[i];
        if (neighbors.empty()) {
            neighbors = find_neighbors(X, i);
        }
        
        // If not enough neighbors, mark as noise
        if (neighbors.size() < static_cast<size_t>(min_samples)) {
            labels[i] = -1;  // Noise
            n_noise++;
            continue;
        }
        
        // Start a new cluster
        int cluster_id = n_clusters++;
        
        // Expand cluster
        expand_cluster(X, i, cluster_id, visited, neighbors_cache);
    }
    
    // Count final noise points
    n_noise = std::count(labels.begin(), labels.end(), -1);
    
    return labels;
}
