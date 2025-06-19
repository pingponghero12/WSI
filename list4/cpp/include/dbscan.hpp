#pragma once

#include <vector>

class DBSCAN {
private:
    double eps;                                 // Radius of neighborhood
    int min_samples;                            // Minimum points to form a core point
    std::vector<int> labels;                    // Cluster labels for each point
    int n_clusters;                             // Number of clusters found
    int n_noise;                                // Number of noise points

    // Helper methods
    std::vector<int> find_neighbors(const std::vector<std::vector<double>>& X, int point_idx);
    void expand_cluster(const std::vector<std::vector<double>>& X, int point_idx, 
                       int cluster_id, std::vector<bool>& visited,
                       std::vector<std::vector<int>>& neighbors_cache);
    double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b);
    
public:
    DBSCAN(double eps_param = 0.5, int min_samples_param = 5);
    
    // Main method to fit the model and return cluster labels
    std::vector<int> fit_predict(const std::vector<std::vector<double>>& X);

    // Getters
    std::vector<int> get_labels() const { return labels; }
    int get_n_clusters() const { return n_clusters; }
    int get_n_noise() const { return n_noise; }
};
