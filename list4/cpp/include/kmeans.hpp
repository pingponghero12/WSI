#pragma once

#include <vector>
#include <random>
#include <string>

class KMeans {
private:
    int n_clusters;
    std::string init;
    int n_init;
    int max_iter;
    int random_state;
    double inertia;
    std::vector<std::vector<double>> cluster_centers;
    std::vector<int> labels;
    
    // Helper functions
    std::vector<std::vector<double>> initialize_centroids_kmeanspp(
        const std::vector<std::vector<double>>& X, 
        std::mt19937& gen);
    double squared_distance(const std::vector<double>& a, const std::vector<double>& b);
    int nearest_centroid(const std::vector<double>& point, 
                         const std::vector<std::vector<double>>& centroids);
    std::vector<int> run_kmeans_once(const std::vector<std::vector<double>>& X, 
                               std::vector<std::vector<double>>& centroids,
                               double& inertia);

public:
    KMeans(int n_clusters=10, 
           std::string init="k-means++", 
           int n_init=1, 
           int max_iter=300, 
           int random_state=0);
    
    std::vector<int> fit_predict(const std::vector<std::vector<double>>& X);
    
    // Getters
    double get_inertia() const { return inertia; }
    std::vector<std::vector<double>> get_cluster_centers() const { return cluster_centers; }

    std::vector<int> get_labels() const { return labels; }
};
