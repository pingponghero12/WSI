#include "kmeans.hpp"
#include <limits>
#include <algorithm>
#include <cmath>
#include <stdexcept>

KMeans::KMeans(int n_clusters, std::string init, int n_init, int max_iter, int random_state) :
    n_clusters(n_clusters), 
    init(init), 
    n_init(n_init), 
    max_iter(max_iter), 
    random_state(random_state), 
    inertia(std::numeric_limits<double>::max()) {}

double KMeans::squared_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

int KMeans::nearest_centroid(const std::vector<double>& point, 
                            const std::vector<std::vector<double>>& centroids) {
    int closest = 0;
    double min_dist = squared_distance(point, centroids[0]);
    
    for (size_t i = 1; i < centroids.size(); ++i) {
        double dist = squared_distance(point, centroids[i]);
        if (dist < min_dist) {
            min_dist = dist;
            closest = i;
        }
    }
    
    return closest;
}

std::vector<std::vector<double>> KMeans::initialize_centroids_kmeanspp(
    const std::vector<std::vector<double>>& X, 
    std::mt19937& gen) {
    
    std::vector<std::vector<double>> centroids;
    std::uniform_int_distribution<> dis(0, X.size() - 1);
    
    // Choose first centroid randomly
    int first_centroid_idx = dis(gen);
    centroids.push_back(X[first_centroid_idx]);
    
    // Choose the rest of the centroids with probability proportional to squared distance
    while (centroids.size() < static_cast<size_t>(n_clusters)) {
        std::vector<double> distances(X.size(), 0.0);
        double sum_distances = 0.0;
        
        // Calculate distance to nearest centroid for each point
        for (size_t i = 0; i < X.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& centroid : centroids) {
                double dist = squared_distance(X[i], centroid);
                min_dist = std::min(min_dist, dist);
            }
            distances[i] = min_dist;
            sum_distances += min_dist;
        }
        
        // Convert distances to probabilities
        std::vector<double> probabilities(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            probabilities[i] = distances[i] / sum_distances;
        }
        
        // Create a distribution
        std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
        
        // Choose the next centroid
        int next_centroid_idx = dist(gen);
        centroids.push_back(X[next_centroid_idx]);
    }
    
    return centroids;
}

std::vector<int> KMeans::run_kmeans_once(const std::vector<std::vector<double>>& X, 
                                   std::vector<std::vector<double>>& centroids,
                                   double& inertia) {
    std::vector<int> labels(X.size());
    bool converged = false;
    int iteration = 0;
    
    const size_t n_features = X[0].size();
    
    while (!converged && iteration < max_iter) {
        // Assign points to nearest centroids
        for (size_t i = 0; i < X.size(); ++i) {
            labels[i] = nearest_centroid(X[i], centroids);
        }
        
        // Calculate new centroids
        std::vector<std::vector<double>> new_centroids(n_clusters, std::vector<double>(n_features, 0.0));
        std::vector<int> counts(n_clusters, 0);
        
        for (size_t i = 0; i < X.size(); ++i) {
            int cluster = labels[i];
            counts[cluster]++;
            
            for (size_t j = 0; j < n_features; ++j) {
                new_centroids[cluster][j] += X[i][j];
            }
        }
        
        // Normalize centroids
        for (int i = 0; i < n_clusters; ++i) {
            if (counts[i] > 0) {
                for (size_t j = 0; j < n_features; ++j) {
                    new_centroids[i][j] /= counts[i];
                }
            }
        }
        
        // Check for convergence
        converged = true;
        for (int i = 0; i < n_clusters; ++i) {
            if (squared_distance(centroids[i], new_centroids[i]) > 1e-4) {
                converged = false;
                break;
            }
        }
        
        centroids = new_centroids;
        iteration++;
    }
    
    // Calculate inertia
    inertia = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        inertia += squared_distance(X[i], centroids[labels[i]]);
    }
    
    return labels;
}

std::vector<int> KMeans::fit_predict(const std::vector<std::vector<double>>& X) {
    if (X.empty() || X[0].empty()) {
        throw std::invalid_argument("Input data cannot be empty");
    }
    
    double best_inertia = std::numeric_limits<double>::max();
    std::vector<std::vector<double>> best_centroids;
    std::vector<int> best_labels;
    
    for (int trial = 0; trial < n_init; ++trial) {
        std::mt19937 gen(random_state + trial);
        
        // Initialize centroids
        std::vector<std::vector<double>> centroids;
        if (init == "k-means++") {
            centroids = initialize_centroids_kmeanspp(X, gen);
        } else {
            throw std::invalid_argument("Only k-means++ initialization is supported");
        }
        
        // Run k-means
        double current_inertia = 0.0;
        std::vector<int> current_labels = run_kmeans_once(X, centroids, current_inertia);
        
        // Update best result if this trial has lower inertia
        if (current_inertia < best_inertia) {
            best_inertia = current_inertia;
            best_centroids = centroids;
            best_labels = current_labels;
        }
    }
    
    // Save best
    inertia = best_inertia;
    cluster_centers = best_centroids;
    labels = best_labels;
    
    return labels;
}
