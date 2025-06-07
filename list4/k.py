import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')

# Import cpp module
import cl_module

class EMNISTClusteringAnalysis:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_scaled = None
        self.results = {}
        
    def load_data(self, n_samples=10000):
        print("Loading EMNIST MNIST dataset...")
        
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        
        # Sample data
        indices = np.random.choice(len(mnist.data), n_samples, replace=False)
        self.X = mnist.data[indices]
        self.y = mnist.target[indices].astype(int)
        
        # Normalize pixel values
        self.X = self.X / 255.0
        
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        print(f"Dataset loaded: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Digit distribution: {Counter(self.y)}")
        
    def perform_kmeans_clustering(self, n_clusters, n_trials=10):
        print(f"\nPerforming k-means clustering for {n_clusters}")
        
        best_inertia = float('inf')
        best_kmeans = None
        best_labels = None
        
        for trial in range(n_trials):
            kmeans = cl_module.KMeans(
                n_clusters=n_clusters, 
                init='k-means++', 
                n_init=1,
                max_iter=300,
                random_state=trial
            )
            
            # Convert numpy array to list of lists for cpp module
            X_list = self.X_scaled.tolist()
            labels = np.array(kmeans.fit_predict(X_list))
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans
                best_labels = labels
                
        print(f"Best inertia after {n_trials} trials: {best_inertia:.2f}")
        
        return best_kmeans, best_labels, best_inertia
    
    def create_assignment_matrix(self, labels, n_clusters):
        assignment_matrix = np.zeros((10, n_clusters))
        
        for digit in range(10):
            digit_indices = np.where(self.y == digit)[0]
            digit_labels = labels[digit_indices]
            
            for cluster in range(n_clusters):
                count = np.sum(digit_labels == cluster)
                percentage = (count / len(digit_indices)) * 100
                assignment_matrix[digit, cluster] = percentage
                
        return assignment_matrix
    
    def plot_assignment_matrix(self, assignment_matrix, n_clusters, title_suffix=""):
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(
            assignment_matrix,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=[f'Cluster {i}' for i in range(n_clusters)],
            yticklabels=[f'Digit {i}' for i in range(10)],
            cbar_kws={'label': 'Percentage (%)'}
        )
        
        plt.title(f'Digit Assignment to Clusters{title_suffix}\n({n_clusters} clusters)')
        plt.xlabel('Clusters')
        plt.ylabel('Digits')
        plt.tight_layout()
        
        # Save the figure
        filename = f"img/kmeans_{n_clusters}_assignment_matrix.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_centroids(self, kmeans, n_clusters, title_suffix=""):
        """Visualize cluster centroids as images"""
        centroids = np.array(kmeans.cluster_centers_)
        
        if n_clusters == 10:
            grid_size = (2, 5)
        elif n_clusters == 15:
            grid_size = (3, 5)
        elif n_clusters == 20:
            grid_size = (4, 5)
        elif n_clusters == 30:
            grid_size = (5, 6)
        else:
            grid_size = (int(np.ceil(np.sqrt(n_clusters))), int(np.ceil(np.sqrt(n_clusters))))
        
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 10))
        axes = axes.flatten() if n_clusters > 1 else [axes]
        
        for i in range(n_clusters):
            # Reshape centroid back to 28x28 image
            centroid_image = centroids[i].reshape(28, 28)
            
            axes[i].imshow(centroid_image, cmap='gray')
            axes[i].set_title(f'Cluster {i}')
            axes[i].axis('off')
            
        # Hide unused subplots
        for i in range(n_clusters, len(axes)):
            axes[i].axis('off')
            
        plt.suptitle(f'Cluster Centroids{title_suffix}\n({n_clusters} clusters)', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        filename = f"img/kmeans_{n_clusters}_centroids.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_cluster_purity(self, labels, n_clusters):
        """Analyze cluster purity and dominant digits"""
        cluster_analysis = {}
        
        for cluster in range(n_clusters):
            cluster_indices = np.where(labels == cluster)[0]
            cluster_digits = self.y[cluster_indices]
            
            # Count digits in this cluster
            digit_counts = Counter(cluster_digits)
            total_in_cluster = len(cluster_digits)
            
            # Find dominant digit
            dominant_digit = digit_counts.most_common(1)[0][0]
            dominant_percentage = (digit_counts[dominant_digit] / total_in_cluster) * 100
            
            cluster_analysis[cluster] = {
                'dominant_digit': dominant_digit,
                'dominant_percentage': dominant_percentage,
                'total_samples': total_in_cluster,
                'digit_distribution': dict(digit_counts)
            }
            
        return cluster_analysis
    
    def run_complete_analysis(self):
        self.load_data(n_samples=15000)
        
        cluster_numbers = [10, 15, 20, 30]
        
        for n_clusters in cluster_numbers:
            print(f"\n{'='*60}")
            print(f"ANALYSIS FOR {n_clusters} CLUSTERS")
            print(f"{'='*60}")
            
            # Perform clustering
            kmeans, labels, inertia = self.perform_kmeans_clustering(n_clusters, n_trials=10)
            
            # Store results
            self.results[n_clusters] = {
                'kmeans': kmeans,
                'labels': labels,
                'inertia': inertia
            }
            
            assignment_matrix = self.create_assignment_matrix(labels, n_clusters)
            self.plot_assignment_matrix(assignment_matrix, n_clusters, f" ({n_clusters} clusters)")
            
            # Plot centroids
            self.plot_centroids(kmeans, n_clusters, f" ({n_clusters} clusters)")
            
            cluster_analysis = self.analyze_cluster_purity(labels, n_clusters)
            
            ari = adjusted_rand_score(self.y, labels)
            nmi = normalized_mutual_info_score(self.y, labels)
            
            print(f"\nClustering Metrics:")
            print(f"Inertia: {inertia:.2f}")
            print(f"Adjusted Rand Index: {ari:.3f}")
            print(f"Normalized Mutual Information: {nmi:.3f}")

def main():
    print("EMNIST MNIST Clustering Analysis")
    print("=" * 50)
    
    analyzer = EMNISTClusteringAnalysis()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
