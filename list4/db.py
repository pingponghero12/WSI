import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter, defaultdict
import os

import cl_module


class DBSCANMNISTAnalyzer:
    def __init__(self, n_samples=12000, pca_components=20):
        self.n_samples = n_samples
        self.pca_components = pca_components
        self.X = None
        self.y = None
        self.X_pca = None
        self.labels = None
        self.best_params = None
        self.best_metrics = None

    def load_data(self):
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(int)

        # subsample
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), self.n_samples, replace=False)
        X, y = X[idx], y[idx]

        # 3) standardize + PCA
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=self.pca_components, random_state=42)
        self.X_pca = pca.fit_transform(X_scaled)

        self.X, self.y = X, y
        var = pca.explained_variance_ratio_.sum()
        print(f"Data loaded: {self.n_samples} samples → PCA({self.pca_components}) "
              f"captures {var:.2%} variance.")

    def calculate_metrics(self, labels):
        # cluster counts
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        noise_ratio = n_noise / len(labels)

        # purity and accuracy
        total_correct = 0
        total_non_noise = 0
        purities = []
        cluster_stats = {}
        
        for cl in set(labels):
            if cl == -1:
                continue
            mask = labels == cl
            counts = Counter(self.y[mask])
            dom, dom_count = counts.most_common(1)[0]
            purity = dom_count / mask.sum()
            purities.append(purity)
            total_correct += dom_count
            total_non_noise += mask.sum()
            
            # Store cluster statistics
            cluster_stats[cl] = {
                'size': mask.sum(),
                'dominant_digit': dom,
                'purity': purity,
                'digit_counts': dict(counts)
            }
            
        purity = np.mean(purities) if purities else 0
        accuracy = total_correct / total_non_noise if total_non_noise else 0
        error_rate = 1 - accuracy

        # silhouette
        if n_clusters > 1 and n_noise < len(labels):
            sil = silhouette_score(self.X_pca[labels != -1], labels[labels != -1])
        else:
            sil = 0

        # combined score
        score = (0.4 * purity +
                 0.3 * accuracy +
                 0.2 * (1 - noise_ratio) +
                 0.1 * sil)

        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'purity': purity,
            'accuracy': accuracy,
            'error_rate': error_rate,
            'silhouette': sil,
            'combined_score': score,
            'cluster_stats': cluster_stats
        }

    def run(self, eps, min_samples):
        print(f"\nRunning DBSCAN with eps={eps}, min_samples={min_samples}")
        
        db = cl_module.DBSCAN(eps=eps, min_samples=min_samples)
        
        # Convert numpy array to list of lists for cpp
        X_list = self.X_pca.tolist()
        self.labels = np.array(db.fit_predict(X_list))
        
        # Calculate metrics
        metrics = self.calculate_metrics(self.labels)
        
        print(f"Clusters: {metrics['n_clusters']}")
        print(f"Noise:{metrics['n_noise']}")
        print(f"Purity: {metrics['purity']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Error rate: {metrics['error_rate']:.4f}")
        print(f"Silhouette: {metrics['silhouette']:.4f}")
        print(f"Combined score: {metrics['combined_score']:.4f}")
        
        return metrics

    def analyze_clusters(self):
        if self.labels is None:
            print("No clustering has been performed yet.")
            return
        
        digit_totals = Counter(self.y)
        
        print("\nDetailed cluster breakdown:")
        print("-" * 70)
        print(f"{'Cluster':<10}{'Size':<8}{'Dominant':<10}{'Purity':<10}{'Distribution'}")
        print("-" * 70)
        
        noise_mask = self.labels == -1
        if noise_mask.any():
            noise_count = noise_mask.sum()
            noise_dist = Counter(self.y[noise_mask])
            dist_str = ", ".join([f"{d}:{c}" for d, c in noise_dist.most_common(3)])
            print(f"{'Noise':<10}{noise_count:<8}{'N/A':<10}{'N/A':<10}{dist_str}")
        
        metrics = self.calculate_metrics(self.labels)
        cluster_stats = metrics['cluster_stats']
        
        for cl in sorted(cluster_stats.keys()):
            stats = cluster_stats[cl]
            dominant = stats['dominant_digit']
            size = stats['size']
            purity = stats['purity']
            
            digit_capture = size * purity / digit_totals[dominant]
            
            counts = stats['digit_counts']
            dist_str = ", ".join([f"{d}:{c}" for d, c in 
                                 sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]])
            
            print(f"{cl:<10}{size:<8}{dominant:<10}{purity:.4f}{dist_str}")
            
        print("\nDigit capture analysis:")
        print("-" * 70)
        print(f"{'Digit':<8}{'Total':<8}{'Clustered':<10}{'Noise':<8}{'Capture':<10}{'Accuracy'}")
        print("-" * 70)
        
        digit_clusters = defaultdict(list)
        digit_correct = defaultdict(int)
        digit_in_clusters = defaultdict(int)
        
        for cl, stats in cluster_stats.items():
            dom_digit = stats['dominant_digit']
            digit_clusters[dom_digit].append((cl, stats['purity'], stats['size']))
            digit_correct[dom_digit] += int(stats['size'] * stats['purity'])
            
        for digit in range(10):
            for cl in set(self.labels):
                if cl == -1:
                    continue
                mask = (self.labels == cl) & (self.y == digit)
                digit_in_clusters[digit] += mask.sum()
        
        for digit in range(10):
            total = digit_totals[digit]
            clustered = digit_in_clusters[digit]
            noise = total - clustered
            capture = clustered / total if total > 0 else 0
            
            correct = digit_correct[digit]
            accuracy = correct / clustered if clustered > 0 else 0
            
            clusters_str = ", ".join([f"{cl}({p:.2f})" for cl, p, _ in digit_clusters[digit]])
            
            print(f"{digit:<8}{total:<8}{clustered:<10}{noise:<8}{capture:.4f}    {accuracy:.4f}")
            if digit_clusters[digit]:
                print(f"  Dominant in clusters: {clusters_str}")
        
        print(f"Total samples: {len(self.y)}")
        print(f"Clustered samples: {len(self.y) - metrics['n_noise']} ({1-metrics['noise_ratio']:.2%})")
        print(f"Noise samples: {metrics['n_noise']} ({metrics['noise_ratio']:.2%})")
        print(f"Number of clusters: {metrics['n_clusters']}")
        print(f"Average cluster purity: {metrics['purity']:.4f}")
        print(f"Overall accuracy: {metrics['accuracy']:.4f}")
        print(f"Error rate within clusters: {metrics['error_rate']:.4f}")

    def grid_search(self, eps_values, min_samples_values):
        print(f"Performing grid search over {len(eps_values) * len(min_samples_values)} parameter combinations...")
        
        best_score = -1
        best_params = None
        best_metrics = None
        results = []
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                metrics = self.run(eps, min_samples)
                
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    **metrics
                })
                
                if metrics['combined_score'] > best_score:
                    best_score = metrics['combined_score']
                    best_params = (eps, min_samples)
                    best_metrics = metrics
        
        print("\nGrid search complete!")
        print(f"Best parameters: eps={best_params[0]}, min_samples={best_params[1]}")
        print(f"Best score: {best_score:.4f}")
        print(f"Clusters: {best_metrics['n_clusters']}, "
              f"Noise: {best_metrics['noise_ratio']:.2%}, "
              f"Purity: {best_metrics['purity']:.4f}, "
              f"Accuracy: {best_metrics['accuracy']:.4f}")
        
        self.best_params = best_params
        self.best_metrics = best_metrics
        
        # Run with best parameters to set labels
        self.run(*best_params)
        
        return results
    
    def visualize_clusters(self, n_examples=5):
        if self.labels is None:
            print("No clustering has been performed yet.")
            return
        
        metrics = self.calculate_metrics(self.labels)
        cluster_stats = metrics['cluster_stats']
        
        # Visualize some samples from each cluster
        for cl in sorted(set(self.labels)):
            if cl == -1:
                title = f"Noise examples (total: {(self.labels == -1).sum()})"
                filename = f"img/dbscan_noise_examples.png"
            else:
                stats = cluster_stats[cl]
                dominant = stats['dominant_digit']
                purity = stats['purity']
                size = stats['size']
                title = f"Cluster {cl}: size={size}, dominant={dominant} ({purity:.2%})"
                filename = f"img/dbscan_cluster_{cl}_dominant_{dominant}.png"
            
            indices = np.where(self.labels == cl)[0]
            
            if len(indices) == 0:
                continue
                
            sample_indices = np.random.choice(indices, 
                                             size=min(n_examples, len(indices)), 
                                             replace=False)
            
            # Plot examples
            fig, axes = plt.subplots(1, len(sample_indices), figsize=(len(sample_indices) * 2, 2))
            if len(sample_indices) == 1:
                axes = [axes]
                
            for i, idx in enumerate(sample_indices):
                axes[i].imshow(self.X[idx].reshape(28, 28), cmap='gray')
                axes[i].set_title(f"True: {self.y[idx]}")
                axes[i].axis('off')
                
            plt.suptitle(title)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()

def main():
    analyzer = DBSCANMNISTAnalyzer(n_samples=12000)
    analyzer.load_data()
    
    # analyzer.run(eps=6.0, min_samples=10)
    
    eps_values = [5.0, 6.0, 7.0, 8.0]
    min_samples_values = [3, 5, 10, 15]
    results = analyzer.grid_search(eps_values, min_samples_values)
    
    analyzer.analyze_clusters()
    analyzer.visualize_clusters()

if __name__ == "__main__":
    main()
