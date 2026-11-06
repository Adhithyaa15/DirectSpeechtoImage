"""
Utility functions to load K-means centroids from CSV/NPZ files
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_centroids_from_csv(csv_path):
    """
    Load centroids from CSV file
    
    Args:
        csv_path: Path to CSV file with centroids
        
    Returns:
        centroids: numpy array [n_clusters, 4096]
    """
    df = pd.read_csv(csv_path)
    
    # Skip first 2 columns (cluster_id, cluster_size)
    # Columns 2-4097 are dim_0 to dim_4095
    centroids = df.iloc[:, 2:].values
    
    print(f"Loaded {centroids.shape[0]} centroids from {csv_path}")
    print(f"Centroid shape: {centroids.shape}")
    
    return centroids.astype(np.float32)


def load_centroids_from_npz(npz_path):
    """
    Load centroids from NPZ file (more efficient)
    
    Args:
        npz_path: Path to NPZ file with centroids
        
    Returns:
        centroids: numpy array [n_clusters, 4096]
    """
    data = np.load(npz_path)
    centroids = data['centroids']
    
    print(f"Loaded {centroids.shape[0]} centroids from {npz_path}")
    print(f"Centroid shape: {centroids.shape}")
    
    return centroids.astype(np.float32)


def find_latest_centroid_file(base_dir, n_clusters=None, format='csv'):
    """
    Find the most recent centroid file
    
    Args:
        base_dir: Base directory to search
        n_clusters: Optional - specific number of clusters to look for
        format: 'csv' or 'npz'
        
    Returns:
        path: Path to latest file, or None if not found
    """
    base_dir = Path(base_dir)
    
    if format == 'csv':
        if n_clusters:
            pattern = f"t5_kmeans_centroids_{n_clusters}clusters_*.csv"
        else:
            pattern = "t5_kmeans_centroids_*clusters_*.csv"
    elif format == 'npz':
        if n_clusters:
            pattern = f"t5_kmeans_{n_clusters}clusters_*.npz"
        else:
            pattern = "t5_kmeans_*clusters_*.npz"
    else:
        raise ValueError(f"Unknown format: {format}")
    
    files = sorted(list(base_dir.glob(pattern)))
    
    if not files:
        # Try the pattern without n_clusters restriction
        if format == 'csv':
            pattern = "t5_kmeans_centroids_*clusters_*.csv"
        else:
            pattern = "t5_kmeans_*clusters_*.npz"
        files = sorted(list(base_dir.glob(pattern)))
    
    if not files:
        return None
    
    # Return most recent (sorted by timestamp in filename)
    return files[-1]


def load_latest_centroids(base_dir, n_clusters=None, format='csv'):
    """
    Convenience function to automatically find and load latest centroids
    
    Args:
        base_dir: Base directory containing centroid files
        n_clusters: Optional - specific number of clusters
        format: 'csv' or 'npz'
        
    Returns:
        centroids: numpy array [n_clusters, 4096]
    """
    centroid_file = find_latest_centroid_file(base_dir, n_clusters, format)
    
    if centroid_file is None:
        raise FileNotFoundError(f"No centroid files found in {base_dir}")
    
    print(f"Auto-detected centroid file: {centroid_file}")
    
    if format == 'csv':
        return load_centroids_from_csv(centroid_file)
    else:
        return load_centroids_from_npz(centroid_file)


if __name__ == "__main__":
    # Test loading
    base_dir = "/scratch/IITB/ai-at-ieor/20004596/T5_Alignment_V0/outputs/kmeans"
    
    print("Testing centroid loading...")
    print("="*60)
    
    try:
        centroids = load_latest_centroids(base_dir, format='csv')
        print(f"\n✓ Successfully loaded centroids")
        print(f"  Shape: {centroids.shape}")
        print(f"  Dtype: {centroids.dtype}")
        print(f"  Mean: {centroids.mean():.4f}")
        print(f"  Std: {centroids.std():.4f}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
