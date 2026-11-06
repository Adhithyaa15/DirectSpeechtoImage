import numpy as np
import os
from pathlib import Path


def load_centroids_from_csv(filepath):
    import csv
    centroids = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            centroids.append([float(x) for x in row[1:]])
    return np.array(centroids, dtype=np.float32)


def find_latest_centroid_file(base_dir, n_clusters=None, format='csv'):
    base = Path(base_dir)
    if not base.exists():
        return None
    pattern = f"kmeans_centroids_*clusters_*.{format}"
    files = sorted(base.glob(pattern))
    if not files:
        return None
    if n_clusters is not None:
        files = [f for f in files if f"_{n_clusters}clusters_" in f.name]
    return str(files[-1]) if files else None
