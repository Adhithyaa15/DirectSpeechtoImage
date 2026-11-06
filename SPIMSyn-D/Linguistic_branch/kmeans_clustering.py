import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

EMB_DIM = 4096
SEQ_LEN = 256
DEFAULT_N_CLUSTERS = 2950


class EmbeddingKMeans:
    def __init__(self, emb_dir, n_clusters=None, auto_select=True,
                 k_range=(100, 1000, 10), batch_size=10000,
                 max_samples=None, n_epochs=3, random_state=42):
        self.emb_dir = Path(emb_dir)
        self.n_clusters = n_clusters
        self.auto_select = auto_select
        self.k_range = k_range
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.files = sorted(self.emb_dir.glob("*.npy"))
        if max_samples:
            self.files = self.files[:max_samples]
        print(f"Found {len(self.files)} embedding files")

    def _load(self, path):
        emb = np.load(path).astype(np.float32)
        return emb.reshape(-1, EMB_DIM) if emb.ndim == 1 else emb

    def _find_optimal_k(self, sample_size=50000):
        n = min(sample_size // SEQ_LEN, len(self.files))
        sample_files = np.random.choice(self.files, n, replace=False)
        samples = np.vstack([self._load(f) for f in tqdm(sample_files, desc="Sampling")])
        k_min, k_max, k_step = self.k_range
        best_k, best_score = k_min, -np.inf
        for k in tqdm(range(k_min, k_max + 1, k_step), desc="Evaluating k"):
            km = MiniBatchKMeans(n_clusters=k, batch_size=min(5000, len(samples)),
                                 max_iter=100, random_state=self.random_state, n_init=3)
            labels = km.fit_predict(samples)
            sil = silhouette_score(samples, labels, sample_size=min(10000, len(samples)))
            db  = davies_bouldin_score(samples, labels)
            score = 0.7 * sil - 0.3 * db
            print(f"  k={k}: silhouette={sil:.4f}, davies-bouldin={db:.4f}")
            if score > best_score:
                best_score, best_k = score, k
        print(f"Optimal k = {best_k}")
        return best_k

    def train(self):
        if self.auto_select and self.n_clusters is None:
            self.n_clusters = self._find_optimal_k()
        elif self.n_clusters is None:
            self.n_clusters = DEFAULT_N_CLUSTERS
        print(f"Training k={self.n_clusters}, epochs={self.n_epochs} ...")
        km = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size,
                             max_iter=1, tol=0.0, random_state=self.random_state,
                             compute_labels=False, init="k-means++", n_init=1)
        t0 = time.time()
        for epoch in range(self.n_epochs):
            batch = []
            for f in tqdm(self.files, desc=f"Epoch {epoch+1}/{self.n_epochs}"):
                try:
                    batch.append(self._load(f))
                    if sum(len(b) for b in batch) >= self.batch_size:
                        km.partial_fit(np.vstack(batch))
                        batch = []
                except Exception as e:
                    print(f"Skip {f.name}: {e}")
            if batch:
                km.partial_fit(np.vstack(batch))
        print(f"Done in {(time.time()-t0)/60:.1f} min")
        return km

    def save(self, km, output_dir):
        from datetime import datetime
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out / f"kmeans_centroids_{self.n_clusters}clusters_{ts}.csv"
        npz_path = out / f"kmeans_centroids_{self.n_clusters}clusters_{ts}.npz"
        df = pd.DataFrame(km.cluster_centers_,
                          columns=[f"dim_{i}" for i in range(EMB_DIM)])
        df.insert(0, "cluster_id", range(self.n_clusters))
        df.to_csv(csv_path, index=False)
        np.savez_compressed(npz_path, centroids=km.cluster_centers_)
        print(f"CSV: {csv_path}")
        print(f"NPZ: {npz_path}")
        return str(csv_path)

    def run(self, output_dir="./outputs/kmeans"):
        return self.save(self.train(), output_dir)


def main():
    parser = argparse.ArgumentParser(description="K-means for T5 embeddings")
    parser.add_argument("--emb_dir",      required=True)
    parser.add_argument("--n_clusters",   type=int,  default=None)
    parser.add_argument("--no_auto",      action="store_true")
    parser.add_argument("--k_min",        type=int,  default=100)
    parser.add_argument("--k_max",        type=int,  default=1000)
    parser.add_argument("--k_step",       type=int,  default=10)
    parser.add_argument("--batch_size",   type=int,  default=10000)
    parser.add_argument("--n_epochs",     type=int,  default=3)
    parser.add_argument("--max_samples",  type=int,  default=None)
    parser.add_argument("--output_dir",   default="./outputs/kmeans")
    parser.add_argument("--random_state", type=int,  default=42)
    args = parser.parse_args()
    clustering = EmbeddingKMeans(
        emb_dir=args.emb_dir,
        n_clusters=args.n_clusters or (DEFAULT_N_CLUSTERS if args.no_auto else None),
        auto_select=not args.no_auto,
        k_range=(args.k_min, args.k_max, args.k_step),
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        n_epochs=args.n_epochs,
        random_state=args.random_state,
    )
    clustering.run(args.output_dir)


if __name__ == "__main__":
    main()
