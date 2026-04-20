"""
Module 11: K-Means with Auto-K Selection.

Two-stage approach:
  1. Sample stand vectors to Python. Sweep K = k_min..k_max. Score each K
     with silhouette (higher=better) and Davies-Bouldin (lower=better).
  2. Pick optimal K. Run final GEE wekaKMeans at that K for the full map.

IMPORTANT: The sample in step 1 should come from a MATERIALIZED asset, not
from the deep lazy computation chain. Otherwise you hit EE memory limits.
(This is why notebooks/01_build_assets.ipynb exports normalized_stands first.)

Exposed functions:
  - sweep_k(config, stand_sample_df)
      -> (scores_df, optimal_k)
  - run_final_clustering(config, normalized_stands, valid_mask, optimal_k, roi=None)
      -> final_clusters image
  - plot_k_scores(scores_df, optimal_k, save_path=None)
      -> matplotlib figure
"""

import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def sample_stand_vectors(config, source_image, n_sample=None, verbose=True):
    """Sample stand vectors from an image (usually a materialized asset).

    Returns a pandas DataFrame. Drops rows with nulls.
    """
    roi   = config['roi']
    scale = config['analysis_scale']
    if n_sample is None:
        n_sample = config['k_sample_pts']

    if verbose:
        print(f"Sampling {n_sample} stand vectors for K evaluation ...")

    sample_fc = source_image.sample(
        region     = roi,
        scale      = scale,
        numPixels  = n_sample,
        geometries = False,
    )
    features = sample_fc.getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in features])
    df = df.dropna()

    if verbose:
        print(f"Sampled {len(df)} valid stand vectors "
              f"({df.shape[1]} features each).")
    return df


def sweep_k(config, sample_df, verbose=True):
    """Sweep K over the configured range. Return (scores_df, optimal_k)."""
    k_min, k_max = config['k_range']
    k_range = range(k_min, k_max + 1)

    X = sample_df.values
    if verbose:
        print(f"Evaluating K from {k_min} to {k_max} ...")

    rows = []
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        sil = silhouette_score(
            X, labels,
            sample_size=min(500, len(X)),
            random_state=42,
        )
        db = davies_bouldin_score(X, labels)
        rows.append({"K": k, "Silhouette": sil, "Davies_Bouldin": db})

        if verbose:
            print(f"  K={k:2d}  Silhouette={sil:.3f}  DB={db:.3f}")

    scores_df = pd.DataFrame(rows)

    best_k_sil = int(scores_df.loc[scores_df["Silhouette"].idxmax(), "K"])
    best_k_db  = int(scores_df.loc[scores_df["Davies_Bouldin"].idxmin(), "K"])

    if best_k_sil == best_k_db:
        optimal_k = best_k_sil
        if verbose:
            print(f"\nConsensus: OPTIMAL_K = {optimal_k}")
    else:
        optimal_k = best_k_sil  # Silhouette preferred; override if needed
        if verbose:
            print(f"\nBest K by Silhouette : {best_k_sil}")
            print(f"Best K by Davies-Bouldin : {best_k_db}")
            print(f"Disagreement -> defaulting to Silhouette: K = {optimal_k}")

    return scores_df, optimal_k


def plot_k_scores(scores_df, optimal_k, save_path=None):
    """Plot silhouette and Davies-Bouldin curves. Returns the figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(scores_df["K"], scores_df["Silhouette"], "b-o")
    ax1.axvline(optimal_k, color="red", linestyle="--",
                label=f"Chosen K={optimal_k}")
    ax1.set_xlabel("K"); ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Silhouette Score (higher better)"); ax1.legend()
    ax1.grid(True)

    ax2.plot(scores_df["K"], scores_df["Davies_Bouldin"], "g-o")
    ax2.axvline(optimal_k, color="red", linestyle="--",
                label=f"Chosen K={optimal_k}")
    ax2.set_xlabel("K"); ax2.set_ylabel("Davies-Bouldin Index")
    ax2.set_title("Davies-Bouldin Index (lower better)"); ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {save_path}")
    return fig


def run_final_clustering(config, normalized_stands, valid_mask, optimal_k,
                         verbose=True):
    """Run final GEE wekaKMeans and return the cluster map image."""
    roi   = config['roi']
    scale = config['analysis_scale']

    if verbose:
        print(f"\nRunning final K-means (K={optimal_k}) on GEE ...")

    training_data = normalized_stands.sample(
        region     = roi,
        scale      = scale,
        numPixels  = 5000,
        geometries = True,
    )
    clusterer = ee.Clusterer.wekaKMeans(optimal_k, seed=42).train(training_data)

    final_clusters = (
        normalized_stands
        .cluster(clusterer)
        .rename("Management_Unit")
        .updateMask(valid_mask)
    )

    if verbose:
        print(f"K-means complete. {optimal_k} management units mapped.")
    return final_clusters


# ---- Default palette for cluster maps (up to 15 classes) ----
CLUSTER_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
    "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854", "#ffd92f",
]
