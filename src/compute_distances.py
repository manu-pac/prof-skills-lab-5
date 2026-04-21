"""
compute_distances.py

For each precision level (float64, float32, float16, int8):
  1. Loads the representations
  2. Reconstructs approximate float vectors (for int8: multiply by scales)
  3. Computes cosine distances between all relevant pairs:
       - intra-speaker: same speaker, same word, different recordings
       - inter-speaker: different speakers, same word
  4. Records disk space and computation time

Outputs:
  metrics/distances.json        -- summary metrics (git-tracked)
  data/pipeline/distances.npz   -- full distance arrays (dvc-tracked)
"""

import os
import time
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from itertools import combinations

# Paths
MANIFEST_PATH   = os.path.join("data", "pipeline", "manifest.csv")
REPS_BASE       = os.path.join("data", "pipeline", "representations")
DISTANCES_PATH  = os.path.join("data", "pipeline", "distances.npz")
METRICS_DIR     = "metrics"
METRICS_PATH    = os.path.join(METRICS_DIR, "distances.json")

PRECISIONS = ["float64", "float32", "float16", "int8"]


def load_representations(precision):
    """
    Loads representations for a given precision level.
    For int8, reconstructs approximate float vectors by multiplying
    each int8 vector by its corresponding scale factor.
    Returns float64 arrays for distance computation in all cases.
    """
    path = os.path.join(REPS_BASE, precision, "representations.npz")
    data = np.load(path)
    reps    = data["representations"]
    indices = data["indices"]

    if precision == "int8":
        scales = data["scales"]  # shape (N,)
        # Reconstruct: for each vector i, multiply by its scale
        # scales[:, None] reshapes (N,) to (N,1) so broadcasting works
        reps = reps.astype(np.float64) * scales[:, None]
    else:
        # Cast to float64 for consistent distance computation
        reps = reps.astype(np.float64)

    return reps, indices


def compute_distances(reps, manifest):
    """
    Computes intra- and inter-speaker cosine distances.

    Intra-speaker: same speaker, same word, different recordings
    - all pairs (i, j) where speaker_i == speaker_j and word_i == word_j
    - measures within-speaker pronunciation variability

    Inter-speaker: different speakers, same word
    - all pairs (i, j) where speaker_i != speaker_j and word_i == word_j
    - measures between-speaker pronunciation variability

    Returns:
      intra_distances: 1D array of all intra-speaker cosine distances
      inter_distances: 1D array of all inter-speaker cosine distances
      ordering_preserved: bool, whether mean(intra) < mean(inter) per word
    """
    intra_distances = []
    inter_distances = []
    # For ordering analysis: per-word check of intra < inter
    word_ordering   = {}

    words    = manifest["word"].values
    speakers = manifest["speaker"].values

    # Get unique words
    unique_words = manifest["word"].unique()

    for word in unique_words:
        # Get all indices in manifest for this word
        word_mask    = words == word
        word_indices = np.where(word_mask)[0]

        word_intra = []
        word_inter = []

        # All pairs of occurrences of this word
        for i, j in combinations(word_indices, 2):
            vec_i = reps[i]
            vec_j = reps[j]

            # scipy cosine() returns cosine distance = 1 - cosine_similarity
            dist = cosine(vec_i, vec_j)

            if speakers[i] == speakers[j]:
                # Same speaker → intra
                word_intra.append(dist)
                intra_distances.append(dist)
            else:
                # Different speakers → inter
                word_inter.append(dist)
                inter_distances.append(dist)

        # Check if ordering is preserved for this word
        if word_intra and word_inter:
            word_ordering[word] = np.mean(word_intra) < np.mean(word_inter)

    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)

    # Overall ordering: is it preserved for all words?
    ordering_preserved = all(word_ordering.values())

    return intra_distances, inter_distances, ordering_preserved, word_ordering


def get_disk_space(precision):
    """Returns the size of the representations file in bytes."""
    path = os.path.join(REPS_BASE, precision, "representations.npz")
    return os.path.getsize(path)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest: {len(manifest)} rows")

    all_distances = {}   # for the NPZ file
    all_metrics   = {}   # for the JSON file

    for precision in PRECISIONS:
        print(f"\n── {precision} ──")

        # Disk space (before timing, not part of compute time)
        disk_bytes = get_disk_space(precision)
        print(f"  Disk space: {disk_bytes / 1e6:.2f} MB")

        # Load representations
        reps, indices = load_representations(precision)
        print(f"  Loaded representations: shape={reps.shape}, dtype={reps.dtype}")

        # Time the distance computation
        t0 = time.time()
        intra, inter, ordering_preserved, word_ordering = compute_distances(reps, manifest)
        compute_time = time.time() - t0

        print(f"  Intra-speaker distances: n={len(intra)}, mean={intra.mean():.6f}")
        print(f"  Inter-speaker distances: n={len(inter)}, mean={inter.mean():.6f}")
        print(f"  Ratio (intra/inter): {intra.mean() / inter.mean():.6f}")
        print(f"  Ordering preserved (intra < inter): {ordering_preserved}")
        print(f"  Compute time: {compute_time:.2f}s")

        # Store raw distances for visualisation
        all_distances[f"intra_{precision}"] = intra
        all_distances[f"inter_{precision}"] = inter

        # Store per-word ordering for analysis
        all_distances[f"word_ordering_{precision}"] = np.array(
            list(word_ordering.values()), dtype=bool
        )

        # Store summary metrics
        all_metrics[precision] = {
            "intra_speaker_mean":    float(intra.mean()),
            "inter_speaker_mean":    float(inter.mean()),
            "ratio":                 float(intra.mean() / inter.mean()),
            "n_intra_pairs":         int(len(intra)),
            "n_inter_pairs":         int(len(inter)),
            "ordering_preserved":    bool(ordering_preserved),
            "disk_space_bytes":      int(disk_bytes),
            "disk_space_mb":         round(disk_bytes / 1e6, 2),
            "compute_time_seconds":  round(compute_time, 2),
        }

    # Saves raw distances
    np.savez(DISTANCES_PATH, **all_distances)
    print(f"\nRaw distances saved to {DISTANCES_PATH}")

    # Saves summary metrics
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Summary metrics saved to {METRICS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()