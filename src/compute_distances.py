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
  metrics/distances.json -- summary metrics (git-tracked)
  data/pipeline/distances.npz -- full distance arrays (dvc-tracked)
"""

import os
import time
import json
import numpy as np
import pandas as pd
from itertools import combinations

# Paths
MANIFEST_PATH = os.path.join("data", "pipeline", "manifest.csv")
REPS_BASE = os.path.join("data", "pipeline", "representations")
DISTANCES_PATH = os.path.join("data", "pipeline", "distances.npz")
METRICS_DIR = "metrics"
METRICS_PATH = os.path.join(METRICS_DIR, "distances.json")

PRECISIONS = ["float64", "float32", "float16", "int8"]


def cosine_distance(a, b):
    """
    Computes cosine distance in the native dtype of the input vectors.
    """
    dot = np.dot(a, b)
    norm_a = np.sqrt(np.dot(a, a))
    norm_b = np.sqrt(np.dot(b, b))
    return 1.0 - dot / (norm_a * norm_b)


def load_representations(precision):
    path = os.path.join(REPS_BASE, precision, "representations.npz")
    data = np.load(path)
    reps = data["representations"]
    indices = data["indices"]

    if precision == "int8":
        scales = data["scales"]
        # Reconstruct in float64 — int8 has no native float ops
        reps = reps.astype(np.float64) * scales[:, None]
    else:
        # Keep native precision
        pass

    return reps, indices


def compute_distances(reps, manifest):
    """
    Computes intra- and inter-speaker cosine distances.

    Intra-speaker: same speaker, same word, different recordings
    Inter-speaker: different speakers, same word

    Also checks, for each word, whether mean(intra) < mean(inter) —
    i.e. whether intra-speaker pairs are closer than inter-speaker pairs,
    as expected if the representations carry speaker-independent word identity.

    Returns:
      intra_distances:    1D array of all intra-speaker cosine distances
      inter_distances:    1D array of all inter-speaker cosine distances
      global_ordering:    bool — mean(intra) < mean(inter) across all words
      word_ordering:      dict[word -> bool] — per-word ordering check
      word_intra_means:   dict[word -> float] — per-word mean intra distance
      word_inter_means:   dict[word -> float] — per-word mean inter distance
    """
    intra_distances = []
    inter_distances = []
    word_ordering = {}
    word_intra_means = {}
    word_inter_means = {}

    words = manifest["word"].values
    speakers = manifest["speaker"].values

    unique_words = manifest["word"].unique()

    for word in unique_words:
        word_mask = words == word
        word_indices = np.where(word_mask)[0]

        word_intra = []
        word_inter = []

        for i, j in combinations(word_indices, 2):
            vec_i = reps[i]
            vec_j = reps[j]
            dist = cosine_distance(vec_i, vec_j)

            if speakers[i] == speakers[j]:
                word_intra.append(dist)
                intra_distances.append(dist)
            else:
                word_inter.append(dist)
                inter_distances.append(dist)

        if word_intra and word_inter:
            word_ordering[word] = bool(np.mean(word_intra) < np.mean(word_inter))
            word_intra_means[word] = float(np.mean(word_intra))
            word_inter_means[word] = float(np.mean(word_inter))

    intra_distances = np.array(intra_distances)
    inter_distances = np.array(inter_distances)
    global_ordering = all(word_ordering.values())

    return intra_distances, inter_distances, global_ordering, word_ordering, word_intra_means, word_inter_means


def get_disk_space(precision):
    """Returns the size of the representations file in bytes."""
    path = os.path.join(REPS_BASE, precision, "representations.npz")
    return os.path.getsize(path)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest: {len(manifest)} rows")

    # Consistent word order for aligned numpy arrays across precision levels
    unique_words = sorted(manifest["word"].unique())

    all_distances = {}
    all_metrics = {}

    for precision in PRECISIONS:
        print(f"\n── {precision} ──")

        disk_bytes = get_disk_space(precision)
        print(f"  Disk space: {disk_bytes / 1e6:.2f} MB")

        reps, indices = load_representations(precision)
        print(f"  Loaded representations: shape={reps.shape}, dtype={reps.dtype}")

        t0 = time.time()
        intra, inter, global_ordering, word_ordering, word_intra_means, word_inter_means = compute_distances(reps, manifest)
        compute_time = time.time() - t0

        print(f"  Intra-speaker distances: n={len(intra)}, mean={intra.mean():.6f}")
        print(f"  Inter-speaker distances: n={len(inter)}, mean={inter.mean():.6f}")
        print(f"  Ratio (intra/inter): {intra.mean() / inter.mean():.6f}")
        print(f"  Global ordering preserved (intra < inter): {global_ordering}")
        print(f"  Per-word ordering preserved: {all(word_ordering.values())}")
        print(f"  Compute time: {compute_time:.2f}s")

        # Raw distance arrays for visualisation (dvc-tracked)
        all_distances[f"intra_{precision}"] = intra
        all_distances[f"inter_{precision}"] = inter
        # Per-word means in consistent word order
        all_distances[f"word_intra_means_{precision}"] = np.array(
            [word_intra_means[w] for w in unique_words]
        )
        all_distances[f"word_inter_means_{precision}"] = np.array(
            [word_inter_means[w] for w in unique_words]
        )

        all_metrics[precision] = {
            "intra_speaker_mean":        float(intra.mean()),
            "inter_speaker_mean":        float(inter.mean()),
            "ratio":                     float(intra.mean() / inter.mean()),
            "n_intra_pairs":             int(len(intra)),
            "n_inter_pairs":             int(len(inter)),
            # Ordering checks — used to write the ordering statement in the report;
            # no plot is produced since ordering is fully preserved across all precisions.
            "global_ordering_preserved": bool(global_ordering),
            "word_ordering":             word_ordering,  # {word: bool}, human-readable
            "disk_space_bytes":          int(disk_bytes),
            "disk_space_mb":             round(disk_bytes / 1e6, 2),
            "compute_time_seconds":      round(compute_time, 2),
        }

    # Save word list in consistent order for use in visualisation
    all_distances["words"] = np.array(unique_words)

    np.savez(DISTANCES_PATH, **all_distances)
    print(f"\nRaw distances saved to {DISTANCES_PATH}")

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Summary metrics saved to {METRICS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()