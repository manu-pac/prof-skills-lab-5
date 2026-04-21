"""
convert_precision.py

Takes the float64 reference representations and converts them to:
  - float32
  - float16
  - int8  (symmetric per-vector min-max quantisation)

For int8, I also save the scale factors needed to reconstruct
approximate float values for distance computation.

Outputs:
  data/pipeline/representations/float32/representations.npz
  data/pipeline/representations/float16/representations.npz
  data/pipeline/representations/int8/representations.npz
    (contains 'representations' in int8, 'scales' in float64,
     and 'indices' in all cases)
"""

import os
import numpy as np

# Paths
INPUT_PATH  = os.path.join("data", "pipeline", "representations", "float64", "representations.npz")
OUTPUT_BASE = os.path.join("data", "pipeline", "representations")


def convert_float32(reps, indices):
    # .astype() creates a new array with the target dtype.
    reps_f32 = reps.astype(np.float32)
    return reps_f32


def convert_float16(reps, indices):
    # same as above
    reps_f16 = reps.astype(np.float16)
    return reps_f16


def quantise_int8(reps):
    n_vectors, n_dims = reps.shape
    # n_vectors = 1367 (one per word occurrence)
    # n_dims    = 768  (wav2vec2-base hidden size)

    # Allocate output arrays
    reps_int8 = np.zeros((n_vectors, n_dims), dtype=np.int8)
    # scales: one float64 value per vector, needed for reconstruction
    scales = np.zeros(n_vectors, dtype=np.float64)

    for i in range(n_vectors):
        x = reps[i]  # shape (768,), dtype float64

        # Step 1: find the maximum absolute value in this vector
        max_abs = np.max(np.abs(x))

        # Step 2: compute scale
        scale = max_abs / 127.0

        # Step 3: quantise
        # x / scale yields float values in approximately [-127, 127]
        # np.round() rounds to nearest integer
        # np.clip() ensures we stay within int8 range [-127, 127]
        # .astype(np.int8) stores as 8-bit signed integer
        quantised = np.clip(np.round(x / scale), -127, 127).astype(np.int8)

        reps_int8[i] = quantised
        scales[i]    = scale

    return reps_int8, scales


def save(output_dir, representations, indices, extra=None):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "representations.npz")

    # np.savez allows saving multiple named arrays into one .npz file.
    # **extra unpacks the dict so e.g. {'scales': scales} becomes scales=scales.
    save_dict = {"representations": representations, "indices": indices}
    if extra:
        save_dict.update(extra)

    np.savez(output_path, **save_dict)
    print(f"  Saved to {output_path} | dtype: {representations.dtype} | shape: {representations.shape}")


def main():
    print("Loading float64 representations...")
    data    = np.load(INPUT_PATH)
    reps    = data["representations"]   # shape (1367, 768), dtype float64
    indices = data["indices"]           # shape (1367,)
    print(f"  Loaded: shape={reps.shape}, dtype={reps.dtype}")

    print("\nConverting to float32...")
    reps_f32 = convert_float32(reps, indices)
    save(os.path.join(OUTPUT_BASE, "float32"), reps_f32, indices)

    print("\nConverting to float16...")
    reps_f16 = convert_float16(reps, indices)
    save(os.path.join(OUTPUT_BASE, "float16"), reps_f16, indices)

    print("\nQuantising to int8 (symmetric per-vector)...")
    reps_int8, scales = quantise_int8(reps)
    # We save scales alongside the int8 values because we need them
    # to reconstruct approximate float vectors for distance computation.
    save(os.path.join(OUTPUT_BASE, "int8"), reps_int8, indices, extra={"scales": scales})

    print("\nDone. All precision levels saved.")


if __name__ == "__main__":
    main()