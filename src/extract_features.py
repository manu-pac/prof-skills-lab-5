"""
extract_features.py

For each word occurrence in the manifest, extracts wav2vec2 frame-level
representations and aggregates them via mean pooling over time.
Output is saved as float64 (reference precision).

Output:
  data/pipeline/representations/float64/representations.npz
    - 'representations': array of shape (N, 768)
    - 'indices':         array of manifest row indices (for traceability)
"""

import os
import time
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# ── Paths ──────────────────────────────────────────────────────────────────
MANIFEST_PATH = os.path.join("data", "pipeline", "manifest.csv")
OUTPUT_DIR    = os.path.join("data", "pipeline", "representations", "float64")
OUTPUT_PATH   = os.path.join(OUTPUT_DIR, "representations.npz")

# ── Model config ───────────────────────────────────────────────────────────
MODEL_NAME    = "facebook/wav2vec2-base"
SAMPLE_RATE   = 16000  # wav2vec2 expects 16kHz audio

def extract_representation(wav_path, start, end, processor, model):
    """
    Loads the audio segment [start, end] from wav_path,
    runs it through wav2vec2, and returns a mean-pooled
    vector of shape (768,) in float64.
    """
    duration = end - start
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, offset=start, duration=duration)

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # hidden_states: (1, T, 768) → mean over T → (768,)
    frame_representations = outputs.last_hidden_state.squeeze(0)  # (T, 768)
    pooled = frame_representations.mean(dim=0).numpy()            # (768,)

    return pooled.astype(np.float64)

def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest: {len(manifest)} rows")

    print(f"Loading model: {MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model     = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.")

    representations = []
    indices         = []

    start_time = time.time()

    for idx, row in manifest.iterrows():
        try:
            rep = extract_representation(
                row["wav_path"], row["start"], row["end"],
                processor, model
            )
            representations.append(rep)
            indices.append(idx)

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {idx + 1}/{len(manifest)} ({elapsed:.1f}s elapsed)")

        except Exception as e:
            print(f"  Warning: failed on row {idx} ({row['wav_path']}): {e}")

    representations = np.array(representations, dtype=np.float64)  # (N, 768)
    indices         = np.array(indices)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(OUTPUT_PATH, representations=representations, indices=indices)

    elapsed = time.time() - start_time
    print(f"\nDone. Shape: {representations.shape}, dtype: {representations.dtype}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()