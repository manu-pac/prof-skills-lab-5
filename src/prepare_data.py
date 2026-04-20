"""
prepare_data.py

Parses the ru-fr_interference corpus and builds a manifest CSV with one row
per target word occurrence, containing:
  speaker, L1, word, wav_path, start, end
"""

import os
import re
import pandas as pd
import tgt

# ── Paths ──────────────────────────────────────────────────────────────────
CORPUS_ROOT   = os.path.join("ru-fr_interference", "2")
METADATA_PATH = os.path.join(CORPUS_ROOT, "metadata_RUFR.csv")
SPEAKERS_DIR  = os.path.join(CORPUS_ROOT, "wav_et_textgrids", "FRcorp_textgrids_only")
OUTPUT_PATH   = os.path.join("data", "pipeline", "manifest.csv")

# Target words to keep (excludes the distractor)
TARGET_WORDS = {
    "tsarine", "j'en chie", "sérieux", "cache cache",
    "hier", "divan", "pour gabriel", "louche",
    "tulle", "juxtaposer", "pas ceux", "garage"
}

def parse_textgrid(tg_path):
    """
    Returns a list of (word, xmin, xmax) for all non-empty intervals
    in the 'words' tier that match a target word.
    """
    tg = tgt.io.read_textgrid(tg_path)
    words_tier = tg.get_tier_by_name("words")
    results = []
    for interval in words_tier.intervals:
        text = interval.text.strip().lower()
        if text in TARGET_WORDS:
            results.append((text, interval.start_time, interval.end_time))
    return results

def main():
    # Load metadata: speaker id and L1
    metadata = pd.read_csv(METADATA_PATH, sep=",")
    # Build a dict: speaker_id (uppercase) -> L1
    speaker_l1 = dict(zip(metadata["spk"], metadata["L1"]))

    rows = []

    for speaker_id in os.listdir(SPEAKERS_DIR):
        speaker_dir = os.path.join(SPEAKERS_DIR, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue

        l1 = speaker_l1.get(speaker_id)
        if l1 is None:
            print(f"Warning: speaker {speaker_id} not found in metadata, skipping.")
            continue

        # Find all TextGrid files in this speaker's folder
        for filename in os.listdir(speaker_dir):
            if not filename.endswith(".TextGrid"):
                continue

            tg_path  = os.path.join(speaker_dir, filename)
            wav_path = tg_path.replace(".TextGrid", ".wav")

            if not os.path.exists(wav_path):
                print(f"Warning: no WAV found for {tg_path}, skipping.")
                continue

            matches = parse_textgrid(tg_path)
            for word, start, end in matches:
                rows.append({
                    "speaker": speaker_id,
                    "L1":      l1,
                    "word":    word,
                    "wav_path": wav_path,
                    "start":   start,
                    "end":     end
                })

    manifest = pd.DataFrame(rows, columns=["speaker", "L1", "word", "wav_path", "start", "end"])
    manifest = manifest.sort_values(["speaker", "word"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False)
    print(f"Manifest saved to {OUTPUT_PATH} ({len(manifest)} rows)")

if __name__ == "__main__":
    main()