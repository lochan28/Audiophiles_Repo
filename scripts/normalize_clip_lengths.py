"""
normalize_clip_lengths.py

Normalizes every clip in the project to at most 3 seconds at 16 kHz.

Speech (dataset/train, validation, testing):
    Anything longer than 3 seconds keeps its first 3 seconds and the tail is
    discarded. Clips already at or under 3 seconds keep their length. No
    speech file is added or removed, so the existing per-language numbering
    stays as it is. Everything is resampled to 16 kHz, which is what the
    English half and the whole noise set already use.

Noise (dataset/noise_data):
    Each category within each split is treated as one stream and re-cut into
    exact 3-second clips. Files are appended in order until 3 seconds are
    available, that clip is emitted, and whatever is left over carries into
    the next clip -- but only if the leftover is longer than 1 second.
    A leftover of 1 second or less is discarded and the next file starts a
    fresh clip. Whatever remains at the end of a category, being shorter than
    3 seconds, is discarded too.

    A stream never crosses a category boundary or a split boundary, so no
    clip ever mixes two noise types and no training audio leaks into
    validation or testing. Output files are renumbered from 001 with no gaps.

This rewrites audio in place and discards the trimmed material. Run without
arguments for a dry run that only reports what would change; pass --apply to
actually write.

    python scripts/normalize_clip_lengths.py
    python scripts/normalize_clip_lengths.py --apply
"""

import os
import csv
import sys
import glob
import shutil
import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
NOISE_DIR = os.path.join(DATASET_DIR, "noise_data")

SPLITS = ["train", "validation", "testing"]
NOISE_SPLIT_DIRS = {
    "train": "noise_train",
    "validation": "noise_validation",
    "testing": "noise_testing",
}

TARGET_SR = 16000
CLIP_SECONDS = 3.0
MIN_CARRY_SECONDS = 1.0     # leftover at or under this is dropped, not carried
SUBTYPE = "PCM_16"

CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)
MIN_CARRY_SAMPLES = int(TARGET_SR * MIN_CARRY_SECONDS)

STAGING_NAME = "__new__"
MANIFEST_FIELDNAMES = ["filename", "language", "speaker_id", "duration_seconds"]


def load_16k(path):
    """Read a wav as mono float32 at TARGET_SR, resampling if necessary."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        tensor = torch.from_numpy(data).unsqueeze(0)
        data = torchaudio.functional.resample(tensor, sr, TARGET_SR).squeeze(0).numpy()
    return data, sr


def write_atomic(path, data):
    """Write via a temp file in the same folder so a crash cannot truncate."""
    tmp = os.path.join(os.path.dirname(path), ".tmp_" + os.path.basename(path))
    sf.write(tmp, data, TARGET_SR, subtype=SUBTYPE)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# speech
# --------------------------------------------------------------------------

def normalize_speech(apply):
    changed_duration = 0
    resampled = 0
    new_durations = {}

    for split in SPLITS:
        split_dir = os.path.join(DATASET_DIR, split)
        for path in sorted(glob.glob(os.path.join(split_dir, "*", "*.wav"))):
            data, orig_sr = load_16k(path)
            needs_resample = orig_sr != TARGET_SR
            needs_trim = len(data) > CLIP_SAMPLES

            if needs_trim:
                data = data[:CLIP_SAMPLES]
                changed_duration += 1
            if needs_resample:
                resampled += 1

            rel = os.path.relpath(path, DATASET_DIR)
            new_durations[rel] = round(len(data) / TARGET_SR, 3)

            if (needs_resample or needs_trim) and apply:
                write_atomic(path, data)

        print(f"  {split}: scanned {len(glob.glob(os.path.join(split_dir, '*', '*.wav')))} clips")

    print(f"speech: {changed_duration} clips trimmed to 3.0s, "
          f"{resampled} clips resampled to {TARGET_SR} Hz")
    return new_durations


def update_speech_manifests(new_durations, apply):
    for split in SPLITS:
        path = os.path.join(DATASET_DIR, f"{split}_manifest.csv")
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        updated = 0
        for row in rows:
            key = row["filename"]
            if key in new_durations:
                fresh = f"{new_durations[key]:g}"
                if row["duration_seconds"] != fresh:
                    row["duration_seconds"] = fresh
                    updated += 1
            else:
                print(f"    !! manifest row has no file on disk: {key}")

        if apply:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
                writer.writeheader()
                writer.writerows(rows)
        print(f"  {split}_manifest.csv: {updated} durations updated")


# --------------------------------------------------------------------------
# noise
# --------------------------------------------------------------------------

def recut_category(paths):
    """Re-cut one category+split stream into exact 3-second clips.

    Returns (clips, consumed_seconds, dropped_seconds).
    """
    clips = []
    buffer = np.zeros(0, dtype=np.float32)
    consumed = 0.0
    dropped = 0.0

    for path in paths:
        data, _ = load_16k(path)
        consumed += len(data) / TARGET_SR
        buffer = np.concatenate([buffer, data])

        while len(buffer) >= CLIP_SAMPLES:
            clips.append(buffer[:CLIP_SAMPLES])
            buffer = buffer[CLIP_SAMPLES:]
            if len(buffer) <= MIN_CARRY_SAMPLES:
                dropped += len(buffer) / TARGET_SR
                buffer = np.zeros(0, dtype=np.float32)

    dropped += len(buffer) / TARGET_SR      # final tail, always under 3s
    return clips, consumed, dropped


def normalize_noise(apply):
    total_in = total_out = 0
    total_dropped = 0.0

    for split in SPLITS:
        split_dir = os.path.join(NOISE_DIR, NOISE_SPLIT_DIRS[split])
        for category in sorted(os.listdir(split_dir)):
            folder = os.path.join(split_dir, category)
            if not os.path.isdir(folder) or category == STAGING_NAME:
                continue

            paths = sorted(glob.glob(os.path.join(folder, "*.wav")))
            clips, consumed, dropped = recut_category(paths)
            total_in += len(paths)
            total_out += len(clips)
            total_dropped += dropped

            print(f"  {split}/{category}: {len(paths)} files ({consumed:.1f}s) "
                  f"-> {len(clips)} clips ({len(clips) * CLIP_SECONDS:.1f}s), "
                  f"{dropped:.1f}s dropped")

            if not apply:
                continue

            # stage the new clips, then swap, so originals survive a crash
            staging = os.path.join(folder, STAGING_NAME)
            os.makedirs(staging, exist_ok=True)
            for index, clip in enumerate(clips, start=1):
                sf.write(os.path.join(staging, f"{category}_{index:03d}.wav"),
                         clip, TARGET_SR, subtype=SUBTYPE)

            staged = sorted(glob.glob(os.path.join(staging, "*.wav")))
            if len(staged) != len(clips):
                raise RuntimeError(f"staging incomplete for {folder}")

            for path in paths:
                os.remove(path)
            for path in staged:
                os.replace(path, os.path.join(folder, os.path.basename(path)))
            os.rmdir(staging)

    print(f"noise: {total_in} files -> {total_out} clips, "
          f"{total_dropped:.1f}s dropped")


def main():
    apply = "--apply" in sys.argv
    print(f"mode: {'APPLY (writes to disk)' if apply else 'dry run'}\n")

    print("--- speech ---")
    new_durations = normalize_speech(apply)
    update_speech_manifests(new_durations, apply)

    print("\n--- noise ---")
    normalize_noise(apply)

    if not apply:
        print("\nnothing written. re-run with --apply to commit.")
    else:
        print("\ndone. re-run build_balanced_manifests.py to refresh the manifests.")


if __name__ == "__main__":
    main()
