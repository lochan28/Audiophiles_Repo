"""
collect_english.py

Standalone companion to split_speech_dataset.py -- downloads ONLY the
English share, into the dataset/train, dataset/validation, and
dataset/testing folders that already exist (including the English
subfolder already created). Doesn't touch any other language.

Source: openslr/librispeech_asr (public on Hugging Face, no login
needed -- unlike ai4bharat/indicvoices_r).

Same rules enforced as the main script:
    1. No audio file repeats across splits.
    2. No speaker appears in more than one split (locked to whichever
       split first picks them up).
    3. Inside one split, the same speaker can repeat, capped at
       MAX_PER_SPEAKER_PER_SPLIT.

Resumable: counts what's already sitting in dataset/<split>/English/
and reads whatever's already in the *_manifest.csv files, and only
fetches whatever's still missing. Safe to just run as-is, any time --
writes into the same shared manifest CSVs the main script uses, it
just adds English's rows to them.
"""

import os
try:
    os.add_dll_directory(r"C:\Users\k.lochan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build-shared\bin")
except Exception:
    pass
import csv
import glob
from datasets import load_dataset
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset")
LANGUAGE = "English"

SPLIT_TARGETS = {
    "train": 410,
    "validation": 50,
    "testing": 50,
}

# How many times the same speaker may repeat WITHIN one split.
# Kept at 3, matching the main script -- change this if you meant a
# different number.
MAX_PER_SPEAKER_PER_SPLIT = 3

MANIFEST_FIELDNAMES = ["filename", "language", "speaker_id", "duration_seconds"]


def english_targets():
    """English's 25% share of each split -- same formula used in
    split_speech_dataset.py's build_targets(), kept identical so the
    numbers match whatever's already been collected."""
    return {split: round(total * 0.25) for split, total in SPLIT_TARGETS.items()}


def stream_english():
    return load_dataset(
        "openslr/librispeech_asr",
        split="train.clean.100",
        streaming=True,
    )


def count_existing_clips(split):
    folder = os.path.join(OUTPUT_DIR, split, LANGUAGE)
    if not os.path.isdir(folder):
        return 0
    return len(glob.glob(os.path.join(folder, f"{LANGUAGE}_*.wav")))


def load_speaker_history():
    """Best-effort: reads whatever English rows already exist in the
    manifest CSVs, so a re-run keeps honoring rule 2 above."""
    speaker_split = {}
    speaker_count_in_split = {}
    for split in SPLIT_TARGETS:
        manifest_path = os.path.join(OUTPUT_DIR, f"{split}_manifest.csv")
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["language"] != LANGUAGE:
                    continue
                speaker_id = row["speaker_id"]
                speaker_split[speaker_id] = split
                key = (split, speaker_id)
                speaker_count_in_split[key] = speaker_count_in_split.get(key, 0) + 1
    return speaker_split, speaker_count_in_split


def open_manifest_writers():
    """Appends to the same shared *_manifest.csv files the main script
    writes -- adds a header only if that file doesn't exist yet."""
    files, writers = {}, {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in SPLIT_TARGETS:
        manifest_path = os.path.join(OUTPUT_DIR, f"{split}_manifest.csv")
        is_new = not os.path.exists(manifest_path)
        f = open(manifest_path, "a", newline="")
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        if is_new:
            writer.writeheader()
            f.flush()
        files[split] = f
        writers[split] = writer
    return files, writers


def main():
    targets = english_targets()
    for split in SPLIT_TARGETS:
        os.makedirs(os.path.join(OUTPUT_DIR, split, LANGUAGE), exist_ok=True)

    counts_so_far = {s: count_existing_clips(s) for s in SPLIT_TARGETS}
    speaker_split, speaker_count_in_split = load_speaker_history()
    unknown_speaker_counter = 0

    remaining = {s: max(0, targets[s] - counts_so_far[s]) for s in SPLIT_TARGETS}
    total_needed = sum(remaining.values())

    if total_needed == 0:
        print("English already complete: " +
              ", ".join(f"{s}={counts_so_far[s]}/{targets[s]}" for s in SPLIT_TARGETS))
        return

    print("English targets: " +
          ", ".join(f"{s}={targets[s]} (have {counts_so_far[s]})" for s in SPLIT_TARGETS))

    manifest_files, manifest_writers = open_manifest_writers()
    total_saved = 0
    seen_first_example = False

    try:
        for example in stream_english():
            if total_saved >= total_needed:
                break

            if not seen_first_example:
                print(f"example fields for English: {list(example.keys())}")
                seen_first_example = True

            audio = example.get("audio")
            if audio is None:
                continue

            speaker_id = example.get("speaker_id", "")
            if not speaker_id:
                unknown_speaker_counter += 1
                speaker_id = f"__unknown_{unknown_speaker_counter}"

            if speaker_id in speaker_split:
                target_split = speaker_split[speaker_id]
                if counts_so_far[target_split] >= targets[target_split]:
                    continue  # that split is already full
                if speaker_count_in_split.get((target_split, speaker_id), 0) >= MAX_PER_SPEAKER_PER_SPLIT:
                    continue  # this speaker already repeated enough here
            else:
                open_splits = [s for s in SPLIT_TARGETS if counts_so_far[s] < targets[s]]
                if not open_splits:
                    break
                target_split = min(
                    open_splits,
                    key=lambda s: counts_so_far[s] / targets[s] if targets[s] else 1
                )
                speaker_split[speaker_id] = target_split

            array = audio["array"]
            sr = audio["sampling_rate"]
            idx = counts_so_far[target_split]
            out_name = f"{LANGUAGE}_{idx:03d}.wav"
            out_path = os.path.join(OUTPUT_DIR, target_split, LANGUAGE, out_name)
            sf.write(out_path, array, sr)

            manifest_writers[target_split].writerow({
                "filename": os.path.join(target_split, LANGUAGE, out_name),
                "language": LANGUAGE,
                "speaker_id": speaker_id,
                "duration_seconds": round(len(array) / sr, 3),
            })
            manifest_files[target_split].flush()

            counts_so_far[target_split] += 1
            speaker_count_in_split[(target_split, speaker_id)] = (
                speaker_count_in_split.get((target_split, speaker_id), 0) + 1
            )
            total_saved += 1
    finally:
        for f in manifest_files.values():
            f.close()

    for s in SPLIT_TARGETS:
        got, want = counts_so_far[s], targets[s]
        status = "OK" if got >= want else "SHORT"
        print(f"English -> {s}: {got}/{want} [{status}]")

    print("\nDone.")


if __name__ == "__main__":
    main()