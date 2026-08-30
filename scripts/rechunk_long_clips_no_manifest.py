"""
rechunk_long_clips_no_manifest.py

Same as rechunk_long_clips.py, EXCEPT it does not touch the manifest
CSVs at all -- it only splits and renumbers the .wav files on disk.

For every existing clip, in every language folder, across all three
splits (train, validation, testing):
    - 5 seconds or shorter -> left completely untouched.
    - longer than 5 seconds -> cut into consecutive 3-second chunks.
      The leftover at the end (whatever doesn't divide evenly into 3s)
      is kept as its own final chunk if it's 1 second or longer,
      otherwise it's dropped.

Numbering: within each language/split folder, everything gets
renumbered from 0 in the original clip order, gap-free -- a clip that
gets split takes as many consecutive numbers as it produced chunks,
and every clip after it shifts up accordingly.

Run it with:
    python rechunk_long_clips_no_manifest.py
"""

import os
import re
import glob
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset")
SPLITS = ["train", "validation", "testing"]

MAX_CLIP_SECONDS = 5.0      # clips at or under this length are left alone
CHUNK_SECONDS = 3.0         # length of each cut piece for longer clips
MIN_LEFTOVER_SECONDS = 1.0  # trailing leftover shorter than this is dropped

FILENAME_RE = re.compile(r"^(?P<language>.+)_(?P<index>\d+)\.wav$")


def find_language_folders(split):
    split_dir = os.path.join(OUTPUT_DIR, split)
    if not os.path.isdir(split_dir):
        return []
    return sorted(
        name for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    )


def list_existing_clips(split, language):
    """Returns [(original_index, filepath), ...] sorted by index."""
    folder = os.path.join(OUTPUT_DIR, split, language)
    pattern = os.path.join(folder, f"{language}_*.wav")
    found = []
    for path in glob.glob(pattern):
        m = FILENAME_RE.match(os.path.basename(path))
        if not m or m.group("language") != language:
            continue
        found.append((int(m.group("index")), path))
    found.sort(key=lambda pair: pair[0])
    return found


def build_output_chunks(array, sr):
    """
    Returns a list of chunk arrays -- a single-item list if the clip is
    short enough to leave alone, or several 3-second (plus maybe one
    shorter trailing) pieces if it's longer than MAX_CLIP_SECONDS.
    """
    duration = len(array) / sr

    if duration <= MAX_CLIP_SECONDS:
        return [array]

    chunk_samples = int(round(CHUNK_SECONDS * sr))
    full_chunks = len(array) // chunk_samples
    leftover_samples = len(array) - full_chunks * chunk_samples
    leftover_seconds = leftover_samples / sr

    chunks = []
    for i in range(full_chunks):
        start = i * chunk_samples
        chunks.append(array[start:start + chunk_samples])

    if leftover_samples > 0 and leftover_seconds >= MIN_LEFTOVER_SECONDS:
        start = full_chunks * chunk_samples
        chunks.append(array[start:])
    # otherwise the leftover is simply dropped

    return chunks


def process_language(split, language):
    """Rechunks every clip for one language/split and renumbers
    everything from 0. Doesn't touch any manifest."""
    existing = list_existing_clips(split, language)
    if not existing:
        return

    folder = os.path.join(OUTPUT_DIR, split, language)
    planned_chunks = []  # list of (array, sr) in final order

    split_count = 0
    for original_index, filepath in existing:
        array, sr = sf.read(filepath)
        chunks = build_output_chunks(array, sr)
        if len(chunks) > 1:
            split_count += 1
        for chunk_array in chunks:
            planned_chunks.append((chunk_array, sr))

    # Stage new files under temp names first so we never overwrite an
    # old file we haven't read yet.
    staged_paths = []
    for new_index, (chunk_array, sr) in enumerate(planned_chunks):
        temp_path = os.path.join(folder, f".tmp_{language}_{new_index:03d}.wav")
        sf.write(temp_path, chunk_array, sr)
        staged_paths.append(temp_path)

    # Now it's safe to remove every original file...
    for _, filepath in existing:
        os.remove(filepath)

    # ...and rename the staged files into their final names.
    for new_index, temp_path in enumerate(staged_paths):
        out_name = f"{language}_{new_index:03d}.wav"
        final_path = os.path.join(folder, out_name)
        os.replace(temp_path, final_path)

    print(f"  {language}: {len(existing)} clip(s) -> {len(planned_chunks)} clip(s) "
          f"({split_count} clip(s) were split)")


def main():
    for split in SPLITS:
        languages = find_language_folders(split)
        if not languages:
            continue

        print(f"{split}:")
        for language in languages:
            process_language(split, language)

    print("\nDone. (Manifest CSVs were not touched.)")


if __name__ == "__main__":
    main()