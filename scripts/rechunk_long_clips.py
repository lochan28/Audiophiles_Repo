"""
rechunk_long_clips.py

Rechunks clips longer than 5 seconds into 3-second pieces, leaving
the leftover if it's 1 second or longer. 

This version safely updates the dataset's *_manifest.csv files 
to match the new filenames and durations, and it renames original 
files to .old safely before writing new chunks to prevent data loss 
on crash. It is fully memory-efficient.
"""

import os
import re
import glob
import csv
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset")
SPLITS = ["train", "validation", "testing"]

MAX_CLIP_SECONDS = 5.0      # clips at or under this length are left alone
CHUNK_SECONDS = 3.0         # length of each cut piece for longer clips
MIN_LEFTOVER_SECONDS = 1.0  # trailing leftover shorter than this is dropped

FILENAME_RE = re.compile(r"^(?P<language>.+)_(?P<index>\d+)\.wav$")

MANIFEST_FIELDNAMES = ["filename", "language", "speaker_id", "duration_seconds"]

def find_language_folders(split):
    split_dir = os.path.join(OUTPUT_DIR, split)
    if not os.path.isdir(split_dir):
        return []
    return sorted(
        name for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    )

def list_existing_clips(split, language):
    folder = os.path.join(OUTPUT_DIR, split, language)
    pattern = os.path.join(folder, f"{language}_*.wav")
    found = []
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        m = FILENAME_RE.match(name)
        if not m or m.group("language") != language:
            continue
        found.append((int(m.group("index")), path))
    found.sort(key=lambda pair: pair[0])
    return found

def build_output_chunks(array, sr):
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

    return chunks

def process_split(split):
    manifest_path = os.path.join(OUTPUT_DIR, f"{split}_manifest.csv")
    
    rows = []
    row_by_filename = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                norm_path = row["filename"].replace("\\", "/")
                row_by_filename[norm_path] = row
                
    languages = find_language_folders(split)
    if not languages:
        return

    print(f"Processing split: {split}")
    new_manifest_rows = []
    processed_languages = set()

    for language in languages:
        existing = list_existing_clips(split, language)
        if not existing:
            continue
            
        processed_languages.add(language)
        folder = os.path.join(OUTPUT_DIR, split, language)
        
        old_paths = []
        for orig_idx, filepath in existing:
            old_path = filepath + ".old"
            os.rename(filepath, old_path)
            old_paths.append((orig_idx, old_path))
            
        new_index = 0
        split_count = 0
        planned_count = 0
        
        for orig_idx, old_path in old_paths:
            orig_rel_path = f"{split}/{language}/{language}_{orig_idx:03d}.wav"
            orig_row = row_by_filename.get(orig_rel_path)
            
            if not orig_row:
                orig_rel_path_win = f"{split}\\{language}\\{language}_{orig_idx:03d}.wav"
                orig_row = row_by_filename.get(orig_rel_path_win)
                
            speaker_id = orig_row["speaker_id"] if orig_row else f"__unknown_{language}_{orig_idx}"
            
            array, sr = sf.read(old_path)
            chunks = build_output_chunks(array, sr)
            
            if len(chunks) > 1:
                split_count += 1
                
            for chunk_array in chunks:
                out_name = f"{language}_{new_index:03d}.wav"
                final_rel_path = os.path.join(split, language, out_name)
                final_path = os.path.join(OUTPUT_DIR, final_rel_path)
                
                sf.write(final_path, chunk_array, sr)
                
                new_row = {
                    "filename": final_rel_path,
                    "language": language,
                    "speaker_id": speaker_id,
                    "duration_seconds": round(len(chunk_array) / sr, 3)
                }
                new_manifest_rows.append(new_row)
                new_index += 1
                planned_count += 1
                
        for _, old_path in old_paths:
            os.remove(old_path)
            
        print(f"  {language}: {len(existing)} clip(s) -> {planned_count} clip(s) ({split_count} clip(s) were split)")

    for row in rows:
        if row["language"] not in processed_languages:
            new_manifest_rows.append(row)
            
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(new_manifest_rows)

def main():
    for split in SPLITS:
        process_split(split)
    print("\nDone. Manifest CSVs and wav files successfully updated.")

if __name__ == "__main__":
    main()
