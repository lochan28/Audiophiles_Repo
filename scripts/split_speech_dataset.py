"""
split_speech_dataset.py
 
Builds three SEPARATE clean-speech sets for the ANC project:
    dataset/train/...        (default target: 410 clips)
    dataset/validation/...   (default target: 50 clips)
    dataset/testing/...      (default target: 50 clips)
 
Sources:
    - English            -> NOT YET WIRED UP. The indicvoices_r link
                             you're using doesn't include English, so
                             this is left as a placeholder below
                             (see stream_english()) -- English's slot
                             in the folders/manifests/targets is still
                             reserved, it just won't collect anything
                             until you plug in a real source later
                             (for example VCTK, LibriSpeech, or
                             Common Voice).
    - Hindi + 21 others  -> ai4bharat/indicvoices_r    (GATED, needs login)
 
Proportions per split: 25% English, 25% Hindi, 50% spread across the
other 21 scheduled Indian languages (as evenly as possible).
 
Guarantees enforced while assigning clips:
    1. No audio file is ever placed in more than one split
       (each streamed clip is only ever written to disk once).
    2. No speaker appears in more than one split. The first time a
       speaker is seen, they get "locked" to whichever split picked
       them up, and every later clip from that same speaker is only
       accepted into that same split (or skipped otherwise).
    3. Inside a single split, the same speaker can repeat, but never
       more than MAX_PER_SPEAKER_PER_SPLIT times.
 
------------------------------------------------------------------------
BEFORE RUNNING
------------------------------------------------------------------------
    pip install datasets huggingface_hub soundfile numpy
 
    huggingface-cli login
    (after first accepting the access conditions on
     https://huggingface.co/datasets/ai4bharat/indicvoices_r)
 
------------------------------------------------------------------------
NOTE ON FIELD NAMES
------------------------------------------------------------------------
indicvoices_r's dataset card doesn't publish exact field names publicly
(it's behind the access gate), so the "audio" / "speaker_id" keys below
are a best guess. The first example pulled for each language prints its
real field names to the console -- if they don't match, edit the two
spots marked FIELD NAME BELOW.
"""
 
import os
try:
    os.add_dll_directory(r"C:\Users\k.lochan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build-shared\bin")
except Exception:
    pass
import csv
from datasets import load_dataset
import soundfile as sf
 
# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset")
 
SPLIT_TARGETS = {
    "train": 410,
    "validation": 50,
    "testing": 50,
}
 
# how many times the same speaker is allowed to repeat WITHIN one split
MAX_PER_SPEAKER_PER_SPLIT = 3
 
OTHER_INDIAN_LANGUAGES = [
    "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Kannada",
    "Kashmiri", "Konkani", "Maithili", "Malayalam", "Manipuri",
    "Marathi", "Nepali", "Odia", "Punjabi", "Sanskrit", "Santali",
    "Sindhi", "Tamil", "Telugu", "Urdu",
]
 
 
def distribute_count(total, n_items):
    """Split `total` into `n_items` integers, as evenly as possible."""
    base = total // n_items
    remainder = total % n_items
    return [base + 1 if i < remainder else base for i in range(n_items)]
 
 
def build_targets():
    """
    Returns targets[language][split] = number of clips wanted,
    for every language across all three splits.
    """
    targets = {lang: {} for lang in ["English", "Hindi"] + OTHER_INDIAN_LANGUAGES}
 
    for split, total in SPLIT_TARGETS.items():
        english_n = round(total * 0.25)
        hindi_n = round(total * 0.25)
        others_total = total - english_n - hindi_n
 
        targets["English"][split] = english_n
        targets["Hindi"][split] = hindi_n
 
        other_counts = distribute_count(others_total, len(OTHER_INDIAN_LANGUAGES))
        for lang, count in zip(OTHER_INDIAN_LANGUAGES, other_counts):
            targets[lang][split] = count
 
    return targets
 
 
# ------------------------------------------------------------------
# Streaming sources
# ------------------------------------------------------------------
 
def stream_english():
    """
    PLACEHOLDER. ai4bharat/indicvoices_r (the dataset you're pulling
    the Indian languages from) does not include English, so there's
    nothing to stream from yet -- this just returns an empty stream
    for now, so the rest of the script still runs normally and
    English's 25% share stays reserved in the folders and manifests.
 
    Once you pick a real English source, swap the `return []` line
    below for something like:
 
        return load_dataset(
            "openslr/librispeech_asr",   # or VCTK, Common Voice, etc.
            split="train.clean.100",
            streaming=True,
        )
    """
    return []
 
 
def stream_indic(language):
    return load_dataset(
        "ai4bharat/indicvoices_r",
        data_dir=language,
        split="train",
        streaming=True,
    )
 
 
# ------------------------------------------------------------------
# Core assignment logic (shared by every language)
# ------------------------------------------------------------------
 
def collect_language(language, stream, targets_for_language, manifest_rows):
    """
    Pulls clips for one language from `stream` and assigns each one to
    train / validation / testing, respecting the speaker-isolation and
    per-speaker-repeat rules described at the top of this file.
    """
    split_names = list(targets_for_language.keys())
    counts_so_far = {s: 0 for s in split_names}
    speaker_split = {}                  # speaker_id -> split it's locked to
    speaker_count_in_split = {}         # (split, speaker_id) -> how many used
    unknown_speaker_counter = 0
 
    for lang_dir_split in split_names:
        os.makedirs(os.path.join(OUTPUT_DIR, lang_dir_split, language), exist_ok=True)
 
    total_needed = sum(targets_for_language.values())
    total_saved = 0
    seen_first_example = False
 
    for example in stream:
        if total_saved >= total_needed:
            break
 
        if not seen_first_example:
            print(f"  example fields for {language}: {list(example.keys())}")
            seen_first_example = True
 
        audio = example.get("audio")  # FIELD NAME BELOW
        if audio is None:
            continue
 
        speaker_id = example.get("speaker_id", "")  # FIELD NAME BELOW
        if not speaker_id:
            unknown_speaker_counter += 1
            speaker_id = f"__unknown_{unknown_speaker_counter}"
 
        # Decide which split this clip is allowed to go to.
        if speaker_id in speaker_split:
            target_split = speaker_split[speaker_id]
            if counts_so_far[target_split] >= targets_for_language[target_split]:
                continue  # that split is already full, skip this clip
            if speaker_count_in_split.get((target_split, speaker_id), 0) >= MAX_PER_SPEAKER_PER_SPLIT:
                continue  # this speaker has already repeated enough in that split
        else:
            # brand new speaker: send them to whichever open split is
            # proportionally furthest behind its target
            open_splits = [
                s for s in split_names
                if counts_so_far[s] < targets_for_language[s]
            ]
            if not open_splits:
                break  # every split for this language is full
            target_split = min(
                open_splits,
                key=lambda s: counts_so_far[s] / targets_for_language[s] if targets_for_language[s] else 1
            )
            speaker_split[speaker_id] = target_split
 
        # Save the clip
        array = audio["array"]
        sr = audio["sampling_rate"]
        idx = counts_so_far[target_split]
        out_name = f"{language}_{idx:03d}.wav"
        out_path = os.path.join(OUTPUT_DIR, target_split, language, out_name)
        sf.write(out_path, array, sr)
 
        manifest_rows[target_split].append({
            "filename": os.path.join(target_split, language, out_name),
            "language": language,
            "speaker_id": speaker_id,
            "duration_seconds": round(len(array) / sr, 3),
        })
 
        counts_so_far[target_split] += 1
        speaker_count_in_split[(target_split, speaker_id)] = (
            speaker_count_in_split.get((target_split, speaker_id), 0) + 1
        )
        total_saved += 1
 
    for s in split_names:
        got, want = counts_so_far[s], targets_for_language[s]
        status = "OK" if got >= want else "SHORT"
        print(f"  {language} -> {s}: {got}/{want} [{status}]")
 
 
# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
 
def main():
    targets = build_targets()
    manifest_rows = {split: [] for split in SPLIT_TARGETS}
 
    print("English (from openslr/librispeech_asr) ...")
    collect_language("English", stream_english(), targets["English"], manifest_rows)
 
    print("Hindi (from ai4bharat/indicvoices_r) ...")
    collect_language("Hindi", stream_indic("Hindi"), targets["Hindi"], manifest_rows)
 
    for language in OTHER_INDIAN_LANGUAGES:
        print(f"{language} (from ai4bharat/indicvoices_r) ...")
        collect_language(language, stream_indic(language), targets[language], manifest_rows)
 
    for split in SPLIT_TARGETS:
        manifest_path = os.path.join(OUTPUT_DIR, f"{split}_manifest.csv")
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["filename", "language", "speaker_id", "duration_seconds"]
            )
            writer.writeheader()
            writer.writerows(manifest_rows[split])
        print(f"{split}: {len(manifest_rows[split])} clips -> {manifest_path}")
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()
    