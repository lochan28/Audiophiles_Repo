"""
yamnet_probe.py

Run Google's YAMNet over our own hand-labelled noise clips and report what it
hears. This is a MEASUREMENT, not a pipeline stage -- nothing here writes into
`dataset/` or `dataset_v2/`, and no label is changed by anything in this file.

    .venv_yamnet/Scripts/python.exe scripts/yamnet_probe.py --limit 5
    .venv_yamnet/Scripts/python.exe scripts/yamnet_probe.py

WHY THIS EXISTS, AND WHY IT COMES FIRST

The plan for semi-automatic pre-labelling rests on an assumption that has never
been tested on this data: that YAMNet's view of a battlefield noise clip is
informative about our seven categories. If it is not, the rest of the plan is
wasted effort, and that is much cheaper to learn now than after building a
review queue around it.

So this script asks three questions of the 636 clips we already have:

  1. WHAT DOES IT HEAR? For each of our categories, the AudioSet classes
     YAMNet most often reports. If our `helicopter` folder does not make
     YAMNet say helicopter, something is wrong with either the labels or the
     assumption.

  2. IS THERE JUNK IN OUR OWN DATA? YAMNet is reliable at detecting speech and
     music, which are exactly what must not be in a noise clip -- music is
     tonal and structured, so training a speech mask against it teaches the
     model to suppress the kind of sound speech is made of. Any clip where
     YAMNet reports speech or music is a clip worth listening to by hand.
     This audits work already done, which is the cheapest audit available.

  3. DO THE EMBEDDINGS SEPARATE OUR CLASSES? YAMNet's real value is not its
     class list, which is AudioSet's rather than ours, but its 1024-number
     summary of each sound. Saving those here lets a small classifier be
     trained on OUR boundaries later, and lets near-duplicate clips be found
     by comparing fingerprints instead of filenames.

WHAT THIS DELIBERATELY DOES NOT DO

It does not label anything, does not move or copy audio, and does not touch
the frozen evaluation mixtures. It writes one CSV and one .npy of embeddings
to its own output directory, and prints a report. Reading it changes nothing.

YAMNET NEVER SHIPS. It is a workshop tool for building the dataset. It is not
part of the enhancement model, adds nothing to what a listener hears, and
never runs on the Raspberry Pi.
"""

import os
import re
import csv
import sys
import argparse
from collections import Counter, defaultdict

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

TOP_K_REPORT = 5
# How many AudioSet classes to list per category in the report. Raising it
# shows more of the tail, which is where confusions hide; 5 is enough to see
# whether the top guess is stable or a coin flip between two classes.

JUNK_CLASSES = ("Speech", "Conversation", "Narration, monologue", "Whispering",
                "Child speech, kid speaking", "Speech synthesizer",
                "Hubbub, speech noise, speech babble",
                "Music", "Singing", "Musical instrument")
# Class names that must NOT be present in a noise clip. Speech is obvious.
# Music matters more than it looks: it is tonal and harmonic, so a mask
# trained against it learns to suppress exactly the structure that speech has.
# "Hubbub, speech noise, speech babble" is here deliberately -- babble is a
# real battlefield sound and a classic noise type elsewhere, but it is made of
# voices, so using it as noise would teach the model to suppress speech. None
# of our seven categories covers it, so it does not belong in the set.
#
# These are YAMNET's 521 classes, which are a SUBSET of the full 632-class
# AudioSet ontology -- "Male speech, man speaking" exists in AudioSet and not
# here, which is exactly the kind of near-miss the check below catches. Names
# must match the class map EXACTLY; the script refuses to start if one does
# not, rather than silently checking nothing.

JUNK_THRESHOLD = 0.30
# A clip is FLAGGED (not rejected -- nothing here rejects anything) when a junk
# class scores above this in any single frame. LOWER catches more borderline
# clips and produces more listening work; HIGHER trusts the existing labels
# more. 0.30 is deliberately loose for an audit: the cost of a false flag is
# one clip listened to, the cost of a miss is bad data left in the set.

AGGREGATE = "mean"
# How the ~5 per-frame opinions become one clip-level answer. "mean" is the
# stable choice and the right default for steady sounds like a rotor. "max"
# suits brief events -- a single gunshot may occupy one frame out of five, and
# a mean dilutes it fivefold. Worth trying both on the impulsive categories;
# this is the single most consequential knob here.

EMBED_AGGREGATE = "mean"
# Same choice for the 1024-number fingerprint. "mean" over frames is the
# standard clip-level summary and is what a classifier trained on top would
# expect. Changing this invalidates saved embeddings -- re-run if you do.


# ═══════════════════════════════════════════════════════════════════
#  LOCKED -- set by YAMNet and by the recorded data, do not change
# ═══════════════════════════════════════════════════════════════════

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
TARGET_SR = 16000          # YAMNet's required input rate; our clips already match
CLIP_SAMPLES = 48000       # 3.000 s, the annotator's contract
FRAME_SECONDS = 0.96       # YAMNet's analysis window
FRAME_HOP_SECONDS = 0.48   # and its hop -- so a 3 s clip yields 5 frames
EMBED_DIM = 1024

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE_DIR = os.path.join(PROJECT_ROOT, "dataset", "noise_data")
SPLIT_DIRS = {"train": "noise_train",
              "validation": "noise_validation",
              "testing": "noise_testing"}
OUT_DIR = os.path.join(PROJECT_ROOT, "yamnet_probe_out")

# Keep the downloaded model inside the venv that uses it: that directory is
# already gitignored, and deleting the venv cleans the cache with it.
CACHE_DIR = os.path.join(PROJECT_ROOT, ".venv_yamnet", "tfhub_cache")


def load_yamnet():
    """Load YAMNet and its class map. Explains itself if TF is missing."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.environ.setdefault("TFHUB_CACHE_DIR", CACHE_DIR)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TF's banner
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as e:
        raise SystemExit(
            f"TensorFlow is not available ({e}).\n"
            "  This script needs the ISOLATED environment, not .venv:\n"
            "      .venv_yamnet/Scripts/python.exe scripts/yamnet_probe.py\n"
            "  .venv is Python 3.14, which TensorFlow does not support.\n"
            "  To rebuild the environment:\n"
            "      py -3.12 -m venv .venv_yamnet\n"
            "      .venv_yamnet/Scripts/python.exe -m pip install "
            "tensorflow-cpu tensorflow_hub soundfile")

    model = hub.load(YAMNET_HANDLE)
    path = model.class_map_path().numpy().decode("utf-8")
    with open(path, newline="", encoding="utf-8") as f:
        names = [row["display_name"] for row in csv.DictReader(f)]

    # Fail loudly if a junk class name does not exist in the map. Silently
    # checking nothing would look exactly like finding nothing.
    unknown = [c for c in JUNK_CLASSES if c not in names]
    if unknown:
        raise SystemExit(
            "these JUNK_CLASSES are not AudioSet class names: "
            + ", ".join(repr(c) for c in unknown)
            + "\n  fix the spelling in the tuning block; the map has "
            + f"{len(names)} classes")
    return model, names


def clips(limit=None):
    """Every hand-labelled noise clip, as (path, category, split)."""
    out = []
    for split, folder in SPLIT_DIRS.items():
        root = os.path.join(NOISE_DIR, folder)
        if not os.path.isdir(root):
            continue
        for category in sorted(os.listdir(root)):
            cat_dir = os.path.join(root, category)
            if not os.path.isdir(cat_dir):
                continue
            for name in sorted(os.listdir(cat_dir)):
                if name.lower().endswith(".wav"):
                    out.append((os.path.join(cat_dir, name), category, split))
    out.sort()
    if limit:
        # Spread the limit across categories rather than taking the first N,
        # which would be all `ambient` and would say nothing.
        step = max(1, len(out) // limit)
        out = out[::step][:limit]
    return out


def read_clip(path):
    """16 kHz mono float32 in [-1, 1]. Refuses anything else."""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        raise SystemExit(f"{path} is {sr} Hz; YAMNet needs {TARGET_SR}")
    return wav


def aggregate(frame_scores, how):
    return frame_scores.max(axis=0) if how == "max" else frame_scores.mean(axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--limit", type=int, default=None,
                    help="probe only this many clips, spread across categories")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    items = clips(args.limit)
    if not items:
        raise SystemExit(f"no .wav files found under {NOISE_DIR}")
    print(f"clips        {len(items)}"
          + (f" (limited from the full set)" if args.limit else "")
          + f"\nmodel        {YAMNET_HANDLE}\n"
          f"aggregate    scores {AGGREGATE}, embeddings {EMBED_AGGREGATE}\n"
          f"cache        {os.path.relpath(CACHE_DIR, PROJECT_ROOT)}\n")

    print("loading YAMNet (first run downloads ~17 MB) ... ", end="", flush=True)
    model, names = load_yamnet()
    print(f"ok, {len(names)} classes")

    junk_idx = {names.index(c): c for c in JUNK_CLASSES}

    rows = []
    embeddings = np.zeros((len(items), EMBED_DIM), dtype=np.float32)
    per_category = defaultdict(Counter)
    flagged = []
    n_frames_seen = Counter()

    for i, (path, category, split) in enumerate(items):
        wav = read_clip(path)
        scores, embeds, _ = model(wav)
        scores = scores.numpy()          # (frames, 521)
        embeds = embeds.numpy()          # (frames, 1024)
        n_frames_seen[scores.shape[0]] += 1

        clip_scores = aggregate(scores, AGGREGATE)
        embeddings[i] = aggregate(embeds, EMBED_AGGREGATE)

        top = int(clip_scores.argmax())
        per_category[category][names[top]] += 1

        # Junk is judged per FRAME, not on the aggregate: half a second of
        # speech in a 3-second clip still ruins the clip, and averaging over
        # five frames would hide it.
        worst_junk, worst_score = "", 0.0
        for idx, label in junk_idx.items():
            s = float(scores[:, idx].max())
            if s > worst_score:
                worst_junk, worst_score = label, s
        if worst_score >= JUNK_THRESHOLD:
            flagged.append((worst_score, worst_junk, category, split,
                            os.path.basename(path)))

        rows.append({
            "filename": os.path.relpath(path, PROJECT_ROOT),
            "category": category,
            "split": split,
            "yamnet_top": names[top],
            "yamnet_top_score": round(float(clip_scores[top]), 4),
            "worst_junk_class": worst_junk,
            "worst_junk_score": round(worst_score, 4),
            "frames": scores.shape[0],
        })
        if (i + 1) % 50 == 0 or i + 1 == len(items):
            print(f"  {i + 1}/{len(items)}")

    # ---------------------------------------------------------------- report
    print(f"\nframes per clip: "
          + ", ".join(f"{k} frames x{v}" for k, v in sorted(n_frames_seen.items())))

    print(f"\n1. WHAT YAMNET HEARS IN EACH OF OUR CATEGORIES")
    print(f"   (top {TOP_K_REPORT} AudioSet classes by clip count)\n")
    for category in sorted(per_category):
        total = sum(per_category[category].values())
        print(f"   {category}  ({total} clips)")
        for label, n in per_category[category].most_common(TOP_K_REPORT):
            print(f"       {100.0 * n / total:5.1f}%  {label}")
        print()

    print(f"2. CLIPS THAT MAY CONTAIN SPEECH OR MUSIC")
    print(f"   (any frame scoring over {JUNK_THRESHOLD} -- these are worth "
          f"LISTENING to, not deleting)\n")
    if not flagged:
        print("   none. YAMNet found no speech or music above the threshold.\n")
    else:
        print(f"   {len(flagged)} of {len(items)} clips "
              f"({100.0 * len(flagged) / len(items):.1f}%)\n")
        for score, label, category, split, name in sorted(flagged, reverse=True)[:25]:
            print(f"       {score:.2f}  {label:<12} {category}/{name} ({split})")
        if len(flagged) > 25:
            print(f"       ... and {len(flagged) - 25} more, all in the CSV")
        print()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "yamnet_probe.csv")
    npy_path = os.path.join(args.out, "yamnet_embeddings.npy")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(npy_path, embeddings)
    print(f"3. EMBEDDINGS SAVED for a later classifier and for duplicate\n"
          f"   detection. Row order matches the CSV exactly.\n")
    print(f"   {os.path.relpath(csv_path, PROJECT_ROOT)}")
    print(f"   {os.path.relpath(npy_path, PROJECT_ROOT)}  "
          f"{embeddings.shape[0]}x{embeddings.shape[1]} float32, "
          f"{embeddings.nbytes / 1e6:.1f} MB")
    print(f"\nNothing in dataset/ or dataset_v2/ was read for labels or "
          f"written to. No clip was moved, copied or relabelled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
