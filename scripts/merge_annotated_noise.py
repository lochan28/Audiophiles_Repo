"""
merge_annotated_noise.py

Merges annotator output from the Noise Annotator tool into dataset_v2/ --
the multi-noise expansion set, kept as a sibling of dataset/ so every run
trained on the old data stays a valid, testable fallback. See HANDOFF.md
Section 10 for the original spec; the target moved from dataset/noise_data/
to dataset_v2/noise_data/ once the project decided v2 had to be a separate
set rather than merged into the one every past run was scored against.

Expected input, one folder per person:

    incoming_noise/<person>/classified/
    |-- training/<category>/<category>_001.wav, _002.wav, ...
    |-- validation/<category>/...
    |-- testing/<category>/...
    `-- manifest.csv   clip, split, category, source, start, end,
                       annotator, created, level_dbfs  (one row per clip)

Every clip is contractually exactly 48000 samples, 16 kHz, mono, PCM_16 --
the annotator's own server enforces this. This script re-checks it anyway
and refuses the WHOLE run if anything fails, rather than installing part of
one person's work and not another's.

Steps:
  1. Read each person's manifest.csv; verify every listed clip exists and is
     the exact contracted format. Any miss refuses the entire run.
  2. Content-hash every incoming clip against every other incoming clip and
     against everything already in dataset_v2/noise_data/. Overlapping
     windows from the same raw file are fine (different bytes); a byte-for-
     byte duplicate is refused rather than installed twice.
  3. Map training -> noise_train, validation -> noise_validation,
     testing -> noise_testing (the annotator's names, ours).
  4. Category-in-all-three-splits rule: a category is installed only if it
     exists in all three splits AFTER the merge. One a single annotator
     invented and only put in training is held back and listed, not
     silently dropped and not silently installed.
  5. Write into dataset_v2/noise_data/noise_<split>/<category>/, numbered
     from the next free index already in that folder. Existing v2 files are
     never renumbered.
  6. Append newly-installed categories to dataset_v2/noise_classes.txt at
     the bottom only, never reordering.
  7. Append dataset_v2/noise_provenance.csv: new filename, split, category,
     annotator, raw source file, start, end -- and, when
     C:\\mad_splits\\assignment.csv is reachable, the MAD label, class name
     and YouTube URL joined in for traceability back to the source video.
  8. Rebuild dataset_v2/noise_manifest.csv from what is now on disk (this
     script's own writer, not build_balanced_manifests.py -- that script
     also rebalances speech, which v2 does not touch).
  9. Never touches dataset/ or generated_test_mixtures/. The v1 frozen
     evaluation set contains none of this and stays valid for comparison.

    python scripts/merge_annotated_noise.py            # dry run
    python scripts/merge_annotated_noise.py --apply    # install

Source files are copied, not moved, so incoming_noise/ stays intact until
cleared by hand.
"""

import os
import re
import csv
import sys
import shutil
import hashlib
import argparse
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DROP_DIR = os.path.join(PROJECT_ROOT, "incoming_noise")
DEFAULT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset_v2")
DEFAULT_ASSIGNMENT_CSV = r"C:\mad_splits\assignment.csv"

ANNOTATOR_SPLIT_DIRS = {"training": "train", "validation": "validation",
                        "testing": "testing"}
NOISE_SPLIT_DIRS = {"train": "noise_train", "validation": "noise_validation",
                    "testing": "noise_testing"}
SPLITS = ["train", "validation", "testing"]

TARGET_SR = 16000
CLIP_SAMPLES = 48000             # 3.000 s at 16 kHz -- the annotator's contract
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

NOISE_FIELDNAMES = ["filename", "category", "split", "duration_seconds"]
PROVENANCE_FIELDNAMES = ["filename", "split", "category", "annotator",
                         "source_file", "start", "end",
                         "mad_label", "mad_class", "youtube_url"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_format(path):
    """None if the clip matches the contract; a reason string if not."""
    try:
        info = sf.info(path)
    except Exception as e:
        return f"unreadable ({e})"
    if info.samplerate != TARGET_SR:
        return f"{info.samplerate} Hz, expected {TARGET_SR}"
    if info.channels != 1:
        return f"{info.channels} channels, expected mono"
    if info.frames != CLIP_SAMPLES:
        return f"{info.frames} samples, expected {CLIP_SAMPLES}"
    if info.subtype != "PCM_16":
        return f"{info.subtype}, expected PCM_16"
    return None


def find_people(drop_dir):
    if not os.path.isdir(drop_dir):
        return []
    return sorted(p for p in os.listdir(drop_dir)
                  if os.path.isfile(os.path.join(drop_dir, p, "classified",
                                                 "manifest.csv")))


def read_person_manifest(drop_dir, person):
    path = os.path.join(drop_dir, person, "classified", "manifest.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def existing_hashes(dataset_root):
    """{sha256: relative path} for everything already installed in v2."""
    out = {}
    noise_dir = os.path.join(dataset_root, "noise_data")
    if not os.path.isdir(noise_dir):
        return out
    for split_dir in NOISE_SPLIT_DIRS.values():
        base = os.path.join(noise_dir, split_dir)
        if not os.path.isdir(base):
            continue
        for cat in os.listdir(base):
            cat_dir = os.path.join(base, cat)
            if not os.path.isdir(cat_dir):
                continue
            for name in os.listdir(cat_dir):
                p = os.path.join(cat_dir, name)
                out[sha256_of(p)] = os.path.relpath(p, dataset_root)
    return out


def next_free_index(cat_dir, category):
    if not os.path.isdir(cat_dir):
        return 1
    pat = re.compile(rf"^{re.escape(category)}_(\d+)\.wav$")
    used = [int(m.group(1)) for n in os.listdir(cat_dir)
            if (m := pat.match(n))]
    return max(used, default=0) + 1


def pinned_classes(dataset_root):
    path = os.path.join(dataset_root, "noise_classes.txt")
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    out.append(line)
    return out


def load_assignment(path):
    """{source_filename: row} from the MAD answer key, if reachable."""
    if not path or not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {r["file"]: r for r in rows if "file" in r}


def collect_and_validate(drop_dir, people):
    """Returns (clips, problems). clips is a flat list of dicts, one per
    incoming clip, ready to install. problems is a list of strings; if
    non-empty the whole run is refused before anything is written.
    """
    clips, problems = [], []
    for person in people:
        rows = read_person_manifest(drop_dir, person)
        if not rows:
            problems.append(f"{person}: manifest.csv has no rows")
            continue
        for row in rows:
            ann_split = row.get("split", "")
            if ann_split not in ANNOTATOR_SPLIT_DIRS:
                problems.append(f"{person}: manifest row has unknown split "
                                f"{ann_split!r} (clip {row.get('clip')})")
                continue
            category = row.get("category", "")
            if not NAME_RE.match(category):
                problems.append(f"{person}: category {category!r} is not "
                                f"lowercase_with_underscores (clip {row.get('clip')})")
                continue
            path = os.path.join(drop_dir, person, "classified",
                                ann_split, category, row["clip"])
            if not os.path.exists(path):
                problems.append(f"{person}: {row['clip']} listed in manifest "
                                f"but missing on disk ({path})")
                continue
            reason = check_format(path)
            if reason:
                problems.append(f"{person}: {row['clip']} fails format check: {reason}")
                continue
            clips.append({
                "person": person,
                "path": path,
                "split": ANNOTATOR_SPLIT_DIRS[ann_split],
                "category": category,
                "source_file": row.get("source", ""),
                "start": row.get("start", ""),
                "end": row.get("end", ""),
            })
    return clips, problems


def check_duplicates(clips, dataset_root):
    """Hashes everything incoming against everything else incoming and
    against what is already installed. Returns (clips_with_hash, dupe_report).
    Duplicates are dropped from what gets installed, never installed twice.
    """
    seen = dict(existing_hashes(dataset_root))   # hash -> where it already lives
    dupe_report = []
    kept = []
    for clip in clips:
        label = (f"{clip['person']}/{clip['split']}/{clip['category']}/"
                f"{os.path.basename(clip['path'])}")
        h = sha256_of(clip["path"])
        if h in seen:
            dupe_report.append(f"{label} is byte-identical to {seen[h]} -- skipped")
            continue
        seen[h] = f"{label} (incoming)"
        clip["hash"] = h
        kept.append(clip)
    return kept, dupe_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the files and manifests; default is a dry run")
    ap.add_argument("--drop-dir", default=DEFAULT_DROP_DIR, metavar="DIR",
                    help="folder holding <person>/classified/ subfolders")
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, metavar="DIR",
                    help="v2 dataset root to install into (default dataset_v2/)")
    ap.add_argument("--assignment-csv", default=DEFAULT_ASSIGNMENT_CSV, metavar="PATH",
                    help="MAD answer key for provenance enrichment; skipped "
                         "with a warning if not reachable")
    args = ap.parse_args()
    print(f"mode: {'APPLY' if args.apply else 'dry run'}")

    people = find_people(args.drop_dir)
    if not people:
        print(f"nothing to merge. Drop <person>/classified/ folders under "
              f"{os.path.relpath(args.drop_dir, PROJECT_ROOT)}/ first.")
        return
    print(f"people found: {', '.join(people)}\n")

    clips, problems = collect_and_validate(args.drop_dir, people)
    if problems:
        print(f"REFUSED -- {len(problems)} problem(s), installing nothing:")
        for p in problems:
            print(f"    {p}")
        return
    print(f"{len(clips)} clips pass the format check across {len(people)} "
          f"people\n")

    clips, dupes = check_duplicates(clips, args.dataset_root)
    print(f"duplicate check: {len(dupes)} byte-identical clip(s) found "
          f"{'and skipped' if dupes else '(none)'}")
    for d in dupes:
        print(f"    {d}")
    if not clips:
        print("\nnothing left to install after duplicate removal.")
        return

    existing = pinned_classes(args.dataset_root)
    by_category = {}
    for clip in clips:
        by_category.setdefault(clip["category"], {"train": [], "validation": [], "testing": []})
        by_category[clip["category"]][clip["split"]].append(clip)

    # category-in-all-three-splits rule, applied against what WOULD exist
    # after the merge, not just what is incoming
    already_have = {}
    for cat in set(existing):
        already_have[cat] = {
            split: len(os.listdir(os.path.join(args.dataset_root, "noise_data",
                                                NOISE_SPLIT_DIRS[split], cat)))
                   if os.path.isdir(os.path.join(args.dataset_root, "noise_data",
                                                 NOISE_SPLIT_DIRS[split], cat)) else 0
            for split in SPLITS
        }

    installable, held_back = {}, {}
    for cat, per_split in by_category.items():
        after = {
            split: len(per_split[split]) + already_have.get(cat, {}).get(split, 0)
            for split in SPLITS
        }
        if all(after[s] > 0 for s in SPLITS):
            installable[cat] = per_split
        else:
            held_back[cat] = after

    print(f"\ncategories ready to install: "
          f"{', '.join(sorted(installable)) if installable else '(none)'}")
    if held_back:
        print("categories HELD BACK (missing from at least one split after "
              "the merge, not installed):")
        for cat, after in held_back.items():
            print(f"    {cat}: " + ", ".join(f"{s} {after[s]}" for s in SPLITS))

    if not installable:
        print("\nnothing installed.")
        return

    assignment = load_assignment(args.assignment_csv)
    if assignment is None:
        print(f"\nnote: {args.assignment_csv} not reachable -- provenance "
              f"rows will be written without MAD label/class/URL")

    provenance_rows = []
    total_installed = 0
    for cat, per_split in installable.items():
        print(f"\ninstalling {cat!r}:")
        for split in SPLITS:
            split_clips = per_split[split]
            if not split_clips:
                print(f"    {split:<11} 0 new (already satisfied)")
                continue
            dest_dir = os.path.join(args.dataset_root, "noise_data",
                                    NOISE_SPLIT_DIRS[split], cat)
            idx = next_free_index(dest_dir, cat)
            print(f"    {split:<11} {len(split_clips)} new, numbered from {idx}")
            if args.apply:
                os.makedirs(dest_dir, exist_ok=True)
            for clip in split_clips:
                name = f"{cat}_{idx:03d}.wav"
                dest_path = os.path.join(dest_dir, name)
                if args.apply:
                    shutil.copyfile(clip["path"], dest_path)
                mad = assignment.get(clip["source_file"]) if assignment else None
                provenance_rows.append({
                    "filename": os.path.join("noise_data", NOISE_SPLIT_DIRS[split],
                                             cat, name),
                    "split": split,
                    "category": cat,
                    "annotator": clip["person"],
                    "source_file": clip["source_file"],
                    "start": clip["start"],
                    "end": clip["end"],
                    "mad_label": mad.get("label", "") if mad else "",
                    "mad_class": mad.get("class_name", "") if mad else "",
                    "youtube_url": mad.get("youtube_url", "") if mad else "",
                })
                idx += 1
                total_installed += 1

    new_categories = sorted(set(installable) - set(existing))
    if args.apply:
        classes_path = os.path.join(args.dataset_root, "noise_classes.txt")
        if new_categories:
            with open(classes_path, "a", encoding="utf-8") as f:
                for cat in new_categories:
                    print(cat, file=f)

        prov_path = os.path.join(args.dataset_root, "noise_provenance.csv")
        write_header = not os.path.exists(prov_path)
        with open(prov_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PROVENANCE_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(provenance_rows)

        rebuild_noise_manifest(args.dataset_root)
        print(f"\ninstalled {total_installed} clips, {len(new_categories)} "
              f"new categor{'y' if len(new_categories) == 1 else 'ies'} "
              f"({', '.join(new_categories) if new_categories else 'none'}), "
              f"rebuilt noise_manifest.csv.")
    else:
        print(f"\nnothing written ({total_installed} clips would be "
              f"installed). re-run with --apply to commit.")


def rebuild_noise_manifest(dataset_root):
    """Scans dataset_v2/noise_data/ and writes noise_manifest.csv fresh --
    the same schema mixing.py's read_noise_manifest expects. Deliberately
    not build_balanced_manifests.py: that script also rebalances the speech
    manifests, which v2 never touches."""
    noise_dir = os.path.join(dataset_root, "noise_data")
    rows = []
    for split, split_dir in NOISE_SPLIT_DIRS.items():
        base = os.path.join(noise_dir, split_dir)
        if not os.path.isdir(base):
            continue
        for cat in sorted(os.listdir(base)):
            cat_dir = os.path.join(base, cat)
            if not os.path.isdir(cat_dir):
                continue
            for name in sorted(os.listdir(cat_dir)):
                path = os.path.join(cat_dir, name)
                duration = sf.info(path).frames / TARGET_SR
                rows.append({
                    "filename": os.path.relpath(path, dataset_root),
                    "category": cat,
                    "split": split,
                    "duration_seconds": round(duration, 3),
                })
    path = os.path.join(dataset_root, "noise_manifest.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NOISE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
