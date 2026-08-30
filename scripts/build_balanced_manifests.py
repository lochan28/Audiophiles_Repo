"""
build_balanced_manifests.py

Applies the two dataset selection rules to the collected data and writes the
result out as new manifest CSVs. Nothing on disk is deleted or renamed --
the surplus clips stay where they are, they simply do not appear in the
balanced manifests.

Rule 1 (clean speech): the target language proportions are the authority and
any surplus above them is ignored. Each split is trimmed to the largest total
that still gives English 25%, Hindi 25%, and the remaining 50% to the other
21 scheduled languages. That 50% is water-filled across those 21: a language
with few clips contributes everything it has, and the shortfall is absorbed
by the languages that still have spare clips.

Rule 2 (noise): a noise category counts only if it is present in all three
splits. A category that appears in some splits but not others is dropped
entirely, since a class with no validation or test data cannot be scored.

Writes:
    dataset/balanced_train_manifest.csv
    dataset/balanced_validation_manifest.csv
    dataset/balanced_testing_manifest.csv
    dataset/noise_manifest.csv
"""

import os
import csv
import collections
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
NOISE_DIR = os.path.join(DATASET_DIR, "noise_data")

SPLITS = ["train", "validation", "testing"]
NOISE_SPLIT_DIRS = {
    "train": "noise_train",
    "validation": "noise_validation",
    "testing": "noise_testing",
}

MAJOR_LANGUAGES = ["English", "Hindi"]
MAJOR_SHARE = 0.25          # each of English and Hindi
OTHER_LANGUAGE_COUNT = 21   # the other scheduled languages sharing the rest

SPEECH_FIELDNAMES = ["filename", "language", "speaker_id", "duration_seconds"]
NOISE_FIELDNAMES = ["filename", "category", "split", "duration_seconds"]


def read_manifest(split):
    path = os.path.join(DATASET_DIR, f"{split}_manifest.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def water_fill(available, budget):
    """Spread budget over the languages in available as evenly as supply allows.

    Languages that cannot meet the equal share give everything they have and
    drop out; the shortfall is redistributed over the ones that still have
    spare clips, repeating until the budget is placed.
    """
    remaining = dict(available)
    allocation = {}
    left = budget

    while remaining and left > 0:
        share = left / len(remaining)
        exhausted = {k: v for k, v in remaining.items() if v <= share}

        if not exhausted:
            # everyone left can meet the share, so split it evenly and stop
            base = left // len(remaining)
            extra = left - base * len(remaining)
            ordered = sorted(remaining, key=lambda k: (-remaining[k], k))
            for i, lang in enumerate(ordered):
                allocation[lang] = base + (1 if i < extra else 0)
            left = 0
            break

        for lang, count in exhausted.items():
            allocation[lang] = count
            left -= count
            del remaining[lang]

    for lang in available:
        allocation.setdefault(lang, 0)
    return allocation


def pick_clips(rows, keep):
    """Pick keep rows from one language, spreading the choice over speakers.

    Taking clips in file order would drop whole speakers off the end of a
    language, so speakers are visited round-robin instead and each gives up
    one clip per pass.
    """
    if keep >= len(rows):
        return list(rows)

    by_speaker = collections.OrderedDict()
    for row in sorted(rows, key=lambda r: r["filename"]):
        by_speaker.setdefault(row["speaker_id"], []).append(row)

    chosen = []
    queues = list(by_speaker.values())
    depth = 0
    while len(chosen) < keep:
        placed_any = False
        for queue in queues:
            if depth < len(queue):
                chosen.append(queue[depth])
                placed_any = True
                if len(chosen) == keep:
                    break
        if not placed_any:
            break
        depth += 1

    return sorted(chosen, key=lambda r: r["filename"])


def balance_split(split):
    rows = read_manifest(split)
    by_language = collections.OrderedDict()
    for row in rows:
        by_language.setdefault(row["language"], []).append(row)

    counts = {lang: len(v) for lang, v in by_language.items()}
    others = {k: v for k, v in counts.items() if k not in MAJOR_LANGUAGES}
    others_total = sum(others.values())

    # largest total that keeps every share satisfiable from what is on disk
    limits = [int(others_total / (1 - 2 * MAJOR_SHARE))]
    for lang in MAJOR_LANGUAGES:
        limits.append(int(counts.get(lang, 0) / MAJOR_SHARE))
    total = min(limits)

    major_keep = int(total * MAJOR_SHARE)
    others_budget = total - 2 * major_keep

    allocation = water_fill(others, others_budget)
    for lang in MAJOR_LANGUAGES:
        allocation[lang] = major_keep

    selected = []
    for lang, lang_rows in by_language.items():
        selected.extend(pick_clips(lang_rows, allocation.get(lang, 0)))
    selected.sort(key=lambda r: r["filename"])

    out_path = os.path.join(DATASET_DIR, f"balanced_{split}_manifest.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPEECH_FIELDNAMES)
        writer.writeheader()
        writer.writerows(selected)

    kept_others = sum(v for k, v in allocation.items() if k not in MAJOR_LANGUAGES)
    print(f"{split}: {len(rows)} on disk -> {len(selected)} kept "
          f"(English {allocation['English']}, Hindi {allocation['Hindi']}, "
          f"others {kept_others} over {len(others)} languages)")
    for lang in sorted(others):
        dropped = others[lang] - allocation[lang]
        if dropped:
            print(f"    {lang}: kept {allocation[lang]} of {others[lang]}")
    for lang in MAJOR_LANGUAGES:
        print(f"    {lang}: kept {allocation[lang]} of {counts.get(lang, 0)}")
    return out_path


def categories_in(split):
    root = os.path.join(NOISE_DIR, NOISE_SPLIT_DIRS[split])
    if not os.path.isdir(root):
        return set()
    return {d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}


def build_noise_manifest():
    per_split = {split: categories_in(split) for split in SPLITS}
    shared = set.intersection(*per_split.values())
    dropped = set.union(*per_split.values()) - shared

    rows = []
    for split in SPLITS:
        split_dir = NOISE_SPLIT_DIRS[split]
        for category in sorted(shared):
            folder = os.path.join(NOISE_DIR, split_dir, category)
            for name in sorted(os.listdir(folder)):
                if not name.lower().endswith(".wav"):
                    continue
                path = os.path.join(folder, name)
                info = sf.info(path)
                rows.append({
                    "filename": os.path.relpath(path, DATASET_DIR),
                    "category": category,
                    "split": split,
                    "duration_seconds": round(info.frames / info.samplerate, 3),
                })

    out_path = os.path.join(DATASET_DIR, "noise_manifest.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NOISE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nnoise: {len(shared)} categories usable -> {sorted(shared)}")
    if dropped:
        for category in sorted(dropped):
            present = [s for s in SPLITS if category in per_split[s]]
            print(f"    dropped {category}: only in {', '.join(present)}")
    print(f"    {len(rows)} noise clips listed")
    return out_path


def main():
    for split in SPLITS:
        balance_split(split)
    build_noise_manifest()


if __name__ == "__main__":
    main()
