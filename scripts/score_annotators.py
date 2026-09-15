"""
score_annotators.py

Audits annotator output against the MAD answer key, before
merge_annotated_noise.py installs anything. Not an accuracy score against
ground truth -- MAD's own class is a coarse, single label for a whole raw
clip that may contain more than one real sound, and the three annotators'
sets are disjoint by design, so there is no inter-annotator agreement to
compute either. What this reports:

  - a tabulation of each person's chosen category against MAD's class, so a
    systematic mislabelling shows up as an off-diagonal cluster
  - files never used at all (every window from that raw file discarded)
  - clips produced per raw file that WAS used
  - clips flagged quiet (level_dbfs < QUIET_THRESHOLD_DBFS) -- the annotator
    tool exports these anyway rather than dropping them; they need a listen
    before merge_annotated_noise.py installs them

The category mapping below is loose and stated here rather than assumed
silently -- MAD's seven classes do not line up one-to-one with our seven:

    MAD class      -> our category (expected)
    helicopter     -> helicopter
    gunshot        -> assault_rifle
    vehicle        -> engine
    shelling       -> heavy_artillery OR mortar (either is a reasonable call)
    fighter        -> engine at best (loose -- jet noise is not rotor noise)
    footsteps      -> ambient, or discarded (footsteps alone rarely reads as
                      a battlefield noise category)
    communication  -> should have been discarded (people talking is not
                      noise for this project)

A category outside this table (mortar, rocket -- MAD has no equivalent
class for either) is reported as "no MAD equivalent" rather than flagged
wrong; the annotators are trusted on these since nothing here can check them.

    python scripts/score_annotators.py
    python scripts/score_annotators.py --drop-dir incoming_noise --assignment-csv C:\\mad_splits\\assignment.csv
"""

import os
import csv
import argparse
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DROP_DIR = os.path.join(PROJECT_ROOT, "incoming_noise")
DEFAULT_ASSIGNMENT_CSV = r"C:\mad_splits\assignment.csv"

QUIET_THRESHOLD_DBFS = -45.0

# MAD class -> set of our categories considered a correct call. A MAD class
# absent from this dict, or an incoming category with no key pointing at
# it, is not scored as right or wrong -- see module docstring.
EXPECTED = {
    "helicopter":    {"helicopter"},
    "gunshot":       {"assault_rifle"},
    "vehicle":       {"engine"},
    "shelling":      {"heavy_artillery", "mortar"},
    "fighter":       {"engine"},
    "footsteps":     {"ambient"},
    "communication": set(),          # nothing is a correct call -- should be discarded
}


def find_people(drop_dir):
    if not os.path.isdir(drop_dir):
        return []
    return sorted(p for p in os.listdir(drop_dir)
                  if os.path.isfile(os.path.join(drop_dir, p, "classified", "manifest.csv")))


def read_manifest(drop_dir, person):
    path = os.path.join(drop_dir, person, "classified", "manifest.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_assignment(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_person = defaultdict(dict)
    for r in rows:
        by_person[r["person"]][r["file"]] = r
    return by_person


def score_person(person, manifest_rows, assigned_files):
    """Returns a dict of everything reportable for one person."""
    confusion = Counter()               # (our_category, mad_class) -> count
    unmapped_mad_class = Counter()      # mad classes not in EXPECTED, seen anyway
    no_equivalent_category = Counter()  # our categories with no MAD key pointing at them
    unknown_source = []                 # source files not found in assignment.csv at all
    used_files = set()
    clips_per_file = Counter()
    quiet = []
    correct = wrong = unscored = 0

    for row in manifest_rows:
        source = row.get("source", "")
        clips_per_file[source] += 1
        used_files.add(source)

        try:
            level = float(row.get("level_dbfs", "nan"))
        except ValueError:
            level = float("nan")
        if level == level and level < QUIET_THRESHOLD_DBFS:   # NaN-safe check
            quiet.append((row.get("clip", ""), level))

        a_row = assigned_files.get(source)
        if a_row is None:
            unknown_source.append(source)
            continue

        mad_class = a_row["class"]
        our_cat = row.get("category", "")
        confusion[(our_cat, mad_class)] += 1

        expected = EXPECTED.get(mad_class)
        if expected is None:
            unmapped_mad_class[mad_class] += 1
            unscored += 1
        elif not expected:
            wrong += 1   # communication: nothing is correct
        elif our_cat in expected:
            correct += 1
        else:
            wrong += 1

        reverse_mapped = any(our_cat in exp for exp in EXPECTED.values())
        if not reverse_mapped and mad_class not in EXPECTED:
            no_equivalent_category[our_cat] += 1

    never_used = sorted(set(assigned_files) - used_files)
    files_with_clips = {f: c for f, c in clips_per_file.items() if f in assigned_files or f}
    avg_clips = (sum(files_with_clips.values()) / len(files_with_clips)
                if files_with_clips else 0.0)

    return {
        "n_clips": len(manifest_rows),
        "confusion": confusion,
        "unmapped_mad_class": unmapped_mad_class,
        "unknown_source": unknown_source,
        "never_used_files": never_used,
        "avg_clips_per_used_file": avg_clips,
        "quiet": quiet,
        "correct": correct,
        "wrong": wrong,
        "unscored": unscored,
    }


def print_report(person, s, total_assigned):
    scored = s["correct"] + s["wrong"]
    rate = f"{s['correct'] / scored:.1%}" if scored else "n/a"
    print(f"\n=== {person} ===")
    print(f"  {s['n_clips']} clips, from {total_assigned} raw files assigned")
    print(f"  {len(s['never_used_files'])} raw files never used at all "
          f"(no window exported from them)")
    print(f"  {s['avg_clips_per_used_file']:.2f} clips per raw file that WAS used")
    print(f"  category matched MAD's loose expectation: {rate} "
          f"({s['correct']} of {scored} scoreable; {s['unscored']} clips had "
          f"no MAD equivalent to check against, see module docstring)")
    if s["quiet"]:
        print(f"  {len(s['quiet'])} clips flagged quiet (< {QUIET_THRESHOLD_DBFS:g} dBFS) "
              f"-- LISTEN before installing:")
        for clip, level in s["quiet"][:10]:
            print(f"      {clip}  {level:.1f} dBFS")
        if len(s["quiet"]) > 10:
            print(f"      ... and {len(s['quiet']) - 10} more")
    if s["unknown_source"]:
        print(f"  WARNING: {len(s['unknown_source'])} clips reference a source "
              f"file not in assignment.csv for this person -- manifest/answer-key "
              f"mismatch, investigate before trusting this report")
    if s["unmapped_mad_class"]:
        print(f"  MAD classes with no scoring rule (see EXPECTED in this script): "
              + ", ".join(f"{c} x{n}" for c, n in s["unmapped_mad_class"].most_common()))

    print(f"  category x MAD-class tabulation (our_category | mad_class | count):")
    for (our_cat, mad_class), n in sorted(s["confusion"].items(),
                                          key=lambda kv: -kv[1]):
        expected = EXPECTED.get(mad_class)
        mark = ("+" if expected and our_cat in expected else
               " " if expected is None else "x")
        print(f"      [{mark}] {our_cat:<16} | {mad_class:<14} | {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-dir", default=DEFAULT_DROP_DIR, metavar="DIR")
    ap.add_argument("--assignment-csv", default=DEFAULT_ASSIGNMENT_CSV, metavar="PATH")
    args = ap.parse_args()

    if not os.path.exists(args.assignment_csv):
        raise SystemExit(f"assignment.csv not found at {args.assignment_csv} "
                         f"-- this script needs the MAD answer key to score anything")

    people = find_people(args.drop_dir)
    if not people:
        print(f"nothing to score. Drop <person>/classified/ folders under "
              f"{os.path.relpath(args.drop_dir, PROJECT_ROOT)}/ first.")
        return

    assignment = read_assignment(args.assignment_csv)
    print(f"scoring {len(people)} annotator(s) against {args.assignment_csv}")
    print("(no inter-annotator agreement to compute -- the three sets are "
          "disjoint by design; see module docstring for the mapping used)")

    for person in people:
        manifest_rows = read_manifest(args.drop_dir, person)
        assigned = assignment.get(person, {})
        if not assigned:
            print(f"\n=== {person} ===\n  WARNING: no rows for {person!r} in "
                  f"assignment.csv -- cannot score")
            continue
        s = score_person(person, manifest_rows, assigned)
        print_report(person, s, len(assigned))


if __name__ == "__main__":
    main()
