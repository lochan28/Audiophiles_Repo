"""
add_noise_class.py

Installs a new noise category dropped into `new_noise_class/`.

Put a folder there named after the class, containing `train/`, `validation/`
and `testing/` subfolders with wav files in each, then run this. It will:

  1. check the class appears in all three splits, refusing otherwise
  2. cut every clip to exact 3.000 s at 16 kHz mono, using the same stream
     rule as the rest of the noise set
  3. write them into dataset/noise_data/noise_<split>/<class>/ renumbered
     from 001
  4. append the class to dataset/noise_classes.txt, taking the next free
     label so existing labels never move
  5. rebuild the manifests

It stops there. It does not train, and it deliberately does not touch the
frozen evaluation mixtures -- those contain no examples of the new class, so
they stay valid for comparison against results measured before it existed.
Generating a second frozen set is a separate, deliberate step.

    python scripts/add_noise_class.py            # dry run
    python scripts/add_noise_class.py --apply    # install

Source files are copied, not moved, so the drop folder stays intact until you
clear it yourself.
"""

import os
import re
import sys
import shutil
import subprocess
import soundfile as sf

from normalize_clip_lengths import (PROJECT_ROOT, NOISE_DIR, NOISE_SPLIT_DIRS,
                                    SPLITS, TARGET_SR, SUBTYPE, CLIP_SECONDS,
                                    recut_category)

DROP_DIR = os.path.join(PROJECT_ROOT, "new_noise_class")
CLASSES_FILE = os.path.join(PROJECT_ROOT, "dataset", "noise_classes.txt")

DROP_SPLIT_DIRS = {"train": "train", "validation": "validation",
                   "testing": "testing"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aif", ".aiff"}


def pinned_classes():
    out = []
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    out.append(line)
    return out


def find_candidates():
    if not os.path.isdir(DROP_DIR):
        return []
    return sorted(d for d in os.listdir(DROP_DIR)
                  if os.path.isdir(os.path.join(DROP_DIR, d)))


def audio_in(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, n) for n in os.listdir(folder)
                  if os.path.splitext(n)[1].lower() in AUDIO_EXT)


def validate(name, existing):
    """Every reason to refuse, collected so the user sees them all at once."""
    problems = []
    if not NAME_RE.match(name):
        problems.append(
            f"name {name!r} is not lowercase_with_underscores -- folder names "
            f"become class labels, so the convention is enforced")
    if name in existing:
        problems.append(f"class {name!r} already exists; this script only adds "
                        f"new ones")

    counts = {}
    for split in SPLITS:
        files = audio_in(os.path.join(DROP_DIR, name, DROP_SPLIT_DIRS[split]))
        counts[split] = len(files)
        if not files:
            problems.append(
                f"no audio in {name}/{DROP_SPLIT_DIRS[split]}/ -- a category "
                f"missing from any split is dropped by the intersection rule, "
                f"so all three are required")
    return problems, counts


def install(name, apply):
    total_in = total_out = 0
    for split in SPLITS:
        src = audio_in(os.path.join(DROP_DIR, name, DROP_SPLIT_DIRS[split]))
        clips, consumed, dropped = recut_category(src)
        total_in += len(src)
        total_out += len(clips)

        print(f"    {split:<11} {len(src):>3} files ({consumed:.1f}s) -> "
              f"{len(clips):>3} clips ({len(clips) * CLIP_SECONDS:.1f}s), "
              f"{dropped:.1f}s dropped")

        if not clips:
            raise RuntimeError(
                f"{name}/{split} yielded no full 3 s clip -- it needs at least "
                f"{CLIP_SECONDS:g}s of audio in total")
        if not apply:
            continue

        dest = os.path.join(NOISE_DIR, NOISE_SPLIT_DIRS[split], name)
        os.makedirs(dest, exist_ok=True)
        if os.listdir(dest):
            raise RuntimeError(f"{dest} already has files in it; refusing")
        for index, clip in enumerate(clips, start=1):
            sf.write(os.path.join(dest, f"{name}_{index:03d}.wav"),
                     clip, TARGET_SR, subtype=SUBTYPE)
    return total_in, total_out


def main():
    apply = "--apply" in sys.argv
    print(f"mode: {'APPLY' if apply else 'dry run'}")

    existing = pinned_classes()
    print(f"existing classes ({len(existing)}): {', '.join(existing)}\n")

    candidates = find_candidates()
    if not candidates:
        print(f"nothing to install. Create a folder under "
              f"{os.path.relpath(DROP_DIR, PROJECT_ROOT)}/ named after the "
              f"class, with train/ validation/ testing/ inside it. "
              f"See its README.md.")
        return

    ok = []
    for name in candidates:
        problems, counts = validate(name, existing)
        print(f"{name}: " + ", ".join(f"{s} {counts[s]}" for s in SPLITS))
        if problems:
            for p in problems:
                print(f"    REFUSED: {p}")
        else:
            ok.append(name)

    if not ok:
        print("\nnothing installed.")
        return

    for name in ok:
        print(f"\ninstalling {name!r} as label {len(existing)}:")
        install(name, apply)
        if apply:
            with open(CLASSES_FILE, "a", encoding="utf-8") as f:
                print(name, file=f)
            existing.append(name)

    if apply:
        print("\nrebuilding manifests...")
        subprocess.run([sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "build_balanced_manifests.py")], check=True)
        print(f"\ndone. N_CLASSES is now {len(existing)} -- update the model "
              f"config before training, and note the frozen evaluation "
              f"mixtures contain no examples of the new class.")
    else:
        print("\nnothing written. re-run with --apply to install.")


if __name__ == "__main__":
    main()
