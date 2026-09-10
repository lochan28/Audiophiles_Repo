"""
make_demo_clips.py

Builds long clips for demonstrating the system to an audience.

    python scripts/make_demo_clips.py            # dry run
    python scripts/make_demo_clips.py --apply

WHY THESE EXIST

The frozen evaluation mixtures are 3 seconds each, which is the right length
for training and scoring and the wrong length for showing somebody. Three
seconds is over before a listener has settled, and a single 3-second clip
contains one noise type, so it cannot show the one thing that is most worth
showing: that the noise CHANGES mid-sentence and the model keeps up.

Each demo clip is three frozen TESTING mixtures joined end to end -- about nine
seconds -- deliberately chosen so that **every segment is a different noise
category**. A listener hears gunfire give way to a helicopter give way to
mortar without the speech breaking up.

WHY THEY ARE STILL HONEST

They are built only from the held-out TESTING split, and both halves of each
pair are joined the same way, so a demo clip still has a true clean reference
and can be scored with the same code as anything else. Nothing is regenerated
and nothing in `generated_test_mixtures/` is touched -- these are written to
their own directory, so the frozen evaluation set stays byte-identical.

THE JOINS ARE CROSSFADED, AND THAT IS NOT COSMETIC

Butting two clips together leaves a step discontinuity that clicks. On a demo
that click would be heard as an artefact OF THE MODEL, which would be actively
misleading. A short equal-power crossfade removes it. The identical fade is
applied to the noisy and the clean side, so the pair stays aligned sample for
sample and the scores stay valid.
"""

import os
import csv
import sys
import argparse

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mixing import TARGET_SR, CLIP_SAMPLES, PROJECT_ROOT, load_wav


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS — safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

SEGMENTS = 3
# How many 3-second mixtures are joined into one demo clip. 3 gives ~9 s,
# which is long enough to settle into and short enough to hold attention.
# Raising this makes a longer clip; the model has no length limit.

N_CLIPS = 8
# How many demo clips to build. Eight is enough to pick a good one on stage
# without hunting, and cheap: the whole set is under 90 seconds of audio.

CROSSFADE_MS = 12.0
# Length of the equal-power crossfade at each join, in milliseconds. LONGER
# hides the seam more thoroughly but eats real audio from both sides; SHORTER
# risks an audible click that a listener would blame on the model. 12 ms is
# about one STFT frame, so it is invisible to the model and inaudible to a
# listener. Set to 0 to butt-join with no fade.

SEED = 7
# Chooses which mixtures get combined. Change it for a different selection;
# the same seed always produces the same demo set.


# ═══════════════════════════════════════════════════════════════════
#  LOCKED — set by the frozen evaluation data, do not change
# ═══════════════════════════════════════════════════════════════════

SOURCE_SPLIT = "testing"        # held-out only; never demo on tuning data
MIXTURE_DIR = os.path.join(PROJECT_ROOT, "generated_test_mixtures")
OUT_DIR = os.path.join(PROJECT_ROOT, "generated_demo_clips")
MANIFEST = "demo_clips_manifest.csv"
SUBTYPE = "PCM_16"

FADE = int(TARGET_SR * CROSSFADE_MS / 1000.0)


def read_manifest():
    path = os.path.join(MIXTURE_DIR, "test_mixtures_manifest.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r["split"] == SOURCE_SPLIT]


def join(pieces):
    """Concatenate with an equal-power crossfade at every join.

    Equal power (cos/sin) rather than linear, because a linear crossfade of two
    uncorrelated signals dips about 3 dB in the middle -- an audible hole right
    where the listener is being asked to judge continuity.
    """
    if FADE <= 0:
        return np.concatenate(pieces).astype(np.float32)
    t = np.linspace(0.0, np.pi / 2.0, FADE, dtype=np.float32)
    fade_out, fade_in = np.cos(t), np.sin(t)

    out = pieces[0].astype(np.float32).copy()
    for nxt in pieces[1:]:
        nxt = nxt.astype(np.float32).copy()
        out[-FADE:] = out[-FADE:] * fade_out + nxt[:FADE] * fade_in
        out = np.concatenate([out, nxt[FADE:]])
    return out.astype(np.float32)


def build(apply):
    rows = read_manifest()
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    cats = sorted(by_cat)
    rng = np.random.default_rng(SEED)

    if len(cats) < SEGMENTS:
        raise RuntimeError(f"need {SEGMENTS} categories, found {len(cats)}")

    noisy_dir = os.path.join(OUT_DIR, "noisy")
    clean_dir = os.path.join(OUT_DIR, "clean")
    if apply:
        os.makedirs(noisy_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)

    used, out_rows = set(), []
    for i in range(N_CLIPS):
        picked = []
        for cat in rng.permutation(cats)[:SEGMENTS]:
            pool = [r for r in by_cat[cat] if r["index"] not in used]
            if not pool:
                pool = by_cat[cat]
            choice = pool[int(rng.integers(len(pool)))]
            used.add(choice["index"])
            picked.append(choice)

        noisy = join([load_wav(os.path.join(MIXTURE_DIR, r["noisy_file"]))
                      for r in picked])
        clean = join([load_wav(os.path.join(MIXTURE_DIR, r["clean_file"]))
                      for r in picked])
        assert len(noisy) == len(clean), "pair went out of alignment"

        name = f"demo_{i:02d}.wav"
        seconds = len(noisy) / TARGET_SR
        # Where each segment starts in the joined clip, so the page can draw
        # the boundaries and name the noise in each stretch.
        step = (CLIP_SAMPLES - FADE) / TARGET_SR
        bounds = [round(k * step, 3) for k in range(SEGMENTS)]

        print(f"  {name}  {seconds:5.2f}s  "
              + " -> ".join(f"{r['category']}@{float(r['snr_db']):.1f}dB"
                            for r in picked))

        if apply:
            sf.write(os.path.join(noisy_dir, name), noisy, TARGET_SR,
                     subtype=SUBTYPE)
            sf.write(os.path.join(clean_dir, name), clean, TARGET_SR,
                     subtype=SUBTYPE)

        out_rows.append({
            "index": i,
            "noisy_file": f"noisy/{name}",
            "clean_file": f"clean/{name}",
            "seconds": round(seconds, 3),
            "segments": SEGMENTS,
            "categories": "|".join(r["category"] for r in picked),
            "snr_db": "|".join(f"{float(r['snr_db']):.2f}" for r in picked),
            "languages": "|".join(r["language"] for r in picked),
            "source_indices": "|".join(str(r["index"]) for r in picked),
            "boundaries_s": "|".join(str(b) for b in bounds),
        })

    if apply:
        with open(os.path.join(OUT_DIR, MANIFEST), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwrote {len(out_rows)} clips and the manifest under "
              f"{os.path.relpath(OUT_DIR, PROJECT_ROOT)}/")
    else:
        print(f"\nnothing written ({len(out_rows)} clips would be created). "
              f"re-run with --apply to commit.")
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    print(f"source: frozen {SOURCE_SPLIT} split (held out)")
    print(f"{SEGMENTS} segments per clip, {CROSSFADE_MS:g} ms crossfade, "
          f"seed {SEED}\n")
    build(args.apply)


if __name__ == "__main__":
    main()
