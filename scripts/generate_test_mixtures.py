"""
generate_test_mixtures.py

Writes the fixed validation and testing mixtures to disk, once.

Training mixtures are regenerated fresh on every run and never stored, but
evaluation audio has to stay identical across model versions -- otherwise a
CRN scored against one set of random mixtures cannot be honestly compared
against a TCN scored on different ones. This script freezes those mixtures
so every model is measured on byte-identical audio.

It drives `mixing.py` with a fixed seed, so the output is reproducible: a
deleted folder can be regenerated exactly, and the same seed always yields
the same speech clip, noise clip, offset, and SNR for a given index.

Both halves of each pair are written out. The clean side is not simply a
copy of the original file -- short clips are padded and randomly placed
inside the 3 second window, and the pair is rescaled together when the sum
would clip -- so the clean target has to be stored as it actually was.

Output:

    generated_test_mixtures/
    ├── validation/noisy/validation_000.wav
    ├── validation/clean/validation_000.wav
    ├── testing/noisy/testing_000.wav
    ├── testing/clean/testing_000.wav
    └── test_mixtures_manifest.csv

Run without arguments for a dry run; pass --apply to write.
"""

import os
import csv
import sys
import numpy as np
import soundfile as sf

from mixing import (SpeechNoiseMixtures, PROJECT_ROOT, TARGET_SR,
                    CLIP_SAMPLES, EPS)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "generated_test_mixtures")
FROZEN_SPLITS = ["validation", "testing"]

SEED = 20260830              # change this only to deliberately reroll the set
SUBTYPE = "PCM_16"

MANIFEST_FIELDNAMES = [
    "split", "index", "noisy_file", "clean_file",
    "speech_file", "language", "speaker_id",
    "noise_file", "category", "label",
    "snr_db", "measured_snr_db", "speech_start_sample", "speech_end_sample",
]


def measured_snr(clean, noisy, span):
    """SNR actually present in the written pair, verifying the mixer.

    This has to use the same definition `mix_at_snr` targets: speech power
    over the speech-active region, noise power over the whole window. The
    two are not interchangeable. Noise energy is not spread evenly across a
    clip -- a mortar burst may sit entirely outside the speech span -- so
    measuring the residual only inside that span reads several dB high for
    the impulsive categories. Window-wide noise power is the stable choice
    for setting the level, since a near-silent speech region would otherwise
    demand a huge scale factor and make the burst outside it deafening.

    Rescaling to avoid clipping multiplies both signals equally and so
    cancels out here.
    """
    start, end = span
    residual = noisy - clean
    sp = float(np.mean(clean[start:end] ** 2))
    np_ = float(np.mean(residual ** 2))
    if sp < EPS or np_ < EPS:
        return float("nan")
    return 10.0 * np.log10(sp / np_)


def generate(apply):
    rows = []
    for split in FROZEN_SPLITS:
        ds = SpeechNoiseMixtures(split, seed=SEED)
        noisy_dir = os.path.join(OUTPUT_DIR, split, "noisy")
        clean_dir = os.path.join(OUTPUT_DIR, split, "clean")
        if apply:
            os.makedirs(noisy_dir, exist_ok=True)
            os.makedirs(clean_dir, exist_ok=True)

        counts, errors = {}, []
        for idx in range(len(ds)):
            m = ds.describe(idx)
            name = f"{split}_{idx:03d}.wav"
            counts[m["category"]] = counts.get(m["category"], 0) + 1

            got = measured_snr(m["clean"], m["noisy"],
                               (m["speech_start"], m["speech_end"]))
            errors.append(abs(got - m["snr_db"]))

            if apply:
                sf.write(os.path.join(noisy_dir, name), m["noisy"],
                         TARGET_SR, subtype=SUBTYPE)
                sf.write(os.path.join(clean_dir, name), m["clean"],
                         TARGET_SR, subtype=SUBTYPE)

            rows.append({
                "split": split,
                "index": idx,
                "noisy_file": os.path.join(split, "noisy", name),
                "clean_file": os.path.join(split, "clean", name),
                "speech_file": m["speech_file"],
                "language": m["language"],
                "speaker_id": m["speaker_id"],
                "noise_file": m["noise_file"],
                "category": m["category"],
                "label": m["label"],
                "snr_db": round(m["snr_db"], 3),
                "measured_snr_db": round(float(got), 3),
                "speech_start_sample": m["speech_start"],
                "speech_end_sample": m["speech_end"],
            })

        print(f"{split}: {len(ds)} mixtures, "
              f"{len(ds)*CLIP_SAMPLES/TARGET_SR/60:.1f} min of audio")
        print(f"    categories: "
              + ", ".join(f"{c} {counts.get(c, 0)}" for c in ds.categories))
        print(f"    requested vs measured SNR: max error "
              f"{max(errors):.4f} dB")

    if apply:
        path = os.path.join(OUTPUT_DIR, "test_mixtures_manifest.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} pairs and the manifest under "
              f"{os.path.relpath(OUTPUT_DIR, PROJECT_ROOT)}/")
    else:
        print(f"\nnothing written ({len(rows)} pairs would be created). "
              f"re-run with --apply to commit.")


if __name__ == "__main__":
    print(f"seed {SEED}\n")
    generate("--apply" in sys.argv)
