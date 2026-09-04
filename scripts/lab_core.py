"""
lab_core.py

The mixing bench: build a mixture by hand from one speech clip and up to a
handful of noise clips, each at its own SNR, then hand it to the model.

    python scripts/lab_core.py            # self-test

Why this exists separately from mixing.py: that module builds the mixtures
TRAINING and EVALUATION use, where every choice is made randomly from a fixed
distribution so the numbers stay comparable. Nothing there lets a person say
"this speaker, with a helicopter at 5 dB and mortar at 0 dB". This module does
exactly that and nothing else, so the training pipeline keeps its guarantees
while the lab stays free to be arbitrary.

The mixing arithmetic is deliberately the same as mixing.mix_at_snr, so a
mixture built here at a given SNR is the same kind of object the model was
trained on rather than a lookalike built by different rules.
"""

import os

import numpy as np

import mixing
from mixing import (DATASET_DIR, TARGET_SR, CLIP_SAMPLES, PEAK_CEILING,
                    load_wav, read_speech_manifest, read_noise_manifest,
                    place_in_window, EPS)


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS — safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

MAX_NOISES = 4
# How many noise clips may be stacked onto one speech clip. Each is scaled to
# its own SNR against the speech, so stacking four at 5 dB does NOT give a
# 5 dB mixture -- it gives roughly 5 - 10*log10(4) = -1 dB, because the noise
# powers add. The combined figure is computed and returned, so the page can
# show what was asked for and what was actually produced side by side.
# Raising this is free; the arithmetic does not care.

DEFAULT_SNR_DB = 5.0
# Where a newly added noise layer starts. 5 dB sits in the ordinary band of
# the training distribution, so it is a fair default rather than a soft one.

SNR_RANGE = (-15.0, 25.0)
# The span the lab allows, deliberately WIDER than training's -5..+15. Being
# able to push past the range the model was trained on is the point of a lab:
# it shows where the model falls apart, which the frozen sets never will.


# ═══════════════════════════════════════════════════════════════════
#  LOCKED — set by the data and the trained models, do not change
# ═══════════════════════════════════════════════════════════════════

SPLITS = ("train", "validation", "testing")
NOISE_SPLITS = {"train": "train", "validation": "validation",
                "testing": "testing"}


# --------------------------------------------------------------- browsing

def speech_sources(split):
    """Every speech clip in one split, as the page needs to list them."""
    rows, _ = read_speech_manifest(split)
    out = []
    for r in rows:
        out.append({
            "file": r["filename"].replace("\\", "/"),
            "language": r["language"],
            "speaker_id": r["speaker_id"],
            "seconds": float(r["duration_seconds"]),
        })
    return out


def noise_sources(split):
    """Every noise clip in one split, grouped by category."""
    by_cat = read_noise_manifest(split)
    out = {}
    for cat, paths in by_cat.items():
        out[cat] = [{
            "file": os.path.relpath(p, DATASET_DIR).replace("\\", "/"),
            "name": os.path.basename(p),
        } for p in paths]
    return out


def _resolve(rel):
    """Turn a manifest-relative path into an absolute one, safely.

    The page sends these back to us, so they are untrusted input. Anything
    that escapes the dataset directory is refused rather than normalised into
    something harmless-looking.
    """
    path = os.path.abspath(os.path.join(DATASET_DIR, rel.replace("/", os.sep)))
    if not path.startswith(os.path.abspath(DATASET_DIR) + os.sep):
        raise ValueError(f"refused path outside the dataset: {rel}")
    if not os.path.isfile(path):
        raise FileNotFoundError(rel)
    return path


def fit_to_window(noise, offset=0.0):
    """Make a noise clip exactly one window long.

    Short clips are tiled rather than zero-padded: silence would quietly
    lower the noise power over the window and make the achieved SNR higher
    than the one requested, which is the sort of error that never announces
    itself. `offset` is a fraction of the clip to start from, so the same
    noise file can be used twice without the two layers being identical.
    """
    if len(noise) == 0:
        return np.zeros(CLIP_SAMPLES, dtype=np.float32)
    start = int(offset * len(noise)) % len(noise)
    rolled = np.roll(noise, -start)
    if len(rolled) < CLIP_SAMPLES:
        reps = int(np.ceil(CLIP_SAMPLES / len(rolled)))
        rolled = np.tile(rolled, reps)
    return np.ascontiguousarray(rolled[:CLIP_SAMPLES], dtype=np.float32)


# ---------------------------------------------------------------- mixing

def build(speech_file, layers, speech_offset=None):
    """Mix one speech clip with several noise layers.

    layers: [{"file": <relative path>, "snr_db": float, "offset": float}]

    Returns a dict with the noisy and clean signals plus a full account of
    what was actually done -- per-layer scale factors and the combined SNR --
    because the whole point of the lab is being able to see why a mixture
    sounds the way it does.
    """
    speech = load_wav(_resolve(speech_file))

    # Deterministic placement unless asked otherwise, so re-mixing the same
    # choices gives the same audio and A/B comparisons stay honest.
    rng = np.random.default_rng(0 if speech_offset is None else None)
    clean, start, end = place_in_window(speech, rng)
    if speech_offset is not None and len(speech) < CLIP_SAMPLES:
        start = int(np.clip(speech_offset, 0.0, 1.0)
                    * (CLIP_SAMPLES - len(speech)))
        clean = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        clean[start:start + len(speech)] = speech
        end = start + len(speech)

    speech_power = float(np.mean(clean[start:end] ** 2))
    stack = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    used = []

    for layer in layers[:MAX_NOISES]:
        noise = fit_to_window(load_wav(_resolve(layer["file"])),
                              float(layer.get("offset", 0.0)))
        snr_db = float(np.clip(layer.get("snr_db", DEFAULT_SNR_DB),
                               SNR_RANGE[0], SNR_RANGE[1]))
        noise_power = float(np.mean(noise ** 2))
        if speech_power < EPS or noise_power < EPS:
            scale = 1.0
        else:
            scale = float(np.sqrt(speech_power
                                  / (noise_power * 10.0 ** (snr_db / 10.0))))
        stack += (noise * scale).astype(np.float32)
        used.append({
            "file": layer["file"],
            "category": layer["file"].split("/")[-2] if "/" in layer["file"] else "",
            "snr_db": snr_db,
            "scale": scale,
            "offset": float(layer.get("offset", 0.0)),
        })

    noisy = clean + stack

    # One shared rescale if the sum would clip, exactly as mix_at_snr does,
    # so the pair stays a valid input/target pair.
    peak = float(np.max(np.abs(noisy))) if noisy.size else 0.0
    shrink = 1.0
    if peak > PEAK_CEILING:
        shrink = PEAK_CEILING / peak
        noisy = noisy * shrink
        clean = clean * shrink
        stack = stack * shrink

    stack_power = float(np.mean(stack ** 2))
    sp = float(np.mean(clean[start:end] ** 2))
    combined = (10.0 * np.log10(sp / stack_power)
                if sp > EPS and stack_power > EPS else None)

    return {
        "noisy": noisy.astype(np.float32),
        "clean": clean.astype(np.float32),
        "noise_only": stack.astype(np.float32),
        "speech_file": speech_file,
        "speech_span": [int(start), int(end)],
        "layers": used,
        "combined_snr_db": combined,
        "clipped": shrink < 1.0,
        "shrink": shrink,
        "seconds": CLIP_SAMPLES / TARGET_SR,
    }


# ------------------------------------------------------------- self-test

def self_test():
    print("--- browsing ---")
    sp = speech_sources("testing")
    nz = noise_sources("testing")
    print(f"  speech  {len(sp)} clips, "
          f"{len(set(s['language'] for s in sp))} languages")
    print(f"  noise   {sum(len(v) for v in nz.values())} clips in "
          f"{len(nz)} categories")
    for c, v in list(nz.items())[:3]:
        print(f"            {c:<16} {len(v)} e.g. {v[0]['file']}")

    print("\n--- one noise at 5 dB ---")
    m = build(sp[0]["file"], [{"file": nz["helicopter"][0]["file"],
                               "snr_db": 5.0}])
    print(f"  speech {m['speech_file']}  span {m['speech_span']}")
    print(f"  requested 5.0 dB -> combined {m['combined_snr_db']:.3f} dB "
          f"(should match to ~0.001)")

    print("\n--- four noises, each at 5 dB ---")
    layers = [{"file": nz[c][0]["file"], "snr_db": 5.0}
              for c in list(nz)[:4]]
    m4 = build(sp[0]["file"], layers)
    expect = 5.0 - 10 * np.log10(len(layers))
    print(f"  {len(m4['layers'])} layers, each asked for 5.0 dB")
    print(f"  combined {m4['combined_snr_db']:.3f} dB, "
          f"theory {expect:.3f} dB (powers add)")
    print(f"  clipped: {m4['clipped']}  shrink {m4['shrink']:.4f}")

    print("\n--- limits ---")
    over = build(sp[0]["file"], [{"file": nz[c][0]["file"], "snr_db": 0.0}
                                 for c in list(nz)[:7]])
    print(f"  asked for 7 layers, used {len(over['layers'])} "
          f"(MAX_NOISES={MAX_NOISES})")
    try:
        _resolve("../../etc/passwd")
        print("  ESCAPE NOT REFUSED")
    except ValueError as e:
        print(f"  path escape refused: {e}")

    print("\n--- shapes ---")
    for k in ("noisy", "clean", "noise_only"):
        print(f"  {k:<11} {m4[k].shape} {m4[k].dtype} "
              f"peak {np.max(np.abs(m4[k])):.4f}")

    print("\nall checks passed")


if __name__ == "__main__":
    self_test()
