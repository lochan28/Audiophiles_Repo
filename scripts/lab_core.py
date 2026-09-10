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

MAX_UPLOAD_SECONDS = 120.0
# Longest uploaded clip the lab will mix. Two minutes is far more than any
# demonstration needs and keeps one request from tying the server up. The model
# itself has no length limit, so raising this costs memory and time only.

SPEECH_GATE_DB = -35.0
# How far below the loudest 10 ms frame a frame may sit and still count as
# speech, when measuring the level of an UPLOADED clip.
# WHY IT EXISTS: SNR is defined against the power of the speech, not of the
# whole file. A recording that opens with three seconds of room tone would
# have its average power dragged down by that silence, and every noise layer
# would then be scaled quieter than the number you asked for. Gating fixes it.
# LOWER (-50) counts quiet breaths and room tone as speech; HIGHER (-25)
# keeps only the loud parts and makes noise louder for a given SNR.


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


def fit_to_window(noise, offset=0.0, n=CLIP_SAMPLES):
    """Make a noise clip exactly one window long.

    Short clips are tiled rather than zero-padded: silence would quietly
    lower the noise power over the window and make the achieved SNR higher
    than the one requested, which is the sort of error that never announces
    itself. `offset` is a fraction of the clip to start from, so the same
    noise file can be used twice without the two layers being identical.
    """
    if len(noise) == 0:
        return np.zeros(n, dtype=np.float32)
    start = int(offset * len(noise)) % len(noise)
    rolled = np.roll(noise, -start)
    if len(rolled) < n:
        reps = int(np.ceil(n / len(rolled)))
        rolled = np.tile(rolled, reps)
    return np.ascontiguousarray(rolled[:n], dtype=np.float32)


def active_mask(wav, frame=160):
    """Which samples of an arbitrary recording count as speech.

    A dataset clip has a known speech span because we placed it ourselves. An
    uploaded file does not, so its level has to be measured rather than looked
    up. This gates on per-frame RMS relative to the loudest frame, which is
    crude but enough to stop leading and trailing silence from making every
    noise layer quieter than requested.

    Falls back to "all of it" if the gate would reject everything, so a very
    flat recording still mixes rather than erroring.
    """
    n = (len(wav) // frame) * frame
    if n == 0:
        return np.ones(len(wav), dtype=bool)
    rms = np.sqrt((wav[:n].reshape(-1, frame) ** 2).mean(axis=1) + 1e-20)
    keep = rms >= rms.max() * (10.0 ** (SPEECH_GATE_DB / 20.0))
    mask = np.repeat(keep, frame)
    if len(mask) < len(wav):
        mask = np.concatenate([mask, np.zeros(len(wav) - len(mask), bool)])
    return mask if mask.any() else np.ones(len(wav), dtype=bool)


# ---------------------------------------------------------------- mixing

MAX_SPEECH_CLIPS = 4
# How many speech clips may be chained into one lab mixture. Each adds three
# seconds, so 3 gives the ~9 s length used for demonstrations. The model has no
# length limit; this only bounds how much a single request can ask for.


def build(speech_file, layers, speech_offset=None, speech_wav=None,
          speech_name=None):
    """Mix speech with several noise layers.

    Speech comes from one of two places:

      * `speech_file` -- one dataset clip, or a list chained end to end. Each
        sits in its own 3 second window, which is how the model was trained.
      * `speech_wav`  -- raw float32 at 16 kHz, ANY length. This is the upload
        path: the recording is used exactly as given, no padding and no
        windowing, and its speech level is measured rather than assumed.

    layers: [{"file": <relative path>, "snr_db": float, "offset": float}]

    Every noise layer is stretched to cover the whole recording, however long
    that is -- a 3 second noise clip under a 40 second upload is repeated
    until it reaches the end. See fit_to_window.

    Returns a dict with the noisy and clean signals plus a full account of
    what was actually done -- per-layer scale factors and the combined SNR --
    because the whole point of the lab is being able to see why a mixture
    sounds the way it does.
    """
    if speech_wav is not None:
        clean = np.ascontiguousarray(speech_wav, dtype=np.float32)
        cap = int(MAX_UPLOAD_SECONDS * TARGET_SR)
        truncated = len(clean) > cap
        clean = clean[:cap]
        if clean.size == 0:
            raise ValueError("uploaded audio is empty")
        mask = active_mask(clean)
        idx = np.flatnonzero(mask)
        return _mix(clean, [(int(idx[0]), int(idx[-1]) + 1)], mask, layers,
                    files=[speech_name or "uploaded audio"],
                    truncated=truncated)

    # One clip or several chained end to end. Several is what makes a lab
    # mixture long enough to demonstrate: the listener settles in, and the
    # noise underneath can change while the speech keeps going.
    files = [speech_file] if isinstance(speech_file, str) else list(speech_file)
    files = files[:MAX_SPEECH_CLIPS]

    # Deterministic placement unless asked otherwise, so re-mixing the same
    # choices gives the same audio and A/B comparisons stay honest.
    rng = np.random.default_rng(0 if speech_offset is None else None)
    blocks, spans = [], []
    for k, f in enumerate(files):
        speech = load_wav(_resolve(f))
        placed, s0, e0 = place_in_window(speech, rng)
        if speech_offset is not None and len(speech) < CLIP_SAMPLES:
            s0 = int(np.clip(speech_offset, 0.0, 1.0)
                     * (CLIP_SAMPLES - len(speech)))
            placed = np.zeros(CLIP_SAMPLES, dtype=np.float32)
            placed[s0:s0 + len(speech)] = speech
            e0 = s0 + len(speech)
        blocks.append(placed)
        spans.append((k * CLIP_SAMPLES + s0, k * CLIP_SAMPLES + e0))

    clean = np.concatenate(blocks).astype(np.float32)
    return _mix(clean, spans, None, layers, files=files, truncated=False)


def _mix(clean, spans, mask, layers, files, truncated):
    """The mixing itself, shared by the dataset path and the upload path.

    `spans` are the speech-active regions used to measure level. `mask` is an
    optional per-sample version of the same thing, which the upload path uses
    because its active region is gated rather than contiguous.

    Keeping one implementation matters: if the two paths measured level
    differently, an SNR chosen in the lab would mean two different things
    depending on where the speech came from.
    """
    window = len(clean)
    start, end = spans[0][0], spans[-1][1]

    # Speech power over the speech-active regions ONLY, exactly as
    # mixing.mix_at_snr does. Averaging across the padding or the silence
    # would understate it and make every layer quieter than requested.
    def level(sig):
        act = sig[mask] if mask is not None else             np.concatenate([sig[a:b] for a, b in spans])
        return float(np.mean(act ** 2)) if act.size else 0.0

    speech_power = level(clean)
    stack = np.zeros(window, dtype=np.float32)
    used = []

    for layer in layers[:MAX_NOISES]:
        # fit_to_window repeats a short noise clip until it covers the whole
        # recording, so a 3 second siren under a 40 second upload simply
        # continues rather than stopping a tenth of the way in.
        noise = fit_to_window(load_wav(_resolve(layer["file"])),
                              float(layer.get("offset", 0.0)), window)
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
            "repeats": window / max(len(load_wav(_resolve(layer["file"]))), 1),
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
    sp = level(clean)
    combined = (10.0 * np.log10(sp / stack_power)
                if sp > EPS and stack_power > EPS else None)

    return {
        "noisy": noisy.astype(np.float32),
        "clean": clean.astype(np.float32),
        "noise_only": stack.astype(np.float32),
        "speech_file": " + ".join(files),
        "speech_files": files,
        "speech_span": [int(start), int(end)],
        "spans": [[int(a), int(b)] for a, b in spans],
        "layers": used,
        "combined_snr_db": combined,
        "clipped": shrink < 1.0,
        "shrink": shrink,
        "truncated": truncated,
        "seconds": window / TARGET_SR,
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
