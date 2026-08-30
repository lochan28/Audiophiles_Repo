"""
mixing.py

Turns the clean speech set and the noise set into training pairs.

A mixture is built by taking one clean speech clip, one noise clip from a
uniformly chosen category, and adding them at a randomly chosen SNR. The
network is trained on the noisy result with the clean clip as its target,
and the noise category is carried alongside as the label for the auxiliary
classification branch described in PROJECT_CONTEXT_4.md Section 11.

Rules this follows, all settled deliberately:

- **SNR is drawn uniformly from -5 to +15 dB.** At the low end the noise is
  louder than the speech, which is what a gunshot or rotor wash over a
  talker actually looks like. The output target is SNR > 15 dB, so this
  range forces real improvement rather than light polishing.
- **Noise categories are sampled uniformly**, one in seven each, not in
  proportion to how many clips each category has on disk. The on-disk counts
  are lopsided (assault_rifle 137 against mortar 45) and proportional
  sampling would bias both the classifier and the denoiser toward the common
  categories.
- **Speech shorter than the 3 second window is zero-padded** and placed at a
  random offset inside it, so the model does not learn that speech always
  starts at t=0. Clips shorter than 1 second are skipped entirely, being too
  sparse to be worth a training slot. Noise still fills the whole window, so
  the padded regions are noise-only — realistic, since real transmissions
  have pauses.
- **Speech power is measured over the real speech samples only**, never over
  the zero padding, so a heavily padded clip is not handed a quieter noise
  floor than a full-length one.
- **A stream never crosses a split.** Validation speech is only ever mixed
  with validation noise.

Training mixtures are generated fresh every run and never stored. The fixed
validation and testing mixtures are written once by
`generate_test_mixtures.py`, which drives this same code with a fixed seed.

Run this file directly for a self-test over all three splits.
"""

import os
import csv
import sys
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

SPLITS = ["train", "validation", "testing"]

TARGET_SR = 16000
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)

SNR_MIN_DB = -5.0
SNR_MAX_DB = 15.0

MIN_SPEECH_SECONDS = 1.0        # shorter clean clips are skipped entirely
MIN_SPEECH_SAMPLES = int(TARGET_SR * MIN_SPEECH_SECONDS)

PEAK_CEILING = 0.99             # mixtures are rescaled below this if they clip
EPS = 1e-10


def load_wav(path):
    """Read a wav as mono float32. Everything in this project is already
    16 kHz mono, so a mismatch means something upstream went wrong."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        raise ValueError(f"{path} is {sr} Hz, expected {TARGET_SR}")
    return data


def read_speech_manifest(split):
    """The balanced manifest is the selection training reads -- see
    PROJECT_CONTEXT_4.md Section 9."""
    path = os.path.join(DATASET_DIR, f"balanced_{split}_manifest.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    keep, skipped = [], 0
    for row in rows:
        if float(row["duration_seconds"]) < MIN_SPEECH_SECONDS:
            skipped += 1
            continue
        keep.append(row)
    return keep, skipped


def read_noise_manifest(split):
    """Returns {category: [paths]} for one split."""
    path = os.path.join(DATASET_DIR, "noise_manifest.csv")
    by_category = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            by_category.setdefault(row["category"], []).append(
                os.path.join(DATASET_DIR, row["filename"]))
    return {k: sorted(v) for k, v in sorted(by_category.items())}


def place_in_window(speech, rng):
    """Pad a short clip out to the full window at a random offset.

    Returns (padded, start, end) so the caller can measure speech power over
    the real samples rather than the padding.
    """
    if len(speech) >= CLIP_SAMPLES:
        return speech[:CLIP_SAMPLES], 0, CLIP_SAMPLES
    padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    start = int(rng.integers(0, CLIP_SAMPLES - len(speech) + 1))
    padded[start:start + len(speech)] = speech
    return padded, start, start + len(speech)


def mix_at_snr(speech, speech_span, noise, snr_db):
    """Scale noise so the mixture sits at snr_db, then add.

    Returns (noisy, clean). Both are rescaled by the same factor if the sum
    would clip, so the pair stays consistent as an input/target pair.
    """
    start, end = speech_span
    speech_power = float(np.mean(speech[start:end] ** 2))
    noise_power = float(np.mean(noise ** 2))
    if speech_power < EPS or noise_power < EPS:
        # a silent clip on either side makes SNR meaningless; add unscaled
        scale = 1.0
    else:
        scale = np.sqrt(speech_power / (noise_power * 10.0 ** (snr_db / 10.0)))

    noisy = speech + (noise * scale).astype(np.float32)

    peak = float(np.max(np.abs(noisy)))
    if peak > PEAK_CEILING:
        shrink = PEAK_CEILING / peak
        noisy = noisy * shrink
        speech = speech * shrink

    return noisy.astype(np.float32), speech.astype(np.float32)


class SpeechNoiseMixtures(Dataset):
    """Pairs of (noisy, clean, noise_category) built on the fly.

    seed=None gives fresh randomness every run, which is what training wants.
    Passing an integer makes every index reproducible regardless of how the
    DataLoader distributes work, which is what the frozen validation and
    testing mixtures need.
    """

    def __init__(self, split, seed=None):
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        self.split = split
        self.seed = seed

        self.speech, self.skipped = read_speech_manifest(split)
        self.noise = read_noise_manifest(split)
        self.categories = sorted(self.noise)
        self.category_index = {c: i for i, c in enumerate(self.categories)}
        if not self.categories:
            raise RuntimeError(f"no noise found for split {split!r}")

        self._shared = np.random.default_rng()

    def __len__(self):
        return len(self.speech)

    def _rng_for(self, idx):
        # a seeded dataset must give the same mixture for the same index no
        # matter which worker asks for it
        return np.random.default_rng([self.seed, idx]) if self.seed is not None \
            else self._shared

    def describe(self, idx):
        """Everything about one mixture, without loading audio twice.

        Returns the arrays plus the bookkeeping the frozen-mixture manifest
        needs, so nothing has to be recomputed or guessed afterwards.
        """
        rng = self._rng_for(idx)
        row = self.speech[idx]

        speech = load_wav(os.path.join(DATASET_DIR, row["filename"]))
        speech, start, end = place_in_window(speech, rng)

        category = self.categories[int(rng.integers(len(self.categories)))]
        noise_path = self.noise[category][int(rng.integers(len(self.noise[category])))]
        noise = load_wav(noise_path)
        if len(noise) < CLIP_SAMPLES:
            noise = np.pad(noise, (0, CLIP_SAMPLES - len(noise)))
        elif len(noise) > CLIP_SAMPLES:
            offset = int(rng.integers(0, len(noise) - CLIP_SAMPLES + 1))
            noise = noise[offset:offset + CLIP_SAMPLES]

        snr_db = float(rng.uniform(SNR_MIN_DB, SNR_MAX_DB))
        noisy, clean = mix_at_snr(speech, (start, end), noise, snr_db)

        return {
            "noisy": noisy,
            "clean": clean,
            "category": category,
            "label": self.category_index[category],
            "snr_db": snr_db,
            "speech_file": row["filename"],
            "language": row["language"],
            "speaker_id": row["speaker_id"],
            "noise_file": os.path.relpath(noise_path, DATASET_DIR),
            "speech_start": start,
            "speech_end": end,
        }

    def __getitem__(self, idx):
        m = self.describe(idx)
        return m["noisy"], m["clean"], m["label"]


def self_test():
    print(f"categories are sampled uniformly, SNR ~ U({SNR_MIN_DB:g}, "
          f"{SNR_MAX_DB:g}) dB, window {CLIP_SECONDS:g}s @ {TARGET_SR} Hz\n")
    for split in SPLITS:
        ds = SpeechNoiseMixtures(split, seed=0)
        print(f"{split}: {len(ds)} speech clips usable "
              f"({ds.skipped} skipped under {MIN_SPEECH_SECONDS:g}s), "
              f"{len(ds.categories)} noise categories")

        counts, snrs, peaks = {}, [], []
        for i in range(len(ds)):
            m = ds.describe(i)
            counts[m["category"]] = counts.get(m["category"], 0) + 1
            snrs.append(m["snr_db"])
            peaks.append(float(np.max(np.abs(m["noisy"]))))
            assert len(m["noisy"]) == CLIP_SAMPLES == len(m["clean"])
        print(f"    SNR {min(snrs):+.1f} to {max(snrs):+.1f} dB "
              f"(mean {sum(snrs)/len(snrs):+.1f}), peak max {max(peaks):.3f}")
        print(f"    category draws: "
              + ", ".join(f"{c} {counts.get(c, 0)}" for c in ds.categories))

        # a seeded dataset must be byte-identical when asked twice
        again = SpeechNoiseMixtures(split, seed=0)
        assert np.array_equal(ds.describe(0)["noisy"], again.describe(0)["noisy"])
    print("\nreproducible for a fixed seed, all windows exact, no clipping.")


if __name__ == "__main__":
    self_test()
