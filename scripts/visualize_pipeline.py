"""
visualize_pipeline.py

Draws what the audio pipeline is actually doing, on real project data.

Takes one mixture out of the frozen evaluation set and shows every stage as a
picture: the clean speech we want back, the noise that buried it, the mixture
the model receives, the spectrogram the network actually sees, and what comes
back out after the spectrogram is converted to audio again.

The bottom panel is the point of the whole figure. It is the difference
between the audio going in and the audio coming out of the round trip, and it
should be a flat line at zero. That turns the abstract "round-trip error
9.5e-07" from the step 1 self-test into something you can look at.

This grows as the model does. Once there is a trained network, the same
script will show the predicted mask and the enhanced output alongside these,
so the model can be watched working rather than only scored.

    python scripts/visualize_pipeline.py                     # pick a good one
    python scripts/visualize_pipeline.py --split testing --index 39
    python scripts/visualize_pipeline.py --category mortar

Writes a PNG under logs/figures/ and prints where it went.
"""

import os
import csv
import sys
import argparse

import numpy as np
import soundfile as sf
import torch

import matplotlib
matplotlib.use("Agg")           # no display on this machine; render to file
import matplotlib.pyplot as plt

from model_crn import (SAMPLE_RATE, N_FFT, HOP_LENGTH,
                       to_spectrogram, to_waveform)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MIX_DIR = os.path.join(PROJECT_ROOT, "generated_test_mixtures")
OUT_DIR = os.path.join(PROJECT_ROOT, "logs", "figures")

# a low-SNR impulsive example: the noise is louder than the speech, and the
# bursts are visually unmistakable against the speech
DEFAULT_SPLIT = "testing"
DEFAULT_INDEX = 39

WAVE_COLOR = "#1f77b4"
NOISE_COLOR = "#d62728"
MIX_COLOR = "#7f4fbf"
ERR_COLOR = "#2ca02c"


def load_manifest():
    path = os.path.join(MIX_DIR, "test_mixtures_manifest.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pick(rows, split, index, category):
    if category:
        hits = [r for r in rows if r["category"] == category]
        if not hits:
            raise SystemExit(f"no mixture uses category {category!r}")
        # the loudest noise relative to speech is the clearest to look at
        return min(hits, key=lambda r: float(r["snr_db"]))
    for r in rows:
        if r["split"] == split and int(r["index"]) == index:
            return r
    raise SystemExit(f"no mixture {split} #{index}")


def read(path):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SAMPLE_RATE, f"{path} is {sr} Hz"
    return data


def draw(row):
    noisy = read(os.path.join(MIX_DIR, row["noisy_file"]))
    clean = read(os.path.join(MIX_DIR, row["clean_file"]))
    noise_src = read(os.path.join(DATASET_DIR, row["noise_file"]))

    # what the mixer actually added: whatever is left when the clean speech is
    # taken back out. Reading it from the pair rather than re-deriving it means
    # the picture cannot disagree with the audio on disk.
    noise_in_mix = noisy - clean

    wav = torch.from_numpy(noisy).unsqueeze(0)
    spec = to_spectrogram(wav)
    back = to_waveform(spec, wav.shape[-1])
    recon = back.squeeze(0).numpy()
    error = recon - noisy

    mag = spec.squeeze(0).abs().numpy().T                  # (freq, time)
    db = 20.0 * np.log10(np.maximum(mag, 1e-8))
    db = np.maximum(db, db.max() - 80.0)                   # 80 dB of range

    t = np.arange(len(noisy)) / SAMPLE_RATE
    dur = len(noisy) / SAMPLE_RATE
    peak = max(np.abs(noisy).max(), np.abs(clean).max()) * 1.1

    fig, axes = plt.subplots(6, 1, figsize=(13, 15),
                             gridspec_kw={"height_ratios": [1, 1, 1, 2.2, 1, 1]})

    fig.suptitle(
        f"Audio pipeline on a real mixture  —  {row['split']} #{row['index']}   "
        f"{row['category']}   SNR {float(row['snr_db']):+.2f} dB   "
        f"speech: {row['language']}",
        fontsize=13, y=0.995)

    def wave(ax, y, color, title, ylim=None):
        ax.plot(t, y, color=color, linewidth=0.4)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(0, dur)
        ax.set_ylim(-(ylim or peak), (ylim or peak))
        ax.set_ylabel("amp", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.15)

    wave(axes[0], clean, WAVE_COLOR,
         "1. CLEAN SPEECH — what we eventually want the model to recover")
    wave(axes[1], noise_in_mix, NOISE_COLOR,
         f"2. THE NOISE that was added ({row['category']}, "
         f"from {os.path.basename(row['noise_file'])})")
    wave(axes[2], noisy, MIX_COLOR,
         "3. THE MIXTURE — this, and only this, is what the model receives")

    axes[3].imshow(db, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, dur, 0, SAMPLE_RATE / 2000])
    axes[3].set_title(
        f"4. THE SPECTROGRAM of the mixture — what the network actually sees "
        f"({db.shape[1]} frames x {db.shape[0]} freq bins)",
        fontsize=10, loc="left")
    axes[3].set_ylabel("kHz", fontsize=8)
    axes[3].tick_params(labelsize=8)

    wave(axes[4], recon, MIX_COLOR,
         "5. BACK TO AUDIO — the spectrogram converted straight back, no model yet")

    axes[5].plot(t, error, color=ERR_COLOR, linewidth=0.4)
    axes[5].set_title(
        f"6. THE ERROR (5 minus 3) — flat means nothing was lost. "
        f"Worst: {np.abs(error).max():.2e}",
        fontsize=10, loc="left")
    axes[5].set_xlim(0, dur)
    axes[5].set_ylim(-peak, peak)
    axes[5].set_xlabel("seconds", fontsize=9)
    axes[5].set_ylabel("amp", fontsize=8)
    axes[5].tick_params(labelsize=8)
    axes[5].grid(alpha=0.15)

    fig.tight_layout(rect=[0, 0, 1, 0.985])

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR,
                       f"pipeline_{row['split']}_{int(row['index']):03d}_"
                       f"{row['category']}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)

    return out, error, noisy, clean, noise_in_mix


def draw_zoom(row, t0, t1, fmax=4.0):
    """Clean, noise and mixture spectrograms side by side over a short window.

    The full three-second view squeezes too much in to see the shapes. Up
    close the contrast is the whole argument for using spectrograms at all:
    voice draws horizontal lines, an impact draws a vertical one.

    All three panels share one brightness scale, taken from the mixture.
    Auto-scaling each panel separately would make the quiet clean speech look
    as loud as the mortar and destroy the comparison.
    """
    noisy = read(os.path.join(MIX_DIR, row["noisy_file"]))
    clean = read(os.path.join(MIX_DIR, row["clean_file"]))
    noise = noisy - clean

    def spec_db(y):
        s = to_spectrogram(torch.from_numpy(y).unsqueeze(0))
        return 20.0 * np.log10(np.maximum(s.squeeze(0).abs().numpy().T, 1e-8))

    dbs = [spec_db(x) for x in (clean, noise, noisy)]
    top = dbs[2].max()                       # one shared scale, from the mixture
    lo, hi = top - 70.0, top

    n_freq, n_frames = dbs[0].shape
    f0 = int(fmax * 1000 / (SAMPLE_RATE / 2) * n_freq)
    a = int(t0 * SAMPLE_RATE / HOP_LENGTH)
    b = int(t1 * SAMPLE_RATE / HOP_LENGTH)

    titles = [
        "CLEAN SPEECH alone — held notes draw HORIZONTAL lines",
        f"THE {row['category'].upper()} alone — an impact hits every pitch at "
        f"once, drawing a VERTICAL line",
        "THE MIXTURE — both at once. This is all the model gets.",
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(
        f"Zoomed to {t0:.2f}–{t1:.2f} s, up to {fmax:.0f} kHz  —  "
        f"{row['split']} #{row['index']}, {row['category']}, "
        f"SNR {float(row['snr_db']):+.2f} dB",
        fontsize=13, y=0.985)

    for ax, db, title in zip(axes, dbs, titles):
        ax.imshow(db[:f0, a:b], origin="lower", aspect="auto", cmap="magma",
                  vmin=lo, vmax=hi, extent=[t0, t1, 0, fmax])
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("kHz", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("seconds", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR,
                       f"zoom_{row['split']}_{int(row['index']):03d}_"
                       f"{row['category']}.png")
    fig.savefig(out, dpi=115)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--index", type=int, default=DEFAULT_INDEX)
    ap.add_argument("--category", default=None,
                    help="pick the lowest-SNR mixture of this noise category")
    ap.add_argument("--zoom", nargs=2, type=float, metavar=("START", "END"),
                    help="instead of the full pipeline, draw clean / noise / "
                         "mixture spectrograms over this time window")
    ap.add_argument("--fmax", type=float, default=4.0,
                    help="top of the frequency axis in kHz for --zoom")
    args = ap.parse_args()

    row = pick(load_manifest(), args.split, args.index, args.category)

    if args.zoom:
        out = draw_zoom(row, args.zoom[0], args.zoom[1], args.fmax)
        print(f"mixture     {row['split']} #{row['index']}  {row['category']}  "
              f"SNR {float(row['snr_db']):+.2f} dB")
        print(f"window      {args.zoom[0]:.2f}-{args.zoom[1]:.2f} s, "
              f"0-{args.fmax:.0f} kHz")
        print(f"\nwrote {os.path.relpath(out, PROJECT_ROOT)}")
        return

    out, error, noisy, clean, noise = draw(row)

    print(f"mixture     {row['split']} #{row['index']}  {row['category']}  "
          f"SNR {float(row['snr_db']):+.2f} dB  speech in {row['language']}")
    print(f"speech      {row['speech_file']}")
    print(f"noise       {row['noise_file']}")
    print(f"levels      clean rms {np.sqrt((clean**2).mean()):.4f}, "
          f"noise rms {np.sqrt((noise**2).mean()):.4f}  "
          f"(noise is {np.sqrt((noise**2).mean())/max(np.sqrt((clean**2).mean()),1e-9):.1f}x "
          f"the speech)")
    print(f"round trip  worst error {np.abs(error).max():.2e}")
    print(f"\nwrote {os.path.relpath(out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
