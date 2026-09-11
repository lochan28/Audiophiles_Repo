"""
erb_ceiling.py -- what does hybrid frequency resolution cost, before training?

The GTCRN idea: keep the lowest bins at full resolution and merge everything
above into ERB bands, so the model spends its capacity where speech has fine
harmonic structure and treats the aperiodic top band coarsely. It roughly
halves the LSTM input, which is the Raspberry Pi's bottleneck.

The question this answers without training anything: if the mask above the
cutoff can only be ONE gain per ERB band, how much of the ideal-mask ceiling
survives? The full-resolution ideal mask (cap 1.0) sets the ceiling every
run in this project is measured against. Project that same ideal mask
through bins -> bands -> bins and score the result. That is the best any
band-constrained model could do.

Read it like this: run 0026 reaches PESQ ~2.11. If the constrained ceiling
is still far above that, the resolution given up above the cutoff was never
going to be used and the experiment is worth an afternoon. If the ceiling
falls toward 2.1, the idea is capped before it starts.

The band gain used here is the energy-weighted average of the ideal mask
across the band -- the least-squares gain for the bins it covers -- so this
is a slightly pessimistic ceiling: a trained model with a learned inverse
could do marginally better, never worse.
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_crn
from mixing import FrozenMixtures
from evaluate import si_snr, safe_pesq, safe_stoi, _mean

# ----------------------------------------------------------- TUNING KNOBS
#
# CONFIGS       (full-resolution low bins, ERB bands for the rest). The first
#               is GTCRN's own choice at 16 kHz / 512 FFT: bins 0-64 are
#               0-2 kHz kept exactly, the remaining 192 bins (2-8 kHz)
#               become 64 bands. The others bracket it: fewer full bins and
#               fewer bands is more aggressive, more of each is safer.
CONFIGS = [(65, 64), (65, 32), (33, 32), (97, 32), (129, 16)]

EPS = 1e-8
N_BINS = model_crn.N_FFT // 2 + 1        # 257
SR = model_crn.SAMPLE_RATE


def hz_to_erb(f):
    return 21.4 * np.log10(1.0 + 0.00437 * f)


def erb_to_hz(e):
    return (10.0 ** (e / 21.4) - 1.0) / 0.00437


def erb_matrix(low_bins, n_bands):
    """(n_bands, N_BINS - low_bins) triangular filters on the ERB scale.

    Rows are analysis weights and sum to one, so a row applied to a mask
    gives a weighted average gain for that band. Same construction GTCRN
    uses; centres are spaced evenly in ERB-rate between the cutoff and
    Nyquist.
    """
    f_lo = low_bins * SR / model_crn.N_FFT
    f_hi = SR / 2
    e = np.linspace(hz_to_erb(f_lo), hz_to_erb(f_hi), n_bands + 2)
    centres = erb_to_hz(e)                                    # n_bands + 2
    freqs = np.arange(low_bins, N_BINS) * SR / model_crn.N_FFT
    W = np.zeros((n_bands, len(freqs)), dtype=np.float32)
    for b in range(n_bands):
        lo, c, hi = centres[b], centres[b + 1], centres[b + 2]
        up = (freqs - lo) / max(c - lo, 1e-6)
        down = (hi - freqs) / max(hi - c, 1e-6)
        W[b] = np.clip(np.minimum(up, down), 0.0, None)
    # Every bin must belong to something, or the top and bottom bins fall
    # between triangles and get a gain of zero.
    empty = W.sum(axis=0) < 1e-6
    for j in np.where(empty)[0]:
        W[np.argmin(np.abs(centres[1:-1] - freqs[j])), j] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def constrain(mask, mag_y, low_bins, n_bands, W):
    """Ideal per-bin mask -> one gain per band above the cutoff -> per-bin."""
    hi_mask = mask[:, low_bins:]                              # (T, F_hi)
    hi_pow = mag_y[:, low_bins:] ** 2
    # Energy-weighted mean of the mask within each band: the least-squares
    # single gain for those bins. Weighted by W (band membership) and by the
    # noisy energy (louder bins matter more to the error).
    num = (hi_mask * hi_pow) @ W.T                            # (T, B)
    den = hi_pow @ W.T + EPS
    band_gain = num / den
    # Synthesis: each bin takes the membership-weighted mix of its bands.
    Ws = W / (W.sum(axis=0, keepdims=True) + EPS)             # columns sum 1
    hi_out = band_gain @ Ws                                   # (T, F_hi)
    return np.concatenate([mask[:, :low_bins], hi_out], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testing")
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    ds = FrozenMixtures(args.split)
    n = len(ds) if args.max_clips is None else min(args.max_clips, len(ds))
    mats = {cfg: erb_matrix(*cfg) for cfg in CONFIGS}
    names = ["noisy", "ideal mask (cap 1.0)"] + \
            [f"ERB {lo} bins + {b} bands" for lo, b in CONFIGS]
    scores = {k: {"pesq": [], "stoi": [], "sisnr": []} for k in names}

    print(f"ERB-constrained ideal-mask ceiling on {n} frozen {args.split} "
          f"mixtures\n")
    for i in range(n):
        d = ds.describe(i)
        clean, noisy = d["clean"], d["noisy"]
        L = len(clean)
        S = model_crn.to_spectrogram(torch.from_numpy(clean).unsqueeze(0))
        Y = model_crn.to_spectrogram(torch.from_numpy(noisy).unsqueeze(0))
        mag_s, mag_y = S.abs()[0].numpy(), Y.abs()[0].numpy()
        phase_y = (Y / (Y.abs() + EPS))[0]
        ideal = np.clip(mag_s / (mag_y + EPS), 0.0, 1.0)

        outs = {"noisy": noisy}

        def render(m):
            spec = torch.from_numpy((m * mag_y).astype(np.float32)) * phase_y
            return model_crn.to_waveform(spec.unsqueeze(0), L)[0].numpy()

        outs["ideal mask (cap 1.0)"] = render(ideal)
        for cfg in CONFIGS:
            lo, b = cfg
            outs[f"ERB {lo} bins + {b} bands"] = render(
                constrain(ideal, mag_y, lo, b, mats[cfg]))

        for k, y in outs.items():
            scores[k]["pesq"].append(safe_pesq(clean, y))
            scores[k]["stoi"].append(safe_stoi(clean, y))
            scores[k]["sisnr"].append(si_snr(y, clean))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n}")

    print(f"\n{'condition':<28}{'PESQ':>8}{'STOI':>8}{'SI-SNR':>9}"
          f"{'features':>10}")
    print("-" * 63)
    for k in names:
        s = scores[k]
        feat = ""
        for lo, b in CONFIGS:
            if k == f"ERB {lo} bins + {b} bands":
                feat = f"{lo + b:>6} / 257"
        if k == "ideal mask (cap 1.0)":
            feat = f"{257:>6} / 257"
        print(f"{k:<28}{_mean(s['pesq']):>8.3f}{_mean(s['stoi']):>8.3f}"
              f"{_mean(s['sisnr']):>9.2f}{feat:>10}")
    print("-" * 63)
    print(f"{'run 0026 + post-filter':<28}{1.983:>8.3f}{0.866:>8.3f}"
          f"{14.02:>9.2f}   (held-out, what we reach)")


if __name__ == "__main__":
    main()
