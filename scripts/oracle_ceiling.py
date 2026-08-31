"""How good could this architecture POSSIBLY get?

Nine training runs all land at PESQ ~1.68 regardless of capacity, data volume,
mask ceiling or auxiliary loss. Two explanations remain, and they point at
opposite next moves:

  A. The model is under-trained / mis-trained. Then a better loss or a longer
     schedule lifts it, and the architecture is fine.
  B. The architecture itself cannot go higher, because it only masks the
     MAGNITUDE and reuses the noisy PHASE. Then no amount of training helps
     and the answer is complex-valued layers (DCCRN).

This script settles it without training anything. It cheats: it uses the clean
speech, which the model never sees, to build the *best possible* mask, and then
scores the result. That score is a hard ceiling -- a real model can approach it
but can never beat it.

The ladder it prints:

  noisy                 what goes in
  ideal mask (cap 1.0)  best any attenuate-only magnitude mask can do
  ideal mask (no cap)   best any magnitude mask can do, boosting allowed
                        (identical to: clean magnitude + noisy phase)
  noisy mag + clean phase   what fixing ONLY the phase would buy
  clean mag + clean phase   sanity check, should be near 4.5

Read it like this: our trained model scores 1.68. If "ideal mask (no cap)" is
far above that, the architecture has headroom and the problem is how we train.
If it sits close to 1.68, the ceiling is the design and phase is the culprit.
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

EPS = 1e-8


def stft(y):
    return model_crn.to_spectrogram(torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0))


def istft(spec, length):
    return model_crn.to_waveform(spec, length).squeeze(0).numpy()


def build_conditions(clean, noisy):
    """Every cheating reconstruction, as waveforms the same length as clean."""
    n = len(clean)
    S, Y = stft(clean), stft(noisy)
    mag_s, mag_y = S.abs(), Y.abs()
    phase_y = Y / (mag_y + EPS)
    phase_s = S / (mag_s + EPS)

    # The ideal amplitude mask: scale each bin so the noisy magnitude becomes
    # the clean magnitude. Capping it at 1.0 forbids boosting, which is exactly
    # the restriction MASK_MAX = 1.0 puts on the trained model.
    mask = mag_s / (mag_y + EPS)

    return {
        "ideal mask (cap 1.0)":      istft(mask.clamp(max=1.0) * mag_y * phase_y, n),
        "ideal mask (no cap)":       istft(mag_s * phase_y, n),
        "noisy mag + clean phase":   istft(mag_y * phase_s, n),
        "clean mag + clean phase":   istft(mag_s * phase_s, n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    ds = FrozenMixtures(args.split)
    n = len(ds) if args.max_clips is None else min(args.max_clips, len(ds))
    print(f"oracle ceiling on {n} frozen {args.split} mixtures\n")

    names = None
    scores = {}
    for i in range(n):
        d = ds.describe(i)
        clean, noisy = d["clean"], d["noisy"]
        conds = {"noisy": noisy}
        conds.update(build_conditions(clean, noisy))
        if names is None:
            names = list(conds)
            scores = {k: {"pesq": [], "stoi": [], "sisnr": []} for k in names}
        for k, y in conds.items():
            scores[k]["pesq"].append(safe_pesq(clean, y))
            scores[k]["stoi"].append(safe_stoi(clean, y))
            scores[k]["sisnr"].append(si_snr(y, clean))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}")

    print(f"\n{'condition':<26}{'PESQ':>8}{'STOI':>8}{'SI-SNR':>9}")
    print("-" * 51)
    for k in names:
        s = scores[k]
        print(f"{k:<26}{_mean(s['pesq']):>8.3f}{_mean(s['stoi']):>8.3f}"
              f"{_mean(s['sisnr']):>9.2f}")
    print("-" * 51)
    print(f"{'our trained model':<26}{1.68:>8.3f}{0.842:>8.3f}{11.6:>9.2f}"
          f"   (run 0009, converged mean)")
    print(f"{'problem-statement target':<26}{2.50:>8.3f}{0.850:>8.3f}{15.0:>9.2f}")


if __name__ == "__main__":
    main()
