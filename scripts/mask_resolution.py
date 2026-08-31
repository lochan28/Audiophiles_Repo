"""How much detail does the mask actually need?

oracle_ceiling.py showed the architecture could in principle reach PESQ 3.31
while run 0010 sits at 1.78. Something stops the model producing the mask it
would need. One candidate is resolution: the encoder halves the frequency axis
five times, 257 bands down to 9, before the LSTM ever sees it.

That "9 bands" figure overstates the bottleneck on its own -- the decoder
upsamples back and the skip connections carry full-resolution detail past the
narrow point, so the model is not literally restricted to 9 bands. What is
genuinely unknown is how much fine detail the ideal mask *contains*, and
therefore how much a smooth approximation of it would cost.

So this measures the mask rather than the model. It takes the ideal mask, blurs
it deliberately -- averaging over groups of frequency bands, or groups of time
frames -- and scores the result. Read it as: if the model could only ever
produce a mask this smooth, this is the best it could possibly do.

If blurring across 32 bands barely dents the score, resolution is not the
problem and the next suspect is elsewhere. If it collapses, the encoder's
squeeze is worth redesigning.
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_crn
from mixing import FrozenMixtures
from evaluate import si_snr, safe_pesq, safe_stoi, _mean

EPS = 1e-8

# Frequency groups to average the mask over. 257 bands in, so 32 is roughly
# the encoder's five halvings and 1 is the untouched ideal mask.
FREQ_FACTORS = (1, 2, 4, 8, 16, 32, 64)

# Time frames to average over. The hop is 10 ms, so 8 frames is 80 ms --
# longer than most consonants, which is the point of testing it.
TIME_FACTORS = (1, 2, 4, 8, 16)


def coarsen(mask, freq_factor=1, time_factor=1):
    """Blur a mask (1, T, F) by averaging over groups, then stretch it back.

    Averaging and re-stretching is what a model with limited resolution is
    effectively restricted to: it can place broad strokes but not fine ones.
    Bilinear interpolation on the way back, rather than nearest, because a
    real decoder produces smooth output rather than blocky steps.
    """
    if freq_factor == 1 and time_factor == 1:
        return mask
    x = mask.permute(0, 2, 1).unsqueeze(0)          # (1, 1, F, T)
    _, _, n_freq, n_time = x.shape
    # int(): the factors may be fractional when something is searching for a
    # blur level between two whole steps.
    small = F.adaptive_avg_pool2d(
        x, (max(1, int(n_freq // freq_factor)),
            max(1, int(n_time // time_factor))))
    back = F.interpolate(small, size=(n_freq, n_time), mode="bilinear",
                         align_corners=False)
    return back.squeeze(0).permute(0, 2, 1)


def score_split(split, max_clips=None):
    ds = FrozenMixtures(split)
    n = len(ds) if max_clips is None else min(max_clips, len(ds))

    conditions = ([(f, 1) for f in FREQ_FACTORS]
                  + [(1, t) for t in TIME_FACTORS if t != 1])
    scores = {c: {"pesq": [], "stoi": [], "sisnr": []} for c in conditions}

    for i in range(n):
        d = ds.describe(i)
        clean, noisy = d["clean"], d["noisy"]
        S = model_crn.to_spectrogram(torch.from_numpy(clean).unsqueeze(0))
        Y = model_crn.to_spectrogram(torch.from_numpy(noisy).unsqueeze(0))
        mag_y = Y.abs()
        phase_y = Y / (mag_y + EPS)
        ideal = (S.abs() / (mag_y + EPS)).clamp(max=1.0)

        for cond in conditions:
            m = coarsen(ideal, *cond)
            est = model_crn.to_waveform(m * mag_y * phase_y,
                                        len(clean)).squeeze(0).numpy()
            scores[cond]["pesq"].append(safe_pesq(clean, est))
            scores[cond]["stoi"].append(safe_stoi(clean, est))
            scores[cond]["sisnr"].append(si_snr(est, clean))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}")
    return scores, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    scores, n = score_split(args.split, args.max_clips)
    print(f"\nideal mask, deliberately blurred -- {n} frozen "
          f"{args.split} mixtures\n")

    print(f"{'blur':<28}{'PESQ':>8}{'STOI':>8}{'SI-SNR':>9}")
    print("-" * 53)
    for f in FREQ_FACTORS:
        s = scores[(f, 1)]
        label = "none (the ideal mask)" if f == 1 else f"{f} frequency bands"
        print(f"{label:<28}{_mean(s['pesq']):>8.3f}{_mean(s['stoi']):>8.3f}"
              f"{_mean(s['sisnr']):>9.2f}")
    print("-" * 53)
    for t in TIME_FACTORS:
        if t == 1:
            continue
        s = scores[(1, t)]
        print(f"{f'{t} time frames ({t * 10} ms)':<28}"
              f"{_mean(s['pesq']):>8.3f}{_mean(s['stoi']):>8.3f}"
              f"{_mean(s['sisnr']):>9.2f}")
    print("-" * 53)
    print(f"{'run 0010, trained':<28}{1.784:>8.3f}{0.845:>8.3f}{11.59:>9.2f}")
    print(f"{'target':<28}{2.500:>8.3f}{0.850:>8.3f}{15.00:>9.2f}")


if __name__ == "__main__":
    main()
