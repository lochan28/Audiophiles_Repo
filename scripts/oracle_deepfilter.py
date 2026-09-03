"""Would a multi-frame filter raise our ceiling above a per-bin mask?

`oracle_ceiling.py` measured what the BEST POSSIBLE per-bin magnitude mask
scores: one real gain per time-frequency point, noisy phase kept. That is the
ceiling of the operator our CRN actually produces.

DeepFilterNet uses a richer operator -- "deep filtering" -- which predicts a
few COMPLEX coefficients per bin and combines several neighbouring frames in
time rather than scaling one value. Speech is correlated across adjacent
frames and impulsive noise is not, so in principle a multi-frame filter can
use information a per-bin mask structurally cannot. This asks whether that
raises the ceiling on OUR data, before anyone spends a week building one.

THE DESIGN DECISION THAT MAKES THIS MEANINGFUL
----------------------------------------------
An unconstrained oracle here is degenerate. Given free complex coefficients at
every (frame, bin) you can always choose them to reproduce the clean spectrum
exactly -- one tap set to S/Y does it -- so the "ceiling" would be perfect
reconstruction and the measurement would say nothing.

The constraint that makes it informative is STATIONARITY: the filter is solved
by least squares and held CONSTANT over a block of frames. A real network
cannot know the per-frame ground truth either; it must infer a filter from
local statistics. Block length is therefore the knob that trades expressiveness
against realism:

    block = whole clip   one filter per bin for 3 seconds. Very constrained,
                         a genuine lower bound on multi-frame filtering.
    block = 20 frames    200 ms. Roughly the timescale a network could track.
    block = 5 frames     50 ms. Approaching the degenerate case; included to
                         show the trend, NOT as an achievable number.

Read the table by comparing against the ideal mask on the same audio. If the
multi-frame rows sit below it at realistic block lengths, the extra machinery
buys nothing here and DeepFilterNet's architecture is not our missing piece.
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

# Filter taps per bin. DeepFilterNet uses about 5.
ORDER = 5

# How many of those taps look into the future. Our model already spends 6
# frames of lookahead, so 2 here costs nothing we are not already paying.
LOOKAHEAD = 2

# Frames the filter is held constant over. None means the whole clip.
#
# BLOCK 10 IS THE FAIR COMPARISON against the ideal per-bin mask, because it is
# where the two operators have the SAME NUMBER OF FREE PARAMETERS. The mask
# gets one real gain per frame per bin, so 10 reals per 10 frames. A filter of
# order 5 is 5 complex taps, also 10 reals, and at block 10 it spends them over
# the same 10 frames. Longer blocks give the filter FEWER degrees of freedom
# than the mask and shorter ones give it more, so only this row isolates the
# operator rather than the parameter budget.
BLOCKS = (None, 40, 20, 10, 5)

# Ridge term on the normal equations. Without it a bin that is near-silent for
# a whole block gives a singular system and the solve blows up.
RIDGE = 1e-6


def multiframe_oracle(noisy, clean, order=ORDER, lookahead=LOOKAHEAD,
                      block=None):
    """Best least-squares multi-frame filter, held constant per block.

    Returns the filtered waveform. For each frequency bin the taps solve
        min over c of  || sum_n c[n] * Y[t - n + lookahead, f]  -  S[t, f] ||
    over the frames of one block, which is a small complex normal-equation
    system per bin per block.
    """
    Y = model_crn.to_spectrogram(
        torch.from_numpy(np.ascontiguousarray(noisy)).unsqueeze(0)).squeeze(0)
    S = model_crn.to_spectrogram(
        torch.from_numpy(np.ascontiguousarray(clean)).unsqueeze(0)).squeeze(0)
    n_frames, n_bins = Y.shape

    # X[t, f, n] = Y[t - n + lookahead, f], zero outside the clip
    X = torch.zeros(n_frames, n_bins, order, dtype=Y.dtype)
    for n in range(order):
        shift = lookahead - n
        if shift > 0:                       # reads the future
            X[:n_frames - shift, :, n] = Y[shift:, :]
        elif shift < 0:                     # reads the past
            X[-shift:, :, n] = Y[:n_frames + shift, :]
        else:
            X[:, :, n] = Y

    out = torch.zeros_like(Y)
    size = n_frames if block is None else block
    for start in range(0, n_frames, size):
        stop = min(start + size, n_frames)
        Xb = X[start:stop]                              # (T, F, N)
        Sb = S[start:stop]                              # (T, F)
        Xh = Xb.conj().permute(1, 2, 0)                 # (F, N, T)
        A = Xh @ Xb.permute(1, 0, 2)                    # (F, N, N)
        b = (Xh @ Sb.permute(1, 0).unsqueeze(-1))       # (F, N, 1)
        eye = torch.eye(A.shape[-1], dtype=A.dtype).expand_as(A)
        scale = A.diagonal(dim1=-2, dim2=-1).abs().mean(-1)
        A = A + eye * (RIDGE * scale.clamp(min=EPS)).view(-1, 1, 1)
        c = torch.linalg.solve(A, b)                    # (F, N, 1)
        out[start:stop] = (Xb.permute(1, 0, 2) @ c).squeeze(-1).permute(1, 0)

    return model_crn.to_waveform(out.unsqueeze(0), len(noisy)).squeeze(0).numpy()


def ideal_mask(noisy, clean):
    """The per-bin magnitude-mask ceiling, for comparison on the same audio."""
    Y = model_crn.to_spectrogram(
        torch.from_numpy(np.ascontiguousarray(noisy)).unsqueeze(0))
    S = model_crn.to_spectrogram(
        torch.from_numpy(np.ascontiguousarray(clean)).unsqueeze(0))
    mag_y = Y.abs()
    mask = (S.abs() / (mag_y + EPS)).clamp(max=1.0)
    return model_crn.to_waveform(mask * Y, len(noisy)).squeeze(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testing")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--order", type=int, default=ORDER)
    args = ap.parse_args()

    ds = FrozenMixtures(args.split)
    n = len(ds) if args.max_clips is None else min(args.max_clips, len(ds))
    print(f"multi-frame filter oracle, order {args.order}, "
          f"{LOOKAHEAD} frames of lookahead")
    print(f"{n} frozen {args.split} mixtures\n")

    names = ["noisy", "ideal per-bin mask"] + [
        f"multi-frame, {'whole clip' if b is None else str(b) + ' frames'}"
        for b in BLOCKS]
    scores = {k: {"pesq": [], "stoi": [], "sisnr": []} for k in names}

    for i in range(n):
        d = ds.describe(i)
        clean, noisy = d["clean"], d["noisy"]
        outs = {"noisy": noisy, "ideal per-bin mask": ideal_mask(noisy, clean)}
        for b in BLOCKS:
            key = f"multi-frame, {'whole clip' if b is None else str(b) + ' frames'}"
            outs[key] = multiframe_oracle(noisy, clean, order=args.order,
                                          block=b)
        for k, y in outs.items():
            scores[k]["pesq"].append(safe_pesq(clean, y))
            scores[k]["stoi"].append(safe_stoi(clean, y))
            scores[k]["sisnr"].append(si_snr(y, clean))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}")

    print(f"\n{'condition':<34}{'PESQ':>8}{'STOI':>9}{'SI-SNR':>9}")
    print("-" * 60)
    for k in names:
        s = scores[k]
        print(f"{k:<34}{_mean(s['pesq']):>8.3f}{_mean(s['stoi']):>9.4f}"
              f"{_mean(s['sisnr']):>9.2f}")
    print("-" * 60)
    print(f"{'our model (run 0026)':<34}{1.944:>8.3f}{0.8625:>9.4f}{13.66:>9.2f}")
    print(f"{'DeepFilterNet v0.5.6':<34}{1.947:>8.3f}{0.8508:>9.4f}{11.67:>9.2f}")
    print(f"{'target':<34}{2.500:>8.3f}{0.8500:>9.4f}{15.00:>9.2f}")
    print("\nShorter blocks approach exact reconstruction and are NOT")
    print("achievable numbers; they show the trend, not a target.")


if __name__ == "__main__":
    main()
