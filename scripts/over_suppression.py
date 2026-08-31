"""Why does the model cut away more than it should, and does it cost anything?

`mask_diagnosis.py` found a bias rather than noise: the trained mask averages
0.358 where the ideal mask averages 0.426. The model systematically
over-suppresses. A bias usually has a nameable cause and a cheap fix, which is
worth establishing before spending another architecture experiment on it.

Three questions, in order:

1. WHERE is the bias? A mask average taken over every bin is dominated by the
   near-silent ones, so it can look like over-suppression when the model is
   really just confident about silence. Splitting the error by how much the
   ideal mask wanted to keep, and weighting by energy rather than by bin count,
   separates those two stories.

2. Is it uniform, or does it depend on conditions? A bias that grows as SNR
   falls is a model hedging when the evidence gets thin. A flat one points at
   the objective instead.

3. DOES IT COST ANYTHING? The decisive test. The model's mask is corrected
   after the fact -- multiplied by a constant, or passed through a power that
   lifts middling values -- and rescored. If a one-parameter correction the
   model could trivially have learned recovers real PESQ, the bias is a genuine
   loss and worth fixing at source. If it recovers nothing, the bias is a
   symptom of the mask being wrong in the places that matter and correcting the
   average is beside the point.
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_crn
from mixing import FrozenMixtures
from evaluate import si_snr, safe_pesq, safe_stoi, _mean
from mask_diagnosis import load_model

EPS = 1e-8

# Constant multipliers applied to the mask, then clipped back to the model's
# own ceiling. 1.0 is the model untouched.
SCALES = (1.0, 1.1, 1.2, 1.3, 1.45, 1.6)

# Powers applied to the mask. Below 1 they lift middling values toward 1 while
# leaving 0 and 1 alone, which is what "stop over-suppressing" looks like if
# the bias is worst in the uncertain middle rather than uniform.
POWERS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)

# Contrast expansion. The decile table shows the model does not under-predict
# everywhere -- it over-keeps where the ideal wanted 0 and under-keeps where the
# ideal wanted 1. That is shrinkage toward the middle, and undoing it needs the
# two halves pushed in OPPOSITE directions, which neither a multiplier nor a
# power can do. This curve leaves 0, 0.5 and 1 fixed and pushes everything else
# outward as k rises.
CONTRASTS = (1.0, 1.3, 1.6, 2.0, 2.5, 3.0)


def expand(mask, k):
    """Push a shrunk mask back out toward 0 and 1, sparing the endpoints."""
    if k == 1.0:
        return mask
    m = mask.clamp(EPS, 1 - EPS)
    up = m ** k
    return up / (up + (1 - m) ** k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-clips", type=int, default=50)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(args.run, device)
    mask_max = model.decoder.mask_max
    ds = FrozenMixtures(args.split)
    n = min(args.max_clips, len(ds))
    print(f"{args.run}, epoch {epoch}, {n} clips, mask ceiling {mask_max}\n")

    # question 1: where the bias lives
    bins = np.linspace(0, 1, 11)
    by_decile = defaultdict(list)
    kept_model, kept_ideal = [], []
    by_snr = defaultdict(list)

    # question 3: does correcting it recover anything
    scale_scores = {s: {"pesq": [], "stoi": [], "sisnr": []} for s in SCALES}
    power_scores = {p: {"pesq": [], "stoi": [], "sisnr": []} for p in POWERS}
    contrast_scores = {k: {"pesq": [], "stoi": [], "sisnr": []}
                       for k in CONTRASTS}

    for i in range(n):
        d = ds.describe(i)
        clean, noisy = d["clean"], d["noisy"]
        x = torch.from_numpy(noisy).unsqueeze(0).to(device)
        with torch.no_grad():
            _, mask, _, _ = model(x)
        mask = mask.cpu()

        S = model_crn.to_spectrogram(torch.from_numpy(clean).unsqueeze(0))
        Y = model_crn.to_spectrogram(torch.from_numpy(noisy).unsqueeze(0))
        mag_y = Y.abs()
        ideal = (S.abs() / (mag_y + EPS)).clamp(max=1.0)

        m = mask.numpy().ravel()
        idl = ideal.numpy().ravel()
        which = np.clip(np.digitize(idl, bins) - 1, 0, 9)
        for k in range(10):
            sel = which == k
            if sel.any():
                by_decile[k].append((float(idl[sel].mean()),
                                     float(m[sel].mean()), int(sel.sum())))

        # energy-weighted: what FRACTION of the incoming energy is kept
        w = (mag_y ** 2).numpy().ravel()
        kept_model.append(float((m * w).sum() / (w.sum() + EPS)))
        kept_ideal.append(float((idl * w).sum() / (w.sum() + EPS)))
        by_snr[round(d["snr_db"] / 5) * 5].append(
            float(m.mean() - idl.mean()))

        phase_y = Y / (mag_y + EPS)

        def score(mod, store):
            y = model_crn.to_waveform(mod * mag_y * phase_y,
                                      len(clean)).squeeze(0).numpy()
            store["pesq"].append(safe_pesq(clean, y))
            store["stoi"].append(safe_stoi(clean, y))
            store["sisnr"].append(si_snr(y, clean))

        for s in SCALES:
            score((mask * s).clamp(max=mask_max), scale_scores[s])
        for p in POWERS:
            score(mask.clamp(min=0) ** p, power_scores[p])
        for k in CONTRASTS:
            score(expand(mask, k).clamp(max=mask_max), contrast_scores[k])

    print("1. where the bias lives -- model's mask vs what the ideal wanted")
    print(f"{'ideal mask in range':<22}{'ideal':>8}{'model':>8}"
          f"{'gap':>8}{'share of bins':>15}")
    total = sum(sum(c for _, _, c in v) for v in by_decile.values())
    for k in range(10):
        if not by_decile[k]:
            continue
        rows = by_decile[k]
        cnt = sum(c for _, _, c in rows)
        iv = np.average([a for a, _, _ in rows],
                        weights=[c for _, _, c in rows])
        mv = np.average([b for _, b, _ in rows],
                        weights=[c for _, _, c in rows])
        print(f"{f'{bins[k]:.1f} - {bins[k + 1]:.1f}':<22}{iv:>8.3f}{mv:>8.3f}"
              f"{mv - iv:>+8.3f}{cnt / total:>14.1%}")

    print(f"\n   energy actually kept, model {np.mean(kept_model):.3f}   "
          f"ideal {np.mean(kept_ideal):.3f}   "
          f"gap {np.mean(kept_model) - np.mean(kept_ideal):+.3f}")

    print("\n2. does the bias depend on how hard the clip is")
    print(f"{'SNR':>8}{'mean gap':>12}{'clips':>8}")
    for snr in sorted(by_snr):
        print(f"{snr:>+8}{np.mean(by_snr[snr]):>+12.3f}{len(by_snr[snr]):>8}")

    print("\n3. correcting it after the fact -- does it recover anything")
    print(f"{'multiply mask by':<20}{'PESQ':>8}{'STOI':>8}{'SI-SNR':>9}")
    for s in SCALES:
        v = scale_scores[s]
        print(f"{s:<20.2f}{_mean(v['pesq']):>8.3f}{_mean(v['stoi']):>8.3f}"
              f"{_mean(v['sisnr']):>9.2f}")
    print(f"{'raise mask to power':<20}{'PESQ':>8}{'STOI':>8}{'SI-SNR':>9}")
    for p in POWERS:
        v = power_scores[p]
        print(f"{p:<20.2f}{_mean(v['pesq']):>8.3f}{_mean(v['stoi']):>8.3f}"
              f"{_mean(v['sisnr']):>9.2f}")
    print(f"{'expand contrast by':<20}{'PESQ':>8}{'STOI':>8}{'SI-SNR':>9}")
    for k in CONTRASTS:
        v = contrast_scores[k]
        print(f"{k:<20.2f}{_mean(v['pesq']):>8.3f}{_mean(v['stoi']):>8.3f}"
              f"{_mean(v['sisnr']):>9.2f}")


if __name__ == "__main__":
    main()
