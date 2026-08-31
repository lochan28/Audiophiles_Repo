"""What is actually wrong with the mask the model paints?

Two facts are established and they do not fit together comfortably:

  - mask_resolution.py: the ideal mask needs fine detail. Blurring it across
    two neighbouring bins costs 0.90 of the 2.13 PESQ available.
  - run 0013: giving the LSTM a four times finer view of frequency made things
    slightly WORSE, not better. The skip connections already hand full
    257-band detail to the decoder, so the model was never unable to draw a
    sharp mask.

So the model *can* be sharp. The open question is whether it *is*. A model
that is unsure hedges -- under a squared-error objective the safest guess in an
uncertain bin is a middling value, and averaging over uncertainty produces
exactly the smooth mask the blur test says is fatal. That would make the
smoothness a symptom of the objective and the uncertainty, not of architecture,
and it would explain why the stride change did nothing.

This script measures it directly. For each clip it takes the model's mask and
the ideal mask and compares them, then does the decisive test: it blurs the
IDEAL mask until its roughness matches the MODEL's, and scores that. Read the
last table like this --

  if blurred-ideal scores about what the model scores, then smoothness explains
  the model's deficit almost entirely, and the fix is whatever makes it commit;

  if blurred-ideal still scores far higher, the model is wrong in ways that
  have nothing to do with sharpness, and smoothness is a red herring.
"""

import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_crn
import train as T
from runlog import RunLog
from mixing import FrozenMixtures
from evaluate import si_snr, safe_pesq, safe_stoi, _mean
from mask_resolution import coarsen

EPS = 1e-8

# Blur levels to search over when matching the model's roughness. Fractional
# steps because the model may sit between whole factors.
BLUR_CANDIDATES = (1, 1.5, 2, 3, 4, 6, 8, 12, 16)


def roughness(mask):
    """Mean absolute change between neighbouring frequency bins.

    A sharp mask that cuts between harmonics swings up and down constantly and
    scores high. A smooth, hedged mask barely changes from bin to bin and
    scores low. This is the number the blur test is implicitly about.
    """
    return float((mask[:, :, 1:] - mask[:, :, :-1]).abs().mean())


def load_model(run_name, device):
    run = RunLog.open(run_name)
    cfg = run.config()
    channels = cfg.get("encoder_channels", model_crn.ENCODER_CHANNELS)
    # Runs before 0013 predate ENCODER_FREQ_STRIDES and every one of them
    # halved at every layer. Falling back to the CURRENT default would silently
    # build the wrong shape for them.
    strides = cfg.get("encoder_freq_strides", [2] * len(channels))
    model = T.ARCHITECTURES[cfg.get("arch", "crn")](
        encoder_channels=channels,
        mask_max=cfg.get("mask_max", 1.0),
        freq_strides=strides).to(device)
    ck = torch.load(run.checkpoint_path("best.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    return model.eval(), ck["epoch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-clips", type=int, default=40)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_model(args.run, device)
    ds = FrozenMixtures(args.split)
    n = min(args.max_clips, len(ds))
    print(f"{args.run}, best checkpoint from epoch {epoch}, {n} clips\n")

    stats = {"rough_model": [], "rough_ideal": [], "abs_err": [],
             "model_mean": [], "ideal_mean": [], "corr": []}
    blur_scores = {b: {"pesq": [], "stoi": []} for b in BLUR_CANDIDATES}
    model_scores = {"pesq": [], "stoi": [], "sisnr": []}

    for i in range(n):
        d = ds.describe(i)
        clean, noisy = d["clean"], d["noisy"]
        x = torch.from_numpy(noisy).unsqueeze(0).to(device)
        with torch.no_grad():
            est, mask, _, _ = model(x)
        est = est.squeeze(0).cpu().numpy()
        mask = mask.cpu()

        S = model_crn.to_spectrogram(torch.from_numpy(clean).unsqueeze(0))
        Y = model_crn.to_spectrogram(torch.from_numpy(noisy).unsqueeze(0))
        mag_y = Y.abs()
        phase_y = Y / (mag_y + EPS)
        ideal = (S.abs() / (mag_y + EPS)).clamp(max=1.0)

        stats["rough_model"].append(roughness(mask))
        stats["rough_ideal"].append(roughness(ideal))
        stats["abs_err"].append(float((mask - ideal).abs().mean()))
        stats["model_mean"].append(float(mask.mean()))
        stats["ideal_mean"].append(float(ideal.mean()))
        a = (mask - mask.mean()).flatten()
        b = (ideal - ideal.mean()).flatten()
        stats["corr"].append(float((a * b).sum()
                                   / (a.norm() * b.norm() + EPS)))

        model_scores["pesq"].append(safe_pesq(clean, est))
        model_scores["stoi"].append(safe_stoi(clean, est))
        model_scores["sisnr"].append(si_snr(est, clean))

        for blur in BLUR_CANDIDATES:
            m = coarsen(ideal, freq_factor=blur)
            y = model_crn.to_waveform(m * mag_y * phase_y,
                                      len(clean)).squeeze(0).numpy()
            blur_scores[blur]["pesq"].append(safe_pesq(clean, y))
            blur_scores[blur]["stoi"].append(safe_stoi(clean, y))

    rm = float(np.mean(stats["rough_model"]))
    ri = float(np.mean(stats["rough_ideal"]))
    print("how the model's mask compares with the ideal one")
    print(f"  roughness, model         {rm:.4f}")
    print(f"  roughness, ideal         {ri:.4f}   "
          f"({'model is SMOOTHER' if rm < ri else 'model is rougher'}, "
          f"ratio {ri / max(rm, EPS):.2f}x)")
    print(f"  mean absolute difference {np.mean(stats['abs_err']):.4f}")
    print(f"  correlation              {np.mean(stats['corr']):.4f}")
    print(f"  average value, model     {np.mean(stats['model_mean']):.4f}")
    print(f"  average value, ideal     {np.mean(stats['ideal_mean']):.4f}")

    print(f"\nblurring the IDEAL mask, to find the blur that matches the model")
    print(f"{'blur':>6}{'roughness':>12}{'PESQ':>9}{'STOI':>8}")
    best, best_gap = None, 1e9
    for blur in BLUR_CANDIDATES:
        with torch.no_grad():
            d0 = ds.describe(0)
            Y0 = model_crn.to_spectrogram(
                torch.from_numpy(d0["noisy"]).unsqueeze(0))
            S0 = model_crn.to_spectrogram(
                torch.from_numpy(d0["clean"]).unsqueeze(0))
            i0 = (S0.abs() / (Y0.abs() + EPS)).clamp(max=1.0)
            r = roughness(coarsen(i0, freq_factor=blur))
        gap = abs(r - rm)
        if gap < best_gap:
            best, best_gap = blur, gap
        mark = ""
        print(f"{blur:>6}{r:>12.4f}{_mean(blur_scores[blur]['pesq']):>9.3f}"
              f"{_mean(blur_scores[blur]['stoi']):>8.3f}{mark}")

    print(f"\nthe model's roughness {rm:.4f} matches a blur of about {best}")
    print(f"{'ideal mask blurred to match':<32}"
          f"{_mean(blur_scores[best]['pesq']):>8.3f}"
          f"{_mean(blur_scores[best]['stoi']):>8.3f}")
    print(f"{'the model itself':<32}{_mean(model_scores['pesq']):>8.3f}"
          f"{_mean(model_scores['stoi']):>8.3f}")
    print(f"{'the unblurred ideal mask':<32}"
          f"{_mean(blur_scores[1]['pesq']):>8.3f}"
          f"{_mean(blur_scores[1]['stoi']):>8.3f}")


if __name__ == "__main__":
    main()
