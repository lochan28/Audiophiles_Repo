"""Score an external system on our frozen test set, beside our own model.

Every number this project has produced is relative to its own baselines, which
leaves one question unanswerable: is our score weak, or is our test set hard?
A published system scored on OUR mixtures answers it.

    python scripts/benchmark_external.py --enhanced-dir <dir> --run 0026_...

The external system is given the same noisy wavs and its output is scored with
the same PESQ, STOI and SI-SNR code, so the only difference is the model.

TWO THINGS THAT WOULD MAKE THE COMPARISON DISHONEST, both handled here:

- **Length.** A system with lookahead may return fewer samples than it was
  given, because it compensates its own delay by trimming. Padding the shorter
  one with silence would penalise it for audio we deleted. Instead every signal
  -- clean, noisy, ours, theirs -- is truncated to the shortest common length,
  so all four are scored over exactly the same window.
- **Alignment.** A time-shifted output scores terribly no matter how clean it
  is. The lag between each enhanced file and its input is measured by
  cross-correlation and reported; anything other than 0 invalidates the run.
"""

import os
import sys
import argparse

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train as T
from runlog import RunLog
from mixing import FrozenMixtures
from evaluate import si_snr, safe_pesq, safe_stoi, _mean


def measure_lag(reference, signal):
    """Sample offset of `signal` against `reference`, by cross-correlation."""
    n = min(len(reference), len(signal))
    x = reference[:n] - reference[:n].mean()
    y = signal[:n] - signal[:n].mean()
    return int(np.correlate(y, x, mode="full").argmax() - (n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enhanced-dir", required=True,
                    help="the external system's output wavs")
    ap.add_argument("--run", required=True, help="our run to compare against")
    ap.add_argument("--split", default="testing")
    ap.add_argument("--label", default="external")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck, epoch = T.load_run_model(RunLog.open(args.run), device)
    ds = FrozenMixtures(args.split)
    print(f"ours   {args.run}, {ck} epoch {epoch}")
    print(f"theirs {args.enhanced_dir}")
    print(f"{len(ds)} frozen {args.split} mixtures\n")

    rows = {k: {"pesq": [], "stoi": [], "sisnr": []}
            for k in ("noisy", "ours", args.label)}
    lags, trims, missing = [], [], 0

    for i in range(len(ds)):
        d = ds.describe(i)
        stem = os.path.basename(ds.rows[i]["noisy_file"])
        path = os.path.join(args.enhanced_dir, stem)
        if not os.path.exists(path):
            missing += 1
            continue
        theirs, sr = sf.read(path, dtype="float32")
        if theirs.ndim > 1:
            theirs = theirs.mean(axis=1)
        assert sr == 16000, f"{stem} is {sr} Hz, expected 16000"

        with torch.no_grad():
            est, _, _, _ = model(
                torch.from_numpy(d["noisy"]).unsqueeze(0).to(device))
        ours = est.squeeze(0).cpu().numpy()

        lags.append(measure_lag(d["noisy"], theirs))
        n = min(len(d["clean"]), len(d["noisy"]), len(ours), len(theirs))
        trims.append(len(d["clean"]) - n)

        clean = d["clean"][:n]
        for name, sig in ((("noisy"), d["noisy"][:n]),
                          ("ours", ours[:n]),
                          (args.label, theirs[:n])):
            rows[name]["pesq"].append(safe_pesq(clean, sig))
            rows[name]["stoi"].append(safe_stoi(clean, sig))
            rows[name]["sisnr"].append(si_snr(sig, clean))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ds)}")

    # A tolerance of one sample (0.0625 ms at 16 kHz) rather than exact zero:
    # cross-correlation ties on near-silent clips give a +-1 result that is
    # measurement noise, not misalignment. Anything larger is a real shift and
    # would invalidate the scores.
    worst = max(map(abs, lags))
    nonzero = sum(1 for x in lags if x)
    verdict = "OK" if worst <= 1 else "NOT ALIGNED - INVALID"
    print(f"\nalignment: lag min {min(lags)}, max {max(lags)} samples, "
          f"{nonzero} of {len(lags)} clips non-zero ({verdict})")
    print(f"window: trimmed {min(trims)}-{max(trims)} samples "
          f"({max(trims) / 16000 * 1000:.0f} ms) so all systems share one window")
    if missing:
        print(f"WARNING: {missing} clips had no external output and were skipped")

    print(f"\n{'system':<14}{'PESQ':>9}{'STOI':>9}{'SI-SNR':>9}")
    print("-" * 41)
    for name in ("noisy", "ours", args.label):
        r = rows[name]
        print(f"{name:<14}{_mean(r['pesq']):>9.3f}{_mean(r['stoi']):>9.4f}"
              f"{_mean(r['sisnr']):>9.2f}")
    print("-" * 41)
    print(f"{'target':<14}{2.5:>9.3f}{0.85:>9.4f}{15.0:>9.2f}")


if __name__ == "__main__":
    main()
