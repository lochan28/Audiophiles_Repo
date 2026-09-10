"""
quantize_test.py

Does int8 quantisation make the two-stage model fast enough for a Raspberry
Pi, and what does it cost in quality?

The deck claims int8 without ever having measured it. This measures it: the
same 125 held-out testing clips through fp32 and int8, same machine, same
threads, reporting PESQ / STOI / SI-SNR and CPU time side by side.

WHAT IS ACTUALLY QUANTISED

Dynamic quantisation, which stores weights as int8 and computes activations
in float, chosen because it needs no calibration set and no retraining. It
applies to Linear and LSTM modules only -- Conv2d is untouched, because
dynamic quantisation has no conv kernel.

That sounds like a limitation and is mostly the opposite. The LSTM holds
6.69 M of the model's 9.26 M parameters and is the memory bottleneck (669
MB/s just to re-read its weights every frame), so quantising the LSTM alone
addresses the part that actually hurts on a Pi. The decoder convolutions are
the compute bottleneck and stay in fp32; fixing those needs static
quantisation with a calibration pass, which is a bigger job and only worth
starting if these numbers justify it.

Everything runs on CPU. GPU numbers would say nothing about a Pi.
"""

import os
import sys
import time
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import train
import evaluate as ev
import model_crn
import postfilter as pf
from runlog import RunLog
from mixing import FrozenMixtures


# ----------------------------------------------------------- TUNING KNOBS
#
# THREADS      how many CPU cores to allow. The point of the sweep is that a
#              Pi has four slow cores, so 1 and 4 bracket the real board.
#              Measured earlier: four threads buy only 1.23x in streaming
#              mode, because the LSTM is sequential across frames.
# SPLIT        which frozen set to score on. "testing" is the held-out one
#              and the only honest choice for a number that gets quoted.
# N_CLIPS      how many clips to score. None means all of them. Lowering it
#              makes a quick check; the reported numbers then are not
#              comparable to anything else in the project.
# WARMUP       untimed passes before timing starts. Without these the first
#              clip carries lazy kernel selection and reads several times
#              slower than the steady state.
THREADS = (1, 4)
SPLIT = "testing"
N_CLIPS = None
WARMUP = 3

# ----------------------------------------------------------------- LOCKED
#
# Set by data on disk and by what the checkpoint was trained against.
# Changing these does not make a faster model, it makes an invalid comparison.
QUANT_MODULES = {torch.nn.Linear, torch.nn.LSTM}
QUANT_DTYPE = torch.qint8


def size_mb(model):
    """Bytes actually occupied by the parameters, quantised or not."""
    total = 0
    for p in model.state_dict().values():
        if hasattr(p, "numel"):
            if p.dtype == torch.qint8:
                total += p.numel()                       # 1 byte each
            elif p.is_floating_point():
                total += p.numel() * p.element_size()
            else:
                total += p.numel() * p.element_size()
    # Quantised LSTM weights hide inside packed params rather than the
    # state_dict, so fall back to serialising when that undercounts.
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return max(total, buf.tell()) / 1e6


def quantize(model):
    """int8 copy of a model. The original is left untouched."""
    return torch.quantization.quantize_dynamic(
        model, QUANT_MODULES, dtype=QUANT_DTYPE)


def run(stage1, post, clips, device):
    """Score and time one model pair over the clips. Returns metrics + ms."""
    rows, times = [], []
    for k, (noisy, clean) in enumerate(clips):
        x = torch.from_numpy(noisy).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _, mask, _, _ = stage1(x)
            nspec = model_crn.to_spectrogram(x)
            ospec = post(nspec * mask, nspec) if post is not None else nspec * mask
            out = model_crn.to_waveform(ospec, x.shape[-1])
        dt = (time.perf_counter() - t0) * 1000.0
        if k >= WARMUP:
            times.append(dt)
        o = out[0].cpu().numpy()
        rows.append((ev.safe_pesq(clean, o), ev.safe_stoi(clean, o),
                     ev.si_snr(o, clean)))
    a = np.array([[r for r in row] for row in rows], dtype=np.float64)
    good = ~np.isnan(a).any(axis=1)
    m = a[good].mean(axis=0)
    return {"pesq": m[0], "stoi": m[1], "si_snr": m[2],
            "ms": float(np.median(times)), "ms_worst": float(np.max(times)),
            "clips": int(good.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="stage 1 run name")
    ap.add_argument("--clips", type=int, default=N_CLIPS)
    ap.add_argument("--no-postfilter", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu")

    import demo_core as core
    run_name = args.run or core.stage1_for_postfilter()
    stage1 = train.load_run_model(RunLog.open(run_name), device)[0].eval()
    stage1.strip_aux()

    post = None
    if not args.no_postfilter and os.path.exists(core.POSTFILTER_PATH):
        post = pf.load(core.POSTFILTER_PATH, device)[0].eval()

    ds = FrozenMixtures(SPLIT)
    n = len(ds) if args.clips is None else min(args.clips, len(ds))
    clips = []
    for i in range(n):
        d = ds.describe(i)
        clips.append((d["noisy"].astype(np.float32),
                      d["clean"].astype(np.float32)))

    q1 = quantize(stage1)
    qp = quantize(post) if post is not None else None

    dur = len(clips[0][0]) / model_crn.SAMPLE_RATE
    print(f"stage 1     {run_name}")
    print(f"post-filter {'none' if post is None else os.path.basename(core.POSTFILTER_PATH)}")
    print(f"clips       {n} from {SPLIT}, {dur:.1f} s each\n")

    print(f"size        fp32 {size_mb(stage1):6.2f} MB"
          f"   int8 {size_mb(q1):6.2f} MB   stage 1")
    if post is not None:
        print(f"            fp32 {size_mb(post):6.2f} MB"
              f"   int8 {size_mb(qp):6.2f} MB   post-filter")
    print()

    base = {}
    for th in THREADS:
        torch.set_num_threads(th)
        for tag, s, p in (("fp32", stage1, post), ("int8", q1, qp)):
            r = run(s, p, clips, device)
            rt = dur * 1000.0 / r["ms"]
            if tag == "fp32" and th == THREADS[0]:
                base = r
            d = "" if tag == "fp32" else (
                f"   dPESQ {r['pesq'] - base['pesq']:+.4f}"
                f"  dSTOI {r['stoi'] - base['stoi']:+.4f}"
                f"  dSI-SNR {r['si_snr'] - base['si_snr']:+.3f}")
            print(f"{th} thread{'s' if th > 1 else ' '}  {tag}  "
                  f"PESQ {r['pesq']:.4f}  STOI {r['stoi']:.4f}  "
                  f"SI-SNR {r['si_snr']:6.3f}  "
                  f"{r['ms']:6.1f} ms ({rt:5.2f}x RT){d}")
        print()

    print("Deltas are against fp32 at "
          f"{THREADS[0]} thread. Noise floor: PESQ 0.0061, STOI 0.0004, "
          "SI-SNR 0.0218.")


if __name__ == "__main__":
    main()
