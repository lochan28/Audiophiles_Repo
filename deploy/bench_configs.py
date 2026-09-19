"""
bench_configs.py -- the four ways to run the model, timed on this board.

    python3 bench_configs.py

Streams a clip through every combination of {fp32, int8} x {two-stage,
stage 1 only} and prints median and worst chunk time against the 100 ms
deadline. This is the table that decided what the Pi 5 demo runs:

    Pi 5, 4 threads             median   worst
    fp32 two-stage              145.5    170.1   MISSED
    fp32 stage 1                100.4    126.4   MISSED (just)
    int8 two-stage              110.8    144.6   MISSED
    int8 stage 1                 59.7     85.4   ok

Pi 4: 424 / 290 / 308 / 191 -- every one MISSED.

int8 here is dynamic quantisation of the LSTM and Linear layers via
qnnpack, ARM's int8 backend. It bought nothing on x86 (fbgemm; fp32 was
already fast) and 1.45-1.7x on ARM, at a max error of 1.8e-3 on rms 0.063.

Run it after selfcheck.py, and run it COOL: the Pi 5 with no cooler hits
its 80 C soft limit in about a minute of continuous inference, after which
every number here drifts upward. The temperature is printed before and
after so a throttled run is recognisable as one.
"""

import os
import sys
import time
import argparse
import subprocess
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "bundle")
sys.path.insert(0, BUNDLE)

import torch

# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS
# ═══════════════════════════════════════════════════════════════════
DEFAULT_RUN_NAME = "0026_crn_comp03_spec600_la6_s0"
POSTFILTER_WEIGHTS = os.path.join(BUNDLE, "models",
                                  "postfilter_ch48_la0_snr25.pt")
CLIP = os.path.join(BUNDLE, "generated_test_mixtures", "testing", "noisy",
                    "testing_000.wav")
THREADS = 4
REPEATS = 2          # the clip is looped this many times per configuration
WARMUP_CHUNKS = 4    # untimed chunks at the start of each configuration
DEADLINE_MS = 100.0
# ═══════════════════════════════════════════════════════════════════


def temp():
    try:
        return subprocess.check_output(["vcgencmd", "measure_temp"],
                                       text=True).strip()
    except Exception:
        return "temp=n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN_NAME, metavar="RUN_NAME",
                    help="run folder under bundle/runs/ to benchmark "
                        f"(default: {DEFAULT_RUN_NAME})")
    args = ap.parse_args()
    run_dir = os.path.join(BUNDLE, "runs", args.run)

    import soundfile as sf
    import runlog
    import train
    import postfilter as pf
    import stream

    torch.set_num_threads(THREADS)
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in \
        torch.backends.quantized.supported_engines else \
        torch.backends.quantized.engine
    dev = torch.device("cpu")

    print(f"run: {args.run}")
    m = train.load_run_model(runlog.RunLog(run_dir), dev, "auto")[0].eval()
    m.strip_aux()
    post = pf.load(POSTFILTER_WEIGHTS, dev)[0].eval()
    q = torch.quantization.quantize_dynamic(
        m, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)

    w, _ = sf.read(CLIP, dtype="float32")
    wav = np.concatenate([np.zeros(512, dtype=np.float32)] + [w] * REPEATS)
    N = stream.CHUNK_SAMPLES

    print(f"{THREADS} threads, engine {torch.backends.quantized.engine}, "
          f"{temp()}")
    print(f"{'configuration':<22} {'median':>8} {'worst':>8}   verdict")
    ref = None
    for tag, model, p in (("fp32 two-stage", m, post), ("fp32 stage 1", m, None),
                          ("int8 two-stage", q, post), ("int8 stage 1", q, None)):
        s = stream.Streamer(model, p, dev)
        out, t = [], []
        for k, i in enumerate(range(0, len(wav) - N + 1, N)):
            t0 = time.perf_counter()
            out.append(s.push(wav[i:i + N]))
            if k >= WARMUP_CHUNKS:
                t.append((time.perf_counter() - t0) * 1000.0)
        t = np.array(t)
        ok = "ok" if t.max() <= DEADLINE_MS else "MISSED"
        print(f"{tag:<22} {np.median(t):8.1f} {t.max():8.1f}   {ok}")
        o = np.concatenate(out)
        if tag == "fp32 stage 1":
            ref = o
        elif tag == "int8 stage 1" and ref is not None:
            n = min(len(o), len(ref))
            print(f"{'':<22} int8 vs fp32 max diff "
                  f"{np.abs(o[:n] - ref[:n]).max():.1e} "
                  f"(rms {np.sqrt((ref ** 2).mean()):.3f})")
    print(f"after: {temp()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
