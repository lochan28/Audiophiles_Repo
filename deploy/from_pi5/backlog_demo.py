"""
backlog_demo.py -- show WHY this board cannot run the model live.

    python3 backlog_demo.py                  microphone in, headset out
    python3 backlog_demo.py --seconds 40     stop after 40 s
    python3 backlog_demo.py --no-postfilter  stage 1 only (slower growth)
    python3 backlog_demo.py --file in.wav    no microphone, same numbers

The board is not out of memory and it does not crash. It is simply slower
than the audio arrives: every second of speech takes about four seconds to
process, so the unprocessed audio piles up and the output falls further
behind for as long as you keep talking. That is the real reason a slower
board cannot be deployed live, and this script makes it visible:

    t=  5 s   captured   5.0 s   processed  1.2 s   BACKLOG   3.8 s
    t= 10 s   captured  10.0 s   processed  2.5 s   BACKLOG   7.5 s
    t= 20 s   captured  20.0 s   processed  5.0 s   BACKLOG  15.0 s

and, through the headset, you hear yourself from further and further ago.

A board that keeps up shows BACKLOG holding near zero. Run the same script
on a Pi 5 and that is what it should print -- the honest before/after.

Nothing here is rigged. The input queue is unbounded on purpose, so that
the backlog is measured rather than dropped; live_mic.py bounds it and
counts drops, which is the right behaviour for deployment and the wrong one
for showing the problem.
"""

import os
import sys
import time
import queue
import argparse
import threading
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

RUN_DIR = os.path.join(BUNDLE, "runs", "0026_crn_comp03_spec600_la6_s0")
POSTFILTER_WEIGHTS = os.path.join(BUNDLE, "models",
                                  "postfilter_ch48_la0_snr25.pt")
THREADS = 4
REPORT_EVERY_S = 1.0
DEFAULT_SECONDS = 30
# How long to run before stopping on its own. Thirty seconds is enough for
# the backlog to reach ~20 s on a Pi 4, which nobody mistakes for jitter.

# ═══════════════════════════════════════════════════════════════════
SR = 16000                       # what the model needs
CHUNK = 1600                     # 100 ms at 16 kHz


def load(device):
    import runlog
    import train
    import postfilter as pf
    m = train.load_run_model(runlog.RunLog(RUN_DIR), device, "auto")[0].eval()
    m.strip_aux()
    post = None
    if POSTFILTER_WEIGHTS and os.path.exists(POSTFILTER_WEIGHTS):
        post = pf.load(POSTFILTER_WEIGHTS, device)[0].eval()
    return m, post


class Backlog:
    """Unbounded input queue, so the pile-up is measured, not hidden."""

    def __init__(self, model, post, device):
        import stream
        self.s = stream.Streamer(model, post, device)
        self.s.push(np.zeros(CHUNK, dtype=np.float32))
        self.s.reset()
        self.q = queue.Queue()
        self.out = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.captured = 0            # samples that have arrived
        self.processed = 0           # samples the model has finished
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                c = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            o = self.s.push(c)
            self.processed += len(c)
            if len(o):
                with self.lock:
                    self.out = np.concatenate([self.out, o])

    def feed(self, c):
        self.captured += len(c)
        self.q.put(c.copy())

    def take(self, n):
        with self.lock:
            if len(self.out) >= n:
                r, self.out = self.out[:n], self.out[n:]
                return r
            r = np.zeros(n, dtype=np.float32)
            r[:len(self.out)] = self.out
            self.out = self.out[:0]
            return r

    def line(self, t):
        cap, pro = self.captured / SR, self.processed / SR
        return (f"  t={t:4.0f} s   captured {cap:5.1f} s   "
                f"processed {pro:5.1f} s   BACKLOG {cap - pro:5.1f} s")


def run_mic(bl, seconds, in_dev, out_dev):
    import sounddevice as sd
    # USB headsets speak 48 kHz, not 16. Run the card at its own rate and
    # convert on the way through: average every 3 samples in, repeat each
    # sample 3 times out. Crude, but this is a demo of timing, not fidelity.
    rate = int(sd.query_devices(out_dev, "output")["default_samplerate"])
    k = rate // SR
    if rate != k * SR:
        raise SystemExit(f"device rate {rate} is not a multiple of {SR}")
    block = CHUNK * k

    def cb(indata, outdata, frames, t, status):
        mono = indata[:, 0]
        bl.feed(mono[:len(mono) // k * k].reshape(-1, k).mean(axis=1)
                .astype(np.float32))
        o = np.repeat(bl.take(frames // k), k)
        outdata[:, 0] = o
        if outdata.shape[1] > 1:
            outdata[:, 1] = o

    print(f"  listening for {seconds} s at {rate} Hz -- talk continuously, "
          f"and listen to how far behind the echo falls\n")
    t0 = time.time()
    with sd.Stream(samplerate=rate, blocksize=block, dtype="float32",
                   channels=(1, 2), device=(in_dev, out_dev), callback=cb):
        last = 0
        while time.time() - t0 < seconds:
            time.sleep(0.05)
            el = time.time() - t0
            if el - last >= REPORT_EVERY_S:
                print(bl.line(el))
                last = el


def run_file(bl, seconds, path):
    """Same numbers with a WAV standing in for the microphone."""
    import soundfile as sf
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"{path} is {sr} Hz, need {SR}")
    reps = int(np.ceil(seconds * SR / len(w))) + 1
    w = np.tile(w, reps)
    print(f"  feeding {os.path.basename(path)} on a loop for {seconds} s, "
          f"at real-time pace\n")
    t0 = time.time()
    last = 0
    i = 0
    while time.time() - t0 < seconds:
        bl.feed(w[i:i + CHUNK])
        i += CHUNK
        bl.take(CHUNK)
        time.sleep(CHUNK / SR)
        el = time.time() - t0
        if el - last >= REPORT_EVERY_S:
            print(bl.line(el))
            last = el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=DEFAULT_SECONDS)
    ap.add_argument("--no-postfilter", action="store_true")
    ap.add_argument("--int8", action="store_true",
                    help="dynamic int8 on the LSTM (qnnpack); 1.45-1.7x on ARM")
    ap.add_argument("--file", default=None)
    ap.add_argument("--in", dest="in_dev", type=int, default=None)
    ap.add_argument("--out", dest="out_dev", type=int, default=None)
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    model, post = load(device)
    if args.no_postfilter:
        post = None
    if args.int8:
        # ARM has real int8 kernels (qnnpack); x86 did not benefit, ARM does.
        # Measured cost: 1.8e-3 max error on rms 0.063 -- inaudible.
        torch.backends.quantized.engine = "qnnpack"
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8)

    print("=" * 66)
    print("  Why this board cannot run the model live: the backlog")
    print("=" * 66)
    print(f"  model: stage 1{' + post-filter' if post else ' only'}"
          f"{', int8' if args.int8 else ', fp32'}, {args.threads} threads")

    bl = Backlog(model, post, device)
    try:
        if args.file:
            run_file(bl, args.seconds, args.file)
        else:
            import sounddevice as sd
            if args.in_dev is None or args.out_dev is None:
                devs = list(enumerate(sd.query_devices()))
                usb = [i for i, d in devs if "USB" in d["name"].upper()]
                if args.in_dev is None:
                    ins = [i for i in usb if devs[i][1]["max_input_channels"]]
                    args.in_dev = ins[0] if ins else sd.default.device[0]
                if args.out_dev is None:
                    outs = [i for i in usb if devs[i][1]["max_output_channels"]]
                    args.out_dev = outs[0] if outs else sd.default.device[1]
            print(f"  mic: [{args.in_dev}]  out: [{args.out_dev}]")
            run_mic(bl, args.seconds, args.in_dev, args.out_dev)
    finally:
        # Let the worker finish its current chunk. Killing it mid-op is
        # what prints 'terminate called without an active exception'.
        bl.stop.set()
        bl.worker.join(timeout=3.0)

    cap, pro = bl.captured / SR, bl.processed / SR
    print(f"\n  after {args.seconds} s: {cap:.1f} s of audio arrived, "
          f"{pro:.1f} s was processed.")
    if cap > 0:
        print(f"  this board keeps up with {100 * pro / cap:.0f}% of real "
              f"time; the other {100 * (1 - pro / cap):.0f}% piles up.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
