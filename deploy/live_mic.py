"""
live_mic.py -- run the model on a live microphone, on the Pi.

    python3 live_mic.py --list                 what devices exist
    python3 live_mic.py                        run it, default devices
    python3 live_mic.py --in 2 --out 3         pick devices by index
    python3 live_mic.py --bypass               raw microphone, for A/B
    python3 live_mic.py --file in.wav out.wav  no hardware needed

The last one is the point of this file being testable at all: it drives the
exact same code path with a WAV file standing in for the sound card, so the
buffering, the chunking and the model wiring can all be checked before the
board exists. Only the two sounddevice calls differ between that and the
real thing.

WHY A WORKER THREAD

The audio callback runs on a real-time thread owned by the sound card. It is
handed a buffer and must return before the next one arrives -- around 100 ms
here -- and if it does not, the driver drops audio and the listener hears a
gap. Model inference takes 17-77 ms depending on how many cores are free,
which is under the deadline but not by a margin worth gambling a live demo
on, and a single slow chunk inside the callback stalls the whole stream.

So the callback does nothing but move memory: it drops the new samples into
a queue and takes whatever finished audio is waiting. A worker thread does
the inference. If the worker falls behind for a moment, the callback emits
silence for that chunk and carries on, which is a brief dropout rather than
a collapsed stream. The count is reported at the end so a dropout is never
silent in both senses.
"""

import os
import sys
import time
import queue
import argparse
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "bundle")
sys.path.insert(0, BUNDLE)

import torch
import stream as st


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

RUN_DIR = os.path.join(BUNDLE, "runs", "0026_crn_comp03_spec600_la6_s0")
# The trained stage-1 run to load.

POSTFILTER_WEIGHTS = os.path.join(BUNDLE, "models",
                                  "postfilter_ch48_la0_snr25.pt")
# The 128 K post-filter. Set to None to run stage 1 alone, which is about
# 3x cheaper and gives up roughly 0.05 PESQ -- the fallback if the board
# cannot keep up.

THREADS = 4
# Cores the model may use. A Pi 5 has four. Lower this if something else
# needs a core; the self-check prints what each count costs.

QUEUE_CHUNKS = 8
# How many chunks may wait between the callback and the worker. Larger
# absorbs a longer stall at the cost of more delay when one happens; at 8
# this can ride out roughly 800 ms of trouble before it starts dropping.

REPORT_EVERY_S = 2.0
# How often to print the running latency line. Set to 0 for silence.

# ═══════════════════════════════════════════════════════════════════
#  LOCKED -- set by the trained model and the STFT
# ═══════════════════════════════════════════════════════════════════
SR = 16000
CHUNK = st.CHUNK_SAMPLES              # 1600 samples, 100 ms


def load_models(device):
    import runlog
    import train
    import postfilter as pf
    run = runlog.RunLog(RUN_DIR)
    model, ck, epoch = train.load_run_model(run, device, "auto")
    model.eval()
    model.strip_aux()
    post = None
    if POSTFILTER_WEIGHTS and os.path.exists(POSTFILTER_WEIGHTS):
        post = pf.load(POSTFILTER_WEIGHTS, device)[0].eval()
    return model, post, ck, epoch


class Engine:
    """Owns the streamer and the worker thread. Feed it, then read it."""

    def __init__(self, model, post, device, bypass=False):
        self.streamer = st.Streamer(model, post, device)
        self.bypass = bypass
        self.inq = queue.Queue(maxsize=QUEUE_CHUNKS)
        self.outbuf = np.zeros(0, dtype=np.float32)
        self.outlock = threading.Lock()
        self.stop = threading.Event()
        self.primed = False        # has the first real output arrived?
        self.dropped_in = 0        # chunks the worker never saw
        self.starved = 0           # callbacks that found no output ready
        self.times = []
        self.worker = threading.Thread(target=self._run, daemon=True)

    def start(self):
        # Warm up before any audio arrives. Without this the first chunk
        # carries lazy kernel selection and takes several times as long as
        # the steady state, which lands as a dropout in the first second.
        warm = np.zeros(CHUNK, dtype=np.float32)
        for _ in range(6):
            self.streamer.push(warm)
        self.streamer.reset()
        self.worker.start()

    def _run(self):
        while not self.stop.is_set():
            try:
                chunk = self.inq.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            out = (chunk if self.bypass else self.streamer.push(chunk))
            self.times.append((time.perf_counter() - t0) * 1000.0)
            if len(out):
                with self.outlock:
                    self.outbuf = np.concatenate([self.outbuf, out])

    def feed(self, chunk):
        try:
            self.inq.put_nowait(chunk.copy())
        except queue.Full:
            self.dropped_in += 1

    def take(self, n):
        """Exactly n samples, padded with silence if the worker is behind.

        The pre-roll matters more than it looks. The model holds 88 ms back
        before a sample is final, so the first couple of calls have nothing
        to give. Emitting silence for those is correct -- that IS the
        latency. But every later shortfall inserts silence into the middle
        of the stream and shifts everything after it permanently, so those
        are counted separately: startup silence is expected, mid-stream
        silence is a dropout worth knowing about.
        """
        with self.outlock:
            if not self.primed:
                if len(self.outbuf) < n:
                    return np.zeros(n, dtype=np.float32)   # pre-roll
                self.primed = True
            if len(self.outbuf) >= n:
                out, self.outbuf = self.outbuf[:n], self.outbuf[n:]
                return out
            out = np.zeros(n, dtype=np.float32)
            have = len(self.outbuf)
            if have:
                out[:have] = self.outbuf
                self.outbuf = self.outbuf[:0]
        self.starved += 1
        return out

    def drain(self):
        """Everything the worker has finished, however much that is."""
        with self.outlock:
            out, self.outbuf = self.outbuf, np.zeros(0, dtype=np.float32)
        return out

    def close(self):
        self.stop.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)

    def report(self):
        t = np.array(self.times[3:]) if len(self.times) > 3 else \
            np.array(self.times)
        if not len(t):
            return "no chunks processed"
        return (f"{np.median(t):.1f} ms median, {t.max():.1f} ms worst "
                f"per {CHUNK / 16:.0f} ms chunk "
                f"({100.0 / max(np.median(t), 1e-9):.1f}x headroom); "
                f"{self.dropped_in} chunks dropped, {self.starved} underruns")


# --------------------------------------------------------------- file mode
def run_file(engine, src, dst):
    import soundfile as sf
    wav, sr = sf.read(src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"{src} is {sr} Hz; this model needs {SR} Hz")

    # A silent lead-in, exactly as a real stream has: the offline path
    # reflect-pads the start of a clip and a live stream cannot.
    wav = np.concatenate([np.zeros(512, dtype=np.float32), wav])

    out = []
    for i in range(0, len(wav) - CHUNK + 1, CHUNK):
        engine.feed(wav[i:i + CHUNK])
        # Real time is what the sound card would impose. Without the wait the
        # worker is raced and every chunk looks starved, which would measure
        # this loop rather than the model.
        time.sleep(CHUNK / SR)
        out.append(engine.take(CHUNK))
    time.sleep(0.5)
    out.append(engine.drain())
    sf.write(dst, np.concatenate(out), SR)
    return dst


# --------------------------------------------------------------- live mode
def _sounddevice():
    """Import sounddevice, or explain what to do instead of a traceback."""
    try:
        import sounddevice as sd
        return sd
    except Exception as e:
        raise SystemExit(
            f"sounddevice is not available ({e}).\n"
            "  Install it:   pip install --no-index "
            "--find-links=wheels sounddevice\n"
            "  On Linux it also needs the PortAudio library:\n"
            "      sudo apt install -y libportaudio2\n"
            "  To check everything else without a sound card, use:\n"
            "      python3 live_mic.py --file in.wav out.wav")


def run_live(engine, in_dev, out_dev):
    sd = _sounddevice()

    def callback(indata, outdata, frames, t, status):
        if status:
            print(f"  [audio] {status}", file=sys.stderr)
        engine.feed(indata[:, 0])
        outdata[:, 0] = engine.take(frames)

    print("listening -- ctrl-c to stop")
    with sd.Stream(samplerate=SR, blocksize=CHUNK, dtype="float32",
                   channels=1, device=(in_dev, out_dev), callback=callback):
        last = time.time()
        try:
            while True:
                time.sleep(0.2)
                if REPORT_EVERY_S and time.time() - last > REPORT_EVERY_S:
                    print("  " + engine.report())
                    last = time.time()
        except KeyboardInterrupt:
            print("\nstopping")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--list", action="store_true", help="list audio devices")
    ap.add_argument("--in", dest="in_dev", type=int, default=None)
    ap.add_argument("--out", dest="out_dev", type=int, default=None)
    ap.add_argument("--bypass", action="store_true",
                    help="pass the microphone through untouched, for A/B")
    ap.add_argument("--file", nargs=2, metavar=("IN", "OUT"),
                    help="process a WAV instead of using the sound card")
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()

    if args.list:
        print(_sounddevice().query_devices())
        return 0

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    model, post, ck, epoch = load_models(device)
    print(f"model       {ck} epoch {epoch}, "
          f"post-filter {'none' if post is None else os.path.basename(POSTFILTER_WEIGHTS)}")
    print(f"threads     {args.threads}   chunk {CHUNK / 16:.0f} ms   "
          f"added latency ~{CHUNK / 16 + 88:.0f} ms")
    if args.bypass:
        print("bypass      ON -- the model is not running")

    engine = Engine(model, post, device, bypass=args.bypass)
    engine.start()
    try:
        if args.file:
            dst = run_file(engine, args.file[0], args.file[1])
            print(f"wrote       {dst}")
        else:
            run_live(engine, args.in_dev, args.out_dev)
    finally:
        engine.close()
    print("timing      " + engine.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
