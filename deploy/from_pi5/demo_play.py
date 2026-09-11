"""
demo_play.py -- show the model working, on the Pi, in front of people.

    python3 demo_play.py                  default clip
    python3 demo_play.py testing_003      a specific clip
    python3 demo_play.py --list           what clips are available
    python3 demo_play.py --out 2          pick an audio output device
    python3 demo_play.py --no-audio       just the numbers, no playback

What the audience sees and hears, in order:

  1. the NOISY input plays
  2. the model runs on it, chunk by chunk, timing scrolling on screen,
     ending with the honest real-time factor for this board
  3. the ENHANCED output plays
  4. the score: SI-SNR before and after, against the clean reference

The point of step 2 is that people watch the Pi do the work rather than
being told it did. On a board that is slower than real time, that turns
"not real time" from a weakness into a visible, honest number -- and makes
it obvious why a faster board is the deployment target.

Every sample played in step 3 came out of stream.py, the same path that
selfcheck.py measures. Nothing is precomputed.
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "bundle")
sys.path.insert(0, BUNDLE)

import torch


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

RUN_DIR = os.path.join(BUNDLE, "runs", "0026_crn_comp03_spec600_la6_s0")
POSTFILTER_WEIGHTS = os.path.join(BUNDLE, "models",
                                  "postfilter_ch48_la0_snr25.pt")
# Set to None to demo stage 1 alone -- about 1.4x faster on this board.

CLIPS_DIR = os.path.join(BUNDLE, "generated_test_mixtures", "testing")
DEFAULT_CLIP = "testing_003"
# The most dramatic of the bundled clips: 4.8 -> 12.8 dB, gunfire over
# speech. testing_000 is the hardest (-1.6 dB in), testing_007 the cleanest.

THREADS = 4

PAUSE_S = 0.8
# Silence between "noisy" and "enhanced" so the two do not run together.

# ═══════════════════════════════════════════════════════════════════
SR = 16000


def si_snr(est, ref):
    est = est - est.mean()
    ref = ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    t = a * ref
    return float(10 * np.log10(((t ** 2).sum() + 1e-8) /
                               (((est - t) ** 2).sum() + 1e-8)))


def align(out, ref, search=4000):
    """The stream output lags its input by a fixed few hundred samples.
    Find that lag once so the score compares like with like."""
    L = min(len(ref), len(out) - search)
    best = max(((d, si_snr(out[d:d + L], ref[:L])) for d in range(0, search, 8)),
               key=lambda z: z[1])
    return out[best[0]:best[0] + L], ref[:L], best[0]


def play(sd, wav, device, label):
    # USB headsets commonly refuse 16 kHz outright (paInvalidSampleRate) and
    # only speak 48 kHz. Resample to whatever the device says it wants;
    # linear interpolation is fine for playback and needs no extra library.
    rate = int(sd.query_devices(device, "output")["default_samplerate"])
    if rate != SR:
        n = int(len(wav) * rate / SR)
        wav = np.interp(np.linspace(0, len(wav) - 1, n),
                        np.arange(len(wav)), wav).astype(np.float32)
    print(f"\n  ▶ playing {label} ({len(wav) / rate:.1f} s at {rate} Hz) ...",
          end="", flush=True)
    sd.play(wav, rate, device=device, blocking=True)
    print(" done")
    time.sleep(PAUSE_S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default=DEFAULT_CLIP)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=int, default=None, help="output device")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()

    noisy_dir = os.path.join(CLIPS_DIR, "noisy")
    if args.list:
        for f in sorted(os.listdir(noisy_dir)):
            print(f[:-4])
        return 0

    sd = None
    if not args.no_audio:
        try:
            import sounddevice as sd
            # Resolve the default explicitly. Passing None lets PortAudio
            # pick a route that rejected float32 on the Pi ("Sample format
            # not supported"); naming the same device by index works.
            if args.out is None:
                # PortAudio's idea of "default" changed between two runs on
                # the same Pi (USB headset once, HDMI the next). Prefer a USB
                # device by name when one exists; that is what a headset is.
                outs = [(i, d) for i, d in enumerate(sd.query_devices())
                        if d["max_output_channels"] > 0]
                usb = [i for i, d in outs if "USB" in d["name"].upper()]
                args.out = usb[0] if usb else sd.default.device[1]
            print(f"  audio out: [{args.out}] "
                  f"{sd.query_devices(args.out, 'output')['name']}")
        except Exception as e:
            print(f"sounddevice unavailable ({e}); running with --no-audio.")
            print("  fix: sudo apt install -y libportaudio2")

    import soundfile as sf
    import runlog
    import train
    import postfilter as pf
    import stream

    torch.set_num_threads(args.threads)
    dev = torch.device("cpu")

    print("=" * 62)
    print("  ANC on Raspberry Pi -- live processing demo")
    print("=" * 62)
    print(f"  loading model ...", end="", flush=True)
    m = train.load_run_model(runlog.RunLog(RUN_DIR), dev, "auto")[0].eval()
    m.strip_aux()
    post = None
    if POSTFILTER_WEIGHTS and os.path.exists(POSTFILTER_WEIGHTS):
        post = pf.load(POSTFILTER_WEIGHTS, dev)[0].eval()
    n1 = sum(p.numel() for p in m.parameters())
    n2 = sum(p.numel() for p in post.parameters()) if post else 0
    print(f" {n1 / 1e6:.2f} M + {n2 / 1e3:.0f} K parameters, {args.threads} threads")

    noisy, _ = sf.read(os.path.join(noisy_dir, args.clip + ".wav"),
                       dtype="float32")
    clean, _ = sf.read(os.path.join(CLIPS_DIR, "clean", args.clip + ".wav"),
                       dtype="float32")
    print(f"  clip: {args.clip}, {len(noisy) / SR:.1f} s, "
          f"input SI-SNR {si_snr(noisy, clean):.1f} dB")

    # -------------------------------------------------- 1. the noisy input
    if sd:
        play(sd, noisy, args.out, "NOISY input")

    # -------------------------------------------------- 2. the model, live
    N = stream.CHUNK_SAMPLES
    wav = np.concatenate([np.zeros(512, dtype=np.float32), noisy,
                          np.zeros(2048, dtype=np.float32)])
    s = stream.Streamer(m, post, dev)
    s.push(np.zeros(N, dtype=np.float32))            # warm the kernels
    s.reset()

    n_chunks = (len(wav) - N) // N + 1
    print(f"\n  processing {n_chunks} chunks of {N / 16:.0f} ms:")
    out, times = [], []
    t_all = time.perf_counter()
    for k, i in enumerate(range(0, len(wav) - N + 1, N)):
        t0 = time.perf_counter()
        out.append(s.push(wav[i:i + N]))
        dt = (time.perf_counter() - t0) * 1000.0
        times.append(dt)
        bar = "#" * int(min(dt, 600) / 15)
        flag = "" if dt <= N / 16 else "  (over the 100 ms deadline)"
        print(f"    chunk {k + 1:2d}/{n_chunks}  {dt:6.1f} ms  {bar}{flag}")
    total = time.perf_counter() - t_all
    enhanced = np.concatenate(out)

    audio_s = len(noisy) / SR
    rtf = audio_s / total
    print(f"\n  {audio_s:.1f} s of audio processed in {total:.1f} s")
    print(f"  = {rtf:.2f}x real time   "
          f"(median {np.median(times):.0f} ms per 100 ms chunk)")
    if rtf < 1:
        print(f"  this board is {1 / rtf:.1f}x too slow for live use; "
              f"the output is still exact")

    # -------------------------------------------------- 3. the result
    if sd:
        play(sd, enhanced, args.out, "ENHANCED output")

    # -------------------------------------------------- 4. the score
    o, c, lag = align(enhanced, clean)
    n = noisy[:len(c)]
    print(f"\n  SI-SNR   {si_snr(n, c):6.2f} dB  ->  {si_snr(o, c):6.2f} dB   "
          f"({si_snr(o, c) - si_snr(n, c):+.1f} dB)")
    try:
        from pesq import pesq
        from pystoi import stoi
        print(f"  PESQ     {pesq(SR, c, n, 'wb'):6.3f}     ->  "
              f"{pesq(SR, c, o, 'wb'):6.3f}")
        print(f"  STOI     {stoi(c, n, SR):6.3f}     ->  {stoi(c, o, SR):6.3f}")
    except ImportError:
        pass
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
