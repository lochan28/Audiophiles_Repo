"""
selfcheck.py -- one command, one answer, on the Raspberry Pi.

Run this first, before anything else, on a board you have just set up:

    python3 selfcheck.py

It ends with a single line, GO or NO-GO, and if NO-GO it says which check
failed. Everything it needs is inside deploy/bundle; it makes no network
calls and takes no arguments.

WHAT IT DECIDES, AND WHY THAT IS THE GATE

Correctness is judged on SI-SNR alone. That is deliberate. SI-SNR is a dozen
lines of numpy, so it runs anywhere Python does, while PESQ is a C extension
with no aarch64 wheel -- it has to be compiled from source on the board, and
needs a toolchain and a network. Gating the whole check on PESQ would mean a
correctly working Pi reporting NO-GO because a scoring library would not
build, which is worse than not checking at all. PESQ and STOI are still
printed when they happen to be available, and still compared, but they never
decide the answer.

Speed is judged against the streaming path in stream.py, not the whole-clip
path. This matters more than it sounds. Processing a 3-second clip in 200 ms
looks like 15x real time and tells you almost nothing, because a live system
does not get whole clips -- it gets 100 ms at a time and has 100 ms to
answer. The number that decides deployability is the WORST chunk, not the
average: one chunk over deadline is a dropout a listener hears.
"""

import os
import sys
import time
import platform
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "bundle")
sys.path.insert(0, BUNDLE)

import torch
import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

RUN_DIR = os.path.join(BUNDLE, "runs", "0026_crn_comp03_spec600_la6_s0")
# The trained stage-1 run. Swapping this changes which model is checked.

POSTFILTER_WEIGHTS = os.path.join(BUNDLE, "models",
                                  "postfilter_ch48_la0_snr25.pt")
# The 128 K post-filter, not the 505 K one. A third of the compute for
# 1.978 PESQ against 1.983 -- a trade that is irrelevant on a desktop GPU
# and decisive on a Pi. Point this at the ch96 file to see what the wider
# filter costs on this board.

CLIPS_DIR = os.path.join(BUNDLE, "generated_test_mixtures", "testing")
# The bundled clips, with 'clean' and 'noisy' subfolders.

N_THREADS_TO_TEST = (1, 2, 4)
# Thread counts to benchmark. A Pi 5 has four cores; 1 shows what a single
# core manages, which is the honest floor if anything else is running.

SI_SNR_TOLERANCE = 0.05
# How far the board's SI-SNR may drift from the reference before this is
# called a failure, in dB. The project's measured run-to-run noise floor is
# 0.0218 dB, so this is a little over twice that: loose enough to absorb a
# different BLAS library and different float rounding on ARM, tight enough
# that a genuinely wrong model cannot slip through.

DEADLINE_MS = 100.0
# The budget one chunk of audio must be processed within. Equal to the chunk
# length, because a live system must finish each chunk before the next one
# has finished arriving.

# ═══════════════════════════════════════════════════════════════════
#  LOCKED -- reference numbers, measured on x86 with THIS bundle
# ═══════════════════════════════════════════════════════════════════
#
# These are not the project's headline figures. The headline numbers
# (PESQ 1.9826 / STOI 0.8660 / SI-SNR 14.019) come from all 125 held-out
# clips with the 505 K filter. These come from the 20 clips in this bundle
# with the 128 K filter, which is a different measurement and must have its
# own reference -- comparing against the headline numbers was the earlier
# mistake here, and produced deltas so large the check had to be switched
# off to pass.
REF_SI_SNR = 14.741
REF_PESQ = 2.0799
REF_STOI = 0.8614


def get_cpu_model():
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line or "Model" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    elif platform.system() == "Windows":
        try:
            out = subprocess.check_output(["wmic", "cpu", "get", "name"])
            return out.decode().strip().split("\n")[1].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def si_snr(est, ref):
    eps = 1e-8
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + eps) /
                               (np.sum(noise ** 2) + eps)))


def main():
    fails = []

    print(f"python      {platform.python_version()}")
    print(f"torch       {torch.__version__}")
    print(f"cpu         {get_cpu_model()}")
    print(f"cores       {os.cpu_count()}, torch default "
          f"{torch.get_num_threads()} threads")
    print("-" * 62)

    # ---------------------------------------------------------- imports
    try:
        import model_crn
        import postfilter
        import runlog
        import train
        import stream
        import soundfile as sf
    except ImportError as e:
        print(f"could not import the bundle: {e}")
        print("\nNO-GO: bundle import failed")
        return 1

    try:
        from pesq import pesq as pesq_score
        from pystoi import stoi as stoi_score
        extras = True
    except ImportError:
        pesq_score = stoi_score = None
        extras = False
        print("note        pesq/pystoi not installed -- SI-SNR still decides "
              "the answer,\n            PESQ and STOI are skipped. See "
              "RUNSHEET.md if you want them.")
        print("-" * 62)

    # ---------------------------------------------------------- self-tests
    print("self-tests...")
    for name, fn in (("model_crn", model_crn.self_test),
                     ("postfilter", postfilter.self_test)):
        try:
            import io as _io
            import contextlib
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                fn()
            print(f"  {name:<12} passed")
        except Exception as e:
            print(f"  {name:<12} FAILED: {e}")
            fails.append(f"{name} self-test")
    print("-" * 62)

    # ---------------------------------------------------------- load
    device = torch.device("cpu")
    try:
        run = runlog.RunLog(RUN_DIR)
        model, ck_name, epoch = train.load_run_model(run, device, "auto")
        model.eval()
        model.strip_aux()
        pf_model, _ = postfilter.load(POSTFILTER_WEIGHTS, device)
        pf_model.eval()
        print(f"loaded      {ck_name} epoch {epoch}, "
              f"{sum(p.numel() for p in model.parameters()):,} params")
        print(f"            {os.path.basename(POSTFILTER_WEIGHTS)}, "
              f"{sum(p.numel() for p in pf_model.parameters()):,} params")
    except Exception as e:
        print(f"could not load the models: {e}")
        print("\nNO-GO: model load failed")
        return 1
    # The streaming loop, checked against the whole-clip path on THIS
    # bundle's model rather than by stream.self_test(), which reaches for
    # repo modules that are deliberately not shipped here. A silent lead-in
    # first: the offline path reflect-pads the start of a clip and a live
    # stream has nothing to reflect, so without it the two disagree on the
    # opening frames and the LSTM carries that forward for ever.
    try:
        import soundfile as _sf
        raw, _ = _sf.read(os.path.join(CLIPS_DIR, "noisy",
                                       sorted(os.listdir(os.path.join(
                                           CLIPS_DIR, "noisy")))[0]),
                          dtype="float32")
        w = np.concatenate([np.zeros(512, dtype=np.float32), raw])
        xt = torch.from_numpy(w).unsqueeze(0)
        with torch.no_grad():
            _, mk, _, _ = model(xt)
            sp = model_crn.to_spectrogram(xt)
            off = model_crn.to_waveform(pf_model(sp * mk, sp),
                                        xt.shape[-1])[0].numpy()
        st = stream.Streamer(model, pf_model, device)
        live = np.concatenate([st.push(w[i:i + stream.CHUNK_SAMPLES])
                               for i in range(0, len(w),
                                              stream.CHUNK_SAMPLES)])
        n = min(len(live), len(off))
        err = float(np.abs(live[512:n] - off[512:n]).max())
        good = err < 1e-3
        print(f"  {'streaming':<12} {'passed' if good else 'FAILED'}  "
              f"(max difference from whole-clip path {err:.1e})")
        if not good:
            fails.append("streaming does not match the whole-clip path")
    except Exception as e:
        print(f"  {'streaming':<12} FAILED: {e}")
        fails.append("streaming check")

    print("-" * 62)

    # ---------------------------------------------------------- score
    noisy_dir = os.path.join(CLIPS_DIR, "noisy")
    clean_dir = os.path.join(CLIPS_DIR, "clean")
    if not (os.path.isdir(noisy_dir) and os.path.isdir(clean_dir)):
        print("NO-GO: the bundled clips are missing")
        return 1

    clips = sorted(f for f in os.listdir(noisy_dir) if f.endswith(".wav"))
    snrs, pesqs, stois = [], [], []
    for clip in clips:
        noisy, _ = sf.read(os.path.join(noisy_dir, clip), dtype="float32")
        clean, _ = sf.read(os.path.join(clean_dir, clip), dtype="float32")
        x = torch.from_numpy(noisy).unsqueeze(0)
        with torch.no_grad():
            _, mask, _, _ = model(x)
            out = postfilter.apply(pf_model, x, mask).squeeze(0).numpy()
        snrs.append(si_snr(out, clean))
        if extras:
            try:
                pesqs.append(pesq_score(16000, clean, out, "wb"))
            except Exception:
                pass
            try:
                stois.append(stoi_score(clean, out, 16000, extended=False))
            except Exception:
                pass

    avg_snr = float(np.mean(snrs))
    drift = avg_snr - REF_SI_SNR
    ok = abs(drift) <= SI_SNR_TOLERANCE
    print(f"scored      {len(clips)} clips")
    print(f"  SI-SNR    {avg_snr:7.3f}   reference {REF_SI_SNR:7.3f}   "
          f"drift {drift:+.3f} dB   {'ok' if ok else 'TOO FAR'}")
    if not ok:
        fails.append(f"SI-SNR drifted {drift:+.3f} dB from the reference")
    if pesqs:
        print(f"  PESQ      {np.mean(pesqs):7.4f}   reference {REF_PESQ:7.4f}"
              f"   drift {np.mean(pesqs) - REF_PESQ:+.4f}")
    if stois:
        print(f"  STOI      {np.mean(stois):7.4f}   reference {REF_STOI:7.4f}"
              f"   drift {np.mean(stois) - REF_STOI:+.4f}")
    print("-" * 62)

    # ---------------------------------------------------------- speed
    #
    # The streaming path, chunk by chunk, against the deadline. A silent
    # lead-in first because the offline path reflect-pads the start of a clip
    # and a live stream has nothing to reflect.
    print(f"streaming   {stream.CHUNK_FRAMES} frames "
          f"({stream.CHUNK_SAMPLES / 16.0:.0f} ms) per chunk, "
          f"deadline {DEADLINE_MS:.0f} ms")
    src, _ = sf.read(os.path.join(noisy_dir, clips[0]), dtype="float32")
    wav = np.concatenate([np.zeros(512, dtype=np.float32)] + [src] * 3)
    N = stream.CHUNK_SAMPLES
    best = None
    for threads in N_THREADS_TO_TEST:
        torch.set_num_threads(threads)
        s = stream.Streamer(model, pf_model, device)
        for i in range(0, 8000, N):                       # warm up untimed
            s.push(wav[i:i + N])
        times = []
        for i in range(8000, len(wav) - N, N):
            t0 = time.perf_counter()
            s.push(wav[i:i + N])
            times.append((time.perf_counter() - t0) * 1000.0)
        t = np.array(times)
        missed = int((t > DEADLINE_MS).sum())
        head = DEADLINE_MS / float(np.median(t))
        flag = "ok" if missed == 0 else f"{missed}/{len(t)} MISSED"
        print(f"  {threads} thread{'s' if threads > 1 else ' '}  "
              f"median {np.median(t):6.1f} ms   worst {t.max():6.1f} ms   "
              f"{head:4.1f}x headroom   {flag}")
        if missed == 0:
            best = threads
    if best is None:
        fails.append("every thread count missed the deadline")
    print("-" * 62)

    # ---------------------------------------------------------- verdict
    if fails:
        for f in fails:
            print(f"  failed: {f}")
        print("\nNO-GO")
        return 1
    print(f"real-time with {best} thread{'s' if best > 1 else ''}, "
          f"output matches the reference")
    if not extras:
        print("(PESQ and STOI were skipped; SI-SNR decided this.)")
    print("\nGO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
