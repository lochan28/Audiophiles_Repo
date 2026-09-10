"""
demo_core.py

The model-facing half of the live demo: pick a trained run, load it, enhance
audio, and turn spectrograms into something a browser can draw.

Kept separate from demo.py (the web server) for one reason: this half imports
torch and takes seconds to start, while the server half is stdlib-only and
trivial. Splitting them keeps the server file readable and lets this half be
exercised on its own:

    python scripts/demo_core.py            # self-test on a frozen test clip

Everything here works on plain numpy float32 at 16 kHz mono, which is what the
rest of the project uses.
"""

import os
import io
import json
import time
import base64

import numpy as np
import threading

import torch

import model_crn
import train
from runlog import RunLog, RUNS_DIR
import mixing


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS — safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

RUN_NAME = None
# Which trained run the demo serves. None picks the best available run by its
# peak PESQ, so the demo follows your best model without editing anything.
# Set a run directory name (e.g. "0023_crn_comp03_spec600_la10_s0") to pin one,
# which is what you want when demonstrating a specific result.

CHECKPOINT = "auto"
# "auto" walks train.CHECKPOINT_PREFERENCE (best_pesq, then last, then best).
# best_pesq is deliberately first: best.pt is chosen by validation LOSS, and
# loss and PESQ diverge by over 0.1 PESQ. Override with "best_pesq" / "last" /
# "best" to hear what different checkpoints of the same run sound like.

POSTFILTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "models", "postfilter_ch96_la0_snr10.pt")
# The optional SECOND STAGE. If this file is missing the bench simply does not
# offer it, so nothing here depends on the post-filter existing.
#
# A small strictly-causal filter that predicts a residual to ADD to the mask's
# output -- see scripts/postfilter.py. Measured on the held-out testing split
# with run 0026 frozen:
#
#                     PESQ      STOI    SI-SNR    latency
#   stage 1 alone   1.9343    0.8626    13.623    176.6 ms
#   + ch48          1.9775    0.8646    13.966    176.7 ms
#   + ch96          1.9826    0.8660    14.019    177.0 ms   <- served
#
# The narrower one is kept beside it because it is a third of the compute for
# most of the gain, which is the trade that matters if this ever runs on the
# Jetson rather than a 5060.
#
# Set to None to serve stage 1 alone, which is what every number recorded
# before the post-filter refers to.

DB_FLOOR = -80.0
# The quietest thing drawn on a spectrogram, in dB below that clip's loudest
# point. LOWER (-100) reveals faint noise floors and reverb tails but makes the
# picture washy; HIGHER (-60) gives punchy contrast that hides quiet detail.
# This only affects the PICTURE, never the audio and never the metrics.

MAX_INPUT_SECONDS = 30.0
# Longest clip accepted from an upload or a microphone recording. Guards
# against someone dropping in a podcast and waiting forever. The model has no
# length limit of its own -- it is convolutional plus an LSTM -- so raising
# this costs memory and time, nothing else.


# ═══════════════════════════════════════════════════════════════════
#  LOCKED — set by the data and the trained models, do not change
# ═══════════════════════════════════════════════════════════════════

SAMPLE_RATE = model_crn.SAMPLE_RATE          # 16 kHz, what every model saw
EPS = 1e-12

_CACHE = {}


# --------------------------------------------------------------- choosing

def peak_pesq(run_dir):
    """Best PESQ this run ever reached, or None if it was never evaluated."""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    best = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = row.get("pesq_enhanced")
            if v is not None and (best is None or v > best):
                best = v
    return best


def available_runs():
    """Every run with a loadable checkpoint, best PESQ first.

    The demo offers this list so two models can be compared on the same audio,
    which is far more convincing than two numbers in a table.
    """
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    for name in sorted(os.listdir(RUNS_DIR)):
        d = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(d):
            continue
        ck = os.path.join(d, "checkpoints")
        if not os.path.isdir(ck) or not any(
                f.endswith(".pt") for f in os.listdir(ck)):
            continue
        run = RunLog(d)
        cfg = run.config()
        out.append({
            "name": name,
            "pesq": peak_pesq(d),
            # Which SNR distribution this run was trained and scored on.
            # A peak PESQ from the old flat U(-5,+15) exam is not
            # comparable with one from the current 90/10 exam -- the change
            # alone is worth about +0.19 PESQ -- so the sort below groups
            # by profile first and only then by score.
            "snr_profile": cfg.get("snr_profile", "unrecorded"),
            "current_profile": cfg.get("snr_profile") == mixing.SNR_PROFILE,
            "parameters": cfg.get("parameters"),
            "latency_ms": cfg.get("total_latency_ms"),
            "lookahead_ms": cfg.get("lookahead_ms"),
            "lstm_hidden": cfg.get("lstm_hidden"),
            "note": run.notes()[:200],
        })
    # Runs on the CURRENT distribution first, then by score within each
    # group. Ranking the two together would let an old run outscore a new
    # one purely because it sat an easier or harder exam.
    out.sort(key=lambda r: (not r["current_profile"],
                            r["pesq"] is None, -(r["pesq"] or 0)))
    return out


def best_run_name():
    runs = available_runs()
    if not runs:
        raise RuntimeError("no trained runs with checkpoints under runs/")
    return runs[0]["name"]


# --------------------------------------------------------------- loading

def stage1_for_postfilter():
    """The run the installed post-filter was TRAINED on, if there is one.

    This matters more than it looks. best_run_name() ranks by peak PESQ, and
    runs 0026, 0027 and 0028 are seed replicates within 0.01 PESQ of each
    other, so it happily returned 0028 -- while the post-filter had learned to
    correct 0026's particular mistakes. A second stage is fitted to ONE first
    stage; pairing it with a different one silently serves a mismatched system
    and every number on the page becomes a number for something we never
    measured.
    """
    loaded = get_postfilter()
    if loaded is None:
        return None
    name = loaded[2].get("trained_on")
    return name if name and os.path.isdir(os.path.join(RUNS_DIR, name)) else None


def get_model(run_name=None):
    """Load (and cache) a run's model. Returns (model, device, info)."""
    run_name = (run_name or RUN_NAME or stage1_for_postfilter()
                or best_run_name())
    if run_name in _CACHE:
        return _CACHE[run_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = RunLog(os.path.join(RUNS_DIR, run_name))
    model, ck_name, epoch = train.load_run_model(run, device, CHECKPOINT)
    cfg = run.config()

    info = {
        "run": run_name,
        "checkpoint": ck_name,
        "epoch": epoch,
        "device": str(device),
        "parameters": cfg.get("parameters"),
        "lookahead_frames": cfg.get("lookahead_frames"),
        "lookahead_ms": cfg.get("lookahead_ms"),
        "chunk_ms": cfg.get("chunk_ms"),
        "latency_ms": cfg.get("total_latency_ms"),
        "lstm_hidden": cfg.get("lstm_hidden"),
        "sample_rate": SAMPLE_RATE,
        "n_fft": model_crn.N_FFT,
        "hop_length": model_crn.HOP_LENGTH,
        "peak_pesq": peak_pesq(run.path),
        "note": run.notes(),
    }
    _CACHE[run_name] = (model, device, info)
    return _CACHE[run_name]


def get_postfilter():
    """Load (and cache) the second stage, or None if none is installed."""
    if "__post__" in _CACHE:
        return _CACHE["__post__"]
    loaded = None
    if POSTFILTER_PATH and os.path.exists(POSTFILTER_PATH):
        import postfilter as pf
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta = pf.load(POSTFILTER_PATH, device)
        model.eval()
        loaded = (model, device, {
            "parameters": sum(p.numel() for p in model.parameters()),
            "lookahead_frames": model.lookahead,
            "past_frames": model.past,
            "added_latency_ms": model.lookahead * 10,
            "trained_on": meta.get("stage1"),
        })
    _CACHE["__post__"] = loaded
    return loaded


# ---------------------------------------------------------------- audio

def postfilter(wav, mask):
    """Stage 1's mask and the original noisy audio -> corrected waveform.

    Takes the MASK rather than the enhanced waveform so stage 1 does not have
    to run twice; the caller already has it from enhance().
    Returns (waveform, milliseconds), or (None, None) when no filter exists.
    """
    loaded = get_postfilter()
    if loaded is None:
        return None, None
    import postfilter as pf
    model, device, _ = loaded
    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    x = x.unsqueeze(0).to(device)
    m = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
    m = m.unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = pf.apply(model, x, m)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out[0].cpu().numpy(), (time.perf_counter() - t0) * 1000.0


def enhance(wav, run_name=None):
    """Noisy float32 mono at 16 kHz -> (enhanced, mask, logits, ms taken)."""
    model, device, _ = get_model(run_name)
    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    x = x.unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out, mask, logits, _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    return (out[0].cpu().numpy(), mask[0].cpu().numpy(),
            None if logits is None else logits[0].cpu().numpy(), ms)


def stream_timing(wav, run_name=None, chunk_ms=100.0):
    """Run the model the way a live system would: one 100 ms chunk at a time.

    Returns (per-chunk timings, summary).

    This is a TIMING measurement, not a second copy of the audio. The waveform
    the page plays still comes from the whole-clip path. Per-chunk compute is
    identical either way -- same tensors, same arithmetic -- but a deployed
    post-filter would also keep a ring buffer of past frames that this loop
    does not, so audio produced here would differ slightly at chunk
    boundaries. Measuring speed with it is honest; playing it would not be.

    The point of it is that a single average hides the question that actually
    matters for a real-time system: did EVERY chunk finish before its
    deadline? One chunk over 100 ms is a dropout a listener hears.
    """
    model, device, info = get_model(run_name)
    pf_loaded = get_postfilter()
    n = int(SAMPLE_RATE * chunk_ms / 1000.0)
    chunks = [wav[i:i + n] for i in range(0, len(wav), n)]
    chunks = [c for c in chunks if len(c) == n]     # drop a short final tail
    if not chunks:
        return [], {"chunks": 0}

    # Warm up first. Without this the opening chunk is many times slower than
    # the rest -- lazy kernel selection, not system behaviour -- and the chart
    # would lead with a spike that means nothing.
    import postfilter as pf
    warm = torch.zeros((1, n), dtype=torch.float32, device=device)
    state = None
    with torch.no_grad():
        for _ in range(8):
            _, wmask, _, state = model(warm, state)
            # The post-filter needs warming too. Warming only stage 1 left the
            # first real chunk at 200 ms -- lazy kernel selection inside the
            # filter, reported as if the system had stalled.
            if pf_loaded is not None:
                pf.apply(pf_loaded[0], warm, wmask)
    if device.type == "cuda":
        torch.cuda.synchronize()

    rows, state = [], None
    for c in chunks:
        x = torch.from_numpy(np.ascontiguousarray(c, dtype=np.float32))
        x = x.unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _, mask, _, state = model(x, state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        post_ms = 0.0
        if pf_loaded is not None:
            with torch.no_grad():
                pf.apply(pf_loaded[0], x, mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            post_ms = (time.perf_counter() - t1) * 1000.0

        stage1_ms = (t1 - t0) * 1000.0
        rows.append({"stage1_ms": stage1_ms, "postfilter_ms": post_ms,
                     "total_ms": stage1_ms + post_ms})

    totals = [r["total_ms"] for r in rows]
    worst = max(totals)
    return rows, {
        "chunks": len(rows),
        "chunk_ms": chunk_ms,
        "deadline_ms": chunk_ms,
        "worst_ms": worst,
        "median_ms": float(np.median(totals)),
        "mean_ms": float(np.mean(totals)),
        "headroom_pct": 100.0 * (1.0 - worst / chunk_ms),
        "missed": sum(1 for t in totals if t > chunk_ms),
        "realtime_factor": chunk_ms / max(float(np.mean(totals)), 1e-9),
        "lookahead_ms": info.get("lookahead_ms"),
        "total_latency_ms": info.get("latency_ms"),
        "device": info.get("device"),
    }



# ---------------------------------------------------------------- live stream
#
# TUNABLE -- live streaming
#
# CONTEXT_MS   how much past audio is re-processed ahead of each new chunk.
#              Both stages have memory: stage 1 an LSTM, the post-filter a
#              63-frame receptive field. Feeding independent chunks would
#              click at every boundary. Raising this removes boundary error
#              but costs compute linearly; below ~400 ms the LSTM has not
#              re-warmed and the first frames of each chunk are wrong.
# SESSION_IDLE_S  a browser that closes mid-stream never says so; sessions
#              older than this are dropped so memory does not grow.
# CHUNK_MS     how much audio the browser sends per request. Compute per call
#              is ~40 ms almost regardless of this (the context dominates),
#              so this is really the headroom dial: 50 ms missed its deadline
#              4 times in 50, 100 ms leaves 2.5x margin. Total added latency
#              is CHUNK_MS + HOLD_BACK, so 228 ms at the default.
CHUNK_MS = 100.0
CONTEXT_MS = 2560.0
SESSION_IDLE_S = 60.0

# LOCKED -- samples held back before a chunk is trusted. Set by the STFT
# window plus the model's 6-frame lookahead, not a free choice. Measured:
# 1024 still leaves 7e-4 of error, 2048 leaves 1e-8. That is 128 ms, which
# together with a 50 ms capture chunk keeps the live path inside the
# project's 200-250 ms budget.
HOLD_BACK = 2048


class StreamSession:
    """One live microphone connection.

    Holds the tail of the audio already seen. Each push re-runs the whole
    two-stage path on (context + new chunk) and returns only the new part,
    so the output is sample-for-sample what the offline path would produce,
    with no state to get wrong.
    """

    def __init__(self, run_name=None):
        self.run_name = run_name
        self.buf = np.zeros(0, dtype=np.float32)   # recent input, absolute
        self.base = 0                              # stream index of buf[0]
        self.pos = 0                               # input samples consumed
        self.emitted = 0                           # output samples emitted
        self.n_context = int(SAMPLE_RATE * CONTEXT_MS / 1000.0)
        self.touched = time.time()

    def push(self, chunk):
        """New input samples in, finished output samples out (may be empty).

        HOLD_BACK is why this is not just "process and take the tail". The
        inverse STFT builds each output sample by summing overlapping
        windows, so the last window's worth of a buffer is still missing the
        contributions of frames that have not arrived. Emitting it produced
        an error at EVERY chunk boundary, not just the first. Holding one
        window back costs 32 ms of delay and makes the live output match the
        offline path to within 1e-7.
        """
        self.touched = time.time()
        chunk = np.asarray(chunk, dtype=np.float32)
        self.buf = np.concatenate([self.buf, chunk])
        self.pos += len(chunk)

        safe_end = self.pos - HOLD_BACK                # last trustworthy sample
        if safe_end <= self.emitted:
            return np.zeros(0, dtype=np.float32)

        out, _, _, _ = enhance(self.buf, self.run_name)
        piece = out[self.emitted - self.base:safe_end - self.base]
        self.emitted = safe_end

        # Drop context we will never need again, keeping base honest.
        keep = self.n_context + HOLD_BACK
        if len(self.buf) > keep:
            drop = len(self.buf) - keep
            self.buf = self.buf[drop:].copy()
            self.base += drop
        return piece.astype(np.float32)


_SESSIONS = {}
_SESSION_LOCK = threading.Lock()


def stream_session(token, run_name=None):
    """Fetch or create the session for a token, reaping stale ones."""
    now = time.time()
    with _SESSION_LOCK:
        for k in [k for k, v in _SESSIONS.items()
                  if now - v.touched > SESSION_IDLE_S]:
            _SESSIONS.pop(k, None)
        s = _SESSIONS.get(token)
        if s is None or s.run_name != run_name:
            s = _SESSIONS[token] = StreamSession(run_name)
        return s


def stream_close(token):
    with _SESSION_LOCK:
        _SESSIONS.pop(token, None)


def spectrogram(wav, ref_peak=None):
    """Spectrogram as a uint8 picture the browser can draw directly.

    Returns (base64 string, n_frames, n_bins, ref_peak).

    The reference peak is returned so the CALLER can reuse it across the
    noisy, clean and enhanced versions of the same clip. That matters: if each
    picture normalised to its own loudest point, a model that merely turned
    everything down would look like it had removed noise. One shared scale
    makes the three panels honestly comparable.
    """
    x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    mag = model_crn.to_spectrogram(x.unsqueeze(0)).abs()[0]       # (T, F)
    mag = mag.cpu().numpy()
    peak = float(ref_peak if ref_peak else max(mag.max(), EPS))
    db = 20.0 * np.log10(mag / peak + EPS)
    img = np.clip((db - DB_FLOOR) / (0.0 - DB_FLOOR), 0.0, 1.0)
    return (base64.b64encode((img * 255).astype(np.uint8).tobytes()).decode(),
            int(img.shape[0]), int(img.shape[1]), peak)


def mask_picture(mask):
    """The predicted mask, 0..MASK_MAX, as the same kind of uint8 picture."""
    m = np.clip(mask / max(model_crn.MASK_MAX, EPS), 0.0, 1.0)
    return (base64.b64encode((m * 255).astype(np.uint8).tobytes()).decode(),
            int(m.shape[0]), int(m.shape[1]))


def wav_bytes(wav, sample_rate=SAMPLE_RATE):
    """Float32 mono -> a 16-bit WAV file in memory, ready to serve."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(wav, -1.0, 1.0), sample_rate,
             subtype="PCM_16", format="WAV")
    return buf.getvalue()


# ------------------------------------------------------------- self-test

def self_test():
    from mixing import FrozenMixtures
    import evaluate as ev

    print("--- picking a model ---")
    for r in available_runs()[:5]:
        print(f"  {r['name']:<36} PESQ {r['pesq']}")
    print(f"  -> best is {best_run_name()}\n")

    model, device, info = get_model()
    print("--- loaded ---")
    for k in ("run", "checkpoint", "epoch", "device", "parameters",
              "lookahead_ms", "latency_ms", "peak_pesq"):
        print(f"  {k:<16} {info[k]}")

    print("\n--- enhancing one frozen TEST clip ---")
    ds = FrozenMixtures("testing")
    m = ds.describe(0)
    out, mask, logits, ms = enhance(m["noisy"])
    cat_str = ", ".join(m["category"]) if isinstance(m["category"], (list, tuple)) else m["category"]
    print(f"  clip 0, category {cat_str}, {m['snr_db']:.1f} dB SNR")
    print(f"  in {m['noisy'].shape} -> out {out.shape}, took {ms:.1f} ms")
    for label, sig in (("noisy", m["noisy"]), ("enhanced", out)):
        print(f"  {label:<9} SI-SNR {ev.si_snr(sig, m['clean']):6.2f}  "
              f"STOI {ev.safe_stoi(m['clean'], sig):.3f}  "
              f"PESQ {ev.safe_pesq(m['clean'], sig):.3f}")

    print("\n--- pictures ---")
    b64, T, F, peak = spectrogram(m["noisy"])
    print(f"  noisy spectrogram {T} frames x {F} bins, "
          f"{len(b64)} b64 chars, peak {peak:.4f}")
    b64e, *_ = spectrogram(out, ref_peak=peak)
    print(f"  enhanced drawn on the SAME scale ({len(b64e)} chars)")
    mb, mT, mF = mask_picture(mask)
    print(f"  mask {mT} x {mF}")
    print(f"  wav bytes {len(wav_bytes(out)):,}")

    print("\n--- a non-3-second clip (uploads are any length) ---")
    odd = np.concatenate([m["noisy"], m["noisy"][:8000]])          # 4.5 s
    out2, _, _, ms2 = enhance(odd)
    print(f"  in {odd.shape} -> out {out2.shape} in {ms2:.1f} ms  "
          f"(lengths match: {odd.shape == out2.shape})")

    print("\nall checks passed")


if __name__ == "__main__":
    self_test()
