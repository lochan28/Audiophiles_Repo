"""
demo.py

A live bench for listening to the trained model work, and for seeing what it
is doing while it works.

    python scripts/demo.py
    python scripts/demo.py --port 8020

Three ways to feed it audio:

  * pick a clip from the frozen evaluation set, which is the only case where a
    CLEAN reference exists, so it is the only case that can be scored;
  * upload a file;
  * record from the microphone.

Uploads and recordings are decoded and resampled to 16 kHz mono *in the
browser* using WebAudio, and arrive here as raw float32 samples. That is
deliberate: it means this server needs no ffmpeg, no codec libraries and no
temporary files, and it accepts whatever the browser can play -- mp3, m4a,
ogg, webm from the microphone -- without knowing anything about formats.

Model work lives in demo_core.py. This file is the web layer only.
"""

import os
import re
import sys
import json
import time
import argparse
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
PAGE = os.path.join(HERE, "demo.html")
MIXTURE_DIR = os.path.join(PROJECT_ROOT, "generated_test_mixtures")

SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

# Enhanced audio is generated, not stored, so it is held here and handed out
# by token rather than written to disk. Bounded so a long session cannot grow
# without limit; the oldest entries simply fall out.
_AUDIO = {}
_AUDIO_ORDER = []
_AUDIO_LIMIT = 64
_LOCK = threading.Lock()


_AUDIO_SEQ = 0


def stash_wav(data):
    """Hold one generated WAV and return the URL that serves it.

    The counter is what makes the token unique, and it has to be. An earlier
    version used len(_AUDIO_ORDER) as the suffix, which stops changing the
    moment the cache reaches its limit -- from then on every stash within the
    same millisecond produced the SAME token, so all four or five clips of one
    analysis collapsed onto one entry and every player on the page returned
    whichever had been stored last. The symptom was that pressing play on the
    mixture played the enhanced audio.

    Generated inside the lock too, so two concurrent requests cannot read the
    same counter value before either increments it.
    """
    global _AUDIO_SEQ
    with _LOCK:
        _AUDIO_SEQ += 1
        token = f"{int(time.time() * 1000):x}{_AUDIO_SEQ:06x}"
        _AUDIO[token] = data
        _AUDIO_ORDER.append(token)
        while len(_AUDIO_ORDER) > _AUDIO_LIMIT:
            _AUDIO.pop(_AUDIO_ORDER.pop(0), None)
    return f"/audio/{token}.wav"


# --------------------------------------------------------------- analysis

def analyse(noisy, clean=None, run=None, meta=None):
    """Run the model and package everything the page needs to draw.

    `clean` is optional and only exists for frozen evaluation clips. Without
    it there is nothing to score against, so the metrics block is omitted
    rather than filled with guesses -- an uploaded clip has no ground truth
    and pretending otherwise would be the easiest way to mislead ourselves.
    """
    import demo_core as core
    from mixing import read_noise_classes

    enhanced, mask, logits, ms = core.enhance(noisy, run)
    # The optional second stage. None when no post-filter is installed, in
    # which case every block below simply omits it.
    filtered, post_ms = core.postfilter(noisy, mask)

    # One shared dB reference across all three pictures. Without this, a model
    # that only turned the volume down would look like it had removed noise.
    nb, T, F, peak = core.spectrogram(noisy)
    eb, *_ = core.spectrogram(enhanced, ref_peak=peak)
    spec = {"noisy": nb, "enhanced": eb}
    if clean is not None:
        spec["clean"], *_ = core.spectrogram(clean, ref_peak=peak)
    if filtered is not None:
        spec["postfiltered"], *_ = core.spectrogram(filtered, ref_peak=peak)

    mb, mT, mF = core.mask_picture(mask)

    out = {
        "meta": meta or {},
        "frames": T, "bins": F,
        "seconds": len(noisy) / core.SAMPLE_RATE,
        "spectrograms": spec,
        "mask": {"data": mb, "frames": mT, "bins": mF},
        "audio": {
            "noisy": stash_wav(core.wav_bytes(noisy)),
            "enhanced": stash_wav(core.wav_bytes(enhanced)),
        },
        "timing": {
            "inference_ms": ms,
            "audio_ms": 1000.0 * len(noisy) / core.SAMPLE_RATE,
            "realtime_factor": (1000.0 * len(noisy) / core.SAMPLE_RATE) / max(ms, 1e-6),
        },
        "model": core.get_model(run)[2],
    }

    # The latency evidence: run the model the way a live system would, one
    # 100 ms chunk at a time, and time every chunk against its deadline. A
    # single average hides the only question that matters for real time --
    # did EVERY chunk make it? -- so the page gets all of them.
    try:
        chunks, summary = core.stream_timing(noisy, run)
        out["stream"] = {"chunks": chunks, "summary": summary}
    except Exception as e:                       # never break the page for it
        out["stream"] = {"error": f"{type(e).__name__}: {e}"}
    if clean is not None:
        out["audio"]["clean"] = stash_wav(core.wav_bytes(clean))
    if filtered is not None:
        out["audio"]["postfiltered"] = stash_wav(core.wav_bytes(filtered))
        out["postfilter"] = dict(core.get_postfilter()[2], inference_ms=post_ms)
        out["timing"]["postfilter_ms"] = post_ms

    # What the auxiliary head thinks the noise is. It is never used to clean
    # anything -- it exists to force the encoder to learn noise identity --
    # but it is the one place the model says out loud what it thinks it is
    # hearing, which makes it worth showing.
    if logits is not None:
        classes = read_noise_classes()
        z = np.asarray(logits, dtype=np.float64)
        p = np.exp(z - z.max())
        p = p / p.sum()
        order = np.argsort(-p)
        out["aux"] = {
            "classes": classes,
            "probs": [float(x) for x in p],
            "predicted": classes[int(order[0])],
            "confidence": float(p[order[0]]),
        }

    if clean is not None:
        import evaluate as ev
        scores = {}
        signals = [("noisy", noisy), ("enhanced", enhanced)]
        if filtered is not None:
            signals.append(("postfiltered", filtered))
        for label, sig in signals:
            scores[label] = {
                "si_snr": ev.si_snr(sig, clean),
                "stoi": ev.safe_stoi(clean, sig),
                "pesq": ev.safe_pesq(clean, sig),
            }
        scores["gain"] = {
            k: (None if scores["enhanced"][k] is None or scores["noisy"][k] is None
                else scores["enhanced"][k] - scores["noisy"][k])
            for k in ("si_snr", "stoi", "pesq")
        }
        out["metrics"] = scores
    return out


# ----------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 60

    def log_message(self, *a):
        pass

    # ------------------------------------------------------------- send

    def send_bytes(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        self.send_bytes(json.dumps(obj, default=str).encode("utf-8"),
                        "application/json", code)

    def send_file(self, path, ctype):
        if not os.path.isfile(path):
            return self.send_json({"error": "not found"}, 404)
        with open(path, "rb") as f:
            self.send_bytes(f.read(), ctype)

    # -------------------------------------------------------------- get

    def do_GET(self):
        p = urlparse(self.path)
        route, q = unquote(p.path), parse_qs(p.query)
        try:
            return self.route_get(route, q)
        except Exception as e:
            return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def route_get(self, route, q):
        if route in ("/", "/index.html"):
            return self.send_file(PAGE, "text/html; charset=utf-8")

        if route == "/api/info":
            import demo_core as core
            return self.send_json({
                "model": core.get_model()[2],
                "runs": core.available_runs(),
                "splits": list(SPLITS),
            })

        if route == "/api/testset":
            return self.send_json({"clips": testset_index(
                q.get("split", ["testing"])[0])})

        if route == "/api/analyze/test":
            from mixing import FrozenMixtures
            split = q.get("split", ["testing"])[0]
            if split not in SPLITS:
                return self.send_json({"error": "bad split"}, 400)
            idx = int(q.get("index", ["0"])[0])
            run = q.get("run", [None])[0]
            if run is not None and not SAFE.match(run):
                return self.send_json({"error": "bad run"}, 400)
            if split == "demo":
                rows = demo_rows()
                if not 0 <= idx < len(rows):
                    return self.send_json({"error": "index out of range"}, 404)
                m = demo_describe(idx)
            else:
                ds = FrozenMixtures(split)
                if not 0 <= idx < len(ds):
                    return self.send_json({"error": "index out of range"}, 404)
                m = ds.describe(idx)
            meta = {"source": "testset", "split": split, "index": idx,
                    "category": m["category"], "snr_db": m["snr_db"],
                    "language": m["language"]}
            if split == "demo":
                meta.update(boundaries_s=m["boundaries_s"],
                            segments=m["segments"])
            return self.send_json(
                analyse(m["noisy"], m["clean"], run, meta))

        if route.startswith("/audio/"):
            token = route[len("/audio/"):].replace(".wav", "")
            with _LOCK:
                data = _AUDIO.get(token)
            if data is None:
                return self.send_json({"error": "expired"}, 404)
            return self.send_bytes(data, "audio/wav")

        return self.send_json({"error": f"no route {route}"}, 404)

    # ------------------------------------------------------------- post

    def do_POST(self):
        p = urlparse(self.path)
        route, q = unquote(p.path), parse_qs(p.query)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self.send_json({"error": "bad length"}, 400)
        if n > MAX_UPLOAD_BYTES:
            return self.send_json({"error": "too large"}, 413)
        raw = self.rfile.read(n) if n > 0 else b""

        try:
            if route == "/api/analyze/upload":
                import demo_core as core
                wav = np.frombuffer(raw, dtype="<f4").astype(np.float32)
                if wav.size == 0:
                    return self.send_json({"error": "no audio"}, 400)
                cap = int(core.MAX_INPUT_SECONDS * core.SAMPLE_RATE)
                truncated = wav.size > cap
                wav = np.ascontiguousarray(wav[:cap])
                run = q.get("run", [None])[0]
                if run is not None and not SAFE.match(run):
                    return self.send_json({"error": "bad run"}, 400)
                meta = {"source": q.get("source", ["upload"])[0],
                        "name": q.get("name", [""])[0][:120],
                        "truncated": truncated}
                return self.send_json(analyse(wav, None, run, meta))

            # Live microphone. Raw float32 in, raw float32 out -- no JSON and
            # no base64 around the audio, because this runs every 100 ms and
            # encoding overhead would eat the deadline. The reply may be
            # shorter than the request (or empty) while the stream fills its
            # hold-back window; the page appends whatever it gets.
            if route == "/api/stream":
                import demo_core as core
                token = q.get("token", [""])[0]
                if not token or not SAFE.match(token):
                    return self.send_json({"error": "bad token"}, 400)
                run = q.get("run", [None])[0]
                if run is not None and not SAFE.match(run):
                    return self.send_json({"error": "bad run"}, 400)
                if q.get("close", [""])[0] == "1":
                    core.stream_close(token)
                    return self.send_json({"ok": True})
                chunk = np.frombuffer(raw, dtype="<f4").astype(np.float32)
                out = core.stream_session(token, run).push(chunk)
                return self.send_bytes(
                    np.ascontiguousarray(out, dtype="<f4").tobytes(),
                    "application/octet-stream")
        except Exception as e:
            return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

        return self.send_json({"error": f"no route {route}"}, 404)


DEMO_DIR = os.path.join(PROJECT_ROOT, "generated_demo_clips")
SPLITS = ("testing", "validation", "demo")


def demo_rows():
    """The long presentation clips, if make_demo_clips.py has been run."""
    import csv
    path = os.path.join(DEMO_DIR, "demo_clips_manifest.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def demo_describe(idx):
    """Same shape FrozenMixtures.describe returns, for a demo clip.

    A demo clip is three held-out testing mixtures joined end to end, so it
    still has a true clean reference and can be scored with the same code.
    """
    import soundfile as sf
    rows = demo_rows()
    r = rows[idx]
    def rd(rel):
        w, _ = sf.read(os.path.join(DEMO_DIR, rel.replace("/", os.sep)),
                       dtype="float32")
        return w
    return {
        "noisy": rd(r["noisy_file"]), "clean": rd(r["clean_file"]),
        "category": r["categories"], "snr_db": r["snr_db"],
        "language": r["languages"], "index": int(r["index"]),
        "boundaries_s": [float(x) for x in r["boundaries_s"].split("|")],
        "segments": int(r["segments"]),
    }


def testset_index(split):
    """The frozen clips, with enough detail to choose one worth listening to."""
    import csv
    if split == "demo":
        return [{"index": int(r["index"]), "category": r["categories"],
                 "snr_db": r["snr_db"], "language": r["languages"],
                 "speaker_id": "", "seconds": float(r["seconds"]),
                 "segments": int(r["segments"])}
                for r in demo_rows()]
    path = os.path.join(MIXTURE_DIR, "test_mixtures_manifest.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{"index": int(r["index"]), "category": r["category"],
                 "snr_db": float(r["snr_db"]), "language": r["language"],
                 "speaker_id": r["speaker_id"]}
                for r in csv.DictReader(f) if r["split"] == split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8020)
    ap.add_argument("--run", default=None,
                    help="run to serve; default is the best by PESQ")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if args.run:
        import demo_core as core
        core.RUN_NAME = args.run

    print("loading the model (first import of torch is the slow part)...")
    import demo_core as core
    _, _, info = core.get_model()
    # Warm the kernels so the first clip a visitor tries is not 20x slower
    # than every one after it, which would make the timing panel a lie.
    core.enhance(np.zeros(core.SAMPLE_RATE, dtype=np.float32))

    print(f"serving       {info['run']} / {info['checkpoint']} "
          f"(epoch {info['epoch']}) on {info['device']}")
    print(f"latency       {info['latency_ms']:.0f} ms "
          f"({info['lookahead_ms']:.0f} ms lookahead + "
          f"{info['chunk_ms']:.0f} ms chunk)")

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); lan = s.getsockname()[0]; s.close()
    except Exception:
        lan = None
    url = f"http://localhost:{args.port}/"
    print(f"open          {url}")
    if lan:
        print(f"on your phone {f'http://{lan}:{args.port}/'}   (same wifi)")
        print("              the microphone needs HTTPS or localhost, so on a "
              "phone\n              use the test set and upload rather than "
              "recording")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
