"""
serve.py

One server, three pages. Start this instead of dashboard.py and demo.py:

    python scripts/serve.py
    python scripts/serve.py --port 8000

    /dashboard   training runs, curves, comparisons, the stop switch
    /bench       run the model on a test clip, an upload, or the microphone
    /lab         build a mixture by hand: one speaker, up to four noises

HOW IT COMPOSES THE OTHER TWO

dashboard.py and demo.py are NOT modified and still run standalone. This file
owns a route table saying which of them handles what, and hands the request to
that module's existing handler. Their route logic lives in one place still --
theirs -- so there is nothing here to keep in sync except the list of paths.

WHY TORCH IS NOT IMPORTED AT THE TOP

dashboard.py is deliberately stdlib-only and starts instantly; demo.py pulls in
torch and a checkpoint, which costs seconds. Importing both eagerly would make
opening the runs page wait for a model nobody asked for. The demo and lab
modules are therefore imported inside the handlers, on first use, so this
server starts as fast as the dashboard alone and only pays for the model when
someone actually opens the bench or the lab.
"""

import os
import re
import json
import threading
import argparse
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

import dashboard
import demo

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = {
    "/dashboard": os.path.join(HERE, "dashboard.html"),
    "/bench": os.path.join(HERE, "demo.html"),
    "/lab": os.path.join(HERE, "lab.html"),
}
DEFAULT_PAGE = "/dashboard"

SAFE = re.compile(r"^[A-Za-z0-9._-]+$")
MAX_BODY = 64 * 1024 * 1024

# Which module answers which path. Kept explicit rather than guessed from a
# prefix, so adding a route to either module is a deliberate line here and a
# typo cannot silently shadow the other server's API.
DASHBOARD_GET = ("/api/runs", "/api/compare", "/api/source")
DASHBOARD_GET_PREFIX = ("/api/run/", "/files/")
DASHBOARD_POST = ("/api/stop", "/api/stop/cancel")

# Uploaded lab audio, held in memory by token. Kept OUT of the mix request so
# the SNR sliders can be re-tried without sending the file again -- which is
# the difference between a lab you can iterate in and one you cannot.
_UPLOADS = {}
_UPLOAD_ORDER = []
_UPLOAD_LIMIT = 8
_UPLOAD_LOCK = threading.Lock()
_UPLOAD_SEQ = 0

DEMO_GET = ("/api/info", "/api/testset", "/api/analyze/test")
DEMO_GET_PREFIX = ("/audio/",)
DEMO_POST = ("/api/analyze/upload", "/api/stream")


class Handler(BaseHTTPRequestHandler):
    # Threaded and HTTP/1.1, for the reason recorded in dashboard.py: a
    # single-threaded server stalls every other client while one browser
    # holds a connection open, which made the page take over a minute to
    # load on a phone.
    protocol_version = "HTTP/1.1"
    timeout = 60

    def log_message(self, *a):
        pass

    # ---- the helper set both borrowed handlers expect ----------------
    #
    # dashboard.Handler calls send_file(path); demo.Handler calls
    # send_file(path, ctype). One optional argument satisfies both.

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

    def send_file(self, path, ctype=None):
        if not os.path.isfile(path):
            return self.send_json({"error": f"not found: {path}"}, 404)
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb") as f:
            self.send_bytes(f.read(), ctype or dashboard.MIME.get(
                ext, "application/octet-stream"))

    def read_body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return b""
        return self.rfile.read(min(n, MAX_BODY)) if n > 0 else b""

    @staticmethod
    def _safe(name):
        return bool(name) and bool(SAFE.match(name))

    # ---- dispatch ----------------------------------------------------

    def do_GET(self):
        p = urlparse(self.path)
        route, q = unquote(p.path), parse_qs(p.query)

        if route in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", DEFAULT_PAGE)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if route in PAGES:
            return self.send_file(PAGES[route], "text/html; charset=utf-8")

        if route.startswith("/api/lab/"):
            return self.lab_get(route, q)

        if route in DEMO_GET or route.startswith(DEMO_GET_PREFIX):
            return demo.Handler.route_get(self, route, q)
        if route in DASHBOARD_GET or route.startswith(DASHBOARD_GET_PREFIX):
            return dashboard.Handler.do_GET(self)

        self.send_json({"error": f"no route {route}"}, 404)

    def do_POST(self):
        route = unquote(urlparse(self.path).path)
        if route.startswith("/api/lab/"):
            return self.lab_post(route, parse_qs(urlparse(self.path).query))
        if route in DEMO_POST:
            return demo.Handler.do_POST(self)
        if route in DASHBOARD_POST:
            return dashboard.Handler.do_POST(self)
        self.read_body()                    # drain, or keep-alive desyncs
        self.send_json({"error": f"no route {route}"}, 404)

    # ---- the lab -----------------------------------------------------

    def lab_get(self, route, q):
        import lab_core
        if route == "/api/lab/sources":
            split = q.get("split", ["testing"])[0]
            if split not in lab_core.SPLITS:
                return self.send_json({"error": "bad split"}, 400)
            return self.send_json({
                "split": split,
                "speech": lab_core.speech_sources(split),
                "noise": lab_core.noise_sources(split),
                "max_noises": lab_core.MAX_NOISES,
                "snr_range": lab_core.SNR_RANGE,
                "default_snr": lab_core.DEFAULT_SNR_DB,
            })
        return self.send_json({"error": f"no route {route}"}, 404)

    def lab_post(self, route, q):
        import lab_core
        raw = self.read_body()

        if route == "/api/lab/upload":
            global _UPLOAD_SEQ
            import numpy as np
            wav = np.frombuffer(raw, dtype="<f4").astype(np.float32)
            if wav.size == 0:
                return self.send_json({"error": "no audio"}, 400)
            cap = int(lab_core.MAX_UPLOAD_SECONDS * lab_core.TARGET_SR)
            truncated = wav.size > cap
            wav = np.ascontiguousarray(wav[:cap])
            name = q.get("name", [""])[0][:120] or "uploaded audio"
            with _UPLOAD_LOCK:
                _UPLOAD_SEQ += 1
                token = f"u{_UPLOAD_SEQ:06x}"
                _UPLOADS[token] = (wav, name)
                _UPLOAD_ORDER.append(token)
                while len(_UPLOAD_ORDER) > _UPLOAD_LIMIT:
                    _UPLOADS.pop(_UPLOAD_ORDER.pop(0), None)
            return self.send_json({
                "token": token, "name": name,
                "seconds": wav.size / lab_core.TARGET_SR,
                "truncated": truncated,
                "max_seconds": lab_core.MAX_UPLOAD_SECONDS,
            })

        if route != "/api/lab/mix":
            return self.send_json({"error": f"no route {route}"}, 404)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return self.send_json({"error": "bad json"}, 400)

        speech = body.get("speech")
        upload = body.get("speech_upload")
        layers = body.get("layers") or []
        run = body.get("run")
        if not speech and not upload:
            return self.send_json({"error": "no speech clip chosen"}, 400)
        if run is not None and not self._safe(run):
            return self.send_json({"error": "bad run"}, 400)

        wav = name = None
        if upload:
            with _UPLOAD_LOCK:
                held = _UPLOADS.get(upload)
            if held is None:
                return self.send_json(
                    {"error": "that upload has expired - send the file again"},
                    404)
            wav, name = held

        try:
            mix = lab_core.build(speech, layers,
                                 speech_offset=body.get("speech_offset"),
                                 speech_wav=wav, speech_name=name)
        except (ValueError, FileNotFoundError) as e:
            return self.send_json({"error": str(e)}, 400)

        meta = {
            "source": "lab",
            "speech_file": mix["speech_file"],
            "layers": mix["layers"],
            "combined_snr_db": mix["combined_snr_db"],
            "clipped": mix["clipped"],
            # The bench shows meta.category when it has one. The lab often has
            # several noises at once, so there is no single true answer for
            # the classifier to be right about; it is left unset rather than
            # picking one arbitrarily and marking the model wrong for it.
            "category": (mix["layers"][0]["category"]
                         if len(mix["layers"]) == 1 else None),
        }
        out = demo.analyse(mix["noisy"], mix["clean"], run, meta)

        import demo_core as core
        out["audio"]["noise_only"] = demo.stash_wav(
            core.wav_bytes(mix["noise_only"]))
        out["lab"] = {
            "layers": mix["layers"],
            "combined_snr_db": mix["combined_snr_db"],
            "clipped": mix["clipped"],
            "speech_span": mix["speech_span"],
            "seconds": mix["seconds"],
            "truncated": mix.get("truncated", False),
            "uploaded": bool(upload),
        }
        return self.send_json(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--run", default=None,
                    help="model for the bench and lab; default is best by PESQ")
    ap.add_argument("--warm", action="store_true",
                    help="load the model at startup instead of on first use")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if args.run:
        import demo_core as core
        core.RUN_NAME = args.run
    if args.warm:
        import demo_core as core
        print("loading the model...")
        _, _, info = core.get_model()
        print(f"  {info['run']} / {info['checkpoint']} on {info['device']}")

    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); lan = s.getsockname()[0]; s.close()
    except Exception:
        lan = None

    base = f"http://localhost:{args.port}"
    print(f"runs        {base}/dashboard")
    print(f"bench       {base}/bench")
    print(f"lab         {base}/lab")
    if lan:
        print(f"on a phone  http://{lan}:{args.port}/   (same wifi)")
    if not args.no_browser:
        try:
            webbrowser.open(base + DEFAULT_PAGE)
        except Exception:
            pass

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
