"""
dashboard.py

A web view of the training runs. Start it and open the address it prints:

    python scripts/dashboard.py
    python scripts/dashboard.py --port 8080

It reads the run directories on every request, so a run that is training right
now and a run from three weeks ago are shown the same way. There is no
connection to the training process and nothing to start in a particular
order -- training writes files, this reads them. Kill either one and the other
carries on.

Standard library only: no Flask, no CDN, no build step. The page draws its own
charts as inline SVG, so it works with no network at all.

What it is for: comparing complete training curves across runs, whether that
is CRN against CRN or CRN against TCN, and showing exactly WHAT CHANGED
between any two of them -- both the settings and the actual source code, which
each run keeps a copy of.
"""

import os
import io
import re
import json
import argparse
import posixpath
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

import runlog
from runlog import RunLog, RUNS_DIR, SOURCE_DIR, config_diff, source_diff

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "dashboard.html")

MIME = {".html": "text/html; charset=utf-8", ".wav": "audio/wav",
        ".png": "image/png", ".json": "application/json",
        ".css": "text/css", ".js": "text/javascript"}

# The problem statement's targets, drawn as lines on the charts.
TARGETS = {"si_snr_enhanced": 15.0, "stoi_enhanced": 0.85,
           "pesq_enhanced": 2.5}

SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def run_summary(name):
    """Everything the run list needs, without sending the whole history."""
    run = RunLog(os.path.join(RUNS_DIR, name))
    cfg = run.config()
    hist = run.history()
    last = hist[-1] if hist else None

    best = {}
    for key, mode in (("val_loss", "min"), ("val_si_snr", "max"),
                      ("pesq_enhanced", "max"), ("stoi_enhanced", "max"),
                      ("si_snr_enhanced", "max")):
        b = run.best(key, mode)
        if b:
            best[key] = {"value": b[key], "epoch": b["epoch"]}

    return {
        "name": name,
        "arch": cfg.get("arch"),
        "sequence": cfg.get("sequence"),
        "started": cfg.get("started"),
        "note": run.notes(),
        "epochs": len(hist),
        "planned_epochs": cfg.get("epochs"),
        "parameters": cfg.get("parameters"),
        "latency_ms": cfg.get("total_latency_ms"),
        "lookahead_ms": cfg.get("lookahead_ms"),
        "git_commit": cfg.get("git_commit"),
        "changed_from": cfg.get("changed_from"),
        "changes": cfg.get("changes", {}),
        "source_changed": cfg.get("source_changed", {}),
        "last": last,
        "best": best,
        "running": run.looks_running(),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                                    # quiet; this is a local tool

    # ---------------------------------------------------------------- send

    def send_json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path):
        if not os.path.isfile(path):
            return self.send_json({"error": f"not found: {path}"}, 404)
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(ext, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ----------------------------------------------------------------- get

    def do_GET(self):
        parsed = urlparse(self.path)
        route = unquote(parsed.path)
        q = parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            return self.send_file(PAGE)

        if route == "/api/runs":
            names = []
            if os.path.isdir(RUNS_DIR):
                names = sorted((d for d in os.listdir(RUNS_DIR)
                                if os.path.isdir(os.path.join(RUNS_DIR, d))),
                               reverse=True)
            return self.send_json({"runs": [run_summary(n) for n in names],
                                   "targets": TARGETS})

        if route.startswith("/api/run/"):
            name = route[len("/api/run/"):]
            if not self._safe(name):
                return self.send_json({"error": "bad name"}, 400)
            run = RunLog(os.path.join(RUNS_DIR, name))
            return self.send_json({
                "name": name,
                "config": run.config(),
                "history": run.history(),
                "notes": run.notes(),
                "samples": run.sample_index(),
                "evaluations": run.evaluations(),
            })

        if route == "/api/compare":
            a, b = (q.get("a", [None])[0], q.get("b", [None])[0])
            if not (self._safe(a) and self._safe(b)):
                return self.send_json({"error": "bad name"}, 400)
            ra = RunLog(os.path.join(RUNS_DIR, a))
            rb = RunLog(os.path.join(RUNS_DIR, b))
            return self.send_json({
                "a": a, "b": b,
                "config_changes": config_diff(ra.config(), rb.config()),
                "source_changed": source_diff(
                    os.path.join(ra.path, SOURCE_DIR),
                    os.path.join(rb.path, SOURCE_DIR)),
            })

        if route == "/api/source":
            a, b = (q.get("a", [None])[0], q.get("b", [None])[0])
            fn = q.get("file", [None])[0]
            if not (self._safe(a) and self._safe(b) and self._safe(fn)):
                return self.send_json({"error": "bad name"}, 400)
            return self.send_json({
                "diff": runlog.unified_source_diff(a, b, fn)})

        if route.startswith("/files/"):
            rel = route[len("/files/"):]
            parts = [p for p in rel.split("/") if p]
            if len(parts) < 2 or not all(self._safe(p) for p in parts):
                return self.send_json({"error": "bad path"}, 400)
            path = os.path.join(RUNS_DIR, *parts)
            # never serve outside the runs directory
            if not os.path.abspath(path).startswith(os.path.abspath(RUNS_DIR)):
                return self.send_json({"error": "refused"}, 403)
            return self.send_file(path)

        self.send_json({"error": f"no route {route}"}, 404)

    @staticmethod
    def _safe(name):
        return bool(name) and bool(SAFE_NAME.match(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)
    n = len([d for d in os.listdir(RUNS_DIR)
             if os.path.isdir(os.path.join(RUNS_DIR, d))])
    url = f"http://localhost:{args.port}/"
    print(f"runs directory  {RUNS_DIR}")
    print(f"runs found      {n}")
    print(f"serving         {url}")
    print("\nReads the run files on every request, so a training run in "
          "progress\nupdates as it goes. Ctrl+C to stop.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
