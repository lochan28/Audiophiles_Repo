"""
runlog.py

The record of a training run. Both the CRN and the TCN write through this, and
the dashboard reads it back, so neither ever needs to know what the other is.

The whole design rests on one rule: **training writes, nothing reads back.**
There is no server to connect to and no process to keep alive. A run appends
lines to a file, and anything that wants to watch reads that file. Which means
a run that crashes at epoch 40 still has 40 epochs of history on disk, and a
run from three weeks ago is viewed exactly like a live one.

Layout, one directory per run:

    runs/0007_crn_h256_la2/
        config.json       every knob, the git commit, the measured latency
        metrics.jsonl     one JSON object per epoch, appended, never rewritten
        checkpoints/      best.pt and last.pt
        samples/          audio and mask images at evaluation points
        notes.md          why you ran it (optional, written by hand)

`metrics.jsonl` is append-only and flushed to disk on every write. That is
deliberate: a rewritten file can be truncated by a crash mid-write, and losing
the whole history to a power cut at epoch 60 would be worse than any amount of
disk saved.

Nothing here imports a model. The caller hands over a plain dictionary, so
adding the TCN needs no change to this file.
"""

import os
import re
import io
import csv
import json
import time
import shutil
import subprocess
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

CONFIG_NAME = "config.json"
METRICS_NAME = "metrics.jsonl"
NOTES_NAME = "notes.md"
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"
SOURCE_DIR = "source"
STOP_NAME = "STOP"          # asks a running trainer to finish early; see below

# Copied into every run. Comparing two runs then means diffing two snapshots
# rather than trusting that the working tree has not moved since -- which it
# usually has, because the whole point of a second run is that something
# changed. A git commit alone is not enough: runs are routinely launched with
# uncommitted edits, and that is exactly when knowing the code matters most.
SOURCE_FILES = ("model_crn.py", "model_tcn.py", "train.py", "mixing.py",
                "runlog.py", "evaluate.py")


def git_commit():
    """The commit the run was launched from, so a result can be traced back."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True,
                             timeout=10)
        commit = out.stdout.strip() if out.returncode == 0 else None
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               cwd=PROJECT_ROOT, capture_output=True, text=True,
                               timeout=10)
        if commit and dirty.stdout.strip():
            commit += "-dirty"          # uncommitted edits: not reproducible
        return commit
    except Exception:
        return None


def next_sequence():
    """Lowest unused run number. Sequential so runs can be named out loud."""
    if not os.path.isdir(RUNS_DIR):
        return 1
    used = []
    for name in os.listdir(RUNS_DIR):
        head = name.split("_", 1)[0]
        if head.isdigit():
            used.append(int(head))
    return max(used) + 1 if used else 1


def build_tag(config, keys=("lstm_hidden", "lookahead_frames")):
    """A short label from the knobs actually being varied.

    The point is that a directory name should say what the run *was* without
    opening anything: 0007_crn_h256_la2 is legible at a glance, where
    0007_crn or a bare timestamp is not.
    """
    short = {"lstm_hidden": "h", "lookahead_frames": "la", "n_fft": "fft",
             "aux_loss_weight": "aux", "batch_size": "b",
             "learning_rate": "lr", "encoder_width": "w",
             "spectral_loss_weight": "spec", "mask_max": "mask",
             "lstm_bands": "bands", "data_fraction": "data", "seed": "s",
             "mask_loss_weight": "mloss",
             "spectral_compression": "comp", "n_resolutions": "res",
             "critic_loss_weight": "gan",
             # The aux target names itself, so it needs no prefix.
             "aux_target": ""}
    # Some knobs are words, not numbers, and would be unreadable spelled out.
    short_value = {"local_snr": "lsnr", "category": "cat"}
    parts = []
    for k in keys:
        if k in config and config[k] is not None:
            v = config[k]
            v = (short_value.get(v, v.replace("_", "")) if isinstance(v, str)
                 else f"{v:g}".replace(".", "").replace("-", "m"))
            parts.append(f"{short.get(k, k)}{v}")
    return "_".join(parts)


def config_diff(old, new):
    """What changed between two runs' settings.

    Answering "why is run 8 better than run 5" starts with knowing what was
    actually different, and memory is not reliable three experiments later.
    Bookkeeping fields are excluded so the diff shows decisions, not clock
    ticks.
    """
    skip = {"run_name", "sequence", "started", "git_commit", "git_dirty_files",
            "changed_from", "changes", "source_changed"}
    keys = (set(old) | set(new)) - skip
    out = {}
    for k in sorted(keys):
        a, b = old.get(k), new.get(k)
        if a != b:
            out[k] = {"from": a, "to": b}
    return out


def source_diff(old_dir, new_dir):
    """Which copied source files differ between two runs, and by how much."""
    changed = {}
    names = set()
    for d in (old_dir, new_dir):
        if os.path.isdir(d):
            names.update(os.listdir(d))
    for name in sorted(names):
        a = os.path.join(old_dir, name)
        b = os.path.join(new_dir, name)
        ta = io.open(a, encoding="utf-8").read() if os.path.exists(a) else None
        tb = io.open(b, encoding="utf-8").read() if os.path.exists(b) else None
        if ta == tb:
            continue
        if ta is None:
            changed[name] = "added"
        elif tb is None:
            changed[name] = "removed"
        else:
            import difflib
            plus = minus = 0
            for line in difflib.unified_diff(ta.splitlines(), tb.splitlines(),
                                             n=0, lineterm=""):
                if line.startswith("+") and not line.startswith("+++"):
                    plus += 1
                elif line.startswith("-") and not line.startswith("---"):
                    minus += 1
            changed[name] = f"+{plus} -{minus}"
    return changed


def unified_source_diff(old_run, new_run, filename):
    """The actual line-by-line diff of one file between two runs."""
    import difflib
    a = os.path.join(RUNS_DIR, old_run, SOURCE_DIR, filename)
    b = os.path.join(RUNS_DIR, new_run, SOURCE_DIR, filename)
    ta = io.open(a, encoding="utf-8").read().splitlines() if os.path.exists(a) else []
    tb = io.open(b, encoding="utf-8").read().splitlines() if os.path.exists(b) else []
    return "\n".join(difflib.unified_diff(ta, tb, fromfile=f"{old_run}/{filename}",
                                          tofile=f"{new_run}/{filename}",
                                          lineterm=""))


class RunLog:
    """One training run's directory, and the only thing that writes into it."""

    def __init__(self, path):
        self.path = path
        self.metrics_path = os.path.join(path, METRICS_NAME)
        self.checkpoints = os.path.join(path, CHECKPOINT_DIR)
        self.samples = os.path.join(path, SAMPLE_DIR)
        self.stop_path = os.path.join(path, STOP_NAME)
        self.name = os.path.basename(path)

    # --------------------------------------------------------- stop request
    #
    # Asking a run to stop is done by dropping a file in its directory, not by
    # signalling or killing the process. That keeps the same one-way rule as
    # everything else here -- training writes files, other tools read them --
    # so the dashboard never needs a process id, never has to survive a
    # restart to stay in control, and cannot half-kill a run mid-checkpoint.
    #
    # The trainer looks for this at the end of an epoch, once the checkpoints
    # and the metrics row for that epoch are already on disk, so stopping
    # this way costs nothing at all: the run ends exactly as it would have
    # ended had that been its last planned epoch.

    def request_stop(self, reason=""):
        """Ask a running trainer to finish after the epoch it is on."""
        with open(self.stop_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "requested": datetime.datetime.now().isoformat(timespec="seconds"),
                "reason": reason,
            }, indent=2))

    def stop_requested(self):
        return os.path.exists(self.stop_path)

    def stop_request(self):
        """The details of a pending stop, or None."""
        if not self.stop_requested():
            return None
        try:
            with open(self.stop_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def clear_stop(self):
        """Remove a stop request, so the run directory can be resumed."""
        try:
            os.remove(self.stop_path)
            return True
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------ creating

    @classmethod
    def create(cls, arch, config, tag_keys=("lstm_hidden", "lookahead_frames"),
               notes=None):
        os.makedirs(RUNS_DIR, exist_ok=True)
        seq = next_sequence()
        tag = build_tag(config, tag_keys)
        name = f"{seq:04d}_{arch}" + (f"_{tag}" if tag else "")
        path = os.path.join(RUNS_DIR, name)
        os.makedirs(path)
        os.makedirs(os.path.join(path, CHECKPOINT_DIR))
        os.makedirs(os.path.join(path, SAMPLE_DIR))

        run = cls(path)
        run.snapshot_source()

        full = dict(config)
        full.update({
            "run_name": name,
            "sequence": seq,
            "arch": arch,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
        })

        # What changed since the previous run -- both settings and code. This
        # is what turns a list of runs into an experiment log: three runs
        # later, nobody remembers whether run 6 changed the width or the
        # learning rate.
        previous = [r for r in sorted(os.listdir(RUNS_DIR)) if r != name
                    and os.path.isdir(os.path.join(RUNS_DIR, r))]
        if previous:
            prev = cls(os.path.join(RUNS_DIR, previous[-1]))
            full["changed_from"] = prev.name
            full["changes"] = config_diff(prev.config(), full)
            full["source_changed"] = source_diff(
                os.path.join(prev.path, SOURCE_DIR),
                os.path.join(path, SOURCE_DIR))
        else:
            full["changed_from"] = None
            full["changes"] = {}
            full["source_changed"] = {}

        run.write_config(full)
        if notes:
            with open(os.path.join(path, NOTES_NAME), "w",
                      encoding="utf-8") as f:
                f.write(notes.rstrip() + "\n")
        return run

    @classmethod
    def open(cls, name):
        path = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"no run named {name!r} in {RUNS_DIR}")
        return cls(path)

    # ------------------------------------------------------------- writing

    def snapshot_source(self):
        """Copy the scripts that produced this run into the run directory."""
        here = os.path.dirname(os.path.abspath(__file__))
        dest = os.path.join(self.path, SOURCE_DIR)
        os.makedirs(dest, exist_ok=True)
        copied = []
        for name in SOURCE_FILES:
            src = os.path.join(here, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest, name))
                copied.append(name)
        return copied

    def source_file(self, name):
        path = os.path.join(self.path, SOURCE_DIR, name)
        return io.open(path, encoding="utf-8").read() if os.path.exists(path) else None

    def write_config(self, config):
        with open(os.path.join(self.path, CONFIG_NAME), "w",
                  encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True, default=str)

    def update_config(self, **fields):
        """Add to the config after creation -- measured latency, dataset sizes."""
        c = self.config()
        c.update(fields)
        self.write_config(c)

    def log(self, **metrics):
        """Append one epoch's numbers. Flushed immediately.

        fsync is deliberate: without it a crash can leave the last lines in an
        OS buffer and lose exactly the epochs that mattered.
        """
        metrics.setdefault("wall_time", time.time())
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, default=float) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def checkpoint_path(self, name):
        return os.path.join(self.checkpoints, name)

    def sample_path(self, name):
        return os.path.join(self.samples, name)

    # ------------------------------------------------------------- reading

    def config(self):
        path = os.path.join(self.path, CONFIG_NAME)
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def history(self):
        """Every epoch logged so far.

        A half-written final line is skipped rather than raised on: a run
        killed mid-write should still be readable, which is the entire reason
        for the append-only format.
        """
        if not os.path.exists(self.metrics_path):
            return []
        rows = []
        with open(self.metrics_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return rows

    def best(self, key, mode="max"):
        """The epoch with the best value of one metric, ignoring gaps.

        Metrics that only run every few epochs are simply absent on the
        others, so rows lacking the key are skipped rather than counted as
        zero.
        """
        rows = [r for r in self.history() if r.get(key) is not None]
        if not rows:
            return None
        pick = max if mode == "max" else min
        return pick(rows, key=lambda r: r[key])

    def notes(self):
        path = os.path.join(self.path, NOTES_NAME)
        return io.open(path, encoding="utf-8").read().strip() \
            if os.path.exists(path) else ""

    def looks_running(self, stale_after=180):
        """Whether this run is probably still training.

        A heuristic, not a fact: it compares the last logged epoch's wall
        clock against now. There is deliberately no lock file or process id,
        because those go stale exactly when a run dies badly, which is when a
        wrong answer would be most misleading. A long gap simply reads as
        "not running", which is the safe way to be wrong.
        """
        rows = self.history()
        if not rows or "wall_time" not in rows[-1]:
            return False
        return (time.time() - float(rows[-1]["wall_time"])) < stale_after

    def sample_index(self):
        """The saved audio and mask pictures, grouped by clip then epoch.

        Listening to the same clip across epochs is the point, so the grouping
        is clip-first: each clip carries its unchanging noisy and clean
        references plus one enhanced version per evaluation.
        """
        if not os.path.isdir(self.samples):
            return {}
        clips = {}
        for name in sorted(os.listdir(self.samples)):
            m = re.match(r"^e(\d+)_(.+)_(\d+)_(enhanced)\.wav$", name)
            if m:
                epoch, cat, idx = int(m.group(1)), m.group(2), m.group(3)
                key = f"{cat}_{idx}"
                clips.setdefault(key, {"category": cat, "index": int(idx),
                                       "epochs": [], "refs": {}})
                clips[key]["epochs"].append({"epoch": epoch, "audio": name,
                                             "mask": name.replace(
                                                 "_enhanced.wav", "_mask.png")})
                continue
            m = re.match(r"^ref_(.+)_(\d+)_(noisy|clean)\.wav$", name)
            if m:
                key = f"{m.group(1)}_{m.group(2)}"
                clips.setdefault(key, {"category": m.group(1),
                                       "index": int(m.group(2)),
                                       "epochs": [], "refs": {}})
                clips[key]["refs"][m.group(3)] = name
        for c in clips.values():
            c["epochs"].sort(key=lambda e: e["epoch"])
        return clips

    def evaluations(self):
        """Any standalone evaluate.py results saved into this run."""
        out = {}
        for name in sorted(os.listdir(self.path)):
            m = re.match(r"^evaluation_(\w+)\.json$", name)
            if m:
                try:
                    with open(os.path.join(self.path, name),
                              encoding="utf-8") as f:
                        out[m.group(1)] = json.load(f).get("summary")
                except Exception:
                    pass
        return out

    def export_csv(self, path=None):
        """Flatten the history to CSV, for a spreadsheet or a report."""
        rows = self.history()
        if not rows:
            return None
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        path = path or os.path.join(self.path, "metrics.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        return path


def all_runs():
    """Every run on disk, newest first, with enough detail to choose one.

    The directory is scanned rather than an index file being maintained,
    because an index can go stale and a directory listing cannot.
    """
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        path = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(path):
            continue
        run = RunLog(path)
        hist = run.history()
        out.append({
            "name": name,
            "config": run.config(),
            "epochs": len(hist),
            "last": hist[-1] if hist else None,
        })
    return out


def self_test():
    import tempfile
    global RUNS_DIR
    original = RUNS_DIR
    RUNS_DIR = tempfile.mkdtemp(prefix="runlog_test_")
    try:
        cfg = {"lstm_hidden": 256, "lookahead_frames": 2, "n_fft": 512,
               "learning_rate": 3e-4}
        run = RunLog.create("crn", cfg, notes="first check")
        print(f"  created       {run.name}")
        assert run.name.startswith("0001_crn_h256_la2"), run.name

        src = os.listdir(os.path.join(run.path, "source"))
        print(f"  source        snapshotted {sorted(src)}")
        assert "model_crn.py" in src and "runlog.py" in src

        second = RunLog.create("tcn", {"lstm_hidden": 128,
                                       "lookahead_frames": 4, "n_fft": 512,
                                       "learning_rate": 1e-3})
        print(f"  second run    {second.name}")
        assert second.name.startswith("0002_tcn_h128_la4"), second.name

        ch = second.config()["changes"]
        print(f"  changed from  {second.config()['changed_from']}")
        for k, v in ch.items():
            print(f"    {k}: {v['from']} -> {v['to']}")
        assert ch["lstm_hidden"] == {"from": 256, "to": 128}
        assert ch["arch"] == {"from": "crn", "to": "tcn"}
        assert "n_fft" not in ch, "unchanged settings must not appear"

        c = run.config()
        print(f"  config        {len(c)} fields, git {c.get('git_commit')}")
        assert c["arch"] == "crn" and c["sequence"] == 1

        for e in range(1, 11):
            row = {"epoch": e, "train_loss": 1.0 / e, "val_loss": 1.1 / e}
            if e % 5 == 0:                      # the periodic expensive ones
                row.update({"pesq": 1.5 + e * 0.05, "stoi": 0.6 + e * 0.01})
            run.log(**row)
        hist = run.history()
        print(f"  logged        {len(hist)} epochs, "
              f"{sum(1 for r in hist if 'pesq' in r)} with pesq")
        assert len(hist) == 10

        b = run.best("pesq", "max")
        print(f"  best pesq     {b['pesq']:.3f} at epoch {b['epoch']} "
              f"(gaps skipped, not read as zero)")
        assert b["epoch"] == 10
        assert run.best("val_loss", "min")["epoch"] == 10

        # a crash mid-write must not destroy the history
        with open(run.metrics_path, "a", encoding="utf-8") as f:
            f.write('{"epoch": 11, "train_loss": 0.0')       # truncated
        survived = run.history()
        print(f"  torn write    {len(survived)} epochs still readable "
              f"(the broken line is skipped)")
        assert len(survived) == 10

        csv_path = run.export_csv()
        print(f"  csv           {os.path.basename(csv_path)}, "
              f"{len(open(csv_path).readlines())} lines")

        listed = all_runs()
        print(f"  listing       {[r['name'] for r in listed]}")
        assert len(listed) == 2 and listed[0]["name"].startswith("0002")
        assert listed[1]["epochs"] == 10

        print("\nrunlog passed.")
    finally:
        shutil.rmtree(RUNS_DIR, ignore_errors=True)
        RUNS_DIR = original


if __name__ == "__main__":
    self_test()
