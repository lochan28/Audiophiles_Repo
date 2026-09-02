"""Score every past run on the CURRENT frozen set, keeping its old score beside it.

The SNR distribution changed from a flat U(-5,+15) to 90% U(0,15) / 10%
U(-5,0), and the frozen evaluation sets were regenerated to match. That makes
every score recorded before the change unreadable against every score recorded
after it -- not because the models changed, but because the exam did. Measured
on run 0025, the change alone is worth about +0.13 PESQ, +0.026 STOI and
+1.5 dB with the weights untouched.

Rather than strand 25 runs of history, this re-scores their saved checkpoints
on the new set. Each run gets `evaluation_<split>_current.json` holding the new
numbers, the old converged mean from its own metrics.jsonl for reference, and
the difference between them. Nothing already on disk is modified: metrics.jsonl
and the original evaluation_*.json files are left exactly as they were.

    python scripts/rescore_runs.py --apply
    python scripts/rescore_runs.py --apply --split testing --only 0016 0025

Runs whose checkpoints predate best_pesq.pt fall back to last.pt, which for a
converged run sits near its plateau. The checkpoint actually used is recorded,
because for runs 0001-0015 the alternative is best.pt, which is chosen by
validation loss and was 39 epochs early in one measured case.
"""

import os
import sys
import json
import argparse
import statistics as st

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train as T
import mixing
from runlog import RunLog, RUNS_DIR
from mixing import FrozenMixtures
import evaluate as ev

# Evaluations at or after this epoch count as converged, per PROJECT_CONTEXT_4
# Section 19. 150-epoch runs use 110; shorter ones fall back to 45.
CONVERGED_LONG = 110
CONVERGED_SHORT = 45


def old_converged(run):
    """The run's own recorded score, from the set it was actually trained against."""
    rows = []
    for line in open(run.metrics_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue          # a torn final line from a killed process
        if "pesq_enhanced" in r:
            rows.append(r)
    if not rows:
        return None
    last = rows[-1]["epoch"]
    lo = CONVERGED_LONG if last >= CONVERGED_LONG else CONVERGED_SHORT
    conv = [r for r in rows if r["epoch"] >= lo] or rows[-3:]
    return {
        "window": f"epoch >= {lo}",
        "evals": len(conv),
        "pesq": st.mean([r["pesq_enhanced"] for r in conv]),
        "stoi": st.mean([r["stoi_enhanced"] for r in conv]),
        "si_snr": st.mean([r["si_snr_enhanced"] for r in conv]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation",
                    choices=["validation", "testing"])
    ap.add_argument("--only", nargs="*", default=None,
                    help="substrings of run names to limit to")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--apply", action="store_true",
                    help="without this, only lists what would be scored")
    args = ap.parse_args()

    names = sorted(d for d in os.listdir(RUNS_DIR)
                   if os.path.isdir(os.path.join(RUNS_DIR, d)))
    if args.only:
        names = [n for n in names if any(o in n for o in args.only)]

    print(f"{len(names)} runs, split {args.split}, "
          f"current profile {mixing.SNR_PROFILE}")
    if not args.apply:
        for n in names:
            print(f"  would score {n}")
        print("\nre-run with --apply")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FrozenMixtures(args.split)
    print(f"{len(dataset)} frozen {args.split} mixtures\n")
    print(f"{'run':<40}{'ckpt':>13}{'PESQ':>8}{'STOI':>8}{'SNR':>7}"
          f"{'  (old ->  new PESQ)':>22}")

    summary = []
    for name in names:
        run = RunLog.open(name)
        try:
            model, ck_name, epoch = T.load_run_model(run, device)
        except Exception as exc:
            print(f"{name:<40}  SKIPPED: {type(exc).__name__}: {exc}")
            continue

        result, _ = ev.evaluate(model, dataset, device, max_clips=args.max_clips)
        o = result["overall"]
        old = old_converged(run)

        record = {
            "run": name,
            "split": args.split,
            "checkpoint": ck_name,
            "epoch": epoch,
            "snr_profile_scored_on": mixing.SNR_PROFILE,
            "snr_profile_trained_on": run.config().get("snr_profile"),
            "current": {"pesq": o["pesq_enhanced"], "stoi": o["stoi_enhanced"],
                        "si_snr": o["si_snr_enhanced"],
                        "pesq_noisy": o["pesq_noisy"],
                        "stoi_noisy": o["stoi_noisy"],
                        "si_snr_noisy": o["si_snr_noisy"]},
            # metrics.jsonl only ever holds VALIDATION scores, so this is
            # meaningless beside a testing result and is omitted there.
            "as_originally_recorded_validation":
                old if args.split == "validation" else None,
            "note": ("`as_originally_recorded_validation` was measured on the archived set "
                     "in generated_test_mixtures_snr_uniform/ under a flat "
                     "U(-5,+15) SNR draw. The two are not comparable; the "
                     "difference is mostly the exam, not the model."),
        }
        with open(os.path.join(run.path,
                               f"evaluation_{args.split}_current.json"),
                  "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        delta = (f"{old['pesq']:.3f} -> {o['pesq_enhanced']:.3f}"
                 if old else "no prior score")
        print(f"{name:<40}{ck_name:>13}{o['pesq_enhanced']:>8.3f}"
              f"{o['stoi_enhanced']:>8.4f}{o['si_snr_enhanced']:>7.2f}"
              f"{delta:>22}")
        summary.append(record)

    out = os.path.join(RUNS_DIR, f"rescore_{args.split}_current.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"snr_profile": mixing.SNR_PROFILE,
                   "split": args.split, "runs": summary}, f, indent=2)
    print(f"\nwrote {len(summary)} records and {out}")


if __name__ == "__main__":
    main()
