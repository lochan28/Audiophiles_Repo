"""
evaluate.py

Scores a model against the frozen mixtures, on the three numbers the problem
statement actually asks for: SNR, STOI and PESQ.

    python scripts/evaluate.py 0001_crn_h256_la2
    python scripts/evaluate.py 0001_crn_h256_la2 --split validation
    python scripts/evaluate.py 0001_crn_h256_la2 --checkpoint last

Also imported by train.py, which calls evaluate() every few epochs.

**Every metric is reported three ways: the noisy input, the enhanced output,
and the gain between them.** The absolute number alone is close to
meaningless -- PESQ 2.4 is a triumph on audio that started at 1.1 and a
failure on audio that started at 2.3. What the model is worth is the
difference it made, and only the noisy baseline shows that. The baseline is
also a property of the frozen set alone, so it is identical for every model
and every run: a fixed yardstick, not a moving one.

Results are broken down per noise category as well as overall, because the
seven categories are not equally hard and an average hides which ones fail.
Note that `heavy_artillery` and `mortar` have only 4 and 6 validation clips,
so their per-category numbers are noisy -- read them as hints, not verdicts.
"""

import os
import sys
import json
import time
import argparse
import collections

import numpy as np
import torch

from pesq import pesq as pesq_score, NoUtterancesError
from pystoi import stoi as stoi_score

from mixing import FrozenMixtures, read_noise_classes
from runlog import RunLog
import model_crn


SAMPLE_RATE = model_crn.SAMPLE_RATE
EPS = 1e-8

# The problem statement's targets, drawn on every report.
TARGETS = {"si_snr": 15.0, "stoi": 0.85, "pesq": 2.5}


def si_snr(est, ref):
    """Scale-invariant SNR in dB, on numpy arrays."""
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + EPS)
                               / (np.sum(noise ** 2) + EPS)))


def safe_pesq(clean, other):
    """PESQ, or None when it refuses.

    PESQ rejects signals it cannot align or that contain no detectable
    speech. A heavily padded clip with a short utterance can trip it. Returning
    None keeps that clip out of the average rather than poisoning it with a
    fabricated score.
    """
    try:
        return float(pesq_score(SAMPLE_RATE, clean, other, "wb"))
    except (NoUtterancesError, ValueError, RuntimeError):
        return None


def safe_stoi(clean, other):
    try:
        v = float(stoi_score(clean, other, SAMPLE_RATE, extended=False))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def evaluate(model, dataset, device, max_clips=None, batch_size=16,
             progress=False):
    """Score a model over a frozen dataset.

    Returns overall metrics, per-category metrics, and the classifier's
    accuracy and confusion.
    """
    model.eval()
    classes = read_noise_classes()
    n = len(dataset) if max_clips is None else min(max_clips, len(dataset))

    per_clip = []
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    t0 = time.perf_counter()

    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        metas = [dataset.describe(i) for i in idxs]
        noisy = torch.from_numpy(np.stack([m["noisy"] for m in metas])).to(device)

        with torch.no_grad():
            est, _, logits, _ = model(noisy)
        est = est.cpu().numpy()

        for k, m in enumerate(metas):
            clean, dirty, out = m["clean"], m["noisy"], est[k]
            row = {
                "index": m["index"], "category": m["category"],
                "snr_db": m["snr_db"], "language": m["language"],
                "si_snr_noisy": si_snr(dirty, clean),
                "si_snr_enhanced": si_snr(out, clean),
                "stoi_noisy": safe_stoi(clean, dirty),
                "stoi_enhanced": safe_stoi(clean, out),
                "pesq_noisy": safe_pesq(clean, dirty),
                "pesq_enhanced": safe_pesq(clean, out),
            }
            per_clip.append(row)
            # Only a category head produces something a confusion matrix can
            # hold. A local-SNR head emits one number per FRAME, so argmax
            # here would silently index the matrix with a frame number.
            if logits is not None and getattr(
                    model, "aux_target", "category") == "category":
                confusion[m["label"], int(logits[k].argmax())] += 1

        if progress:
            done = min(start + batch_size, n)
            print(f"\r  scoring {done}/{n}", end="", flush=True)
    if progress:
        print()

    def summarise(rows):
        out = {"clips": len(rows)}
        for base in ("si_snr", "stoi", "pesq"):
            a = _mean([r[base + "_noisy"] for r in rows])
            b = _mean([r[base + "_enhanced"] for r in rows])
            out[base + "_noisy"] = a
            out[base + "_enhanced"] = b
            out[base + "_gain"] = (b - a) if (a is not None and b is not None) \
                else None
        return out

    by_category = {}
    grouped = collections.defaultdict(list)
    for r in per_clip:
        grouped[r["category"]].append(r)
    for cat in classes:
        if grouped[cat]:
            by_category[cat] = summarise(grouped[cat])

    result = {
        "overall": summarise(per_clip),
        "by_category": by_category,
        "classes": classes,
        "confusion": confusion.tolist(),
        "aux_target": getattr(model, "aux_target", "category"),
        # None rather than 0.0 when there is no classifier: a zero here would
        # read as "the classifier is at chance", which is a different claim.
        "aux_accuracy": (float(np.trace(confusion) / confusion.sum())
                         if confusion.sum() else None),
        "pesq_failures": sum(1 for r in per_clip if r["pesq_enhanced"] is None),
        "seconds": time.perf_counter() - t0,
    }
    result["targets_met"] = {
        "si_snr": (result["overall"]["si_snr_enhanced"] or -99) >= TARGETS["si_snr"],
        "stoi": (result["overall"]["stoi_enhanced"] or 0) >= TARGETS["stoi"],
        "pesq": (result["overall"]["pesq_enhanced"] or 0) >= TARGETS["pesq"],
    }
    return result, per_clip


def print_report(result, title=""):
    o = result["overall"]
    print(f"\n{title}  ({o['clips']} clips, {result['seconds']:.1f} s)")
    print(f"{'':16}{'noisy':>9}{'enhanced':>11}{'gain':>9}{'target':>9}  ")
    print("  " + "-" * 54)
    for key, label in (("si_snr", "SNR (SI-SNR)"), ("stoi", "STOI"),
                       ("pesq", "PESQ")):
        a, b, g = o[key + "_noisy"], o[key + "_enhanced"], o[key + "_gain"]
        tgt = TARGETS[key]
        hit = "MET" if result["targets_met"][key] else ""
        fmt = "{:>9.3f}" if key != "si_snr" else "{:>9.2f}"
        print(f"  {label:<14}" + fmt.format(a) + fmt.format(b).rjust(2)
              + f"{g:>+9.3f}{tgt:>9.2f}  {hit}")

    if result.get("aux_accuracy") is not None:
        print(f"\n  noise classifier accuracy: {result['aux_accuracy']:.3f} "
              f"(chance is {1/len(result['classes']):.3f})")
    else:
        print(f"\n  auxiliary head: {result.get('aux_target', 'none')} "
              f"- not a classifier, so no accuracy to report")
    if result["pesq_failures"]:
        print(f"  PESQ could not score {result['pesq_failures']} clips; "
              f"they are excluded from the average")

    print(f"\n  per category" + " " * 14 + "SNR gain   STOI gain   PESQ gain"
          f"   clips")
    for cat, m in result["by_category"].items():
        def g(k):
            v = m[k + "_gain"]
            return f"{v:+.3f}" if v is not None else "    -"
        print(f"    {cat:<20}{g('si_snr'):>9}{g('stoi'):>12}"
              f"{g('pesq'):>12}{m['clips']:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory name, e.g. 0001_crn_h256_la2")
    ap.add_argument("--split", default="testing",
                    choices=["validation", "testing"])
    ap.add_argument("--checkpoint", default="auto",
                    choices=["auto", "best_pesq", "best", "last"],
                    help="auto prefers best_pesq.pt; 'best' is chosen by "
                         "validation LOSS and can be tens of epochs early")
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    run = RunLog.open(args.run)
    cfg = run.config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import train
    # Shared loader: it rebuilds the model from the run's OWN config, so a run
    # made before a default changed still loads with the right shapes.
    model, ck_name, epoch = train.load_run_model(run, device, args.checkpoint)

    print(f"run           {run.name}")
    print(f"checkpoint    {ck_name} from epoch {epoch}")
    print(f"split         {args.split} (frozen mixtures)")

    dataset = FrozenMixtures(args.split)
    result, per_clip = evaluate(model, dataset, device,
                                max_clips=args.max_clips, progress=True)
    result.update({"run": run.name, "split": args.split,
                   "checkpoint": args.checkpoint, "epoch": epoch})

    print_report(result, f"{run.name} / {args.split} / epoch {epoch}")

    out = os.path.join(run.path, f"evaluation_{args.split}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": result, "per_clip": per_clip}, f, indent=2,
                  default=float)
    print(f"\nwrote {os.path.relpath(out, os.path.dirname(run.path))}")


if __name__ == "__main__":
    main()
