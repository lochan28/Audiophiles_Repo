"""
yamnet_head.py

Train a small classifier on top of YAMNet's embeddings and measure honestly
whether it could ever carry part of the labelling work.

    .venv_yamnet/Scripts/python.exe scripts/yamnet_head.py

Requires `yamnet_probe.py` to have run first -- it reads that script's saved
embeddings and never touches audio.

THE QUESTION THIS ANSWERS

YAMNet's own class list is AudioSet's, not ours: it says "Explosion" where we
need mortar-versus-artillery, and the probe showed it cannot tell those apart
at all. The hope is that its 1024-number fingerprint still CONTAINS that
distinction even though its output layer throws it away, so a small classifier
trained on our own 636 labelled clips could recover it.

That is a testable claim, and this script tests it three ways.

  1. THE SEVEN FINE CATEGORIES. The full job. If this worked well the machine
     could propose fine labels -- which, to be clear, we would still not
     auto-accept (see the circularity note below).

  2. IMPULSIVE VERSUS CONTINUOUS. The coarse job, and the one actually worth
     automating. A bang and a drone are different in ways a machine can see.

  3. WHERE DOES `rocket` BELONG? An open question in this project with two
     contradictory answers already in the code: mixing.py's
     REPEATABLE_CATEGORIES treats rocket as NOT impulsive, train.py's
     pick_sample_clips treats it as impulsive. Rather than pick by taste,
     this trains the impulsive/continuous classifier with rocket held out
     entirely, then asks which side it lands on. That is evidence, not an
     opinion -- though it is evidence about how a sound BEHAVES, which is not
     automatically the same as how the mixing rule should treat it.

WHY THE SPLITS ARE THE EXISTING ONES

Training on train/ and testing on validation/ + testing/ is not a detail. The
noise splits were dealt so that no source YouTube video feeds more than one
split, so a score measured across them is a real generalisation score. Shuffling
all 636 and cross-validating would put slices of ONE recording on both sides and
report a number several points too high -- the same mistake this project already
guards against everywhere else.

WHY THE CONFIDENCE INTERVALS MATTER MORE THAN THE ACCURACIES

Two of our categories are tiny in the held-out splits -- heavy_artillery has 4
validation and 8 testing clips, mortar 6 and 8. A precision of "1.00" on 4
examples is not evidence of precision; it is evidence of 4 examples. This script
therefore prints a Wilson interval beside every per-class number, because the
plan's whole premise is picking a threshold where precision is high enough to
auto-accept, and that premise dies quietly if nobody looks at the interval.

THIS SCRIPT DECIDES NOTHING. It writes no labels, moves no audio, and does not
touch dataset/, dataset_v2/ or the frozen mixtures. It prints a measurement.
"""

import os
import csv
import sys
import argparse

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

CLASSIFIER = "logreg"
# "logreg" is multinomial logistic regression: a single linear layer on the
# embedding. "mlp" adds one hidden layer of HIDDEN units. Start with logreg --
# if a LINEAR probe separates the classes, the information is plainly there and
# a bigger head is unnecessary; if logreg fails and the MLP succeeds, the
# information is present but tangled, which is a different and weaker claim.

HIDDEN = 128
# Hidden units, "mlp" only. Larger fits the 420 training clips more closely and
# generalises worse -- with this little data, treat anything above ~256 as
# fitting noise.

C_REG = 1.0
# Inverse regularisation strength for logreg. LOWER (0.01-0.1) constrains the
# model harder, which is usually right when 1024 features meet 420 examples;
# HIGHER lets it fit the training set more exactly. Worth sweeping: with
# 1024-d inputs and 420 rows, this is the knob that matters most.

STANDARDISE = True
# Scale each of the 1024 features to zero mean and unit variance using the
# TRAINING rows only. Logistic regression converges far better with it, and
# fitting the scaler on training data alone is what keeps the held-out score
# honest.

AUTO_ACCEPT_PRECISION = 0.95
# The precision a class must reach before its proposals could be auto-accepted
# without a human. The script reports, for each class, the confidence threshold
# that achieves this on held-out data AND how many examples that figure rests
# on -- because a threshold calibrated on 4 clips is not calibrated.

IMPULSIVE = ("assault_rifle", "mortar", "heavy_artillery")
# The coarse grouping used for test 2. This is mixing.py's REPEATABLE_CATEGORIES
# set, chosen here because it is the one that currently drives real behaviour.
# `rocket` is deliberately absent: test 3 asks the data where it belongs rather
# than assuming. Anything not listed and not rocket counts as continuous.

ROCKET_CATEGORY = "rocket"
# The category held out of test 3's training to be classified blind. Set to
# None to skip that test.

SEED = 0
# Fixed so a re-run reproduces. Changing it re-rolls the MLP's initialisation
# only; logreg is deterministic.


# ═══════════════════════════════════════════════════════════════════
#  LOCKED -- set by yamnet_probe.py's output, do not change
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_DIR = os.path.join(PROJECT_ROOT, "yamnet_probe_out")
CSV_PATH = os.path.join(PROBE_DIR, "yamnet_probe.csv")
NPY_PATH = os.path.join(PROBE_DIR, "yamnet_embeddings.npy")
TRAIN_SPLIT = "train"
HELD_OUT_SPLITS = ("validation", "testing")


def wilson(hits, n, z=1.96):
    """95% Wilson interval for a proportion.

    Chosen over the textbook normal interval because that one is badly wrong
    for small n and for proportions near 0 or 1 -- it happily reports
    [1.00, 1.00] for 4 out of 4, which is the exact situation here.
    """
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load():
    if not (os.path.exists(CSV_PATH) and os.path.exists(NPY_PATH)):
        raise SystemExit(
            "no probe output found. Run this first:\n"
            "    .venv_yamnet/Scripts/python.exe scripts/yamnet_probe.py")
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.load(NPY_PATH)
    if len(rows) != X.shape[0]:
        raise SystemExit(f"{len(rows)} CSV rows but {X.shape[0]} embeddings; "
                         "re-run yamnet_probe.py so the two match")
    y = np.array([r["category"] for r in rows])
    split = np.array([r["split"] for r in rows])
    return X, y, split, rows


def build():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if CLASSIFIER == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(HIDDEN,), max_iter=2000,
                            random_state=SEED)
    else:
        clf = LogisticRegression(max_iter=5000, C=C_REG, random_state=SEED)
    return make_pipeline(StandardScaler(), clf) if STANDARDISE else clf


def evaluate(X, y, split, labels, title, note=""):
    """Fit on the training split, score on held-out. Returns (model, report)."""
    from sklearn.metrics import confusion_matrix

    tr = split == TRAIN_SPLIT
    te = np.isin(split, HELD_OUT_SPLITS)
    model = build().fit(X[tr], y[tr])
    pred = model.predict(X[te])
    truth = y[te]

    acc = float((pred == truth).mean())
    lo, hi = wilson(int((pred == truth).sum()), len(truth))
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"  trained on {tr.sum()} clips, scored on {te.sum()} held out")
    chance = max((truth == l).mean() for l in labels)
    print(f"  accuracy {acc:.3f}  [{lo:.2f}-{hi:.2f}]   "
          f"always-guess-the-commonest would get {chance:.3f}")

    print(f"\n  {'class':<17}{'n':>4}{'recall':>9}{'precision':>11}"
          f"   precision 95% CI")
    for lab in labels:
        n = int((truth == lab).sum())
        if n == 0:
            print(f"  {lab:<17}{0:>4}        --         --   (absent held out)")
            continue
        rec = float((pred[truth == lab] == lab).mean())
        npred = int((pred == lab).sum())
        if npred:
            hits = int((truth[pred == lab] == lab).sum())
            prec = hits / npred
            plo, phi = wilson(hits, npred)
            flag = "  <-- interval too wide to calibrate" if phi - plo > 0.35 else ""
            print(f"  {lab:<17}{n:>4}{rec:>9.3f}{prec:>11.3f}"
                  f"   [{plo:.2f}-{phi:.2f}] on {npred:>3}{flag}")
        else:
            print(f"  {lab:<17}{n:>4}{rec:>9.3f}{'never':>11}"
                  f"   (predicted for nothing)")

    if len(labels) <= 8:
        cm = confusion_matrix(truth, pred, labels=labels)
        print(f"\n  confusion (rows = truth, cols = predicted)")
        print("  " + " " * 17 + "".join(f"{l[:6]:>7}" for l in labels))
        for lab, row in zip(labels, cm):
            print(f"  {lab:<17}" + "".join(f"{v:>7}" for v in row))
    return model, (acc, truth, pred)


def threshold_report(model, X, y, split, labels):
    """For each class, the confidence needed to reach AUTO_ACCEPT_PRECISION."""
    te = np.isin(split, HELD_OUT_SPLITS)
    proba = model.predict_proba(X[te])
    truth = y[te]
    classes = list(model.classes_)

    print(f"\n  could any class be AUTO-ACCEPTED at "
          f"{AUTO_ACCEPT_PRECISION:.0%} precision?")
    print(f"  {'class':<17}{'threshold':>10}{'kept':>7}{'of':>5}"
          f"   verdict")
    for lab in labels:
        if lab not in classes:
            continue
        p = proba[:, classes.index(lab)]
        best = None
        for t in np.arange(0.30, 1.00, 0.01):
            sel = p >= t
            if sel.sum() == 0:
                continue
            hits = int((truth[sel] == lab).sum())
            prec = hits / int(sel.sum())
            if prec >= AUTO_ACCEPT_PRECISION:
                plo, _ = wilson(hits, int(sel.sum()))
                best = (t, int(sel.sum()), int((truth == lab).sum()), plo)
                break
        if best is None:
            print(f"  {lab:<17}{'none':>10}{'-':>7}{int((truth == lab).sum()):>5}"
                  f"   never reaches it -- always human")
        else:
            t, kept, total, plo = best
            # The interval's LOWER bound is the honest claim: a threshold whose
            # precision could plausibly be 0.6 is not a 0.95 threshold.
            ok = plo >= 0.80 and kept >= 20
            print(f"  {lab:<17}{t:>10.2f}{kept:>7}{total:>5}"
                  f"   {'usable' if ok else 'NOT trustworthy: lower bound '
                                           f'{plo:.2f} on {kept} clips'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--classifier", choices=("logreg", "mlp"), default=None)
    args = ap.parse_args()
    global CLASSIFIER
    if args.classifier:
        CLASSIFIER = args.classifier

    X, y, split, rows = load()
    print(f"embeddings   {X.shape[0]} clips x {X.shape[1]} features")
    print(f"classifier   {CLASSIFIER}"
          + (f", C={C_REG}" if CLASSIFIER == "logreg" else f", hidden={HIDDEN}")
          + f", standardised={STANDARDISE}")
    print(f"splits       train -> {list(HELD_OUT_SPLITS)} "
          f"(no source video crosses a split, so this generalises)")

    fine = sorted(set(y))
    model, _ = evaluate(X, y, split, fine,
                        "TEST 1 -- the seven fine categories",
                        "the full job: can it propose mortar vs artillery?")
    threshold_report(model, X, y, split, fine)

    # ---- test 2: impulsive vs continuous, rocket excluded entirely ---------
    keep = y != ROCKET_CATEGORY if ROCKET_CATEGORY else np.ones(len(y), bool)
    coarse = np.where(np.isin(y, IMPULSIVE), "impulsive", "continuous")
    m2, _ = evaluate(X[keep], coarse[keep], split[keep],
                     ["impulsive", "continuous"],
                     "TEST 2 -- impulsive vs continuous (the coarse job)",
                     f"{ROCKET_CATEGORY!r} excluded; IMPULSIVE = {IMPULSIVE}")
    threshold_report(m2, X[keep], coarse[keep], split[keep],
                     ["impulsive", "continuous"])

    # ---- test 3: ask the data where rocket belongs ------------------------
    if ROCKET_CATEGORY:
        rk = y == ROCKET_CATEGORY
        if rk.sum():
            proba = m2.predict_proba(X[rk])
            cls = list(m2.classes_)
            p_imp = proba[:, cls.index("impulsive")]
            votes = int((p_imp > 0.5).sum())
            print(f"\nTEST 3 -- where does {ROCKET_CATEGORY!r} belong?")
            print(f"  The test-2 classifier never saw a rocket clip. Asked to "
                  f"place all {rk.sum()} of them:")
            print(f"    called impulsive   {votes:>4} ({100.0 * votes / rk.sum():.1f}%)")
            print(f"    called continuous  {rk.sum() - votes:>4} "
                  f"({100.0 * (rk.sum() - votes) / rk.sum():.1f}%)")
            print(f"    mean P(impulsive)  {p_imp.mean():.3f}   "
                  f"median {np.median(p_imp):.3f}")
            verdict = ("IMPULSIVE" if p_imp.mean() > 0.65 else
                       "CONTINUOUS" if p_imp.mean() < 0.35 else "GENUINELY MIXED")
            print(f"  reads as: {verdict}")
            print(f"  Note this says how rocket clips SOUND, which is not "
                  f"automatically\n  how the mixing rule should treat them -- "
                  f"that is still a decision.")

    print(f"\nNothing was labelled, moved or written. dataset/, dataset_v2/ "
          f"and the\nfrozen mixtures were not touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
