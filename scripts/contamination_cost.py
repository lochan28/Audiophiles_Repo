"""
contamination_cost.py

Measure what the speech and music found inside our NOISE clips costs us, by
scoring the shipped system separately on the mixtures that use a contaminated
noise clip and the mixtures that do not.

    .venv/Scripts/python.exe scripts/contamination_cost.py
    .venv/Scripts/python.exe scripts/contamination_cost.py --split validation

Requires `yamnet_probe.py` to have run: it reads that script's per-clip
speech/music scores to decide which noise clips are suspect.

WHY THIS IS WORTH MEASURING

yamnet_probe.py found that about 12% of our hand-cut noise clips contain speech
or music, and because each noise clip is reused across several mixtures, a third
of the frozen TESTING set is affected. A human has since listened and confirmed
the flags are real.

That matters beyond tidiness. Our model removes noise by predicting a mask over
the spectrogram -- it turns bands down. A mask can suppress a rotor, because a
rotor does not look like speech. It cannot suppress an interfering TALKER,
because a talker looks exactly like the thing we are trying to keep. Separating
two voices is a different problem (speaker separation) that this architecture
cannot do at all.

So every mixture whose "noise" contains a voice is an exam question with no
right answer available to us. If those questions are dragging the average down,
then our real score on the task we actually built for is higher than the
headline, and cleaning the data is a bigger lever than any architecture change
left on the list.

THE CONFOUND, AND WHY THIS SCRIPT TAKES IT SERIOUSLY

The contaminated clips are NOT spread evenly across categories -- `ambient` and
`mortar` are heavily over-represented, `helicopter` and `heavy_artillery` almost
absent. Those categories also differ in intrinsic difficulty. So a raw
clean-versus-flagged gap could be measuring "ambient is hard" rather than
"speech in the noise is hard", and reporting it as the latter would be exactly
the kind of confident nonsense this project's measurement rules exist to
prevent.

Two things are done about it. The comparison is repeated WITHIN each category,
where the confound cannot operate. And the flagged group is split by how
confident YAMNet was, because a dose-response -- worse scores for clips with
more speech in them -- is evidence a category effect cannot easily produce.

WHAT THIS DOES NOT DO

It scores an existing model on existing frozen mixtures. It trains nothing,
regenerates nothing, relabels nothing, and writes no audio. Every number the
project already reports stays valid; this only slices the same exam differently.
"""

import os
import csv
import sys
import argparse
import collections

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

WEAK_THRESHOLD = 0.30
STRONG_THRESHOLD = 0.80
# How the noise clips are bucketed by YAMNet's worst speech/music frame score:
#   clean   below WEAK
#   weak    between the two -- suspect, some will be false alarms
#   strong  above STRONG -- a human confirmed these are really contaminated
# The three-way split exists to look for a DOSE-RESPONSE. Raising STRONG makes
# that group smaller and purer; lowering WEAK moves borderline clips out of
# "clean", which is the conservative direction if you want the clean group to
# really be clean.

BOOTSTRAP = 10000
# Resamples used for the confidence interval on each difference. The groups are
# small (40-85 mixtures) and PESQ is not normally distributed, so a bootstrap is
# more honest here than a standard error. 10000 is plenty; lower it only if you
# are impatient.

MIN_CATEGORY_N = 4
# A category needs at least this many mixtures on BOTH sides before its
# within-category comparison is printed. Below that the comparison is noise and
# printing it invites over-reading.

STAGE1_RUN = "0036_crn_cat_comp03_res1_spec600_la6_s0"
POSTFILTER = "postfilter_ch48_la0_snr25.pt"
# The SHIPPED pairing as of 2026-09-19 (HANDOFF section 2), which is what the
# Pi runs. Note this is NOT what the bench and lab web pages load -- those use
# postfilter_ch96_la0_snr10.pt, which is paired with run 0026. Both pairings are
# self-consistent; they are simply different systems, so a number from one is
# not comparable with a number from the other.


# ═══════════════════════════════════════════════════════════════════
#  LOCKED -- set by the frozen data and by yamnet_probe.py's output
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE_CSV = os.path.join(PROJECT_ROOT, "yamnet_probe_out", "yamnet_probe.csv")
MIXTURE_MANIFEST = os.path.join(PROJECT_ROOT, "generated_test_mixtures",
                                "test_mixtures_manifest.csv")
NOISE_FLOOR_PESQ = 0.0061      # run-to-run spread, HANDOFF section 4
METRICS = ("pesq", "stoi", "si_snr")


def noise_file_by_index(split):
    """{mixture index: noise clip basename} from the frozen manifest.

    FrozenMixtures.describe() does not carry the noise filename -- it returns
    the audio, the category and the SNR -- so the join back to which noise clip
    a mixture used has to come from the manifest on disk. That file is frozen,
    so this mapping is stable.
    """
    with open(MIXTURE_MANIFEST, newline="", encoding="utf-8-sig") as f:
        return {int(r["index"]): _key(r["noise_file"])
                for r in csv.DictReader(f) if r["split"] == split}


def _key(path):
    """A join key that identifies a noise clip UNIQUELY.

    It must not be the basename. Noise clips are numbered from 001 inside each
    split's folder, so `ambient_003.wav` exists three times over -- once in
    noise_train, once in noise_validation, once in noise_testing -- and they
    are different recordings with different contamination. 116 basenames
    collide this way, and their scores genuinely disagree: ambient_002.wav is
    0.021 in testing and 0.896 in validation. Keying on the basename silently
    hands most mixtures another split's score, which is a join that looks like
    it worked while producing entirely fictional groups.

    So the key is the path from `noise_data` onward, which carries the split
    folder. Separators are normalised because the manifest is written with
    Windows backslashes.
    """
    parts = path.replace("\\", "/").split("/")
    if "noise_data" in parts:
        parts = parts[parts.index("noise_data"):]
    return "/".join(parts).lower()


def contamination_by_clip():
    """{split-qualified clip key: worst speech/music score} from the probe."""
    if not os.path.exists(PROBE_CSV):
        raise SystemExit(
            "no probe output. Run this first (note the ISOLATED venv):\n"
            "    .venv_yamnet/Scripts/python.exe scripts/yamnet_probe.py")
    with open(PROBE_CSV, newline="", encoding="utf-8") as f:
        out = {}
        for r in csv.DictReader(f):
            k = _key(r["filename"])
            if k in out:
                raise SystemExit(f"duplicate join key {k!r} -- the key is not "
                                 f"unique, so results would be fiction")
            out[k] = float(r["worst_junk_score"])
    return out


def bucket(score):
    if score >= STRONG_THRESHOLD:
        return "strong"
    if score >= WEAK_THRESHOLD:
        return "weak"
    return "clean"


def boot_diff(a, b, rng):
    """Bootstrap 95% CI for mean(a) - mean(b). Empty groups give None."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    diffs = np.array([
        rng.choice(a, len(a), replace=True).mean()
        - rng.choice(b, len(b), replace=True).mean()
        for _ in range(BOOTSTRAP)])
    return (float(a.mean() - b.mean()),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def score_split(split):
    """Every mixture in the split, scored with the shipped two-stage system."""
    import train
    import postfilter as pf
    import postfilter_test as pt
    from runlog import RunLog
    from mixing import FrozenMixtures
    from evaluate import si_snr, safe_stoi, safe_pesq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = RunLog(os.path.join(PROJECT_ROOT, "runs", STAGE1_RUN))
    stage1 = train.load_run_model(run, device)[0].eval()
    stage1.strip_aux()
    post = pf.load(os.path.join(PROJECT_ROOT, "models", POSTFILTER),
                   device)[0].eval()

    data = FrozenMixtures(split)
    noise_of = noise_file_by_index(split)
    print(f"stage 1      {STAGE1_RUN}")
    print(f"post-filter  {POSTFILTER}")
    print(f"device       {device}\n"
          f"split        {split}, {len(data)} frozen mixtures\n")

    rows = []
    for i in range(len(data)):
        m = data.describe(i)
        noisy = torch.from_numpy(m["noisy"][None]).to(device)
        with torch.no_grad():
            two, _ = pt.two_stage(stage1, post, noisy)
        out = two[0].cpu().numpy()
        clean = m["clean"]
        cat = m["category"]
        rows.append({
            "index": m["index"],
            "category": cat[0] if isinstance(cat, (list, tuple)) else cat,
            "noise_file": noise_of.get(m["index"], ""),
            "snr_db": m["snr_db"],
            "pesq": safe_pesq(clean, out),
            "stoi": safe_stoi(clean, out),
            "si_snr": si_snr(out, clean),
        })
        if (i + 1) % 25 == 0 or i + 1 == len(data):
            print(f"\r  scored {i + 1}/{len(data)}", end="", flush=True)
    print("\n")
    return rows


def report(rows, contam):
    rng = np.random.default_rng(0)
    for r in rows:
        r["bucket"] = bucket(contam.get(r["noise_file"], 0.0))
        r["contam"] = contam.get(r["noise_file"], 0.0)

    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["bucket"]].append(r)
    missing = [r for r in rows if r["noise_file"] not in contam]
    if missing:
        # An unmatched key means the JOIN is broken, not that a clip is clean.
        # The first version of this script defaulted them to clean, which is
        # how a basename collision produced fictional groups without anything
        # looking wrong. Refuse instead.
        raise SystemExit(
            f"{len(missing)} of {len(rows)} mixtures reference a noise clip "
            f"the probe has no score for.\n  That is a BROKEN JOIN, not clean "
            f"data -- defaulting them to clean would invent the result.\n"
            f"  example unmatched key: {missing[0]['noise_file']!r}")

    print("GROUP SIZES")
    for b in ("clean", "weak", "strong"):
        n = len(groups[b])
        print(f"  {b:<8}{n:>4} mixtures  ({100.0 * n / len(rows):5.1f}%)")

    print(f"\nSCORES BY CONTAMINATION LEVEL")
    print(f"  {'group':<8}{'n':>4}{'PESQ':>9}{'STOI':>9}{'SI-SNR':>9}")
    for b in ("clean", "weak", "strong"):
        g = groups[b]
        if not g:
            continue
        vals = []
        for k in METRICS:
            v = [r[k] for r in g if r[k] is not None]
            vals.append(np.mean(v) if v else float("nan"))
        print(f"  {b:<8}{len(g):>4}{vals[0]:>9.4f}{vals[1]:>9.4f}{vals[2]:>9.3f}")

    print(f"\nDIFFERENCE vs the clean group   (95% bootstrap CI, "
          f"{BOOTSTRAP} resamples)")
    print(f"  PESQ run-to-run noise floor is {NOISE_FLOOR_PESQ:.4f}, so a PESQ "
          f"gap below that is nothing.")
    for b in ("weak", "strong"):
        if not groups[b]:
            continue
        print(f"\n  {b} vs clean:")
        for k in METRICS:
            d = boot_diff([r[k] for r in groups[b]],
                          [r[k] for r in groups["clean"]], rng)
            if d is None:
                print(f"    {k:<8} too few clips")
                continue
            diff, lo, hi = d
            sig = "" if (lo < 0 < hi) else "  <-- interval excludes zero"
            print(f"    {k:<8}{diff:+9.4f}   [{lo:+.4f}, {hi:+.4f}]{sig}")

    # also the whole flagged set together, which is the headline comparison
    flagged = groups["weak"] + groups["strong"]
    if flagged and groups["clean"]:
        print(f"\n  ALL flagged ({len(flagged)}) vs clean ({len(groups['clean'])}):")
        for k in METRICS:
            d = boot_diff([r[k] for r in flagged],
                          [r[k] for r in groups["clean"]], rng)
            if d:
                diff, lo, hi = d
                sig = "" if (lo < 0 < hi) else "  <-- interval excludes zero"
                print(f"    {k:<8}{diff:+9.4f}   [{lo:+.4f}, {hi:+.4f}]{sig}")

    # ---- the confound control ------------------------------------------
    print(f"\nWITHIN EACH CATEGORY -- this is the comparison that controls for\n"
          f"category difficulty. Categories need {MIN_CATEGORY_N}+ mixtures on "
          f"both sides.")
    print(f"\n  {'category':<17}{'clean n':>8}{'flag n':>7}"
          f"{'PESQ clean':>12}{'PESQ flag':>11}{'diff':>9}")
    by_cat = collections.defaultdict(lambda: {"clean": [], "flag": []})
    for r in rows:
        by_cat[r["category"]]["clean" if r["bucket"] == "clean" else "flag"].append(r)
    shown = 0
    for cat in sorted(by_cat):
        c, f = by_cat[cat]["clean"], by_cat[cat]["flag"]
        if len(c) < MIN_CATEGORY_N or len(f) < MIN_CATEGORY_N:
            continue
        pc = np.mean([r["pesq"] for r in c if r["pesq"] is not None])
        pf_ = np.mean([r["pesq"] for r in f if r["pesq"] is not None])
        print(f"  {cat:<17}{len(c):>8}{len(f):>7}{pc:>12.4f}{pf_:>11.4f}"
              f"{pf_ - pc:>+9.4f}")
        shown += 1
    if not shown:
        print(f"\n  No category has {MIN_CATEGORY_N}+ mixtures on both sides, so"
              f" the confound CANNOT be\n  ruled out from this split alone. "
              f"Treat the headline gap as suggestive only,\n  and re-run on the "
              f"validation split for more clips.")

    # ---- what a clean exam would read ---------------------------------
    clean_pesq = [r["pesq"] for r in groups["clean"] if r["pesq"] is not None]
    all_pesq = [r["pesq"] for r in rows if r["pesq"] is not None]
    if clean_pesq:
        print(f"\nWHAT THIS WOULD MEAN")
        print(f"  reported now, whole split     PESQ {np.mean(all_pesq):.4f}")
        print(f"  clean mixtures only           PESQ {np.mean(clean_pesq):.4f}"
              f"   ({len(clean_pesq)} of {len(all_pesq)} mixtures)")
        print(f"\n  The second number is NOT a new headline -- it is a different,"
              f" easier exam\n  on a smaller sample, and swapping to it because "
              f"it looks better is exactly\n  the move HANDOFF section 4 warns "
              f"about. It is an estimate of what the\n  system would score on "
              f"noise that is really noise.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--split", default="testing",
                    choices=("testing", "validation"))
    ap.add_argument("--save", metavar="CSV", default=None,
                    help="also write per-mixture scores here, so splits can be "
                         "pooled for more statistical power")
    args = ap.parse_args()
    contam = contamination_by_clip()
    print(f"contamination from {os.path.relpath(PROBE_CSV, PROJECT_ROOT)}: "
          f"{len(contam)} noise clips known, "
          f"{sum(1 for v in contam.values() if v >= WEAK_THRESHOLD)} flagged\n")
    rows = score_split(args.split)
    report(rows, contam)
    if args.save:
        for r in rows:
            r["split"] = args.split
        with open(args.save, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-mixture scores written to {args.save}")
    print(f"\nNothing was trained, regenerated or relabelled. The frozen "
          f"mixtures and\nevery number already reported remain valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
