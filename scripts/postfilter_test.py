"""Train the second stage: a filter that corrects run 0026's output.

The network itself lives in scripts/postfilter.py; this trains it, scores it,
and saves it. Stage 1 is frozen throughout, so this only ever learns a
correction on top of a fixed model.

    python scripts/postfilter_test.py --lookahead 0          the deployable one
    python scripts/postfilter_test.py                        the upper bound
    python scripts/postfilter_test.py --lookahead 0 --channels 96 --batch 4

WHY --lookahead MATTERS MOST

Omitting it gives centred convolutions: 650 ms of future context, which no
real-time system can have. That was deliberate and it was the FIRST thing run.
The logic: if a post-filter with unlimited future cannot beat stage 1, a causal
one is strictly weaker and the streaming work never needs to start. It came back
positive, which turned the question into "what does causality cost".

--lookahead 0 is strictly causal and is what ships. Measured on the held-out
testing split: PESQ 1.9343 -> 1.9775, STOI 0.8626 -> 0.8646, SI-SNR 13.623 ->
13.966, for 0.3 ms of extra latency.

A TRAP WORTH KNOWING ABOUT

At 40 epochs the causal version kept only 45% of the non-causal gain, which read
as the price of causality. It was not -- all the early runs were still climbing
when they hit their epoch limit. Given 150 epochs the causal version reached
+0.0499 on validation against the non-causal run's +0.0394 at 40. Do not compare
two of these runs unless both converged.

WHAT WOULD MAKE A RESULT WORTH KEEPING

PESQ, STOI and SI-SNR all improving, by more than the noise floor (sd 0.0061 /
0.0004 / 0.0218). Stage 1 and the two-stage system are both measured here, in
one process through identical code, so the comparison has no wiggle room.
"""

import os
import sys
import copy
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_crn
import train as T
from postfilter import PostFilter, CHANNELS, DILATIONS
from runlog import RunLog
from mixing import SpeechNoiseMixtures, FrozenMixtures
from evaluate import si_snr, safe_pesq, safe_stoi, _mean
from pesq_labels import PesqPool

# ══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change while exploring
# ══════════════════════════════════════════════════════════════════




BATCH = 8
EPOCHS = 40
LR = 3e-4
EVAL_EVERY = 5

TRAIN_SNR_LOW_FRACTION = 0.25
# Share of TRAINING mixtures drawn from the hard -5..0 dB band. The project
# default is 0.10, chosen to reflect expected operating conditions; 0.25 makes
# every decibel from -5 to +15 equally represented instead.
#
# WHY: the first post-filter gained +0.0506 PESQ in the 0-5 dB band and only
# +0.0249 below 0 dB, and listening agreed it helped least on the quiet clips.
# One training example in ten was below 0 dB, so a 128k-parameter filter
# learning a correction had almost no practice on the hardest case. How often a
# condition OCCURS and how much practice it DESERVES are different questions.
#
# This changes the training diet only. Validation and testing are pre-mixed on
# disk, so the exam does not move and results stay comparable.
# Set to mixing.SNR_LOW_FRACTION to reproduce the first post-filter runs.

PATIENCE = 6
# Evaluations without a new best validation PESQ before stopping. At
# EVAL_EVERY 5 that is 30 epochs of no improvement, which is generous -- the
# first three runs were all still climbing when they hit their epoch limit, so
# stopping too eagerly is the more likely mistake here.



# ══════════════════════════════════════════════════════════════════
#  FIXED
# ══════════════════════════════════════════════════════════════════

STAGE1 = "0026_crn_comp03_spec600_la6_s0"
EPS = 1e-8


def two_stage(stage1, post, noisy):
    """Noisy waveform in, post-filtered waveform out. Stage 1 stays frozen."""
    with torch.no_grad():
        _, mask, _, _ = stage1(noisy)
        noisy_spec = model_crn.to_spectrogram(noisy)
        enh_spec = noisy_spec * mask
    out_spec = post(enh_spec, noisy_spec)
    return (model_crn.to_waveform(out_spec, noisy.shape[-1]),
            model_crn.to_waveform(enh_spec, noisy.shape[-1]))


@torch.no_grad()
def score_by_band(stage1, post, split, device,
                  bands=((-5, 0), (0, 5), (5, 10), (10, 15))):
    """PESQ change per input-SNR band, which is where the trade shows up.

    An overall mean can hide a filter that got better at quiet clips by getting
    worse at loud ones, and that trade is the whole risk of rebalancing the
    training diet. It is only visible split by band.
    """
    ds = FrozenMixtures(split)
    out = {b: [] for b in bands}
    for i in range(len(ds)):
        d = ds.describe(i)
        noisy = torch.from_numpy(np.ascontiguousarray(d["noisy"]))
        noisy = noisy.unsqueeze(0).to(device)
        two, enh = two_stage(stage1, post, noisy)
        a = safe_pesq(d["clean"], enh[0].cpu().numpy())
        b = safe_pesq(d["clean"], two[0].cpu().numpy())
        if a is None or b is None:
            continue
        for lo, hi in bands:
            if lo <= d["snr_db"] < hi:
                out[(lo, hi)].append(b - a)
                break
    return out


@torch.no_grad()
def score(stage1, post, loader, device, pool):
    """PESQ, STOI and SI-SNR for stage 1 alone and for the two-stage system.

    Both are computed here, in one pass, through identical code. Quoting a
    stored number for stage 1 instead would leave room for the comparison to be
    wrong in a way nobody could see.
    """
    rows = {k: {"pesq": [], "stoi": [], "sisnr": []}
            for k in ("stage1", "two_stage")}
    for noisy, clean, _ in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        out, enh = two_stage(stage1, post, noisy)
        c = clean.cpu().numpy()
        for name, sig in (("stage1", enh), ("two_stage", out)):
            s = sig.cpu().numpy()
            for i in range(len(c)):
                rows[name]["pesq"].append(safe_pesq(c[i], s[i]))
                rows[name]["stoi"].append(safe_stoi(c[i], s[i]))
                rows[name]["sisnr"].append(si_snr(s[i], c[i]))
    return {k: {m: _mean(v[m]) for m in v} for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=STAGE1)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--no-noisy-input", action="store_true",
                    help="strict post-filter: only sees stage 1's output")
    ap.add_argument("--channels", type=int, default=CHANNELS,
                    help="width of the post-filter")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--snr-low-fraction", type=float,
                    default=TRAIN_SNR_LOW_FRACTION,
                    help="share of TRAINING mixtures from -5..0 dB")
    ap.add_argument("--lookahead", type=int, default=None,
                    help="total future frames allowed, 10 ms each. 0 is "
                         "strictly causal. Omit for the non-causal upper bound")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    stage1, ck, ep = T.load_run_model(RunLog.open(args.stage1), device)
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad_(False)

    post = PostFilter(channels=args.channels,
                      use_noisy=not args.no_noisy_input,
                      lookahead=args.lookahead).to(device)
    n_post = sum(p.numel() for p in post.parameters())
    print(f"stage 1       {args.stage1}, {ck} epoch {ep}, frozen")
    print(f"post-filter   {n_post:,} parameters, "
          f"{'sees noisy input too' if post.use_noisy else 'output only'}")
    print(f"context       {post.past} frames of past ({post.past * 10} ms), "
          f"{post.lookahead} of future ({post.lookahead * 10} ms)")
    if args.lookahead is None:
        print(f"              NON-CAUSAL upper bound, not deployable")
    else:
        print(f"              adds {post.lookahead * 10} ms to the 177 ms "
              f"budget, giving {177 + post.lookahead * 10} ms\n")

    # draw_snr reads this at call time, and only the TRAINING mixtures are made
    # on the fly -- the frozen sets are already wavs on disk -- so this changes
    # what the filter practises on and nothing about what it is marked against.
    import mixing
    was = mixing.SNR_LOW_FRACTION
    mixing.SNR_LOW_FRACTION = args.snr_low_fraction
    print(f"training SNR diet: {was:.0%} -> "
          f"{args.snr_low_fraction:.0%} drawn from -5..0 dB\n")

    train_loader = DataLoader(SpeechNoiseMixtures("train"), batch_size=args.batch,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(FrozenMixtures("validation"), batch_size=args.batch,
                            shuffle=False, num_workers=0)
    opt = torch.optim.Adam(post.parameters(), lr=LR)

    with PesqPool() as pool:
        # The zero-init identity check, made ON THE WAVEFORM. Comparing the two
        # PESQ scores instead looks equivalent and is not: PESQ is not a smooth
        # function of its input, so the 1e-7 float differences that cuDNN
        # produces by choosing a different convolution algorithm come out as a
        # PESQ wobble around 1e-6 that changes from run to run. Measured here:
        # waveforms agree to 1.8e-07 on a signal of rms 8.9e-02, while the PESQ
        # difference was 2.7e-07 one run and 1.4e-06 the next. The waveform is
        # the claim; the PESQ number is that claim seen through a chaotic lens.
        with torch.no_grad():
            noisy, _, _ = next(iter(val_loader))
            out, enh = two_stage(stage1, post, noisy.to(device))
            drift = float((out - enh).abs().max())
            rms = float(enh.pow(2).mean().sqrt())
        print(f"identity check at init: max waveform difference {drift:.2e} "
              f"on a signal of rms {rms:.2e}")
        assert drift < 1e-5, "zero-initialised post-filter is not the identity"

        base = score(stage1, post, val_loader, device, pool)
        print(f"stage 1 baseline: PESQ {base['stage1']['pesq']:.4f}  "
              f"STOI {base['stage1']['stoi']:.4f}  "
              f"SI-SNR {base['stage1']['sisnr']:.3f}")

        print(f"\n{'ep':>4}{'train loss':>12}{'PESQ':>9}{'STOI':>9}"
              f"{'SI-SNR':>9}{'sec':>7}")
        best, best_state, best_epoch, stale = None, None, 0, 0
        for epoch in range(1, args.epochs + 1):
            post.train()
            t0, total, n = time.perf_counter(), 0.0, 0
            for noisy, clean, _ in train_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out, _ = two_stage(stage1, post, noisy)
                loss = (T.si_snr_loss(out, clean)
                        + T.SPECTRAL_LOSS_WEIGHT * T.spectral_loss(out, clean))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(noisy)
                n += len(noisy)
            secs = time.perf_counter() - t0
            if epoch % EVAL_EVERY == 0 or epoch == args.epochs:
                post.eval()
                s = score(stage1, post, val_loader, device, pool)["two_stage"]
                mark = ""
                if best is None or s["pesq"] > best["pesq"]:
                    best, best_epoch, stale = s, epoch, 0
                    best_state = copy.deepcopy(post.state_dict())
                    mark = " *"
                else:
                    stale += 1
                print(f"{epoch:>4}{total / n:>12.4f}{s['pesq']:>9.4f}"
                      f"{s['stoi']:>9.4f}{s['sisnr']:>9.3f}{secs:>7.1f}{mark}")
                if stale >= PATIENCE:
                    print(f"  no new best for {stale} evaluations, stopping")
                    break
            else:
                print(f"{epoch:>4}{total / n:>12.4f}{'':>27}{secs:>7.1f}")

        # The held-out check, run ONCE, on the weights validation chose. The
        # testing split has never been used for training and never for picking
        # anything, which is the only reason a number from it means something.
        post.load_state_dict(best_state)
        post.eval()
        test_loader = DataLoader(FrozenMixtures("testing"), batch_size=args.batch,
                                 shuffle=False, num_workers=0)
        print(f"\nheld-out testing split, using the epoch {best_epoch} weights")
        held = score(stage1, post, test_loader, device, pool)

    a, b = base["stage1"], best
    print(f"\n{'':<14}{'PESQ':>10}{'STOI':>10}{'SI-SNR':>10}")
    print("-" * 44)
    print(f"{'stage 1 alone':<14}{a['pesq']:>10.4f}{a['stoi']:>10.4f}"
          f"{a['sisnr']:>10.3f}")
    print(f"{'two-stage':<14}{b['pesq']:>10.4f}{b['stoi']:>10.4f}"
          f"{b['sisnr']:>10.3f}")
    print(f"{'difference':<14}{b['pesq'] - a['pesq']:>+10.4f}"
          f"{b['stoi'] - a['stoi']:>+10.4f}{b['sisnr'] - a['sisnr']:>+10.3f}")
    print("-" * 44)
    floor = (0.0061, 0.0004, 0.0218)
    sig = [(b[k] - a[k]) / (f * 2 ** 0.5)
           for k, f in zip(("pesq", "stoi", "sisnr"), floor)]
    print(f"{'sigma':<14}{sig[0]:>+10.1f}{sig[1]:>+10.1f}{sig[2]:>+10.1f}")

    ha, hb = held["stage1"], held["two_stage"]
    print(f"\nHELD-OUT TESTING{'':<8}{'PESQ':>10}{'STOI':>10}{'SI-SNR':>10}")
    print("-" * 54)
    print(f"{'stage 1 alone':<24}{ha['pesq']:>10.4f}{ha['stoi']:>10.4f}"
          f"{ha['sisnr']:>10.3f}")
    print(f"{'two-stage':<24}{hb['pesq']:>10.4f}{hb['stoi']:>10.4f}"
          f"{hb['sisnr']:>10.3f}")
    print(f"{'difference':<24}{hb['pesq'] - ha['pesq']:>+10.4f}"
          f"{hb['stoi'] - ha['stoi']:>+10.4f}"
          f"{hb['sisnr'] - ha['sisnr']:>+10.3f}")
    print(f"{'target':<24}{2.5:>10.4f}{0.85:>10.4f}{15.0:>10.3f}")
    print("-" * 54)

    bands = score_by_band(stage1, post, "testing", device)
    print(f"\n{'held-out by input SNR':<22}{'clips':>7}{'mean dPESQ':>13}"
          f"{'worse':>8}")
    for (lo, hi), v in bands.items():
        if v:
            print(f"{f'{lo:>3} to {hi:>3} dB':<22}{len(v):>7}"
                  f"{sum(v) / len(v):>+13.4f}"
                  f"{100 * sum(1 for x in v if x < 0) / len(v):>7.0f}%")
    print("\nthe first post-filter, 10% low-SNR diet, scored:")
    print("   -5 to   0 dB      12      +0.0249     17%")
    print("    0 to   5 dB      42      +0.0506      5%")
    print("    5 to  10 dB      31      +0.0361     10%")
    print("   10 to  15 dB      40      +0.0484     15%")

    won = all(x > 0 for x in sig)
    print(f"\n{'WORTH CONTINUING' if won else 'NOT WORTH CONTINUING'}")
    if not won:
        print("A post-filter with over a second of centred context, seeing the")
        print("noisy input as well, cannot improve all three metrics. A causal")
        print("one is strictly weaker, so the streaming version is closed too.")
    else:
        # EVERY knob that changes the model goes in the filename. This bit has
        # bitten twice. First a single fixed name meant three runs all wrote to
        # "postfilter_noncausal.pt", leaving the 40 ms causal weights under a
        # name that said otherwise. Then a name carrying only the lookahead let
        # the 25% diet run overwrite the 10% one, which scored better. Twice is
        # the pattern, not the instance: if a knob changes what the weights ARE,
        # it belongs in what they are CALLED.
        la = "noncausal" if args.lookahead is None else f"la{args.lookahead}"
        name = (f"postfilter_ch{args.channels}_{la}"
                f"_snr{round(args.snr_low_fraction * 100)}.pt")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "models", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": post.state_dict(),
                    "channels": args.channels,
                    "lookahead": args.lookahead,
                    "past_frames": post.past,
                    "future_frames": post.lookahead,
                    "use_noisy_input": post.use_noisy,
                    "snr_low_fraction": args.snr_low_fraction,
                    "dilations": list(DILATIONS),
                    "stage1": args.stage1,
                    "validation": b, "held_out": hb}, path)
        print(f"weights saved to models/{name}")


if __name__ == "__main__":
    main()
