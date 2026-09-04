"""Can a second stage improve run 0026's output at all?

THIS IS AN UPPER BOUND, NOT A DEPLOYABLE THING

The eventual idea is a causal, streaming post-filter that cleans up what the
mask leaves behind. This is deliberately NOT that. It is the same idea with
every constraint removed: it sees the whole clip including the future, it has a
receptive field of over a second, and it has no latency budget at all.

The logic is the one that has paid off repeatedly in this project. If a
post-filter with unlimited future context cannot beat stage 1, then a causal
one -- which is strictly weaker -- certainly cannot, and the week of streaming
engineering never needs to start. An hour here decides it.

THREE DESIGN CHOICES, EACH LOAD-BEARING

**Additive, not multiplicative.** Our output is mask1 * noisy. A post-filter
that applies its own mask gives mask2 * (mask1 * noisy) = (mask1*mask2) * noisy,
which is algebraically just ONE mask on the noisy input -- a re-parameterisation
of stage 1, bounded by the same ideal-mask ceiling of 3.389 we already know we
fall short of. Predicting a RESIDUAL TO ADD is not collapsible that way, and it
can put back energy the mask destroyed.

**It sees the noisy input as well as the enhanced output.** Masking is lossy:
whatever stage 1 crushed toward zero is gone, so a filter looking only at the
output is solving the same problem with strictly less information. Giving it
both makes this a generous test, which is the point -- it makes a null result
mean something.

**It starts as the identity.** The final convolution is zero-initialised, so at
epoch 0 the residual is exactly zero and the two-stage output is bit-identical
to stage 1. Any gain is therefore something the post-filter learned, not an
accident of initialisation, and the run can only be read as "what did adding
this buy".

WHAT WOULD MAKE IT WORTH CONTINUING

Beating stage 1 on PESQ, STOI and SI-SNR together, by more than the noise floor
(sd 0.0061 / 0.0004 / 0.0218). Both are measured here in the same process,
through the same code, so the comparison has no wiggle room.
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
from runlog import RunLog
from mixing import SpeechNoiseMixtures, FrozenMixtures
from evaluate import si_snr, safe_pesq, safe_stoi, _mean
from pesq_labels import PesqPool

# ══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change while exploring
# ══════════════════════════════════════════════════════════════════

CHANNELS = 48
# Width of the post-filter. WIDER can express more correction and costs memory
# quadratically in activations, since nothing here downsamples the spectrogram.
# At 48 with batch 8 the whole thing fits beside the frozen stage 1 on an 8 GB
# card; raising it means lowering BATCH.

DILATIONS = (1, 2, 4, 8, 16, 32)
# Time dilation per layer. This is where the context comes from: the receptive
# field is 1 + 2*sum(dilations) = 127 frames, or 1.27 seconds, CENTRED on the
# frame being corrected. That is roughly 0.6 s of future -- deliberately far
# more than any real-time system could ever have.

LOOKAHEAD_FRAMES = None
# How many frames of the FUTURE the post-filter may use in total, at 10 ms each.
#   None  centred convolutions, which is the non-causal upper bound: 65 frames
#         = 650 ms of future. Not deployable; it is what the first run measured.
#   0     strictly causal. Every layer is fed entirely from the past, so the
#         filter costs NO extra latency at all -- it can run inside the delay
#         the system already pays.
#   n     n frames of future, added directly to the 177 ms budget.
#
# The budget is handed to the EARLIEST layers first, where the dilation is
# smallest, so what it buys is fine-grained access to the next few frames
# rather than a coarse peek far ahead. That is the same reasoning
# lookahead_plan uses in the encoder.

BATCH = 8
EPOCHS = 40
LR = 3e-4
EVAL_EVERY = 5

PATIENCE = 6
# Evaluations without a new best validation PESQ before stopping. At
# EVAL_EVERY 5 that is 30 epochs of no improvement, which is generous -- the
# first three runs were all still climbing when they hit their epoch limit, so
# stopping too eagerly is the more likely mistake here.

RESIDUAL_COMPRESSION = 0.3
# The residual is predicted in a compressed complex domain, the same power the
# spectral grader uses. Raw complex values span orders of magnitude and are
# dominated by loud vowels, so a residual predicted on them would only ever
# learn to adjust the loudest parts.

USE_NOISY_INPUT = True
# Whether the post-filter also sees the original noisy audio. False makes it a
# strict post-filter that only sees stage 1's output, which is the weaker and
# more honest version of the question. True is the generous version and is the
# default because a null result then rules out both.

# ══════════════════════════════════════════════════════════════════
#  FIXED
# ══════════════════════════════════════════════════════════════════

STAGE1 = "0026_crn_comp03_spec600_la6_s0"
EPS = 1e-8


def compress(spec, p=RESIDUAL_COMPRESSION):
    """Complex spectrum -> same phase, magnitude raised to p."""
    mag = spec.abs()
    return ((mag + EPS) ** p) * (spec / (mag + EPS))


def decompress(z, p=RESIDUAL_COMPRESSION):
    mag = z.abs()
    return ((mag + EPS) ** (1.0 / p)) * (z / (mag + EPS))


class TimeConv(nn.Module):
    """A 3x3 convolution whose past and future reach are set explicitly.

    A dilated 3-wide kernel spans offsets -d, 0, +d, so it needs 2d frames of
    padding to keep the length. WHERE that padding goes is the whole question:
    all of it on the left and the kernel is fed entirely from the past; split
    evenly and it reads the future. `right` is how many of those frames the
    layer is allowed to take from ahead.
    """

    def __init__(self, in_ch, out_ch, dilation, right):
        super().__init__()
        self.right = right
        # ZeroPad2d takes (freq_left, freq_right, time_left, time_right).
        self.pad = nn.ZeroPad2d((1, 1, 2 * dilation - right, right))
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=0,
                              dilation=(dilation, 1))

    def forward(self, x):
        return self.conv(self.pad(x))


def allocate_lookahead(dilations, budget):
    """Hand a total future-frame budget to the earliest layers first."""
    rights = []
    for d in dilations:
        take = d if budget is None else min(d, budget)
        rights.append(take)
        if budget is not None:
            budget -= take
    return rights


class PostFilter(nn.Module):
    """Predicts a residual to ADD to stage 1's spectrum.

    How far into the future it may look is set by `lookahead`: None gives
    centred convolutions -- the non-causal upper bound -- and an integer caps
    the total future reach at that many frames. 0 is strictly causal.
    """

    def __init__(self, channels=CHANNELS, dilations=DILATIONS,
                 use_noisy=USE_NOISY_INPUT, lookahead=LOOKAHEAD_FRAMES):
        super().__init__()
        self.use_noisy = use_noisy
        in_ch = 4 if use_noisy else 2
        # The input and output convolutions are dilation 1 but still read a
        # frame either side, so they are counted in the budget like any other.
        plan = (1,) + tuple(dilations) + (1,)
        rights = allocate_lookahead(plan, lookahead)
        self.lookahead = sum(rights)
        self.past = sum(2 * d - r for d, r in zip(plan, rights))

        layers = [TimeConv(in_ch, channels, 1, rights[0]), nn.ELU()]
        for d, r in zip(dilations, rights[1:-1]):
            layers += [TimeConv(channels, channels, d, r),
                       nn.BatchNorm2d(channels), nn.ELU()]
        self.body = nn.Sequential(*layers)
        self.out = TimeConv(channels, 2, 1, rights[-1])
        # Zero-initialised so the post-filter starts as the identity and the
        # experiment reads as "what did adding this buy", not "what did
        # replacing stage 1 with a random correction do".
        nn.init.zeros_(self.out.conv.weight)
        nn.init.zeros_(self.out.conv.bias)

    def forward(self, enhanced_spec, noisy_spec):
        e = compress(enhanced_spec)
        chans = [e.real, e.imag]
        if self.use_noisy:
            n = compress(noisy_spec)
            chans += [n.real, n.imag]
        x = torch.stack(chans, dim=1)                    # (B, C, T, F)
        r = self.out(self.body(x))                       # (B, 2, T, F)
        corrected = e + torch.complex(r[:, 0], r[:, 1])
        return decompress(corrected)


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

    post = PostFilter(use_noisy=not args.no_noisy_input,
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

    train_loader = DataLoader(SpeechNoiseMixtures("train"), batch_size=BATCH,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(FrozenMixtures("validation"), batch_size=BATCH,
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
        test_loader = DataLoader(FrozenMixtures("testing"), batch_size=BATCH,
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

    won = all(x > 0 for x in sig)
    print(f"\n{'WORTH CONTINUING' if won else 'NOT WORTH CONTINUING'}")
    if not won:
        print("A post-filter with over a second of centred context, seeing the")
        print("noisy input as well, cannot improve all three metrics. A causal")
        print("one is strictly weaker, so the streaming version is closed too.")
    else:
        # Named by configuration. A single fixed filename meant three runs in
        # a row all wrote to "postfilter_noncausal.pt", so the file ended up
        # holding the 40 ms causal weights under a name that said otherwise --
        # which is how a wrong conclusion gets drawn six weeks later.
        tag = ("noncausal" if args.lookahead is None
               else f"causal_la{args.lookahead}")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "backups", f"postfilter_{tag}.pt")
        torch.save({"state_dict": post.state_dict(),
                    "lookahead": args.lookahead,
                    "past_frames": post.past,
                    "future_frames": post.lookahead,
                    "use_noisy_input": post.use_noisy,
                    "stage1": args.stage1,
                    "validation": b}, path)
        print(f"weights saved to backups/{os.path.basename(path)}")


if __name__ == "__main__":
    main()
