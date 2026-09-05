"""The second stage: a small causal filter that corrects stage 1's output.

Stage 1 (the CRN) can only turn sound DOWN -- it produces a mask, and a mask
cannot restore what the noise destroyed. This stage predicts a residual to ADD
to that output, which is not the same operation and is not bounded by the same
ceiling.

WHY ADDITIVE MATTERS, ALGEBRAICALLY

A post-filter that applies its own mask gives

    mask2 * (mask1 * noisy)  =  (mask1 * mask2) * noisy

which is one mask on the noisy input -- a re-parameterisation of stage 1, capped
by the same ideal-mask ceiling of 3.389 that stage 1 already falls short of.
Adding a residual is not collapsible that way.

MEASURED, on the held-out testing split, with run 0026 frozen:

                              PESQ      STOI    SI-SNR
    stage 1 alone           1.9343    0.8626    13.623
    two-stage               1.9775    0.8646    13.966

Strictly causal, so it buys no lookahead: total latency goes 176.6 -> 176.9 ms,
all of it compute. 128k parameters against stage 1's 9.3M.

This module holds only the network. scripts/postfilter_test.py trains it and
scripts/demo_core.py serves it.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model_crn

# ══════════════════════════════════════════════════════════════════
#  TUNING KNOBS -- safe to change while exploring
# ══════════════════════════════════════════════════════════════════

CHANNELS = 48
# Width of the filter. WIDER can express more correction and costs memory
# quadratically in activations, since nothing here downsamples the spectrogram.
# At 48 with batch 8 it fits beside the frozen stage 1 on an 8 GB card. The
# trained model had not fully plateaued, so this is the first knob to raise.

DILATIONS = (1, 2, 4, 8, 16, 32)
# Time dilation per layer, which is where the context comes from: the receptive
# field is 1 + 2*sum(dilations) frames. Causal, that is 1300 ms all behind.

LOOKAHEAD_FRAMES = 0
# Total future frames the filter may use, at 10 ms each, added directly to the
# latency budget.
#   0     strictly causal -- costs NO extra latency, and is what shipped.
#   4     40 ms. Measured +0.0054 PESQ over strictly causal at 40 epochs, which
#         did not look worth 40 ms.
#   None  centred convolutions: 650 ms of future. The non-causal UPPER BOUND
#         used to decide whether the idea was worth pursuing at all. Never
#         deployable.

RESIDUAL_COMPRESSION = 0.3
# The residual is predicted in a compressed complex domain, the same power the
# spectral grader uses. Raw complex values span orders of magnitude and are
# dominated by loud vowels, so a residual predicted on them would only ever
# learn to adjust the loudest parts.

USE_NOISY_INPUT = True
# Whether the filter also sees the original noisy audio. Masking is lossy, so
# a filter seeing only stage 1's output is solving the same problem with less
# information than stage 1 had.

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
    layer may take from ahead.
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
    """Hand a total future-frame budget to the earliest layers first.

    Earliest first, where the dilation is smallest, so the budget buys
    fine-grained access to the next few frames rather than a coarse peek far
    ahead. Same reasoning as lookahead_plan in the encoder.
    """
    rights = []
    for d in dilations:
        take = d if budget is None else min(d, budget)
        rights.append(take)
        if budget is not None:
            budget -= take
    return rights


class PostFilter(nn.Module):
    """Predicts a residual to ADD to stage 1's spectrum."""

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
        # Zero-initialised so the filter starts as the identity: at step 0 the
        # residual is exactly zero and the two-stage output equals stage 1.
        # Any gain is therefore learned rather than an artefact of the start.
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
        return decompress(e + torch.complex(r[:, 0], r[:, 1]))


def load(path, device):
    """Rebuild a saved post-filter from its own recorded configuration.

    The checkpoint carries its lookahead and input choice, so a filter trained
    with one setting cannot be silently rebuilt with another -- which is how a
    causal model quietly becomes a non-causal one on the way to deployment.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in ck:                    # a bare state_dict
        model = PostFilter().to(device)
        model.load_state_dict(ck)
        return model, {}
    model = PostFilter(use_noisy=ck.get("use_noisy_input", USE_NOISY_INPUT),
                       lookahead=ck.get("lookahead", LOOKAHEAD_FRAMES)).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, ck


def apply(model, noisy, mask):
    """Noisy waveform and stage 1's mask, both batched tensors -> waveform."""
    noisy_spec = model_crn.to_spectrogram(noisy)
    out_spec = model(noisy_spec * mask, noisy_spec)
    return model_crn.to_waveform(out_spec, noisy.shape[-1])


def self_test():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for la in (0, 4, None):
        m = PostFilter(lookahead=la).to(device)
        x = torch.randn(2, 301, 257, dtype=torch.complex64, device=device)
        with torch.no_grad():
            out = m(x, x)
        print(f"  lookahead {str(la):<5} past {m.past:>3} future {m.lookahead:>2}"
              f"   out {tuple(out.shape)}   identity "
              f"{float((out - x).abs().max()):.1e}")
        assert out.shape == x.shape

    # Causality, CHECKED rather than trusted -- the padding arithmetic is one
    # off-by-one away from leaking the future, and nothing downstream would say.
    print("\n  causality: change every frame after 150, see what moves")
    for la in (0, 4):
        m = PostFilter(lookahead=la).to(device)
        nn.init.normal_(m.out.conv.weight, std=0.1)
        m.eval()      # BatchNorm in TRAIN mode averages across all time
        a = torch.randn(1, 301, 257, dtype=torch.complex64, device=device)
        b = a.clone()
        b[:, 151:] = torch.randn_like(b[:, 151:])
        with torch.no_grad():
            d = (m(a, a) - m(b, b)).abs().amax(dim=(0, 2))
        first = int((d > 1e-6).nonzero()[0])
        print(f"    lookahead {la}: first frame affected {first}, "
              f"expected {151 - la}")
        assert first == 151 - la, "future information is leaking"
    print("\npostfilter self-test passed.")


if __name__ == "__main__":
    self_test()
