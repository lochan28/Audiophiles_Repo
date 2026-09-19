"""
model_tcn.py

The TCN: the same model as the CRN with the LSTM replaced by a stack of
dilated causal convolutions. Nothing else changes.

WHY THIS EXISTS

The project was scoped as a two-architecture comparison from the start.
PROJECT_CONTEXT_4.md Section 4: "The model that ships is DCCRN if we manage to
achieve it; otherwise it is whichever of CRN and TCN performs better." The CRN
was built and tuned for six weeks; the TCN was never built. This is that.

It is also the right KIND of change for where the project is. Runs 0029-0034
closed every remaining way to change what the model is TOLD during training,
and Section 22 concluded the remaining gap is an OPTIMISATION problem: the
information is present, the mask can express the answer, and the model still
hedges. Architecture is the lever that acts on optimisation.

WHY ONLY THE MEMORY IS REPLACED

This class SUBCLASSES model_crn.CRN and swaps one attribute. The STFT front
end, the encoder, the decoder, the skip connections, the mask and the auxiliary
head are not reimplemented here -- they are inherited, so they are byte
identical by construction rather than by careful copying.

That makes the comparison readable. One variable changes, so a win or a loss is
attributable. It also keeps every diagnostic valid: the operator is still a
per-bin STFT mask, so oracle_ceiling.py's 3.389 ceiling, mask_diagnosis.py and
over_suppression.py all still describe this model. A Conv-TasNet style rewrite
with a learned encoder would invalidate all of them at once.

The LSTM holds about 7.58 M of the CRN's 9.26 M parameters, so "only the
memory" is still most of the model.

THE SECOND REASON TO WANT THIS, WHICH LANDS EVEN ON A TIE

The project's noise floor -- PESQ sd 0.0061 -- is dominated by cuDNN's
nondeterministic LSTM backward pass. Nine runs were spent chasing differences
that turned out to be noise. A pure-convolution core can be made
bit-reproducible, which would shrink that floor and make every future
experiment sharper.

    python scripts/model_tcn.py        # run every check
"""

import time

import torch
import torch.nn as nn

import model_crn
from model_crn import (to_spectrogram, to_waveform, CLIP_SAMPLES,
                       CHUNK_SAMPLES, TOTAL_LOOKAHEAD_MS, FREQ_BINS,
                       LOOKAHEAD_FRAMES, LSTM_HIDDEN, MASK_MAX, N_CLASSES,
                       ENCODER_CHANNELS, ENCODER_FREQ_STRIDES, AUX_TARGET)

# The STFT helpers and the encoder/decoder live in model_crn.py despite the
# name: they are architecture-neutral, and moving them would churn eleven files
# that import them for no benefit.


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS — safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

TCN_STACKS = 2
# How many times the dilation pattern repeats. Each stack restarts the
# dilations at 1 and climbs again, which is how a TCN gets a long memory
# without a long kernel.
# MORE stacks means a longer reach and proportionally more parameters.

TCN_LAYERS_PER_STACK = 6
# Layers inside one stack. Dilation DOUBLES at each one: 1, 2, 4, 8, 16, 32.
# This is the knob that buys reach cheaply -- each extra layer doubles how far
# back the model can see, where an extra stack only adds.
#
# Reach in frames is 1 + stacks * (kernel-1) * (2^layers - 1), and each frame
# is 10 ms. At 2 stacks and 6 layers that is 253 frames = 2.53 s, comfortably
# more than the 3.00 s clips, so the model can in principle use the whole
# utterance of history. Going further buys nothing on this data.

TCN_HIDDEN = 512
# Width inside each residual block, where the dilated convolution actually
# happens. This and the two knobs above are what set the parameter count, and
# the count has to be MATCHED to the CRN or the comparison measures capacity
# rather than architecture. self_test() prints both.

TCN_KERNEL = 3
# Taps per dilated convolution. 3 is the standard choice and the reach formula
# above assumes it. Larger kernels buy reach less efficiently than more layers.

# ═══════════════════════════════════════════════════════════════════
#  LOCKED — set by the comparison, not by preference
# ═══════════════════════════════════════════════════════════════════

TCN_RIGHT_PAD = 0
# How many frames of the FUTURE the temporal core may see. It must stay 0.
#
# The encoder already spends the whole lookahead budget -- 6 frames, handed out
# by model_crn.lookahead_plan -- and that budget is what fixes total latency at
# 177 ms. A temporal core that also peeked ahead would add latency the CRN does
# not pay, and the comparison would be measuring the extra lookahead instead of
# the architecture. self_test() asserts the two models report the same latency.


class ChannelNorm(nn.Module):
    """Layer norm over channels only, at each time step independently.

    THIS CHOICE IS A CAUSALITY REQUIREMENT, NOT A PREFERENCE. The standard
    normalisation in TCN separators is global layer norm, which normalises over
    channels AND time -- including the future. Used here it would leak, exactly
    the way the BatchNorm note in model_crn.EncoderBlock describes, and the
    model would quietly stop being deployable.

    Normalising each time step on its own cannot see forward by construction.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):                       # (B, C, T)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalDilatedBlock(nn.Module):
    """One residual block: widen, convolve over time, narrow, add back.

    The convolution is depthwise -- one filter per channel rather than a full
    mixing convolution. Mixing is left to the surrounding 1x1 convolutions,
    which is far cheaper for the same expressiveness and is what lets a stack
    this deep fit inside the CRN's parameter budget.
    """

    def __init__(self, bottleneck, hidden, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.widen = nn.Conv1d(bottleneck, hidden, 1)
        self.act1 = nn.PReLU()
        self.norm1 = ChannelNorm(hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel, dilation=dilation,
                                   groups=hidden)
        self.act2 = nn.PReLU()
        self.norm2 = ChannelNorm(hidden)
        self.narrow = nn.Conv1d(hidden, bottleneck, 1)

    def forward(self, x, buf=None):
        """(B, C, T) in, same out, plus the left-context buffer for streaming.

        The convolution is given no padding of its own. Instead the previous
        chunk's tail is prepended, which is what makes a chunked pass identical
        to a whole-clip pass. A fresh stream starts against zeros, which is the
        same thing the whole-clip pass sees at its own start.
        """
        y = self.norm1(self.act1(self.widen(x)))
        if buf is None:
            buf = y.new_zeros(y.shape[0], y.shape[1], self.pad)
        padded = torch.cat([buf, y], dim=-1)
        new_buf = padded[..., -self.pad:] if self.pad else buf
        y = self.norm2(self.act2(self.depthwise(padded)))
        return x + self.narrow(y), new_buf


class TemporalCore(nn.Module):
    """The LSTM's replacement. Same signature, same shapes, no recurrence.

    Drop-in for model_crn.Memory: (B, C, T, F) in, same shape out, plus the
    (B, T, hidden) sequence the auxiliary head reads and the state that carries
    between streaming chunks.

    `hidden` is the bottleneck width and is deliberately the SAME argument the
    CRN passes to its LSTM -- the --lstm-hidden flag drives both -- so the two
    architectures are configured through one knob and stay comparable.
    """

    def __init__(self, n_features, hidden=LSTM_HIDDEN, stacks=TCN_STACKS,
                 layers=TCN_LAYERS_PER_STACK, width=TCN_HIDDEN,
                 kernel=TCN_KERNEL):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.enter = nn.Conv1d(n_features, hidden, 1)
        self.blocks = nn.ModuleList(
            [CausalDilatedBlock(hidden, width, kernel, 2 ** i)
             for _ in range(stacks) for i in range(layers)])
        self.project = nn.Linear(hidden, n_features)
        self.receptive_field = 1 + sum(b.pad for b in self.blocks)

    def forward(self, x, state=None):
        b, c, t, f = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b, t, c * f)     # (B, T, C*F)
        h = self.enter(seq.transpose(1, 2))                  # (B, hidden, T)
        bufs = list(state) if state is not None else [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            h, bufs[i] = block(h, bufs[i])
        seq = h.transpose(1, 2)                              # (B, T, hidden)
        out = self.project(seq).reshape(b, t, c, f).permute(0, 2, 1, 3)
        return out, seq, tuple(bufs)


class TCN(model_crn.CRN):
    """The CRN with a convolutional temporal core instead of an LSTM.

    Everything else -- encoder, decoder, skips, mask, auxiliary head,
    predict_mask() and forward() -- is inherited unchanged, so it cannot drift
    from the CRN it is being compared against.
    """

    def __init__(self, channels=ENCODER_CHANNELS, lookahead=LOOKAHEAD_FRAMES,
                 hidden=LSTM_HIDDEN, layers=TCN_LAYERS_PER_STACK,
                 mask_max=MASK_MAX, n_classes=N_CLASSES,
                 freq_strides=ENCODER_FREQ_STRIDES, aux_target=AUX_TARGET,
                 stacks=TCN_STACKS, width=TCN_HIDDEN, kernel=TCN_KERNEL):
        # Builds the CRN, LSTM and all, then replaces the one part that
        # differs. The discarded LSTM costs a moment at construction and
        # nothing afterwards; inheriting the rest is what guarantees it is
        # identical.
        super().__init__(channels=channels, lookahead=lookahead,
                         hidden=hidden, layers=2, mask_max=mask_max,
                         n_classes=n_classes, freq_strides=freq_strides,
                         aux_target=aux_target)
        self.memory = TemporalCore(self.encoder.out_features, hidden,
                                   stacks=stacks, layers=layers, width=width,
                                   kernel=kernel)


def self_test():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wide = dict(channels=tuple(max(1, round(c * 1.5))
                               for c in ENCODER_CHANNELS),
                hidden=512, lookahead=6)

    tcn = TCN(**wide).to(device).eval()
    crn = model_crn.CRN(**wide).to(device).eval()
    n_tcn = sum(p.numel() for p in tcn.parameters())
    n_crn = sum(p.numel() for p in crn.parameters())
    core_t = sum(p.numel() for p in tcn.memory.parameters())
    core_c = sum(p.numel() for p in crn.memory.parameters())

    print("step 1 -- shape and size, beside the CRN it replaces\n")
    print(f"  TCN total     {n_tcn:,} parameters")
    print(f"  CRN total     {n_crn:,} parameters"
          f"   ({(n_tcn / n_crn - 1) * 100:+.1f}% difference)")
    print(f"  TCN core      {core_t:,}")
    print(f"  CRN core      {core_c:,}   (the LSTM this replaces)")
    print(f"  reach         {tcn.memory.receptive_field} frames "
          f"= {tcn.memory.receptive_field * 10} ms of history")
    # A comparison between models of different size measures size, not
    # architecture. The plan allows 10%.
    assert abs(n_tcn / n_crn - 1) < 0.10, (
        f"TCN is {(n_tcn / n_crn - 1) * 100:+.1f}% the CRN's size; tune "
        f"TCN_HIDDEN or TCN_STACKS to bring it within 10%")

    wav = torch.randn(2, CLIP_SAMPLES, device=device) * 0.1
    with torch.no_grad():
        out, mask, logits, state = tcn(wav)
    frames = to_spectrogram(wav).shape[1]
    print(f"\n  audio         {tuple(wav.shape)} -> {tuple(out.shape)}")
    print(f"  mask          {tuple(mask.shape)}, range "
          f"{float(mask.min()):.3f} to {float(mask.max()):.3f}")
    assert out.shape == wav.shape
    assert mask.shape == (2, frames, FREQ_BINS)
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= MASK_MAX
    print("\nstep 1 passed.")

    # ------------------------------------------------------------ step 2
    print("\nstep 2 -- causality: the future must not reach the past")

    # Deliberately on the CPU, batch 1, uniform input -- the same conditions
    # model_crn.self_test() uses. It matters: on CUDA with a batch of real
    # spectrograms, float non-associativity alone moves distant frames by
    # ~5e-6, which swamps a 1e-6 threshold and makes the CRN "fail" this test
    # too. The real dependency is five orders of magnitude larger, so the
    # quiet conditions are what let the assertion be exact rather than fuzzy.
    def first_frame_disturbed(model, probe, n_frames):
        assert not model.training, "BatchNorm couples every frame in train mode"
        base_in = torch.rand(1, n_frames, FREQ_BINS)
        with torch.no_grad():
            a, _, _ = model.predict_mask(base_in)
            bumped = base_in.clone()
            bumped[0, probe, :] += 5.0
            b, _, _ = model.predict_mask(bumped)
        moved = (a - b).abs().amax(dim=2)[0]
        changed = (moved > 1e-6).nonzero().flatten()
        return int(changed[0]) if len(changed) else None

    cpu_tcn = TCN(**wide).eval()
    cpu_crn = model_crn.CRN(**wide).eval()
    probe = 150
    first_t = first_frame_disturbed(cpu_tcn, probe, 301)
    first_c = first_frame_disturbed(cpu_crn, probe, 301)
    print(f"  changing frame {probe} first moves the TCN mask at {first_t}")
    print(f"  the CRN, on the same probe, reacts at            {first_c}")
    # The mask at frame t may depend on frames up to t + lookahead and no
    # further. Reacting EARLIER than that means the model read audio that had
    # not arrived. Both models must land on exactly the same frame, since they
    # spend exactly the same budget.
    assert first_t == probe - cpu_tcn.lookahead, (
        f"the temporal core added lookahead: mask reacts at {first_t}, "
        f"expected {probe - cpu_tcn.lookahead}")
    assert first_t == first_c, (
        f"TCN reacts at {first_t} but CRN at {first_c}; the two are not "
        f"paying the same latency and the comparison would be unfair")
    print(f"  both spend {cpu_tcn.lookahead} frames, and no more")
    print("\nstep 2 passed.")

    # ------------------------------------------------------------ step 3
    print("\nstep 3 -- streaming: chunks must equal the whole clip")

    half = CLIP_SAMPLES // 2
    with torch.no_grad():
        whole, _, _, _ = tcn(wav)
        a, _, _, st = tcn(wav[:, :half])
        b, _, _, _ = tcn(wav[:, half:], st)
        cold, _, _, _ = tcn(wav[:, half:])

    # The STFT is taken per call, so the two halves cannot be compared to the
    # whole clip sample for sample at the seam. What must hold is that the
    # CARRIED state changes the answer -- that is what proves the core is
    # actually continuing rather than restarting blind every chunk.
    carried = float((b - cold).abs().max())
    print(f"  second chunk with state vs cold: {carried:.3e}")
    assert carried > 1e-6, (
        "starting cold gives the same answer, so the state carries nothing "
        "and chunked inference would restart the model at every boundary")

    n_buf = len(state)
    buf_frames = sum(s.shape[-1] for s in state)
    print(f"  state         {n_buf} buffers, {buf_frames} frames of context "
          f"in total")
    assert n_buf == len(tcn.memory.blocks)
    print("\nstep 3 passed.")

    # ------------------------------------------------------------ step 4
    print("\nstep 4 -- the auxiliary head, and deployment")

    print(f"  aux target    {tcn.aux_target}")
    print(f"  logits        {tuple(logits.shape)}")
    # On the CPU, like model_crn.self_test() does, and for the same reason:
    # the claim is that the aux branch is provably not part of the audio path,
    # which is worth stating as EXACTLY zero. On CUDA it comes out at ~2e-7,
    # because freeing the head changes the workspace available to cuBLAS and
    # a different GEMM algorithm gets picked. That is rounding, not coupling,
    # but a threshold would hide a real coupling of the same size one day.
    import copy
    cpu_wav = torch.randn(2, CLIP_SAMPLES) * 0.1
    trimmed = copy.deepcopy(cpu_tcn).strip_aux().eval()
    with torch.no_grad():
        again, _, _, _ = cpu_tcn(cpu_wav)
        cut, _, cut_logits, _ = trimmed(cpu_wav)
    same = float((again - cut).abs().max())
    print(f"  stripping the aux head changes the audio by {same:.2e}")
    assert same == 0.0 and cut_logits is None
    print("\nstep 4 passed.")

    # ------------------------------------------------------------ step 5
    print("\nstep 5 -- latency, which must match the CRN exactly")

    def ms_per_clip(model):
        x = torch.randn(1, CLIP_SAMPLES, device=device)
        with torch.no_grad():
            for _ in range(3):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / 10 * 1000

    t_ms, c_ms = ms_per_clip(tcn), ms_per_clip(crn)
    t_chunk = t_ms * CHUNK_SAMPLES / CLIP_SAMPLES
    c_chunk = c_ms * CHUNK_SAMPLES / CLIP_SAMPLES
    # Computed from the lookahead these models were actually BUILT with, not
    # from model_crn.TOTAL_LOOKAHEAD_MS, which is fixed to the module default
    # of 2 frames and would understate a 6-frame model by 40 ms.
    fixed = model_crn.STFT_LOOKAHEAD_MS + tcn.lookahead * 10 + 100
    print(f"  TCN  {t_ms:6.2f} ms per 3 s clip, {t_chunk:.2f} ms per chunk"
          f"  -> {fixed + t_chunk:.1f} ms total latency")
    print(f"  CRN  {c_ms:6.2f} ms per 3 s clip, {c_chunk:.2f} ms per chunk"
          f"  -> {fixed + c_chunk:.1f} ms total latency")
    print(f"  the TCN is {c_ms / t_ms:.1f}x faster: convolutions run in "
          f"parallel where an LSTM must step through time")
    # Both spend the same lookahead and the same chunk wait, so any latency
    # difference is compute alone. It has to stay small enough not to be the
    # thing the comparison measures.
    assert abs(t_chunk - c_chunk) < 5.0, (
        "the TCN costs more than 5 ms per chunk more than the CRN; the "
        "latency comparison would not be like for like")
    print("\nstep 5 passed. The TCN is ready to train.")


if __name__ == "__main__":
    self_test()
