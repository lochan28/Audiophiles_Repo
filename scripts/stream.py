"""
stream.py

Run the two-stage model on a live audio stream, one chunk at a time.

WHY THIS EXISTS

demo_core.StreamSession re-processes 2.56 s of context for every 100 ms
chunk. That is sample-exact and needs no state, which is the right trade on
a laptop -- and hopeless on a Raspberry Pi, where it asks for about 26x real
time. This module does the same job by REMEMBERING instead of recomputing.

WHAT HAS TO BE REMEMBERED, AND WHY EACH PIECE IS DIFFERENT

Three parts of the model have a past, and they need three different
treatments. Getting any of them wrong produces audio that still sounds like
speech and merely clicks softly at every chunk boundary -- which is why this
file is checked against the offline path rather than trusted.

  The LSTM is stateful. Its hidden state IS the memory, so it is carried
  forward and every frame is fed to it exactly once, in order. Nothing is
  recomputed and the result is exact.

  The encoder and decoder are convolutions over time, KERNEL_TIME 3. A
  convolution has no state of its own, but every output frame reads two
  earlier input frames. So each layer keeps a copy of the last two frames it
  was given and glues them in front of the next chunk: the new frames then
  see exactly the neighbours they would have seen offline, and only the new
  frames are computed. This is the INCREMENTAL form. The first version of
  this file instead re-ran the whole encoder and decoder over a 24-frame
  window every push and kept the last few outputs; that was correct, and it
  was also why a 20 ms chunk cost almost as much as a 100 ms one -- thirty
  frames of work to finish two.

  The post-filter is a dilated causal convolution stack. Same treatment,
  layer by layer, with each layer keeping 2 x its own dilation of history.

LOOKAHEAD IS A DELAY, NOT A PROBLEM

Stage 1 is allowed to see 6 frames of the future, handed out across the
encoder layers as right-padding. A layer given r frames of right context
cannot finish an output frame until r more input frames exist, so its output
simply runs r frames behind its input. Those lags add up through the encoder
to the model's total lookahead, and the frames that fall out of the last
layer are, by construction, exactly the ones whose future has arrived. No
separate "trusted frames" bookkeeping is needed: the encoder emits a frame
when, and only when, it is final.

Because the encoder outputs lag by different amounts at different depths,
the skip connections the decoder needs run slightly AHEAD of the frames the
decoder is processing. Each skip is therefore queued and released in step.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import model_crn
from model_crn import N_FFT, HOP_LENGTH, SAMPLE_RATE, KERNEL_TIME


# ----------------------------------------------------------- TUNING KNOBS
#
# CHUNK_FRAMES  frames consumed per push. 10 frames is 100 ms at the 160-sample
#               hop; 2 frames is 20 ms. The output audio is identical at any
#               chunk size (verified by self_test), so this trades only two
#               things against each other: SMALLER chunks lower the end-to-end
#               delay -- the live path in deploy/live_mic.py costs about
#               (chunk + lookahead rounded up to whole chunks + chunk) -- and
#               raise the fixed per-call overhead, which is the Python and
#               launch cost of running every layer once per push. Whether a
#               given size holds its deadline on a given board is a
#               measurement (deploy/bench_configs.py), not a setting.
CHUNK_FRAMES = 10

# ----------------------------------------------------------------- LOCKED
#
# Set by the STFT and by the trained checkpoint. Changing these does not make
# a faster stream, it makes a different model.
CHUNK_SAMPLES = CHUNK_FRAMES * HOP_LENGTH
PAD = N_FFT // 2                 # what center=True adds at the front
HISTORY = KERNEL_TIME - 1        # input frames a time-conv reads before t


class _ConvStream:
    """One time-convolution layer, run on new frames only.

    Wraps the layer's conv + norm + activation and owns a buffer of the last
    HISTORY input frames. Each call glues the buffer in front of the new
    frames, runs the bare convolution (no time padding -- the buffer IS the
    padding), and keeps the newest HISTORY frames for next time.

    `left_init` is how many zero frames the buffer starts with. Offline, a
    layer with right context r is left-padded with (HISTORY - r) zeros, so
    the stream starts the same way; a strictly causal layer starts with the
    full HISTORY of zeros. With fewer zeros than HISTORY the first call emits
    r fewer frames than it was given, which is precisely the layer's lag.

    Works for Conv2d and ConvTranspose2d alike: both are configured in
    model_crn so that a T-frame input with no time padding yields T - HISTORY
    frames.
    """

    def __init__(self, conv, norm, act, left_init):
        self.conv, self.norm, self.act = conv, norm, act
        self.left_init = int(left_init)
        self.buf = None

    def __call__(self, x):                       # (1, C, n, F)
        if self.buf is None:
            shape = list(x.shape)
            shape[2] = self.left_init
            self.buf = torch.zeros(shape, dtype=x.dtype, device=x.device)
        xi = torch.cat([self.buf, x], dim=2)
        self.buf = xi[:, :, max(0, xi.shape[2] - HISTORY):, :]
        if xi.shape[2] <= HISTORY:
            return None                          # nothing finishable yet
        y = self.conv(xi)
        if self.norm is not None:
            y = self.norm(y)
        return self.act(y)


class _PostFilterStream:
    """The post-filter, run incrementally instead of from raw history.

    The straightforward way to stream a dilated convolution stack is to keep
    the last `past` input frames and re-run the whole stack over them. That
    is correct and it is what this class replaces, because it is also 13x
    more work than necessary: 130 frames are recomputed to produce 10, and
    the post-filter then costs 90% of the streaming budget.

    The observation that fixes it: each layer separately needs only ITS OWN
    small history -- 2*dilation frames of the tensor that feeds it, not of
    the raw input. Six layers each keeping a few frames costs a fraction of
    one layer re-reading 130. BatchNorm and ELU act on one frame at a time
    and need no memory at all.
    """

    def __init__(self, post, device):
        import postfilter as _pf
        self.post = post
        self.mods = list(post.body) + [post.out]
        self.buf = [None] * len(self.mods)
        self.TimeConv = _pf.TimeConv
        self.compress, self.decompress = _pf.compress, _pf.decompress
        for m in self.mods:
            if isinstance(m, _pf.TimeConv) and m.pad.padding[3] != 0:
                raise ValueError(
                    "this streamer handles strictly causal post-filters only; "
                    f"a layer asks for {m.pad.padding[3]} future frames")

    def _step(self, i, mod, x):
        left = mod.pad.padding[2]
        if self.buf[i] is None:
            sh = list(x.shape)
            sh[2] = left
            self.buf[i] = torch.zeros(sh, dtype=x.dtype, device=x.device)
        xi = torch.cat([self.buf[i], x], dim=2)
        self.buf[i] = xi[:, :, xi.shape[2] - left:, :] if left else xi[:, :, :0, :]
        return mod.conv(torch.nn.functional.pad(xi, (1, 1, 0, 0)))

    def __call__(self, enhanced_spec, noisy_spec):
        e = self.compress(enhanced_spec)
        chans = [e.real, e.imag]
        if self.post.use_noisy:
            n = self.compress(noisy_spec)
            chans += [n.real, n.imag]
        x = torch.stack(chans, dim=1)
        for i, mod in enumerate(self.mods):
            x = self._step(i, mod, x) if isinstance(mod, self.TimeConv) else mod(x)
        return self.decompress(e + torch.complex(x[:, 0], x[:, 1]))


class Streamer:
    """Feed it audio in pieces; it hands back finished audio in pieces.

    Usage:

        s = Streamer(stage1, postfilter)
        out = s.push(samples)      # may be shorter than the input, or empty

    The output lags the input by the model's lookahead plus one analysis
    window. push() returns only samples that will never change again, so
    what comes out can be played immediately. Chunks may be any size,
    including sizes that are not whole frames; the streamer keeps the
    remainder.
    """

    def __init__(self, stage1, post=None, device=None):
        self.m = stage1.eval()
        self.post = post.eval() if post is not None else None
        self.device = device or torch.device("cpu")
        self.lookahead = int(getattr(stage1, "lookahead", 0))
        self.pf = _PostFilterStream(self.post, self.device) if post else None
        self.win = model_crn._window(self.device)
        self.reset()

    # ------------------------------------------------------------ state
    def reset(self):
        self.samples = np.zeros(PAD, dtype=np.float32)  # front pad, once
        self.specq = None          # (1, T, F) input frames not yet masked
        self.state = None          # the memory's carried state
        self.done = 0              # frames whose mask has been emitted
        self.ola = np.zeros(0, dtype=np.float32)   # overlap-add numerator
        self.den = np.zeros(0, dtype=np.float32)   # and its window envelope
        self.ola_base = 0          # sample index of self.ola[0], padded frame
        self.emitted = 0           # padded-domain samples already returned

        enc, dec = self.m.encoder, self.m.decoder
        self.enc = [_ConvStream(b.conv, b.norm, b.act, HISTORY - b.right_pad)
                    for b in enc.blocks]
        self.dec = [_ConvStream(b.conv, b.norm, b.act, HISTORY)
                    for b in dec.blocks]
        self.skipq = [None] * len(enc.blocks)   # per layer: frames the
                                                # decoder has not used yet
        if self.pf is not None:
            self.pf = _PostFilterStream(self.post, self.device)

    # ------------------------------------------------------------ helpers
    def _cut_frames(self):
        """Turn buffered samples into whole STFT frames."""
        avail = (len(self.samples) - N_FFT) // HOP_LENGTH + 1
        if avail <= 0:
            return None
        idx = np.arange(N_FFT)[None, :] + \
            (np.arange(avail) * HOP_LENGTH)[:, None]
        blocks = torch.from_numpy(self.samples[idx]).to(self.device)
        spec = torch.fft.rfft(blocks * self.win, n=N_FFT)       # (avail, F)
        keep = avail * HOP_LENGTH
        self.samples = self.samples[keep:].copy()
        return spec.unsqueeze(0)                                # (1, T, F)

    @staticmethod
    def _cat(a, b):
        return b if a is None else torch.cat([a, b], dim=2)

    # ------------------------------------------------------------ the loop
    @torch.no_grad()
    def push(self, chunk):
        """New input samples in; finished output samples out."""
        chunk = np.asarray(chunk, dtype=np.float32)
        self.samples = np.concatenate([self.samples, chunk])

        new = self._cut_frames()
        if new is None:
            return np.zeros(0, dtype=np.float32)
        self.specq = new if self.specq is None else \
            torch.cat([self.specq, new], dim=1)

        # --- encoder, layer by layer, new frames only ----------------------
        feat = new.abs()
        erb = getattr(self.m, "erb", None)
        if erb is not None:
            feat = erb.bm(feat)                    # per-frame, no state
        x = feat.unsqueeze(1)                      # (1, 1, n, F)
        for i, layer in enumerate(self.enc):
            x = layer(x)
            if x is None:
                return np.zeros(0, dtype=np.float32)
            self.skipq[i] = self._cat(self.skipq[i], x)
        n_new = x.shape[2]

        # --- memory over exactly those frames, state carried --------------
        m_new, _, self.state = self.m.memory(x, self.state)

        # --- decoder: each block takes its skip from the front of the queue -
        y = m_new
        for j, layer in enumerate(self.dec):
            i = len(self.enc) - 1 - j              # matching encoder layer
            q = self.skipq[i]
            assert q is not None and q.shape[2] >= n_new, \
                "skip queue ran dry -- encoder and decoder out of step"
            skip, self.skipq[i] = q[:, :, :n_new, :], q[:, :, n_new:, :]
            y = layer(torch.cat([y, skip], dim=1))
        mask = y.squeeze(1) * self.m.decoder.mask_max      # (1, n_new, F')
        if erb is not None:
            mask = erb.bs(mask).clamp(0.0, self.m.decoder.mask_max)

        # --- stage 2 ------------------------------------------------------
        sp, self.specq = self.specq[:, :n_new, :], self.specq[:, n_new:, :]
        enh = sp * mask
        if self.pf is not None:
            enh = self.pf(enh, sp)

        # --- overlap-add back to a waveform -------------------------------
        out = self._overlap_add(enh, self.done)
        self.done += n_new
        return out

    def _overlap_add(self, spec, first_frame):
        """Windowed inverse FFT of new frames, summed into the output."""
        blocks = torch.fft.irfft(spec[0], n=N_FFT) * self.win      # (T, N)
        blocks = blocks.cpu().numpy()
        w2 = (self.win ** 2).cpu().numpy()

        start = first_frame * HOP_LENGTH
        need = start + (len(blocks) - 1) * HOP_LENGTH + N_FFT - self.ola_base
        if need > len(self.ola):
            self.ola = np.concatenate(
                [self.ola, np.zeros(need - len(self.ola), dtype=np.float32)])
            self.den = np.concatenate(
                [self.den, np.zeros(need - len(self.den), dtype=np.float32)])
        for i, b in enumerate(blocks):
            s = start + i * HOP_LENGTH - self.ola_base
            self.ola[s:s + N_FFT] += b.astype(np.float32)
            self.den[s:s + N_FFT] += w2

        # A sample is final once the last frame that touches it has landed:
        # frame t covers [t*HOP, t*HOP + N_FFT), so after frames up to
        # `last`, everything below (last + 1) * HOP is complete.
        final = (first_frame + len(blocks)) * HOP_LENGTH
        lo, hi = self.emitted, final
        if hi <= lo:
            return np.zeros(0, dtype=np.float32)
        num = self.ola[lo - self.ola_base:hi - self.ola_base]
        den = self.den[lo - self.ola_base:hi - self.ola_base]
        piece = np.where(den > 1e-8, num / np.maximum(den, 1e-8), 0.0)
        self.emitted = hi

        # Drop what can never be touched again.
        cut = max(0, self.emitted - N_FFT - self.ola_base)
        if cut > 0:
            self.ola = self.ola[cut:].copy()
            self.den = self.den[cut:].copy()
            self.ola_base += cut

        # The first PAD samples are the centre padding, not real audio.
        piece = piece.astype(np.float32)
        if lo < PAD:
            piece = piece[max(0, PAD - lo):]
        return piece


def _offline(m, post, wav):
    """The reference: the whole clip through the ordinary forward pass."""
    x = torch.from_numpy(wav).unsqueeze(0)
    with torch.no_grad():
        _, mask, _, _ = m(x)
        spec = model_crn.to_spectrogram(x)
        o = spec * mask
        if post is not None:
            o = post(o, spec)
        return model_crn.to_waveform(o, x.shape[-1])[0].numpy()


def _streamed(m, post, wav, chunk):
    s = Streamer(m, post, torch.device("cpu"))
    return np.concatenate([s.push(wav[i:i + chunk])
                           for i in range(0, len(wav), chunk)])


def _check(live, offline, label, tol=1e-3):
    """Max error overall and at the chunk seams, against the offline path."""
    n = min(len(live), len(offline))
    skip = N_FFT
    err = np.abs(live[skip:n] - offline[skip:n]).max()
    print(f"  {label:<28} {len(live):6d} samples out, "
          f"max diff {err:.2e}")
    assert err < tol, f"{label}: streaming does not match the offline path"
    return err


def self_test():
    """Stream real clips and demand they match the offline path -- at
    several chunk sizes, with and without the post-filter, and for every
    architecture in the deploy bundle that can be loaded."""
    import train
    import postfilter as pf
    import demo_core as core
    from runlog import RunLog
    from mixing import FrozenMixtures

    torch.set_num_threads(4)
    dev = torch.device("cpu")
    name = core.stage1_for_postfilter()
    m = train.load_run_model(RunLog.open(name), dev)[0].eval()
    m.strip_aux()
    post = (pf.load(core.POSTFILTER_PATH, dev)[0].eval()
            if os.path.exists(core.POSTFILTER_PATH) else None)
    print(f"  stage 1 {name}, lookahead {m.lookahead} frames, "
          f"right pads {m.encoder.right_pads}")
    print(f"  post-filter past {getattr(post, 'past', 0)} frames")

    # A silent lead-in, because the offline path reflect-pads the very start
    # of a clip and a live stream has nothing to reflect. Without this the two
    # disagree on frames 0-1, the LSTM carries that difference forward for
    # ever, and the whole comparison measures the padding rather than the
    # streaming. Real audio from a microphone begins in silence anyway.
    raw = FrozenMixtures("testing").describe(0)["noisy"].astype(np.float32)
    wav = np.concatenate([np.zeros(N_FFT, dtype=np.float32), raw])
    rms = np.sqrt((wav ** 2).mean())
    print(f"  clip rms {rms:.2e}\n")

    for label, p in (("two-stage", post), ("stage 1 only", None)):
        offline = _offline(m, p, wav)
        # 100 ms, 20 ms, 10 ms (one frame), a size that is not a whole
        # number of frames, and a size smaller than one hop.
        for chunk in (CHUNK_SAMPLES, 320, 160, 700, 100):
            live = _streamed(m, p, wav, chunk)
            _check(live, offline, f"{label}, {chunk / 16:g} ms chunks")

            # A stream that ignored its history would still look fine on an
            # average; the damage lands at the chunk seams. Check them.
            n = min(len(live), len(offline))
            seams = [i * chunk for i in range(1, len(live) // chunk)]
            worst = max(np.abs(live[max(N_FFT, t - 8):t + 8]
                               - offline[max(N_FFT, t - 8):t + 8]).max()
                        for t in seams if t + 8 < n and t > N_FFT)
            assert worst < 1e-3, f"chunk boundaries do not line up at {chunk}"
        held = len(wav) - len(_streamed(m, p, wav, CHUNK_SAMPLES))
        print(f"  {label}: {held} samples held back = {held / 16:.0f} ms\n")

    print("all checks passed")


if __name__ == "__main__":
    self_test()
