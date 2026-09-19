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

  The encoder and decoder are convolutions over time. They hold no state,
  but each output frame needs a window of neighbouring input frames --
  KERNEL_TIME 3 over five layers, so ten frames of past each. They are
  recomputed over a short context window, which is cheap because the window
  is measured in tens of frames rather than the 256 the offline path uses.

  The post-filter is a dilated convolution with a receptive field of `past`
  frames. Same treatment as the encoder: keep that many frames of history.

LOOKAHEAD IS A DELAY, NOT A PROBLEM

Stage 1 is allowed to see 6 frames of the future. A frame therefore cannot
be finished until 6 more have arrived, so this class holds the newest 6
frames back and emits them next call. That is not overhead added here -- it
is the same 60 ms of lookahead the model always had, made visible.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import model_crn
from model_crn import N_FFT, HOP_LENGTH, SAMPLE_RATE


# ----------------------------------------------------------- TUNING KNOBS
#
# CONV_CONTEXT  frames of history kept so the encoder and decoder convolutions
#               see the same neighbours they would offline. The true need is
#               about 20 (KERNEL_TIME 3, five layers, both directions of the
#               network). Raising it costs compute linearly and buys nothing
#               once it exceeds the receptive field; lowering it below 20
#               silently corrupts the frames nearest each boundary, which is
#               audible as a click and invisible in any error that is averaged
#               over a whole clip. Verified by self_test().
# CHUNK_FRAMES  frames consumed per push. 10 frames is 100 ms at the 160-sample
#               hop. Smaller chunks lower latency and raise per-call overhead.
CONV_CONTEXT = 24
CHUNK_FRAMES = 10

# ----------------------------------------------------------------- LOCKED
#
# Set by the STFT and by the trained checkpoint. Changing these does not make
# a faster stream, it makes a different model.
CHUNK_SAMPLES = CHUNK_FRAMES * HOP_LENGTH
PAD = N_FFT // 2                 # what center=True adds at the front



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
            x = self._step(i, mod, x) if isinstance(mod, self.TimeConv)                 else mod(x)
        return self.decompress(e + torch.complex(x[:, 0], x[:, 1]))


class Streamer:
    """Feed it audio in pieces; it hands back finished audio in pieces.

    Usage:

        s = Streamer(stage1, postfilter)
        out = s.push(samples)      # may be shorter than the input, or empty

    The output lags the input by the model's lookahead plus one analysis
    window. push() returns only samples that will never change again, so
    what comes out can be played immediately.
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
        self.n_framed = 0          # frames cut from self.samples so far
        self.spec = None           # (1, T, F) recent input frames
        self.spec_base = 0         # stream frame index of self.spec[0]
        self.mem = None            # (1, C, T, F') recent memory outputs
        self.state = None          # the LSTM's memory
        self.done = 0              # frames whose mask has been emitted
        self.ola = np.zeros(0, dtype=np.float32)   # overlap-add numerator
        self.den = np.zeros(0, dtype=np.float32)   # and its window envelope
        self.ola_base = 0          # sample index of self.ola[0], padded frame
        self.emitted = 0           # padded-domain samples already returned

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
        self.n_framed += avail
        return spec.unsqueeze(0)                                # (1, T, F)

    # ------------------------------------------------------------ the loop
    @torch.no_grad()
    def push(self, chunk):
        """New input samples in; finished output samples out."""
        chunk = np.asarray(chunk, dtype=np.float32)
        self.samples = np.concatenate([self.samples, chunk])

        new = self._cut_frames()
        if new is not None:
            if self.spec is None:
                self.spec = new
            else:
                self.spec = torch.cat([self.spec, new], dim=1)

        if self.spec is None:
            return np.zeros(0, dtype=np.float32)

        total = self.spec_base + self.spec.shape[1]
        trusted = total - self.lookahead          # frames with their future in
        n_new = trusted - self.done
        if n_new <= 0:
            return np.zeros(0, dtype=np.float32)

        # --- encoder over the whole retained window -----------------------
        feat = self.spec.abs()
        erb = getattr(self.m, "erb", None)
        if erb is not None:
            feat = erb.bm(feat)                    # per-frame, no state
        x, skips = self.m.encoder(feat)

        # --- LSTM over the new frames only, state carried -----------------
        a = self.done - self.spec_base
        m_new, _, self.state = self.m.memory(
            x[:, :, a:a + n_new, :], self.state)

        # --- decoder needs its own past, so glue on the memory history ----
        m_full = m_new if self.mem is None else torch.cat([self.mem, m_new],
                                                          dim=2)
        span = m_full.shape[2]
        end = a + n_new
        sk = [s[:, :, end - span:end, :] for s in skips]
        mask = self.m.decoder(m_full, sk)[:, -n_new:, :]
        if erb is not None:
            mask = erb.bs(mask).clamp(0.0, self.m.decoder.mask_max)

        # --- stage 2 ------------------------------------------------------
        sp = self.spec[:, a:a + n_new, :]
        enh = sp * mask
        if self.pf is not None:
            enh = self.pf(enh, sp)

        # --- overlap-add back to a waveform -------------------------------
        out = self._overlap_add(enh, self.done)

        # --- retire what is no longer needed ------------------------------
        self.done = trusted
        self.mem = m_full[:, :, -CONV_CONTEXT:, :]
        keep = CONV_CONTEXT + self.lookahead + CHUNK_FRAMES
        if self.spec.shape[1] > keep:
            drop = self.spec.shape[1] - keep
            self.spec = self.spec[:, drop:, :].contiguous()
            self.spec_base += drop
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


def self_test():
    """Stream a real clip and demand it match the offline path."""
    import train
    import postfilter as pf
    import demo_core as core
    from runlog import RunLog
    from mixing import FrozenMixtures

    dev = torch.device("cpu")
    torch.set_num_threads(4)
    name = core.stage1_for_postfilter()
    m = train.load_run_model(RunLog.open(name), dev)[0].eval()
    m.strip_aux()
    post = (pf.load(core.POSTFILTER_PATH, dev)[0].eval()
            if os.path.exists(core.POSTFILTER_PATH) else None)
    print(f"  stage 1 {name}, lookahead {m.lookahead} frames")
    print(f"  post-filter past {getattr(post, 'past', 0)} frames")

    # A silent lead-in, because the offline path reflect-pads the very start
    # of a clip and a live stream has nothing to reflect. Without this the two
    # disagree on frames 0-1, the LSTM carries that difference forward for
    # ever, and the whole comparison measures the padding rather than the
    # streaming. Real audio from a microphone begins in silence anyway.
    raw = FrozenMixtures("testing").describe(0)["noisy"].astype(np.float32)
    wav = np.concatenate([np.zeros(N_FFT, dtype=np.float32), raw])
    x = torch.from_numpy(wav).unsqueeze(0)
    with torch.no_grad():
        _, mask, _, _ = m(x)
        spec = model_crn.to_spectrogram(x)
        o = spec * mask
        if post is not None:
            o = post(o, spec)
        offline = model_crn.to_waveform(o, x.shape[-1])[0].numpy()

    s = Streamer(m, post, dev)
    live = np.concatenate([s.push(wav[i:i + CHUNK_SAMPLES])
                           for i in range(0, len(wav), CHUNK_SAMPLES)])
    n = min(len(live), len(offline))
    skip = N_FFT
    err = np.abs(live[skip:n] - offline[skip:n]).max()
    print(f"  streamed {len(live)} of {len(wav)} samples "
          f"({(len(wav) - len(live)) / 16.0:.0f} ms held back)")
    print(f"  max difference vs offline: {err:.2e}  "
          f"(rms {np.sqrt((offline ** 2).mean()):.2e})")
    assert err < 1e-3, "streaming does not match the offline path"

    # A stream that ignored its history would still look fine on an average;
    # the damage lands at the chunk seams. Check them specifically.
    seams = [i * CHUNK_SAMPLES for i in range(1, len(live) // CHUNK_SAMPLES)]
    worst = max(np.abs(live[max(skip, t - 8):t + 8]
                       - offline[max(skip, t - 8):t + 8]).max()
                for t in seams if t + 8 < n)
    print(f"  worst error at a chunk seam:  {worst:.2e}")
    assert worst < 1e-3, "chunk boundaries do not line up"

    print("\nall checks passed")


if __name__ == "__main__":
    self_test()
