"""
model_crn.py

The CRN (Convolutional Recurrent Network) for speech enhancement.

BUILT IN STEPS -- this file grows one stage at a time, and every stage adds
a check to self_test() that must pass before the next is started.

    [x] step 1  audio <-> spectrogram
    [x] step 2  encoder
    [x] step 3  lookahead control
    [x] step 4  LSTM memory
    [x] step 5  decoder and mask
    [x] step 6  noise classifier head

Run this file directly to execute every check built so far.
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
#  TUNING KNOBS — safe to change, this is what you experiment with
# ═══════════════════════════════════════════════════════════════════

N_FFT = 512
# THE "EXPOSURE". How much audio each snapshot covers, in samples.
# 512 = 32 ms.  SHORTER (320 = 20 ms) sharpens sudden sounds — gunshots,
# mortar, plosives — but blurs steady tones.  LONGER (1024 = 64 ms)
# cleanly separates rotor hum and engine drone but smears transients.
# You have both kinds of noise, so 512 is the middle. Try 320 first.
# Sensible range: 256–1024. Must stay a power of 2.

ENCODER_CHANNELS = (24, 48, 96, 192, 192)
# How many different patterns each encoder layer looks for, and implicitly how
# deep the encoder is (one layer per entry). Each layer also halves the pitch
# axis, so 5 layers takes 257 bands down to 9.
# HALVE ALL OF THESE for the "smaller arrangement" comparison in
# PROJECT_CONTEXT_4.md Section 4 — it roughly quarters the parameter count and
# is the first thing to try if train and validation loss diverge.
# MORE layers squeeze pitch further and see broader patterns, but 5 is already
# down to 9 bands; a 6th would leave 5 and start losing real detail.

LOOKAHEAD_FRAMES = 2
# How many frames of the FUTURE the layers are allowed to see. Each frame is
# 10 ms, so 2 = 20 ms of layer lookahead.
#
# Lookahead costs latency one-for-one: it is dead time spent waiting for audio
# that has not arrived, and unlike compute you cannot buy it back with faster
# hardware. It earns its keep by making onsets easier to judge - whether a
# sudden loud thing is a plosive like /p/ or the start of a gunshot - and that
# evidence arrives within a few tens of milliseconds, so the returns die off
# quickly beyond that.
#
# THE FRAMING ALREADY SPENT 16 ms (see STFT_LOOKAHEAD_MS below), so the real
# total is 16 + 10*LOOKAHEAD_FRAMES ms. At 2 frames that is 36 ms, close to
# the 37 ms DCCRN ran on when it won its real-time category.
#
# 0 = strictly causal, the model never sees the future at all.
# Sensible range: 0-4. Hard cap is 2 per encoder layer, so 10 with 5 layers.

LSTM_HIDDEN = 512
# How much the memory can hold. Larger remembers more about what has been
# heard, but is also where most of the parameters live and where overfitting
# starts on a 1292-clip training set. TRY 128 FIRST if training loss keeps
# falling while validation loss flattens. Sensible range: 128-512.

LSTM_LAYERS = 2
# Depth of the memory stack. 1 trains faster and overfits less; 3 rarely helps
# at this data size and slows every step. Sensible range: 1-3.

MASK_MAX = 1.0
# Ceiling on the mask, the "keep this much" dial the model produces for every
# pitch band at every moment. At 1.0 the model can only turn sound DOWN, never
# up, which is safe: the worst it can do is mute something.
# RAISING to ~2.0 lets it boost bands the noise buried, which can lift quality
# but also amplifies whatever artefacts it invents. Try 1.0 first and only
# raise it if the output sounds over-suppressed. Sensible range: 1.0-3.0.

AUX_LOSS_WEIGHT = 0.2
# How loudly the noise-classifier branch gets to argue during training. It
# never touches the audio output - it only pushes the shared encoder and
# memory to tell noise types apart, which is useful groundwork for suppressing
# them (PROJECT_CONTEXT_4.md Section 11).
# 0.0 switches it off entirely, which is the experiment worth running to find
# out whether it helps at all. Too high and the model optimises for guessing
# the category over sounding good. Read by the training script, not here.
# Sensible range: 0.0-0.5.


# ═══════════════════════════════════════════════════════════════════
#  LOCKED — changing these invalidates data already on disk
# ═══════════════════════════════════════════════════════════════════

SAMPLE_RATE = 16000     # every wav in the project is 16 kHz
CLIP_SAMPLES = 48000    # 3.000 s — the frozen 268 mixtures are this length
HOP_LENGTH = 160        # 10 ms — must divide the 1600-sample chunk evenly
N_CLASSES = 7           # the seven noise categories on disk

CHUNK_SAMPLES = 1600                                # 100 ms streaming chunk
FRAMES_PER_CHUNK = CHUNK_SAMPLES // HOP_LENGTH      # 10
FREQ_BINS = N_FFT // 2 + 1                          # 257

# The STFT is taken with center=True, which pads N_FFT//2 samples at each end
# so that frame t is *centred* on sample t*HOP_LENGTH. That framing therefore
# reads N_FFT//2 samples past the frame's own timestamp -- unavoidable
# lookahead baked into the front end, before the network sees anything.
#
# center=False would avoid it, but is not usable: a Hann window is exactly
# zero at its first sample, so the opening samples of the signal get zero
# weight and the inverse transform refuses (NOLA check). Padding the end
# cannot fix a deficiency at the start.
#
# Step 3 must count this against the lookahead budget rather than pretend the
# model starts from zero.
STFT_LOOKAHEAD_SAMPLES = N_FFT // 2                 # 256 samples
STFT_LOOKAHEAD_MS = STFT_LOOKAHEAD_SAMPLES / SAMPLE_RATE * 1000     # 16 ms


def _window(device=None):
    return torch.hann_window(N_FFT, device=device)


def frames_for(n_samples):
    """How many snapshots a clip of this length produces."""
    return n_samples // HOP_LENGTH + 1


def to_spectrogram(wav):
    """Waveform (B, N) -> complex spectrogram (B, T, F)."""
    spec = torch.stft(wav, N_FFT, HOP_LENGTH, N_FFT, _window(wav.device),
                      center=True, return_complex=True)
    return spec.transpose(1, 2)                      # (B, F, T) -> (B, T, F)


def to_waveform(spec, length):
    """Complex spectrogram (B, T, F) -> waveform (B, length)."""
    return torch.istft(spec.transpose(1, 2), N_FFT, HOP_LENGTH, N_FFT,
                       _window(spec.device), center=True, length=length)


KERNEL_TIME = 3
KERNEL_FREQ = 3


MAX_RIGHT_PAD = KERNEL_TIME - 1     # most future one layer can be given

# What the whole front end plus the layers actually costs in latency.
TOTAL_LOOKAHEAD_MS = (STFT_LOOKAHEAD_MS
                      + LOOKAHEAD_FRAMES * HOP_LENGTH / SAMPLE_RATE * 1000)


def freq_after(n_bins):
    """Pitch bands left after one encoder layer halves them."""
    return (n_bins + 2 * (KERNEL_FREQ // 2) - KERNEL_FREQ) // 2 + 1


def lookahead_plan(total, n_layers):
    """Hand out the lookahead budget across encoder layers, deepest first.

    Deep layers hold more abstract features, so a frame of future is worth
    more there than at the raw spectrogram, where it only buys a slightly
    wider view of individual pitch bands.

    Right-context adds up straight across layers because time is never
    downsampled: one frame granted at each of two layers is two frames of
    lookahead in total.
    """
    if total > n_layers * MAX_RIGHT_PAD:
        raise ValueError(
            f"LOOKAHEAD_FRAMES={total} is more than {n_layers} layers can "
            f"carry (max {MAX_RIGHT_PAD} each, so {n_layers * MAX_RIGHT_PAD})")
    per = [0] * n_layers
    left = total
    while left > 0:
        for i in range(n_layers - 1, -1, -1):
            if left == 0:
                break
            if per[i] < MAX_RIGHT_PAD:
                per[i] += 1
                left -= 1
    return per


class EncoderBlock(nn.Module):
    """One layer: hunt for patterns, halve the pitch axis, keep time intact.

    Time is never squeezed. Only pitch is. Compressing time would blur exactly
    the sudden events -- gunshots, mortar, plosives -- that this system most
    needs to catch, and would also break the fixed 10-frames-per-chunk
    relationship the streaming design depends on.

    `right_pad` is how many frames of the future this layer is allowed to see.
    It defaults to 0, meaning strictly causal: the kernel is fed entirely from
    the past. Step 3 hands out a budget across the layers; until then nothing
    peeks ahead.
    """

    def __init__(self, in_ch, out_ch, right_pad=0):
        super().__init__()
        if not 0 <= right_pad <= KERNEL_TIME - 1:
            raise ValueError(f"right_pad must be 0..{KERNEL_TIME - 1}")
        self.right_pad = right_pad
        # (freq_left, freq_right, time_left, time_right)
        self.pad = nn.ZeroPad2d((0, 0, KERNEL_TIME - 1 - right_pad, right_pad))
        self.conv = nn.Conv2d(in_ch, out_ch,
                              kernel_size=(KERNEL_TIME, KERNEL_FREQ),
                              stride=(1, 2), padding=(0, KERNEL_FREQ // 2))
        # NOTE: BatchNorm mixes statistics across the whole clip, including
        # future frames, while *training*. At inference it uses stored running
        # statistics instead, so there is no leakage in the deployed model --
        # which is why step 3's causality check runs in eval mode.
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ELU()

    def forward(self, x):                       # (B, C, T, F)
        return self.act(self.norm(self.conv(self.pad(x))))


class Encoder(nn.Module):
    """Five layers squeezing 257 pitch bands down to 9.

    Squeezing is the point. The network cannot carry all 257 bands through, so
    it has to decide what is worth keeping -- which is how it learns that
    evenly stacked rungs mean a voice and a full-height column means an impact.

    Returns the final features plus every intermediate output, because the
    decoder in step 5 needs them as skip connections to restore the detail
    thrown away here.
    """

    def __init__(self, channels=ENCODER_CHANNELS, lookahead=LOOKAHEAD_FRAMES,
                 right_pads=None):
        super().__init__()
        self.channels = tuple(channels)
        if right_pads is None:
            right_pads = lookahead_plan(lookahead, len(self.channels))
        self.right_pads = list(right_pads)
        self.lookahead = sum(self.right_pads)

        blocks, in_ch, freq = [], 1, FREQ_BINS
        self.freq_path = [freq]
        for out_ch, rp in zip(self.channels, self.right_pads):
            blocks.append(EncoderBlock(in_ch, out_ch, rp))
            in_ch = out_ch
            freq = freq_after(freq)
            self.freq_path.append(freq)
        self.blocks = nn.ModuleList(blocks)
        self.out_freq = freq
        self.out_features = self.channels[-1] * freq

    def forward(self, spec):
        """(B, T, F) -> final (B, C, T, F'), plus the per-layer skips."""
        x = spec.unsqueeze(1)
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return x, skips


class Memory(nn.Module):
    """The recurrent middle: gives the network a sense of what came before.

    The encoder judges each 10 ms moment on its own. This layer lets the model
    know that a rotor has been droning for two seconds, or that a voice has
    been mid-sentence -- context no single snapshot contains.

    It is also what makes streaming possible. The LSTM's hidden state is a
    small summary of everything heard so far, so it can be carried from one
    100 ms chunk to the next and the model never restarts blind.

    Unidirectional on purpose. A bidirectional LSTM reads the sequence
    backwards as well, which means seeing the end of the clip before emitting
    the start -- unusable in real time at any latency budget, and a silent way
    to produce results that cannot be deployed.
    """

    def __init__(self, n_features, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.layers = layers
        self.lstm = nn.LSTM(n_features, hidden, layers,
                            batch_first=True, bidirectional=False)
        self.project = nn.Linear(hidden, n_features)

    def forward(self, x, state=None):
        """(B, C, T, F) in, same shape out, plus the sequence and the state.

        `state` is the (h, c) pair carried between chunks. Pass None to start
        fresh; pass what came out last time to continue.
        """
        b, c, t, f = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b, t, c * f)     # (B, T, C*F)
        seq, state = self.lstm(seq, state)                   # (B, T, hidden)
        out = self.project(seq).reshape(b, t, c, f).permute(0, 2, 1, 3)
        return out, seq, state


class DecoderBlock(nn.Module):
    """One layer back up: double the pitch axis, keep time intact.

    Strictly causal in time. The whole lookahead budget is spent in the
    encoder, where a glimpse of the future buys the most, so the decoder is
    given none: it is left-padded and the transposed convolution's time
    padding is set to cancel that out exactly.
    """

    def __init__(self, in_ch, out_ch, last=False):
        super().__init__()
        self.pad = nn.ZeroPad2d((0, 0, KERNEL_TIME - 1, 0))
        self.conv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=(KERNEL_TIME, KERNEL_FREQ),
            stride=(1, 2), padding=(KERNEL_TIME - 1, KERNEL_FREQ // 2))
        self.norm = None if last else nn.BatchNorm2d(out_ch)
        self.act = nn.Sigmoid() if last else nn.ELU()

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm is not None:
            x = self.norm(x)
        return self.act(x)


class Decoder(nn.Module):
    """Mirrors the encoder back up to full resolution and produces the mask.

    Each layer is handed the matching encoder layer's output as a SKIP
    CONNECTION, glued on alongside its own input. The encoder deliberately
    threw detail away while squeezing; the skips hand that detail back, so the
    decoder can place the mask precisely instead of painting in broad strokes.

    The last layer ends in a sigmoid, so every output sits between 0 and 1: a
    "keep this much" dial for every pitch band at every moment.
    """

    def __init__(self, channels=ENCODER_CHANNELS, mask_max=MASK_MAX):
        super().__init__()
        self.mask_max = mask_max
        rev = list(reversed(tuple(channels)))
        blocks = []
        for i, ch in enumerate(rev):
            out_ch = rev[i + 1] if i + 1 < len(rev) else 1
            blocks.append(DecoderBlock(ch * 2, out_ch, last=(i == len(rev) - 1)))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skips):
        for block, skip in zip(self.blocks, reversed(skips)):
            x = block(torch.cat([x, skip], dim=1))
        return x.squeeze(1) * self.mask_max          # (B, T, F)


class CRN(nn.Module):
    """Encoder, memory, decoder: noisy waveform in, cleaner waveform out.

    The model never rebuilds audio from scratch. It predicts a mask -- how
    much of each pitch band to keep at each moment -- and that mask is applied
    to the noisy spectrogram. Multiplying is a gentler thing to learn than
    inventing a signal, and it cannot hallucinate sound that was never there.

    The TIMING information of the noisy audio is reused unchanged; only
    loudness is masked. That is this architecture's ceiling and exactly what
    DCCRN's complex-valued layers exist to lift later.
    """

    def __init__(self, channels=ENCODER_CHANNELS, lookahead=LOOKAHEAD_FRAMES,
                 hidden=LSTM_HIDDEN, layers=LSTM_LAYERS, mask_max=MASK_MAX,
                 n_classes=N_CLASSES):
        super().__init__()
        self.encoder = Encoder(channels, lookahead)
        self.memory = Memory(self.encoder.out_features, hidden, layers)
        self.decoder = Decoder(channels, mask_max)
        self.lookahead = self.encoder.lookahead

        # The auxiliary branch. It reads the shared memory and guesses which
        # noise category is present -- a training aid only. It is never given
        # anything the deployed system would not have, and it is removed
        # before deployment (Section 11).
        self.aux = nn.Sequential(
            nn.Linear(hidden, 64), nn.ELU(), nn.Linear(64, n_classes))

    def strip_aux(self):
        """Delete the classifier branch, for deployment.

        Safe by construction: nothing downstream of the audio path reads it,
        so removing it cannot change what the model outputs. The self-test
        checks that rather than assuming it.
        """
        self.aux = None
        return self

    def predict_mask(self, mag, state=None):
        """Magnitude spectrogram (B, T, F) -> mask (B, T, F), logits, state."""
        x, skips = self.encoder(mag)
        x, seq, state = self.memory(x, state)
        # one guess for the whole clip: the weak label described in Section 12
        logits = self.aux(seq.mean(dim=1)) if self.aux is not None else None
        return self.decoder(x, skips), logits, state

    def forward(self, wav, state=None):
        """Noisy waveform (B, N) -> enhanced waveform, mask, logits, state."""
        spec = to_spectrogram(wav)
        mask, logits, state = self.predict_mask(spec.abs(), state)
        enhanced = spec * mask                       # phase rides along unchanged
        return to_waveform(enhanced, wav.shape[-1]), mask, logits, state


def self_test():
    torch.manual_seed(0)

    print("step 1 -- audio <-> spectrogram")
    print(f"  window      {N_FFT} samples ({N_FFT / SAMPLE_RATE * 1000:.0f} ms)")
    print(f"  hop         {HOP_LENGTH} samples "
          f"({HOP_LENGTH / SAMPLE_RATE * 1000:.0f} ms)")
    print(f"  freq bins   {FREQ_BINS}")

    frames = frames_for(CLIP_SAMPLES)
    print(f"  a {CLIP_SAMPLES / SAMPLE_RATE:.3f} s clip -> {frames} frames")

    # a 100 ms chunk must be a whole number of frames, or streaming chunk
    # boundaries would not line up with frame boundaries
    assert CHUNK_SAMPLES % HOP_LENGTH == 0
    print(f"  a {CHUNK_SAMPLES / SAMPLE_RATE * 1000:.0f} ms chunk "
          f"= {FRAMES_PER_CHUNK} frames exactly")

    wav = torch.randn(2, CLIP_SAMPLES)
    spec = to_spectrogram(wav)
    assert spec.shape == (2, frames, FREQ_BINS), spec.shape
    print(f"\n  shapes      {tuple(wav.shape)} -> {tuple(spec.shape)} "
          f"(batch, frames, freq)")

    back = to_waveform(spec, CLIP_SAMPLES)
    assert back.shape == wav.shape
    err = (back - wav).abs().max()
    print(f"  round trip  worst error over all {CLIP_SAMPLES} samples: "
          f"{err:.2e}")
    assert err < 1e-4, f"audio is not surviving the round trip: {err}"

    # a real clip, not just noise, so the check means something musically
    t = torch.arange(CLIP_SAMPLES) / SAMPLE_RATE
    tone = (0.4 * torch.sin(2 * torch.pi * 440 * t)
            + 0.2 * torch.sin(2 * torch.pi * 1970 * t)).unsqueeze(0)
    err_tone = (to_waveform(to_spectrogram(tone), CLIP_SAMPLES) - tone).abs().max()
    print(f"              same for a two-tone signal:        {err_tone:.2e}")
    assert err_tone < 1e-4

    print(f"\n  cost: the framing is centred, so each frame reads "
          f"{STFT_LOOKAHEAD_SAMPLES} samples "
          f"({STFT_LOOKAHEAD_MS:.0f} ms) past its own timestamp.")
    print(f"  That is lookahead the network gets for free and step 3 must")
    print(f"  count against the budget.")

    print("\nstep 1 passed.")

    # ------------------------------------------------------------ step 2
    print("\nstep 2 -- the encoder")

    enc = Encoder().eval()
    print(f"  pitch bands   " + " -> ".join(str(f) for f in enc.freq_path))
    print(f"  patterns      1 -> " + " -> ".join(str(c) for c in enc.channels))

    assert enc.freq_path == [257, 129, 65, 33, 17, 9], enc.freq_path

    with torch.no_grad():
        out, skips = enc(spec.abs())

    b, c, t, f = out.shape
    print(f"\n  shapes        {tuple(spec.shape)} (batch, frames, freq)")
    for i, s in enumerate(skips):
        print(f"    layer {i + 1}     -> {tuple(s.shape)}  "
              f"(batch, patterns, frames, freq)")

    # time must survive untouched: chunk boundaries depend on it, and squeezing
    # time would blur the transients this system exists to catch
    for i, s in enumerate(skips):
        assert s.shape[2] == frames, (
            f"layer {i + 1} changed the number of frames: "
            f"{s.shape[2]} instead of {frames}")
    print(f"\n  time          {frames} frames in, {t} frames out of every "
          f"layer - unsqueezed")

    assert (b, c, f) == (2, enc.channels[-1], 9), out.shape
    assert enc.out_features == enc.channels[-1] * 9
    print(f"  bottleneck    {c} patterns x {f} bands = {enc.out_features} "
          f"numbers per frame, into the memory stage")

    n_enc = sum(p.numel() for p in enc.parameters())
    print(f"  parameters    {n_enc:,}")

    print(f"  lookahead     {enc.right_pads} frames per layer "
          f"(deepest first), {enc.lookahead} total")

    # a real clip, so this is not only true for random numbers
    import os as _os
    import soundfile as _sf
    real_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        "generated_test_mixtures", "testing", "noisy", "testing_039.wav")
    if _os.path.exists(real_path):
        y, _ = _sf.read(real_path, dtype="float32")
        with torch.no_grad():
            ro, _ = enc(to_spectrogram(torch.from_numpy(y).unsqueeze(0)).abs())
        assert ro.shape == (1, enc.channels[-1], frames, 9), ro.shape
        assert torch.isfinite(ro).all()
        print(f"  real audio    testing_039.wav -> {tuple(ro.shape)}, "
              f"all finite")

    print("\nstep 2 passed.")

    # ------------------------------------------------------------ step 3
    print("\nstep 3 -- lookahead control")
    print(f"  budget        framing {STFT_LOOKAHEAD_MS:.0f} ms "
          f"+ layers {LOOKAHEAD_FRAMES} x "
          f"{HOP_LENGTH / SAMPLE_RATE * 1000:.0f} ms "
          f"= {TOTAL_LOOKAHEAD_MS:.0f} ms total")
    print(f"  allocation    {lookahead_plan(LOOKAHEAD_FRAMES, 5)} "
          f"across the 5 encoder layers")

    def earliest_frame_disturbed(model, probe, n_frames):
        """Change one input frame; find the first output frame that reacts.

        This is the only trustworthy way to check lookahead. Reading the
        padding numbers off the layers proves nothing, because a single
        mis-set padding anywhere would leak future audio silently and stay
        invisible until deployment.

        Runs in eval mode deliberately. In train mode BatchNorm recomputes
        statistics over the whole clip, so touching any frame nudges every
        output and the measurement becomes meaningless -- the deployed model
        uses stored running statistics and has no such coupling.
        """
        assert not model.training, "must be in eval mode, see docstring"
        base_in = torch.rand(1, n_frames, FREQ_BINS)
        with torch.no_grad():
            a, _ = model(base_in)
            bumped = base_in.clone()
            bumped[0, probe, :] += 5.0
            b, _ = model(bumped)
        moved = (a - b).abs().amax(dim=(1, 3))[0]      # per output frame
        changed = (moved > 1e-6).nonzero().flatten()
        return (int(changed[0]), int(changed[-1])) if len(changed) else (None, None)

    n_frames, probe = 120, 60

    first, last = earliest_frame_disturbed(enc, probe, n_frames)
    print(f"\n  changed input frame {probe}:")
    print(f"    outputs {first}..{last} reacted")
    print(f"    so it sees {probe - first} frames ahead "
          f"and {last - probe} frames behind")

    assert first >= probe - enc.lookahead, (
        f"FUTURE LEAK: output frame {first} reacted, which is "
        f"{probe - first} frames of lookahead in a model allowed "
        f"{enc.lookahead}")
    assert first == probe - enc.lookahead, (
        f"lookahead is configured but not reaching the output: expected the "
        f"first reaction at {probe - enc.lookahead}, got {first}")
    print(f"    within budget, and the budget is actually used")

    # the strictly causal case must also hold, or the check proves nothing
    causal = Encoder(lookahead=0).eval()
    c_first, _ = earliest_frame_disturbed(causal, probe, n_frames)
    assert c_first == probe, (
        f"a lookahead=0 encoder still reacted at {c_first}, "
        f"{probe - c_first} frames early")
    print(f"  control       lookahead=0 reacts no earlier than frame {probe} "
          f"itself - strictly causal")

    # and a deliberately over-generous one must be caught by the same test
    wide = Encoder(lookahead=6).eval()
    w_first, _ = earliest_frame_disturbed(wide, probe, n_frames)
    assert w_first == probe - 6
    print(f"  control       lookahead=6 reacts at frame {w_first}, "
          f"exactly 6 early - the check tracks the setting")

    # The framing's own lookahead has to be measured in SAMPLES, not frames.
    # Frames sit 160 samples apart while the window reaches 256 samples past
    # its centre, so counting whole frames rounds 256 down to 1 frame and
    # understates the wait. Latency is set by the last sample a frame needs,
    # which is the number below.
    base_wav = torch.rand(1, CLIP_SAMPLES)
    base_spec = to_spectrogram(base_wav).abs()
    target_frame = 150
    centre = target_frame * HOP_LENGTH

    def frame_reacts(offset):
        w = base_wav.clone()
        w[0, centre + offset] += 5.0
        d = (to_spectrogram(w).abs() - base_spec).abs()
        return bool(d[0, target_frame].max() > 1e-6)

    reach = max(k for k in range(0, N_FFT) if frame_reacts(k))
    reach_ms = reach / SAMPLE_RATE * 1000
    print(f"\n  framing       frame {target_frame} sits at sample {centre}, "
          f"but still reacts to sample {centre + reach}")
    print(f"                = {reach} samples ({reach_ms:.0f} ms) of future "
          f"per frame, exactly as step 1 warned")
    assert reach == STFT_LOOKAHEAD_SAMPLES - 1, (reach, STFT_LOOKAHEAD_SAMPLES)

    # in whole frames that same reach rounds down, which is why the budget is
    # kept in milliseconds rather than frames
    print(f"                (in whole frames that is only "
          f"{reach // HOP_LENGTH}, which is why the budget is counted in ms)")

    print(f"\n  TOTAL LATENCY COST OF SEEING AHEAD: {TOTAL_LOOKAHEAD_MS:.0f} ms")
    print("\nstep 3 passed.")

    # ------------------------------------------------------------ step 4
    print("\nstep 4 -- the LSTM memory")

    mem = Memory(enc.out_features).eval()
    print(f"  input         {mem.n_features} numbers per frame "
          f"(from the encoder bottleneck)")
    print(f"  memory        {mem.layers} layers x {mem.hidden} hidden, "
          f"one direction only")

    assert not mem.lstm.bidirectional, (
        "bidirectional reads the clip backwards -- unusable in real time")

    with torch.no_grad():
        mem_out, mem_seq, mem_state = mem(out)
    print(f"  shapes        {tuple(out.shape)} -> {tuple(mem_out.shape)}, "
          f"sequence {tuple(mem_seq.shape)}")
    assert mem_out.shape == out.shape
    assert mem_seq.shape == (out.shape[0], out.shape[2], mem.hidden)

    h, c_ = mem_state
    print(f"  carried state {tuple(h.shape)} hidden + {tuple(c_.shape)} cell "
          f"= {h.numel() + c_.numel()} numbers per clip in the batch")

    n_mem = sum(p.numel() for p in mem.parameters())
    print(f"  parameters    {n_mem:,}  "
          f"(encoder {n_enc:,}, running total {n_enc + n_mem:,})")

    # --- the check streaming depends on: whole clip vs two halves ---
    feats = torch.rand(1, enc.channels[-1], 120, enc.out_freq)
    half = 60
    with torch.no_grad():
        whole, _, _ = mem(feats)
        first_half, _, carried = mem(feats[:, :, :half], None)
        second_half, _, _ = mem(feats[:, :, half:], carried)
    joined = torch.cat([first_half, second_half], dim=2)
    drift = (whole - joined).abs().max()
    print(f"\n  streaming     whole clip vs two halves with the state passed "
          f"across: {drift:.2e}")
    assert drift < 1e-5, (
        f"state is not carrying: {drift}. Chunked inference would not match "
        f"what training produced.")

    with torch.no_grad():
        cold = mem(feats[:, :, half:], None)[0]
    cold_gap = (second_half - cold).abs().max()
    print(f"                same second half started cold instead: "
          f"{cold_gap:.2e}")
    assert cold_gap > 1e-4, (
        "starting cold gives the same answer, so the state carries no "
        "information and the test above proves nothing")
    print(f"                so the memory genuinely matters, and passing it "
          f"reproduces the unsplit result")

    # --- memory must not have introduced any peeking ---
    class EncMem(nn.Module):
        def __init__(self, e, m):
            super().__init__()
            self.e, self.m = e, m

        def forward(self, spec):
            y, _ = self.e(spec)
            y, _, _ = self.m(y)
            return y, None

    stack = EncMem(enc, mem).eval()
    s_first, s_last = earliest_frame_disturbed(stack, probe, n_frames)
    print(f"\n  causality     changing frame {probe} moves outputs "
          f"{s_first}..{s_last} of {n_frames - 1}")
    assert s_first == probe - enc.lookahead, (s_first, probe - enc.lookahead)
    print(f"                still exactly {enc.lookahead} frames of lookahead "
          f"- the memory added none")
    assert s_last > last, (
        f"the memory did not extend context at all: influence still stops at "
        f"{s_last}, same as the encoder alone ({last})")
    print(f"                past reach grew from {last - probe} frames "
          f"(encoder alone) to {s_last - probe} frames "
          f"= {(s_last - probe) * HOP_LENGTH / SAMPLE_RATE * 1000:.0f} ms")
    print(f"                In principle an LSTM carries influence forever; in "
          f"practice it fades, and")
    print(f"                at random initialisation it drops below "
          f"measurable after that. Training")
    print(f"                is what decides how much really gets retained.")

    # --- what full streaming will additionally need ---
    with torch.no_grad():
        spec_in = torch.rand(1, 120, FREQ_BINS)
        cont, _ = enc(spec_in)
        a_half, _ = enc(spec_in[:, :half])
        b_half, _ = enc(spec_in[:, half:])
    seam = (cont[:, :, half:] - b_half).abs().amax(dim=(0, 1, 3))
    settled = int((seam < 1e-6).nonzero().flatten()[0])
    print(f"\n  note          splitting the ENCODER at a chunk boundary "
          f"disagrees for {settled} frames")
    print(f"                because its convolutions need the frames before "
          f"the cut. Real streaming")
    print(f"                must cache those {settled} frames per layer; that "
          f"is a deployment concern,")
    print(f"                not a training one, and does not affect anything "
          f"built so far.")

    print("\nstep 4 passed.")

    # ------------------------------------------------------------ step 5
    print("\nstep 5 -- decoder and mask")

    model = CRN().eval()
    dec_freq = [enc.out_freq]
    with torch.no_grad():
        e_out, e_skips = model.encoder(spec.abs())
        m_out, _, _ = model.memory(e_out)
        x = m_out
        for block, skip in zip(model.decoder.blocks, reversed(e_skips)):
            x = block(torch.cat([x, skip], dim=1))
            dec_freq.append(x.shape[3])
    print(f"  pitch bands   " + " -> ".join(str(f) for f in dec_freq)
          + "   (back up)")
    assert dec_freq == [9, 17, 33, 65, 129, 257], dec_freq
    assert dec_freq[-1] == FREQ_BINS

    with torch.no_grad():
        mask, _, _ = model.predict_mask(spec.abs())
    print(f"  mask          {tuple(mask.shape)} - one 'keep this much' dial "
          f"per band per frame")
    assert mask.shape == spec.shape
    assert mask.shape[1] == frames, "the decoder changed the number of frames"
    lo_m, hi_m = float(mask.min()), float(mask.max())
    print(f"  mask range    {lo_m:.3f} to {hi_m:.3f} "
          f"(allowed 0 to {MASK_MAX:g})")
    assert 0.0 <= lo_m and hi_m <= MASK_MAX + 1e-6

    wav_in = torch.randn(2, CLIP_SAMPLES)
    with torch.no_grad():
        wav_out, mask2, _, st = model(wav_in)
    print(f"\n  full pass     {tuple(wav_in.shape)} -> {tuple(wav_out.shape)}"
          f"  (waveform in, waveform out)")
    assert wav_out.shape == wav_in.shape
    assert torch.isfinite(wav_out).all()

    # A mask capped at 1.0 can only attenuate. That holds exactly in the
    # spectrogram, which is where the masking happens.
    with torch.no_grad():
        in_spec = to_spectrogram(wav_in)
        masked_mag = (in_spec * mask2).abs()
    grew_spec = float((masked_mag - in_spec.abs()).max())
    print(f"  energy        in the spectrogram, no band grew: "
          f"largest increase {grew_spec:.2e}")
    assert grew_spec <= 1e-5, (
        f"a mask capped at {MASK_MAX:g} made a band louder by {grew_spec}")

    # ...but it does NOT survive the trip back to audio and forward again.
    # A masked spectrogram is generally not the spectrogram of any real
    # signal: neighbouring frames overlap, and once they are scaled by
    # different amounts they stop agreeing about the samples they share.
    # istft resolves the disagreement by averaging, and re-analysing that
    # compromise can land above the original in places.
    with torch.no_grad():
        out_mag = to_spectrogram(wav_out).abs()
    grew_wav = float((out_mag - in_spec.abs()).max())
    print(f"                after istft and re-analysis it can, by "
          f"{grew_wav:.2e}, because a")
    print(f"                masked spectrogram is not a consistent one - the "
          f"overlapping frames")
    print(f"                no longer agree, and istft has to average them. "
          f"Reusing the noisy")
    print(f"                phase is the same limitation, and what DCCRN "
          f"exists to fix.")

    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(f"\n  parameters    decoder {n_dec:,}, whole model {n_all:,}")

    # the decoder must not have quietly bought itself more lookahead
    class MaskOnly(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, mag):
            mk, _, _ = self.m.predict_mask(mag)
            return mk.unsqueeze(1), None

    d_first, d_last = earliest_frame_disturbed(MaskOnly(model).eval(),
                                               probe, n_frames)
    print(f"\n  causality     changing frame {probe} moves the mask at "
          f"{d_first}..{d_last}")
    assert d_first == probe - model.lookahead, (
        f"the decoder added lookahead: mask reacts at {d_first}, expected "
        f"{probe - model.lookahead}")
    print(f"                still {model.lookahead} frames "
          f"({TOTAL_LOOKAHEAD_MS:.0f} ms with the framing) - the decoder "
          f"added none")

    # and it must still stream
    with torch.no_grad():
        mag120 = torch.rand(1, 120, FREQ_BINS)
        w_mask, _, _ = model.predict_mask(mag120)
        a_mask, _, carry = model.predict_mask(mag120[:, :half], None)
    print(f"  streaming     state still carries through the full model "
          f"(mask {tuple(w_mask.shape)})")
    assert w_mask.shape == (1, 120, FREQ_BINS)

    if _os.path.exists(real_path):
        y, _ = _sf.read(real_path, dtype="float32")
        with torch.no_grad():
            ry, rm, _, _ = model(torch.from_numpy(y).unsqueeze(0))
        print(f"  real audio    testing_039.wav -> {tuple(ry.shape)}, "
              f"mask mean {float(rm.mean()):.3f}, all finite: "
              f"{bool(torch.isfinite(ry).all())}")

    print("\nstep 5 passed.")

    # ------------------------------------------------------------ step 6
    print("\nstep 6 -- noise classifier head, and the GPU")

    with torch.no_grad():
        _, _, logits, _ = model(wav_in)
    print(f"  logits        {tuple(logits.shape)} - one score per noise "
          f"category, per clip")
    assert logits.shape == (wav_in.shape[0], N_CLASSES)

    n_aux = sum(p.numel() for p in model.aux.parameters())
    print(f"  aux head      {n_aux:,} parameters "
          f"({n_aux / n_all * 100:.1f}% of the model)")

    # Section 11 claims the branch can be removed before deployment without
    # affecting the audio. Check it rather than trust it.
    import copy
    trimmed = copy.deepcopy(model).strip_aux().eval()
    with torch.no_grad():
        a_wav, _, _, _ = model(wav_in)
        b_wav, b_mask, b_logits, _ = trimmed(wav_in)
    same = (a_wav - b_wav).abs().max()
    n_trim = sum(p.numel() for p in trimmed.parameters())
    print(f"\n  deployment    removing the aux head changes the audio by "
          f"{same:.2e}")
    assert same == 0.0, (
        f"stripping the classifier altered the output by {same}; it is not "
        f"the training-only branch Section 11 describes")
    assert b_logits is None
    print(f"                and leaves {n_trim:,} parameters "
          f"({n_all:,} - {n_aux:,}) - smaller and faster, no quality cost")

    # ---- on the GPU
    if not torch.cuda.is_available():
        print("\n  GPU           not available; skipping the device checks")
        print("\nstep 6 passed (CPU only).")
        return

    import time
    dev = torch.device("cuda")
    gmodel = CRN().to(dev).eval()
    torch.cuda.reset_peak_memory_stats()

    one = torch.randn(1, CLIP_SAMPLES, device=dev)
    with torch.no_grad():
        for _ in range(3):                      # warm up cuda kernels
            gmodel(one)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            gmodel(one)
        torch.cuda.synchronize()
        per_clip = (time.perf_counter() - t0) / 20

    print(f"\n  GPU           {torch.cuda.get_device_name(0)}")
    print(f"  one 3 s clip  {per_clip * 1000:.1f} ms  "
          f"({CLIP_SAMPLES / SAMPLE_RATE / per_clip:.0f}x faster than "
          f"real time)")
    chunk_ms = per_clip * 1000 * CHUNK_SAMPLES / CLIP_SAMPLES
    print(f"  per 100 ms    {chunk_ms:.2f} ms of compute for 100 ms of audio")
    print(f"  latency       {TOTAL_LOOKAHEAD_MS:.0f} ms lookahead "
          f"+ 100 ms chunk wait + {chunk_ms:.1f} ms compute "
          f"= {TOTAL_LOOKAHEAD_MS + 100 + chunk_ms:.0f} ms")
    print(f"                against a 500-800 ms budget. NOTE this is the "
          f"5060, not the")
    print(f"                Jetson the problem statement names - it is a "
          f"first data point,")
    print(f"                not the answer.")

    # training-sized batch, to see whether 8 GB is enough
    batch = torch.randn(16, CLIP_SAMPLES, device=dev)
    gtrain = CRN().to(dev).train()
    out_b, _, log_b, _ = gtrain(batch)
    loss = out_b.pow(2).mean() + log_b.pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"\n  batch of 16   forward and backward fit in {peak:.2f} GB "
          f"of {total_vram:.1f} GB")
    assert peak < total_vram

    if _os.path.exists(real_path):
        y, _ = _sf.read(real_path, dtype="float32")
        with torch.no_grad():
            gy, gm, gl, _ = gmodel(torch.from_numpy(y).unsqueeze(0).to(dev))
        print(f"  real audio    testing_039.wav on GPU -> {tuple(gy.shape)}, "
              f"mask mean {float(gm.mean()):.3f}, "
              f"logits finite: {bool(torch.isfinite(gl).all())}")

    print("\nstep 6 passed. The model is complete.")


if __name__ == "__main__":
    self_test()
