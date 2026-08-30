"""
model_crn.py

The CRN (Convolutional Recurrent Network) for speech enhancement.

BUILT IN STEPS -- this file grows one stage at a time, and every stage adds
a check to self_test() that must pass before the next is started.

    [x] step 1  audio <-> spectrogram
    [ ] step 2  encoder
    [ ] step 3  lookahead control
    [ ] step 4  LSTM memory
    [ ] step 5  decoder and mask
    [ ] step 6  noise classifier head

Run this file directly to execute every check built so far.
"""

import torch


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


if __name__ == "__main__":
    self_test()
