# CRN build — step tracker

The model is being built in small stages. Each one is written, run, and
checked before the next begins, so a failure is always traceable to the piece
that just changed.

## Where we are

```
[x] step 1   Audio <-> spectrogram          DONE
[ ] step 2   The encoder                    <-- WE ARE HERE
[ ] step 3   Lookahead control
[ ] step 4   The LSTM memory
[ ] step 5   Decoder and mask
[ ] step 6   Noise classifier head, on GPU
```

Everything lives in `scripts/model_crn.py`. It grows one stage at a time, and
every stage adds a check to its `self_test()`.

**To see everything built so far still works:**

```
python scripts/model_crn.py
```

That re-runs every check from every completed step, so it doubles as a
regression test. If it passes, the foundation is sound.

---

## step 1 — Audio to spectrogram and back    ✅ DONE

**What it does, plainly.** Turns sound into a picture, and the picture back
into sound.

Your 3-second clip is 48,000 numbers describing a wiggly line. A network
cannot easily see "that is a helicopter" in a wiggly line. So we convert it
into a **spectrogram**: imagine a piano with 257 keys, photographed every 10
milliseconds, recording which keys are pressed and how hard. Time runs left to
right, pitch runs bottom to top, brightness is loudness.

This matters because the two things we care about look completely different in
that picture. A voice holds a few pitches for a moment, drawing **horizontal
lines**. A gunshot or mortar hits every pitch at once for an instant, drawing
a **vertical line**. Separating horizontal from vertical in a picture is
something a network is good at. Separating two tangled wiggly lines is not.

**The numbers.**

| | |
| --- | --- |
| 3-second clip | becomes 301 snapshots |
| each snapshot | splits sound into 257 pitch bands |
| 100 ms streaming chunk | exactly 10 snapshots, no ragged edges |
| round-trip error | 9.5e-07 — lossless for practical purposes |

**What went wrong, and what it cost.** The first attempt used `center=False`,
meaning the first snapshot starts exactly at sample 0. It crashed. The reason
is that the Hann window — the soft fade applied to each snapshot so its edges
do not click — is exactly zero at its first sample. Sample 0 therefore gets
multiplied by zero and cannot be recovered, and PyTorch refused rather than
returning quietly wrong audio.

The fix is `center=True`, which pads a little silence at both ends so every
real sample is covered by several overlapping snapshots. Reconstruction is
then essentially perfect.

**But it is not free, and step 3 must account for it.** Each snapshot now sits
centred on its timestamp, so it reads 256 samples — **16 ms** — past the
moment it claims to represent. That is lookahead the network receives before a
single layer exists. With 16 ms from the framing plus 2 frames of layer
lookahead, the total lands at 36 ms, close to the 37 ms DCCRN operated on.

**Seeing it work.**

```
python scripts/visualize_pipeline.py
```

Draws a real mixture from the frozen evaluation set through every stage: clean
speech, the noise added, the mixture, the spectrogram, the reconstruction, and
the error between input and output. The error panel is a flat line at zero,
which is the point.

```
python scripts/visualize_pipeline.py --zoom 0.40 1.05 --fmax 4
```

Zooms in and shows clean, noise and mixture spectrograms stacked, sharing one
brightness scale. This is where the horizontal-versus-vertical contrast
becomes obvious.

Other options: `--split`, `--index`, and `--category mortar` to auto-pick the
loudest example of a noise category. Figures land in `logs/figures/` and are
gitignored, being reproducible in seconds.

---

## step 2 — The encoder    ⬅ CURRENT

**What it will do, plainly.** Slide small windows over the spectrogram
hunting for shapes, the same machinery that finds edges in photographs. Five
layers, each squeezing the pitch axis in half while learning more patterns.

The point of squeezing is to force the network to summarise. It cannot keep
all 257 pitch bands, so it must decide what matters — which is how it learns
that stacked rungs mean a voice and a full-height column means an impact.

**Frequency shrinks:** 257 → 129 → 65 → 33 → 17 → 9
**Patterns learned per layer:** 16 → 32 → 64 → 128 → 128

Time length is never squeezed. Only pitch is. Compressing time would blur
exactly the sudden events we most need to catch.

**How it gets checked.** Shapes at every layer are what they should be, time
length is preserved end to end, and the output is the expected size going into
the memory stage.

---

## step 3 — Lookahead control

**What it will do.** Decide how far into the future the model may peek.

Peeking helps: seeing a few milliseconds ahead makes it far easier to judge
whether a sudden loud thing is a plosive like /p/ or the start of a gunshot.
But peeking is the most expensive thing in a real-time system, because waiting
for future audio is dead time — you cannot optimise it away with faster
hardware.

The budget is roughly 40 ms total, of which **16 ms is already spent** by the
step 1 framing.

**How it gets checked.** Mechanically, not by inspection: change the input at
one moment in time, then confirm that every output earlier than the allowed
window is bit-for-bit unchanged. This is the only reliable way to catch
accidental future leakage, which otherwise stays invisible until deployment.

---

## step 4 — The LSTM memory

**What it will do.** Give the network a memory, so it understands "this rotor
has been droning for two seconds" instead of judging each 10 ms snapshot in
isolation.

This is also what makes streaming work. The memory is a small piece of state
carried from one 100 ms chunk to the next, so the model never starts blind.

**How it gets checked.** Feed a clip whole, then feed it in two halves passing
the memory across the join, and confirm the results match. If they do not, the
model cannot stream.

---

## step 5 — Decoder and mask

**What it will do.** Expand back to full resolution and produce a **mask** —
a value between 0 and 1 for every pitch band at every moment, meaning "keep
this much of it". Multiply the mask by the noisy spectrogram, convert back to
audio, and that is the enhanced output.

The decoder mirrors the encoder and receives **skip connections** from it, so
the fine detail thrown away during squeezing can be restored.

Note this masks loudness only and reuses the noisy timing information
unchanged. That is this architecture's ceiling, and precisely what DCCRN's
complex-valued layers exist to lift later.

**How it gets checked.** A full pass, waveform in to waveform out, with the
output the same length as the input and the mask staying within range.

---

## step 6 — Noise classifier head, on GPU

**What it will do.** Add the second small branch that guesses which of the
seven noise categories is present. It never receives anything the deployed
system would not have — it only reads the shared trunk — and it gets removed
before deployment.

**How it gets checked.** Seven output numbers per clip, the whole model runs
on the RTX 5060, and we record the real parameter count and memory use.

---

## After the six steps

Not part of this sequence, but next in line:

1. **Training script** — the loop that actually teaches it.
2. **Evaluation script** — PESQ, STOI and SI-SNR against the frozen 268
   mixtures. Needs `pesq` and `pystoi`, which are **not installed yet**.
3. **A short real run** to measure seconds-per-epoch on the 5060 before
   committing to anything long.

Then the same sequence for the TCN, and a comparison between them. DCCRN
remains the stretch upgrade, attempted only if a working system already
exists.
