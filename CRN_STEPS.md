# CRN build — step tracker

The model was built in small stages. Each one was written, run, and checked
before the next began, so a failure was always traceable to the piece that had
just changed.

## Where we are

```
[x] step 1   Audio <-> spectrogram          DONE
[x] step 2   The encoder                    DONE
[x] step 3   Lookahead control              DONE
[x] step 4   The LSTM memory                DONE
[x] step 5   Decoder and mask               DONE
[x] step 6   Noise classifier head, on GPU  DONE

    THE MODEL IS COMPLETE.  Next: the training script.  <-- WE ARE HERE
```

Everything is in `scripts/model_crn.py`, 3,018,072 parameters, running on the
RTX 5060.

**To check the whole model still works:**

```
python scripts/model_crn.py
```

That runs every check from all six steps in one go, so it doubles as a full
regression test. If it passes, nothing has been broken.

**To see it working on real audio:**

```
python scripts/visualize_pipeline.py
python scripts/visualize_pipeline.py --zoom 0.40 1.05 --fmax 4
```

Figures land in `logs/figures/` and are gitignored, being reproducible in
seconds. Other options: `--split`, `--index`, `--category mortar`.

## The finished shape

```
noisy waveform
   |  STFT                    301 snapshots, 257 pitch bands
   v
ENCODER     5 layers, pitch 257 -> 129 -> 65 -> 33 -> 17 -> 9      245,472
   |        time never squeezed; 128 patterns x 9 bands per frame
   v
MEMORY      2 LSTM layers x 256, one direction                   2,266,240
   |        carries state between 100 ms chunks
   +------> AUX HEAD  7 noise categories, removed for deployment     16,903
   |
   v
DECODER     5 layers back up 9 -> 257, fed by skip connections      489,457
   |        ends in a sigmoid: a "keep this much" dial per band
   v
mask x noisy spectrogram, noisy timing reused, back to audio
                                                          TOTAL  3,018,072
                                                       DEPLOYED  3,001,169
```

---

## step 1 — Audio to spectrogram and back    ✅

**What it does.** Turns sound into a picture and back. Imagine a piano with
257 keys photographed every 10 ms, recording which keys are pressed and how
hard. Time runs left to right, pitch bottom to top, brightness is loudness.

This matters because a voice holds a few pitches and draws **horizontal
lines**, while a gunshot hits every pitch at once and draws a **vertical
line**. Separating horizontal from vertical in a picture is something a
network is good at. Separating two tangled wiggly lines is not.

| | |
| --- | --- |
| 3-second clip | 301 snapshots |
| each snapshot | 257 pitch bands |
| 100 ms chunk | exactly 10 snapshots, no ragged edges |
| round-trip error | 9.5e-07 — lossless in practice |

**What went wrong.** The first attempt used `center=False` and crashed. A Hann
window is exactly zero at its first sample, so sample 0 gets multiplied by
zero and cannot be recovered; PyTorch refused rather than returning quietly
wrong audio. The fix, `center=True`, pads silence at both ends so every real
sample is covered several times.

**What that cost.** Each snapshot now sits centred on its timestamp and reads
256 samples — **16 ms** — past the moment it claims to represent. That is
lookahead the network gets before a single layer exists, and step 3 had to
budget for it.

---

## step 2 — The encoder    ✅

**What it does.** Slides small windows over the spectrogram hunting for
shapes, the same machinery that finds edges in photographs. Five layers, each
looking for more kinds of pattern while halving the pitch axis.

```
pitch      257 -> 129 -> 65 -> 33 -> 17 -> 9
patterns     1 ->  16 -> 32 -> 64 -> 128 -> 128
time                    301 frames throughout
```

**Why squeezing is the point.** The network cannot carry all 257 bands
through, so it has to decide what is worth keeping. That constraint is what
forces it to learn the general rule — rungs mean voice, columns mean impact —
rather than memorising particular clips.

**Time is never squeezed.** Compressing it would blur exactly the sudden
events this system exists to catch, and would break the fixed
10-frames-per-chunk relationship streaming depends on.

Ends at 128 patterns x 9 bands = **1152 numbers per frame**. 245,472
parameters.

**One note in the code.** BatchNorm mixes statistics across the whole clip
while *training*, including future frames. At inference it uses stored running
statistics, so the deployed model has no leakage — which is why step 3's
causality check runs in eval mode.

---

## step 3 — Lookahead control    ✅

**What it does.** Lets the model peek at the future, but only a measured
amount, and proves it cannot cheat.

Peeking helps judge whether a sudden loud thing is a /p/ or the start of a
gunshot. But it is **dead time**: waiting for audio that has not arrived, and
unlike compute you cannot buy it back with faster hardware.

```
framing   16 ms   (inherited from step 1, unavoidable)
layers    20 ms   (2 frames x 10 ms, our choice)
          ------
total     36 ms
```

Essentially what DCCRN ran on (37 ms) when it won its real-time category. The
original plan was 200–300 ms.

The two frames go to the **deepest** layers, `[0, 0, 0, 1, 1]`, where abstract
features make a glimpse of the future worth more.

**How it is checked.** Not by reading padding numbers off the layers — one
mis-set padding anywhere would leak silently and stay invisible until
deployment. Instead the test changes one input frame and watches which outputs
react:

```
changed input frame 60 -> outputs 58..68 reacted
   2 frames ahead (the budget, used fully), 8 frames behind (free)
```

Two controls keep it honest: `lookahead=0` reacts no earlier than frame 60
itself, and `lookahead=6` reacts at 54, so the check genuinely tracks the
setting rather than always passing.

**A mistake worth keeping.** The framing measurement first printed 10 ms,
contradicting the 16 ms in the budget. Frames sit 160 samples apart but each
window reaches 256 samples past its centre, so counting whole frames rounds
256 down to 1 and understates the wait. Latency is set by the last sample a
frame needs, so the test now measures that directly: frame 150 sits at sample
24000 and still reacts to sample 24255. **This is why the budget is counted in
milliseconds, not frames.**

---

## step 4 — The LSTM memory    ✅

**What it does.** Gives the model a sense of what came before. The encoder
judges each 10 ms snapshot alone; the memory knows a rotor has been droning
for two seconds. Think of a notebook carried from moment to moment, with
dials deciding what to rub out, what to write in, and what to say out loud.

Two layers of 256, **one direction only**. A bidirectional LSTM reads the clip
backwards, so it needs the end before it can produce the start — unusable in
real time at any budget, and a quiet way to build something undeployable. The
code asserts against it.

**The result streaming depends on:**

```
whole clip vs two halves with the state passed across:  0.00e+00
same second half started cold instead:                  1.10e-01
```

Bit-identical, not merely close. Processing 100 ms at a time gives exactly the
same answer as processing the whole clip. The second line proves the first
means something: without the carried memory the answer visibly changes, so the
state really carries information.

**Where the model's size lives.** 2,266,240 parameters, about 90% of the model
at that point. This is also where overfitting will start on 1292 clips, so
`LSTM_HIDDEN` is the first knob to reach for.

**A mistake worth keeping.** The test first asserted that changing frame 60
would ripple to the clip's last frame, since an LSTM's memory is unbounded in
theory. It stopped at frame 88 and failed. Influence *fades*: at random
initialisation it drops below measurable after about 28 frames — **280 ms** of
effective memory. Real, but not infinite. How much is retained after training
is something training decides, not architecture.

**Found for later.** Splitting the *encoder* at a chunk boundary disagrees for
8 frames, because its convolutions need the frames before the cut. Real
streaming must cache those 8 frames per layer. That is a deployment concern,
not a training one.

---

## step 5 — Decoder and mask    ✅

**What it does.** Walks back up, `9 -> 17 -> 33 -> 65 -> 129 -> 257`, with
each layer handed the matching encoder layer's output as a **skip
connection**. The encoder deliberately threw detail away; the skips hand it
back, so the mask can be placed precisely instead of painted in broad strokes.

**The output is not audio.** It is a **mask**: a "keep this much" dial between
0 and 1 for every pitch band at every moment, 257 x 301 dials per clip. The
model never rebuilds sound from scratch, it decides how much of what is
already there to let through. Multiplying is gentler to learn than inventing a
signal, and cannot hallucinate sound that was never present.

Untrained, the mask averages **0.484** — "keep half of everything, everywhere",
exactly what random weights should give.

Decoder 489,457 parameters; model 3,001,169. Causality still exactly 2 frames,
so the decoder bought itself none.

**A finding worth keeping.** In the spectrogram, a mask capped at 1.0 never
increases any band — largest increase exactly **0.00**. But after converting
back to audio and re-analysing, bands **can** exceed the original, by 5.21 in
one test. A masked spectrogram is usually not the spectrogram of any real
sound: neighbouring frames overlap, and once scaled differently they stop
agreeing about the samples they share, so `istft` averages the disagreement.
This is the same limitation as reusing the noisy timing information, and is
exactly the ceiling DCCRN's complex-valued layers exist to lift.

---

## step 6 — Noise classifier head, on GPU    ✅

**What it does.** Seven scores per clip, one per noise category, read off the
shared memory. It never receives anything the deployed system would not have.
Its job is to push the encoder and memory to tell noise types apart, which is
groundwork for suppressing them.

**16,903 parameters — 0.6% of the model.** Deliberately tiny, because it gets
thrown away.

**The deployment claim, verified rather than trusted:**

```
removing the aux head changes the audio by  0.00e+00
```

Bit-identical, so the trim is safe. Deployed model: 3,001,169 parameters.

**On the RTX 5060:**

```
one 3 s clip     13.9 ms      (216x faster than real time)
per 100 ms       0.46 ms of compute for 100 ms of audio
batch of 16      1.62 GB of 8.0 GB, forward and backward
```

**Compute is not the constraint.** The machine is idle 99.5% of the time.

```
36 ms lookahead + 100 ms chunk wait + 0.5 ms compute = 136 ms
```

Against a 500–800 ms budget, and inside ITU's 150 ms comfortable-conversation
range. The budget is dominated by *waiting*, not thinking.

Two caveats: this is the 5060, **not** the Jetson AGX Orin the problem
statement names, so it is a first data point rather than the answer. And an
untrained model runs at the same speed as a trained one — this measures the
architecture, which is what needed knowing before committing to long runs.

The 1.62 GB also means there is room for a much larger batch, or the wider
model, well inside 8 GB.

---

## What comes next

1. **Training script** — the loop that teaches it. SI-SNR on the waveform for
   the audio branch, cross-entropy for the classifier, combined with
   `AUX_LOSS_WEIGHT`. Reads `SpeechNoiseMixtures('train')` from `mixing.py`.
2. **Evaluation script** — PESQ, STOI and SI-SNR against the frozen 268
   mixtures, reported overall and per noise category. **Needs `pesq` and
   `pystoi`, which are not installed yet.**
3. **A short real run** to measure seconds-per-epoch before committing to
   anything long.
4. **Watch for overfitting.** 1292 training clips against 3 M parameters is a
   thin ratio. When validation loss stops tracking training loss, that is the
   trigger for the deferred voice-distortion augmentation
   (`PROJECT_CONTEXT_4.md` Section 7) and for trying `LSTM_HIDDEN = 128`.

## Results so far -- the ceiling is not resources

Five runs. Full detail in `PROJECT_CONTEXT_4.md` Section 18.

```
run    tested                 params  clips   PESQ   STOI  SI-SNR
0001   baseline                3.0M   1292   1.671  0.840   11.53
0002   3x capacity             9.3M   1292   1.694  0.844   11.72
0003   quarter of the data     3.0M    323   1.673  0.841   11.64
0004   half the data           3.0M    646   1.650  0.836   11.50
0005   all the data            3.0M   1292   1.678  0.842   11.59
```

Tripling the parameters changed nothing. Quartering the data changed nothing.
Everything lands at SI-SNR ~11.6, STOI ~0.84, PESQ ~1.67.

**So do not collect more data and do not build a bigger model.** Both were
tested and both are dead ends. The ceiling is in the design:

1. `MASK_MAX = 1.0` means the model can only turn sound DOWN. At -5 dB SNR the
   speech is buried underneath the noise and needs boosting. One line to test.
2. The noisy phase is reused unchanged; only magnitude is masked. That is the
   known ceiling of this kind of model, and what DCCRN exists to fix.
3. Possibly a loss/metric mismatch: SI-SNR judges waveform match, PESQ and
   STOI judge spectral envelope.

Test in that order: 15 minutes, an hour, then DCCRN.

## Tuning ideas, from measurement rather than guesswork

An epoch takes **11.3 seconds** (9.2 s GPU, 0.7 s data) and an evaluation
about 12 more. 40 epochs is under 9 minutes; 150 is about half an hour.

- **Set `EPOCHS` to 150, not 40.** Early stopping at 12 epochs of patience
  will halt it when it plateaus, so there is no need to guess the number.
- **Run many experiments.** A dozen configurations is an afternoon. This is
  why the comparison dashboard earns its place.
- **Raise `BATCH_SIZE` to 32 or 48.** Only 1.62 GB of 8 GB is in use and the
  GPU is the bottleneck, so a bigger batch should cut epoch time.
- **Do not add DataLoader workers.** Data loading is 6% of an epoch.
  Worth recording because the first assumption was the opposite — mixing
  audio on the fly sounds expensive and is not.

Then the same for the TCN, and a comparison. DCCRN remains the stretch
upgrade, attempted only once a working system exists.
