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

    THE MODEL IS COMPLETE.
    Training, evaluation and the dashboard are built and running.
    26 runs done.

    THE SNR DISTRIBUTION CHANGED after run 0025: mixtures are now
    90% U(0,15) / 10% U(-5,0) instead of a flat U(-5,+15), and the
    frozen sets were rebuilt. That is worth ~+0.19 PESQ on its own,
    so scores either side of it are NOT comparable. Runs from the
    old distribution carry a _snrU suffix.

    Best so far, on held-out audio scored against the new set:
    run 0026 at PESQ 1.934, STOI 0.8630, SI-SNR 13.62.
    STOI clears its 0.85 target; PESQ and SI-SNR do not.
    0026 is the first run trained AND scored on the new
    distribution, and it beat 0025 by only +0.004 PESQ --
    training natively bought far less than the exam change did.

    Settled: the objective was the ceiling, not the architecture.
    Phase, mask ceiling, LSTM resolution, mask post-processing,
    mask supervision and both compression directions are all
    measured dead ends. Only two things ever worked -- fixing the
    training objective, and giving the model more future context.

    Next: re-measure the noise floor on the new exam
    (two seeds), then the SIH deck.                       <-- WE ARE HERE
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
This is the same limitation as reusing the noisy timing information.

**Later correction.** This paragraph used to end by calling that the ceiling
DCCRN's complex-valued layers exist to lift. The effect is real, but it is not
what caps this project's scores: an ideal mask keeping the same noisy phase
reaches PESQ 3.31 against our 1.78. See "Results so far" below.

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

## What came next — all four done ✅

1. **Training script** ✅ `train.py`. The audio loss is SI-SNR plus a
   compressed-magnitude spectral term, and cross-entropy for the classifier,
   combined with `SPECTRAL_LOSS_WEIGHT` and `AUX_LOSS_WEIGHT`. Reads
   `SpeechNoiseMixtures('train')` from `mixing.py`.
2. **Evaluation script** ✅ `evaluate.py`, with `pesq` and `pystoi` installed.
   Reports every metric three ways — noisy, enhanced, and the gain between
   them — overall and per noise category.
3. **Timing** ✅ 11.3 s per epoch; an 80-epoch run is about 25 minutes.
4. **Overfitting** ✅ watched, and it never became the problem. Quartering the
   data changed nothing, so the thin data-to-parameter ratio was not what held
   the scores down. The deferred voice-distortion augmentation is
   correspondingly a weaker bet than it looked.

## Results so far -- two things worked, everything else was a dead end

26 runs. Full detail in `PROJECT_CONTEXT_4.md` Sections 18-21.

**Check which exam a score sat before comparing it.** The SNR distribution
changed after run 0025, from a flat U(-5,+15) to 90% U(0,15) / 10% U(-5,0), and
the frozen sets were rebuilt. That alone moves PESQ by about +0.19 with the
model untouched -- run 0025 scores 1.898 on the old set and 2.098 on the new
one. Old-distribution runs carry a `_snrU` suffix; `rescore_runs.py` has scored
all of them on the current set so the two eras can be lined up.

**Read scores on STOI, not PESQ.** Training is not reproducible -- two runs with
the same settings and the same seed differ by 0.029 PESQ, because cuDNN's LSTM
backward is nondeterministic. Three identical runs measured the cost on the old
exam: PESQ sd 0.0124, STOI sd 0.0004, SI-SNR sd 0.03. PESQ is 13x noisier. That
floor has not yet been re-measured on the new exam.

On the current set, validation:

```
run    tested                    PESQ    STOI    SI-SNR
noisy  input                    1.230  0.7814    6.19
0016   9.3M, lookahead 2        2.058  0.8823   13.01
0023   lookahead 10             2.102  0.8831   13.21
0024   lookahead 4              2.081  0.8799   13.05
0025   lookahead 6  <- best     2.098  0.8824   13.19
```

Run 0025 on held-out testing: **PESQ 1.925, STOI 0.8609, SI-SNR 13.52**, against
targets of 2.5, 0.85 and 15. STOI is met.

### Only two changes ever mattered

**Fixing the training objective** (+0.117 PESQ, 13 sigma) and **giving the model
more future context** (+0.040 PESQ, 7 sigma at lookahead 6). Both change what
the model is *told*, not how big it is. Capacity is real but small; data volume
matters but we have no more of it.

### What the cheating test showed

`scripts/oracle_ceiling.py` builds the best possible mask out of the clean
speech the model never sees, and scores it. That is a hard ceiling. It needs no
training and took twenty minutes to settle a question argued about for weeks:

```
noisy input                          PESQ 1.189
ideal mask, capped 1.0, NOISY phase  PESQ 3.314   <- the real ceiling
ideal mask uncapped, noisy phase     PESQ 3.397
noisy magnitude + clean phase        PESQ 1.289
clean magnitude + clean phase        PESQ 4.644   <- sanity check, passes
```

**Reusing the noisy phase is not the ceiling, and DCCRN is not the next move.**
The ideal mask keeps that same noisy phase and still clears the 2.5 target.
Raising `MASK_MAX` is worth +0.08 even on a *perfect* mask, which is why runs
0006 and 0007 measured nothing.

The gap was between what we trained and what we measured: run 0009 captured
**73% of the achievable SI-SNR but only 23% of the achievable PESQ**, because
SI-SNR was the loss. SI-SNR grades waveform overlap, largely dictated by the
phase we copy and never change; PESQ and STOI grade the spectral envelope,
which is what the mask actually controls.

### Run 0010: the fix

A compressed-magnitude term was added to the loss -- spectrograms compared bin
by bin after raising both to the power 0.3, so quiet consonants count instead
of being drowned by loud vowels. Against run 0009 as its control:

```
PESQ     1.6819 +-0.0109  ->  1.7843 +-0.0062     +0.102
STOI     0.8398 +-0.0022  ->  0.8449 +-0.0004     +0.005
SI-SNR   11.572 +-0.096   ->  11.588 +-0.025       unchanged
```

9.4x the control's standard deviation, ranges not overlapping, and no SI-SNR
surrendered to get it. First real effect in ten runs.

That held up: on STOI it is a 13-sigma effect, by far the largest thing
measured in this project.

### What came after, and what is left

**The spectral weight knob is spent.** Off 1.682, w50 1.780, w200 1.784, w600
1.799. Almost all of the gain is from having the term at all.

**Mask resolution: a good hypothesis, and wrong.** `mask_resolution.py` blurs
the ideal mask and scores it -- a two-bin blur costs 0.90 of the 2.13 PESQ
available, and harmonics sit 3-8 bins apart while the LSTM saw 9 bands of ~29
bins each. So run 0013 gave the LSTM 33 bands, and it was WORSE (STOI 8.4 sigma
down), fitting even the training data worse despite 2.3x the parameters. The
skip connections already carry full 257-band detail to the decoder, so the
model could always draw a sharp mask.

**The mask is sharp and in the wrong places.** It correlates with the ideal
mask at only 0.617 and sits 0.198 away on a 0-to-1 dial, while being just 1.77x
smoother. An ideal mask blurred to the model's own roughness still scores 2.62
against the model's 1.68. The error is a regression toward the middle -- where
the ideal wants 0 the model leaves 0.134, where it wants 1 it keeps 0.707 --
and multiplying, powering or contrast-expanding the mask all make things
monotonically worse. It is a model correctly averaging over what it does not
know.

**Deploy `best_pesq.pt`, not `best.pt`.** The latter is chosen by validation
loss, which on run 0014 landed on epoch 26 at PESQ 1.703 while the best epoch
was 65 at 1.832.

**Next: train the mask against the ideal mask directly.** The model is only
ever told its audio was wrong, never shown the mask it should have drawn --
though the mask is the only thing it makes. That is the one kind of change that
has ever worked here.

## Tuning ideas, from measurement rather than guesswork

An epoch takes **11.3 seconds** (9.2 s GPU, 0.7 s data) and an evaluation
about 12 more. 40 epochs is under 9 minutes; 150 is about half an hour.

- **Set `EPOCHS` to a fixed budget of 150 and leave early stopping off.** This
  advice originally said to rely on early stopping at 12 epochs of patience.
  That was withdrawn: it watched validation *loss* while results are judged on
  PESQ, and the two diverge, so runs stopped at different points and training
  length became a hidden variable in comparisons meant to isolate one thing.
  Models converge by roughly epoch 110 at 9.3M parameters and the curve is
  flat after that, so 150 leaves a margin and nothing hides beyond it.
- **Run many experiments.** A dozen configurations is an afternoon. This is
  why the comparison dashboard earns its place.
- **Raise `BATCH_SIZE` to 32 or 48.** Only 1.62 GB of 8 GB is in use and the
  GPU is the bottleneck, so a bigger batch should cut epoch time.
- **Do not add DataLoader workers.** Data loading is 6% of an epoch.
  Worth recording because the first assumption was the opposite — mixing
  audio on the fly sounds expensive and is not.

Then the same for the TCN, and a comparison. DCCRN remains a stretch upgrade
attempted only once a working system exists -- and note that after the oracle
measurement above, nothing in this project's own results argues for it.
