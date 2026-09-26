# Project handoff — AI/ML Adaptive Noise Cancellation for defence communication

**Read this first if you are new to the project, human or LLM.** It is written
to be self-contained: everything built, everything measured, everything ruled
out, and what to do next. Nothing here requires having seen the conversation
that produced it.

Companion documents: `PROJECT_CONTEXT_4.md` (the long-form record; Sections
18–21 hold the experiment history, 21 covers the SNR change), `AGENTS_3.md` (the short operational brief
to read before suggesting anything), `CRN_STEPS.md` (how the model was built,
step by step, with the checks each step passed).

---

## 1. What the project is

Smart India Hackathon 2026, problem statement **26052** (DRDO / IDEX).

> AI/ML-enabled Adaptive Noise Cancellation for defence communication:
> suppress stationary, non-stationary and impulsive battlefield noise while
> keeping speech intelligible, in real time.

In plain terms: a soldier is speaking on a radio. Around them there is gunfire,
artillery, a helicopter, an engine. We want the listener to hear the words and
not the battlefield — fast enough that a conversation still feels natural.

Three numbers define success, all measured on audio the model never trained on.

| metric | what it measures | target |
| --- | --- | --- |
| PESQ | how *pleasant and undistorted* the speech sounds | 2.5 |
| STOI | how *understandable* the words are | 0.85 |
| SI-SNR | how much noise was actually removed, in dB | 15 |

Latency budget: 500–800 ms was the original allowance. We are far inside it.

---

## 2. Where the project stands right now

**The SNR distribution changed after run 0025.** Mixtures were drawn from a
flat U(-5,+15) dB; they are now drawn **90% from U(0,15) and 10% from
U(-5,0)**, and the frozen evaluation sets were rebuilt to match. Read §4.0
before comparing any two numbers in this project.

The system is **two stages**: a frozen CRN (stage 1) followed by a small
causal post-filter that predicts a residual to *add* to its output. **Since
2026-09-19 stage 1 is run 0036 (the ERB hybrid)**, adopted over run 0026 as a
deliberate speed-for-quality trade so the two-stage system fits the Raspberry
Pi 5 (§13 item 7). Scored on the current held-out testing set -- 125 mixtures
never used for training and never used to choose the model:

```
                 raw input   shipped (0036 + PF)   target     status
PESQ               1.223           1.990            2.5       short by 0.51
STOI               0.754           0.866            0.85      MET
SI-SNR (dB)        6.14           13.95            15.0       short by 1.05
latency (delay)                  ~300 ms           200-250    OVER, see below
```

For reference, the previous pairing (run 0026 + its post-filter) read 1.934 /
0.863 / 13.62 for stage 1 alone and **1.983 / 0.866 / 14.02** two-stage. The
swap is +0.0069 PESQ, -0.0005 STOI, -0.068 SI-SNR: STOI and SI-SNR slipped
past the noise floor, which the acceptance rule in §3 would normally forbid --
accepted anyway, eyes open, for the Pi margin. The DeepFilterNet comparison
and the per-experiment numbers further down were measured with the 0026
pairing and are left as recorded.

**STOI now clears its target on held-out audio.** Under the project's
acceptance rule (§3) it becomes a constraint to protect rather than a goal to
chase. PESQ and SI-SNR are the two still short.

**The two-stage system beats DeepFilterNet v0.5.6 on all three metrics at
once**, scored on our own frozen testing set with the same code: 1.947 /
0.8508 / 11.67 for DeepFilterNet against 1.983 / 0.8660 / 14.02 for us. Before
the post-filter we were only level with it on PESQ. `benchmark_external.py`
runs that comparison and guards the two things that would make it dishonest --
a common truncation window, and a cross-correlation lag check.

On the **validation split** (143 mixtures, also used for checkpoint selection,
so optimistic) it reads PESQ 2.1024 / STOI 0.8855 / SI-SNR 13.26. Quote the
testing numbers externally.

**Training natively on the new distribution bought very little**, and that is
worth knowing. Run 0025, trained on the old flat distribution and merely scored
on the new one, reads 2.0981 / 0.8824 / 13.19 on validation. So 0026 gains
+0.004 PESQ (inside the PESQ noise floor), +0.0031 STOI and +0.07 SI-SNR.
Nothing regressed, so it is adopted under the rule in §3, but the honest
reading is that **almost all of the apparent improvement since the change came
from the exam getting easier, not from the model.** It also suggests training
on harder mixtures generalises downward well, which is the direction you want
if deployment ever sees worse conditions than the test set.

### Latency -- two different numbers, and which one a listener feels

**Read this before quoting any latency figure.** The project conflated two
quantities for a long time, and the widely-quoted **177 ms was wrong**.

**Throughput** is "can the board keep up?" -- compute time per chunk against
the chunk period. Miss it and audio drops out. **Delay** (mouth to ear) is
"how long after I speak does the other end hear me?" -- the sum of every place
a sample WAITS. A system can have excellent throughput and poor delay; ours
does.

```
THROUGHPUT   Pi 5, 4 threads, int8, run 0036 + post-filter
             69.7 ms of compute per 100 ms chunk = 1.4x real time, deadline met

DELAY        the model holds 86 ms back before a sample is final
             (60 ms chosen lookahead + 16 ms STFT framing + 10 ms frame
             quantisation). MEASURED off the streamer, not derived: it emits
             1376 samples fewer than it is fed, always.

             live path, deploy/live_mic.py, 100 ms audio blocks:
              100 ms   wait for the block to fill
              100 ms   the 86 ms hold-back, rounded UP to a whole block
              100 ms   the output block's own slot
              -------
             ~300 ms   + sound-card I/O (NOT yet measured)
```

**Compute does not add to the delay.** `live_mic.py` runs inference on a
worker thread while the next block fills, so the 69.7 ms hides inside a 100 ms
block period. Replace it with 1 ms and the delay is still ~300 ms: the delay
is set by the BLOCK SIZE, not by the model's speed.

**Where 177 ms came from, and why it is wrong.** `train.py`'s
`measure_inference_ms` and `total_latency_ms` compute
`lookahead + chunk + compute`, on the training GPU. Two errors: it ADDS
compute that actually overlaps, and it assumes output can be released at any
instant when in fact the hold-back is rounded up to a whole block. It also
uses 76 ms where the streamer really holds 86. Every `runs/*/config.json`
carries this field -- treat it as a lower bound on an ideal system, not as a
measured delay. `live_mic.py` printing "added latency ~188 ms" is wrong the
same way.

**The lever is the block size, not the model.** Since the encoder and decoder
became incremental (§11) the chunk size is a free knob -- the output audio is
bit-identical at any of them. Delay is roughly
`chunk + roundup(86, chunk) + chunk`:

```
    block    delay        x86 median/worst per push   verdict
    100 ms   ~300 ms      25.1 / 31.1 ms              comfortable, over budget
     50 ms   ~200 ms      19.8 / 25.2 ms              fits budget, good margin
     20 ms   ~140 ms      15.1 / 25.7 ms              worst case misses on x86
```

**50 ms is the recommended target**, and the measurement that decides it has
NOT been taken -- `bench_configs.py` with `CHUNK_FRAMES = 5` on the board
(§13). The x86 numbers above are a laptop CPU; §8 records that x86-to-ARM
scaling was already off by 3x once in this project. Nothing here is a Pi
result.

The post-filter is **strictly causal**, so it buys no lookahead at all: it
adds 0.3 ms to the total, all of it arithmetic. Per 3-second clip the two
stages cost 22.39 ms and 30.26 ms respectively on the GPU.

**It does not need a GPU.** Measured on one CPU core, streaming 100 ms chunks
with state carried: **15.9 ms per chunk, 6.3x faster than real time**, using
16% of the chunk budget. Four cores buy only 1.23x more, because the LSTM is
sequential across frames and the tensors are too small to fill wide cores --
so a *fast* core matters far more than *many* cores when choosing hardware.
Per second of audio the model needs 3.75 GMAC/s, 9.3 MB of int8 weights, and
669 MB/s of memory bandwidth just to re-read the LSTM weights every frame.
That last figure is what rules out microcontroller-class parts.

The 86 ms hold-back is inside the 200-250 ms budget; the ~300 ms LIVE delay
at 100 ms blocks is not, and 50 ms blocks are the intended fix. It is **past**
ITU-T G.114's 150 ms comfort limit, which earlier versions of this project
claimed; that claim no longer holds and was traded deliberately for the
lookahead gain in §7. The GPU idles 99.4% of the time, so there is large
headroom for a slower embedded board. Target hardware is a **Raspberry Pi 5**
-- measured, 2026-09-11, PROJECT_CONTEXT_4.md Section 24. With run 0026,
stage 1 with int8 ran live in real time on a Pi 5 (58 ms per 100 ms chunk,
0 underruns) but the two-stage system did not (93.0 ms, too tight); with run
0036 the **full two-stage system runs int8 in 69.7 ms**, which is why it was
adopted. A Pi 4 runs everything correctly at a quarter of real time. The x86 figures (8.4x on four cores) turned out to say almost
nothing about ARM.

## 3. The acceptance rule — read before proposing any change

**A change is only an improvement if it holds or improves ALL THREE metrics.**

A gain on one metric does not buy a loss on another. When a metric reaches its
target it becomes a *constraint to protect*, not headroom to spend. Report all
three for every comparison and say plainly if any moved the wrong way. Never
lead with the favourable one and bury the rest.

This matters because STOI is our most *measurable* metric (Section 4) and it is
tempting to decide on it alone. Being easiest to measure does not make it the
one that matters.

---

## 4. How to measure things here -- the most important section

Rules learned by getting each one wrong first. Anyone who ignores these will
produce confident nonsense.

### 4.0 &nbsp;Check which exam a score sat before comparing it

The SNR distribution changed after run 0025. **That change alone is worth about
+0.19 PESQ with the model completely untouched** -- run 0025 scores 1.898 on the
old set and 2.098 on the new one, identical weights. The noisy baseline moved
too, from PESQ 1.199 / STOI 0.745 / +4.5 dB to 1.230 / 0.781 / +6.19.

Guards in place:

- every run records an `snr_profile` field
- old-distribution runs are named with a `_snrU` suffix
- the dashboard badges them "old SNR"
- `scripts/rescore_runs.py` has scored all 25 old checkpoints on the current
  set, written to `evaluation_<split>_current.json` inside each run

**Use those re-scored numbers when comparing an old run with a new one.** The
archived mixtures remain in `generated_test_mixtures_snr_uniform/` and load via
`FrozenMixtures(split, root=LEGACY_MIXTURE_DIR)`.

Why it was done: measured on the old testing set, run 0025 scored PESQ 1.320 /
STOI 0.732 / SI-SNR 6.12 below 0 dB and 2.057 / 0.883 / 14.76 at 0 dB and
above. Essentially all the remaining shortfall lived in the low band, which was
27% of the set and is now 10%.

### 4.1 &nbsp;Training is not reproducible. Never trust a single-run difference.

Runs 0014 and 0016 share a configuration **and a seed**, yet diverge at epoch 1
(training loss 25.1153 vs 25.6876) and by 0.029 PESQ over epochs 45-80. cuDNN's
LSTM backward pass is nondeterministic and `torch.manual_seed` does not contain
it.

Three identical runs (0016, 0017, 0018 -- seeds 0, 1, 2) measured the cost **on
the old exam**:

| metric | mean | sd across identical runs | relative |
| --- | --- | --- | --- |
| PESQ | 1.8584 | 0.0124 | 0.67% |
| STOI | 0.8544 | **0.0004** | **0.05%** |
| SI-SNR | 11.84 | 0.03 | 0.25% |

**PESQ is 13x noisier than STOI in relative terms.** Almost every result in
this project is around 0.02 PESQ -- about one standard deviation, and
uncallable. The same comparisons on STOI are decisive.

**This floor has not been re-measured on the new exam.** Until it is, treat
small differences as indicative. Re-measuring needs two more seeds of the
current baseline.

Use STOI to detect whether an effect is real; use all three to decide whether to
adopt it (§3).

### 4.2 &nbsp;The within-run standard deviation is the wrong number

Evaluations within one run vary by 0.005-0.010 PESQ. That is *evaluation
wobble*, not run-to-run variation. Quoting it as the noise floor for comparing
two runs understates the real floor by three to six times, and did so in earlier
versions of these documents.

### 4.3 &nbsp;Compare converged means, never a run's best score

PESQ oscillates once converged, so "best of N evaluations" grows with N and a
longer run wins without being better. Compare the **mean of every evaluation
from epoch 110 onward** (150-epoch runs) or **from epoch 45** (shorter ones),
always on matched windows. Models converge by roughly epoch 110 at 9.3 M
parameters. `EARLY_STOP_PATIENCE` is deliberately 0: it watched validation
*loss* while results are judged on PESQ, making training length a hidden
variable.

### 4.4 &nbsp;Deploy `best_pesq.pt`, never `best.pt`

`best.pt` is selected by validation **loss**. On run 0014 that landed on epoch
26 scoring PESQ 1.703, while the best epoch was 65 scoring 1.832. `train.py`
writes `best_pesq.pt` and everything prefers it, falling back to `last.pt`.
Runs before 0016 have none. Always build models with
`train.build_from_config`, which reads every shape from the run's own recorded
config -- constructing from module defaults silently builds the wrong
architecture for older runs.

### 4.5 &nbsp;Loss values are not comparable across different objectives

A run with an extra loss term has a higher total loss by construction. The
dashboard's overfitting panel cannot compare two runs whose objectives differ.
Only the metric panels are comparable.

## 5. How the system works

### 5.1 The idea, plainly

Sound is turned into a **picture** — a spectrogram, time across the bottom,
pitch up the side, bright where there is energy. This matters because in a
picture, **speech draws horizontal lines** (a voice holds a few pitches and
slides between them) while **a gunshot draws a vertical line** (every pitch at
once, for an instant). Separating horizontal from vertical in an image is
something a neural network is good at. Separating two tangled waveforms is not.

The network paints a **mask** over that picture: a "keep this much" dial between
0 and 1 for every pitch band at every 10 ms moment. Multiply the mask by the
noisy picture, turn it back into sound, and the noise is gone.

The model never invents sound — it can only turn existing sound down. That is
deliberate: the worst it can do is mute something, not hallucinate speech that
was never said.

### 5.2 The pipeline

```
noisy waveform
   |  STFT              301 snapshots, 257 pitch bands, 32 ms window, 10 ms hop
   v
ENCODER     5 conv layers, pitch 257 -> 129 -> 65 -> 33 -> 17 -> 9
   |        time is NEVER squeezed
   v
MEMORY      2 LSTM layers, one direction, carries state between 100 ms chunks
   |
   +------> AUX HEAD   7 noise categories, training aid, deleted for deployment
   |
   v
DECODER     5 layers back up 9 -> 257, fed by skip connections from the encoder
   |        ends in a sigmoid: a "keep this much" dial per band per frame
   v
mask x noisy spectrogram, NOISY PHASE REUSED UNCHANGED, back to audio
   |
   v
POST-FILTER  stage 2: strictly causal, predicts a residual to ADD
   |         505 K parameters, no lookahead
   v
enhanced waveform
```

Base model 3,018,072 parameters. The current best configuration widens it to
**9,264,122** (LSTM 512, encoder channels ×1.5), plus **505 K** for stage 2.

**Why the second stage is additive, not another mask.** A post-filter that
applies its own mask gives `mask2 * (mask1 * noisy)` = `(mask1 * mask2) * noisy`
— one mask on the noisy input, a re-parameterisation of stage 1, capped by the
same ideal-mask ceiling of 3.389 that stage 1 already falls short of. Adding a
residual does not collapse that way, which is the whole reason it can help.
Stage 1 can only turn sound *down*; a mask cannot restore what the noise
destroyed. Stage 2 can put energy back.

Stage 1 stays **frozen** while stage 2 trains. That is deliberate: it means no
number already measured for run 0026 can regress, which is what §3 demands.

### 5.3 Design decisions worth knowing

**Time is never compressed.** Squeezing the time axis would blur exactly the
sudden events this system exists to catch — gunshots, mortar, plosives — and
would break the fixed 10-frames-per-100 ms relationship streaming depends on.

**The LSTM is unidirectional.** A bidirectional one reads the clip backwards, so
it needs the end before it can produce the start: unusable in real time at any
latency budget. The code asserts against it.

**Streaming is bit-identical.** Processing a whole clip versus processing it in
two halves with the LSTM state carried across differs by exactly 0.00e+00.
Starting the second half cold instead differs by 1.10e-01, so the state
genuinely carries information.

**Causality is verified mechanically, not by inspection.** The test changes one
input frame and watches which outputs react. One mis-set padding anywhere would
leak the future silently and stay invisible until deployment.

**The STFT costs 16 ms of lookahead before any layer exists.** `center=True`
padding means each frame reads 256 samples past its own timestamp. The first
attempt used `center=False` and crashed, because a Hann window is exactly zero
at its first sample so that sample cannot be recovered.

**The auxiliary head is deleted for deployment**, verified: removing it changes
the audio by exactly 0.00.

### 5.4 The loss — where the only large gain came from

```
loss = si_snr_loss
     + 600  x compressed_magnitude_loss     <- the one that mattered
     +   0  x mask_loss                     <- tested, made things worse
     + 0.2  x aux_cross_entropy
```

**SI-SNR** compares the output waveform to the clean one. It was the original
and only loss, and it was the project's ceiling — see Section 6.

**Compressed magnitude** compares spectrograms bin by bin after raising both
magnitudes to the power 0.3. The compression is the point: raw magnitudes span
orders of magnitude, so a squared error on them is decided almost entirely by
loud vowels, while the quiet consonants that carry intelligibility contribute
nothing. Raising to a power below 1 squashes the range so quiet bins count — the
same trick the ear plays. A sound 1000× louder than another becomes only 8×
louder at power 0.3.

**Weights are sized by measurement, never by eye.** `python scripts/train.py
--loss-scales` backpropagates each term alone and reports the size of the update
it produces. Never compare the printed loss *values*: their ratio swings from 8
to 451 over a single run and means nothing.

---

## 6. The central finding: the ceiling was the objective, not the architecture

This is the story an LLM most needs, because the wrong version of it was
believed here for weeks and written into three documents.

**Nine runs produced nothing.** Capacity (3×), data volume (a quarter), the mask
ceiling, the auxiliary weight — every result landed between PESQ 1.65 and 1.69,
inside the noise. The documented explanation was that the model reuses the
**noisy phase** (only magnitude is masked), that this is the known ceiling of
magnitude-masking systems, and that the fix was DCCRN's complex-valued layers.

**That explanation was wrong, and `scripts/oracle_ceiling.py` disproved it in
twenty minutes with no training at all.** It cheats: it uses the clean speech
the model never sees to build the *best possible* mask, then scores it. That is
a hard ceiling — a real model can approach it but never beat it.

*Measured on the OLD flat-SNR exam. Recomputed on the current 90/10 exam the
ceiling is **PESQ 3.389**; the conclusions below are unchanged.*

```
condition                                    PESQ    STOI   SI-SNR
raw noisy input                             1.189   0.745    4.46
ideal mask, capped 1.0, NOISY phase kept    3.314   0.966   14.29   <- the real ceiling
ideal mask, uncapped, noisy phase kept      3.397   0.973   15.02
noisy magnitude + clean phase               1.289   0.807    8.59
clean magnitude + clean phase (sanity)      4.644   1.000  106.10
```

Three conclusions follow directly:

1. **Phase is not the constraint.** The ideal mask keeps the *same noisy phase*
   and still reaches 3.389 on the current exam, well past the 2.5 target.
   Fixing phase alone is worth +0.10. DCCRN is not indicated.
2. **The mask ceiling was correctly a dead end.** Letting a *perfect* mask boost
   above 1.0 is worth +0.08.
3. **We were training one thing and grading another.** The model captured 73% of
   the achievable SI-SNR but only 23% of the achievable PESQ — because SI-SNR
   was the loss. SI-SNR rewards waveform overlap, largely dictated by the phase
   we copy and never change. PESQ and STOI reward the spectral envelope, which
   is exactly what the mask controls and what the loss never asked about.

Adding the compressed-magnitude term (run 0010) gained **+0.10 PESQ, 13 sigma on
STOI** — the first and still largest real effect in the project.

---

## 7. Every experiment, and what it showed

**Every run before 0026 was trained on the old flat U(-5,+15) distribution.**
The table below gives each run's score on the **current** frozen set, produced
by `rescore_runs.py`, so every row is comparable with every other. The
old-exam column is the run's own converged validation mean at the time, kept
for reference and **not comparable** with the rest.

| run | what it tested | old exam | current PESQ | current STOI | current SI-SNR |
| --- | --- | --- | --- | --- | --- |
| 0016 | 9.3 M, lookahead 2, seed 0 | 1.857 | 2.058 | 0.8823 | 13.01 |
| 0017 | seed 1, identical | 1.871 | 2.060 | 0.8822 | 12.96 |
| 0018 | seed 2, identical | 1.847 | 2.042 | 0.8806 | 12.86 |
| 0019 | mask-vs-ideal-mask loss | 1.794 | — | — | — |
| 0020 | compression 0.15 | 1.768 | 1.964 | 0.8688 | — |
| 0021 | compression 0.5 | 1.853 | 2.037 | 0.8827 | — |
| 0022 | compression 0.7 | 1.825 | 2.010 | 0.8823 | — |
| 0023 | lookahead 10 | 1.914 | **2.102** | 0.8831 | 13.21 |
| 0024 | lookahead 4 | 1.855 | 2.081 | 0.8799 | 13.05 |
| 0025 | lookahead 6, trained on old distribution | 1.898 | 2.098 | 0.8824 | 13.19 |
| **0026** | **first native 90/10 run — best** | — | **2.102** | **0.8855** | **13.26** |

Runs 0001-0015 are the earlier series under the SI-SNR-only objective and are
kept in `runs/` with their metrics and re-scored numbers.

### What is established

Sigma is against the old-exam noise floor, which is the only one measured.

| comparison | dPESQ | dSTOI | sigma | verdict |
| --- | --- | --- | --- | --- |
| spectral loss off → on | +0.117 | +0.0074 | +13 | **real and large** |
| lookahead 2 → 6 | +0.040 | +0.0030 | +7 | **real** |
| capacity 3.0 M → 9.3 M | +0.019 | +0.0025 | +4.4 | real, small |
| data full → a quarter | −0.021 | −0.0034 | −6.1 | real, and worse |
| mask loss on | −0.059 | −0.0065 | −16 | real, and worse |
| compression 0.3 → 0.15 | −0.090 | −0.0143 | −36 | real, much worse |
| compression 0.3 → 0.5 | −0.006 | +0.0030 | +8 | mixed, not adopted |
| lookahead 6 → 10 | +0.004 | +0.0007 | ~2 | **a tie on the new exam** |

**Only two changes ever mattered: fixing the training objective, and giving the
model more future context.** Both change what the model is *told*, not how big
it is. Everything else was inside the noise or actively worse.

The lookahead 6 → 10 row is worth reading twice. On the old exam those differed
by 0.016; on the new one they are 0.004 apart, inside the noise. The extra
lookahead was earning its keep on the hard clips that are now down-weighted, so
**lookahead 6 gives the same score for 40 ms less latency**.

## 8. Ruled out by measurement — do NOT propose these

Each has been tested. Re-proposing one without new evidence wastes time already
spent.

**DCCRN, complex masks, any phase-based fix.** The ideal mask keeps the same
noisy phase and reaches PESQ 3.389, past the 2.5 target. Fixing phase alone is
worth +0.10. This project's documents once claimed phase reuse was the ceiling;
that claim is withdrawn.

**Raising `MASK_MAX` above 1.0.** Uncapping a *perfect* mask is worth +0.08.
Runs 0006 and 0007 were measuring nothing.

**A bigger model as the answer.** Capacity is real but small (+4.4 sigma on
STOI, +0.019 PESQ) and it invites the overfitting run 0015 showed is now live.
It is a supporting change, not a lever.

**Collecting more noise categories, or assuming more data is the fix.** Data
volume matters (−6.1 sigma when quartered) but we only know that *less is
worse*. Whether *more* helps is untested and untestable with what we have.

**A finer frequency view for the LSTM.** `mask_resolution.py` shows the ideal
mask needs fine detail — a two-bin blur costs 0.90 PESQ of the 2.13 available —
and harmonics sit 3–8 bins apart while the LSTM sees 9 bands of ~29 bins. That
is compelling and it is wrong: run 0013 gave the LSTM 33 bands and was worse on
every metric, fitting even the *training* data worse despite 2.3× the
parameters. The reason: the **skip connections already carry full 257-band
detail to the decoder**, so the model could always draw a sharp mask.

**Post-processing the mask.** Multiplying it by a constant, raising it to a
power, and expanding its contrast all make PESQ monotonically worse.

**Supervising the mask directly against the ideal mask.** Run 0019, −16 sigma.
Section 9 explains why, and the reason generalises.

**The auxiliary noise classifier as a quality lever.** It learns (0.52–0.59
accuracy against 0.143 chance) but does not help: run 0008 with it off matched
run 0005 with it on. It stays because it is free and deleted at deployment.

**Static int8 quantisation.** The input magnitude spectrogram spans 7.3e6:1
and int8 activations have 256 levels; everything quiet collapses to zero and
the mask error is 1.0 on a 0-1 quantity. ConvTranspose2d also has no
per-channel weight observer. Not a calibration problem. Dynamic int8 on the
LSTM is fine and is used.

**Estimating ARM speed from x86.** Off by 3x for a Pi 4. Measure on the board.

---

## 9. What is actually wrong with the model now

`scripts/mask_diagnosis.py` and `scripts/over_suppression.py` compare the
trained mask against the ideal one, on converged weights.

- correlation with the ideal mask: **0.617**
- mean absolute difference: **0.198**, on a dial that only runs 0 to 1
- roughness: only **1.77× smoother** than the ideal, so it is **not blurred**
- an ideal mask blurred to the model's own roughness still scores **2.62**
  against the model's 1.68

**The mask is sharp and in the wrong places.** Smoothness explains almost none
of the gap.

The shape of the error is a **regression toward the middle**:

| what the ideal mask wanted | ideal | model | share of all bins |
| --- | --- | --- | --- |
| 0.0 – 0.1 (delete this) | 0.031 | 0.134 | 32.7% |
| 0.9 – 1.0 (keep this) | 0.987 | 0.707 | 26.0% |

A third of all bins should be deleted outright and the model leaves four times
too much in them — that is the residual hiss. A quarter should be kept fully and
it keeps 71%. The two errors nearly cancel, so total energy kept is 0.690
against the ideal's 0.695: half a percent. An earlier note claimed a 16%
over-suppression bias; that was measured on a checkpoint 39 epochs early and is
**withdrawn**.

### Why this cannot be fixed by pushing the model to commit

Three attempts, all failed:

1. Post-hoc correction — multiply, power, contrast-expand: all monotonically
   worse.
2. A finer view for the LSTM (run 0013): worse.
3. Direct mask supervision (run 0019): worse, by 16 sigma.

Run 0019 is the instructive one. **The term did learn its own task** — mask
error fell from 0.069 to 0.055 — and the audio still got worse. Two reasons, and
the first generalises:

- **Squared error is minimised by predicting the average.** If a bin might be 0
  or might be 1 and the model is unsure, the answer that scores best is the
  middle. A grader built on squared error is therefore *won by hedging* — and
  hedging was the disease. We added a second helping of the cause.
- **It treats every bin as equally important**, but most bins are inaudible, so
  effort went into getting silence exactly right. Compression 0.15 failed the
  same way: squash too hard and near-silent bins dominate the grade.

**The lesson: hedging is the correct response to uncertainty, not a habit to be
trained out.** You cannot punish a model into confidence it has not earned.
Moving *where* the error is measured achieves nothing while *how* it is measured
stays the same.

---

## 10. The data

### Speech — 1292 clips, all exactly 3.000 seconds at 16 kHz

AI4Bharat IndicVoices-R (22 Indian languages) plus LibriSpeech (English).
Composition: 25% English, 25% Hindi, 50% spread across 21 further languages.
Splits are **speaker-disjoint** — no speaker appears in more than one split.

### Noise — 636 clips, 7 categories, all exactly 3.000 seconds

`ambient`, `assault_rifle`, `engine`, `heavy_artillery`, `helicopter`,
`mortar`, `rocket`.

Two standing rules, both deliberate:

- Only categories present in **all three splits** are used. A category in
  training but not testing cannot be evaluated.
- Excess data is **ignored**, not used — balance targets are held exactly rather
  than absorbing whatever surplus exists.

`dataset/noise_classes.txt` pins the label order (line order = label index) so
adding a class appends rather than renumbering, which would silently invalidate
every existing checkpoint.

### Mixing

Speech and noise are combined **on the fly during training**, categories
sampled uniformly, at an SNR drawn **90% from U(0,+15) dB and 10% from
U(-5,0) dB**. The model therefore never sees the same example twice. This
replaced a flat U(-5,+15) after run 0025; see §4.0, and note that frozen sets
get an *exact* 90/10 composition rather than a sampled one, because a random
draw over 143 clips landed at 14% and a yardstick should be exact.

The SNR definition matters and has been "fixed" wrongly before: **noise power is
measured over the whole window, speech power over the speech-active region
only.** Measuring noise inside the speech span would let a near-silent span
demand an enormous scale factor.

**Evaluation uses 268 frozen mixtures on disk** — 143 validation, 125 testing —
generated once at seed 20260830 and never regenerated. If validation audio were
re-rolled each epoch, its curve would wobble with the draw rather than the
model, and comparing epoch 40 with epoch 20 would be meaningless.

Verified after writing: every file is exactly 48000 samples, requested and
measured SNR agree to 0.0000 dB across all 268, no mixture crosses a split, and
validation/testing speakers are disjoint.

### Incoming: a human-annotated noise round from MAD — NOT arrived yet

**Status: sent out 2026-09-13, nothing has come back. No merge script exists.
When the folders arrive, write it to the spec below.** Until then there is
nothing to do here.

**What went out.** Three people — Rashmit, Vaibhav, Lochan — each got a zip
(`C:\Users\k.lochan\Downloads\noise_annotator_<person>.zip`) containing the
Noise Annotator web tool (source at `C:\noise_annotator\`, its own README and
AGENTS.md describe it) with raw files already in its `dataset/` folder.
**Rashmit and Vaibhav: 500 training / 150 validation / 150 testing each.
Lochan: 30 / 6 / 6** (the smaller round-1 hand, kept unchanged because it
was already in use). 1,642 raw files in total. The raw files are from the
Kaggle **MAD** dataset (Kim, Yoon & Jung, *Sci. Data* 2024; 7,466 clips of
2–10 s, 16 kHz mono, sliced from military YouTube videos, 7 classes:
communication, gunshot, footsteps, shelling, vehicle, helicopter, fighter).
They were dealt by `C:\mad_splits\make_mad_splits.py` with these guarantees,
all verified on the written output: no file dealt twice to anyone, by path
or by bytes; **no YouTube video feeds more than one split** — the training,
validation and testing videos are disjoint sets across all three people, so
a slice of a recording used for evaluation was never trained on; training
files from Kaggle's training set only, validation and testing from Kaggle's
test set only; nothing under 3 s; clips spread over as many videos as
possible (each 500-hand spans 330+ videos). Rashmit and Vaibhav share videos
but never files. File names are `<segment>_<n>.wav` and carry no class
information — the annotators label blind. `C:\mad_splits\README.md` has the
round history.

**The answer key is `C:\mad_splits\assignment.csv`** (person, split, file,
Kaggle path, MAD label and class name, duration, YouTube title and URL). It is
not in any zip. The full 1.2 GB Kaggle download is at `C:\mad_splits\kaggle\`;
a second round is one run of the script with a new `SEED`.

**What comes back from each person: a `classified/` folder.** Its format is
deliberately this project's format, so nothing needs converting:

```
classified/
├── training/<category>/<category>_001.wav, _002.wav ...
├── validation/<category>/...
├── testing/<category>/...
└── manifest.csv    clip, split, category, source, start, end, annotator,
                    created, level_dbfs   -- one row per clip
```

Every clip is exactly 48000 samples, 16 kHz, mono, PCM_16 — the annotator's
server enforces this and refuses anything else. Categories are our seven
(`categories.json` in the tool is our `noise_classes.txt` order), normalised
to `lowercase_with_underscores`; an annotator *can* add a new category and it
is appended to their list. Windows may overlap, so one 10 s raw file can yield
several clips; expect on the order of 1,500–4,000 clips in total across the
three people, most of them from Rashmit and Vaibhav, not 1,642. `source` is the raw file name (`232_7.wav`), which joins to
`assignment.csv` → `file`.

**Two things that make a hand-merge wrong.** (1) Each person's numbering
restarts at 001, so Rashmit's `helicopter_003.wav` and Vaibhav's are different
clips with the same name. (2) The split folders are named `training/` etc.,
ours are `noise_train/` etc.

**The script to write: `scripts/merge_annotated_noise.py`.** Same shape as
`add_noise_class.py` (dry run by default, `--apply` to write, copies never
moves). Drop folder `incoming_noise/<person>/classified/`. It must:

1. Read each person's `manifest.csv`; check every listed file exists and is
   exactly 48000 samples / 16 kHz / mono / PCM_16 — refuse the whole run on
   any miss, the way the rest of the pipeline refuses rather than pads.
2. Content-hash every incoming clip against every other incoming clip **and**
   against everything already in `dataset/noise_data/`. Overlapping windows
   from the same raw file are fine (different bytes); byte-identical clips are
   not. Expect zero; report the count either way.
3. Map `training→noise_train`, `validation→noise_validation`,
   `testing→noise_testing`.
4. Apply the standing rule: a category is installed only if, *after* the merge,
   it exists in all three splits. Our seven already do. A category one
   annotator invented and only put in training is held back and listed, not
   silently dropped and not silently installed.
5. Write into `dataset/noise_data/noise_<split>/<category>/` numbered from the
   **next free index in that folder**. Existing files are never renumbered —
   the frozen manifests reference them by name.
6. Append new categories to `dataset/noise_classes.txt` at the bottom only, as
   `add_noise_class.py` does. Never reorder.
7. Write `dataset/noise_provenance.csv` (append if present): new filename,
   split, category, annotator, raw source file, start, end, and — joined from
   `assignment.csv` — the MAD label, class name and YouTube URL. This is what
   makes a clip traceable back to a video later.
8. Rebuild the manifests (`build_balanced_manifests.py`). Remember the balance
   rule: targets are held exactly and surplus is ignored, so more raw material
   in a category does not automatically mean more of it is used — check the
   targets after and decide deliberately.
9. **Do not regenerate the 268 frozen evaluation mixtures.** They contain none
   of the new noise and stay valid for before/after comparison. A second
   frozen set is a separate, deliberate step (see `add_noise_class.py`).

**A second, smaller script, or a flag on the first: score the annotators.**
Join each `manifest.csv` row's `source` to `assignment.csv` and tabulate our
category against MAD's class per person. The mapping is loose and must be
stated in the output, not assumed: helicopter↔helicopter, gunshot↔assault_rifle,
vehicle↔engine, shelling↔heavy_artillery *or* mortar, fighter has no clean
home (engine at best), footsteps→ambient or discarded, communication (people
talking) should have been discarded. Also report per person: files discarded,
clips per file, clips flagged `quiet` (level under −45 dBFS — the annotator
flags these but still exports them; listen before installing). There is no
inter-annotator agreement to compute — the three sets are disjoint by design.

**Before any of this: play a few clips from each person.** The format check
proves the container, not the content.

---

## 11. The code

Everything lives in `scripts/`. Nothing imports a model directly except
`train.py`, which looks one up in `ARCHITECTURES` — adding a TCN is one dict
entry plus its model file.

| file | what it does |
| --- | --- |
| `model_crn.py` | the model. Run it directly to execute every check from all six build steps as a regression test |
| `train.py` | the training loop, the loss functions, `--loss-scales`, and the shared model loader |
| `mixing.py` | on-the-fly mixing and the frozen evaluation set |
| `evaluate.py` | PESQ, STOI, SI-SNR reported three ways (noisy / enhanced / gain) and per noise category |
| `stream.py` | real-time streaming inference: chunk in, finished audio out, state carried. Run it directly to check it against the whole-clip path at five chunk sizes |
| `runlog.py` | one directory per run: config, metrics, source snapshot, checkpoints |
| `dashboard.py` | stdlib-only web dashboard for comparing runs, reachable from a phone on the same network |
| `oracle_ceiling.py` | the cheating test that measures the architecture's hard ceiling |
| `mask_resolution.py` | blurs the ideal mask deliberately to find how much detail it needs |
| `mask_diagnosis.py` | compares the trained mask to the ideal one |
| `over_suppression.py` | where the mask's error lives, and whether correcting it helps |
| `visualize_pipeline.py` | figures showing the pipeline on real audio |
| `build_balanced_manifests.py`, `split_speech_dataset.py`, `normalize_clip_lengths.py`, `generate_test_mixtures.py`, `add_noise_class.py`, `collect_english.py`, `rechunk_long_clips*.py` | the data pipeline, mostly run once |

### Streaming: the same audio, computed incrementally

`stream.py` is what runs on the Pi and what any real deployment uses. The
offline forward pass re-reads the whole clip; a live system cannot, so the
streamer keeps state instead. Three parts of the model have a past and each
needs a different treatment:

- **The LSTM** is genuinely stateful -- its hidden state IS the memory, so it
  is carried and every frame is fed to it exactly once. Exact by construction.
- **The encoder and decoder** are 3-frame convolutions over time. Each layer
  keeps its own last 2 input frames and is handed only the new ones.
- **The post-filter** is a dilated causal stack; each layer keeps 2x its own
  dilation of history.

**This was rewritten on 2026-09-23 and it matters for latency.** The previous
version re-ran the whole encoder and decoder over a retained 24-frame window
on every push and kept the newest outputs. Correct, but it made a push cost
roughly the same regardless of how much new audio it carried -- about thirty
frames of work to finish two -- so a 20 ms chunk cost nearly as much as a
100 ms one, and shrinking the chunk to cut delay bought nothing. The cost now
tracks the new frames.

**The gain is real but smaller than the frame count suggests: 1.5x at 20 ms,
not 5x.** The convolutions are now negligible (2.3 ms of a 15.7 ms push). What
remains is irreducible without further work -- the LSTM re-reads all 6.7 M of
its weights every 10 ms frame (3.6 ms), and the post-filter is ~40 small
tensor operations whose launch overhead is ~0.1 ms each whatever their size
(7.0 ms). Fusing BatchNorm into the convolutions and dropping the per-layer
pad might recover ~2 ms; nobody has tried it.

**Chunk size is now a free knob.** `CHUNK_FRAMES` changes the delay and the
per-push cost and NOTHING else -- verified bit-identical output at 100, 20,
10, 43.75 and 6.25 ms. It is not a model parameter and does not affect PESQ,
STOI or SI-SNR at all.

**It is checked against the offline path, not trusted.** A streamer that
mishandled its history would still produce something that sounds like speech
and merely clicks at the chunk seams -- damage that averages away invisibly.
So `python scripts/stream.py` demands a max difference under 1e-3 from the
whole-clip forward pass at every chunk size, with and without the
post-filter, AND checks the seams specifically. It currently reads ~1e-7. The
rewrite was additionally checked to be sample-for-sample identical to the old
streamer on all three trained runs (0026 CRN, 0036 ERB + post-filter, 0035
TCN), to hold its buffers bounded over five minutes of 20 ms pushes, and to
leave `deploy/selfcheck.py` printing GO with zero metric drift.

### Every run records how to reproduce it

`runs/NNNN_arch_tag/` holds `config.json` (every knob, the git commit, measured
latency), `metrics.jsonl` (one appended line per epoch, flushed on every write),
`notes.md` (why the run was launched), `checkpoints/`, `samples/`, and
**`source/` — a copy of the code that produced it**. Comparing two runs means
diffing two snapshots rather than trusting the working tree has not moved. A git
commit alone is not enough: runs are routinely launched with uncommitted edits,
and that is exactly when knowing the code matters most.

`metrics.jsonl` is append-only by design. A run that crashes at epoch 40 keeps
40 epochs of history.

### Tunable parameters

Every file puts its tunable knobs in a clearly marked block at the top, split
into **TUNING KNOBS** (safe to change, annotated with what changing each does
and a sensible range) and **LOCKED** (fixed by data already on disk — changing
them silently invalidates it). Sample rate and clip length are locked by the
recorded dataset and the frozen evaluation mixtures.

---

## 12. Running it

```bash
# check the model still works - runs every build-step test
python scripts/model_crn.py

# train (the current best configuration)
python scripts/train.py --epochs 150 --eval-every 5 \
    --lstm-hidden 512 --channel-scale 1.5 --note "why you ran this"

# size a new loss term by measurement, not by eye
python scripts/train.py --loss-scales --lstm-hidden 512 --channel-scale 1.5

# score a run on held-out audio (prefers best_pesq.pt automatically)
python scripts/evaluate.py 0026_crn_comp03_spec600_la6_s0 --split testing

# stage 2: the post-filter. The module self-test ASSERTS causality.
python scripts/postfilter.py
python scripts/postfilter_test.py

# streaming: demands the chunked path match the whole-clip path exactly, at
# five chunk sizes, with and without the post-filter
python scripts/stream.py

# compare against a published system on our own frozen set
python scripts/benchmark_external.py --enhanced-dir <dir> --run 0026_crn_comp03_spec600_la6_s0

# the ceiling, and the diagnostics - none of these need training
python scripts/oracle_ceiling.py --split validation
python scripts/mask_resolution.py --split validation
python scripts/mask_diagnosis.py 0026_crn_comp03_spec600_la6_s0
python scripts/over_suppression.py 0026_crn_comp03_spec600_la6_s0

# ONE server, three pages, reachable from a phone on the same wifi:
#   /dashboard  training runs, curves, comparisons, the stop switch
#   /bench      run the model on a test clip, an upload, or the microphone
#   /lab        build a mixture by hand: one speaker, up to four noises
python scripts/serve.py
```

Useful flags: `--seed` (repeat a configuration to measure run-to-run spread),
`--spec-weight`, `--compression`, `--mask-weight`, `--data-fraction`,
`--mask-max`, `--limit-batches` (smoke testing).

An epoch takes 11.3 s at 3.0 M parameters and 17.3 s at 9.3 M. A 150-epoch run
is about an hour.

**Environment:** Python with PyTorch built for CUDA 12.8+ — the RTX 5060 is
Blackwell (sm_120) and older builds will not run on it. The `.venv` is ~5 GB and
gitignored; the exact install command is in `PROJECT_CONTEXT_4.md` Section 6.
Windows Smart App Control blocks unsigned PyTorch DLLs and must be off for the
install.

---

## 13. What to do next

**Run 0026 is still the best model, and every idea on the previous list has
been closed by measurement.** Runs 0029 to 0034 tested the last untried lever --
changing what the model is TOLD during training -- across four approaches.
PROJECT_CONTEXT_4.md Section 22 has the full write-up; the short version:

| tried | outcome |
| --- | --- |
| Multi-resolution spectral loss (0029) | -4.6 sigma PESQ, rejected |
| Local-SNR auxiliary head (0030, 0031) | null at both a whisper and a shout |
| MetricGAN from scratch (0032) | worse on all three metrics |
| MetricGAN as a fine-tune (0033, 0034) | works: +0.0127 PESQ for -0.060 SI-SNR |

### The three findings that should shape what happens next

**We are on a trade-off curve, not below one.** The fine-tuned MetricGAN run
bought PESQ and paid for it in SI-SNR, consistently, at every checkpoint. Gains
in one metric now measurably cost another, which means the earlier nulls were
not near-misses waiting for a better version.

**The wall is not architectural, and it is not missing information.** An ideal
per-bin mask reaches PESQ 3.389 using this same architecture and this same
reused phase, so the answer is expressible by what was built. And
`scripts/probe_local_snr.py` showed a 513-parameter linear readout recovers
speech presence to 0.124 from a frozen run 0026 that was never taught it,
against 0.352 on an untrained model -- the representation already holds what we
kept trying to give it. What remains is that the model HEDGES, and every
squared-error grader that can be written is minimised by hedging.

**We are now AHEAD of published state of the art on this data.** DeepFilterNet
v0.5.6 scores 1.947 / 0.8508 / 11.67 on our frozen test set; the two-stage
system scores 1.983 / 0.8660 / 14.02 — ahead on all three at once. Before the
post-filter we were level on PESQ and ahead on the other two. This is the
single most useful fact for the submission, and it reframes six runs of nulls:
the model is not weak.

**A causal post-filter broke the plateau.** It is the first change since the
loss fix to help the deliverable rather than a diagnostic. Run 0026 frozen, a
505 K second stage predicting an additive residual: +0.0483 PESQ, +0.0034 STOI,
+0.343 SI-SNR on held-out audio, with validation agreeing at +5.8 / +5.3 /
+10.6 sigma. The held-out gain landing close to the validation gain is the part
that matters — it generalises rather than being selection luck.

### What is actually left

1. **Decide whether the 2.50 PESQ target is achievable on this data.** It was
   set before anything was known about how hard this mixture set is, and
   nothing since has demonstrated it is reachable on 3-second clips at these
   SNRs within an 86 ms algorithmic delay. A published system is where we are. This is a
   submission-level judgement, not a coding task, and it should be made
   explicitly rather than by continuing to grind.
2. **More training data.** The one lever never tested, because we have none.
   Less is measurably worse; more is unknown. A week of effort, so it deserves
   a decision rather than a launch.
3. **A larger model with a relaxed latency budget.** Capacity is a real but
   small lever at the current size, and the live delay is already over the
   ITU-T G.114 150 ms figure the deck quotes -- §2 for the real numbers. If the
   budget can move, this becomes a different problem. Note that a bigger model
   costs THROUGHPUT, not delay: delay is set by the lookahead and the block
   size, so a slower model is free until it stops meeting the deadline.
4. **The SIH deck needs its latency slide corrected, and its stage-1 numbers
   checked.** It was updated once and is now partly stale again.
   `SIH2026_26052_ANC_Idea.pptx` carries the two-stage numbers, the
   DeepFilterNet comparison and four regenerated charts, and team name, team
   ID and theme on slide 1 are still `[ FILL IN ]`. Two things are now wrong:

   - **It quotes 177 ms as end-to-end latency**, built as
     `76 + 100 + 0.9` on the training GPU. §2 explains why that figure does not
     survive contact with the live path. A judge who adds the numbers up will
     find the compute term does not match the Pi's 69.7 ms. Replace it with
     the two honest lines: an 86 ms algorithmic delay with 1.4x real-time
     throughput on a Pi 5, and the live mouth-to-ear delay for whatever block
     size ships.
   - **Its metrics may predate the ERB swap.** The shipped stage 1 became run
     0036 on 2026-09-19 (item 7); if the deck still shows 1.983 / 0.866 /
     14.02 it is describing the previous pairing.
5. **Wire `scripts/pesq_labels.py` into `evaluate.py` and `rescore_runs.py`.**
   It scores PESQ across worker processes, 5.7x faster than serial, and is
   already on master. Small, safe, and it speeds up every future measurement.
6. ~~**Quantise to int8 for the embedded board.**~~ Done and measured
   (Section 24). Dynamic int8 on the LSTM is free in quality and buys 1.45-1.7x
   on ARM -- nothing on x86 -- and on a Pi 5 it is what makes stage 1 real time.
   Static int8 is **closed**: the input magnitude spans a 7e6 dynamic range and
   256 levels destroy the mask (Section 22). "Full int8 through ONNX should give
   3-4x" was a guess and is withdrawn. The Pi 5 is the target; a Pi 4 is
   measured too slow; a Pi Zero 2 W is out of the question.
7. ~~**The ERB hybrid is built, measured, and waiting for a board.**~~
   **Adopted, 2026-09-19 -- a deliberate speed-for-quality trade, eyes open.**
   Branch `erb-hybrid` (tag `pre-erb` marks where it forked), merged.
   Rejected as a *free* change (run 0036: PESQ holds at -0.8 sigma, STOI
   -5 sigma, SI-SNR -3 sigma, because halving the frequency axis also halves
   the LSTM's bands from 9 to 5, and run 0013 already showed that matters).
   The Pi trade half of the question came back positive: `bench_configs.py
   --run 0036_crn_cat_comp03_res1_spec600_la6_s0`, int8 two-stage **69.7 ms**,
   comfortably under the 100 ms deadline (run 0026 sits at 93.0 ms the same
   day, on the same board) -- every configuration passed, including fp32
   two-stage at 85.6 ms.

   The post-filter was retrained against 0036's output
   (`postfilter_test.py --stage1 0036... --lookahead 0 --epochs 150`,
   converged at epoch 50) and scored on held-out testing against the
   pairing it replaces, 0026 + the original post-filter (1.983 / 0.8660 /
   14.02):

       PESQ +0.0069 (+1.1 sigma)   STOI -0.0005 (-1.25 sigma)   SI-SNR -0.068 (-3.1 sigma)

   STOI and SI-SNR both regress past the noise floor -- by the standing rule
   (every metric must hold or improve) this is not a free upgrade, and the
   earlier guess on this page ("should pass on net") did not hold up under
   measurement. **Decided anyway**: the regression is small in absolute
   terms and the Pi 5 speed margin was judged worth it. `models/
   postfilter_ch48_la0_snr25.pt` now holds the retrained (0036-paired)
   weights -- the pre-retrain version (trained against 0026) stays
   recoverable from git history if this needs revisiting.
8. ~~**Make the Pi 4 real-time, if it still matters.**~~ **Does not matter
   any more.** Item 7 resolved yes: the ERB model clears the Pi 5 two-stage
   deadline, which was the condition for retiring this.
9. **The TCN (branch `tcn`, run 0035): measured on the Pi 5, does not
   displace the CRN.** Built in an earlier session as a from-scratch
   alternative to the LSTM -- two stacks of dilated causal convolutions,
   same encoder/decoder/mask, a tie on every quality metric (PESQ +0.9
   sigma, STOI -0.1, SI-SNR +1.6 -- a tie, not a win). The reason it was
   worth measuring: an LSTM re-reads all its weights every 10 ms frame
   (758 MB/s) where the TCN's convolutions reuse weights across a whole
   chunk (81 MB/s), and a Pi is exactly the kind of bandwidth-starved board
   where that should matter. It does, partly -- fp32 stage 1 is 64.7 ms
   against the CRN's 80.1 ms -- but int8 erases the gap and then some:
   dynamic quantisation only touches `LSTM` and `Linear` layers, so it
   barely moves the TCN (64.7 -> 68.4 ms, no real Linear/LSTM to quantise)
   while it cuts the CRN by 28% (80.1 -> 57.4 ms). int8 stage 1 ends up
   **57.4 ms for the CRN against 68.4 ms for the TCN** -- the CRN wins where
   it actually matters for deployment. No case to switch architectures.
   `stream.py` needed zero changes to run the TCN -- its streaming state was
   already built to match `Memory`'s exact interface, verified to 1.19e-07
   against the offline path before ever reaching the board.
10. ~~**Measure a smaller audio block on the Pi 5 -- the one thing standing
    between us and the latency budget.**~~ **Measured, 2026-09-23 -- helps
    cold, but this board cannot hold it warm.** `stream.CHUNK_FRAMES = 5`
    (50 ms) on run 0036, int8 two-stage, on the freshly-reflashed Pi 5: a
    short run reproduces the prediction exactly, 40.5 ms median / 41.2 ms
    worst against a 50 ms deadline, comfortable headroom. That result only
    held for about the first minute.

    An 8-minute continuous-load run (chunks pushed back to back with zero
    idle time -- harsher than real use, see caveat below) tells a different
    story. Temperature climbs from 52.7 to 84-86 degC within about 90
    seconds on this board, which has no heatsink, and `vcgencmd
    get_throttled` shows it sitting inside the throttle zone (currently-
    throttled and frequency-capped bits set) for nearly the rest of the
    run. Median drifts from 40.6 ms cold to 52.2 ms hot (+11.6 ms), worst
    case spikes as high as 319 ms, and across all 9,686 chunks **41.29%
    missed the 50 ms deadline** -- a listener would hear that as regular
    dropouts, not a rare edge case.

    **Caveat that keeps this from being a clean no.** The test denies the
    chip any idle time between chunks; real `live_mic.py` only computes
    ~40 ms out of every 50 ms window, leaving it ~20% idle per chunk to
    shed heat that this test never gives it. Real sustained behaviour is
    probably better than this number, but has not been measured -- and
    neither has the 100 ms config's own sustained-load number, so there is
    no clean baseline yet to say how much better.

    **Decided**: 50 ms chunks stay parked as a change to retest once a
    heatsink is fitted, not adopted now. The deployed bundle and
    `live_mic.py` were never touched for either test -- both ran from a
    standalone script that patches `stream.CHUNK_FRAMES` in-process, not on
    disk -- so the shipped config is still 100 ms, unchanged, and there was
    nothing to revert. Still unmeasured either way: the sound card's own
    input and output latency, which adds to every figure above.
    `sd.query_devices()` reports it and an open `sd.Stream` exposes
    `.latency`; a loopback with a click, recorded with and without
    `--bypass`, would give ground truth.

### One experiment recorded as inconclusive, not a clean negative

Raising the low-SNR share of training mixtures from 10% to 25% changed nothing
overall (+0.0425 held-out against +0.0431) and the target band got *nominally*
worse. Two flaws in the test, both worth not repeating: early stopping used
validation PESQ, which is 90/10 weighted and therefore biased against the
hypothesis, so the run was cut at epoch 80 where the control ran to 135; and
the band being targeted holds **twelve clips**, where a 0.0075 PESQ difference
is not a measurement. Testing it properly needs a low-SNR-weighted stopping
criterion and more evaluation clips down there.

### Planned: multiple noises per mixture

Every mixture so far contains exactly ONE noise clip. A real battlefield has
rotor noise *and* gunfire at once, so this is the least realistic simplification
left. Training mixtures would be built dynamically as now; validation and
testing would be frozen on disk as now.

Three decisions come before any code:

- **SNR definition.** The COMBINED noise should hit the target SNR, with the
  power budget split among the clips. Giving each its own SNR makes the total
  louder than any of them and the difficulty drifts.
- **The category label.** The aux head is a 7-way softmax; "the category" is
  undefined with three noises playing. Runs 0008, 0030 and 0031 all showed the
  head contributes nothing, so DELETE it rather than redesign it. It also
  breaks per-category reporting in evaluate.py, the dashboard and the bench.
- **Simultaneous or staggered.** All k spanning the clip is the simple version.
  Continuous noise throughout with impulsive events at random offsets is more
  realistic and a bigger build; do the simple one first.

Steps: `mix_at_snr` becomes `mix_many` (draw k=1..3 from distinct categories,
split the power budget, sum, clip-guard once at the end) -> metadata carries a
LIST of categories and weights -> regenerate the frozen sets, archiving the old
ones the way `generated_test_mixtures_snr_uniform/` preserves the pre-90/10 set
-> update the consumers -> re-measure the noisy baseline and the ideal-mask
ceiling -> retrain stage 1, then the post-filter -> re-measure the noise floor
with three seeds.

**This invalidates every number in the project, not the code.** PESQ 1.983, the
3.389 ceiling, the 0.0061 noise floor and the DeepFilterNet comparison were all
measured on single-noise mixtures and none of them transfer. The SNR
distribution change was worth +0.19 PESQ on its own with the model untouched;
expect the same kind of shift here, in the other direction, because overlapping
noises are harder to separate. **A fall from 1.983 is the exam getting harder,
not the model getting worse** -- but that only holds if it is said before the
run, not after.

Cost: about a day to build and regenerate, then three retrains plus three
noise-floor seeds, so 12-15 GPU hours before there is a trustworthy number. Do
not start it mid-submission: it leaves the old numbers invalid and the new ones
not yet earned.

**Not on the list, deliberately:** DCCRN and phase fixes, a bigger model as the
primary lever, mask post-processing, direct mask supervision, a finer frequency
view for the LSTM, more compression in either direction, deep filtering,
multi-resolution spectral loss, the local-SNR auxiliary head, and MetricGAN.
Section 8 and PROJECT_CONTEXT_4.md Section 22 carry the measurement that killed
each.

### Waiting on: the human-annotated noise round

Three people are annotating MAD files with the Noise Annotator tool — 800
each for Rashmit and Vaibhav, 42 for Lochan (§10, "Incoming"). ~~When their
`classified/` folders come back, the job is to write
`scripts/merge_annotated_noise.py`~~ **The tooling is already built and
tested against synthetic data** — `scripts/merge_annotated_noise.py` (dry
run by default; format check, content-hash dedup, category-in-all-splits
rule) and `scripts/score_annotators.py` (tabulates each person's category
against the MAD answer key). Both install into `dataset_v2/`, a sibling of
`dataset/` kept separate so every past run stays comparable against the
original set. `dataset_v2/` also carries `mixing.py`'s new opt-in
`multi_noise=True` mixing mode (1-4 noise clips per mixture, RMS-summed
so "treated as one for SNR" means what it says) for when this data lands.
Nothing to do until the folders actually arrive in `incoming_noise/`.

### The honest position on the targets

On held-out audio STOI is met. PESQ sits at 1.93 against 2.5 and SI-SNR at
13.62 against 15. Every lever that ever worked has been spent, every remaining
one has been measured and closed, and the model is now ahead of published work on
this data. Closing a 0.57 PESQ gap now needs something categorical -- much more
data, a bigger model with more latency, or a revised target -- rather than
another knob. That is a submission decision as much as a technical one, and it
is the decision the project is now waiting on.

## 14. Standing working rules for this project

- **Ask before committing to git.** Proposing a commit is welcome; running one
  is the user's call.
- **Propose new work, do not start it.** Suggest the next experiment and wait. A
  question like "what is the next step?" is a request for an answer, not for the
  work to begin.
- **Build in small verified steps.** Write one piece, run it, show it works,
  explain it in plain language, then move on. Do not build three stages ahead
  and verify at the end.
- **Explain in simple terms**, with analogies where they help. Technical
  precision only where it changes a decision.
- **Put tunable parameters in an obvious annotated block at the top of the
  file**, separated from constants that must not change.

---

## 15. Glossary

**Spectrogram** — sound drawn as a picture: time across, pitch up, brightness is
loudness.

**Mask** — the "keep this much" dial, 0 to 1, that the model produces for every
pitch band at every moment. It is the only thing the model outputs.

**Phase / timing** — a spectrogram has a loudness part and a timing part. We
mask loudness and copy the timing from the noisy input unchanged. This was long
believed to be the ceiling and is not.

**PESQ** — a score for how pleasant and undistorted speech sounds, roughly 1 to
4.5. Our noisiest measurement.

**STOI** — a score for how understandable the words are, 0 to 1. Our most
precise measurement by a factor of thirteen.

**SI-SNR** — how much louder the speech is than what is left of the noise, in
dB, ignoring overall volume.

**Ideal mask** — the perfect mask, computed by cheating with the clean speech.
Used to measure ceilings; never available at deployment.

**Lookahead** — how far into the future the model is allowed to peek. Costs
latency one-for-one and cannot be bought back with faster hardware.

**Sigma** — how many standard deviations of the *measured run-to-run noise* an
effect is. Under 2 sigma means it cannot be called.
