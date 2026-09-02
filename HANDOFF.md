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

Best model until run 0026 finishes is **run 0025**, scored on the current
held-out testing set -- 125 mixtures never used for training and never used to
choose the model:

```
                 raw input     our model     target     status
PESQ               1.223         1.925        2.5       short by 0.58
STOI               0.754         0.861        0.85      MET
SI-SNR (dB)        6.14         13.52        15.0       short by 1.5
latency                         177 ms       500-800    comfortably inside
```

**STOI now clears its target on held-out audio.** Under the project's
acceptance rule (§3) it becomes a constraint to protect rather than a goal to
chase. PESQ and SI-SNR are the two still short.

On the **validation split** (143 mixtures, also used for checkpoint selection,
so optimistic) the same model reads PESQ 2.098 / STOI 0.8824 / SI-SNR 13.19.
Quote the testing numbers externally.

**Run 0026 is training** as of writing: the same configuration, but the first
model whose training and evaluation distributions agree. Every run up to 0025
was trained on the old flat distribution and is only being *scored* on the new
one. At epoch 80 it reads PESQ 2.107 / STOI 0.8856, already above 0025.

### Latency, measured on an RTX 5060 laptop GPU

```
 76 ms   lookahead    (16 ms unavoidable from STFT framing + 60 ms chosen)
100 ms   chunk wait   (we process 100 ms at a time, so we wait for it to arrive)
0.65 ms  compute
-------
177 ms   end to end
```

Inside the 500-800 ms budget the problem statement allows. It is **past**
ITU-T G.114's 150 ms comfort limit, which earlier versions of this project
claimed; that claim no longer holds and was traded deliberately for the
lookahead gain in §7. The GPU idles 99.4% of the time, so there is large
headroom for a slower embedded board. Target hardware is the NVIDIA Jetson AGX
Orin, not yet tested.

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
```

Base model 3,018,072 parameters. The current best configuration widens it to
**9,264,512** (LSTM 512, encoder channels ×1.5).

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
   and still reaches 3.31, well past the 2.5 target. Fixing phase alone is worth
   +0.10. DCCRN is not indicated.
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

**All 26 runs were trained on the old flat U(-5,+15) distribution except 0026.**
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
| 0025 | **lookahead 6, chosen** | 1.898 | **2.098** | 0.8824 | 13.19 |
| 0026 | first native 90/10 run | — | *training* | | |

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
noisy phase and reaches PESQ 3.31, past the 2.5 target. Fixing phase alone is
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
| `runlog.py` | one directory per run: config, metrics, source snapshot, checkpoints |
| `dashboard.py` | stdlib-only web dashboard for comparing runs, reachable from a phone on the same network |
| `oracle_ceiling.py` | the cheating test that measures the architecture's hard ceiling |
| `mask_resolution.py` | blurs the ideal mask deliberately to find how much detail it needs |
| `mask_diagnosis.py` | compares the trained mask to the ideal one |
| `over_suppression.py` | where the mask's error lives, and whether correcting it helps |
| `visualize_pipeline.py` | figures showing the pipeline on real audio |
| `build_balanced_manifests.py`, `split_speech_dataset.py`, `normalize_clip_lengths.py`, `generate_test_mixtures.py`, `add_noise_class.py`, `collect_english.py`, `rechunk_long_clips*.py` | the data pipeline, mostly run once |

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
python scripts/evaluate.py 0016_crn_spec600_h512_data1 --split testing

# the ceiling, and the diagnostics - none of these need training
python scripts/oracle_ceiling.py --split validation
python scripts/mask_resolution.py --split validation
python scripts/mask_diagnosis.py 0016_crn_spec600_h512_data1
python scripts/over_suppression.py 0016_crn_spec600_h512_data1

# the dashboard, reachable from a phone on the same wifi
python scripts/dashboard.py
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

**In flight:** run 0026, the first model whose training and evaluation
distributions agree. Every run up to 0025 studied for the old exam and is only
being *scored* on the new one. At epoch 80 it read PESQ 2.107 / STOI 0.8856,
already above run 0025's 2.098.

**Then, in order:**

1. **Re-measure the noise floor on the new exam.** The sd of 0.0004 STOI and
   0.0124 PESQ that every judgement here rests on was measured on the old
   mixtures. Two more seeds of the current baseline, about two hours. Until it
   is done, small differences on the new exam are indicative rather than
   established.
2. **Re-run `oracle_ceiling.py`.** The PESQ 3.314 ceiling was computed on the
   old set. It needs recomputing before it is quoted again. No training needed.
3. **Update the SIH deck.** It still carries old-exam numbers and asserts
   ITU-T G.114's 150 ms limit, which 177 ms does not meet.
4. **Push capacity further**, now established as real but small, and watch for
   the overfitting run 0015 showed is live.
5. **The data question.** We know less is worse and have never tested whether
   more helps, because we have none. A week of effort, so it deserves a
   decision rather than a launch.

**Not on the list, deliberately:** DCCRN and phase fixes, a bigger model as the
primary lever, mask post-processing, direct mask supervision, a finer frequency
view for the LSTM, and more compression in either direction. §8 has the
measurement that killed each.

### The honest position on the targets

On held-out audio, STOI is met. PESQ sits around 1.93 against 2.5 and SI-SNR
around 13.5 against 15. The two levers that ever worked have both been spent,
and the remaining knobs are worth hundredths. Closing a 0.58 PESQ gap needs
something categorical -- substantially more data, or a better mask estimator --
rather than another knob. That is a submission decision as much as a technical
one.

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
