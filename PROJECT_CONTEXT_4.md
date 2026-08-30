# Project Context — SIH 26052 Adaptive Noise Cancellation System

This file is a complete record of every decision, idea, and piece of reasoning
worked out for this project so far. It exists so that anyone opening this
folder — a teammate, a future version of yourself, or a coding assistant like
Antigravity or Claude Code — can understand the full project and every choice
behind it without needing to see the original conversation it came from.

This is a companion to the shorter `AGENTS_3.md` file in this folder.
`AGENTS_3.md` is kept short on purpose because Antigravity has a limited
context-file size. This file is the long, complete version — read this one
first if you want the full picture, and treat `AGENTS_3.md` as the quick
summary a coding tool reads on every run.

**Note on the folder name:** the project folder on the working laptop is
literally `C:\speach data\` — "speach", not "speech". Earlier revisions of
this document wrote it as `C:\speech data\`, which is not a real path on
this machine. Paths in this file use the actual spelling.

**Status of the numbers in this file:** every count, duration, and file
listing in Sections 9, 10, 13 and 17 was measured directly off disk rather
than copied forward from planning. Where the plan and reality differ, the
plan is described as a plan and the reality is given alongside it.

---

## 1. The problem being solved

This project is being built for Smart India Hackathon problem statement
26052, issued by DRDO / IDEX. The goal is an AI/ML-based Adaptive Noise
Cancellation (ANC) system for defence communication. It needs to suppress
three broad kinds of noise:

- **Stationary noise** — steady, unchanging background noise.
- **Non-stationary noise** — noise that changes over time, like engine or
  rotor noise as a vehicle moves.
- **Impulsive noise** — sudden, short bursts, like gunshots, artillery, or
  mortar fire.

At the same time, the system has to keep speech clearly intelligible. The
target numbers set by the problem statement are:

- SNR (signal-to-noise ratio) greater than 15 dB
- STOI (short-time objective intelligibility) greater than 0.85
- PESQ (perceptual evaluation of speech quality) greater than 2.5
- Low enough latency to run in real time on embedded defence hardware

## 2. The overall approach decided on

The project uses supervised deep learning, not reinforcement learning. This
question was raised directly early on — whether RL might suit the problem
better — and the conclusion was that supervised learning is the right fit
here, because the project can generate paired examples directly: a clean
speech clip and a noisy version of the exact same clip. Supervised learning
can use that pairing directly as input and target. Reinforcement learning
needs a reward signal and doesn't have a natural advantage here, so it adds
complexity without a real benefit for this problem.

The model is a CNN encoder–decoder (U-Net style): it takes noisy audio in,
compresses it through convolutional layers, then expands it back out. In the
built version this is a **CRN** — the same encoder–decoder with a recurrent
memory in the middle, so the network can use what it heard seconds ago rather
than judging each moment alone. See Section 4.

## 3. How the audio is represented: STFT vs DWT

Two ways of turning raw audio into something a CNN can process were compared:
STFT (short-time Fourier transform, which turns audio into a spectrogram of
frequency content over time) and DWT (discrete wavelet transform, which
represents audio at multiple resolutions at once). STFT is the more standard
choice for speech enhancement and preserves phase information well when kept
in the complex domain. DWT can be useful for capturing sudden, sharp events
particularly well, which matters for impulsive noise like gunfire.

Both are usable, and combining the strengths of each was discussed as
feasible rather than an either-or choice, since STFT handles the steady
frequency-domain structure of speech well while DWT-style multi-resolution
information can help the network notice sudden impulsive events.

Now that every clip in the project is a fixed 3.000 seconds at 16 kHz
(Section 7), an STFT of any clip has a fixed, known frame count, so the
network can be built with a fixed input size rather than needing to handle
variable-length input.

**Settled in the built model: STFT only, no DWT.** The CRN uses a 512-sample
window (32 ms) with a 160-sample hop (10 ms), giving 301 snapshots of 257
pitch bands per clip. Two constraints fixed those numbers. The hop must divide
the 1600-sample streaming chunk exactly, so chunk boundaries land on frame
boundaries — 160 gives exactly 10 frames per chunk. And the window length is a
direct trade between the two noise families: shorter sharpens gunfire and
mortar onsets, longer separates rotor and engine hum, so 32 ms sits in the
middle. It is exposed as `N_FFT` in the model's tuning block and is the first
thing worth sweeping once a baseline exists.

DWT was not used. It remains a possible addition if impulsive noise proves
hard for the STFT front end, but nothing so far suggests it is needed, and it
would complicate the streaming path.

## 4. Model architecture notes

- Structure: CNN encoder–decoder (U-Net style).
- Two candidate sizes were discussed to test against each other: a smaller
  arrangement (fewer layers, fewer nodes per layer, faster to train, likely
  a good fit for the embedded latency requirement) and a larger arrangement
  (more capacity, potentially better quality, slower and heavier). The plan
  is to test both on a small subset of data first before committing to a
  full training run with either one.
- Loss functions considered: SI-SNR (scale-invariant signal-to-noise ratio,
  a common speech-enhancement loss), plain L1 or L2 distance between clean
  and predicted audio, and perceptual loss (which scores audio more like a
  human listener would rather than just comparing raw waveform values).
- Optimizers shortlisted for direct comparison: Adam, AdamW, and RMSprop.
  The plan is to train on all of them at a small scale first and compare
  results before picking one for full-scale training. Weight decay (used by
  AdamW in particular) is a regularization technique that gently shrinks the
  network's weights during training to help prevent overfitting.
- The auxiliary noise-classification branch (Section 11) now has **7 output
  classes**, not the 6 originally planned. See Section 10 for why.
- The network must be **causal** and must carry hidden state forward
  between chunks, because of the real-time streaming decision in
  Section 5. This constrains which layer types either candidate size can
  use — nothing that needs to see the whole clip at once.

### The CRN is built (step 1 of the plan below)

`scripts/model_crn.py` holds a complete, working CRN. It is untrained — no
training script exists yet — but every structural claim in it has been
verified by a self-test that runs with `python scripts/model_crn.py`.
`CRN_STEPS.md` is the stage-by-stage record.

```
noisy waveform
   |  STFT           301 snapshots of 257 pitch bands per 3 s clip
   v
ENCODER   5 conv layers, pitch 257 -> 129 -> 65 -> 33 -> 17 -> 9    245,472
   |      time never squeezed; 128 patterns x 9 bands per frame
   v
MEMORY    2 unidirectional LSTM layers x 256                      2,266,240
   |      state carries between 100 ms chunks
   +----> AUX HEAD   7 noise categories, removed for deployment      16,903
   |
   v
DECODER   5 layers back up to 257, fed by skip connections          489,457
   |      sigmoid output: a "keep this much" dial per band per frame
   v
mask x noisy spectrogram, noisy phase reused, back to audio
                                                          TOTAL  3,018,072
                                                       DEPLOYED  3,001,169
```

**3.0 million parameters**, close to DCCRN's 3.7 M. What the self-test
establishes, beyond shapes:

- **Lookahead is exactly 2 frames**, verified by changing one input frame and
  confirming which outputs react — not by reading padding numbers, which would
  miss a silent leak. Controls at 0 and 6 frames confirm the check tracks the
  setting rather than always passing.
- **Streaming works exactly.** Feeding a clip whole and feeding it in two
  halves with the LSTM state passed across gives bit-identical results
  (0.00e+00). Starting the second half cold gives a visibly different answer,
  so the state genuinely carries information.
- **The aux head really is removable.** Stripping it changes the audio by
  exactly 0.00, confirming the Section 11 claim rather than assuming it.
- **Measured speed on the RTX 5060:** 13.9 ms per 3-second clip, 216x real
  time; a batch of 16 with forward and backward fits in 1.62 GB of 8 GB.

Three findings from building it are recorded in `CRN_STEPS.md` because they
look like bugs and are not: the STFT framing costs 16 ms of lookahead that
cannot be avoided; LSTM influence fades after roughly 280 ms at random
initialisation rather than persisting indefinitely; and a masked spectrogram
can gain energy after `istft` and re-analysis, because a masked spectrogram is
generally not the spectrogram of any real signal — the same limitation as
reusing the noisy phase, and exactly what DCCRN exists to lift.

### The architecture plan: CRN and TCN first, DCCRN as the upgrade

**DCCRN is not the final choice.** The plan is staged:

1. **Start with CRN and TCN.** These are the two architectures actually
   being built and compared first.
2. **If that works out well, DCCRN becomes the next step** toward the final
   model — an upgrade attempted on top of a working system, not the starting
   point.
3. **The model that ships is DCCRN if we manage to achieve it; otherwise it
   is whichever of CRN and TCN performs better.**

So DCCRN is a stretch target throughout. Everything below about DCCRN —
its published results and what building it would involve — describes that
upgrade step, not the architecture the project is currently working in.

### What DCCRN's published numbers do and don't tell us

The published research behind this architecture used a genuinely small
model, about **3.7 million parameters**, and it ranked **first in the
real-time category** of a well known international noise suppression
competition in 2020. The reported figures:

- **PESQ around 3.5** on one test setup, and **around 3.2** on a tougher,
  more realistic test set — both comfortably above our own PESQ target
  of 2.5.
- A human listener quality score of about **3.4 out of 5**.
- On speed, one frame of audio took only about **3 milliseconds** to process
  on an ordinary laptop CPU, using just **37 milliseconds of lookahead** —
  comparable to the 36 ms our own CRN ended up with (Section 5).

**These should not be read as numbers we expect to match directly.** They
came from training on very large, standard datasets and everyday
environmental noise types — cafe chatter, traffic, and similar. Our own
situation differs on both counts. Our clean speech dataset is far smaller
(1293 training clips, and roughly 125–145 each for validation and testing —
see Section 17), and our noise categories — engine and helicopter hum plus
impulsive gunfire, artillery, and mortar bursts (Section 10) — are a harder
and much less standard category than what those published numbers were
tested on.

The right way to record these figures is as **proof that the architecture is
capable of meeting our targets under favourable conditions**, not as a result
we should expect to reproduce. Our actual results will depend on our own,
much smaller and more specialized dataset, and need to be measured ourselves
once a model exists — and in the first instance that model will be a CRN or
a TCN, not a DCCRN, so these figures are not the yardstick for the first
round of results either.

### Build approach and feasibility of the DCCRN upgrade

Should the project get far enough to attempt it, building DCCRN ourselves is
realistic, not purely theoretical:

- **Working open source implementations already exist** that can be studied
  and adapted, rather than needing to implement complex-valued neural network
  layers from scratch based only on the research paper's math.
- **It does not require unusual hardware.** The original model was small
  enough to run in real time even on a modest CPU, so training something
  similar on our own GPU is well within reach (Section 6).

That said, it is genuinely **more implementation work** than the simpler CRN
or TCN options considered earlier, mainly because complex-valued layers are
less common and can be trickier to debug.

**This is why the order above is what it is, given the hackathon timeline.**
Building a complete, working end-to-end pipeline with the simpler **CRN or
TCN** architecture first guarantees that a finished, submittable system
exists. **DCCRN is attempted afterward if time allows**, rather than being
the only path pursued from the start — and if it is not reached, the better
performing of CRN and TCN is what ships.


## 5. Real-time streaming: chunk size, lookahead, and latency budget

**What changed.** The original plan trained the model on whole 3–5 second
audio clips at once, treating each clip as a single block the network sees
all at once. That does not match how this system actually needs to run once
deployed, since a real-time defence communication system has to process
audio as small packets arrive, not wait several seconds for a full clip to
accumulate.

**The decision.** The model processes audio in **100 millisecond chunks**.
Before finalizing its output it looks a little way into the future, since
seeing near-future audio measurably helps quality — especially at chunk
boundaries, where a network with no view past the edge has nothing to work
from. The **total latency budget for the whole system is 500 to 800
milliseconds**.

**REVISED: the lookahead is 36 ms, not the 200–300 ms first planned.** The
original figure was 2–3 whole chunks. That was far more than the evidence
supports: DCCRN won its real-time category on 37 ms, and the benefit of
lookahead is steeply front-loaded — it exists to help judge whether a sudden
loud thing is a plosive or the start of a gunshot, and that evidence arrives
within a few tens of milliseconds. Spending 200–300 ms bought almost nothing
and cost the budget dearly, because lookahead is **dead time**: waiting for
audio that has not arrived, unrecoverable by any amount of faster hardware.

**What the built model actually costs**, measured rather than estimated
(Section 4, and `CRN_STEPS.md` step 3):

| source | cost |
| --- | --- |
| STFT framing, unavoidable | 16 ms |
| layer lookahead, 2 frames x 10 ms | 20 ms |
| **lookahead total** | **36 ms** |
| chunk wait, a full chunk must arrive | 100 ms |
| inference, measured on the RTX 5060 | 0.5 ms |
| **end to end** | **≈136 ms** |

That leaves the 500–800 ms budget almost entirely unspent. It also lands
inside ITU-T G.114's preferred range for two-way conversation, under 150 ms
one-way — worth knowing, because 500–800 ms would have been past the point
where people start talking over each other. The ANC is only one element in a
chain that also has radio, codec and network delay.

**Why not a larger chunk.** Growing the chunk to 200–250 ms would not loosen
the budget, it would *tighten* it: the mandatory chunk-wait scales directly
with chunk size. There is no reason to increase it, and shrinking it would
only add margin — dropping to a 20 ms chunk would put the total near 56 ms,
matching the competition standard, at the cost of five times more frequent
inference calls. That is a tuning step once a model works, not a starting
constraint.

The one thing that could still force a change is the model's **measured
inference time on the real target embedded hardware**. There is now a first
data point — 0.46 ms of compute per 100 ms of audio, 216x faster than real
time — but on the RTX 5060, **not** the Jetson AGX Orin the problem statement
names. Compute is nowhere near the constraint; waiting is.

**This requires an architecture decision, not just a data-chunking
decision.** Two requirements follow for the network itself:

- It must be built as a **causal** network, meaning it only ever looks
  backward in time within its processing, never forward, since it cannot
  access audio that has not arrived yet.
- It should **carry some memory or hidden state forward** from one chunk to
  the next, so that it is not starting blind every 100 milliseconds.

Carrying memory forward is distinct from the lookahead buffering described
above, and the difference is worth being precise about: **carrying memory
forward is free in terms of latency**, since it only uses the past, while
**lookahead buffering deliberately costs latency**, since it waits on the
future. Real systems typically use both together.

## 6. Hardware and training time

Training happens on a laptop with an Intel Core Ultra processor and an RTX
5060 GPU with 8GB of VRAM. This is enough to train a small-to-medium CNN of
the kind planned here in a reasonable number of hours rather than days,
though the exact time depends on the final architecture size chosen and how
much data is used. The agreed approach is to first test a smaller version of
the model on a smaller slice of data to get a feel for real training time and
GPU behavior, before committing to a longer full run. Running extended
training sessions on this hardware is generally fine as long as normal
laptop precautions are followed — decent airflow/cooling and not running it
unattended for excessively long stretches without checking on it.

**GPU is working (resolved).** The environment in `.venv/` originally had
the CPU-only build of PyTorch, so the GPU was unusable. It has been replaced
with the CUDA build and verified end to end:

- `torch 2.13.0+cu130`, `torchaudio 2.11.0+cu130`, CUDA 13.0, cuDNN 9.2.0.
- Device reports as **NVIDIA GeForce RTX 5060 Laptop GPU**, compute
  capability **sm_120**, 7.96 GB usable VRAM. Driver is 610.74 (CUDA 13.3).
- Verified with real kernel execution, not just `cuda.is_available()`:
  a 4096³ fp32 matmul (~4.9 TFLOP/s), a `torchaudio` resample on device,
  a Conv1d forward and backward pass, and a complex-valued STFT.

**Measured, now that a model exists.** The CRN runs a 3-second clip in
13.9 ms, 216x faster than real time, using 0.46 ms of compute per 100 ms of
audio. A training batch of 16 with forward and backward passes peaks at
1.62 GB of the 8 GB available, so there is room for a much larger batch or a
wider model. Compute is nowhere near the constraint on this machine.

**Measured training speed, and what it changes.** One epoch over the 1292
training clips plus the 143 frozen validation mixtures takes **11.3 seconds**,
and a full perceptual evaluation adds about 12 seconds. So 40 epochs is under
9 minutes and 150 epochs is roughly half an hour. Training this model is not
an overnight job, and several planning assumptions were built around the idea
that it would be. Profiled, an epoch is 9.2 s of GPU work against 0.7 s of
data loading, so it is firmly **GPU-bound**.

Four consequences, all worth acting on:

- **`EPOCHS = 40` is too conservative.** With early stopping at 12 epochs of
  patience, set it to 150 and let the model halt itself when it plateaus.
  Guessing the right number in advance is unnecessary when the guard exists.
- **Run many experiments rather than few.** CRN against TCN, hidden 256
  against 128, aux weight 0.2 against 0.0, and the three shortlisted
  optimisers is roughly a dozen runs — an afternoon, not a week. This is what
  makes the comparison dashboard worth building rather than a nicety.
- **Raise the batch size.** A batch of 16 uses 1.62 GB of 8 GB while the GPU
  is the bottleneck, so 32 or 48 would use the card properly and should cut
  epoch time noticeably.
- **Do not add DataLoader workers.** Data loading is 6% of an epoch;
  parallelising it would buy essentially nothing. This was worth measuring
  rather than assuming — the first guess was that mixing audio on the fly
  would dominate, and it does not.

`matplotlib` was added to the environment for `visualize_pipeline.py`, and
`pesq`, `pystoi` and `scipy` for `evaluate.py`. None are in the install
command below; add them when rebuilding.

The RTX 5060 is **Blackwell**, compute capability `sm_120`. This matters when
reinstalling: only CUDA 12.8 and newer builds contain `sm_120` kernels, so
the older `cu126` wheels will install happily and then fail at the first
real operation with "no kernel image is available for execution on the
device". Python in the venv is 3.14.6 (`cp314`), and the matching wheels
come from the `cu130` index:

```
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0+cu130 torchaudio==2.11.0+cu130
```

**Windows gotcha worth recording.** PyTorch ships unsigned DLLs, and
**Smart App Control** on this laptop blocked them outright — `OSError
[WinError 4551]`, with Code Integrity event 3077 naming `torch\lib\torch.dll`
as failing the signing requirement. The CPU build had been allowed on cloud
reputation; the far rarer CUDA wheels had none. Smart App Control had to be
turned off (Windows Security → App & browser control) and the machine
rebooted before the libraries would load. Note that **Smart App Control
cannot be re-enabled without resetting Windows** — it is a one-way switch.
Anyone rebuilding this environment on a different Windows machine should
expect the same block.

## 7. Training data strategy

### Clip length — normalization to a fixed 3 seconds (DONE)

Clean speech clips were originally settled at roughly **3 to 5 seconds**
each (this evolved from an earlier, shorter 2–4 second idea during
planning), and that is the length the collection and rechunking scripts
were written around.

On top of that, **both speech and noise have now been normalized to a fixed
maximum of 3 seconds at 16 kHz mono**, and this step has been carried out —
it is no longer a pending manual task. It is implemented in
`scripts/normalize_clip_lengths.py`. What it actually did:

**Speech** — any clip longer than 3 seconds keeps its **first 3 seconds**
and the tail is discarded. Clips already at or under 3 seconds are left
exactly as they are, so speech clip lengths now range from 0.64 s up to
3.000 s. No speech file was added or removed, so the existing per-language
numbering is unchanged. Every speech clip was also resampled to 16 kHz —
see "Sample rate" below.

**Noise** — each category within each split is treated as **one continuous
stream**. Files are appended in their numbered order until 3 seconds of
audio are available; that 3-second clip is written out, and whatever is left
over carries forward into the next clip — but only if the leftover is
**longer than 1 second**. A leftover of 1 second or less is discarded and
the next source file starts a fresh clip. Whatever remains at the end of a
category, necessarily shorter than 3 seconds, is discarded too. Every
resulting noise clip is exactly 3.000 seconds.

Two hard rules govern this stream, and both were enforced and verified:

1. A stream **never crosses a category boundary** — no clip ever contains
   two different noise types.
2. A stream **never crosses a split boundary** — training audio cannot leak
   into validation or testing.

Output files are renumbered from `001` with no gaps, keeping the existing
`<category>_NNN.wav` convention.

**Known divergence from the original intent.** Earlier revisions of this
document and of `AGENTS.md` specified that noise concatenation should keep
"their natural gaps between events, e.g. real spacing between individual
gunshots, rather than butting them together with no gap at all." The
implemented step does **not** insert any gap at the seams — source
recordings are joined end to end. Silence that already exists *inside* an
individual recording is preserved, but no spacing is added *between* two
recordings. This was reviewed and **accepted as-is** — see Section 12. It is
not an open item.

### Sample rate

Everything in the project is now **16 kHz, mono, 16-bit PCM WAV**. This was
not true before normalization: English speech (from LibriSpeech) and all
noise were already 16 kHz, but all 22 Indic languages (from IndicVoices)
were 48 kHz. The 1216 Indic clips were resampled down to 16 kHz during the
normalization step, using `torchaudio.functional.resample`. Having a single
sample rate across speech and noise is a prerequisite for mixing the two
together at all, so this was done at the same time rather than deferred to
the mixing script.

### Combining clean speech with noise (BUILT)

Noisy examples are not stored. They are generated on the fly by mixing a
random clean speech clip with a random noise clip at a randomized SNR —
many noise clips against many speech clips produces a very large number of
distinct training examples without storing any of them. Training mixtures
are regenerated fresh each run; only a fixed, saved set of validation and
testing mixtures is kept permanently, so different model versions can be
compared fairly against the exact same evaluation audio (see Section 11 for
why this matters, and Section 13 for where this lives on disk).

This is implemented in `scripts/mixing.py`, which exposes a PyTorch
`Dataset` yielding `(noisy, clean, noise_category_label)`. The rules it
follows were each chosen deliberately:

- **SNR is drawn uniformly from −5 to +15 dB.** At the low end the noise is
  louder than the speech, which is what a gunshot or rotor wash over a
  talker actually sounds like. Since the output target is SNR > 15 dB, this
  range forces real improvement rather than light polishing.
- **Noise categories are sampled uniformly**, one in seven each, *not* in
  proportion to the clips on disk. The on-disk counts are lopsided
  (assault_rifle 137 against mortar 45) and proportional sampling would bias
  both the auxiliary classifier and the denoiser toward the common
  categories. Scarce categories simply get reused more often, which is
  harmless because each reuse pairs with a different speech clip and SNR.
- **Speech shorter than the 3 second window is zero-padded and placed at a
  random offset**, so the network cannot learn that speech always starts at
  t=0. Noise fills the whole window regardless, so padded regions are
  noise-only — realistic, since real transmissions contain pauses.
- **Clips shorter than 1 second are skipped.** This costs almost nothing:
  one clip in train, one in validation, none in testing.
- **Speech power is measured over the real speech samples only**, never over
  the zero padding, so a heavily padded clip is not handed a quieter noise
  floor than a full-length one.
- **Mixtures that would clip are rescaled with clean and noisy together**,
  so the pair remains a valid input/target rather than drifting apart.
- **A mixture never crosses a split.** Validation speech is only ever mixed
  with validation noise.

**How SNR is defined here, and why it matters for the impulsive
categories.** The noise level is set from noise power measured across the
whole 3 second window, against speech power measured over the speech-active
region. Window-wide is the right basis for the noise: measuring it only
inside the speech span would mean that a near-silent span there demands a
huge scale factor, making whatever burst sits outside it deafening. The
consequence is that for `assault_rifle`, `heavy_artillery`, `mortar` and
`rocket`, the SNR *local to the speech* varies by several dB even at a fixed
requested SNR, because burst energy is not spread evenly through the clip.
That is physically correct — impulsive noise is not stationary — but it
means per-category PESQ and STOI will be noisier for those four than for
`ambient`, `engine` and `helicopter`. Anything verifying the mixer must use
the same definition, or it will read several dB off.

### The frozen evaluation mixtures (BUILT)

`scripts/generate_test_mixtures.py` drives the same mixing code with a fixed
seed and writes the validation and testing mixtures to disk once, so that
every model version is scored on byte-identical audio. Regenerating with the
same seed reproduces the set exactly.

Both halves of each pair are written out, not just the noisy side. The clean
target is not a copy of the original file — short clips are padded and
randomly placed, and the pair is rescaled together when the sum would clip —
so the clean signal has to be stored as it actually was.

The run produced **268 pairs (536 WAV files, 13.4 minutes)**: 143 validation
and 125 testing. `test_mixtures_manifest.csv` records the speech source,
noise source, category, label, requested and measured SNR, and the speech
placement for every mixture. Verified after writing: every file is exactly
48000 samples at 16 kHz, requested and measured SNR agree to 0.0000 dB
across all 268, no split leaks, and validation/testing speakers stay
disjoint.

### Beyond simple additive noise — voice distortion (DEFERRED, and now a weaker bet)

**Read Section 18 before acting on this.** The data-scaling experiment
showed that quartering the real training data changes nothing, which
undercuts the premise that more diversity is what the model lacks.
Augmentation is still worth doing because the problem statement names it,
but it should no longer be expected to lift the scores.


Realistic distortion beyond just "clean speech plus noise" is worth
including, since real transmitted audio is degraded in more ways than that:
clipping, reverberation (convolving with a room impulse response),
band-limiting (cutting off frequency range, as a radio channel would),
dropout or packet loss (missing chunks of audio), and codec/vocoder-style
distortion that makes speech sound "robotic." The problem statement names
reverberation and clipping explicitly as expected augmentation, so this is a
scored gap rather than only a nice-to-have. Random noise mixing, the third
augmentation it names, is already covered by the mixer itself.

**This is deliberately deferred to a later update.** The decision and the
reasoning behind it:

**Mode A is the chosen form.** The distortion is applied to the speech
*before* it becomes both the target and the input:

```
speech -> distort -> [ target ]
                  -> [ + noise -> input ]
```

The network then learns "remove noise from distorted speech" and keeps the
distortion, rather than trying to undo it. This matches deployment — a radio
operator's reference audio genuinely is band-limited and codec-mangled, and
the system's job is to make that intelligible, not to reconstruct studio
audio that never existed.

The rejected alternative, **Mode B**, distorts only the input and leaves the
target pristine, forcing the network to denoise *and* de-distort. That is
restoration rather than enhancement, a strictly harder problem, and with
only 1292 training clips it is a realistic way to end up below all three
targets at once by half-learning two tasks instead of learning one well.

**Why deferring is safe, and why it is actively better.** Mode A changes
nothing structural: same shape, same sample rate, same sample-for-sample
alignment between input and target, same `(noisy, clean, label)` interface.
The model, the loss, the training loop and the frozen evaluation set are all
untouched, so it drops in later as a single function called before
`mix_at_snr`. Mode B would not have this property — it redefines the task,
so adding it later would invalidate every result measured before it.

Beyond being safe, deferring is the better experiment. Augmentation fixes
overfitting, and whether this project has any overfitting is not yet known.
Adding it before the first baseline exists means that if the first CRN trains
badly, there is no way to tell whether the model is wrong or the augmentation
made the task too hard. Train clean first, get a baseline worth trusting,
then add augmentation and measure whether it helped.

**The trigger for picking this up:** when validation loss stops tracking
training loss — training keeps improving while validation flattens or turns
upward. That is overfitting confirmed, and it is exactly what Mode A
augmentation addresses. Given the dataset size this may well appear early,
so it should not be assumed to stay deferred for long.

**Two rules for when it is implemented.** Nothing that alters time alignment
may be used — no time-stretching, no `speed`, no pitch shifting by resample,
no variable delay. SI-SNR and L1 losses compare input and target sample by
sample, so a desynchronized pair does not raise an error, it silently trains
the network against a meaningless objective. And augmentation must never be
applied to 100% of examples; roughly 40% is the intended rate, so the network
still sees undistorted conditions to anchor against.

**Effort, and what is already available.** `torchaudio` already provides
everything for the cheap effects — `bandpass_biquad` for band-limiting,
`mu_law_encoding`/`mu_law_decoding` for the "robotic" quantized flavour,
plain `numpy` clipping, and simple masking for dropout. That tier is roughly
an hour of work. Reverberation is the one moderate item: `fftconvolve`
exists, but neither `scipy` nor `pyroomacoustics` is installed, so it needs
either a synthetic exponential-decay impulse response or a downloaded RIR set
such as OpenSLR's RIRS_NOISES. A real codec (AMR, Opus, GSM) is the hard case
and is not planned — `torchaudio.functional.apply_codec` has been removed, so
it would require driving ffmpeg as a subprocess, and mu-law approximates the
audible character well enough.

**When it lands, record which runs used it.** An augmented run must never be
compared against a clean one and the difference read as an architecture
result.

### Held-out generalization testing

Part of the testing strategy is to hold out entire noise categories (not
just individual clips) from training, so the model's ability to generalize
to noise types it has never seen can be checked specifically, not just its
ability to handle noise types it was trained on.

**This is not currently set up.** All 7 noise categories are present in all
three splits (that is in fact a requirement now — see Section 10), so no
category is held out from training. Doing this properly needs either an
eighth category collected specifically to be held out, or a deliberate
decision to sacrifice one of the current seven from the training split.

## 8. Datasets used or considered

**Clean speech:**

- `ai4bharat/indicvoices_r` on Hugging Face — the main source for Hindi and
  the other 21 scheduled Indian languages. This is a **gated** dataset
  (requires accepting access conditions and running `huggingface-cli login`)
  and does not include English. Delivers audio at 48 kHz.
- `openslr/librispeech_asr` on Hugging Face — public, no login required,
  used specifically for English. Confirmed working split name is
  `"train.clean.100"`. Delivers audio at 16 kHz.
- Other speech datasets considered but not currently in use: VCTK, TIMIT,
  Common Voice.

**Noise:**

- Built manually by classifying real recordings into named categories
  (see Section 10). This is now complete for the seven categories in use.
- Datasets considered as sources for this: UrbanSound8K, ESC-50, AudioSet,
  the Military Audio Dataset (published via Nature), and a Multi-Firearm
  Gunshot Dataset.

## 9. Clean speech dataset construction

### The collection rules

Three separate sets are built: `train`, `validation`, `testing`. Each set
targets the same proportions: 25% English, 25% Hindi, and the remaining 50%
spread as evenly as possible across the other 21 scheduled Indian languages
(Assamese, Bengali, Bodo, Dogri, Gujarati, Kannada, Kashmiri, Konkani,
Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit,
Santali, Sindhi, Tamil, Telugu, Urdu).

The original targets were 410 / 50 / 50 clips. Actual collection plus the
later rechunking of long clips overshot those numbers considerably; the
real counts are in Section 17.

Three hard rules are enforced while assigning clips to a set:

1. No audio file is ever placed in more than one split.
2. No speaker appears in more than one split — the first time a speaker is
   seen, they get locked to whichever split picked them up. **Verified: zero
   speaker overlap between any pair of splits.**
3. Inside a single split, the same speaker can repeat, but capped at a
   configurable limit (currently 3 in the main collection scripts).

Each split has its own manifest CSV (`train_manifest.csv`,
`validation_manifest.csv`, `testing_manifest.csv`) listing filename,
language, speaker id, and duration for every clip.

### Rebalancing to the target percentages

Rechunking long clips into 3-second pieces inflated the languages that
happened to have long recordings — English most of all — so the collected
data no longer matched the 25 / 25 / 50 target. Two rules were settled to
deal with this:

**Surplus is ignored, not used.** The target percentages are the authority.
Where a language has more clips on disk than its share allows, the extra
clips are simply not selected. They are *not* deleted — they stay on disk
and in the original manifests.

**The 50% share is water-filled, not split flat.** Within the 50% belonging
to the other 21 languages, a language that has very few clips contributes
everything it has and the shortfall is absorbed by languages that still have
spare clips. Per-language counts therefore remain uneven. The stricter
alternative — capping every one of the 21 at whatever the scarcest language
has — was considered and rejected, because in the validation and testing
splits some languages have only a single clip, which would have collapsed
those splits to 41 clips each, far too small to score PESQ and STOI
meaningfully.

This selection is produced by `scripts/build_balanced_manifests.py`, which
writes a parallel set of files:

    dataset/balanced_train_manifest.csv
    dataset/balanced_validation_manifest.csv
    dataset/balanced_testing_manifest.csv

These are the manifests training should read. The original
`*_manifest.csv` files are left untouched as the record of everything that
was actually collected.

When a language is trimmed, clips are chosen **round-robin across that
language's speakers** rather than in file order, so trimming does not
silently delete whole speakers off the end of a language and shrink speaker
diversity.

## 10. Noise dataset construction and classification

Noise recordings were manually sorted into named category folders. The
folder name itself acts as the classification label — any script reading the
data later reads the category straight from the folder path.

### The seven categories actually in use

The originally planned six were *ambient, engine, helicopter, assault rifle,
artillery, mortar*. What exists on disk differs:

| category | note |
| --- | --- |
| `ambient` | as planned |
| `engine` | as planned |
| `helicopter` | as planned |
| `assault_rifle` | as planned (see the renaming note below) |
| `heavy_artillery` | the planned "artillery" category, under its fuller name |
| `mortar` | as planned |
| `rocket` | **new** — not in the original plan |

Two corrections were applied to the folders:

- The folder named `dog_bark` was a **mislabel**. Its contents are assault
  rifle gunfire, and it has been renamed to `assault_rifle`. The `.wav`
  files inside were renamed to match, since the naming convention here is
  `<category>_NNN.wav` inside `<category>/` and leaving `dog_bark_001.wav`
  sitting in `assault_rifle/` would have preserved exactly the mistake being
  corrected.
- Misspelled and inconsistently-cased folder names were corrected:
  `Helecopter` → `helicopter`, `Heavy_artelary` → `heavy_artillery`,
  `Ambient` → `ambient`, `Mortar` → `mortar`, `Rocket` → `rocket`. All
  category folder names are now lowercase `snake_case`, which matters
  because these strings become class labels.

### The all-splits rule

**A category counts only if it is present in all three splits.** A category
that appears in some splits but not others is dropped from consideration
entirely rather than being partially used or backfilled — a class that
cannot be evaluated on validation and test data is not a usable class. This
is what removed a `siren` category that existed only in the training split
with a single file.

### Folder layout

The layout is **split first, category second**, matching how the clean
speech dataset is organized. Note that the actual location differs from what
Section 13 of earlier revisions planned: noise lives **inside** `dataset/`
as `dataset/noise_data/`, and the split folders are named `noise_train`,
`noise_validation`, `noise_testing` rather than plain `train`/`validation`/
`testing`. The full current layout is in Section 13.

### The noise manifest

`dataset/noise_manifest.csv` is generated by
`scripts/build_balanced_manifests.py` and lists every usable noise clip with
columns `filename, category, split, duration_seconds`. It was described as
optional in earlier revisions; in practice it is now the convenient way to
enumerate the noise set and its class labels, though the folder structure
remains the authoritative label.

## 11. The noise-aware auxiliary training idea

This is one of the more significant architectural ideas developed in this
project, so it's documented in full detail here.

**The question that started it:** since the training mixtures are built by
combining known clean speech with known noise, could the noise itself be fed
to the network to help it produce cleaner output?

**Why the direct version of this is wrong:** feeding the isolated noise
waveform to the network as an extra input would make the training task
artificially easy (the network could just learn to subtract), but at real
deployment time the system will only ever have the single noisy microphone
signal — it will never have a separate, isolated copy of just the noise. A
model trained to expect that input would fail once deployed. So noise must
never be fed in as an input.

**The valid version — multi-task auxiliary training:** instead of feeding
noise in as an input, the *known noise category* is used only as an extra
target the network is scored against during training, never as something it
consumes. Concretely: the shared trunk of the CNN (the encoder layers) feeds
into two separate output branches. The main branch produces the clean audio,
exactly as before. A second, small branch tries to guess which of the **seven
noise categories** (ambient, engine, helicopter, assault_rifle,
heavy_artillery, mortar, rocket) is present in the mixture it just saw.
Since the noise category is already known — it was chosen when the mixture
was built — this comparison can be made automatically as an ordinary
classification loss (predicted category vs known true category). Both
branches' errors flow backward through the shared trunk, so the
classification branch's error reshapes the trunk's learned features even
though it never touches the final clean-audio output directly. This is
expected to help the main task, because it pushes the shared features to get
better at telling different noise types apart, which is useful groundwork
for suppressing them.

**Class imbalance to account for.** The seven classes are not evenly sized —
`assault_rifle` has 137 clips against `mortar`'s 45 (full table in Section 17). The classification loss will need class weighting or balanced sampling,
otherwise the branch can score well by favouring the common classes.

**It works.** Across five training runs the branch reaches 0.52 to 0.59
accuracy against a chance level of 0.143, about four times chance. The idea
below is sound and the branch learns what it was meant to. Whether it
*helps* the audio task is separate and still untested; the experiment is a
run with `AUX_LOSS_WEIGHT = 0.0` (Section 18).

**Removing it for deployment (verified).** In the built model this branch is
16,903 parameters, 0.6% of the total, and stripping it changes the enhanced
audio by exactly 0.00 — checked in the self-test rather than assumed. The
deployed model is 3,001,169 parameters instead of 3,018,072.

This classification branch is only a training aid. It is not needed to produce the final clean-audio output, so
it gets physically removed before deployment, leaving only the shared trunk
and the main clean-audio branch. This makes the deployed model smaller and
faster — directly useful for the embedded, low-latency requirement — without
losing any of the benefit, since that benefit is already baked into the
shared trunk's weights by the time training finishes.

**What a trained model actually looks like on disk, and how to trim it:**
a trained CNN is really two separate things — the architecture, which is
ordinary Python code describing the layers and how they connect (e.g.
`model.py`), and the weights, which are the actual learned numbers, saved
separately as a checkpoint file (for PyTorch, typically a single `.pt` or
`.pth` file). To remove the noise-classification branch: (1) make a copy of
the model code with that branch's layer definitions deleted, keeping only
the shared trunk and the main clean-audio branch, and (2) write a small
script that loads the full trained checkpoint and copies over only the
weights whose layer names still exist in the trimmed architecture into a new,
smaller checkpoint file, dropping the rest. The trimmed code plus the
trimmed checkpoint together are the deployment version.

## 12. Handling noise clips shorter than the training window

Noise recordings originally varied wildly in length — some long and
continuous (ambient, engine, helicopter), while gunfire-type noises
(assault_rifle, heavy_artillery, mortar, rocket) were short, often well
under a second. This section describes that original problem and how it was
resolved. **It has been superseded in practice by the normalization step
described in Section 7**, which was carried out and produced 636 noise clips
of exactly 3.000 seconds each. The reasoning is kept here as background.

**Continuous noise types** are easy to handle: any random crop of matching
length taken from a longer recording is still representative, since the
sound is similar throughout.

**Short, bursty noise types** needed more care. A single brief burst dropped
into an otherwise silent multi-second window means most of that window
contains no audible noise at all, even though the whole window carries a
single category label. The original planned handling was to randomize where
the burst sits within the window, or place more than one burst in a window
where realistic, so a meaningful portion of the window actually contains the
sound. This isn't purely a workaround — real deployment audio also has long
quiet stretches interrupted by sudden bangs, so training on this pattern is
realistic, not just a technical fix. In practice, concatenating multiple
bursts up to the fixed 3 second length achieves the same goal more directly,
by construction, rather than needing randomized placement logic in the
mixing script.

**How the classification "checking" actually works:** the approach used is
a whole-clip (weak) label — the entire clip gets one single category label
(e.g. `assault_rifle`). Now that clips are normalized to a fixed 3 second
length and densely filled with their category's sound for nearly the whole
duration, this whole-clip label is an accurate description of the entire
clip, not a rough approximation covering a mostly-silent window. The
classification branch produces one guess for the whole clip, compared
against that one true label with an ordinary classification loss. A more
precise, time-sliced alternative was also considered (labeling only the
exact narrow span containing a sound rather than the whole clip), but is no
longer necessary now that normalization keeps each clip densely
representative throughout — it remains documented here only for reference,
not a planned next step. No separate detection step is needed to create this
feedback either way — it's simply recorded automatically by the mixing
script at the moment each mixture is built, the same bookkeeping idea as the
existing manifests.

**Clicks at the seams — considered and dropped.** Joining two separate
recordings at a hard cut can produce an audible click or pop at the seam,
since the waveform jumps abruptly between two different recordings, and the
normalization step does join them with no crossfade and no inserted gap.
This was raised, weighed, and **deliberately set aside as not worth
addressing** — the noise is background material being actively suppressed
rather than reproduced, so a brief transient at a join does not threaten the
objective. No crossfade pass is planned. It is recorded here only so that
nobody re-derives it later and mistakes it for an outstanding task.

## 13. Full project folder layout

Everything lives inside one main project folder, `C:\speach data\` on the
working laptop. This is the **actual** layout as it exists now, not a plan.

```
speach data/
├── .venv/                          Python 3.14.6 virtualenv
│                                    torch 2.13.0+cu130 (GPU working)
├── AGENTS_3.md                     short context file for coding tools
├── AGENTS_1.md                     older revision, superseded by AGENTS_3
├── PROJECT_CONTEXT_4.md            this file
│
├── dataset/
│   ├── train/                      1440 clips, one subfolder per language
│   ├── validation/                 172 clips, same structure
│   ├── testing/                    163 clips, same structure
│   ├── train_manifest.csv          everything collected
│   ├── validation_manifest.csv       columns: filename, language,
│   ├── testing_manifest.csv          speaker_id, duration_seconds
│   │
│   ├── balanced_train_manifest.csv       the 25/25/50 selection —
│   ├── balanced_validation_manifest.csv  THESE are what training reads
│   ├── balanced_testing_manifest.csv
│   │
│   ├── noise_manifest.csv          columns: filename, category, split,
│   │                                duration_seconds
│   └── noise_data/
│       ├── noise_train/
│       │   ├── ambient/
│       │   ├── assault_rifle/
│       │   ├── engine/
│       │   ├── heavy_artillery/
│       │   ├── helicopter/
│       │   ├── mortar/
│       │   └── rocket/
│       ├── noise_validation/       same seven category folders inside
│       └── noise_testing/          same seven category folders inside
│
├── scripts/                        all Python scripts
├── models/                         (empty) saved checkpoints — full
│                                    training version and the trimmed
│                                    lightweight deployment version
├── logs/                           (empty) training run records
└── generated_test_mixtures/        268 frozen mixtures for consistent
    ├── validation/noisy/            scoring across model versions.
    ├── validation/clean/            Training mixtures are NOT stored —
    ├── testing/noisy/               they are generated fresh each run.
    ├── testing/clean/
    └── test_mixtures_manifest.csv   columns: split, index, noisy_file,
                                      clean_file, speech_file, language,
                                      speaker_id, noise_file, category,
                                      label, snr_db, measured_snr_db,
                                      speech_start_sample, speech_end_sample
```

Two differences from what earlier revisions of this document planned:

- Noise sits at `dataset/noise_data/noise_<split>/<category>/`, not at a
  top-level `noise/<split>/<category>/`. Moving it would be a purely
  cosmetic restructure at this point; every script already targets the
  current location.
- Scripts moved from the project root into `scripts/`. Because of this,
  each script now resolves its paths from a `PROJECT_ROOT` derived from
  `__file__` instead of from the current working directory, so they can be
  run from anywhere.

## 14. Scripts built so far

All live in `scripts/`. All resolve `dataset/` relative to the project root
rather than the working directory, so they can be run from any location.

- **`split_speech_dataset.py`** — the main clean-speech collection script.
  Builds the train/validation/testing split described in Section 9, pulling
  Hindi and the other 21 Indian languages from `ai4bharat/indicvoices_r` and
  English from `openslr/librispeech_asr`. Fully resumable: it counts what's
  already on disk and reads existing manifests before continuing, so it can
  be safely re-run after an interruption without duplicating or losing
  progress. Writes to the manifest CSVs incrementally so progress survives a
  crash.
- **`collect_english.py`** — a standalone companion that pulls only the
  English share, writing into the same shared manifests as the main script.
  Same resumable, rule-enforcing behavior. Note: the per-speaker repeat cap
  in this script is currently left at 3 (matching the main script) because
  an earlier instruction about changing it to a different number was
  ambiguous and never confirmed — worth double-checking this value.
- **`rechunk_long_clips.py`** — a post-processing script that splits any
  clip longer than 5 seconds into 3-second chunks (keeping a final leftover
  chunk only if it's at least 1 second, otherwise dropping it), and
  renumbers all clips per language/split sequentially with no gaps. This
  version also rewrites the manifest CSVs to match the new chunked files.
  Tested against synthetic data before delivery. **Largely superseded** by
  `normalize_clip_lengths.py`, which enforces the stricter 3-second ceiling.
- **`rechunk_long_clips_no_manifest.py`** — identical splitting and
  renumbering behavior to the script above, but does not read or write any
  manifest CSV at all — it only changes the `.wav` files on disk. Also
  largely superseded.
- **`normalize_clip_lengths.py`** — the clip-length normalizer described in
  Section 7. Trims over-length speech to its first 3 seconds, resamples
  everything to 16 kHz, and re-cuts each noise category+split stream into
  exact 3-second clips with contiguous renumbering. Runs as a **dry run by
  default**, reporting what would change without writing; pass `--apply` to
  commit. Writes noise output to a staging folder and swaps it in only after
  every clip is written, so an interrupted run cannot leave a category
  half-converted.
- **`build_balanced_manifests.py`** — applies the two selection rules from
  Sections 9 and 10. Writes the three `balanced_*_manifest.csv` files and
  `noise_manifest.csv`. Purely additive: it never deletes or renames audio.
  Re-run it after anything changes the data on disk.
- **`mixing.py`** — the mixing pipeline described in Section 7. Exposes a
  PyTorch `Dataset`, `SpeechNoiseMixtures(split, seed=None)`, yielding
  `(noisy, clean, label)`. `seed=None` gives fresh randomness every run,
  which is what training wants; an integer makes every index reproducible
  regardless of how the DataLoader distributes work. Run the file directly
  for a self-test across all three splits.
- **`generate_test_mixtures.py`** — drives `mixing.py` with a fixed seed to
  write the frozen validation and testing mixtures. Dry run by default;
  pass `--apply` to write. Already run — the 268 pairs are on disk.

- **`add_noise_class.py`** — installs a new noise category dropped into
  `new_noise_class/` as `<class>/train|validation|testing/`. Cuts the clips to
  exact 3.000 s, renumbers them, appends the label, and rebuilds the
  manifests. Dry run by default. Refuses a class missing from any split, which
  is how an earlier `siren` category disappeared unnoticed.
- **`model_crn.py`** — the CRN itself (Section 4). Run it directly to execute
  the self-test for all six build stages; it doubles as a regression test for
  the whole model. All tunable values sit in a marked block at the top,
  separated from constants that are locked by data already on disk.
- **`visualize_pipeline.py`** — draws what the pipeline is doing on real
  project audio: clean speech, the noise added, the mixture, the spectrogram
  the network sees, the reconstruction, and the error between them. `--zoom
  START END` instead shows clean, noise and mixture spectrograms stacked over
  a short window, which is where the horizontal-versus-vertical contrast that
  makes spectrograms worth using becomes visible. Figures go to
  `logs/figures/` and are gitignored, being reproducible in seconds.

An early draft named `collect_speech_data.py` is referenced in older notes;
it is **not present in this project folder**.

## 15. Open questions / things still to confirm

**Planned, deliberately deferred:**

- **Voice distortion augmentation (Mode A)** — decided on, scoped, and held
  back until a clean baseline exists (Section 7). The trigger for picking it
  up is validation loss ceasing to track training loss. The problem statement
  names reverberation and clipping as expected augmentation, so leaving this
  undone permanently is a scored gap, not merely a missed improvement.

**Data quality items:**

- `heavy_artillery` has only 4 validation clips and `mortar` only 6. That is
  very thin for evaluating those classes on the auxiliary branch — with 4
  clips, one misclassification moves that class's accuracy by 25 points.
- Per-category PESQ and STOI will be noisier for the four impulsive classes
  than for the steady ones, because the SNR local to the speech varies even
  at a fixed requested SNR (Section 7). Worth remembering when reading
  per-category evaluation numbers rather than treating it as a regression.
- Held-out-category generalization testing is not set up, since all seven
  categories appear in all three splits (Section 7).
- The model's inference time on the real target embedded hardware is still
  unknown. There is now a first data point on the RTX 5060 — 0.46 ms of
  compute per 100 ms of audio, leaving end-to-end latency around 136 ms — but
  the problem statement names a Jetson AGX Orin, which has not been tested.
  Compute is far from the constraint here, so this is unlikely to force the
  chunk size or lookahead to change (Section 5).

**Carried over, still unresolved:**

- The exact per-speaker repeat cap intended for `collect_english.py` — an
  earlier mention of a different number was never clearly confirmed, so it's
  still set to 3 by default.
- `ai4bharat/indicvoices_r`'s exact field names for audio and speaker id
  were guessed (`"audio"` and `"speaker_id"`) since the dataset's schema
  wasn't visible without logging in. Both collection scripts print the real
  field names from the first pulled example of each language so this can be
  verified and corrected if needed.
- The final choice of optimizer, loss function, and model size (smaller vs
  larger arrangement) is still pending the planned small-scale comparison
  runs described in Section 4.

**Resolved since earlier revisions:**

- The mixing pipeline exists and the frozen evaluation set is on disk
  (Section 7). This was the last thing standing between the data and a
  trainable model.
- Clicks at concatenation seams — considered and deliberately dropped, not
  worth addressing (Section 12).
- Speech clips under 3 seconds — handled by the mixer, which pads them to
  the window at a random offset and skips anything under 1 second
  (Section 7). Only two clips in the whole project fall below that floor.
- Noise class imbalance — handled by sampling categories uniformly rather
  than in proportion to clip counts (Section 7), so no class weighting is
  needed in the loss.
- Which noise categories the project uses — settled at the seven in
  Section 10, collected manually rather than pulled from one of the candidate
  datasets in Section 8.
- The GPU is usable — the CUDA build of PyTorch is installed and verified
  (Section 6). This was previously the one item blocking any training run.
- Clip-length normalization — done, not pending (Section 7).
- Sample rate consistency — done, everything is 16 kHz (Section 7).

## 16. Quick glossary

- **SNR (signal-to-noise ratio):** how much stronger the speech is than the
  background noise, in decibels — higher is cleaner.
- **STOI:** a score estimating how intelligible speech is to a listener.
- **PESQ:** a score estimating perceived speech quality, closer to how a
  human would judge it.
- **SI-SNR:** a training loss related to SNR, commonly used for speech
  enhancement models.
- **STFT / DWT:** two ways of converting raw audio into a form a CNN can
  process — STFT via frequency content over time, DWT via multiple
  resolutions at once.
- **Weak labeling:** giving one label to an entire clip even though the
  labeled event only occupies part of it — used here for noise categories.
- **Multi-task learning:** training a network on more than one task at once
  by sharing most of its layers, so learning one task also improves the
  other through the shared, improved internal features.
- **Causal network:** a network that only ever looks backward in time,
  never forward, so it can run on audio that is still arriving.
- **Lookahead:** deliberately waiting for a small amount of near-future
  audio before finalizing the output for the current moment. It costs
  latency, unlike carrying hidden state forward, which only uses the past.
- **Water-filling:** the way the 50% language share is distributed — each
  language gets an equal share of the budget, any language that cannot meet
  its share contributes everything it has, and the shortfall is redistributed
  over the languages that still have spare clips (Section 9).

## 17. Current state of the data on disk

All figures measured directly from the files. Every clip in the project is
**16 kHz, mono, 16-bit PCM WAV**.

### Speech

| split | clips collected | balanced selection | English | Hindi | other 21 |
| --- | --- | --- | --- | --- | --- |
| train | 1440 | **1293** | 323 (25.0%) | 323 (25.0%) | 647 (50.0%) |
| validation | 172 | **144** | 36 (25.0%) | 36 (25.0%) | 72 (50.0%) |
| testing | 163 | **125** | 31 (24.8%) | 31 (24.8%) | 63 (50.4%) |

All 23 languages are represented in every split. Speaker overlap between
splits is zero. Clip lengths run from 0.64 s to 3.000 s; 47 clips were
trimmed from over 3 s down to exactly 3 s, and 314 clips remain under 3 s.
Manifest durations match the files exactly, with zero mismatches.

Of the balanced selection, the mixer's 1 second floor makes **1292 train,
143 validation and 125 testing** clips actually usable — two clips in the
whole project fall below it.

### Noise

636 clips, every one exactly **3.000 seconds**, renumbered contiguously
from `001` within each category and split.

| category | total | train | validation | testing |
| --- | --- | --- | --- | --- |
| ambient | 102 | 60 | 22 | 20 |
| assault_rifle | 137 | 108 | 13 | 16 |
| engine | 115 | 77 | 20 | 18 |
| heavy_artillery | 53 | 41 | 4 | 8 |
| helicopter | 73 | 53 | 9 | 11 |
| mortar | 45 | 31 | 6 | 8 |
| rocket | 111 | 50 | 31 | 30 |
| **total** | **636** | **420** | **105** | **111** |

That is 31.8 minutes (1908 s) of noise in total. It was built from 811 source
recordings totalling 2070.1 s; the remaining 162.1 s were discarded as
sub-1-second remainders and end-of-category tails.

### Frozen evaluation mixtures

268 noisy/clean pairs — 536 WAV files, 13.4 minutes — generated once at seed
20260830 and kept, so every model version is scored on identical audio.

| split | mixtures | audio |
| --- | --- | --- |
| validation | 143 | 7.2 min |
| testing | 125 | 6.2 min |

Verified after writing: every file is exactly 48000 samples at 16 kHz,
requested and measured SNR agree to 0.0000 dB across all 268, SNR spans
−4.99 to +14.97 dB, no mixture crosses a split, category labels map cleanly
to 0–6, and validation/testing speakers remain disjoint.

## 18. Training runs, and what they showed

Five runs, all scored on the frozen validation mixtures. The noisy baseline is
PESQ 1.199, STOI 0.745, SI-SNR +4.5 dB, and is identical for every run.

| run | what it tested | params | clips | PESQ | STOI | SI-SNR |
| --- | --- | --- | --- | --- | --- | --- |
| 0001 | baseline | 3.0 M | 1292 | 1.671 | 0.840 | 11.53 |
| 0002 | 3x the capacity | 9.3 M | 1292 | 1.694 | 0.844 | 11.72 |
| 0003 | a quarter of the data | 3.0 M | 323 | 1.673 | 0.841 | 11.64 |
| 0004 | half the data | 3.0 M | 646 | 1.650 | 0.836 | 11.50 |
| 0005 | all the data | 3.0 M | 1292 | 1.678 | 0.842 | 11.59 |

### The finding: neither capacity nor data is the constraint

**Tripling the parameters changed nothing** -- +0.02 PESQ for 50% more time
per epoch, which is noise.

**Quartering the data changed nothing either.** Runs 0003 to 0005 are the
data-scaling experiment: identical in every respect except how many distinct
speech clips were available. The step count was held fixed at 80 batches per
epoch with sampling *with replacement*, so every run received exactly the same
number of gradient updates and only the diversity of source recordings
differed. Without that control a smaller dataset would also have meant less
training, and a worse result would have said nothing about which caused it.
The subset was taken as an evenly spaced slice rather than the first N,
because the manifest is sorted by language and the head would have been
Assamese and English only.

The curve is **flat, not flattening**: 323 clips scores the same as 1292, and
25% even beat 50%. Every configuration converges to SI-SNR around 11.6 dB,
STOI around 0.84, PESQ around 1.67. That consistency across a 4x range of data
and a 3x range of capacity is the result.

**Two earlier recommendations in this document were wrong and are withdrawn.**
Collecting more speech and noise would have been weeks of effort for nothing.
And the deferred voice-distortion augmentation (Section 7) is now a much
weaker bet: if four times less *real* diversity changes nothing, synthetic
diversity is unlikely to change anything either. It is not cancelled, since
the problem statement names it, but it should no longer be expected to lift
the scores.

### What the ceiling probably is

Not resources. Two design choices in the model are the leading suspects, and
both are cheap to test:

1. **The model can only attenuate.** `MASK_MAX = 1.0` caps the mask at 1, so
   it can turn sound down but never up. At -5 dB SNR speech bands sit
   *underneath* the noise, and recovering them needs amplification. A one-line
   change tests this.
2. **The noisy phase is reused unchanged.** Only magnitude is masked. At low
   SNR that phase is badly corrupted and no magnitude mask can repair it.
   This is the known ceiling of magnitude-masking systems, and exactly what
   DCCRN's complex-valued layers exist to lift.

A third possibility is a loss/metric mismatch: SI-SNR optimises waveform
match, while PESQ and STOI care about the spectral envelope.

**This changes the case for DCCRN.** It was a stretch goal held back until a
working system existed (Section 4). That condition is met, and the project's
own measurements now argue for it, rather than it resting on the published
results of others.

### The auxiliary classifier works

The noise-category branch reaches **0.52 to 0.59 accuracy against a chance
level of 0.143**, roughly four times chance, in every run. The Section 11 idea
is sound and the branch learns what it was meant to. Whether it *helps* the
audio task is a separate question, still untested: the experiment is a run
with `AUX_LOSS_WEIGHT = 0.0`.

### Cost

An epoch is 11.3 s at 3.0 M parameters and 16.9 s at 9.3 M. A full run to
early stopping is 10 to 20 minutes. Runs are cheap enough that questions like
these should be settled by experiment rather than by argument.
