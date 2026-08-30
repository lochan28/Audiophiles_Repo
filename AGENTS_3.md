# Project context

This project is for Smart India Hackathon problem statement 26052
(DRDO / IDEX): an AI/ML-based Adaptive Noise Cancellation (ANC) system
for defence communication, meant to suppress stationary, non-stationary,
and impulsive noise (gunshots, artillery, rotor/vehicle noise) while
keeping speech intelligible, targeting SNR > 15 dB, STOI > 0.85,
PESQ > 2.5, and low latency on embedded hardware.

The long-form version of everything below, with the full reasoning behind
each decision, is in `PROJECT_CONTEXT_4.md` in this folder.

The project folder is literally `C:\speach data\` — "speach", not
"speech". Older notes use the wrong spelling.

**Where the project is:** data preparation and the **mixing pipeline**
are done, and the frozen evaluation mixtures are on disk. The GPU works.
The **model and training are not started** — there is no `model.py` and
nothing in `models/`. Building CRN and TCN is the next step.

## Layout

```
speach data/
├── dataset/
│   ├── train/ validation/ testing/          one subfolder per language
│   ├── <split>_manifest.csv                 everything collected
│   ├── balanced_<split>_manifest.csv        READ THESE for training
│   ├── noise_manifest.csv
│   └── noise_data/noise_<split>/<category>/
├── scripts/          all Python scripts
├── models/           (empty) checkpoints
├── logs/             (empty) training run records
└── generated_test_mixtures/          268 frozen eval pairs
    ├── <split>/noisy/ + <split>/clean/
    └── test_mixtures_manifest.csv
```

Scripts resolve paths from a `PROJECT_ROOT` derived from `__file__`, so
they can be run from any working directory.

## The data as it stands

Everything is **16 kHz, mono, 16-bit PCM WAV**, and no clip exceeds
**3 seconds**.

**Speech** — 1440 train / 172 validation / 163 testing clips collected
across 23 languages. Lengths run 0.64 s to 3.000 s. Zero speaker overlap
between splits.

The balanced manifests select a strict 25% English / 25% Hindi / 50%
across the other 21 languages: **1293 train, 144 validation, 125
testing**. Surplus clips stay on disk but are not selected. Within the
50%, a scarce language contributes everything it has and the shortfall
is absorbed by languages with spare clips, so per-language counts are
uneven by design.

**Noise** — 636 clips, every one exactly 3.000 s, in seven categories:

| category | total | train | val | test |
| --- | --- | --- | --- | --- |
| ambient | 102 | 60 | 22 | 20 |
| assault_rifle | 137 | 108 | 13 | 16 |
| engine | 115 | 77 | 20 | 18 |
| heavy_artillery | 53 | 41 | 4 | 8 |
| helicopter | 73 | 53 | 9 | 11 |
| mortar | 45 | 31 | 6 | 8 |
| rocket | 111 | 50 | 31 | 30 |

The folder name is the class label. A category counts only if it exists
in **all three splits** — that rule removed a `siren` folder that was
train-only. Two corrections were applied: a folder named `dog_bark` was
a mislabel and is now `assault_rifle`, and misspellings were fixed
(`Helecopter` → `helicopter`, `Heavy_artelary` → `heavy_artillery`).
All category names are lowercase `snake_case`.

The auxiliary noise-classification branch therefore has **7 classes**,
not the 6 in older notes.

## Mixing pipeline

`scripts/mixing.py` turns the speech and noise sets into training pairs.
`SpeechNoiseMixtures(split, seed=None)` is a PyTorch `Dataset` yielding
`(noisy, clean, label)`. Training mixtures are generated fresh every run
and never stored.

- **SNR ~ Uniform(−5, +15) dB.** Output target is SNR > 15 dB, so this
  forces real improvement, not polishing.
- **Categories sampled uniformly**, 1-in-7 each — *not* proportional to
  clip counts. This is why no class weighting is needed in the loss.
- **Short speech is zero-padded to the 3 s window at a random offset**,
  so the model can't learn speech always starts at t=0. Clips under 1 s
  are skipped (costs 2 clips project-wide: 1292/143/125 usable).
- **Speech power measured over real samples only**, never the padding.
- **Clipping rescales noisy and clean together**, keeping the pair valid.
- **Never crosses a split** — validation speech meets validation noise.

**SNR definition — do not "fix" this.** The noise level is set from noise
power over the **whole 3 s window**, against speech power over the
**speech-active region**. Measuring noise only inside the speech span
would let a near-silent span demand a huge scale factor, making a burst
outside it deafening. Consequence: for the four impulsive categories the
SNR *local to the speech* varies several dB at a fixed requested SNR, so
their per-category PESQ/STOI is noisier than the steady categories'.
That is correct, not a bug. Anything verifying the mixer must use this
same definition or it reads several dB off.

`scripts/generate_test_mixtures.py` drives the same code at a fixed seed
(20260830) to freeze the evaluation set: **268 pairs, 536 WAVs, 13.4
min** (143 validation, 125 testing), already written. Both halves are
stored because the clean target is not the original file — it is padded,
randomly placed, and co-rescaled. Verified: SNR requested vs measured
agrees to 0.0000 dB across all 268, no split leaks, speakers disjoint.

### Voice distortion augmentation — planned, deliberately deferred

Decided on and scoped, but **not implemented, and not to be implemented
until a clean baseline exists.** Do not add it unprompted.

**Mode A** is the chosen form: distortion is applied to the speech
*before* it becomes both target and input, so the network learns to
denoise distorted speech rather than to undo the distortion. Mode B
(distort the input only, keep a pristine target) was rejected — it turns
enhancement into restoration, which is far harder on 1292 clips.

Deferring is safe precisely because Mode A changes nothing structural:
same shapes, same alignment, same `(noisy, clean, label)` interface, and
the frozen evaluation set stays valid. It drops in later as one function
called before `mix_at_snr`.

**Trigger to implement:** validation loss stops tracking training loss.
That is overfitting confirmed, which is what this fixes.

**Two hard rules when it is built:**
- **Nothing that changes time alignment** — no time-stretch, no `speed`,
  no pitch shift by resample, no variable delay. SI-SNR and L1 compare
  sample by sample; a desynchronized pair does not error, it silently
  trains against a meaningless objective.
- **Never augment 100% of examples.** ~40% is the intended rate.

Cheap effects are all available in `torchaudio` already
(`bandpass_biquad`, `mu_law_encoding`/`decoding`, numpy clipping, mask
dropout) — about an hour of work. Reverb needs impulse responses:
`fftconvolve` exists but `scipy` and `pyroomacoustics` do not. A real
codec is not planned; `apply_codec` was removed from torchaudio.

Record which runs used augmentation. Never compare an augmented run
against a clean one and read the difference as an architecture result.

## Real-time processing

The model does **not** see a whole 3-second clip at once. The original
plan did, but a deployed real-time system has to process audio as small
packets arrive rather than waiting several seconds for a full clip.

- **Chunk size: 100 ms.** Audio is processed in 100 ms chunks.
- **Lookahead: 2–3 chunks.** Output for a chunk is finalized only after
  2–3 further chunks have been buffered — a little near-future audio
  measurably helps quality, especially at chunk boundaries.
- **Total latency budget: 500–800 ms.**

The budget breaks down as 100 ms of mandatory chunk-wait (a full chunk
must arrive before it can be processed) plus 200–300 ms of lookahead
buffer, so **300–400 ms is structural delay** spent before the network
computes anything, leaving **100–400 ms** for inference and pipeline
overhead depending on which end of the range is targeted.

**Do not increase the chunk size.** Both delays scale directly with it,
so a 200–250 ms chunk would mean 200–250 ms of chunk-wait plus
400–750 ms of lookahead — chunking and buffering alone could consume the
whole budget. Shrinking it only adds margin. The one thing that could
force a change is measured inference time on the real target embedded
hardware, unknowable until a model exists and is benchmarked there.

**Architecture requirements this imposes:**

- The network must be **causal** — it only looks backward in time within
  its processing, never forward, since audio that hasn't arrived yet
  isn't available to it.
- It should **carry memory or hidden state forward** between chunks so
  it isn't starting blind every 100 ms.

These two are not the same as lookahead. Carrying state forward is free
in latency terms since it only uses the past; lookahead buffering
deliberately costs latency since it waits on the future. Real systems
typically use both together.

## Model architecture

No model code exists yet. **DCCRN is not the final choice** — the plan
is staged:

1. **CRN and TCN first.** These are the two architectures actually being
   built and compared.
2. **If that works out well, DCCRN is the next step** — an upgrade on top
   of a working system, not the starting point.
3. **What ships is DCCRN if it is achieved; otherwise whichever of CRN
   and TCN performs better.**

Everything below about DCCRN describes that upgrade step, not the
architecture currently being worked in.

**What the published DCCRN results show.** The original research model
was small — about **3.7 M parameters** — and ranked first in the
real-time category of a well known international noise suppression
competition in 2020. Reported: **PESQ ~3.5** on one test setup and
**~3.2** on a tougher, more realistic test set (our target is 2.5), a
human listener quality score of **~3.4 out of 5**, and about **3 ms per
frame on an ordinary laptop CPU** using only **37 ms of lookahead** —
well under our planned 200–300 ms buffer.

**Do not treat those as our expected numbers.** They came from very
large standard datasets and everyday environmental noise (cafe chatter,
traffic). Our speech set is far smaller and our noise categories —
engine and helicopter hum plus impulsive gunfire, artillery and mortar
bursts — are harder and much less standard. Read them as proof the
architecture *can* clear our targets under favourable conditions, not as
a forecast. Real numbers have to be measured on our own data once a
model exists.

**Feasibility of the DCCRN upgrade.** Should the project get that far,
building DCCRN is realistic rather than theoretical:
working open source implementations exist to study and adapt, so
complex-valued layers do not have to be written from scratch from the
paper's math, and the model is small enough that training something
similar on our GPU is well within reach. But it is genuinely more
implementation work than the simpler CRN or TCN options, mainly because
complex-valued layers are less common and can be trickier to debug.

**Why that build order.** Getting a complete, working end-to-end
pipeline running with the simpler **CRN or TCN** first guarantees a
finished, submittable system exists. **DCCRN is attempted afterward if
time allows**, never as the only path pursued from the start — and if it
is not reached, the better performing of CRN and TCN is what ships.

## Scripts

### `split_speech_dataset.py`

The main clean-speech collector. Pulls Hindi and the other 21 Indian
languages from `ai4bharat/indicvoices_r` (**gated** — accept the access
conditions on the dataset page and run `huggingface-cli login` first)
and English from `openslr/librispeech_asr` (public), which is wired up
in `stream_english()`. Streaming mode throughout, so the ~1 TB
`indicvoices_r` is never downloaded in bulk.

Hard constraints enforced while assigning clips:
1. No audio file lands in more than one split.
2. No speaker appears in more than one split.
3. Inside one split a speaker can repeat, capped at
   `MAX_PER_SPEAKER_PER_SPLIT` (default 3).

Resumable — reads what's already on disk and in the manifests before
continuing, and writes manifests incrementally.

### `collect_english.py`

Pulls only the English share into the same shared manifests. Same
resumable, rule-enforcing behavior.

### `normalize_clip_lengths.py`

The clip-length normalizer. **Dry run by default; pass `--apply` to
write.** Already applied to the current data.

- **Speech**: anything over 3 s keeps its first 3 s, tail discarded.
  Clips already at or under 3 s are left alone. Everything resampled to
  16 kHz (the 22 Indic languages arrived at 48 kHz; English was already
  16 kHz).
- **Noise**: each category within each split is one continuous stream.
  Files are appended in order until 3 s is available, that clip is
  emitted, and the leftover carries into the next clip **only if it is
  longer than 1 second** — otherwise it is dropped and the next file
  starts fresh. The final under-3 s tail of each category is discarded.
  Output is renumbered from `001` with no gaps.

A noise stream never crosses a category boundary or a split boundary,
so no clip mixes two noise types and no training audio leaks into
validation or testing.

### `build_balanced_manifests.py`

Writes the three `balanced_*_manifest.csv` files and
`noise_manifest.csv`. Purely additive — never deletes or renames audio.
Re-run it after anything changes the data on disk.

### `mixing.py`

The mixing pipeline — see the section above. Import
`SpeechNoiseMixtures`; run the file directly for a self-test across all
three splits.

### `generate_test_mixtures.py`

Freezes the validation and testing mixtures at a fixed seed. Dry run by
default, `--apply` to write. Already run.

### `rechunk_long_clips.py` / `rechunk_long_clips_no_manifest.py`

Earlier 5-second-ceiling rechunkers, largely superseded by
`normalize_clip_lengths.py`. Kept for reference.

## Known caveats

- **The GPU works.** `torch 2.13.0+cu130` / `torchaudio 2.11.0+cu130`,
  CUDA 13.0, cuDNN 9.2, on an RTX 5060 Laptop GPU (**sm_120**, 7.96 GB).
  Verified with real kernels, not just `cuda.is_available()`. Python is
  3.14.6 (`cp314`). If reinstalling: the 5060 is Blackwell, so only
  **CUDA 12.8+** builds carry `sm_120` kernels — `cu126` installs fine and
  then dies with "no kernel image is available". Use the `cu130` index.
- **Windows: Smart App Control blocks PyTorch's unsigned DLLs**
  (`WinError 4551`). It had to be turned off and the machine rebooted
  before the CUDA libraries would load. It cannot be re-enabled without
  resetting Windows. Expect this again on any other Windows machine.
- Noise concatenation joins recordings **end to end with no crossfade
  and no inserted gap**, so a 3-second clip can contain several hard
  seams. This was raised and **deliberately dropped as not worth
  addressing** — the noise is background being suppressed, not
  reproduced. Do not open it again as a task.
- `heavy_artillery` has 4 validation clips and `mortar` has 6 — thin for
  scoring those classes.
- The 7 noise classes are imbalanced (45 to 137 clips). The
  classification loss needs class weighting or balanced sampling.
- Held-out-category generalization testing is **not** set up: all seven
  categories appear in all three splits.
- `indicvoices_r`'s exact column names for the audio and speaker ID
  fields were not visible without logging into Hugging Face, so both
  collection scripts guess `"audio"` and `"speaker_id"`. Both print the
  real field names from the first example of each language — if they
  don't match, fix the two spots marked `# FIELD NAME BELOW`.
- Requires `datasets`, `huggingface_hub`, `soundfile`, `numpy`, `torch`,
  `torchaudio`.
