# The Audiophiles — AI noise cancellation for battlefield radio

**Smart India Hackathon 2026 · Problem statement SIH26052 (DRDO / iDEX)**
Project site: **https://audiophiles-sih26052.vercel.app**

A soldier speaks into a radio. Around them: gunfire, artillery, helicopters,
engines. This project removes that noise in real time and keeps the speech
intelligible, on hardware as small as a Raspberry Pi 5, fully offline.

This branch holds the **code, training records and documentation**. The
project website lives on `main`.

---

## Results

Scored on **125 held-out test mixtures** that were never used for training
or for choosing the model:

| | PESQ (quality) | STOI (intelligibility) | SI-SNR (noise removed) |
|---|---|---|---|
| Noisy input | 1.223 | 0.754 | 6.14 dB |
| Stage 1 alone | 1.957 | 0.864 | 13.72 dB |
| **Our system (two-stage)** | **1.990** | **0.866** | **13.95 dB** |
| DeepFilterNet v0.5.6 *(same test set, same scoring code)* | 1.947 | 0.851 | 11.67 dB |
| Target | 2.5 | 0.85 | 15 dB |

- **STOI meets its target.** PESQ and SI-SNR are still short of theirs.
- **We beat DeepFilterNet**, a published open-source system, on all three
  metrics on our own test set.

## How it works

```
noisy audio ─▶ STFT ─▶ STAGE 1: CRN ─────────▶ mask × spectrogram ─▶ STAGE 2 ─▶ clean audio
                       encoder → LSTM → decoder                       post-filter
                       7.31 M parameters                              128 K parameters
```

- **Stage 1** is a convolutional recurrent network (CRN). It predicts a mask
  that turns down each frequency band at each moment. It uses ERB bands above
  2 kHz to save compute.
- **Stage 2** is a small, strictly causal post-filter. It predicts a
  correction to *add* to stage 1's output, so it can restore energy that a
  mask can only remove.
- **Streaming:** `scripts/stream.py` runs the model chunk by chunk, carrying
  state forward between chunks. Its self-test checks that the streamed audio
  matches whole-clip processing to about 1e-7, at any chunk size.

## Latency

Latency means two different things here, and both are reported:

- **Can the board keep up?** On a **Raspberry Pi 5** (4 threads, int8), the
  two-stage system takes **69.7 ms per 100 ms of audio**, which is 1.4× real
  time.
- **How long until the listener hears it?** The model needs **86 ms of
  lookahead** (60 ms chosen, plus 26 ms from STFT framing). With 100 ms audio
  blocks, the live mouth-to-ear delay is **about 300 ms**, plus the sound
  card's own delay, which has not been measured yet.

Smaller 50 ms blocks would bring the delay down to about 200 ms. On the Pi 5
they meet their deadline while the board is cool (40.5 ms per block). Without
a heatsink, though, the board throttles within about 90 seconds. They are on
hold until it has cooling. `HANDOFF.md` §2 has the full breakdown.

## What is and isn't in this repository

**Included:**
- All code
- Every training run's configuration, metrics, notes and a snapshot of the code
  that produced it
- The post-filter weights (`models/`)
- Dataset manifests
- The Raspberry Pi and ESP32 deployment kits
- The full engineering record

**Not included:**
- **Audio.** The speech and noise datasets are not redistributed here. That
  includes the frozen evaluation mixtures and recordings.
- **Stage-1 model weights.** `runs/*/checkpoints/` is not tracked, because the
  shipped checkpoint alone is 29 MB.

To reproduce the results you need the data (see below) and a training run.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows. On Linux/macOS: source .venv/bin/activate

# PyTorch. RTX 50-series GPUs need a CUDA 12.8+ build; CPU-only also works.
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0+cu130 torchaudio==2.11.0+cu130
pip install numpy soundfile matplotlib pesq pystoi
```

These self-tests need **no data and no checkpoints**. They run straight after
cloning:

```bash
python scripts/model_crn.py     # every build-step check for the CRN
python scripts/postfilter.py    # asserts the post-filter is strictly causal
python scripts/model_tcn.py     # the TCN alternative
```

With the dataset in place:

```bash
python scripts/train.py --epochs 150 --eval-every 5 --lstm-hidden 512 --channel-scale 1.5
python scripts/evaluate.py <run_name> --split testing
python scripts/stream.py        # streaming vs offline, at five chunk sizes
python scripts/serve.py         # web UI: /dashboard, /bench, /lab
```

## Data

- **Speech:** [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r)
  covers 22 Indian languages. [LibriSpeech](https://www.openslr.org/12)
  covers English and is licensed CC BY 4.0. Each source has its own terms;
  check them before redistributing any audio.
- **Noise:** 636 three-second clips in seven categories:
  - ambient
  - assault_rifle
  - engine
  - heavy_artillery
  - helicopter
  - mortar
  - rocket
- **Mixing:** each training mixture puts noise under speech at an SNR drawn
  90% from 0 to 15 dB and 10% from −5 to 0 dB.
- **Evaluation:** a frozen set of 268 mixtures (143 validation, 125 test),
  generated once with seed 20260830.

These scripts rebuild the dataset from source audio:
- `collect_english.py`
- `split_speech_dataset.py`
- `normalize_clip_lengths.py`
- `build_balanced_manifests.py`
- `generate_test_mixtures.py`

Without the exact source files, a rebuilt evaluation set will not be
byte-identical to ours, so expect numbers close to the ones above rather than
identical.

## Repository layout

| Path | What it holds |
|---|---|
| `scripts/` | model, training, evaluation, streaming, data pipeline, web UI |
| `deploy/` | Raspberry Pi 5 kit: live microphone, benchmarks, self-check, runsheet |
| `esp32/` | ESP32 audio-link sketches. `tx_clip/clip.h` is generated, so run `python esp32/make_clip.py <any.wav>` |
| `runs/` | one folder per training run: config, metrics, notes, code snapshot |
| `models/` | post-filter weights |
| `dataset/` | manifests and the noise-class list (no audio) |
| `HANDOFF.md` | **start here.** Current state, every experiment, what was ruled out, what's next |
| `PROJECT_CONTEXT_4.md` | the long-form engineering record |
| `CRN_STEPS.md` | how the model was built, step by step |
| `AGENTS_3.md` | short operating brief for anyone (human or AI) working on the code |

The documents mention commit hashes from the original development history.
That history was rewritten to strip the audio before publishing here, so those
hashes don't exist on this branch.

## Licence

No licence has been chosen yet. Until one is added, the code is all rights
reserved by its authors. Data licences are listed under [Data](#data).
