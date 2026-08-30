# Project context

This project is for Smart India Hackathon problem statement 26052
(DRDO / IDEX): an AI/ML-based Adaptive Noise Cancellation (ANC) system
for defence communication, meant to suppress stationary, non-stationary,
and impulsive noise (gunshots, artillery, rotor/vehicle noise, sirens)
while keeping speech intelligible, targeting SNR > 15 dB, STOI > 0.85,
PESQ > 2.5, and low latency on embedded hardware.

The two scripts in this folder are **only the clean-speech data
collection stage**. They do not touch noise mixing, model training,
or inference -- that work comes later and is out of scope for now.

## What each script does

### `split_speech_dataset.py` (the current/main one -- run this)

Builds three separate clean-speech sets on disk:

    dataset/train/        (target: 410 clips)
    dataset/validation/    (target: 50 clips)
    dataset/testing/       (target: 50 clips)

Each split is targeted at 25% English, 25% Hindi, and the remaining
50% spread as evenly as possible across the other 21 scheduled Indian
languages (Assamese, Bengali, Bodo, Dogri, Gujarati, Kannada, Kashmiri,
Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi,
Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu).

Sources:
- Hindi + the other 21 Indian languages come from `ai4bharat/indicvoices_r`
  on Hugging Face -- this is a **gated** dataset: you must accept its
  access conditions on the dataset page and run `huggingface-cli login`
  with an access token before this script will work.
- English is currently a **placeholder**. `indicvoices_r` doesn't include
  English, so `stream_english()` just returns an empty stream for now --
  English's 25% share is still reserved in the folders/manifests, it
  simply won't collect anything until a real English source (e.g.
  VCTK, LibriSpeech, Common Voice) is wired in where the function's
  docstring shows.

Hard constraints enforced while assigning clips:
1. No audio file is ever placed in more than one split.
2. No speaker appears in more than one split (a speaker is "locked" to
   whichever split first picks them up).
3. Inside one split, the same speaker can repeat, capped at
   `MAX_PER_SPEAKER_PER_SPLIT` (default 3).

Each split also gets its own manifest CSV (`train_manifest.csv`,
`validation_manifest.csv`, `testing_manifest.csv`) listing filename,
language, speaker_id, and duration for every clip.

### `collect_speech_data.py` (earlier draft -- superseded)

An earlier, simpler version that just pulls a fixed number of samples
per language (100 Hindi, 10 each for the other 21, 100 English from
LibriSpeech) into one flat `clean_speech/` folder with one manifest.
Kept for reference; `split_speech_dataset.py` is the one that matters
now since it adds the train/validation/testing split with the speaker
constraints above.

## Known caveats / things to check when running

- `indicvoices_r`'s exact column names (for the audio and speaker ID
  fields) were not visible without logging into Hugging Face, so both
  scripts guess `"audio"` and `"speaker_id"`. Both scripts print the
  real field names from the first example of each language to the
  console -- if they don't match, fix the two spots marked
  `# FIELD NAME BELOW`.
- Requires `pip install datasets huggingface_hub soundfile numpy`.
- Streaming mode is used throughout so the full ~1 TB `indicvoices_r`
  dataset is never downloaded in bulk -- only the clips actually needed.
