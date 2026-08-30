# Dropping in a new noise class

Put a folder here named exactly what you want the class called, with the
three splits inside it, and put your `.wav` files in those:

```
new_noise_class/
└── siren/                 <- the class name; lowercase_with_underscores
    ├── train/             <- put wav files here
    ├── validation/        <- and here
    └── testing/           <- and here
```

Then run:

```
python scripts/add_noise_class.py            # dry run, shows what it would do
python scripts/add_noise_class.py --apply    # actually installs it
```

## Rules the installer enforces

**All three splits must have at least one file.** A category present in only
some splits is dropped by the intersection rule -- this is exactly why an
earlier `siren` folder disappeared, having had one file in train and nothing
elsewhere. The installer refuses rather than letting that happen silently.

**Keep speakers and recordings apart across splits.** Do not put two clips
cut from the same source recording into different splits: the model would be
evaluated on audio it effectively trained on. The installer cannot detect
this for you.

## What you do NOT need to worry about

- **Length.** Clips of any length are fine. They get cut into exact 3.000 s
  pieces, with short ones concatenated together, using the same stream rule
  as the rest of the noise set.
- **Sample rate or channels.** Anything is resampled to 16 kHz mono.
- **File names.** They get renumbered to `<class>_001.wav` and so on.
- **Label numbers.** The class is appended to `dataset/noise_classes.txt` and
  takes the next free label. Existing labels never move.
- **Manifests.** Rebuilt automatically.

## What it will NOT do

It stops before training, and it does not touch the existing frozen
evaluation mixtures in `generated_test_mixtures/`. Those contain no examples
of the new class, so they stay valid for comparison against earlier results.
Generating a second frozen set that includes the new class is a separate,
deliberate step.
