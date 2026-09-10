# Pi 5 deployment runsheet

Everything here has been built and verified on x86 from a copy of `deploy/`
placed outside the repository. **Nothing has run on ARM.** Today is assembly
and measurement, not building.

The goal: replace every scaled-from-x86 estimate in the project documents
with a real measurement from the board, and have a live microphone demo that
works.

Assumes the Pi has **Wi-Fi with internet** (so the bundled wheels are a
safety net, not the only path) and that it is driven **over SSH** from the
laptop holding this repository. If either is untrue, section 2 has the
offline route.

---

## 0. Before the board boots — 5 minutes, at the Imager

Click the gear icon (or `Ctrl+Shift+X`) **before** writing the card:

| setting | value |
| --- | --- |
| hostname | `anc-pi` |
| enable SSH | **Allow public-key authentication only** |
| paste key | the contents of `~/.ssh/id_ed25519.pub` |
| username | `pi` |
| Wi-Fi | your SSID + password, correct country |
| locale | your timezone |

**Choose the OS deliberately: Raspberry Pi OS (64-bit) _Bookworm_.** The
bundled wheels are `cp311` and Bookworm ships Python 3.11. A newer image
shipping 3.13 makes every one of them unusable — survivable with internet,
but it turns a 30-second install into a download.

Doing this at flash time costs five minutes. Retrofitting SSH and Wi-Fi onto
a card later costs an hour and a monitor.

---

## 1. First contact — 10 minutes

Three checks, ordered by how badly each one bites. Do them before anything
depends on them.

```bash
ssh pi@anc-pi.local 'uname -m; python3 --version; free -h; vcgencmd measure_temp'
```

1. **Does `anc-pi.local` resolve?** If not, mDNS is being awkward — take the
   IP from your router's device list and use `pi@192.168.x.x` instead.
   Nothing else in this sheet changes.
2. **Does `uname -m` print `aarch64`?** Every bundled wheel is 64-bit ARM.
   If it prints `armv7l` you are on a 32-bit image: nothing installs, pip
   silently falls back to building torch from source, and that is hours.
   Reflash with the 64-bit image. This cannot happen on a Pi 5, which is
   64-bit only, but a **Pi 4 still commonly runs 32-bit** and this is the
   most likely way to lose a morning.
3. **Is Python 3.11?** If yes, the bundled wheels install instantly. If it is
   3.13, skip `--no-index` in section 2 and let pip fetch from PyPI.
4. **Idle temperature under 60 °C?** A Pi without a heatsink throttles under
   sustained load, and throttling looks exactly like "the model is too slow"
   while being nothing of the kind.
5. **If this is a 2 GB board**, watch `free -h`. Torch is about 1 GB resident
   before the model loads. 4 GB and 8 GB are comfortable; 2 GB is tight.

---

## 2. Get the code onto it — 15 minutes

```bash
scp -r "C:/speach data/deploy" pi@anc-pi.local:~/
```

About 153 MB over Wi-Fi. *(No network? Copy `deploy/` to a USB stick and
`cp -r /media/pi/<stick>/deploy ~/` on the board instead — everything after
this is identical.)*

Then install. **This is the step with the trap:**

```bash
ssh pi@anc-pi.local 'python3 -m venv ~/anc && ~/anc/bin/pip install \
    --no-index --find-links=deploy/wheels \
    torch numpy soundfile sounddevice'
ssh pi@anc-pi.local 'sudo apt install -y libportaudio2'
```

**Why a venv rather than a plain `pip install`.** Raspberry Pi OS marks the
system Python as externally managed (PEP 668), so installing into it fails
outright with a wall of text about `--break-system-packages`. A venv
sidesteps that entirely and is the right habit anyway.

**Why `libportaudio2`.** It is the C library `sounddevice` binds to. Without
it the import *succeeds* and then fails at the first audio stream, which is
a confusing way to lose ten minutes.

Every command from here uses `~/anc/bin/python`, not `python3`.

---

## 3. The go/no-go — 10 minutes

```bash
ssh pi@anc-pi.local 'cd deploy && ~/anc/bin/python selfcheck.py'
```

This is the decision point of the day. One command, no arguments, no
network. It ends in `GO` or `NO-GO` and names the failing check.

**Reference output, measured on x86.** The board should be roughly 3×
slower; the *verdict* should not change:

```
self-tests...
  model_crn    passed
  postfilter   passed
loaded      best_pesq.pt epoch 110, 9,231,225 params
  streaming    passed  (max difference from whole-clip path 1.3e-07)
scored      20 clips
  SI-SNR     14.741   reference  14.741   drift -0.000 dB   ok
streaming   10 frames (100 ms) per chunk, deadline 100 ms
  1 thread   median   54.4 ms   worst   90.9 ms    1.8x headroom   ok
  2 threads  median   27.3 ms   worst   34.5 ms    3.7x headroom   ok
  4 threads  median   17.5 ms   worst   23.8 ms    5.7x headroom   ok

GO
```

**Expect roughly 50 ms median at four threads on a Pi 5.** That number is
what this whole exercise exists to find out.

### On a Raspberry Pi 4

Every step in this sheet is the same. The likely *result* is not. A Pi 4 is
Cortex-A72 at 1.5 GHz against the Pi 5's A76 at 2.4 GHz — roughly **2.5×
slower per core**:

| | 4 threads, median | against the 100 ms deadline |
| --- | --- | --- |
| x86 (measured) | 17.5 ms | 5.7× headroom |
| Pi 5 (estimated) | ~50 ms | ~2× |
| **Pi 4 (estimated)** | **~130 ms** | **misses** |
| Pi 4, stage 1 only | ~45 ms | ~2× |

So on a Pi 4, expect to go straight to **fallback 3 in section 5** — drop
the post-filter, which costs about 0.05 PESQ and buys roughly 3×. Fallback 4
(`CHUNK_FRAMES = 20`) is the other lever, at +100 ms of latency.

Those are scaled from x86, not measured. The board may surprise us either
way, and `selfcheck.py` settles it in ten minutes — which is the point of
running it before anything else.

**The two lines that matter.** `drift` says the model computes the right
answer on ARM — anything inside 0.05 dB is fine, and it is what confirms a
different BLAS and different float rounding have not changed the result.
`worst` against the 100 ms deadline says it keeps up; the worst chunk, not
the median, is what a listener hears, because one late chunk is one dropout.

### About PESQ and STOI

`selfcheck.py` prints them when it can but **does not need them** — SI-SNR
alone decides GO. This is deliberate: `pesq` is a C extension with no aarch64
wheel, verified, so it cannot be installed offline. You will see a note
saying it was skipped. **That is expected and is not a failure.**

To build them anyway (optional, a few minutes, needs network):

```bash
ssh pi@anc-pi.local 'sudo apt install -y build-essential python3-dev'
ssh pi@anc-pi.local '~/anc/bin/pip install pesq pystoi'
```

---

## 4. Audio — 45 minutes

**First without hardware**, so a failure here is unambiguous:

```bash
ssh pi@anc-pi.local 'cd deploy && ~/anc/bin/python live_mic.py \
    --file bundle/generated_test_mixtures/testing/noisy/testing_000.wav out.wav'
```

Look for `0 dropped, 0 underruns`. On x86 this reports about 21 ms median
against the 100 ms budget.

Then plug the USB headset in and find its device numbers — this is the one
part that cannot be prepared in advance:

```bash
ssh pi@anc-pi.local 'cd deploy && ~/anc/bin/python live_mic.py --list'
```

Now run it for real. **Do this from a terminal on the Pi, not over SSH**, so
it keeps running while you listen:

```bash
cd deploy && ~/anc/bin/python live_mic.py --in <N> --out <M>
```

- **Headphones, not speakers.** On speakers the cleaned output feeds back
  into the microphone and the model ends up fighting its own echo.
- It prints a latency line every two seconds. Watch `underruns` — zero is
  what you want; anything else means the board is not keeping up.
- `--bypass` passes the microphone through untouched, which is the honest
  A/B: verified bit-perfect, correlation 1.000000 against the raw input, so
  any difference you hear is the model.
- Added delay is about 190 ms (100 ms to fill a chunk, 88 ms the model holds
  back), plus whatever the USB stack adds — expect roughly a third of a
  second end to end.

---

## 5. If it misses the deadline

Work down this list, stopping as soon as `selfcheck.py` reports `ok` on a
thread count you can live with:

1. **Confirm four threads are actually being used**, nothing else is loading
   the cores, and the board is not throttling:
   ```bash
   ssh pi@anc-pi.local 'vcgencmd measure_temp; vcgencmd get_throttled'
   ```
   Anything other than `throttled=0x0` means it is, and the fix is cooling,
   not code.
2. **Stay on the 128 K post-filter.** `POSTFILTER_WEIGHTS` already points at
   `postfilter_ch48_la0_snr25.pt`. The 505 K `ch96` file is not in the
   bundle precisely so this cannot be got wrong by accident.
3. **Drop the post-filter entirely** — set `POSTFILTER_WEIGHTS = None` in
   both `selfcheck.py` and `live_mic.py`. Stage 1 alone is about 3× cheaper
   and gives up roughly 0.05 PESQ. This is the safe fallback for a live
   demo.
4. **Raise `CHUNK_FRAMES` in `bundle/stream.py`** from 10 to 20. Doubles the
   chunk to 200 ms, halves the per-second overhead, adds 100 ms of latency —
   which pushes the end-to-end figure to the top of the 200–250 ms budget.
   Last resort.

---

## 6. Record what you measured

This is the part that outlasts the day. Do not skip it if things are going
well.

1. **Save the whole `selfcheck.py` output verbatim.**
2. **Copy an enhanced WAV back** and check it scores better than its input:
   ```bash
   scp pi@anc-pi.local:~/deploy/out.wav .
   ```
   Proof the board produces real enhancement, not plausible-sounding
   silence — a distinction that has already caught one bug in this project.
3. **Re-run the benchmark after ~10 minutes of sustained load** and check
   the temperature again. Throttling shows up late, and a demo that works
   for two minutes and dies at five is worse than useless.
4. **Take a short phone video of the live A/B.** It is the strongest single
   thing you can put in front of judges.

Then update the documents. Every Pi figure currently in
`PROJECT_CONTEXT_4.md` and in the deck is an x86 measurement scaled by a
rule of thumb. This run replaces them with real ones, and turns the
Raspberry Pi claim from argued into measured.

---

## Known risks, in the order they are likely to bite

| risk | why it matters | handled in |
| --- | --- | --- |
| PEP 668 blocks `pip install` | fails outright on Bookworm | §2, venv |
| 32-bit image (Pi 4) | no wheel installs, torch builds from source | §1, `uname -m` |
| wheels are `cp311` | unusable on Python 3.13 | §1 check, PyPI fallback |
| Pi 4 is ~2.5× slower | likely misses the deadline two-stage | §3, fallback 3 |
| `.local` does not resolve | cannot connect at all | §1, router IP |
| missing `libportaudio2` | import works, streaming fails | §2, apt |
| thermal throttling | looks like "too slow", is not | §1 and §5 |
| USB audio device numbers | cannot be known in advance | §4, `--list` |

## What "done" looks like

- `selfcheck.py` prints **GO** on the board, drift inside 0.05 dB, zero
  missed chunks at four threads.
- `live_mic.py --file` reports **0 dropped, 0 underruns**, and the returned
  WAV scores better than its noisy input.
- The live microphone runs **ten minutes without underruns**, and A/B
  against `--bypass` is audibly different.
- Real Pi numbers are written into `PROJECT_CONTEXT_4.md`, replacing the
  scaled estimates, and committed.

## Timing

Sections 1–3 take about 35 minutes and answer the question the day exists to
answer. Section 4 is an hour. That leaves the afternoon, which you will
want, because something always breaks.
