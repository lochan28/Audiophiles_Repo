# Pi 5 Deployment Runsheet

This runsheet outlines the step-by-step process for deploying the speech-enhancement model to the Raspberry Pi 5.

## 1. Boot and Prepare Pi
- Boot the Raspberry Pi 5.
- Ensure the Pi is fully powered and disconnected from the internet (offline environment).

## 2. Transfer Deployment Bundle
- Connect the USB drive containing the `deploy/` directory to the Pi.
- Copy the entire `deploy/` directory to the Pi's local storage:
  ```bash
  cp -r /path/to/usb/deploy /home/pi/deploy
  cd /home/pi/
  ```

## 3. Install Offline Wheels
- The bundle includes aarch64-compatible wheels for all necessary packages.
- Install them using the provided offline wheels:
  ```bash
  python3 -m pip install deploy/wheels/*.whl --no-index --find-links deploy/wheels/
  ```

## 4. Run Self-Check
- One command, one answer. It needs no network and no arguments:
  ```bash
  python3 deploy/selfcheck.py
  ```
- **Expected output** (times will be slower than these x86 figures; the
  verdict should not change):
  ```
  self-tests...
    model_crn    passed
    postfilter   passed
  loaded      best_pesq.pt epoch 110, 9,231,225 params
    streaming    passed  (max difference from whole-clip path 1.3e-07)
  scored      20 clips
    SI-SNR     14.741   reference  14.741   drift -0.000 dB   ok
  streaming   10 frames (100 ms) per chunk, deadline 100 ms
    1 thread   median   52.6 ms   worst   76.2 ms    1.9x headroom   ok
    2 threads  median   28.6 ms   worst   32.4 ms    3.5x headroom   ok
    4 threads  median   17.1 ms   worst   21.7 ms    5.8x headroom   ok

  GO
  ```
- **The two lines that matter.** `drift` says the model computes the right
  answer on this board; anything inside 0.05 dB is fine. `worst` against the
  100 ms deadline says it keeps up -- the worst chunk, not the median, is
  what a listener hears, because one late chunk is one dropout.
- If it prints `NO-GO`, the failing check is named just above the verdict.

### About PESQ and STOI
`selfcheck.py` prints them when it can, but **does not need them** -- SI-SNR
alone decides GO. This is deliberate: `pesq` is a C extension with no
aarch64 wheel, so it cannot be installed offline and must be compiled. You
will see a note saying it was skipped. That is normal and is not a failure.

If you do want them on the board (not required for the demo), you need
network and a toolchain:
```bash
sudo apt install -y build-essential python3-dev
pip install pesq pystoi
```
Budget a few minutes for the build. Skip this on a tight day.

## 5. If it misses the deadline
Work down this list, stopping as soon as `selfcheck.py` reports `ok` on a
thread count you can live with:

1. **Confirm four threads are actually being used.** `torch.set_num_threads`
   is already called, but check nothing else is loading the cores, and that
   the Pi is not thermally throttling (`vcgencmd measure_temp`).
2. **Stay on the 128 K post-filter.** `POSTFILTER_WEIGHTS` in `selfcheck.py`
   already points at `postfilter_ch48_la0_snr25.pt`. The 505 K `ch96` file
   is not in the bundle precisely so this cannot be got wrong by accident.
3. **Drop the post-filter entirely.** Stage 1 alone runs about 7x faster than
   real time on x86 and gives up roughly 0.05 PESQ. It is the safe fallback
   for a live demo.
4. **Raise `CHUNK_FRAMES` in `stream.py`** from 10 to 20. Doubles the chunk
   to 200 ms, halves the per-second overhead, and adds 100 ms of latency --
   which pushes the end-to-end figure to the top of the 200-250 ms budget.
   Last resort.

## 6. Record what you measured
Copy the whole `selfcheck.py` output into a file and keep it. Every Pi
number in the project documents is currently an x86 measurement scaled by a
rule of thumb; this run is the first real one, and it replaces them.
