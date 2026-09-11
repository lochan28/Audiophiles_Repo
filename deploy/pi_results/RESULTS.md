# Raspberry Pi 4 results — measured 2026-09-11

Board: Raspberry Pi 4 Model B Rev 1.1, 2 GB, Cortex-A72 x4 @ 1.5 GHz
OS:    Raspberry Pi OS 64-bit (newest), Python 3.13.5, torch 2.14.0+cpu
Model: run 0026 stage 1 (9.23 M) + 128 K post-filter (ch48), fp32, 4 threads

## Correctness — settled

selfcheck.py:  drift -0.000 dB from the x86 reference (14.741 dB SI-SNR)
               streaming vs whole-clip path 1.1e-07
               model loads in 1.8 GB; peak usage ~500 MB

Three clips processed end-to-end on the board by the streaming path
(pi_*.wav in this directory), scored on x86 against the clean reference:

    clip          SI-SNR (dB)         PESQ             STOI
    testing_000   -1.63 ->  6.69     1.113 -> 1.388   0.651 -> 0.744
    testing_003    4.82 -> 12.79     1.244 -> 1.666   0.714 -> 0.870
    testing_007    8.96 -> 16.01     1.227 -> 1.890   0.890 -> 0.941

testing_000 matches the x86 live_mic result (6.69 dB) exactly.

## Speed — NOT real time on this board

Streaming, 100 ms chunks, 4 threads, against a 100 ms deadline:

    two-stage fp32       424 ms median   4.2x over
    stage 1 only fp32    290 ms median   2.9x over
    two-stage int8       308 ms median   3.1x over    (qnnpack, 1.28x gain)
    stage 1 only int8    191 ms median   1.9x over    (qnnpack, 1.45x gain)

    whole clip, two-stage fp32:   3.0 s of audio in 11.7 s  = 0.26x real time

int8 costs 1.8e-3 max error on rms 0.063 -- inaudible, consistent with x86.
Temperature 70-75 C under load, throttled=0x0 (not throttling).

Thread scaling is poor (815 / 549 / 430 ms at 1 / 2 / 4), as predicted:
the LSTM is sequential.

## What this means

The two-stage system is verified correct on a Pi 4 and runs at about a
quarter of real time. Real time on THIS board needs the encoder/decoder
made incremental (per-layer ring buffers, the same fix that bought 3-5.7x
on the post-filter) combined with int8 -- estimated to land at 70-100 ms,
i.e. at the deadline with no headroom, stage 1 only. A Pi 5 (A76 @ 2.4 GHz,
roughly 2.5x faster per core) is expected to run the full two-stage system
in real time without either change; that is an estimate, not a measurement.

The pre-day estimate of ~130 ms for a Pi 4 was wrong by 3x. Torch's fp32
kernels on Cortex-A72 are far weaker than a clock-ratio scaling suggests.

---

# Raspberry Pi 5 results — measured 2026-09-11, same SD card

Board: Raspberry Pi 5 Model B Rev 1.1, 8 GB, Cortex-A76 x4 @ 2.4 GHz
Same card, venv, torch 2.14.0+cpu, deploy/ and scripts as the Pi 4 above.
Only the silicon changed, so every difference below is the board.

## Correctness

selfcheck.py: drift -0.000 dB, streaming vs whole-clip 1.1e-07. Identical
to the Pi 4 and to x86.

## Speed — streaming, 100 ms chunks, 4 threads, 100 ms deadline

                       Pi 4          Pi 5          Pi 5 verdict
    fp32 two-stage     424 ms        145.5 ms      MISSED
    fp32 stage 1       290 ms        100.4 ms      MISSED (just)
    int8 two-stage     308 ms        110.8 ms      MISSED
    int8 stage 1       191 ms         59.7 ms      OK  (worst 85.4)

Pi 5 is 2.8-3.2x faster than Pi 4 per configuration.

## Live — the actual demo

live_mic.py --int8 --no-postfilter, USB headset at 48 kHz, 16 s run:

    58.0 ms median, 60.1 ms worst per 100 ms chunk, 1.7x headroom
    0 chunks dropped, 0 underruns, steady

backlog_demo.py, same configuration, 30 s: BACKLOG 0.0 s throughout,
100% of real time. The identical script on the Pi 4 showed 26% and a
backlog growing 0.75 s per second.

## What runs in real time on a Pi 5

Stage 1 with dynamic int8 on the LSTM. Not the two-stage system, and not
fp32 stage 1 (100.4 ms median is exactly at the deadline, worst 126 ms).
The post-filter's 0.05 PESQ is the price of real time on this board.

int8 costs 1.8e-3 max error on rms 0.063 -- inaudible.

## Thermal — matters for a demo

No cooler fitted. Idle 51-62 C. Under sustained load it reached 81.8 C and
throttled (throttled=0xe0000: frequency capped, soft temperature limit
hit); the first selfcheck ran partly throttled and read 151 ms at 4
threads instead of 145. Short runs hold 2.4 GHz and 65-70 C. For a demo
longer than a minute or two, fit the official active cooler, or expect the
numbers to drift toward the Pi 4's.
