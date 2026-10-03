"""Record the INMP441 mic through the ESP32 and check it works.

The transmitter ESP32 (running mic_usb.ino) streams the microphone down the
USB cable. This script records a few seconds, then reports:

  - the ESP32's wiring check (which channel the mic is talking on),
  - how loud the recording is, every half second, as a bar,
  - whether anything went wrong: lost frames, USB errors, clipping, a mic
    stuck at one value,

and saves the recording as a WAV and plays it on the laptop's speakers.

Close the Arduino Serial Monitor first: only one program can hold the port.

    .venv\\Scripts\\python.exe esp32\\pc_mic.py           # record SECONDS
    .venv\\Scripts\\python.exe esp32\\pc_mic.py 10        # record 10 s

Talk, clap, or tap near the mic while it records.
"""

import math
import struct
import sys
import time
import wave
import winsound
from pathlib import Path

import numpy as np
import serial
from scipy.signal import butter, sosfilt

# ============================================================================
# TUNABLE -- safe to change
# ============================================================================

# The port of the board the mic is wired to (the transmitter).
PORT = "COM9"

# How long to record, in seconds. A number on the command line overrides it.
SECONDS = 5.0

# Play the recording when it is done. A second, louder copy is also saved
# (normalised, see LOUD_COPY_PEAK_DB) so a quiet recording is still audible
# -- the plain copy is the honest one, the loud copy is just for listening.
PLAY = True
LOUD_COPY_PEAK_DB = -3.0

# Speech at arm's length should land roughly in this window (dBFS = decibels
# below the loudest possible sound; 0 is the ceiling, more negative is
# quieter). Outside it, the report suggests changing GAIN_SHIFT on the ESP32.
GOOD_SPEECH_RMS_DB = (-30.0, -12.0)

# Sound below this frequency is ignored when measuring loudness. Speech has
# almost nothing under ~80 Hz; what the mic puts there is slow drift.
AUDIBLE_FROM_HZ = 80

SAVE_DIR = Path(__file__).parent / "received"

# ============================================================================
# FIXED -- must match mic_usb.ino
# ============================================================================

BAUD = 921600
RATE = 16000
MAGIC = b"AUD0"
HEAD = struct.Struct("<IH")       # seq, n
FULL_SCALE = 32768.0


def db(x):
    return 20 * math.log10(max(x, 1e-9))


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else SECONDS
    SAVE_DIR.mkdir(exist_ok=True)

    port = serial.Serial()
    port.port, port.baudrate, port.timeout = PORT, BAUD, 0.1
    port.dtr = port.rts = False
    port.open()
    # Press the board's reset button for it, so we see its start-up wiring
    # check (otherwise it ran long ago, before we were listening).
    port.rts = True
    time.sleep(0.1)
    port.rts = False

    print(f"Listening on {PORT}...")
    buf, blocks, bad, expected_seq, lost = b"", [], 0, None, 0
    want = int(seconds * RATE)
    got = 0
    started = None
    deadline = time.time() + seconds + 15

    while got < want and time.time() < deadline:
        buf += port.read(8192)
        while True:
            i = buf.find(MAGIC)
            if i < 0:
                # Text from the ESP32 before streaming starts: show it.
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    s = line.decode("utf-8", "replace").strip()
                    if s and s.isprintable():
                        print("  ESP32:", s)
                break
            if started is None:
                for line in buf[:i].split(b"\n"):
                    s = line.decode("utf-8", "replace").strip()
                    if s and s.isprintable():
                        print("  ESP32:", s)
                started = time.time()
                print(f"Recording {seconds:g} s -- talk, clap or tap near the mic...")
            frame = buf[i:]
            if len(frame) < 4 + HEAD.size:
                buf = frame
                break
            seq, n = HEAD.unpack_from(frame, 4)
            total = 4 + HEAD.size + 2 * n + 2
            if n == 0 or n > 4096:                 # not a real frame
                buf = frame[4:]
                continue
            if len(frame) < total:
                buf = frame
                break
            audio = frame[4 + HEAD.size: 4 + HEAD.size + 2 * n]
            (checksum,) = struct.unpack_from("<H", frame, total - 2)
            buf = frame[total:]
            if (sum(audio) & 0xFFFF) != checksum:
                bad += 1
                continue
            if expected_seq is not None and seq != expected_seq:
                lost += (seq - expected_seq) & 0xFFFFFFFF
            expected_seq = seq + 1
            blocks.append(np.frombuffer(audio, dtype="<i2"))
            got += n
    port.close()

    if not blocks:
        sys.exit("\nNo audio arrived. Is mic_usb.ino uploaded to this board, "
                 "and the Serial Monitor closed?")

    x = np.concatenate(blocks)[:want].astype(np.float64)
    path = SAVE_DIR / "mic_test.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(x.astype("<i2").tobytes())

    # ---- report ------------------------------------------------------------
    dc = x.mean()
    y = x - dc
    peak = np.abs(y).max()
    clipped = int(np.sum(np.abs(x) >= 32767))
    print(f"\nGot {len(x) / RATE:.2f} s.  Missing frames: {lost} "
          f"(of which {bad} arrived damaged and were dropped).  "
          f"Clipped samples: {clipped}.")
    # Level is measured on the AUDIBLE part only. The INMP441's output also
    # drifts slowly (a few Hz -- far below hearing); on the first real
    # recording that drift alone read ~10-15 dB louder than the actual
    # sound and made a quiet voice look like a good level.
    heard = sosfilt(butter(4, AUDIBLE_FROM_HZ, "highpass", fs=RATE, output="sos"), y)
    print(f"Peak {db(np.abs(heard).max() / FULL_SCALE):6.1f} dBFS (audible)    "
          f"DC offset {dc:+.0f} ({db(abs(dc) / FULL_SCALE):.0f} dBFS, silent, "
          f"filtered out)")
    print(f"\nLoudness every 0.5 s (sound above {AUDIBLE_FROM_HZ} Hz):")
    half = RATE // 2
    rms_list = []
    for k in range(0, len(heard), half):
        seg = heard[k:k + half]
        r = db(np.sqrt(np.mean(seg ** 2)) / FULL_SCALE)
        rms_list.append(r)
        bar = "#" * max(0, int((r + 80) / 2))
        print(f"  {k / RATE:4.1f}s  {r:6.1f} dB  {bar}")

    if np.ptp(x) == 0:
        print("\nVERDICT: the mic is STUCK at one value -- it is not sending "
              "sound. Check VDD (3V3), GND and the SD wire.")
    else:
        loudest = max(rms_list)
        lo, hi = GOOD_SPEECH_RMS_DB
        if clipped > 0:
            print(f"\nVERDICT: working, but {clipped} samples CLIPPED -- lower "
                  "GAIN_SHIFT in mic_usb.ino.")
        elif loudest < lo:
            print(f"\nVERDICT: working but QUIET (loudest {loudest:.0f} dB). If "
                  "you were talking, raise GAIN_SHIFT in mic_usb.ino by 1 or 2.")
        elif loudest > hi:
            print(f"\nVERDICT: working, LOUD (loudest {loudest:.0f} dB) -- fine, "
                  "lower GAIN_SHIFT if it sounds harsh.")
        else:
            print(f"\nVERDICT: working, good level (loudest {loudest:.0f} dB).")

    loud = SAVE_DIR / "mic_test_loud.wav"
    hpeak = np.abs(heard).max()
    if hpeak > 0:
        z = heard / hpeak * (10 ** (LOUD_COPY_PEAK_DB / 20)) * 32767
        with wave.open(str(loud), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
            w.writeframes(z.astype("<i2").tobytes())
    print(f"\nSaved {path.name} (as recorded) and {loud.name} (boosted for "
          f"listening) in {SAVE_DIR}")

    if PLAY and peak > 0:
        print("Playing the boosted copy...")
        winsound.PlaySound(str(loud), winsound.SND_FILENAME)   # waits until done


if __name__ == "__main__":
    main()
