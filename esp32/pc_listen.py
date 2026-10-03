"""Hear the ESP32 radio link through the laptop's speakers.

The receiver ESP32 (running rx_usb.ino) catches a clip over the radio and
sends it down the USB cable. This script picks it up, plays it on the
laptop's speakers, saves it as a WAV, and -- if the original clip.h is
around -- checks the received audio against what the transmitter holds, so
you can see whether the radio delivered it bit for bit.

Close the Arduino Serial Monitor first: only one program can hold the port.

    .venv\\Scripts\\python.exe esp32\\pc_listen.py          # listen forever
    .venv\\Scripts\\python.exe esp32\\pc_listen.py 3        # stop after 3 clips

Ctrl+C stops it.
"""

import struct
import sys
import time
import wave
import winsound
from pathlib import Path

import serial

# ============================================================================
# TUNABLE -- safe to change
# ============================================================================

# The receiver's USB port. Device Manager -> Ports shows it ("Silicon Labs
# CP210x ... (COM10)").
PORT = "COM10"

# Play each clip on the laptop speakers as it arrives. False = save only.
PLAY = True

# Where the received WAVs go. One file per clip, numbered.
SAVE_DIR = Path(__file__).parent / "received"

# Compare each clip against the transmitter's clip.h and report how much of
# it came through identical. Harmless if the file is missing.
COMPARE_WITH = Path(__file__).parent / "tx_clip" / "clip.h"

# ============================================================================
# FIXED -- must match rx_usb.ino, or the data is misread
# ============================================================================

BAUD = 921600
MAGIC = b"CLIP"
HEADER = struct.Struct("<IIHH")    # rate, samples, packets got, packets expected
MAX_SAMPLES = 16000 * 10           # sanity limit: anything bigger is garbage


def load_original():
    """The transmitter's clip as bytes, read out of clip.h."""
    if not COMPARE_WITH.exists():
        return None
    text = COMPARE_WITH.read_text()
    body = text[text.index("{") + 1: text.rindex("}")]
    return bytes(int(v) for v in body.replace("\n", "").split(",") if v.strip())


def read_exact(port, n):
    data = b""
    deadline = time.time() + 5
    while len(data) < n:
        chunk = port.read(n - len(data))
        if chunk:
            data += chunk
            deadline = time.time() + 5
        elif time.time() > deadline:
            raise TimeoutError(f"USB went quiet after {len(data)} of {n} bytes")
    return data


def main():
    stop_after = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    SAVE_DIR.mkdir(exist_ok=True)
    original = load_original()

    port = serial.Serial()
    port.port, port.baudrate, port.timeout = PORT, BAUD, 0.1
    port.dtr = port.rts = False        # opening the port must not reset the board
    port.open()
    print(f"Listening on {PORT}. Waiting for a clip (one arrives every ~5 s)...")

    text, clips = b"", 0
    try:
        while not stop_after or clips < stop_after:
            text += port.read(4096)

            # Print whatever log lines the ESP32 sent before the next clip.
            i = text.find(MAGIC)
            shown = text if i < 0 else text[:i]
            *lines, rest = shown.split(b"\n")
            for line in lines:
                if line.strip():
                    print("  ESP32:", line.decode("utf-8", "replace").strip())
            if i < 0:
                text = rest[-len(MAGIC):] if len(rest) > 200 else rest
                continue

            # A clip starts here. Read it: header, audio, checksum.
            leftover = text[i + len(MAGIC):]
            head = leftover[:HEADER.size]
            head += read_exact(port, HEADER.size - len(head))
            rate, n, got, expected = HEADER.unpack(head)
            if not (0 < n <= MAX_SAMPLES):
                print(f"  garbage header (n={n}) -- skipping")
                text = leftover
                continue
            audio = leftover[HEADER.size:HEADER.size + n]
            audio += read_exact(port, n - len(audio))
            rest = leftover[HEADER.size + n:]
            tail = rest[:4] + read_exact(port, max(0, 4 - len(rest)))
            text = rest[4:]
            (checksum,) = struct.unpack("<I", tail)

            clips += 1
            ok_usb = (sum(audio) & 0xFFFFFFFF) == checksum
            lost = expected - got
            path = SAVE_DIR / f"clip_{clips:03d}.wav"
            with wave.open(str(path), "wb") as w:   # 8-bit WAV is unsigned,
                w.setnchannels(1)                   # the same format the
                w.setsampwidth(1)                   # ESP32 sends -- no conversion
                w.setframerate(rate)
                w.writeframes(audio)

            print(f"\nClip {clips}: {n / rate:.2f} s at {rate} Hz, "
                  f"{got}/{expected} packets ({lost} lost), "
                  f"USB checksum {'OK' if ok_usb else 'WRONG'} -> {path.name}")
            if original is not None and len(original) == n:
                same = sum(a == b for a, b in zip(audio, original))
                verdict = "IDENTICAL to what was sent" if same == n else \
                          f"{n - same} of {n} samples differ (lost packets become silence)"
                print(f"  vs transmitter: {verdict}")

            if PLAY:
                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
    except KeyboardInterrupt:
        pass
    finally:
        port.close()
    print(f"\nStopped. {clips} clip(s) saved in {SAVE_DIR}")


if __name__ == "__main__":
    main()
