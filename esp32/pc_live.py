"""Play the ESP32 radio link LIVE on the laptop.

mic -> tx_live.ino (COM9) -> radio -> rx_live.ino (COM10) -> USB -> this script
-> laptop speakers / headphones.

While it runs it prints, every 2 seconds, how many packets arrived, how many
were lost on the way, and how much sound is waiting in the playback buffer.
When it stops it saves everything it played as a WAV.

    .venv\\Scripts\\python.exe esp32\\pc_live.py          # until Ctrl+C
    .venv\\Scripts\\python.exe esp32\\pc_live.py 20       # stop after 20 s

USE HEADPHONES. With laptop speakers the mic hears its own output and you
get a howl (feedback).

Close the Arduino Serial Monitor first: only one program can hold the port.

This is where our noise-cancelling model will plug in later: every block of
sound passes through on_audio() below on its way to the speakers.
"""

import struct
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import serial
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

# ============================================================================
# TUNABLE -- safe to change
# ============================================================================

PORT = "COM10"            # the RECEIVER's port

# How much sound to collect before playback starts, in ms. Radio packets
# arrive slightly unevenly; this cushion hides that. Bigger = smoother but
# more delay. 40-100 is sensible.
BUFFER_TARGET_MS = 60

# If the cushion grows past this, trim it back to BUFFER_TARGET_MS. It grows
# slowly on its own because the ESP32's clock and the laptop's sound card
# never tick at exactly the same speed; without a trim the delay would creep
# up over minutes.
BUFFER_MAX_MS = 250

# Extra loudness for listening, in dB. Use the headphone volume first; only
# raise this if that is not enough. The first live session ran at +12 and
# CLIPPED 7.4% of the audio (speech was louder than in the mic test), so the
# default is now 0. The final report says how much clipped.
PLAYBACK_GAIN_DB = 0.0

# Remove sound below this frequency. The INMP441 drifts slowly (a few Hz);
# speech has almost nothing under ~80 Hz, so this removes only the drift.
HIGHPASS_HZ = 80

# A gap bigger than this many packets is not "lost packets" but the
# transmitter restarting (its counter went back to 0). Do not fill it.
MAX_FILL_PACKETS = 20

STATUS_EVERY_S = 2.0
SAVE_PATH = Path(__file__).parent / "received" / "live_session.wav"

# ============================================================================
# FIXED -- must match rx_live.ino / tx_live.ino
# ============================================================================

BAUD = 921600
RATE = 16000
MAGIC = b"AUD0"
HEAD = struct.Struct("<IH")


def on_audio(block):
    """Every block of sound passes through here before it is played.

    Today it does nothing. This is the spot where the noise-cancelling model
    goes: take `block` (float32, 16 kHz, mono), return the cleaned block.
    """
    return block


class Playback:
    """A small cushion of sound between the radio and the speakers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.blocks = deque()
        self.queued = 0                # samples waiting
        self.started = False
        self.underruns = 0
        self.trimmed_ms = 0.0
        self.target = RATE * BUFFER_TARGET_MS // 1000
        self.maximum = RATE * BUFFER_MAX_MS // 1000

    def push(self, block):
        with self.lock:
            self.blocks.append(block)
            self.queued += len(block)
            if self.queued > self.maximum:            # clock drift: trim
                while self.queued > self.target:
                    old = self.blocks.popleft()
                    self.queued -= len(old)
                    self.trimmed_ms += 1000 * len(old) / RATE

    def pull(self, out):
        """Fill `out` for the sound card. Called by sounddevice, on its own thread."""
        n = len(out)
        with self.lock:
            if not self.started:
                if self.queued < self.target:
                    out[:] = 0
                    return
                self.started = True
            i = 0
            while i < n and self.blocks:
                b = self.blocks[0]
                k = min(n - i, len(b))
                out[i:i + k] = b[:k]
                i += k
                if k == len(b):
                    self.blocks.popleft()
                else:
                    self.blocks[0] = b[k:]
                self.queued -= k
            if i < n:                                  # ran dry
                out[i:] = 0
                self.underruns += 1
                self.started = False                   # refill the cushion

    def buffered_ms(self):
        return 1000 * self.queued / RATE


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 0
    SAVE_PATH.parent.mkdir(exist_ok=True)

    sos = butter(4, HIGHPASS_HZ, "highpass", fs=RATE, output="sos")
    zi = None
    gain = 10 ** (PLAYBACK_GAIN_DB / 20)
    play = Playback()
    recorded = []

    port = serial.Serial()
    port.port, port.baudrate, port.timeout = PORT, BAUD, 0.05
    port.dtr = port.rts = False
    port.open()
    # The receiver streams all the time, so the port is already full of old
    # sound from before we opened it. Throw it away: playing it would start
    # us ~0.5 s behind (the first live session did exactly that).
    time.sleep(0.05)
    port.reset_input_buffer()

    def callback(outdata, frames, t, status):
        play.pull(outdata[:, 0])

    stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="float32",
                             blocksize=160, latency="low", callback=callback)
    stream.start()
    print(f"Listening on {PORT}, playing on: "
          f"{sd.query_devices(kind='output')['name']}.  Ctrl+C to stop.")
    print("USE HEADPHONES -- laptop speakers will make the mic howl.\n")

    buf = b""
    got = lost = damaged = restarts = 0
    expected = None
    last = np.zeros(120, np.float32)
    t0 = time.time()
    next_status = t0 + STATUS_EVERY_S
    try:
        while not seconds or time.time() - t0 < seconds:
            buf += port.read(4096)
            while True:
                i = buf.find(MAGIC)
                if i < 0:
                    buf = buf[-3:]
                    break
                frame = buf[i:]
                if len(frame) < 4 + HEAD.size:
                    buf = frame
                    break
                seq, n = HEAD.unpack_from(frame, 4)
                total = 4 + HEAD.size + 2 * n + 2
                if n == 0 or n > 1024:
                    buf = frame[4:]
                    continue
                if len(frame) < total:
                    buf = frame
                    break
                raw = frame[4 + HEAD.size: 4 + HEAD.size + 2 * n]
                (checksum,) = struct.unpack_from("<H", frame, total - 2)
                buf = frame[total:]
                if (sum(raw) & 0xFFFF) != checksum:
                    damaged += 1
                    continue

                # Lost packets: fill the gap with the last good packet played
                # BACKWARDS and fading to silence. Backwards, because its
                # first sample is then the same as the sample just played, so
                # there is no jump; fading, so it does not buzz. Much softer
                # on the ear than dropping straight to silence with a click.
                if expected is not None and seq != expected:
                    gap = (seq - expected) & 0xFFFFFFFF
                    if gap <= MAX_FILL_PACKETS:
                        lost += gap
                        fade = last[::-1] * np.linspace(1, 0, len(last), dtype=np.float32)
                        for _ in range(gap):
                            play.push(fade)
                            recorded.append(fade)
                            fade = np.zeros_like(last)
                    else:
                        restarts += 1
                expected = seq + 1

                x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                if zi is None:
                    zi = sosfilt_zi(sos) * x[0]
                x, zi = sosfilt(sos, x, zi=zi)
                x = np.clip(on_audio(x.astype(np.float32)) * gain, -1, 1).astype(np.float32)
                play.push(x)
                recorded.append(x)
                last = x
                got += 1

            if time.time() >= next_status:
                next_status += STATUS_EVERY_S
                tot = got + lost
                print(f"  {time.time() - t0:5.1f}s  packets {got:6d}  lost {lost:4d} "
                      f"({100 * lost / max(tot, 1):4.1f}%)  damaged {damaged}  "
                      f"buffer {play.buffered_ms():4.0f} ms  underruns {play.underruns}")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        port.close()

    if not recorded:
        sys.exit("\nNo audio arrived. Are tx_live.ino (COM9) and rx_live.ino "
                 "(COM10) uploaded, and the Serial Monitor closed?")

    audio = np.concatenate(recorded)
    with wave.open(str(SAVE_PATH), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes((audio * 32767).astype("<i2").tobytes())
    tot = got + lost
    print(f"\nStopped. {got} packets received, {lost} missing "
          f"({100 * lost / max(tot, 1):.2f}%; includes the {damaged} that "
          f"arrived damaged on USB and were dropped), "
          f"{play.underruns} buffer underruns, {play.trimmed_ms:.0f} ms trimmed "
          f"for clock drift, {restarts} transmitter restart(s).")
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    print(f"Clipped: {100 * clipped:.2f}% of samples"
          + ("  -- lower PLAYBACK_GAIN_DB" if clipped > 0.001 else "  (fine)"))
    print(f"Saved {len(audio) / RATE:.1f} s to {SAVE_PATH}")


if __name__ == "__main__":
    main()
