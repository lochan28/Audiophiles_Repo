// ============================================================================
// mic_usb.ino  --  MIC TEST.  Upload to the board the INMP441 is wired to
//                  (the transmitter, COM9).
//
// What it does, in plain words:
//   Listens to the INMP441 microphone 16,000 times a second and sends the
//   sound straight down the USB cable to the laptop, where pc_mic.py saves
//   it, measures how loud it is, and plays it back. No radio involved --
//   this only proves the mic and its wiring work.
//
//   At start-up it also checks the wiring: it listens on both the "left"
//   and "right" channel for half a second and reports which one has sound.
//   The INMP441 talks on the left when its L/R pin is tied to GND.
//
// WIRING (INMP441 -> ESP32):
//   VDD -> 3V3      (NOT 5V / VIN -- 5 V can destroy the mic)
//   GND -> GND
//   L/R -> GND      (selects the left channel; do not leave it floating)
//   WS  -> GPIO 25
//   SCK -> GPIO 33
//   SD  -> GPIO 32
//
// Close the Arduino Serial Monitor before running pc_mic.py -- only one
// program can hold the USB port.
// ============================================================================

#include <ESP_I2S.h>

// ============================================================================
// TUNABLE -- safe to change
// ============================================================================

// Digital volume boost, in powers of two: 0 = as the mic hears it, 1 = x2
// (+6 dB), 2 = x4 (+12 dB), 3 = x8 (+18 dB). The INMP441 is a quiet mic, so
// speech from arm's length may come out low. Raise this if pc_mic.py says
// "quiet"; lower it if it reports clipping. Too high = loud sounds distort.
const int GAIN_SHIFT = 2;

// Throw away this much audio right after start-up, in ms. The mic needs a
// moment to wake up and its first samples are junk.
const uint32_t WARMUP_MS = 300;

// ============================================================================
// FIXED -- pc_mic.py depends on these
// ============================================================================

const int PIN_SCK = 33;
const int PIN_WS  = 25;
const int PIN_SD  = 32;

const uint32_t SAMPLE_RATE = 16000;   // matches our model
const uint32_t USB_BAUD    = 921600;  // 32 KB/s of audio fits easily
const int      BLOCK       = 320;     // samples per USB frame = 20 ms

// One USB frame -- IDENTICAL to pc_mic.py:
//   "AUD0"    4 bytes   marks the start of a frame
//   seq       4 bytes   counts up by one; a gap means a frame was lost
//   n         2 bytes   samples in this frame (BLOCK)
//   audio     n x 2 bytes, signed 16-bit
//   checksum  2 bytes   sum of the audio bytes -- catches USB errors

// ============================================================================

I2SClass i2s;
int32_t raw[BLOCK * 2];   // the mic sends left+right pairs, 32 bits each
int16_t out[BLOCK];
int     channel = 0;      // 0 = left, 1 = right -- chosen at start-up
uint32_t seq = 0;

// Read one block of left/right pairs from the mic.
bool readBlock() {
  size_t want = sizeof(raw);
  return i2s.readBytes((char *)raw, want) == want;
}

void setup() {
  Serial.begin(USB_BAUD);
  delay(300);
  Serial.println("\nMic test starting...");

  // The INMP441 sends 24-bit sound inside 32-bit slots. We read BOTH
  // channels so the wiring check below can see which one is alive.
  i2s.setPins(PIN_SCK, PIN_WS, -1, PIN_SD);   // -1: no speaker output here
  if (!i2s.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT,
                 I2S_SLOT_MODE_STEREO)) {
    Serial.println("I2S failed to start -- this is a code/board problem, not wiring");
    while (true) delay(1000);
  }

  uint32_t until = millis() + WARMUP_MS;
  while (millis() < until) readBlock();

  // Wiring check: average loudness on each channel over ~0.5 s. A dead
  // channel reads exactly 0 (or -1) every time; a live one wobbles even in a
  // quiet room, because the mic hears its own faint hiss.
  double level[2] = {0, 0};
  int32_t first[2] = {raw[0] >> 8, raw[1] >> 8};
  bool changes[2] = {false, false};
  const int blocks = 25;   // 25 x 20 ms = 0.5 s
  for (int b = 0; b < blocks; b++) {
    readBlock();
    for (int i = 0; i < BLOCK; i++) {
      for (int c = 0; c < 2; c++) {
        int32_t v = raw[2 * i + c] >> 8;            // 24-bit value
        level[c] += fabs((double)v);
        if (v != first[c]) changes[c] = true;
      }
    }
  }
  for (int c = 0; c < 2; c++) level[c] /= (double)blocks * BLOCK;

  Serial.printf("Wiring check: LEFT level %.0f%s, RIGHT level %.0f%s\n",
                level[0], changes[0] ? "" : " (stuck -- no signal)",
                level[1], changes[1] ? "" : " (stuck -- no signal)");
  if (!changes[0] && !changes[1]) {
    Serial.println("NO SIGNAL on either channel. Check VDD=3V3, GND, and the "
                   "SCK/WS/SD wires. Streaming anyway so you can see it.");
  }
  channel = (changes[1] && (!changes[0] || level[1] > level[0])) ? 1 : 0;
  Serial.printf("Using the %s channel.%s\n", channel ? "RIGHT" : "LEFT",
                channel ? " (Expected LEFT -- is L/R really tied to GND?)" : "");
  Serial.println("Streaming...");
  Serial.flush();
}

void loop() {
  if (!readBlock()) return;

  uint16_t checksum = 0;
  for (int i = 0; i < BLOCK; i++) {
    // 32-bit slot -> 16-bit sample, with the optional boost, clamped so a
    // loud sound clips cleanly instead of wrapping around into a crackle.
    int32_t v = raw[2 * i + channel] >> (16 - GAIN_SHIFT);
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    out[i] = (int16_t)v;
    const uint8_t *b = (const uint8_t *)&out[i];
    checksum += b[0] + b[1];
  }

  uint16_t n = BLOCK;
  Serial.write((const uint8_t *)"AUD0", 4);
  Serial.write((const uint8_t *)&seq, 4);
  Serial.write((const uint8_t *)&n, 2);
  Serial.write((const uint8_t *)out, sizeof(out));
  Serial.write((const uint8_t *)&checksum, 2);
  seq++;
}
