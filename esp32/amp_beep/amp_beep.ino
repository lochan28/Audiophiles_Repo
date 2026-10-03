// ============================================================================
// amp_beep.ino  --  SPEAKER TEST.  Upload to the board the amp is wired to.
//
// What it does, in plain words:
//   Beeps: half a second of tone, half a second of silence, forever. No
//   radio, no mic -- only ESP32 -> MAX98357A amp -> speaker. If you hear the
//   beep, the amp side is wired correctly and any silence elsewhere is a
//   code or radio problem. If you hear nothing, it is wiring or power.
//
// WIRING (same as rx_speaker.ino):
//   Vin -> VIN (5V)   GND -> GND   LRC -> 25   BCLK -> 26   DIN -> 22
//   GAIN, SD -> unconnected.   Speaker -> amp + and - (soldered).
//
// Serial Monitor (115200) says "beep" each time, so you know it is running.
// ============================================================================

#include <ESP_I2S.h>

// ============================================================================
// TUNABLE -- safe to change
// ============================================================================

const float TONE_HZ = 440.0f;     // the note A; 1000 is more piercing
const float LEVEL   = 0.30f;      // 0.30 = 30% of full -- clearly audible, safe

// ============================================================================
// FIXED
// ============================================================================

const int PIN_BCLK = 26, PIN_LRC = 25, PIN_DIN = 22;
const uint32_t SAMPLE_RATE = 16000;

I2SClass i2s;
int16_t block[160 * 2];           // 10 ms, left+right
float phase = 0;

void writeBlock(bool sound) {
  for (int i = 0; i < 160; i++) {
    int16_t v = 0;
    if (sound) {
      v = (int16_t)(LEVEL * 32767 * sinf(phase));
      phase += 2 * PI * TONE_HZ / SAMPLE_RATE;
      if (phase > 2 * PI) phase -= 2 * PI;
    }
    block[2 * i] = block[2 * i + 1] = v;    // same on both sides
  }
  i2s.write((uint8_t *)block, sizeof(block));
}

void setup() {
  Serial.begin(115200);
  delay(300);
  i2s.setPins(PIN_BCLK, PIN_LRC, PIN_DIN);
  if (!i2s.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT,
                 I2S_SLOT_MODE_STEREO)) {
    Serial.println("I2S failed to start");
    while (true) delay(1000);
  }
  Serial.println("\nAmp beep test running.");
}

void loop() {
  Serial.println("beep");
  for (int i = 0; i < 50; i++) writeBlock(true);    // 0.5 s tone
  for (int i = 0; i < 50; i++) writeBlock(false);   // 0.5 s silence
}
