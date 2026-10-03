// ============================================================================
// rx_speaker.ino  --  RECEIVER that plays through the SPEAKER.  Upload to COM10.
//
// What it does, in plain words:
//   Catches the transmitter's live sound packets (tx_live.ino on COM9) and
//   plays them straight out of the MAX98357A amplifier and speaker. No
//   laptop needed -- this only proves the amp and speaker work. There is NO
//   noise cleaning here: the model cannot run on an ESP32. That comes next,
//   when the laptop/Pi cleans the sound and sends it back to this board.
//
//   Packets wait in a short line (~60 ms) before playing, so uneven radio
//   timing does not cause stutters -- the same idea as pc_live.py.
//
// WIRING (MAX98357A -> ESP32):
//   Vin  -> VIN (5V)     GND -> GND
//   LRC  -> GPIO 25      BCLK -> GPIO 26      DIN -> GPIO 22
//   GAIN, SD -> leave unconnected
//   Speaker -> the amp's + and - holes ONLY (never to GND or the ESP32)
//
// FEEDBACK: if the speaker is near the mic, the mic hears it and the system
// howls. Keep them apart, or turn VOLUME down.
//
// Serial Monitor (115200) prints a status line every 2 seconds.
// ============================================================================

#include <WiFi.h>
#include <esp_now.h>
#include <esp_mac.h>
#include <esp_arduino_version.h>
#include <ESP_I2S.h>

// ============================================================================
// TUNABLE -- safe to change
// ============================================================================

// Loudness. 1.0 = as received, 2.0 = twice as loud (+6 dB). Turn it down if
// the status line shows many "limited" samples or if it howls.
const float VOLUME = 2.0f;

// Sound to collect before playing starts, in packets (1 packet = 7.5 ms).
// 8 = 60 ms. More = smoother on a bad radio link, but more delay.
const int START_AFTER_PACKETS = 8;

// If more than this many packets are waiting, drop the oldest ones until
// only START_AFTER_PACKETS are left. This happens once at start-up (a burst
// piles up while the board boots), and very rarely after that, because the
// two ESP32s' clocks tick at very slightly different speeds. 16 = 120 ms.
// (The first version dropped just ONE packet per check, so a start-up pile
// of 23 packets never shrank: ~170 ms of extra delay, for good.)
const int TRIM_ABOVE_PACKETS = 16;

// Remove sound below this frequency. The mic drifts slowly (a few Hz),
// which a speaker cannot play -- it would only heat the speaker coil and
// waste battery. Speech has almost nothing under 80 Hz.
const float HIGHPASS_HZ = 80.0f;

const uint32_t STATUS_EVERY_MS = 2000;

// ============================================================================
// SAFETY -- read before changing
// ============================================================================

// Hard ceiling on output level, as a fraction of the amp's maximum. The
// speaker is 4 ohm / 2 W, and at 5 V this amp can push ~3 W into 4 ohm.
// 0.70 (-3 dB) keeps it at ~1.5 W or below. Whatever VOLUME says, no
// sample is ever played louder than this. Only raise it for an 8 ohm speaker.
const float OUTPUT_CEILING = 0.70f;

// ============================================================================
// FIXED -- must match tx_live.ino
// ============================================================================

const int PIN_BCLK = 26, PIN_LRC = 25, PIN_DIN = 22;
const uint32_t SAMPLE_RATE = 16000;
const int SAMPLES_PER_PACKET = 120;
const int QUEUE_PACKETS = 32;

struct __attribute__((packed)) LivePacket {   // IDENTICAL to tx_live.ino
  uint32_t seq;
  int16_t  audio[SAMPLES_PER_PACKET];
};

// ============================================================================

I2SClass i2s;
QueueHandle_t queue;

// The amp mixes left and right, (L+R)/2. Sending the same sample on both
// sides makes that mix exactly the sample -- full level, whatever the board.
int16_t stereo[SAMPLES_PER_PACKET * 2];

bool     playing = false;
uint32_t expectedSeq = 0;
bool     haveSeq = false;
uint32_t received = 0, lost = 0, underruns = 0, trimmed = 0, limited = 0;
uint32_t lastStatus = 0;

// High-pass filter memory (one-pole: removes slow drift and DC).
float hpA, hpPrevIn = 0, hpPrevOut = 0;

#if ESP_ARDUINO_VERSION_MAJOR >= 3
void onReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
#else
void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
#endif
  if (len != sizeof(LivePacket)) return;
  xQueueSend(queue, data, 0);          // full line = packet dropped
}

void playPacket(const LivePacket &p) {
  const int16_t ceiling = (int16_t)(OUTPUT_CEILING * 32767);
  for (int i = 0; i < SAMPLES_PER_PACKET; i++) {
    float x = p.audio[i];
    float y = hpA * (hpPrevOut + x - hpPrevIn);     // high-pass
    hpPrevIn = x;
    hpPrevOut = y;
    float v = y * VOLUME;
    if (v > ceiling)  { v = ceiling;  limited++; }
    if (v < -ceiling) { v = -ceiling; limited++; }
    stereo[2 * i] = stereo[2 * i + 1] = (int16_t)v;
  }
  i2s.write((uint8_t *)stereo, sizeof(stereo));     // waits for room: this sets the pace
}

void playSilence() {
  memset(stereo, 0, sizeof(stereo));
  i2s.write((uint8_t *)stereo, sizeof(stereo));
}

void setup() {
  Serial.begin(115200);
  delay(300);
  queue = xQueueCreate(QUEUE_PACKETS, sizeof(LivePacket));
  hpA = 1.0f / (1.0f + 2.0f * PI * HIGHPASS_HZ / SAMPLE_RATE);

  i2s.setPins(PIN_BCLK, PIN_LRC, PIN_DIN);
  if (!i2s.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT,
                 I2S_SLOT_MODE_STEREO)) {
    Serial.println("I2S (amp) failed to start");
    while (true) delay(1000);
  }
  playSilence();

  WiFi.mode(WIFI_STA);
  uint8_t mac[6];
  esp_read_mac(mac, ESP_MAC_WIFI_STA);
  Serial.printf("\nSpeaker receiver %02X:%02X:%02X:%02X:%02X:%02X\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW failed to start -- press EN to retry");
    while (true) delay(1000);
  }
  esp_now_register_recv_cb(onReceive);
  Serial.printf("Waiting for live audio... (volume %.1f, ceiling %.0f%%)\n",
                VOLUME, OUTPUT_CEILING * 100);
}

void loop() {
  LivePacket p;
  int waiting = uxQueueMessagesWaiting(queue);

  if (!playing) {
    // Fill the cushion first. Meanwhile keep the amp fed with silence so
    // it does not pop when the sound starts.
    if (waiting >= START_AFTER_PACKETS) playing = true;
    else playSilence();
  } else if (waiting == 0) {
    underruns++;                 // ran dry: play silence and refill
    playing = false;
    playSilence();
  } else {
    if (waiting > TRIM_ABOVE_PACKETS) {         // too far behind: catch up
      while (uxQueueMessagesWaiting(queue) > START_AFTER_PACKETS) {
        xQueueReceive(queue, &p, 0);
        trimmed++;
      }
      haveSeq = false;      // these were dropped on purpose -- not "lost"
    }
    xQueueReceive(queue, &p, 0);
    if (haveSeq && p.seq != expectedSeq) {
      uint32_t gap = p.seq - expectedSeq;
      if (gap < 1000) lost += gap;              // bigger = transmitter restarted
    }
    expectedSeq = p.seq + 1;
    haveSeq = true;
    received++;
    playPacket(p);
  }

  if (millis() - lastStatus >= STATUS_EVERY_MS) {
    lastStatus = millis();
    Serial.printf("received %lu  lost %lu  underruns %lu  trimmed %lu  "
                  "limited samples %lu  waiting %d\n",
                  (unsigned long)received, (unsigned long)lost,
                  (unsigned long)underruns, (unsigned long)trimmed,
                  (unsigned long)limited, waiting);
  }
}
