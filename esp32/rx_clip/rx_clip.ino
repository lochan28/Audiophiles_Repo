// ============================================================================
// rx_clip.ino  --  RECEIVER.  Upload this to the board on COM10.
//
// What it does, in plain words:
//   Listens on ESP-NOW for the transmitter's packets. Each packet is one
//   numbered piece of a voice clip; the receiver drops each piece into the
//   right place in memory. When the packets stop arriving, the clip is
//   complete, so it plays the whole thing out of pin 25 -- the ESP32's
//   built-in audio output (an 8-bit "DAC": it turns numbers into voltages).
//
//   Any piece that got lost on the way is left as silence, and the Serial
//   Monitor tells you how many were lost.
//
// WIRING -- do NOT connect a speaker straight to pin 25; the pin cannot
// supply enough current and it may be damaged. Use one of these:
//
//   Best:   GPIO25 -> amplifier input (PAM8403 module, or the AUX-in of any
//           powered speaker).  GND -> amplifier GND.
//
//   Quick:  GPIO25 -> 10 uF capacitor (+ leg at the ESP32) -> 220 ohm
//           resistor -> headphone tip.  GND -> headphone sleeve.
//           Quiet, but enough to hear it. Keep the volume LOW at first.
//
// Watch it work: Arduino IDE -> Tools -> Serial Monitor, 115200 baud.
// Sound quality: 8-bit, so expect "old telephone". That is the DAC, not the
// radio -- the audio arrives bit-for-bit what was sent.
// ============================================================================

#include <WiFi.h>
#include <esp_now.h>
#include <esp_mac.h>
#include <esp_arduino_version.h>

// ============================================================================
// TUNABLE -- safe to change
// ============================================================================

// Which pin plays the sound. On the classic ESP32 only 25 and 26 have a DAC.
const int SPEAKER_PIN = 25;

// How long without a packet before we decide the clip is finished and play
// it, in milliseconds. The transmitter sends the whole clip in under a
// second and then waits 5 s, so anything from ~200 to ~2000 works.
const uint32_t END_OF_CLIP_SILENCE_MS = 300;

// Longest clip we can hold, in seconds (16 kHz, 1 byte per sample).
// 4 s = 64 KB of RAM, which the ESP32 has plenty of. Raise it only if
// make_clip.py's SECONDS is raised too.
const uint32_t MAX_SECONDS_ON_RECEIVER = 4;

// ============================================================================
// FIXED -- must be IDENTICAL to tx_clip.ino, or packets are misread
// ============================================================================

const int AUDIO_BYTES_PER_PACKET = 240;

struct __attribute__((packed)) Packet {
  uint8_t  clipId;
  uint16_t index;
  uint16_t count;
  uint32_t totalSamples;
  uint16_t sampleRate;
  uint8_t  audio[AUDIO_BYTES_PER_PACKET];
};

// ============================================================================

const uint32_t MAX_SAMPLES = MAX_SECONDS_ON_RECEIVER * 16000;
const uint32_t MAX_PACKETS = MAX_SAMPLES / AUDIO_BYTES_PER_PACKET + 1;

uint8_t clip[MAX_SAMPLES];        // the clip is rebuilt here
bool    gotPacket[MAX_PACKETS];   // which pieces have arrived

// These are written by the radio (onReceive) and read by loop(), which run
// at the same time on different parts of the chip -- "volatile" tells the
// compiler to always re-read them instead of trusting an old copy.
volatile bool     receiving     = false;
volatile bool     playing       = false;
volatile uint8_t  currentClipId = 0;
volatile uint16_t expectedCount = 0;
volatile uint16_t receivedCount = 0;
volatile uint32_t clipSamples   = 0;
volatile uint16_t clipRate      = 16000;
volatile uint32_t lastPacketMs  = 0;

// Called by the radio every time a packet arrives. Keep it short: copy the
// audio into place and get out.
#if ESP_ARDUINO_VERSION_MAJOR >= 3
void onReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
#else
void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
#endif
  if (playing || len != sizeof(Packet)) return;   // busy, or not our packet
  const Packet *p = (const Packet *)data;
  if (p->count > MAX_PACKETS || p->totalSamples > MAX_SAMPLES) return;  // too long

  // A new clip id means the transmitter started a fresh send: wipe the
  // buffer to silence (128) and forget which pieces we had.
  if (!receiving || p->clipId != currentClipId) {
    memset(clip, 128, sizeof(clip));
    memset(gotPacket, 0, sizeof(gotPacket));
    currentClipId = p->clipId;
    expectedCount = p->count;
    clipSamples   = p->totalSamples;
    clipRate      = p->sampleRate;
    receivedCount = 0;
    lastPacketMs  = millis();   // set BEFORE receiving=true, or loop() could
    receiving     = true;       // see an old time and play an empty clip
  }

  if (p->index >= expectedCount || gotPacket[p->index]) return;  // bad or duplicate

  uint32_t offset = (uint32_t)p->index * AUDIO_BYTES_PER_PACKET;
  if (offset >= clipSamples) return;                               // past the end
  uint32_t n = min((uint32_t)AUDIO_BYTES_PER_PACKET, clipSamples - offset);
  memcpy(clip + offset, p->audio, n);
  gotPacket[p->index] = true;
  receivedCount++;
  lastPacketMs = millis();
}

// Plays the clip out of the DAC at exactly clipRate samples per second.
// micros() is the clock: sample i must come out at i/16000 s after the start.
void playClip() {
  const uint32_t n = clipSamples;
  const uint32_t rate = clipRate;
  const uint32_t t0 = micros();
  for (uint32_t i = 0; i < n; i++) {
    uint32_t due = (uint32_t)((uint64_t)i * 1000000ULL / rate);
    while ((uint32_t)(micros() - t0) < due) { }   // wait for this sample's moment
    dacWrite(SPEAKER_PIN, clip[i]);
  }
  dacWrite(SPEAKER_PIN, 128);                      // back to silence
}

void setup() {
  Serial.begin(115200);
  delay(500);
  dacWrite(SPEAKER_PIN, 128);                      // start silent, no click

  WiFi.mode(WIFI_STA);
  // Read the address straight from the chip -- WiFi.macAddress() can print
  // 00:00:00:00:00:00 on newer ESP32 Arduino versions this early on.
  uint8_t mac[6];
  esp_read_mac(mac, ESP_MAC_WIFI_STA);
  Serial.printf("Receiver address (put this in tx_clip.ino): %02X:%02X:%02X:%02X:%02X:%02X\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW failed to start -- press the EN/RST button to retry");
    while (true) delay(1000);
  }
  esp_now_register_recv_cb(onReceive);
  Serial.println("Waiting for a clip...");
}

void loop() {
  // Finished = we were receiving, and nothing has arrived for a while.
  if (receiving && millis() - lastPacketMs > END_OF_CLIP_SILENCE_MS) {
    playing = true;          // radio ignores packets until we are done
    receiving = false;

    uint16_t lost = expectedCount - receivedCount;
    Serial.printf("Clip %u: %u of %u packets arrived, %u lost (%.1f%%). Playing %.2f s...\n",
                  currentClipId, receivedCount, expectedCount, lost,
                  100.0f * lost / expectedCount, clipSamples / (float)clipRate);

    playClip();
    Serial.println("Done. Waiting for the next clip...");
    playing = false;
  }
}
