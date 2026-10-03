// ============================================================================
// tx_clip.ino  --  TRANSMITTER.  Upload this to the board on COM9.
//
// What it does, in plain words:
//   A 3-second voice clip is stored inside this ESP32 (clip.h, made by
//   make_clip.py). Every few seconds the ESP32 cuts the clip into small
//   pieces ("packets") and sends them one by one over ESP-NOW -- the ESP32's
//   own radio, no Wi-Fi router needed -- to the receiver, which plays it.
//
// Why pieces: one ESP-NOW message can carry at most 250 bytes, and the clip
// is 48,000 bytes. So it goes as ~200 numbered packets, and the receiver
// puts them back in order.
//
// Watch it work: Arduino IDE -> Tools -> Serial Monitor, 115200 baud.
// ============================================================================

#include <WiFi.h>
#include <esp_now.h>
#include <esp_mac.h>
#include "clip.h"

// ============================================================================
// TUNABLE -- safe to change
// ============================================================================

// The RECEIVER's radio address. This one is the board that was on COM10.
// If you swap boards, the receiver prints its own address when it starts --
// copy that here.
uint8_t RECEIVER_MAC[6] = {0x84, 0x1F, 0xE8, 0x1A, 0x9D, 0x28};

// Pause between two sends of the clip, in milliseconds. The receiver needs
// ~3 s to play it, so keep this above ~4000 or clips will overlap.
const uint32_t REPEAT_EVERY_MS = 5000;

// Pause between packets, in milliseconds. The radio has a small send queue;
// 0 fills it and packets get refused. 2-5 is fine. The whole clip still goes
// across in well under a second, much faster than it takes to play.
const uint32_t GAP_BETWEEN_PACKETS_MS = 3;

// ============================================================================
// FIXED -- must be IDENTICAL in rx_clip.ino, or the receiver misreads packets
// ============================================================================

const int AUDIO_BYTES_PER_PACKET = 240;

// One radio message. "packed" = no hidden padding bytes, so both boards lay
// the fields out exactly the same way. Total 249 bytes (limit is 250).
struct __attribute__((packed)) Packet {
  uint8_t  clipId;        // changes every send, so the receiver knows a new clip started
  uint16_t index;         // which piece this is: 0, 1, 2, ...
  uint16_t count;         // how many pieces the whole clip has
  uint32_t totalSamples;  // clip length, so the receiver knows where it ends
  uint16_t sampleRate;    // how fast to play it (16000 per second)
  uint8_t  audio[AUDIO_BYTES_PER_PACKET];
};

// ============================================================================

uint8_t clipId = 0;

void setup() {
  Serial.begin(115200);
  delay(500);
  pinMode(2,OUTPUT);

  // ESP-NOW needs the radio switched on in "station" mode. We never connect
  // to a router -- this only powers the radio up.
  WiFi.mode(WIFI_STA);
  // Read the address straight from the chip -- WiFi.macAddress() can print
  // 00:00:00:00:00:00 on newer ESP32 Arduino versions this early on.
  uint8_t mac[6];
  esp_read_mac(mac, ESP_MAC_WIFI_STA);
  Serial.printf("Transmitter address: %02X:%02X:%02X:%02X:%02X:%02X\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW failed to start -- press the EN/RST button to retry");
    while (true) delay(1000);
  }

  // Tell ESP-NOW about the receiver. Sending to one known address (instead
  // of broadcasting to everyone) means the radio automatically re-sends a
  // packet if the receiver did not confirm it -- fewer gaps in the audio.
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, RECEIVER_MAC, 6);
  peer.channel = 0;         // 0 = whatever channel the radio is on now
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("Could not add the receiver -- check RECEIVER_MAC");
    while (true) delay(1000);
  }

  Serial.printf("Clip: %u samples at %u Hz = %.2f s\n",
                (unsigned)CLIP_LEN, (unsigned)CLIP_RATE, CLIP_LEN / (float)CLIP_RATE);
}

void loop() {
  digitalWrite(2, HIGH);
  const uint16_t count =
      (CLIP_LEN + AUDIO_BYTES_PER_PACKET - 1) / AUDIO_BYTES_PER_PACKET;  // round up
  clipId++;

  Packet p;
  p.clipId = clipId;
  p.count = count;
  p.totalSamples = CLIP_LEN;
  p.sampleRate = CLIP_RATE;

  uint32_t started = millis();
  int refused = 0;

  for (uint16_t i = 0; i < count; i++) {
    uint32_t offset = (uint32_t)i * AUDIO_BYTES_PER_PACKET;
    uint32_t n = min((uint32_t)AUDIO_BYTES_PER_PACKET, CLIP_LEN - offset);

    p.index = i;
    memset(p.audio, 128, sizeof(p.audio));    // 128 = silence, pads the last piece
    memcpy(p.audio, CLIP + offset, n);

    // If the send queue is full, wait a moment and try again (up to 20 times).
    int tries = 0;
    while (esp_now_send(RECEIVER_MAC, (uint8_t *)&p, sizeof(p)) != ESP_OK) {
      if (++tries > 20) { refused++; break; }
      delay(2);
    }
    delay(GAP_BETWEEN_PACKETS_MS);
  }

  Serial.printf("Clip %u sent: %u packets in %lu ms (%d refused by the radio)\n",
                clipId, count, millis() - started, refused);

  delay(REPEAT_EVERY_MS);
}
