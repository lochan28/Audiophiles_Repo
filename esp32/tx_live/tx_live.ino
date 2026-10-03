// ============================================================================
// tx_live.ino  --  LIVE TRANSMITTER.  Upload to the board with the mic (COM9).
//
// What it does, in plain words:
//   Listens to the INMP441 mic and, every 7.5 milliseconds, sends the newest
//   slice of sound over ESP-NOW to the receiver. Each slice carries a number
//   that counts up, so the receiver (and the laptop) can tell when a slice
//   went missing on the way.
//
//   This is the transmitter half of the real demo: mic -> radio -> receiver
//   -> laptop/Pi -> our noise-cancelling model -> headphones.
//
// WIRING: the INMP441 exactly as for mic_usb.ino (VDD->3V3, GND->GND,
// L/R->GND, WS->25, SCK->33, SD->32). The LED on GPIO 2 blinks while sending.
//
// Serial Monitor (115200) prints a status line every 2 seconds.
// ============================================================================

#include <WiFi.h>
#include <esp_now.h>
#include <esp_mac.h>
#include <ESP_I2S.h>

// ============================================================================
// TUNABLE -- safe to change
// ============================================================================

// The RECEIVER's radio address (the board on COM10). The receiver prints its
// own address at start-up if you ever need to check it.
uint8_t RECEIVER_MAC[6] = {0x84, 0x1F, 0xE8, 0x1A, 0x9D, 0x28};

// Digital volume boost in powers of two. 2 (= x4, +12 dB) was measured as a
// good level for speech at ~30 cm; 3 starts clipping on claps and shouts.
const int GAIN_SHIFT = 2;

// How often the status line prints, in ms.
const uint32_t STATUS_EVERY_MS = 2000;

// ============================================================================
// FIXED -- must be IDENTICAL in rx_live.ino
// ============================================================================

const int PIN_SCK = 33, PIN_WS = 25, PIN_SD = 32;
const int LED_PIN = 2;
const uint32_t SAMPLE_RATE = 16000;

// 120 samples x 2 bytes = 240 bytes of sound per radio packet (the limit is
// 250). 120 samples = 7.5 ms, so 133 packets per second.
const int SAMPLES_PER_PACKET = 120;

struct __attribute__((packed)) LivePacket {
  uint32_t seq;                          // counts up by one per packet
  int16_t  audio[SAMPLES_PER_PACKET];    // signed 16-bit sound
};                                       // 244 bytes

// ============================================================================

I2SClass i2s;
int32_t raw[SAMPLES_PER_PACKET * 2];     // mic gives left+right pairs
LivePacket pkt;

uint32_t sent = 0, refused = 0, lastStatus = 0;

void setup() {
  Serial.begin(115200);
  delay(300);
  pinMode(LED_PIN, OUTPUT);

  WiFi.mode(WIFI_STA);
  uint8_t mac[6];
  esp_read_mac(mac, ESP_MAC_WIFI_STA);
  Serial.printf("\nLive transmitter %02X:%02X:%02X:%02X:%02X:%02X\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW failed to start -- press EN to retry");
    while (true) delay(1000);
  }
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, RECEIVER_MAC, 6);
  peer.channel = 0;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("Could not add the receiver -- check RECEIVER_MAC");
    while (true) delay(1000);
  }

  i2s.setPins(PIN_SCK, PIN_WS, -1, PIN_SD);
  if (!i2s.begin(I2S_MODE_STD, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_32BIT,
                 I2S_SLOT_MODE_STEREO)) {
    Serial.println("I2S (mic) failed to start");
    while (true) delay(1000);
  }
  Serial.println("Sending live audio...");
}

void loop() {
  // Wait for the next 7.5 ms of sound. This read is what sets the pace:
  // the mic delivers exactly 16,000 samples a second, so we send exactly
  // 133.3 packets a second -- no timer needed.
  size_t want = sizeof(raw);
  if (i2s.readBytes((char *)raw, want) != want) return;

  for (int i = 0; i < SAMPLES_PER_PACKET; i++) {
    int32_t v = raw[2 * i] >> (16 - GAIN_SHIFT);   // left channel, 32 -> 16 bit
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    pkt.audio[i] = (int16_t)v;
  }

  // Send and move on. If the radio's queue is full we DROP this packet
  // rather than wait: waiting would make us fall behind the mic, and in
  // live audio a short gap is better than a growing delay. The seq number
  // still goes up, so the receiver knows exactly what was lost.
  if (esp_now_send(RECEIVER_MAC, (uint8_t *)&pkt, sizeof(pkt)) == ESP_OK) sent++;
  else refused++;
  pkt.seq++;

  digitalWrite(LED_PIN, (pkt.seq / 33) & 1);       // blink ~2x per second

  if (millis() - lastStatus >= STATUS_EVERY_MS) {
    lastStatus = millis();
    Serial.printf("sent %lu packets, %lu refused by the radio queue\n",
                  (unsigned long)sent, (unsigned long)refused);
  }
}
