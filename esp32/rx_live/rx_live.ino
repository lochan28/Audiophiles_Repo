// ============================================================================
// rx_live.ino  --  LIVE RECEIVER.  Upload to COM10.
//
// What it does, in plain words:
//   Catches the transmitter's sound packets and passes each one straight
//   down the USB cable to the laptop (pc_live.py), which plays it. Later the
//   same cable goes to the Pi, and the Pi cleans the sound with our model
//   before playing it.
//
//   The radio hands packets over at awkward moments, so they first go into a
//   short waiting line (a "queue"); the main loop takes them out in order
//   and writes them to USB. If USB ever falls behind and the line fills up,
//   the newest packet is dropped (the laptop sees it as a lost packet).
//
// Nothing needs to be wired -- just the USB cable.
// Close the Arduino Serial Monitor before running pc_live.py.
// ============================================================================

#include <WiFi.h>
#include <esp_now.h>
#include <esp_mac.h>
#include <esp_arduino_version.h>

// ============================================================================
// TUNABLE -- safe to change
// ============================================================================

// Length of the waiting line, in packets (each = 7.5 ms of sound). 32 holds
// a quarter of a second -- far more than USB ever needs. It does not add
// delay: packets leave as soon as USB takes them.
const int QUEUE_PACKETS = 32;

// ============================================================================
// FIXED -- must match tx_live.ino and pc_live.py
// ============================================================================

const uint32_t USB_BAUD = 921600;
const int SAMPLES_PER_PACKET = 120;

struct __attribute__((packed)) LivePacket {   // IDENTICAL to tx_live.ino
  uint32_t seq;
  int16_t  audio[SAMPLES_PER_PACKET];
};

// USB frame -- the SAME format mic_usb.ino uses, so pc_mic.py can read it too:
//   "AUD0", seq (4 bytes), n (2 bytes), n x int16 audio, checksum (2 bytes)
// seq is the transmitter's own number, so a gap on the laptop means a
// packet was lost over the radio (or in this queue).

// ============================================================================

QueueHandle_t queue;

#if ESP_ARDUINO_VERSION_MAJOR >= 3
void onReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
#else
void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
#endif
  if (len != sizeof(LivePacket)) return;   // not one of our packets
  // 0 = never wait. If the line is full the packet is dropped; the laptop
  // still notices, because that packet's seq number never arrives.
  xQueueSend(queue, data, 0);
}

void setup() {
  Serial.begin(USB_BAUD);
  delay(300);
  queue = xQueueCreate(QUEUE_PACKETS, sizeof(LivePacket));

  WiFi.mode(WIFI_STA);
  uint8_t mac[6];
  esp_read_mac(mac, ESP_MAC_WIFI_STA);
  Serial.printf("\nLive receiver %02X:%02X:%02X:%02X:%02X:%02X\n",
                mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW failed to start -- press EN to retry");
    while (true) delay(1000);
  }
  esp_now_register_recv_cb(onReceive);
  Serial.println("Waiting for live audio...");
}

void loop() {
  LivePacket p;
  if (xQueueReceive(queue, &p, portMAX_DELAY) != pdTRUE) return;  // sleeps until one arrives

  uint16_t n = SAMPLES_PER_PACKET;
  uint16_t checksum = 0;
  const uint8_t *b = (const uint8_t *)p.audio;
  for (size_t i = 0; i < sizeof(p.audio); i++) checksum += b[i];

  Serial.write((const uint8_t *)"AUD0", 4);
  Serial.write((const uint8_t *)&p.seq, 4);
  Serial.write((const uint8_t *)&n, 2);
  Serial.write((const uint8_t *)p.audio, sizeof(p.audio));
  Serial.write((const uint8_t *)&checksum, 2);
}
