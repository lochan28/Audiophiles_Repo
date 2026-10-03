// ============================================================================
// rx_usb.ino  --  RECEIVER that plays through the LAPTOP.  Upload to COM10.
//
// What it does, in plain words:
//   Same radio side as rx_clip.ino: catch the transmitter's numbered packets
//   and put the clip back together. But instead of playing it out of pin 25,
//   it sends the finished clip down the USB cable to the laptop, where
//   pc_listen.py plays it through the laptop's speakers and saves a WAV.
//
//   This is the same path the real demo uses: later this cable goes into the
//   Pi instead of the laptop, and the Pi cleans the audio with our model.
//
// Nothing needs to be wired to this board -- just the USB cable.
//
// IMPORTANT: close the Arduino Serial Monitor before running pc_listen.py.
// Only one program can use the USB port at a time.
// ============================================================================

#include <WiFi.h>
#include <esp_now.h>
#include <esp_mac.h>
#include <esp_arduino_version.h>

// ============================================================================
// TUNABLE -- safe to change
// ============================================================================

// How long without a packet before the clip counts as finished, in ms.
// The transmitter sends a clip in under a second, then waits 5 s.
const uint32_t END_OF_CLIP_SILENCE_MS = 300;

// Longest clip we can hold, in seconds (16 kHz, 1 byte per sample).
const uint32_t MAX_SECONDS_ON_RECEIVER = 4;

// ============================================================================
// FIXED -- must match the other side, or the data is misread
// ============================================================================

// USB speed. pc_listen.py uses the same number. 921600 moves the 48 KB clip
// in about half a second; the default 115200 would take over 4 seconds.
const uint32_t USB_BAUD = 921600;

// Packet layout -- IDENTICAL to tx_clip.ino.
const int AUDIO_BYTES_PER_PACKET = 240;
struct __attribute__((packed)) Packet {
  uint8_t  clipId;
  uint16_t index;
  uint16_t count;
  uint32_t totalSamples;
  uint16_t sampleRate;
  uint8_t  audio[AUDIO_BYTES_PER_PACKET];
};

// What goes down the USB cable for each clip -- IDENTICAL to pc_listen.py:
//   "CLIP"            4 bytes, marks the start so the laptop can find it
//   sampleRate        4 bytes
//   numSamples        4 bytes
//   packetsReceived   2 bytes
//   packetsExpected   2 bytes
//   audio             numSamples bytes (8-bit, 128 = silence)
//   checksum          4 bytes, sum of all audio bytes -- catches USB errors
// Everything else on the port is plain text log lines.

// ============================================================================

const uint32_t MAX_SAMPLES = MAX_SECONDS_ON_RECEIVER * 16000;
const uint32_t MAX_PACKETS = MAX_SAMPLES / AUDIO_BYTES_PER_PACKET + 1;

uint8_t clip[MAX_SAMPLES];
bool    gotPacket[MAX_PACKETS];

volatile bool     receiving     = false;
volatile bool     sending       = false;
volatile uint8_t  currentClipId = 0;
volatile uint16_t expectedCount = 0;
volatile uint16_t receivedCount = 0;
volatile uint32_t clipSamples   = 0;
volatile uint16_t clipRate      = 16000;
volatile uint32_t lastPacketMs  = 0;

#if ESP_ARDUINO_VERSION_MAJOR >= 3
void onReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
#else
void onReceive(const uint8_t *mac, const uint8_t *data, int len) {
#endif
  if (sending || len != sizeof(Packet)) return;
  const Packet *p = (const Packet *)data;
  if (p->count > MAX_PACKETS || p->totalSamples > MAX_SAMPLES) return;

  if (!receiving || p->clipId != currentClipId) {
    memset(clip, 128, sizeof(clip));
    memset(gotPacket, 0, sizeof(gotPacket));
    currentClipId = p->clipId;
    expectedCount = p->count;
    clipSamples   = p->totalSamples;
    clipRate      = p->sampleRate;
    receivedCount = 0;
    lastPacketMs  = millis();   // before receiving=true -- see rx_clip.ino
    receiving     = true;
  }

  if (p->index >= expectedCount || gotPacket[p->index]) return;
  uint32_t offset = (uint32_t)p->index * AUDIO_BYTES_PER_PACKET;
  if (offset >= clipSamples) return;
  uint32_t n = min((uint32_t)AUDIO_BYTES_PER_PACKET, clipSamples - offset);
  memcpy(clip + offset, p->audio, n);
  gotPacket[p->index] = true;
  receivedCount++;
  lastPacketMs = millis();
}

// Little helpers: write a number as raw bytes, lowest byte first -- the
// order pc_listen.py expects.
void writeU32(uint32_t v) { Serial.write((uint8_t *)&v, 4); }
void writeU16(uint16_t v) { Serial.write((uint8_t *)&v, 2); }

void sendClipToLaptop() {
  const uint32_t n = clipSamples;
  uint32_t checksum = 0;
  for (uint32_t i = 0; i < n; i++) checksum += clip[i];

  Serial.write((const uint8_t *)"CLIP", 4);
  writeU32(clipRate);
  writeU32(n);
  writeU16(receivedCount);
  writeU16(expectedCount);
  Serial.write(clip, n);
  writeU32(checksum);
  Serial.flush();                  // wait until every byte has left
}

void setup() {
  Serial.begin(USB_BAUD);
  delay(500);

  WiFi.mode(WIFI_STA);
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
  if (receiving && millis() - lastPacketMs > END_OF_CLIP_SILENCE_MS) {
    sending = true;              // radio ignores packets while we send
    receiving = false;

    Serial.printf("Clip %u: %u of %u packets arrived. Sending to laptop...\n",
                  currentClipId, receivedCount, expectedCount);
    sendClipToLaptop();
    Serial.println("\nSent. Waiting for the next clip...");
    sending = false;
  }
}
