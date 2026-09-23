#include "FS.h"
#include "SD.h"
#include "WiFi.h"
#include "config.h"
#include "esp_camera.h"
#include <Arduino.h>
#include <WiFiAP.h>
#include <driver/i2s.h>

// --- นำเข้าไฟล์ทำงานของโมเดลมังคุด ---
#include "mangosteen_infer.h"
// --------------------------------

HardwareSerial SerialAT(1);
static bool mangosteen_ready = false;
static constexpr uint32_t AI_INTERVAL_MS = 250;
static constexpr uint32_t WIFI_CONNECT_TIMEOUT_MS = 10000;

void mic_init(void);
void check_sound(void);
void sd_test(void);
void wifi_scan_connect(void);
void pcie_test(void);
bool camera_test(void);
void startCameraServer();

void setup() {
    pinMode(PWR_ON_PIN, OUTPUT);
    digitalWrite(PWR_ON_PIN, HIGH);
    delay(100);
    Serial.begin(115200);
    Serial.println("T-SIMCAM self test");

#ifdef CAM_IR_PIN
    //Teset IR Filter
    pinMode(CAM_IR_PIN, OUTPUT);
    Serial.println("Test IR Filter");
    int i = 3;
    while (i--) {
        digitalWrite(CAM_IR_PIN, 1 - digitalRead(CAM_IR_PIN)); delay(1000);
    }
#endif

    sd_test();
    // pcie_test(); // <-- ปิดไว้ก่อนเพื่อแก้ปัญหาค้างตอน Waking up PCI module
    mic_init();

    // เปิด AP และเว็บกล้องก่อนเสมอ จึงเข้ากล้องได้แม้ Wi-Fi ภายนอกใช้ไม่ได้
    if (!camera_test()) {
        Serial.println("Camera unavailable; AI and camera web server are disabled.");
        return;
    }
    wifi_scan_connect(); // เชื่อม Wi-Fi ภายนอกแบบมี timeout โดยไม่ปิด AP

    // --- เริ่มต้น TFLite และโมเดล AI ---
    Serial.println("Initializing Mangosteen model...");
    if (!mangosteen_init()) {
        Serial.println("Mangosteen model init failed!");
    } else {
        Serial.println("Mangosteen model init SUCCESS!");
        mangosteen_ready = true;
    }
    // ---------------------------------

    check_sound();
}

void loop() {
    check_sound();
    delay(5);

    // วิเคราะห์ประมาณ 4 ครั้ง/วินาที; รอบจริงขึ้นกับเวลาที่ Invoke ใช้
    static uint32_t last_inference = 0;
    static uint32_t last_log = 0;
    if (mangosteen_ready && millis() - last_inference >= AI_INTERVAL_MS) {
        const uint32_t frame_started_ms = millis();
        camera_fb_t *fb = esp_camera_fb_get();
        if (fb) {
            float scores[MANGOSTEEN_CLASS_COUNT] = {0};
            uint32_t time_ms = 0;

            // โยนภาพเข้าโมเดล
            const uint32_t predict_started_ms = millis();
            int best_class = mangosteen_predict(fb, scores, MANGOSTEEN_CLASS_COUNT, &time_ms);
            const uint32_t predict_total_ms = millis() - predict_started_ms;
            const uint32_t frame_total_ms = millis() - frame_started_ms;
            
            if (best_class >= 0 && millis() - last_log >= 1000) {
                Serial.println("===================================");
                Serial.printf("Mangosteen Class: %s (%d)\n", mangosteen_class_label(best_class), best_class);
                Serial.printf("Confidence Score: %.2f\n", scores[best_class]);
                Serial.printf("Model Invoke Time: %u ms\n", time_ms);
                Serial.printf("AI Pipeline Time: %u ms\n", predict_total_ms);
                Serial.printf("Capture + AI Time: %u ms\n", frame_total_ms);
                Serial.println("===================================");
                last_log = millis();
            }

            esp_camera_fb_return(fb); // คืนค่าเมมโมรี่รูปภาพเสมอ
        }
        last_inference = millis();
    }
    // ------------------------------------
}

void sd_test(void) {
    SPI.begin(SD_SCLK_PIN, SD_MISO_PIN, SD_MOSI_PIN, SD_CS_PIN);
    if (!SD.begin(SD_CS_PIN, SPI)) {
        Serial.println("Card Mount Failed");
        return;
    }
    uint8_t cardType = SD.cardType();

    if (cardType == CARD_NONE) {
        Serial.println("No SD card attached");
        return;
    }

    Serial.print("SD Card Type: ");
    if (cardType == CARD_MMC)
        Serial.println("MMC");
    else if (cardType == CARD_SD)
        Serial.println("SDSC");
    else if (cardType == CARD_SDHC)
        Serial.println("SDHC");
    else
        Serial.println("UNKNOWN");

    uint64_t cardSize = SD.cardSize() / (1024 * 1024);
    Serial.printf("SD Card Size: %lluMB\n", cardSize);
    SD.end();
    return;
}

void mic_init(void) {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 44100,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 6,
        .dma_buf_len = 160,
        .use_apll = false,
        .tx_desc_auto_clear = true,
        .fixed_mclk = 0,
        .mclk_multiple = I2S_MCLK_MULTIPLE_256,
        .bits_per_chan = I2S_BITS_PER_CHAN_32BIT,
    };

    i2s_pin_config_t pin_config = {-1};
    pin_config.bck_io_num = MIC_IIS_SCK_PIN;
    pin_config.ws_io_num = MIC_IIS_WS_PIN;
    pin_config.data_in_num = MIC_IIS_DATA_PIN;

    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    i2s_zero_dma_buffer(I2S_NUM_0);
}

#define BUFFER_SIZE (4 * 1024)
uint8_t buffer[BUFFER_SIZE] = {0};
const int define_max = 600;
const int define_avg = 150;
const int define_zero = 3900;
String timelong_str = "";
float val_avg = 0;
int16_t val_max = 0;
float val_avg_1 = 0;
int16_t val_max_1 = 0;
float all_val_avg = 0;
int32_t all_val_zero1 = 0;
int32_t all_val_zero2 = 0;
int32_t all_val_zero3 = 0;
int16_t val16 = 0;
uint8_t val1, val2;
uint32_t j = 0;
bool aloud = false;

void check_sound(void) {
    size_t bytes_read;
    j = j + 1;
    i2s_read(I2S_NUM_0, (char *)buffer, BUFFER_SIZE, &bytes_read, portMAX_DELAY);
    for (int i = 0; i < BUFFER_SIZE / 2; i++) {
        val1 = buffer[i * 2];
        val2 = buffer[i * 2 + 1];
        val16 = val1 + val2 * 256;
        if (val16 > 0) {
            val_avg = val_avg + val16;
            val_max = max(val_max, val16);
        }
        if (val16 < 0) {
            val_avg_1 = val_avg_1 + val16;
            val_max_1 = min(val_max_1, val16);
        }
        all_val_avg = all_val_avg + val16;
        if (abs(val16) >= 20)
            all_val_zero1 = all_val_zero1 + 1;
        if (abs(val16) >= 15)
            all_val_zero2 = all_val_zero2 + 1;
        if (abs(val16) > 5)
            all_val_zero3 = all_val_zero3 + 1;
    }

    if (j % 2 == 0 && j > 0) {
        val_avg = val_avg / BUFFER_SIZE;
        val_avg_1 = val_avg_1 / BUFFER_SIZE;
        all_val_avg = all_val_avg / BUFFER_SIZE;
        if (val_max > define_max && val_avg > define_avg && all_val_zero2 > define_zero)
            aloud = true;
        else
            aloud = false;
        timelong_str = " high_max:" + String(val_max) + " high_avg:" + String(val_avg) + " all_val_zero2:" + String(all_val_zero2);
        if (aloud) {
            timelong_str = timelong_str + " ##### ##### ##### ##### ##### #####";
            Serial.println(timelong_str);
        }
        val_avg = 0;
        val_max = 0;
        val_avg_1 = 0;
        val_max_1 = 0;
        all_val_avg = 0;
        all_val_zero1 = 0;
        all_val_zero2 = 0;
        all_val_zero3 = 0;
    }
}

void wifi_scan_connect(void) {
    // camera_test() เปิด SoftAP แล้ว: ห้ามเปลี่ยนเป็น WIFI_STA เพราะจะปิดเว็บกล้อง
    WiFi.mode(WIFI_AP_STA);
    WiFi.disconnect(false, false);
    delay(100);
    Serial.printf("Connecting to Wi-Fi (timeout %lus)",
                  (unsigned long)(WIFI_CONNECT_TIMEOUT_MS / 1000));
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    const uint32_t started_ms = millis();
    while (WiFi.status() != WL_CONNECTED &&
           millis() - started_ms < WIFI_CONNECT_TIMEOUT_MS) {
        Serial.print(".");
        delay(250);
    }
    Serial.println();
    if (WiFi.status() == WL_CONNECTED) {
        Serial.print("Wi-Fi connected. STA IP: http://");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("Wi-Fi not connected; camera AP remains available.");
    }
    Serial.print("Camera AP: http://");
    Serial.println(WiFi.softAPIP());
}

void pcie_test(void) {
    SerialAT.begin(115200, SERIAL_8N1, PCIE_RX_PIN, PCIE_TX_PIN);
    delay(100);
    pinMode(PCIE_PWR_PIN, OUTPUT);
    digitalWrite(PCIE_PWR_PIN, 1);
    delay(500);
    digitalWrite(PCIE_PWR_PIN, 0);
    delay(3000);
    Serial.println("Waking up PCI module");
    do {
        SerialAT.println("AT");
        delay(50);
    } while (!SerialAT.find("OK"));
    Serial.println("The PCI module has been awakened");

    Serial.println("Example Query the SIM card status");
    do {
        SerialAT.println("AT+CPIN?");
        delay(50);
    } while (!SerialAT.find("READY"));
    Serial.println("SIM card has been identified");
}

bool camera_test() {
    Serial.println("Camera init");
    camera_config_t config = {};
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = CAM_Y2_PIN;
    config.pin_d1 = CAM_Y3_PIN;
    config.pin_d2 = CAM_Y4_PIN;
    config.pin_d3 = CAM_Y5_PIN;
    config.pin_d4 = CAM_Y6_PIN;
    config.pin_d5 = CAM_Y7_PIN;
    config.pin_d6 = CAM_Y8_PIN;
    config.pin_d7 = CAM_Y9_PIN;
    config.pin_xclk = CAM_XCLK_PIN;
    config.pin_pclk = CAM_PCLK_PIN;
    config.pin_vsync = CAM_VSYNC_PIN;
    config.pin_href = CAM_HREF_PIN;
    config.pin_sccb_sda = CAM_SIOD_PIN;
    config.pin_sccb_scl = CAM_SIOC_PIN;
    config.pin_pwdn = CAM_PWDN_PIN;
    config.pin_reset = CAM_RESET_PIN;
    config.xclk_freq_hz = 20000000;
    
    // --- เปลี่ยนฟอร์แมตภาพเป็น RGB565 ---
    config.pixel_format = PIXFORMAT_RGB565; 

    // *** ตั้งค่าขนาดภาพเป็น QVGA ทันที เพื่อป้องกัน RAM ล้น ***
    if (psramFound()) {
        config.frame_size = FRAMESIZE_QVGA; 
        config.jpeg_quality = 10;
        config.fb_count = 2;
        config.fb_location = CAMERA_FB_IN_PSRAM;
        config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
        config.frame_size = FRAMESIZE_QVGA; 
        config.jpeg_quality = 12;
        config.fb_count = 1;
        config.fb_location = CAMERA_FB_IN_DRAM;
    }

#if defined(CAMERA_MODEL_ESP_EYE)
    pinMode(13, INPUT_PULLUP);
    pinMode(14, INPUT_PULLUP);
#endif

    Serial.printf("Camera init\n");
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        return false;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        Serial.println("Camera sensor handle unavailable");
        esp_camera_deinit();
        return false;
    }
    if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }
    
    // ลบการตั้งค่า s->set_framesize ทิ้งไป เพราะเราตั้งเป็น QVGA แต่แรกแล้ว

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);
#endif

    String ssid;
    uint8_t mac[8];
    esp_efuse_mac_get_default(mac);
    ssid = WIFI_AP_SSID;
    ssid += mac[0] + mac[1] + mac[2];
    WiFi.mode(WIFI_MODE_APSTA);
    WiFi.softAP(ssid.c_str(), WIFI_AP_PASSWORD);
    startCameraServer();

    Serial.print("Camera Ready! Use 'http://");
    Serial.print(WiFi.softAPIP());
    Serial.println("' to connect");
    return true;
}
