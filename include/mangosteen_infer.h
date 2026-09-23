#pragma once
#include <Arduino.h>
#include "esp_camera.h"

// ---------------------------------------------------------------------------
// ปรับ 3 ค่านี้ให้ตรงกับโมเดลของคุณ
// ---------------------------------------------------------------------------
// 1) ชื่อ array ในไฟล์ mangosteen_int8.h  (เปิดไฟล์ดูบรรทัดแรก ๆ)
#define MODEL_ARRAY      mangosteen_int8_tflite
#define MODEL_ARRAY_LEN  mangosteen_int8_tflite_len

// ลำดับต้องตรงกับ class_indices ตอนเทรนโมเดลเท่านั้น
// โมเดลปัจจุบันมี 3 class: แก้ข้อความ/ลำดับที่นี่หากชุดเทรนใช้ลำดับอื่น
constexpr int MANGOSTEEN_CLASS_COUNT = 3;
#define MANGOSTEEN_LABEL_0 "UNRIPE"  // ดิบ
#define MANGOSTEEN_LABEL_1 "TURNING" // กำลังสุก
#define MANGOSTEEN_LABEL_2 "RIPE"    // สุก

// 2) วิธี normalize ตอนเทรน
//    NORM_0_1   -> Rescaling(1./255)          ค่า 0..1
//    NORM_M1_1  -> Rescaling(1./127.5, -1)    ค่า -1..1  (MobileNet preprocess_input)
#define NORM_0_1
// #define NORM_M1_1

// 3) ขนาด tensor arena (ไบต์). MobileNetV2 96x96 int8 ราว 250-400KB
#define TENSOR_ARENA_SIZE (400 * 1024)

// ---------------------------------------------------------------------------
bool mangosteen_init();
int  mangosteen_num_classes();
int  mangosteen_input_size();   // ความกว้าง/สูงที่โมเดลต้องการ

struct MangosteenResult {
    bool valid;
    int class_index;
    float confidence;
    float scores[MANGOSTEEN_CLASS_COUNT];
    uint32_t inference_ms;
    uint32_t updated_ms;
};

// thread-safe: ใช้ทั้ง loop() และ HTTP server ได้
bool mangosteen_get_latest_result(MangosteenResult *result);
const char *mangosteen_class_label(int class_index);

// รับ frame จากกล้อง (JPEG หรือ RGB565 ก็ได้), คืน index ของคลาสที่คะแนนสูงสุด
// scores = ค่าความน่าจะเป็นหลัง dequantize, ms_out = เวลา inference (ms)
// คืน -1 เมื่อผิดพลาด
int mangosteen_predict(camera_fb_t *fb, float *scores, int max_scores,
                       uint32_t *ms_out = nullptr);
