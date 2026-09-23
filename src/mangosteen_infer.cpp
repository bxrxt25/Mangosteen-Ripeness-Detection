#include "mangosteen_infer.h"
#include "img_converters.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"

#include <math.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h" // <-- เพิ่ม Header สำหรับ Error Reporter
#include "tensorflow/lite/schema/schema_generated.h"

#include "mangosteen_int8.h"   // <-- ไฟล์ C Array โมเดลมังคุดของคุณ

namespace {
const tflite::Model    *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input  = nullptr;
TfLiteTensor *output = nullptr;
uint8_t *tensor_arena = nullptr;
uint8_t *rgb_workspace = nullptr;
size_t rgb_workspace_size = 0;

int in_w = 0, in_h = 0, in_ch = 0, n_classes = 0;

portMUX_TYPE result_mux = portMUX_INITIALIZER_UNLOCKED;
MangosteenResult latest_result = {};
const char *const class_labels[MANGOSTEEN_CLASS_COUNT] = {
    MANGOSTEEN_LABEL_0, MANGOSTEEN_LABEL_1, MANGOSTEEN_LABEL_2};

// ops ที่โมเดล CNN ทั่วไปใช้ ถ้า init แล้วขึ้น error "Didn't find op for builtin
// opcode ..." ให้เปิดดูโมเดลแล้วเพิ่ม op ที่ขาดตรงนี้
constexpr int kNumOps = 5;
tflite::MicroMutableOpResolver<kNumOps> resolver;

// ลดการสั่นของผลลัพธ์จากเฟรมเดียว โดยไม่เปลี่ยนโมเดลหรือ class ที่โมเดลส่งออก
constexpr float kScoreEmaAlpha = 0.35f;
float filtered_scores[MANGOSTEEN_CLASS_COUNT] = {};
bool filtered_scores_initialized = false;

bool ensure_rgb_workspace(size_t required_size)
{
    if (required_size <= rgb_workspace_size) return true;

    // workspace นี้ถูก reuse ทุกเฟรม จึงไม่เกิด alloc/free PSRAM ซ้ำ ๆ ระหว่างรัน
    uint8_t *new_workspace = (uint8_t *)heap_caps_realloc(
        rgb_workspace, required_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!new_workspace) return false;

    rgb_workspace = new_workspace;
    rgb_workspace_size = required_size;
    return true;
}
}

bool mangosteen_init()
{
    // MODEL_ARRAY เป็น const และ align 16 จึงอ่านจาก flash ที่ map ไว้ได้โดยตรง
    // ไม่ต้องเสีย PSRAM และเวลา copy ทุกครั้งที่บูต
    model = tflite::GetModel(MODEL_ARRAY);
    if (!model || model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("[TFLM] schema mismatch %lu != %d\n",
                      model ? (unsigned long)model->version() : 0UL, TFLITE_SCHEMA_VERSION);
        return false;
    }

    // --- arena: ลอง SRAM ก่อน (เร็วกว่ามาก) ถ้าไม่พอค่อยตก PSRAM ---
    tensor_arena = (uint8_t *)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE,
                                                      MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!tensor_arena) {
        Serial.println("[TFLM] arena -> PSRAM (ช้ากว่า)");
        tensor_arena = (uint8_t *)heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE,
                                                          MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    }
    if (!tensor_arena) { Serial.println("[TFLM] alloc arena failed"); return false; }

    // --- โหลด Operations ที่จำเป็น ---
    resolver.AddConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddMaxPool2D();
    resolver.AddMean();          // GlobalAveragePooling2D

    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    static tflite::MicroInterpreter static_interpreter(model, resolver,
                                                       tensor_arena, TENSOR_ARENA_SIZE,
                                                       error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[TFLM] AllocateTensors failed -> เพิ่ม TENSOR_ARENA_SIZE");
        return false;
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);
    if (!input || !output || !input->dims || !output->dims ||
        input->dims->size != 4 || output->dims->size < 2) {
        Serial.println("[TFLM] unexpected tensor rank");
        return false;
    }

    in_h  = input->dims->data[1];
    in_w  = input->dims->data[2];
    in_ch = input->dims->data[3];
    n_classes = output->dims->data[output->dims->size - 1];

    if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
        input->dims->data[0] != 1 || in_w <= 0 || in_h <= 0 ||
        in_ch != 3 || n_classes != MANGOSTEEN_CLASS_COUNT) {
        Serial.printf("[TFLM] unexpected tensor: input type=%d channels=%d, output type=%d classes=%d\n",
                      input->type, in_ch, output->type, n_classes);
        return false;
    }

    Serial.printf("[TFLM] input %dx%dx%d type=%d scale=%.6f zp=%d\n",
                  in_w, in_h, in_ch, input->type,
                  input->params.scale, input->params.zero_point);
    Serial.printf("[TFLM] classes=%d  arena used=%u B\n",
                  n_classes, (unsigned)interpreter->arena_used_bytes());
    return true;
}

int mangosteen_num_classes() { return n_classes; }
int mangosteen_input_size()  { return in_w; }

const char *mangosteen_class_label(int class_index)
{
    return (class_index >= 0 && class_index < MANGOSTEEN_CLASS_COUNT)
               ? class_labels[class_index]
               : "UNKNOWN";
}

bool mangosteen_get_latest_result(MangosteenResult *result)
{
    if (!result) return false;
    portENTER_CRITICAL(&result_mux);
    *result = latest_result;
    portEXIT_CRITICAL(&result_mux);
    return result->valid;
}

int mangosteen_predict(camera_fb_t *fb, float *scores, int max_scores, uint32_t *ms_out)
{
    if (!interpreter || !input || !output || !fb) return -1;

    // --- 1) แปลงเฟรมเป็น RGB888 (รองรับทั้ง JPEG และ RGB565) ---
    if (!fb->buf || fb->width == 0 || fb->height == 0) return -1;
    size_t rgb_len = (size_t)fb->width * fb->height * 3;
    if (!ensure_rgb_workspace(rgb_len)) {
        Serial.printf("[TFLM] RGB workspace alloc failed (%u B)\n", (unsigned)rgb_len);
        return -1;
    }
    if (!fmt2rgb888(fb->buf, fb->len, fb->format, rgb_workspace)) return -1;

    // --- 2) center-crop เป็นสี่เหลี่ยมจัตุรัส แล้วย่อแบบ nearest neighbor ---
    int side = min((int)fb->width, (int)fb->height);
    int x0 = (fb->width  - side) / 2;
    int y0 = (fb->height - side) / 2;

    int8_t *in = input->data.int8;
    const float scale = input->params.scale;
    const int   zp    = input->params.zero_point;

    for (int y = 0; y < in_h; y++) {
        int sy = y0 + (int)((int64_t)y * side / in_h);
        for (int x = 0; x < in_w; x++) {
            int sx = x0 + (int)((int64_t)x * side / in_w);
            const uint8_t *p = rgb_workspace + ((size_t)sy * fb->width + sx) * 3;
            // fmt2rgb888 ให้ลำดับไบต์เป็น B,G,R
            float r = p[2], g = p[1], b = p[0];

            if (in_ch == 1) {
                float v = 0.299f * r + 0.587f * g + 0.114f * b;
#ifdef NORM_0_1
                v = v / 255.0f;
#else
                v = v / 127.5f - 1.0f;
#endif
                int q = lroundf(v / scale) + zp;
                *in++ = (int8_t)constrain(q, -128, 127);
            } else {
                float c[3] = {r, g, b};
                for (int k = 0; k < 3; k++) {
#ifdef NORM_0_1
                    float v = c[k] / 255.0f;
#else
                    float v = c[k] / 127.5f - 1.0f;
#endif
                    int q = lroundf(v / scale) + zp;
                    *in++ = (int8_t)constrain(q, -128, 127);
                }
            }
        }
    }
    // --- 3) inference ---
    uint32_t t0 = millis();
    if (interpreter->Invoke() != kTfLiteOk) return -1;
    if (ms_out) *ms_out = millis() - t0;

    // --- 4) dequantize เอาท์พุต ---
    // โมเดลนี้มี Softmax เป็น operation สุดท้ายแล้ว ดังนั้น v คือ probability
    // ห้ามทำ softmax ซ้ำ เพราะจะทำ confidence เพี้ยน
    int n = min(n_classes, MANGOSTEEN_CLASS_COUNT);
    int best = 0;
    float best_v = -1.0f;
    for (int i = 0; i < n; i++) {
        float v = (output->type == kTfLiteInt8)
                      ? (output->data.int8[i] - output->params.zero_point) * output->params.scale
                      : output->data.f[i];
        v = constrain(v, 0.0f, 1.0f);
        if (!filtered_scores_initialized) {
            filtered_scores[i] = v;
        } else {
            filtered_scores[i] += kScoreEmaAlpha * (v - filtered_scores[i]);
        }
    }
    filtered_scores_initialized = true;

    for (int i = 0; i < n; i++) {
        const float v = filtered_scores[i];
        if (i < max_scores && scores) scores[i] = v;
        if (v > best_v) { best_v = v; best = i; }
    }

    MangosteenResult result = {};
    result.valid = true;
    result.class_index = best;
    result.confidence = best_v;
    result.inference_ms = ms_out ? *ms_out : 0;
    result.updated_ms = millis();
    for (int i = 0; i < n; i++) {
        result.scores[i] = (scores && i < max_scores) ? scores[i] : 0.0f;
    }
    portENTER_CRITICAL(&result_mux);
    latest_result = result;
    portEXIT_CRITICAL(&result_mux);
    return best;
}
