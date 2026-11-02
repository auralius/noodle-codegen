/** Noodle FP32 NHWC reference kernels for ESP32-S3.
 * Explicit graph execution, PSRAM activations and read-only XIP parameters.
 * Deliberately independent of INT8 kernels; no heap allocation inside operators.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
struct NoodleS3F32Tensor { float *data; uint32_t elements; uint16_t h,w,c; };
enum NoodleS3F32Type : uint8_t { NDL_F32_CONV=1, NDL_F32_MAXPOOL=2, NDL_F32_CONCAT=3, NDL_F32_SUB=4, NDL_F32_MUL=5, NDL_F32_MEAN=6, NDL_F32_RESHAPE=7, NDL_F32_AVGPOOL=8, NDL_F32_FCN=9, NDL_F32_SOFTMAX=10 };
struct NoodleS3F32Op {
  uint8_t kind, relu;
  uint16_t src0,src1,dst,kh,kw,sh,sw;
  uint32_t weights,bias;
};
bool noodle_s3_f32_execute(const NoodleS3F32Op &op, NoodleS3F32Tensor *t,
                           size_t count, const uint8_t *xip, size_t xip_bytes);
