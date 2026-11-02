/** Noodle ESP32-S3 quantized NHWC graph operators.
 * Read-only model parameters are byte-addressable XIP flash; activation data
 * is caller-owned (normally PSRAM). No dynamic allocation in kernels.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>

struct NoodleS3Tensor {
  uint8_t *data;
  uint32_t bytes;
  uint16_t h, w, c;
  int32_t zero_point;
  float scale;
  bool is_uint8;
};

enum NoodleS3OpType : uint8_t {
  NDL_S3_CONV = 1, NDL_S3_MAXPOOL = 2, NDL_S3_CONCAT = 3,
  NDL_S3_QUANTIZE = 4, NDL_S3_SUB = 5, NDL_S3_MUL = 6,
  NDL_S3_MEAN = 7, NDL_S3_RESHAPE = 8
};

struct NoodleS3Op {
  uint8_t type, relu;
  uint16_t src0, src1, dst;
  uint16_t kh, kw, stride_h, stride_w;
  uint32_t weights, bias, multipliers, shifts;
  float constant_scale;
  int32_t constant_zero_point;
};

bool noodle_s3_i8_execute(const NoodleS3Op &op, NoodleS3Tensor *t,
                          size_t tensor_count, const uint8_t *xip, size_t xip_bytes);
