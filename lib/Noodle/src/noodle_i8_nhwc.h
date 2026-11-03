/** Portable Noodle INT8 NHWC graph operators.
 * All activations are ordinary NoodleTensor/NoodleBuffer storage.
 * Weights are byte-addressable read-only memory, XIP on supported targets.
 * This backend does not allocate memory or depend on ESP32 headers.
 * Shapes and UINT8/INT8 quantization are per-logical-tensor metadata.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include "noodle_tensor.h"

struct NoodleI8NHWCTensor {
  NoodleTensor *storage; // Shared physical NoodleTensor, sized/bound by caller.
  uint32_t bytes;
  uint16_t h, w, c;
  int32_t zero_point;
  float scale;
  bool is_uint8;
};

enum NoodleI8NHWCOpType : uint8_t {
  NDL_I8_NHWC_CONV = 1, NDL_I8_NHWC_MAXPOOL = 2, NDL_I8_NHWC_CONCAT = 3,
  NDL_I8_NHWC_QUANTIZE = 4, NDL_I8_NHWC_SUB = 5, NDL_I8_NHWC_MUL = 6,
  NDL_I8_NHWC_MEAN = 7, NDL_I8_NHWC_RESHAPE = 8
};
struct NoodleI8NHWCOp {
  uint8_t type, relu;
  uint16_t src0, src1, dst;
  uint16_t kh, kw, stride_h, stride_w;
  uint32_t weights, bias, multipliers, shifts;
  float constant_scale;
  int32_t constant_zero_point;
};

static inline uint8_t *noodle_i8_nhwc_data(NoodleI8NHWCTensor &t) {
  return t.storage ? reinterpret_cast<uint8_t*>(noodle_tensor_data(t.storage)) : nullptr;
}
static inline const uint8_t *noodle_i8_nhwc_data(const NoodleI8NHWCTensor &t) {
  return t.storage ? reinterpret_cast<const uint8_t*>(noodle_tensor_const_data(t.storage)) : nullptr;
}

bool noodle_i8_nhwc_execute(const NoodleI8NHWCOp &op, NoodleI8NHWCTensor *t,
                            size_t tensor_count, const uint8_t *parameter_base, size_t parameter_bytes);
