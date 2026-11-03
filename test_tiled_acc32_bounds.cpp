// Compile this test with NOODLE_INT8_CONV_IMPL=1 and
// NOODLE_INT8_CONV3X3_ACC32=1. Internal function visibility is intentional.
#include "lib/Noodle/src/noodle_int8_layers.cpp"
#include <assert.h>
#include <stdint.h>
int main() {
  assert(q_conv3x3_acc32_is_safe(3, -14, 64492));
  assert(q_conv3x3_acc32_is_safe(64, -128, 8));
  assert(!q_conv3x3_acc32_is_safe(64, -128, INT32_MAX));
  assert(!q_conv3x3_acc32_is_safe(65535, -128, 0));
  assert(!q_conv3x3_acc32_is_safe(64, INT32_MIN, 0));
  assert(!q_conv3x3_acc32_is_safe(1, 127, INT32_MIN));
  return 0;
}
