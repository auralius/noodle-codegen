> Historical implementation note: v0.6.24 always exports unified native CHW for SqueezeNet INT8. Legacy NHWC is no longer selectable.

# Noodle Studio R15 v0.6.19 — portable INT8 NHWC interface (staged migration)

This release **only changes the ESP32-S3 generic INT8 Codegen output**. FP32, Uno/Nano/Mega exporters, the shared sequential detector and the established sequential ping-pong implementation remain unchanged. The previous S3-specific INT8 source is retained in the Studio development library solely for compatibility with existing hand-written projects, but is **not emitted into new INT8 firmware**.

## Architecture

- `lib/Noodle/src/noodle_i8_nhwc.h/.cpp`: portable, MCU-independent INT8 NHWC operator API. It contains the same numerical kernels previously used in the S3 file (Conv2D, Pointwise4, MaxPool, Concat, Quantize, SUB, MUL, MEAN, RESHAPE). No ESP-IDF, Arduino or PSRAM calls occur inside it. The kernel reads byte-addressable constant parameters; ESP32-XIP is one possible backend. AVR `PROGMEM` still requires separate backend support.
- `NoodleI8NHWCTensor` stores per-logical-tensor NHWC shape, INT8/UINT8 quantization, and a pointer to a physical `NoodleTensor`. Different logical tensors can reference the same physical tensor when their lifetimes do not overlap.
- Physical `NoodleTensor` objects use the existing `NoodleBuffer` ABI. The ESP32-S3 generated application allocates exactly 3 PSRAM blocks and binds their addresses using `noodle_buffer_bind_external()`. This is the **existing fixed external binding**, not a grow-only managed allocation. Buffer sizes are calculated automatically by the branch planner and need no manual resizing. `NoodleBuffer` grow-only behavior remains available to other Noodle paths.
- `branch_buffer_planner.js` remains unchanged: it prevents aliasing for each operator and all concat inputs. No graph, weights, quantization, Pointwise4 calculation or operator descriptor is changed.

## Experimental SqueezeNet example

- 44 logical activations / 43 operators / 8 concat ops
- 3 physical `NoodleTensor`/`NoodleBuffer` instances instead of 44 separate allocations
- 1,369,344 B of planned activation storage versus 4,975,136 B legacy
- `settings.reuse_branch_buffers=false` retains the one-allocation-per-logical-tensor fallback, but both modes now call the same portable Noodle API.
- Baseline `-Os` and optional `-O2` configurations and historical `NOODLE_S3_I8_POINTWISE_PIXELS=0` flag continue to work. The generic kernel also accepts `NOODLE_I8_NHWC_POINTWISE_PIXELS` (default 4).

## Validation boundaries

Native C++ tests compare every operator output hash against the previous S3 kernel on three deterministic input patterns. They are not proof of correct camera capture or physical ESP32 execution. Before declaring production readiness, test an on-device `pio run -e esp32-s3-n16r8 -t upload`, run gradient and photograph inputs, compare all 1000 logits, and benchmark PSRAM footprint/latency.

This is a **portable API migration**, not a claim that the older CHW `noodle_conv_float` / `noodle_i8_conv2d_fused` already implement the same NHWC/W8A8 convolution. Removing the legacy S3 source from all projects can wait until hardware compatibility is verified.
