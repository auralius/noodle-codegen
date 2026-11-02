# Noodle Studio R15 v0.6.9 — ESP32-S3 FP32 SqueezeNet experiment

This version **adds** a separate FP32 SqueezeNet code generator. The existing
ESP32-S3 INT8 and Arduino Uno/Nano backends are preserved.

## Using the GUI

1. Open `index.html` (or deploy all its neighboring assets to GitHub Pages).
2. Import `squeezenet1_1_fp32.tflite` or open the separate FP32 project JSON.
3. Set board ESP32-S3 N16R8; all activation backends PSRAM and parameter backends XIP.
4. Leave sequential ping-pong **disabled**; the new FP32 backend uses independent
   liveness-based reuse of dead activation tensors.
5. Generate Firmware ZIP and extract it.
6. Run `pio run -e esp32-s3-n16r8-fp32 -t upload` (includes the model flash image).
7. Connect the board's USB OTG/native USB port and run:
   `python3 test/send_tensor.py --port /dev/ttyACM0 --image dog.jpg`

The FP32 input is a [1,224,224,3] tensor with values RGB/255.0. For serial
convenience, the sender transfers 150,528 RGB bytes and the firmware expands
these in-place into float32 before invoking the imported normalization steps.

The official FP32 TFLite model here has 41 operations, and its output consists of
1000 FLOAT32 logits. No Softmax node appears in the imported model.

## Memory and backends

- INT8 path: independent existing W8A8 generator, 43 operators, no changes.
- FP32 path: 41 operators, 4,358,400-byte PSRAM arena, 4,942,024-byte XIP image.
- FP32 flash partition begins at 0x800000 and spans 8 MiB; no FFat required.
- Flash parameters are written by the normal PlatformIO upload task.

## Verification and limitations

Passed: source TFLite import, graph/shape checks, liveness allocator,
41-operator native inference with AddressSanitizer/UBSan, generated C++ syntax
check, flash image CRC, sender dry-run, INT8 codegen and LeNet5 regression tests.
The native gradient inference's class was 794, also the class observed previously
for the corresponding INT8 input on real hardware; matching top-1 alone is NOT
numerical equivalence.

Not tested: actual ESP32-S3 build (PlatformIO not installed in this environment),
flashing/boot/serial on hardware, 1000-logit agreement with TensorFlow Lite, and
physical inference latency. Published FP32 125.1 s is not guaranteed because
these NHWC kernels and buffer scheduling differ from the old FP32 program.

## Optional compiler speed experiment
See [README_COMPILER_OPTIMIZATION.md](README_COMPILER_OPTIMIZATION.md). The baseline export is preserved; use `-e esp32-s3-n16r8-fp32-o2` to compare an `-O2` build.
