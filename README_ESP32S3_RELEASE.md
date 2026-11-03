> Historical behavior for the release named below. As of **v0.6.23**, the GUI no longer exposes the ping-pong checkbox; see [`README_AUTOMATIC_MEMORY_POLICY.md`](README_AUTOMATIC_MEMORY_POLICY.md). Existing direct Codegen APIs retain their legacy flags for compatibility.

# Noodle Studio R15 v0.6.7 — ESP32-S3 SqueezeNet INT8 preview

Adds a separate **ESP32-S3-N16R8 serial-input firmware exporter** without changing the Arduino Uno/Nano generators.

- Original Qualcomm SqueezeNet 1.1 W8A8 TFLite input/output UINT8, internal INT8; 43 operations (26 convolutions, 8 concats, 3 max pools, 2 quantize, SUB, MUL, spatial MEAN, RESHAPE).
- PSRAM: 44 logical activation tensors (4,975,136 bytes when separately allocated). **R15 v0.6.18 additionally supports automatic branch-aware reuse**: 3 physical buffers, 1,369,344 bytes for this model. Sequential ping-pong remains disabled on SqueezeNet; see `README_BRANCH_BUFFER_REUSE.md`.
- XIP flash: `model/params.bin` (~1.22 MiB) in raw data partition at `0x800000`; weights, INT32 biases, and per-channel quantization coefficients are byte-mapped, not FFat file streaming.
- Library: new `noodle_s3_nhwc_i8` operator module with NHWC, INT8 quantized execution. Original Noodle operators remain unmodified.
- Application: synthetic 224x224 RGB fixed input, executes complete model, emits 1000 quantized output bytes and timing over Serial. Camera and TFT integration **not included** in first stage.
- Generator rejects unsupported operations, mismatched shapes/backend, graph reordering, and active sequential ping-pong, instead of emitting partially working firmware.

### Installation / usage

Extract ZIP, open `index.html` offline (or deploy files to GitHub Pages), open the previously saved SqueezeNet JSON or import Qualcomm `.tflite`, verify target ESP32-S3 and PSRAM/XIP assignments, leave ping-pong unchecked, select **Generate Firmware ZIP**.

Generated firmware contains detailed `README_ESP32S3_SQUEEZENET.md` with compile, flash, partition mapping, reference test commands.

### Validation status

Host tests pass 43/43 graph operators; native C++ runs on x86 using the exact packed XIP model image and new Noodle NHWC kernel. The previous Uno/Nano LeNet-5 Codegen regression tests pass. **No physical ESP32-S3 compile or hardware inference was performed here**, and byte-level TFLite numerical agreement is not yet established. Do not claim deployment accuracy or performance.

## v0.6.8: Serial test + one-step upload
The exported ESP32-S3 project now contains `test/send_tensor.py` and `test/input_spec.json` derived from the TFLite input, with `IMG`/`RDYIMG`/32-byte `K` acknowledgments and `OUTPUT_UINT8` results. Inputs can be raw bytes, generated RGB gradient, or Pillow-converted photographs. The serial receiver accepts repeated inputs. The generated `pio_flash_model.py` adds `model/params.bin` at `0x800000` to the normal PlatformIO `upload` target, removing the separate esptool command. Physical testing remains required.
