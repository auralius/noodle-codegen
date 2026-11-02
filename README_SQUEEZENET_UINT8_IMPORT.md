# SqueezeNet-1.1 UINT8 TFLite import (inspection only)

This update to Noodle Studio R15 v0.6.6 accepts `UINT8` TFLite input tensors, keeping the unchanged embedded model binary and tensor-by-tensor quantization metadata. It also lists and recognizes the `QUANTIZE`, `SUB`, and `MUL` operations used by Qualcomm SqueezeNet-1.1 W8A8.

## Usage

1. Open `index.html` locally (or publish these updated files on GitHub Pages).
2. Choose **Import TFLite** and select your Qualcomm `squeezenet1_1.tflite`.
3. Verify target **ESP32-S3 N16R8**, 43 operations, eight concatenations, and UINT8 external input and output.
4. Keep **Apply ping-pong for sequential network** unchecked (Fire modules branch).
5. Inspect the graph, edit storage assignments if desired, and use **Save Project JSON**. This includes the original TFLite bytes.

## Important limitations

- Import, graph preview and JSON round trip **only**: the current Codegen deliberately rejects ESP32-S3 firmware generation.
- No INT8/UINT8 runtime conversions, Fire-module execution, PSRAM allocation or FFat streaming are added here.
- The TFLite graph begins with `QUANTIZE`, `SUB`, `MUL` and ends with `QUANTIZE`; these are preserved as real graph operations. Do not replace them with unconditional `uint8 - 128` preprocessing.
- LeNet-5 Uno/Nano generator and Noodle operators are **unchanged**.

## Test

```sh
node test_squeezenet_uint8.js /path/to/squeezenet1_1.tflite
node test_tflite.js
node test_core.js
node test_lenet_examples.js
```

The uploaded Qualcomm model is not redistributed in this ZIP.
