# Operator-driven FP32 export (ESP32-S3 N16R8)

Experimental general graph export, not a fixed-model whitelist. For the first validation target, import the included `lenet5_fp32.tflite` (input [1,28,28,1], 9 ops). Choose ESP32-S3 and XIP params / PSRAM activations (importer defaults). Click Generate Firmware ZIP, unzip it, then use:

```sh
pio run -e esp32-s3-n16r8-fp32 -t upload
python3 test/send_tensor.py --port /dev/ttyACM0 --pattern zeros --save-payload input_float32.bin
python3 test/reference.py --input-file input_float32.bin
```

The sender uses raw FLOAT32 transport, not RGB bytes. `--image path.png` works with grayscale/RGB; `--image-scale` defaults to 1/255, but the correct scale depends on the training pipeline. `--input-file` takes exact float32 tensor bytes. No automatic preprocessing is applied on the ESP32.

Backend: graph-ordered operator call plan, liveness-allocated PSRAM arena, XIP flash parameters at `0x800000`; emits FP32 Conv2D (ReLU/none), AveragePool2D (ReLU/none), MaxPool2D, FCN (ReLU/none), Reshape, Softmax (beta=1), and SqueezeNet subset (Concat, spatial MEAN, SUB/MUL with RGB constants). Limitations: one batch-one input/output, XIP params, PSRAM activations, unsupported options rejected. No INT8 model generality is claimed. LeNet-5 FP32 hardware and numeric equivalence must be confirmed. Existing tested SqueezeNet paths remain unchanged.

## Implemented scope and validation

- The Studio routes known SqueezeNet graphs through the existing exporters, then selects the generic FP32 path for other graphs comprised of supported operators. The Model Zoo is unchanged; import a custom `.tflite` with **Import TFLite**.
- Input transport is raw little-endian FLOAT32, so the sender and firmware agree on all tensor dimensions.
- The imported LeNet5 FP32 graph was exported to actual `main.cpp`, model `params.bin`, and PlatformIO project. A native g++ execution of the generated nine operations matched independent NumPy calculations to within 1e-8 maximum absolute error for a ramp input.
- Existing SqueezeNet FP32, Uno INT8 LeNet5, browser, shared-window and core regression checks pass. **No ESP32-S3 hardware test or PlatformIO cross-build has been performed here.**
- Some operator subtypes/configurations remain unsupported. Generation checks dimension and supported-option constraints before producing the bundle; no claim of arbitrary TFLite operator support is made.
