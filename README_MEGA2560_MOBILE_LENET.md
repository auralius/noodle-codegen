# Noodle Studio R15 v0.6.17 — Mega2560 Mobile-LeNet-16

This is an additive, model-family-specific generator based on the supplied `mega2560.zip` reference implementation. Existing Uno/Nano and ESP32-S3 generators are unchanged.

## GUI

Choose **Mobile-LeNet-16 · INT8 (Mega2560)** or **Mobile-LeNet-16 · FP32 (Mega2560)** from the tested-model dropdown, or import either provided `.tflite` file. The TFLite importer auto-selects **Arduino Mega2560 (ATmega2560)** and assigns PROGMEM parameters, SD early activations, and SRAM final block activations. Click **Generate Firmware ZIP**.

The exporter rejects unsupported operator sequences, tensor shapes, activation dtypes, quantization, edited topology, edited parameter backend, and incompatible activation storage assignments. It retains the firmware's validated mixed SD/SRAM strategy, not generic ping-pong. The sequential indicator is informational only for Mega2560.

## Generated firmware

* Includes local `lib/Noodle` from the current R15 Studio bundle; eliminates the machine-specific `symlink:///home/...` reference.
* Regenerates FP32 weights/biases and INT8 W/B/M/S arrays from the imported `.tflite`. TFLite depthwise weights [1,H,W,C] are reordered to Noodle [C,H,W], while standard O,H,W,I kernels are reordered to O,I,H,W.
* Includes the supplied Mega2560 reference application as its per-precision source and retains the tested SPI/SD/PROGMEM and B6-in-SRAM allocation.
* Keeps the original raw-256-byte UART0 sender protocol at 115200 baud and the model's raw uint8-to-[0,1] input preprocessing. FP32 normalization is done before SD writes; INT8 quantizes the normalized byte value according to model input scale/zero point.
* Does NOT need SD model parameter binaries. SD card is used for intermediate feature maps and must be writable.
* No camera or LCD component is added.

## Build

INT8:

```sh
pio run -e case2-sd4-control
pio run -e case2-sd4-control -t upload
```

FP32:

```sh
pio run -e mega2560-fp32-sd4
pio run -e mega2560-fp32-sd4 -t upload
```

SD SPI: Mega2560 CS D53, MOSI D51, MISO D50, SCK D52. Use a level-shifted, 5V-compatible microSD adapter.

## Validation

The exact two provided checkpoints generate parameter arrays **numerically identical** to the reference header arrays (all 14 layers; FP32 weights/biases and INT8 weights/biases/multipliers/shifts). The INT8 input/output and per-layer quantization metadata are also bitwise-equivalent as C `float`. The original runnable Codegen regression tests pass. Tests requiring external SqueezeNet fixture files were not run. PlatformIO AVR toolchain and physical Mega2560 hardware testing were not available in this environment.

Only the 18-operator Mobile-LeNet-16 family (14 weighted ops + 2 spatial PAD + GAP + SOFTMAX) is supported. This is **not** a universal Mega2560 convolutional graph compiler.
