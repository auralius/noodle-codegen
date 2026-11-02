# Model zoo dropdown patch

Replaces the old example dropdown with tested models: FCN-MNIST16 INT8, LeNet-5 INT8, LeNet-5 FP32, SqueezeNet 1.1 INT8, and SqueezeNet 1.1 FP32.

## GUI cleanup (2026-11-02)

- Removed the redundant "Use Uno/SD LeNet-5 preset" toolbar button. The bundled LeNet-5 INT8 project already selects Uno/SD parameters and activation storage.
- Removed LeNet-5 FP32 preview from Model Zoo because it is not an established deployment. No FP32 FCN option is advertised.
- Retained FCN-MNIST16 INT8, LeNet-5 INT8, SqueezeNet-1.1 INT8, and SqueezeNet-1.1 FP32.
- No kernel, codegen, hardware profile, or bundled model content was changed.
