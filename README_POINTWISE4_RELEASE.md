# Noodle Studio R15 v0.6.12 — Model Zoo + Pointwise4

Based on the working v0.6.10 Codegen and Model Zoo dropdown. The only inference change is in the **ESP32-S3 INT8 NHWC convolution kernel**.

## What changed

- Pointwise4 is now the **default** for eligible 1x1, stride-1, same-size INT8 convolutions. Reuses each flash-mapped weight across four output pixels, with a scalar tail loop.
- All 3x3 convolutions, normal PSRAM/XIP memory mapping, graph topology, quantization math, serial sender and FP32 kernels are unchanged.
- Original kernel is still available as `esp32-s3-n16r8-scalar` in generated INT8 PlatformIO firmware.
- Original default environment `esp32-s3-n16r8` now runs Pointwise4; `esp32-s3-n16r8-o2` does so too.
- Uno/Nano LeNet-5 and all FP32 Codegen paths are unchanged.
- Model Zoo dropdown retains FCN-MNIST16 INT8, LeNet-5 INT8 (Uno/SD), and SqueezeNet 1.1 INT8/FP32. The unvalidated LeNet-5 FP32 preview and redundant Uno/SD preset toolbar button have been removed.

## Hardware test

```sh
pio run -e esp32-s3-n16r8 -t upload
python3 test/send_tensor.py --port /dev/ttyACM0 --image test/dog.jpg
```

Connect to **USB OTG** for serial input. Expected for previously tested `dog.jpg`: class 258; the previously measured unoptimized CRC is `746c1040`. Check new optimized CRC with the same image before treating equivalence as hardware-validated.

For strict A/B regression, upload the scalar fallback environment:

```sh
pio run -e esp32-s3-n16r8-scalar -t upload
python3 test/send_tensor.py --port /dev/ttyACM0 --image test/dog.jpg
```

Note: the board already completed Pointwise4 inference at **37.970 s** on bus and cat images vs original INT8 ~**43.327 s** (dog). The pointwise arithmetic was byte-identical on a native host test across 43 operators. Hardware bit-identical comparison on the *same* image between scalar/Pointwise4 still needs confirmation.

## Verification scope

Tests include offline generator regression, browser bundle parity, identical generated parameter images and code structure. Final actual compilation and performance verification depend on your PlatformIO and ESP32-S3 setup.
