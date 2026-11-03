# R15 v0.6.27: production INT8 CHW convolution cleanup

This release keeps one normal ESP32-S3 PlatformIO build (`esp32-s3-n16r8`) that uses:

- Unified CHW Noodle INT8 `noodle_i8_conv2d` (OIHW parameter layout).
- Pointwise4 for eligible in-memory 1x1 Conv2D.
- Spatially tiled 3x3 with a conservative signed INT32 accumulator bound and automatic INT64 fallback for unsafe output filters.
- General direct convolution fallback for other supported kernel sizes/backends.
- XIP model parameters, unchanged quantization, and branch-aware three physical activation buffers.

Historical experimental Pointwise8/16 and multiple test environments are removed from newly generated firmware. The existing generic portable Noodle operator and backends are retained. This is not a rewrite of the FP32, AVR, SD or PROGMEM kernels.

## Build / profile

```bash
pio run -e esp32-s3-n16r8 -t upload
python3 test/send_tensor_profile.py --port /dev/ttyACM0 --image test/bus.jpeg --profile | tee profile_production.log
```

Reference on physical ESP32-S3 N16R8 (previous validated Tiled32+Pointwise4 build): 33.789 s, class 779, 1000-byte CRC32 `a046a893`. The refactored release is native-tested, not yet hardware-flashed; compare it with this reference.

Pointwise16 was measured at 35.944 s on this hardware (6.4% slower); Pointwise8 was not hardware-benchmarked. One input image and one run per configuration are not a statistical performance study.
