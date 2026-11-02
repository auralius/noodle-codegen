# Compiler optimization comparison — ESP32-S3 N16R8

These Noodle Studio ESP32-S3 firmware exports include two PlatformIO environments, both built in release mode:

- **Baseline:** `esp32-s3-n16r8` (INT8) or `esp32-s3-n16r8-fp32` (FP32). PlatformIO's default compiler optimization is preserved (normally `-Os` for Arduino ESP32).
- **Speed experiment:** the matching environment with `-o2` suffix. Explicitly removes `-Os` and applies `-O2` for code compiled from source (including Noodle kernels).

### Benchmark commands

INT8:

```sh
pio run -e esp32-s3-n16r8-o2 -t upload
python3 test/send_tensor.py --port /dev/ttyACM0 --image test/dog.jpg
```

FP32:

```sh
pio run -e esp32-s3-n16r8-fp32-o2 -t upload
python3 test/send_tensor.py --port /dev/ttyACM0 --image test/dog.jpg
```

Use the **USB OTG** port for native USB CDC serial on the Freenove-style ESP32-S3 board. Close the monitor before the sender. Uploading still writes the XIP parameter image at the normal dedicated partition offset.

### Scientific comparison

Record median and variability over repeated identical image inferences, code/flash size, free PSRAM, and output checksums. Changing optimization level may change floating-point rounding; compare numeric output to TFLite rather than only class predictions.

`-Os` is optimized for **code size**. `-O2` targets **execution speed** but can produce **larger code and more instruction-cache pressure**. Neither is guaranteed to be faster on the ESP32-S3. Precompiled Arduino framework binaries may not inherit all of the application optimization settings. Do not use `-Ofast` until numerical equivalence has been established.

The original baseline environments remain available and unchanged except for explicit `build_type = release`.
