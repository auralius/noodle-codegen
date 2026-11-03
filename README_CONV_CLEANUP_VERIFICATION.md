# R15 v0.6.27 Production Cleanup — Verification

## Scope

- Keeps exactly one ESP32-S3 INT8 build environment (`esp32-s3-n16r8`).
- Selects Tiled 3x3 with conservative guarded INT32/INT64 accumulators and Pointwise4.
- Removes Pointwise8/16 implementations, duplicate compiler definitions and obsolete experiment environments.
- Leaves existing AVR, Mega2560, FP32, file-storage/PROGMEM backends and the shared graph/buffer planner in place.

## Validation (development host, not flashed hardware)

- Production `noodle_int8_layers.cpp` byte-for-byte identical to R15 v0.6.25 Noodle kernel.
- 24 offline JavaScript regression tests pass; two fixture-dependent tests cannot run without external TFLite files.
- Native C++ graph matches the old Pointwise4 build at all 43 operator outputs on five inputs.
- Native `bus.jpeg`: class 779, 1000-byte output CRC32 a046a893.
- Sanitizer (`-fsanitize=undefined`) passes on `bus.jpeg`.
- INT32 bounds/unsafe-case tests pass.
- `params.bin`, `src/main.cpp`, `model/manifest.json`, and model/buffer plan are unchanged from the prior release.

## Hardware retest

The previously hardware-tested Tiled32+Pointwise4 baseline is 33.789 seconds (one run). Flash this cleaned release and remeasure to confirm it reproduces that result. The release was not built with PlatformIO or hardware-tested in this environment.
