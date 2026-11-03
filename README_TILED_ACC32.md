# Noodle R15 SqueezeNet INT8: tiled 3x3 INT32 experiment

This is a comparison build, **not a replacement of the measured 43.935 s baseline**.
It preserves the same CHW graph, 43 operators, Pointwise4 (1x1), XIP weights,
PSRAM branch planner, model image, and serial sender.

## Build / upload

Reference tiled 3x3 INT64 (our last confirmed hardware configuration):

```sh
pio run -e esp32-s3-n16r8-tiled64 -t upload
python3 test/send_tensor_profile.py --port /dev/ttyACM0 --image test/bus.jpeg --profile | tee tiled64.log
```

Experimental tiled 3x3 INT32 with per-filter overflow check:

```sh
pio run -e esp32-s3-n16r8-tiled32 -t upload
python3 test/send_tensor_profile.py --port /dev/ttyACM0 --image test/bus.jpeg --profile | tee tiled32.log
```

Use the correct USB CDC port for your board. The unmodified base environment
`esp32-s3-n16r8` remains the original direct-convolution configuration.

## Numeric safety

The same tiled Noodle convolution code is specialized for `int32_t` or `int64_t`
accumulators, not replaced by a new MCU-specific operator. On each output filter:

```
maxInput = max(abs(-128-input_zero_point), abs(127-input_zero_point))
worstCase = abs(bias) + 9 * input_channels * maxInput * 128
if worstCase <= INT32_MAX:
    use INT32 tiled convolution
else:
    use the original INT64 tiled convolution
```

All proof calculations use INT64. The bound is conservative, covers the
absolute value of every possible INT8 weight and input, and includes bias.
Even intermediate partial sums are bounded, so signed overflow is excluded
where INT32 is selected. Input shape and quantization are otherwise unchanged.

For this SqueezeNet model, all nine 3x3 convolution layers are safe. Maximum
computed per-filter conservative bound = **18,800,648**, compared with
INT32_MAX = **2,147,483,647**. See `model/int32_accumulator_audit.csv`.

## Expected output and current measurements

For the supplied `test/bus.jpeg`, the last confirmed ESP32-S3 tiled64 result was:

```
class=779
inference_ms=43935
output_uint8 crc32=a046a893
```

The new INT32 path was NOT yet measured on ESP32-S3 hardware. Confirm the
entire output CRC32 and compare the nine 3x3 operator timings against tiled64.
For useful timing comparison, use the same build optimization, model, input,
clock, and board; repeat several times.

## Validation performed before packaging

- Native C++ execution: matching output hashes after all 43 operators on
  five inputs (zeros, all 255, two deterministic patterns, bus.jpeg).
- Native C++ undefined-behavior sanitizer: bus.jpeg path passes unchanged.
- Explicit guard tests include safe cases and unsafe cases, e.g. INT32_MAX
  bias, very large channel counts, and extreme input zero points.
- Offline Codegen/browser regression suite: passes on runnable tests.
- XIP parameter image and firmware graph are byte-for-byte unchanged.
- ESP32-S3 PlatformIO/hardware build and timing: **not performed here**.

The existing Noodle INT8 operators, Pointwise4 path, and buffer planner stay
unchanged except for the internal selectable width of the tiled 3x3 accumulator.
