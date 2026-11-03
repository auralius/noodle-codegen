# Noodle R15 v0.6.24 — canonical CHW INT8 export

SqueezeNet INT8 now always exports unified CHW through the existing Noodle INT8 operators.
The retired experimental checkbox is removed. Previous saved projects with `experimental_chw=false` or `true` generate the same CHW firmware. The original serial input protocol remains NHWC RGB, and Codegen converts it to internal CHW while receiving bytes. The OHWI→OIHW weight conversion and CRC remain required.

No `noodle_i8_nhwc.*`, `noodle_s3_nhwc_i8.*`, or `noodle_i8_chw.*` operator files are included in generated INT8 firmware. The FP32 SqueezeNet backend remains a separate, unchanged path.

The branch planner and Pointwise4 kernel are unchanged. Full ESP32-S3 hardware validation is still necessary; keep an archived copy of the working v0.6.23 package for rollback.
