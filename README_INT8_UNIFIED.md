# Noodle R15 v0.6.24 — unified native CHW INT8

This is the canonical SqueezeNet INT8 path. No GUI checkbox or old project-setting flag is needed.

- `noodle_int8.h`: native INT8 graph descriptor/API.
- `noodle_int8_api.cpp`: graph dispatch, quantization and elementwise operations.
- `noodle_int8_layers.cpp`: original Noodle convolution, pooling, concat and integrated Pointwise4 specialization.
- `noodle_int8_math.cpp`: common quantization arithmetic.
- No dedicated `noodle_i8_chw.*`, `noodle_i8_nhwc.*` or `noodle_s3_nhwc_i8.*` C++ layer files are emitted.
- All internal activations use CHW. The 224×224 RGB serial input still arrives as NHWC byte order; receiver scatters it into CHW.
- TFLite Conv2D weights are converted OHWI→OIHW, updating the XIP parameter CRC. Flash matching `params.bin`.
- Automatic birth–death planner retains three physical buffers for the provided 43-operator SqueezeNet model.

This changes the default code-generation policy, **not** convolution kernels, buffer assignments, or the FP32 SqueezeNet path. Hardware correctness/performance remains to be verified.
