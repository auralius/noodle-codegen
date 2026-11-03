# R15 v0.6.18 — branch-aware reusable activation buffers

This change is **restricted to the ESP32-S3 generic INT8 (W8A8) SqueezeNet Codegen backend**. It does not modify the sequential/ping-pong path, the FP32 backends, Uno/Nano/Mega exporters, or any Noodle convolution/kernel function.

## Why

Previously, each activation in the imported SqueezeNet graph had a separate PSRAM allocation that stayed allocated throughout inference. The logical tensor graph still requires all 44 tensor descriptors, but not 44 simultaneously resident physical buffers.

Codegen now uses the validated TFLite operator execution order to compute the birth and last consuming operation of each activation. It allocates a tensor's output in a free physical buffer only when the buffer's previous tensor was last used in an **earlier** operator, never in the same operation. This conservatively prevents input/output aliases and preserves both inputs during `CONCATENATION`. Physical buffer capacities are planned from maximum tensor sizes at Codegen time, so no user-managed resize is needed.

### SqueezeNet 1.1 INT8 (44 tensors, 43 operators)

| Metric | Legacy | Branch reuse |
| --- | ---: | ---: |
| Logical tensor descriptors | 44 | 44 |
| Physical PSRAM buffers | 44 | 3 |
| Requested activation storage | 4,975,136 B | 1,369,344 B |
| Computed peak concurrently live tensor bytes | — | 982,144 B |
| XIP parameter image | 1,278,904 B | 1,278,904 B |
| Noodle operator calls | 43 | 43 |

The two memory totals above are **requested allocation capacities**. Neither is an instrumented ESP32-S3 runtime heap measurement; allocator metadata, fragmentation, and runtime scratch are excluded.

## Output and visibility

The generated firmware keeps `NoodleS3Tensor` descriptors for each logical tensor, but assigns `.data` to one of several reusable physical PSRAM allocations. The generated ZIP includes `model/buffer_plan.json` with logical tensor intervals, physical buffer assignments and reserved sizes, and `model/manifest.json` reports the chosen memory strategy.

The GUI shows the physical buffer count and storage reduction after exporting. The planner is backend-neutral, but in this release its allocation output is connected **only** to the ESP32-S3 INT8 generator for non-sequential graphs. The existing ESP32-S3 FP32 generator already has its own lifetime planning and remains unchanged. Standard `NoodleTensor` managed grow-only allocation is not used by this specialized S3 NHWC kernel path; its buffers are pre-sized by Codegen based on known shapes. No new neural-network operators are introduced.

## Revert/fallback

For direct A/B regression comparisons, set in the saved project JSON:

```json
"settings": {"reuse_branch_buffers": false}
```

This reproduces the original one-PSRAM-allocation-per-tensor strategy. The setting defaults to enabled for non-sequential INT8 graphs only. Strictly sequential graph behavior remains on its existing path.

## Validation

- Pure JavaScript tests verify liveness, branch/concat overlap, malformed graph rejection, deterministic assignments and backwards compatibility.
- On the original SqueezeNet-1.1 W8A8 graph, the XIP parameter binary, all 43 Noodle op descriptors, and serial input/output metadata are unchanged.
- Native x86 C++ regression using **the existing Noodle INT8 NHWC operator source** and three distinct inputs compared all 43 per-operation output tensor hashes: all were identical between legacy and reuse allocation. The native-only build replaced Arduino dependencies with a standalone copy of Noodle quantization arithmetic; the operator source itself was not otherwise changed.
- Hardware build, PSRAM runtime measurements, repeated camera inference, and on-device output equivalence remain to be validated on ESP32-S3 N16R8. Do not claim hardware-verified timing or buffer savings yet.

## v0.6.22 lifetime analysis

An explicit per-day birth/death timeline, maximum simultaneous live tensors, and backend-specific buffer-count lower bound are now included in `model/buffer_plan.json` and CSV reports. See `README_BIRTH_DEATH_TIMELINE.md`. **The physical buffer assignments do not change.**
