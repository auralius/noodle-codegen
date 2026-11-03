# Noodle Studio R15 v0.6.22 — birth–death buffer lifetime diagnostics

This revision extends the **existing** branch-aware planner with an explicit, deterministic day-by-day lifetime report. It **does not alter buffer allocation, layer operators, SqueezeNet weights, quantization, the Pointwise4 kernel, or the sequential ping-pong path**.

## Day model

- `day = 0`: input tensor is present, before the first operator.
- `day = i + 1`: execution of operator index `i`, for `i` from `0` through `N-1`.
- `day = N + 1`: retained model output, after the final operator.
- Each tensor has `bornDay` and `dieDay` (both inclusive). An activation remains live on its last-consuming operator's day; that operator's output is also considered live that day. The planner does **not** assume in-place kernels.
- The original `born`/`lastUsed` fields are preserved (old operator index scheme, `-1` for model inputs and `N` for retained outputs) to maintain compatibility with existing tools.

For day `d`, define the set of simultaneously live tensors:

    L(d) = { t : t.bornDay <= d <= t.dieDay }
    maxLiveTensors = max_d |L(d)|
    peakLiveBytes   = max_d sum_{t in L(d)} t.bytes

A buffer may be reused only when its previously assigned tensor **dies strictly before** the new tensor is born. The existing greedy best-fit/largest-fit assignment remains unchanged.

For a single compatible backend, `maxLiveTensors` is the exact minimum number of non-aliasing physical buffers required by the interval model. For multiple independent backends, the buffer-count lower bound is instead the **sum of each backend's maximum live count**; those maxima can occur on different days. Operator scratch, non-activation memory, allocator metadata, and any additional aliasing constraints are not included.

## Generated model diagnostics

- `model/buffer_plan.json`: legacy `lifetimes` and physical assignments, plus `bornDay`/`dieDay`, `maxLiveTensors`, `maxLiveDays`, `peakLiveByteDays`, `minimumPhysicalBuffers`, `maximumLivePerBackend`, and the complete `timeline`.
- `model/lifetime_by_day.csv`: per-day number of live tensors, live bytes, and tensor IDs (easy to plot in Python, Excel, or a paper).
- `model/tensor_lifetimes.csv`: each tensor's birthday, death day, size, backend, and assigned physical buffer.
- `model/manifest.json`: includes the maximum-live count and theoretical buffer-count lower bound.
- The GUI generation status now shows maximum simultaneous live tensors, peak-live bytes, and minimum buffer count.

## SqueezeNet-1.1 INT8 example, 43 operators

| Metric | Result |
| --- | ---: |
| Logical activation tensors | 44 |
| Max simultaneously live tensors | 3 |
| Min possible physical buffers (interval model) | 3 |
| Planner physical buffers | 3 |
| Peak live activation bytes | 982,144 |
| Planner physical buffer capacity | 1,369,344 |
| Legacy physical buffer capacity | 4,975,136 |
| Reduction in requested activation storage | 72.5% |

The planner meets the **buffer-count** lower bound, but the sum of pre-reserved physical buffer capacities is above the *peak-live-byte* lower bound. It is not proven memory-capacity optimal. All numbers here are compile-time analyses, **not** instrumented PSRAM measurements.

## Suggested methods text for a paper

> We represent the execution sequence as discrete operator days. Each logical activation is annotated with its birth day (producing operator) and death day (last consuming operator), both inclusive. Counting the live intervals at each day yields the maximum number of simultaneously live tensors and the peak live activation bytes. Buffer reuse is permitted only when the previous activation's death day precedes the next activation's birth day, preventing input–output aliasing within an operator. For storage backends that cannot share memory, the per-backend peak live counts are added to obtain a lower bound on physical buffer count. The allocator then assigns logical activations to reusable physical buffers using deterministic greedy heuristics, selecting the candidate with lower retained capacity. This analysis takes place entirely in Codegen, without runtime lifetime tracking.

## Reproducibility and scope

Run `node test_branch_buffer_planner.js` and `node test_chw_unification.js`. The diagnostic is deliberately read-only. The ESP32-S3 INT8 generator uses unified CHW by default as of v0.6.24; the birth/death diagnostic has not changed physical tensor assignments or numerical operators. Uno, Mega2560, ESP32-S3 FP32 and sequential ping-pong implementations are not modified. Hardware validation remains separate.
