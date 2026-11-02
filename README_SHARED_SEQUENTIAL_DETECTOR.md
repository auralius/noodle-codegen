# Shared sequential-network detector — R15 v0.6.15 (conservative integration)

`sequential_network_detector.js` exposes `NoodleSequentialNetwork.analyze(model, options)` in the browser, and `require('./sequential_network_detector.js').analyze(...)` in Node.

The detector is pure, graph-based, target-independent, and dtype-independent. It identifies a strict chain with exactly one graph input/output, one activation input/output per operator, and unique activation tensor IDs. It optionally validates an exact set of storage declarations. Return value is `{sequential: true, tensorIds, operatorCount}` or `{sequential: false, reason, operatorIndex}`.

**Scope:** Only the existing ESP32-S3 generic FP32 ping-pong planner uses this shared detector in v0.6.15. Its A/B memory sizing, activation offsets, firmware emission, manifest, GUI option and default liveness plan are unchanged. Other backends remain unchanged and do not automatically enable ping-pong. This intentionally avoids changing currently working INT8 and MCU targets.

**Important:** A sequential graph does not on its own guarantee a runtime operator supports non-overlapping input/output buffers, or that the target has sufficient memory. Backends must verify these separately before enabling ping-pong.

## Validation

Run `node test_sequential_network_detector.js`, `node test_s3_fp32_pingpong.js`, and `node test_generic_fp32_lenet.js`.

The general regression suite also ran; three pre-existing tests require external model/project files not bundled with this archive (`test_fp32_squeezenet.js`, `test_pointwise4_release.js`, `test_squeezenet_uint8.js`). No physical hardware verification has been performed for this refactor.

## R15 v0.6.16 — GUI visibility (read-only)

The Topology panel now shows the classification of the **imported source TFLite graph** for either FP32 or INT8, independent of selected MCU. It displays `Sequential source graph` or `Non-sequential source graph`, with the detector's reason for rejection. The status is informational only: changing board selection, importing a model, or toggling the existing checkbox does not turn on ping-pong for unrelated backends. In ESP32-S3 generic FP32, the checkbox remains opt-in; actual topology/storage suitability is validated at firmware generation. Edited topology is not the source graph and is deliberately not silently classified as the original network.

The regression tests and all firmware generator source files are unchanged except for GUI and documentation. No physical hardware verification implied.
