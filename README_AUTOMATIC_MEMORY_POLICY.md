# R15 v0.6.23 — Automatic memory strategy selection in the GUI

The manual **Apply ping-pong for sequential network** checkbox is removed. The offline GUI chooses the existing allocation mechanism when generating firmware:

- Strict sequential TFLite graphs supported by the generic ESP32-S3 FP32 exporter, Uno FCN or experimental Uno LeNet5 exporter: automatically request the already-implemented A/B ping-pong strategy.
- ESP32-S3 INT8 SqueezeNet: retain automatic branch-aware reuse with the existing birth/death timeline; no checkbox.
- ESP32-S3 FP32 SqueezeNet: retain its existing lifetime-aware allocation.
- Mega2560 Mobile-LeNet and other exporters: retain their tested specialized placement; **no generic planner is claimed** for them.

The strategy is selected from the imported graph at firmware generation time. Each exporter still validates the edited topology, memory and execution constraints. An incompatible project is rejected or uses the existing safe path according to the backend. Historical `apply_pingpong_sequential` values in imported JSON are ignored by the GUI, and new saved projects omit that setting. The old generator flag is retained internally for backward compatibility with existing programmatic tests.

**No Noodle kernels, parameter binaries, layer code, or allocation algorithms were changed.** This is a GUI policy-selection update.

### SqueezeNet INT8 layout
Native Noodle CHW is automatic in v0.6.24; the legacy NHWC checkbox was removed. The model's incoming RGB serial bytes remain in standard NHWC order.
