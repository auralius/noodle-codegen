> Historical behavior for the release named below. As of **v0.6.23**, the GUI no longer exposes the ping-pong checkbox; see [`README_AUTOMATIC_MEMORY_POLICY.md`](README_AUTOMATIC_MEMORY_POLICY.md). Existing direct Codegen APIs retain their legacy flags for compatibility.

# ESP32-S3 FP32 sequential ping-pong experiment (v0.6.14)

On a supported FP32 TFLite graph, **Apply ping-pong for sequential network** now selects two alternating regions (A/B) in one PSRAM allocation. No in-place operator aliases input and output. Each region is sized by the largest of the tensors mapped to that region. The original static liveness arena remains the default when unchecked, preserving SqueezeNet behavior.

Ping-pong only supports strict single-input chains. Graphs with multiple activation inputs, residual/skip connections, branches, repeated activation tensor IDs, or unused storage declarations are rejected with an explicit message. This does **not** add new operators or INT8 ESP32-S3 memory policy.

The full GUI includes the existing Model Zoo, Pointwise4, and generic FP32 generation. The FP32 generator also now supplies an explicit firmware ZIP filename for custom imported FP32 models.

**Validation:** Node codegen regression tests; additional native FP32 arithmetic regression and hardware ESP32-S3 execution should be performed before treating this as a released performance optimization.

As of v0.6.15, the strict graph detector is shared in `sequential_network_detector.js`; ESP32-S3 FP32 memory planning remains unchanged. See `README_SHARED_SEQUENTIAL_DETECTOR.md`.
