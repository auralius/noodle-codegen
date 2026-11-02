# R15 v0.6.5 — AVR INT8 sequential FCN

- Default `q_avr_packed_fcn` streams `w06.bin` etc. in output-major order with one persistent weight handle; auxiliary handle is reused for input and metadata.
- Removes old per-block reopen/seek implementation from the bundled Noodle runtime. No new compile flag required.
- Codegen's **Apply ping-pong for sequential network** checkbox is still opt-in/unchecked by default. It changes tensor/config descriptor reuse, not sequential weight streaming.
- Optional `-D NOODLE_APP_LAYER_TIMING=1` emits `TIME <STAGE> <ms>` from generated application code, never from Noodle. Leave disabled for final latency comparisons.
- Same input protocol, model, weights, preallocated SD files, SRAM buffers, and runtime quantization.
- The previous v0.6.4 GUI and two-handle experiment are available as rollback artifacts.
- Host-side simulation checks arithmetic equivalence; AVR compilation and 10-image hardware tests must still be repeated on your board after export.
