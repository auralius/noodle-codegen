# Noodle runtime changes (experimental)

Base: user-uploaded `src(20261101-153418).zip`, 27 archived entries (25 code files).

Modified library source:
- `noodle.h`: optional `Conv.metadata_fn` for 12-byte packed quantization records.
- `noodle_int8.h`: optional scratch binding and flash Softmax declarations.
- `noodle_int8_layers.cpp`: opt-in single-handle SD fused Conv2D, file flatten, and SD FCN paths; existing public operators dispatch to these paths with `NOODLE_INT8_AVR_LOW_MEMORY=1`.
- `noodle_int8_math.cpp`: `noodle_i8_softmax_progmem`, preserving full INT8 probabilities.
- `noodle_buffer.h/.cpp`: `noodle_buffer_bind_external`, non-owning static SRAM views preserved across `noodle_buffer_require`.

No custom neural-network math is emitted in generated `src/main.cpp`. The experimental low-memory compute path belongs to Noodle; the GUI only generates bindings and calls.
