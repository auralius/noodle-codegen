# R15 v0.6.4 — Noodle/codegen cleanup + opt-in ping-pong

This revision is based on the supplied latest working LeNet-5 firmware and the supplied Noodle Studio GUI.

## Library boundaries

- `lib/Noodle` is the library embedded in `noodle_bundle.js`. The `.cpp` and `.h` source in the bundled runtime comes from the user's latest firmware revision.
- **No serial printing in Noodle**: removed experimental FCN/Conv progress, checkpoints, and direct UART diagnostics. Error and stage reporting is only in generated application code.
- `NOODLE_FS_PREALLOCATED=1` is the single preallocated-output policy. In this mode, `noodle_fs_prepare_output_into` does no filesystem open/check during an operator call. Streaming writers still open existing files as needed. The size and existence of output files are an **export/deployment contract**; Noodle does not verify file sizes on AVR under this policy.
- `NOODLE_FS_PREALLOCATED=0` remains the default for other builds and retains normal create/truncate behavior.
- `noodle_fs_open_read_into`, `noodle_fs_open_write_existing_into` and `noodle_fs_open_write_only_existing_into` are the reference-based filesystem helpers.

## Code generation

- Specialized sequential LeNet-5 path: exports correct-size zero-filled `sdcard/in.bin`, `p1.bin`, `p2.bin`, `flat.bin` (or matching project-assigned SD paths), along with all parameters; optional ping-pong (two tensor descriptors; one reusable Conv config and one FCNFile config) or individual descriptors/configurations when unchecked; existing scratch reuse; AVR-only polling UART in the **application**, not the library.
- Generic sequential FCN/Softmax path: only when the **Apply ping-pong for sequential network** checkbox is checked does a recognized simple linear chain use two ping-pong `NoodleTensor` descriptors plus two fixed external SRAM activation arrays. With the checkbox unchecked (the default), or for nonlinear/destroy/fan-out cases, the previous explicit buffer allocation path remains. This is not a general graph memory planner.
- The generated LeNet `platformio.ini` defaults to a standard Uno bootloader. If the Nano uses USBasp and the full-flash fuse settings, transfer your existing upload configuration, including `pio_usbasp_fullflash.py` if needed.
- Existing `.bin` files must be copied to the microSD **root**, not as a nested `sdcard/` folder.

## Tests and limitations

Node host-side tests verify the export, file sizes, configuration wiring, and browser ZIP. The earlier real Uno successful inference used a different debugging build. **The cleaned output is not yet compiled with PlatformIO or tested on hardware.** In particular, disabling progress output may change flash usage and performance; benchmark again after testing predictions against MNIST/TFLite.

## Opt-in memory reuse (new in v0.6.4)

- UI: **Apply ping-pong for sequential network** under **Topology**, unchecked by default. The option is saved as `settings.apply_pingpong_sequential` in the project JSON. Old JSON files load with the option off.
- **Checked, strict linear chain:** Codegen may reuse two tensor descriptors. The fixed LeNet exporter additionally reuses one Conv and one FCNFile configuration, matching the low-RAM Uno firmware.
- **Unchecked, any graph:** Codegen keeps the original explicit tensor descriptors/configuration layout.
- **Checked, non-sequential/fan-out/destroy:** The generic FCN exporter falls back to explicit buffers and reports that ping-pong was not applicable. There is no implicit reordering or liveness-based graph planner.
- Only the already supported exportable graphs can generate firmware: arbitrary residual Add/Concat models are still not supported by the current Uno generator, regardless of this checkbox.
- **Uno caution:** Unoptimized LeNet-5 might exhaust 2 KB SRAM. Enable ping-pong for the previously demonstrated low-RAM deployment.
