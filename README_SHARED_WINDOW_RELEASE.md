# Noodle Studio R15 v0.6.6 — AVR shared-window convolution

## Changes

- Integrates the 2x2 pooling shared-window AVR INT8 convolution path with compile-time `NOODLE_INT8_AVR_CONV_SHARED_WINDOW=1` (on for supported LeNet-5 exports).
- Reuses the 84-byte later-stage activation scratch as four aligned 32-bit accumulators during Conv1/Conv2. This does not require an additional permanent 16-byte scratch array.
- Reuses the existing application I/O scratch for 12-byte per-channel convolution metadata.
- Keeps the v0.6.5 sequential-weight, two-file-handle AVR FCN path as default.
- Keeps Noodle itself silent, with application-only optional stage timing. Temporary `ERR_CONV_DETAIL` debug/status API is deliberately omitted.
- Preserves the unchecked-by-default *Apply ping-pong for sequential network* option and the original unoptimized path for non-sequential models.
- Keeps `NOODLE_FS_PREALLOCATED=1` for the supported LeNet-5 export and generates the usual activation files.

## Current measured baseline (user's hardware)

The predecessor v0.6.5 shared-window + accumulator-reuse trial completed ten consecutive MNIST **training-set** inferences: 10/10 predictions matched, mean ~24.349 s (vs. ~40.09 s without shared windows, ~57.98 s original). These are hardware results for the trial, **not yet validation of this generated v0.6.6 export**. The trial also showed intermittent failures before the successful 10-image run. Validate multiple cold starts and use unseen test samples before calling the new build fully stable.

## Deployment

- The standard Codegen PlatformIO configuration uses the Uno serial bootloader. Existing Nano + USBasp users must retain their USBasp upload/fuse settings.
- Select *Apply ping-pong for sequential network* for the 2-KB ATmega328P LeNet deployment. It remains unchecked by default.
- Copy all exported `sdcard/` files to the card root.
- Add `-D NOODLE_APP_LAYER_TIMING=1` if per-stage timing is desired.
- To disable the new optimization while leaving sequential FCN streaming intact, remove `-D NOODLE_INT8_AVR_CONV_SHARED_WINDOW=1` from `platformio.ini`.

This release has host-side Codegen checks, but does not claim a compiled AVR binary or live hardware validation of the freshly generated package.
