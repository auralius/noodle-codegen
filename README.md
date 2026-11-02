# Noodle Studio R15 v0.6.6 — Noodle-runtime-first Uno LeNet-5 (EXPERIMENTAL)

**Goal:** Execute the original INT8 LeNet-5 on an Arduino Uno/Nano ATmega328P with only 2 KB SRAM, without implementing CNN/FCN arithmetic in the generated application.

## Architecture

- User topology remains 9 original LeNet-5 operators; the GUI validates the sequence and storage bindings.
- Codegen exports original TFLite weights and quantization metadata as ten SD files.
- Generated `main.cpp` contains static `Conv`, `FCNFile`, tensor, and protocol bindings. **No custom Conv2D, FCN, pooling, flattening or Softmax arithmetic loops.**
- Calls existing Noodle public `noodle_i8_conv2d_fused`, `noodle_i8_flat_file`, `noodle_i8_fcn`, and new low-stack `noodle_i8_softmax_progmem`.
- The optional `NOODLE_INT8_AVR_LOW_MEMORY` backend is implemented **in Noodle's own source**: 1 live SdFat handle, packed 12-byte per-channel metadata, caller-owned 150-byte filter + 32-byte I/O scratch, exact Noodle quantization/mean pooling.
- `noodle_buffer_bind_external` is the new Noodle API for borrowed, capacity-limited SRAM buffers. This avoids re-registering scratch memory in the global arena.
- The earlier hardware-tested Uno FCN PROGMEM/SD generator remains available without the new low-memory compile flag.

## Hardware

Uno/Nano SD SPI: CS D10, MOSI D11, MISO D12, SCK D13. Use a 5V-compatible SD module with proper voltage shifting. FAT32 microSD; copy all ten generated `sdcard/*.bin` **parameter files** to SD root. Activations are written by the firmware. A pre-existing card does not need to be formatted.

## First build

From a generated firmware folder:

```sh
pio run -e uno-int8
pio run -e uno-int8 -t upload
python3 test/send_tensor.py --port /dev/ttyACM0 --manifest test/input_spec.json --pattern zeros --repeat 1 --inference-timeout 86400
```

Close any PlatformIO monitor before using Python on `/dev/ttyACM0`. Use the existing generic 784-byte sender unchanged. This firmware intentionally skips the startup scan of ten SD files (`R15_SKIP_BOOT_SD_VERIFY`) because the old boot scan exhausted memory; the inference kernels must still open and read each parameter file.

**Please share the complete `pio run` flash/SRAM size report before assuming the Uno image fits.** This artifact was not compiled with AVR GCC or flashed onto a real Uno in the authoring environment. Host results do NOT prove AVR stack headroom.

## Verification performed

- Offline GUI TFLite model import, SD parameter extraction/transposition and topology validation passed. Existing R15 FCN generator tests also pass.
- Generated C++ and Noodle sources pass host syntax compilation with a simulated Arduino/SdFat interface.
- Full host filesystem simulation executed Conv1+AvgPool1, Conv2+AvgPool2, flatten, FC1, FC2, FC3, Softmax and returned a class prediction from a deterministic synthetic 784-byte image.
- Both new low-memory pooled tensors matched **Noodle's original fused convolution/mean-pooling operator byte-for-byte** for that input.
- The original custom R15 v0.6.1 implementation returned the same final class for that input but had 136 and 32 differing pooled output bytes due to different mean-rounding semantics. The new version follows the latest Noodle operator semantics.

## Known limitations

- Experimental **fixed-shape INT8 LeNet-5 only**. Not a general CNN compiler.
- Very slow SD seeks and repeated file-open operations; correct execution is the first objective.
- Only 2 KB SRAM; even the single-handle method may still overflow the stack on AVR.
- The `Conv.metadata_fn` field and `noodle_buffer_bind_external` API are new additions to the user-supplied 2026-11-01 Noodle source. The unchanged latest user Noodle zip itself is preserved; modifications are in the bundled experimental fork.
- TFLite numerical parity (not just agreement with Noodle's original operators) and real MNIST accuracy remain to be measured.

**Pass/fail checkpoint:** first `pio run` Flash/SRAM report; then SD initialization and 784-byte serial transfer; then each `STAGE` transition; finally one `PRED` on hardware.

## Latest low-RAM update
See [README_R15_CLEANUP.md](README_R15_CLEANUP.md) for the preallocated-output policy, silent Noodle runtime, bare AVR application UART, and opt-in (unchecked by default) sequential tensor/configuration reuse.
