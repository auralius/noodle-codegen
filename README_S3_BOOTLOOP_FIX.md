# SqueezeNet ESP32-S3-N16R8 flash-size fix

ESP32-S3 Codegen adds `board_upload.flash_size = 16MB` to generated `platformio.ini`, in addition to `board_build.flash_size = 16MB`. This avoids reliance on the generic ESP32-S3 board manifest flash size and addresses a known N16R8 early boot-loop configuration issue. The Arduino Uno/Nano exporter is unchanged.

Regenerate the firmware ZIP, then run `pio run -e esp32-s3-n16r8 -t clean` followed by `pio run -e esp32-s3-n16r8 -t upload`. The generated extra image uploader continues to include model parameters at `0x800000`.
