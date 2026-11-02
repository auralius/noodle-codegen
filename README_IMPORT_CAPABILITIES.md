# R15 Studio — imported TFLite compatibility

TFLite import and firmware export are separate capabilities. This patch fixes dispatch so non-SqueezeNet graphs cannot accidentally be routed to the SqueezeNet ESP32-S3 code generator.

On import, the model can be inspected and saved as a Project JSON. Firmware export is disabled when no suitable backend exists, and the Model Source panel lists model I/O, operator count and the currently missing ESP32-S3 operators. Existing supported presets remain exportable.

Existing backends: LeNet-5 fixed INT8 Uno/SD, FCN-only INT8 Uno, and SqueezeNet-class INT8/FP32 ESP32-S3 with 224x224 RGB and their respective operator profiles. New architectures such as MobileNet/LeNet with depthwise convolutions and PAD require additional Noodle kernels and an explicit exporter; no auto-conversion is implied by successfully parsing a TFLite file.

This is UI validation and dispatch correction, not a new generic MobileNet inference backend.
