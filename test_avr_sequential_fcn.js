require('./bare_uart_bundle.js');
const assert=require('node:assert/strict');
require('./tflite_import.js');require('./lenet5_examples.js');require('./noodle_bundle.js');require('./serial_sender_bundle.js');
const C=require('./lenet_uno_codegen.js');const p=require('./example_project.json');
for(const pingpong of [false,true]){
  p.settings.apply_pingpong_sequential=pingpong;
  const f=C.generate(p).files;const src=f['lib/Noodle/src/noodle_int8_layers.cpp'];
  assert(src.includes('NDL_File weights;') && src.includes('NDL_File auxiliary;'));
  assert(!src.includes('NOODLE_INT8_AVR_FCN_SEQUENTIAL_WEIGHTS'));
  assert(!src.includes('q_avr_seek(f,(uint32_t)oc*n+base)'));
  assert(!src.includes('Serial.'));
  const app=f['src/main.cpp'];
  assert(app.includes('R15_STAGE_TIC()') && app.includes('R15_STAGE_TOC("CONV1_POOL1")'));
  assert(app.includes('R15_STAGE_TOC("CONV2_POOL2")') && app.includes('R15_STAGE_TOC("FC1")'));
  assert(!f['platformio.ini'].includes('NOODLE_INT8_AVR_FCN_SEQUENTIAL_WEIGHTS'));
}
console.log('PASS: sequential FCN is default, profiler in application only, both planner paths preserved');
