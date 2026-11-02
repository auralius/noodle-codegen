require('./bare_uart_bundle.js');
const assert=require('node:assert/strict');
require('./tflite_import.js');require('./lenet5_examples.js');require('./noodle_bundle.js');require('./serial_sender_bundle.js');
const C=require('./lenet_uno_codegen.js');
const p=require('./example_project.json');
for(const pingpong of [false,true]){
  p.settings={...(p.settings||{}),apply_pingpong_sequential:pingpong};
  const f=C.generate(p).files;
  const l=f['lib/Noodle/src/noodle_int8_layers.cpp'],h=f['lib/Noodle/src/noodle_int8.h'],app=f['src/main.cpp'],ini=f['platformio.ini'];
  assert(ini.includes('NOODLE_INT8_AVR_CONV_SHARED_WINDOW=1'));
  assert(ini.includes('NOODLE_FS_PREALLOCATED=1'));
  assert(l.includes('q_avr_conv_pool_2x2_shared'));
  assert(l.includes('noodle_i8_avr_bind_conv_accumulators(int32_t *acc)'));
  assert(h.includes('noodle_i8_avr_bind_conv_accumulators(int32_t *acc)'));
  assert(l.includes('NDL_File weights;') && l.includes('NDL_File auxiliary;'));
  assert(!l.includes('q_avr_conv_fault') && !l.includes('ERR_CONV_DETAIL') && !l.includes('Serial.'));
  assert(app.includes('union R15SharedScratch'));
  assert(app.includes('noodle_i8_avr_bind_conv_accumulators(r15_shared_scratch.conv_acc)'));
  assert(app.includes('r15_shared_scratch.hidden84'));
  assert(!app.includes('ERR_CONV_DETAIL'));
  assert(app.includes('R15_STAGE_TOC("CONV2_POOL2")'));
  assert.equal(f['sdcard/c03.bin'].length,2400);
  assert.equal(f['sdcard/p1.bin']?.length||f['sdcard/t13.bin']?.length,1176);
  if(pingpong) {
    assert(app.includes('static NoodleTensor tensor_a;'));
    assert(!app.includes('static Conv conv1_cfg, conv2_cfg;'));
  } else {
    assert(app.includes('static Conv conv1_cfg, conv2_cfg;'));
    assert(!app.includes('static NoodleTensor tensor_a;'));
  }
}
console.log('PASS v0.6.6: shared-window + scratch union in both LeNet export modes, sequential FCN, silent runtime, SD preallocation');
