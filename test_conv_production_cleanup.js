'use strict';
const assert=require('node:assert/strict'), fs=require('node:fs'), vm=require('node:vm');
const gen=require('./esp32s3_codegen.js');
const p=require('./model_zoo_examples.js').getProject('squeezenet_int8_project');
const result=gen.generate(p), f=result.files;
const ini=f['platformio.ini'], cc=f['lib/Noodle/src/noodle_int8_layers.cpp'];
assert.equal((ini.match(/\[env:/g)||[]).length,1);
for(const token of [
    '-D NOODLE_INT8_POINTWISE_PIXELS=4',
    '-D NOODLE_INT8_CONV_IMPL=1',
    '-D NOODLE_INT8_CONV3X3_ACC32=1']) assert(ini.includes(token),token);
assert.equal((ini.match(/NOODLE_INT8_POINTWISE_PIXELS=/g)||[]).length,1);
assert.doesNotMatch(ini,/pw8|pw16|tiled64|scalar|\-U(?:NOODLE| NOODLE)/i);
assert.doesNotMatch(cc,/q_conv2d_pointwise_chw_wide|q_pointwise_wide_acc32_safe/);
assert.match(cc,/NOODLE_INT8_POINTWISE_PIXELS == 4/);
assert.match(cc,/q_conv2d_filter_streaming_3x3_impl<int32_t>/);
assert.match(cc,/q_conv2d_filter_streaming_3x3_impl<int64_t>/);
assert.match(cc,/q_conv2d_filter_direct/);
assert.equal(cc,fs.readFileSync('lib/Noodle/src/noodle_int8_layers.cpp','utf8'));
assert.equal(result.summary.operators,43);
assert.equal(result.summary.physicalBuffers,3);
assert.equal(result.summary.allocatedBytes,1369344);
assert.equal(result.summary.peakLiveBytes,982144);
assert(!Object.keys(f).some(k=>/noodle_i8_nhwc\.(h|cpp)$/.test(k)));
const scope={console,TextEncoder,TextDecoder,Uint8Array,ArrayBuffer,DataView,atob,btoa};
scope.globalThis=scope;
const ctx=vm.createContext(scope);
for(const n of ['tflite_import.js','noodle_bundle.js','branch_buffer_planner.js','sequential_network_detector.js','s3_serial_sender_bundle.js','esp32s3_chw_transform.js','esp32s3_codegen.js'])
  vm.runInContext(fs.readFileSync(n,'utf8'),ctx,{filename:n});
ctx.p=p;
const browser=vm.runInContext('NoodleESP32S3.generate(p)',ctx).files;
assert.equal(browser['platformio.ini'],ini);
assert.equal(browser['lib/Noodle/src/noodle_int8_layers.cpp'],cc);
assert(Buffer.from(browser['model/params.bin']).equals(Buffer.from(f['model/params.bin'])));
console.log('PASS: one production environment, Pointwise4, guarded tiled32, generic fallback, unchanged graph/buffers, offline browser parity');
