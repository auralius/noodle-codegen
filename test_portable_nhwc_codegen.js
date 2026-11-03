'use strict';
// Migration regression: generic Noodle INT8 NHWC + NoodleTensor/NoodleBuffer bindings.
const fs=require('fs'),assert=require('node:assert/strict'),vm=require('vm');
const P=require('./esp32s3_codegen.js');
const src=process.argv[2];
const header=fs.readFileSync('lib/Noodle/src/noodle_i8_nhwc.h','utf8');
const kernel=fs.readFileSync('lib/Noodle/src/noodle_i8_nhwc.cpp','utf8');
assert.match(header,/NoodleTensor \*storage/);
assert.match(kernel,/noodle_i8_nhwc_execute\(/);
assert.doesNotMatch(kernel,/#include\s*<(?:Arduino|esp_)/);
assert.match(kernel,/#  define NOODLE_I8_NHWC_POINTWISE_PIXELS 4/);
if (src) {
  const project=JSON.parse(fs.readFileSync(src,'utf8'));
  const generated=P.generate(project);
  const f=generated.files,code=f['src/main.cpp'];
  assert.match(code,/#include <noodle_i8_nhwc.h>/);
  assert.match(code,/noodle_i8_nhwc_execute\(/);
  assert.match(code,/static NoodleTensor physicalTensors\[3\]/);
  assert.match(code,/noodle_buffer_bind_external\(/);
  assert(!Object.hasOwn(f,'lib/Noodle/src/noodle_s3_nhwc_i8.cpp'));
  assert.equal(f['lib/Noodle/src/noodle_i8_nhwc.cpp'],kernel);
  assert.equal(f['lib/Noodle/src/noodle_i8_nhwc.h'],header);
  assert.equal(generated.summary.physicalBuffers,3);
  assert.equal(generated.summary.allocatedBytes,1369344);
  assert.equal(generated.summary.operators,43);
  const old=structuredClone(project);old.settings={...(old.settings||{}),reuse_branch_buffers:false};
  const baseline=P.generate(old);
  assert.equal(baseline.summary.physicalBuffers,44);
  assert.match(baseline.files['src/main.cpp'],/static NoodleTensor physicalTensors\[44\]/);
  assert(Buffer.from(f['model/params.bin']).equals(Buffer.from(baseline.files['model/params.bin'])));
  const scope={console, TextEncoder,TextDecoder, Uint8Array,ArrayBuffer,DataView,atob,btoa};scope.globalThis=scope;
  const ctx=vm.createContext(scope);
  for(const name of ['tflite_import.js','noodle_bundle.js','noodle_i8_nhwc_bundle.js','branch_buffer_planner.js','sequential_network_detector.js','s3_serial_sender_bundle.js','esp32s3_codegen.js']){
    vm.runInContext(fs.readFileSync(name,'utf8'),ctx,{filename:name});
  }
  ctx.project=project;
  const exported=vm.runInContext('NoodleESP32S3.generate(project)',ctx);
  assert.equal(exported.files['lib/Noodle/src/noodle_i8_nhwc.cpp'],kernel);
  assert.equal(exported.files['lib/Noodle/src/noodle_i8_nhwc.h'],header);
  assert.equal(exported.summary.allocatedBytes,generated.summary.allocatedBytes);
  console.log('PASS: generated SqueezeNet 3-buffer NHWC NoodleTensor path, legacy 44-buffer fallback, 43 operators, identical XIP weights, browser parity');
} else console.log('PASS: portable NHWC API and default Pointwise4');
