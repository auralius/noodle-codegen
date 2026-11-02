
'use strict';
const fs=require('node:fs'), assert=require('node:assert/strict');
const bundle=require('./esp32s3_codegen.js');
const p=JSON.parse(fs.readFileSync(process.argv[2]||'/mnt/data/SqueezeNet_1_1_W8A8_Studio_Project.json','utf8'));
const e=bundle.generate(p);
assert.equal(e.summary.operators,43);
const ini=e.files['platformio.ini'];
assert.match(ini,/\[env:esp32-s3-n16r8-scalar\]/);
assert.match(ini,/NOODLE_S3_I8_POINTWISE_PIXELS=0/);
assert.match(ini,/\[env:esp32-s3-n16r8-o2\]/);
const cpp=e.files['lib/Noodle/src/noodle_s3_nhwc_i8.cpp'];
assert.match(cpp,/#define NOODLE_S3_I8_POINTWISE_PIXELS 4/);
assert.match(cpp,/return pointwise_1x1/);
assert.match(cpp,/if\(op.kh==1 && op.kw==1 && op.stride_h==1 && op.stride_w==1/);
assert.match(cpp,/for\(uint16_t y=0;y<b.h;y\+\+\)/);
assert.equal(cpp,fs.readFileSync('lib/Noodle/src/noodle_s3_nhwc_i8.cpp','utf8'));
assert.equal(e.files['model/params.bin'].length,1278904);
console.log('PASS Pointwise4 default; scalar regression fallback; 43-op graph; exact source parity; model image');
