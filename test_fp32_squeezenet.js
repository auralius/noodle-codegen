'use strict';
const fs=require('fs'),crypto=require('crypto'),assert=require('assert');
const imp=require('./tflite_import.js'),studio=require('./studio.js'),gen=require('./esp32s3_fp32_codegen.js');
const src=fs.readFileSync('/mnt/data/_fp32_work/squeezenet1_1_fp32.tflite');
let model=imp.inspectTFLite(src), hash=crypto.createHash('sha256').update(src).digest('hex');
const project=studio.fromTFLiteModel(model,'squeezenet1_1_fp32.tflite',src,hash);
project.settings={apply_pingpong_sequential:false};
const v=studio.validate(project);assert.deepStrictEqual(v.errors,[],v.errors.join('|'));
assert.equal(project.target,'esp32s3');assert.equal(project.layers.length,41);
const out=gen.generate(project);
console.log('FP32 model=',hash,'layers=',out.summary.layers.length,'arena=',out.summary.arenaBytes,'live=',out.summary.peakLive,'XIP=',out.summary.paramsBytes,'files=',Object.keys(out.files).length);
assert(out.summary.arenaBytes<=8*1024*1024);assert(out.summary.paramsBytes<8*1024*1024);
assert(out.files['src/main.cpp'].includes('NOODLE_SQUEEZENET_FP32_N16R8'));
const f=out.files;
for(const [name,content] of Object.entries(f)){
 const path=require('path').join('/mnt/data/_fp32_work/generated',name);fs.mkdirSync(require('path').dirname(path),{recursive:true});fs.writeFileSync(path,content);
}
fs.writeFileSync('/mnt/data/_fp32_work/squeezenet1_1_fp32_studio_project.json',JSON.stringify(project));
const saved=JSON.parse(fs.readFileSync('/mnt/data/_fp32_work/squeezenet1_1_fp32_studio_project.json'));
assert.equal(gen.generate(saved).summary.paramsBytes,out.summary.paramsBytes);
console.log('FP32 export PASS');
