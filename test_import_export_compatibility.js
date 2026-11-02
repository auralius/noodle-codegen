'use strict';
const assert=require('node:assert/strict');
const T=require('./tflite_import.js'),Z=require('./model_zoo_examples.js'),S=require('./studio.js');
const expected={lenet_int8_project:'uno_lenet5',squeezenet_int8_project:'s3_int8',squeezenet_fp32_project:'s3_fp32'};
for(const [id,backend] of Object.entries(expected)) {
 const p=Z.getProject(id);const c=S.firmwareCompatibility(p);
 assert.equal(c.supported,true,id);assert.equal(c.backend,backend,id);
}
const item=Z.getTFLite('fcn_mnist16_int8');
const bin=T.base64Decode(item.base64),m=T.inspectTFLite(bin),fcn=S.fromTFLiteModel(m,item.filename,bin,item.sha256);
assert.equal(S.firmwareCompatibility(fcn).backend,'uno_fcn');
const unsupported=structuredClone(Z.getProject('squeezenet_int8_project'));
unsupported.source.name='mobile-lenet-16-int8.tflite';
unsupported.source.tensors[unsupported.source.inputs[0]].dtype='INT8';
unsupported.source.tensors[unsupported.source.inputs[0]].shape=[1,16,16,1];
unsupported.source.operators=[...Array(7).fill(0).map(()=>({op:'CONV_2D'})),...Array(6).fill(0).map(()=>({op:'DEPTHWISE_CONV_2D'})),...Array(2).fill(0).map(()=>({op:'PAD'})),{op:'MEAN'},{op:'FULLY_CONNECTED'},{op:'SOFTMAX'}];
const blocked=S.firmwareCompatibility(unsupported);
assert.equal(blocked.supported,false);
assert.match(blocked.detail,/DEPTHWISE_CONV_2D × 6/);
assert.match(blocked.detail,/PAD × 2/);
assert.match(blocked.detail,/FULLY_CONNECTED × 1/);
assert.match(blocked.detail,/SOFTMAX × 1/);
assert.doesNotMatch(blocked.reason,/UINT8 RGB input required/);
unsupported.target='uno';
assert.equal(S.firmwareCompatibility(unsupported).supported,false);
console.log('PASS: all four Model Zoo backends remain available, unsupported MobileNet/LeNet-like imported graph is inspection-only and lists missing export bindings.');
