'use strict';
const assert=require('node:assert/strict'),fs=require('fs'),T=require('./tflite_import.js');
const D=require('./sequential_network_detector.js');
for(const name of ['fcn_mnist16_fp32','lenet5_fp32','fcn_mnist16_int8']){
 const m=T.inspectTFLite(fs.readFileSync(__dirname+'/'+name+'.tflite'));
 const result=D.analyze(m);
 assert.equal(result.sequential,true,`${name}: ${result.reason}`);
 assert.equal(result.tensorIds.length,m.operators.length+1);
 console.log('PASS sequential:',name);
 const branched=structuredClone(m);
 if(branched.operators.length>1){
  const i=1,op=branched.operators[i],params=new Set(op.parameter_tensors||[]);
  const activeIndex=op.inputs.findIndex(id=>id>=0&&!params.has(id));
  op.inputs[activeIndex]=m.inputs[0];
  assert.equal(D.analyze(branched).sequential,false);
 }
 const reused=structuredClone(m);
 reused.operators[0].outputs[0]=m.inputs[0];
 assert.equal(D.analyze(reused).sequential,false);
}
const model=T.inspectTFLite(fs.readFileSync(__dirname+'/lenet5_fp32.tflite'));
assert.equal(D.analyze(model,{storage:[]}).sequential,false);
console.log('PASS rejects branches, aliases, and mismatched storage');
