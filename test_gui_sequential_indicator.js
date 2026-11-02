'use strict';
const assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm');
const T=require('./tflite_import.js'),D=require('./sequential_network_detector.js');
const context=vm.createContext({NoodleTFLite:T,NoodleSequentialNetwork:D,console});
vm.runInContext(fs.readFileSync(__dirname+'/studio.js','utf8'),context,{filename:'studio.js'});
function runFor(model,expected){
 const elems={};
 context.document={getElementById:id=>elems[id]||(elems[id]={className:'',innerHTML:''})};
 context.project={target:'esp32s3',source:{...model,type:'tflite',name:'Imported',model_base64:'base64-present'},settings:{apply_pingpong_sequential:false},storage:[],layers:[],topology:''};
 const result=vm.runInContext('sourceSequentialAnalysis(project)',context);
 assert.equal(result.sequential,expected, result.reason);
 vm.runInContext('state=project;renderSequentialAnalysis()',context);
 assert.match(elems.sequentialAnalysis.innerHTML,expected?/Sequential source graph/:/Non-sequential source graph/);
 assert.match(elems.sequentialAnalysis.innerHTML,/validated separately during Codegen/);
 return elems.sequentialAnalysis.innerHTML;
}
const f=T.inspectTFLite(fs.readFileSync(__dirname+'/fcn_mnist16_fp32.tflite'));
const i=T.inspectTFLite(fs.readFileSync(__dirname+'/fcn_mnist16_int8.tflite'));
const branched=structuredClone(f);branched.operators[1].inputs[0]=f.inputs[0];
runFor(f,true);runFor(i,true);runFor(branched,false);
context.project={target:'uno',source:{...i,type:'tflite',model_base64:'base64-present'},settings:{},storage:[],layers:[],topology:''};
const crossTarget=vm.runInContext('sourceSequentialAnalysis(project)',context);
assert.equal(crossTarget.sequential,true);
console.log('PASS: GUI classification visible for FP32, INT8, non-sequential and Uno; read-only');
