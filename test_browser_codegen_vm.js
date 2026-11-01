const fs=require('fs'), vm=require('vm'),assert=require('node:assert/strict');
const scope={console,TextEncoder,TextDecoder,Uint8Array,ArrayBuffer,DataView,atob,btoa,setTimeout:undefined};
scope.globalThis=scope;const context=vm.createContext(scope);
for(const name of ['tflite_import.js','noodle_bundle.js','serial_sender_bundle.js','bare_uart_bundle.js','firmware_codegen.js','lenet5_examples.js','lenet_uno_codegen.js','studio.js']){
  vm.runInContext(fs.readFileSync(__dirname+'/'+name,'utf8'),context,{filename:name});
}
context.project=JSON.parse(fs.readFileSync(__dirname+'/example_project.json','utf8'));
const files=vm.runInContext('NoodleLeNetUno.generate(project)',context);
assert.equal(files.summary.layers.length,9);
assert.equal(files.files['test/input_spec.json'].includes('784'),true);
assert.equal(files.files['test/send_tensor.py'].startsWith('#!/usr/bin/env python3'),true);
assert.equal(files.files['lib/Noodle/src/noodle_int8_layers.cpp'].length>0,true);
const zip=vm.runInContext('NoodleCodegen.zip(NoodleLeNetUno.generate(project).files)',context);
assert(zip.byteLength>100000);
console.log('PASS: browser-global offline v0.6 export includes all LeNet ops, generic sender, full Noodle library and self-contained ZIP ('+zip.byteLength+' bytes).');
