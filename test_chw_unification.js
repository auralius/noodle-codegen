'use strict';
const fs=require('fs'),assert=require('node:assert/strict'),vm=require('vm');
const gen=require('./esp32s3_codegen.js');
const project=require('./model_zoo_examples.js').getProject('squeezenet_int8_project');
const stable=gen.generate(project);
assert.equal(stable.summary.operators,43);
assert.equal(stable.summary.physicalBuffers,3);
assert.equal(stable.summary.allocatedBytes,1369344);
assert.equal(stable.summary.maxLiveTensors,3);
assert.equal(stable.summary.peakLiveBytes,982144);
assert.equal(stable.summary.convertedConvolutions,26);
assert.equal(stable.summary.unifiedCHW,true);
assert.match(stable.files['src/main.cpp'],/noodle_i8_graph_execute/);
assert.match(stable.files['src/main.cpp'],/#include <noodle_int8.h>/);
assert(!Object.keys(stable.files).some(f=>/nhwc.*\.(?:cpp|h)$/i.test(f)));
assert(!Object.hasOwn(stable.files,'lib/Noodle/src/noodle_i8_chw.cpp'));
assert(!Object.hasOwn(stable.files,'lib/Noodle/src/noodle_i8_nhwc.cpp'));
assert.equal(stable.files['test/input_spec.json'].includes('"layout": "NHWC"'),true); // WIRE ONLY
assert.equal(stable.files['test/input_spec.json'].includes('"internal_activation_layout": "CHW"'),true);
for(const legacy of [undefined,false,true]) {
 const v=structuredClone(project);
 v.settings={...(v.settings||{})};
 if(legacy!==undefined)v.settings.experimental_chw=legacy;
 const out=gen.generate(v);
 assert.equal(out.files['src/main.cpp'],stable.files['src/main.cpp']);
 assert(Buffer.from(out.files['model/params.bin']).equals(Buffer.from(stable.files['model/params.bin'])));
}
const scope={console,TextEncoder,TextDecoder,Uint8Array,ArrayBuffer,DataView,atob,btoa};scope.globalThis=scope;
const ctx=vm.createContext(scope);
for(const name of ['tflite_import.js','noodle_bundle.js','branch_buffer_planner.js','sequential_network_detector.js','s3_serial_sender_bundle.js','esp32s3_chw_transform.js','esp32s3_codegen.js'])
 vm.runInContext(fs.readFileSync(name,'utf8'),ctx,{filename:name});
ctx.project=project;
const browser=vm.runInContext('NoodleESP32S3.generate(project)',ctx);
assert(Buffer.from(browser.files['model/params.bin']).equals(Buffer.from(stable.files['model/params.bin'])));
assert.equal(browser.files['src/main.cpp'],stable.files['src/main.cpp']);
assert.equal(browser.files['lib/Noodle/src/noodle_int8_api.cpp'],stable.files['lib/Noodle/src/noodle_int8_api.cpp']);
console.log('PASS: canonical CHW, 26 OHWI to OIHW, 3 buffers, no NHWC layer files, old project settings ignored, browser parity');
