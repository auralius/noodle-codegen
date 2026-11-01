require('./bare_uart_bundle.js');
'use strict';
const assert=require('node:assert/strict'),fs=require('fs'),crypto=require('crypto');
require('./tflite_import.js');require('./lenet5_examples.js');require('./noodle_bundle.js');require('./serial_sender_bundle.js');
const S=require('./studio.js'),T=require('./tflite_import.js'),C=require('./lenet_uno_codegen.js');
(async()=>{
 let p=await S.bundledLeNetModel('int8');
 p.target='uno';const sdids=new Set([p.source.inputs[0],...p.source.operators.slice(0,5).map(o=>o.outputs[0])]);
 for(const b of p.storage){b.backend=sdids.has(b.tensor_id)?'SD':'SRAM';b.path=sdids.has(b.tensor_id)?'/t'+String(b.tensor_id).padStart(2,'0')+'.bin':'';}
 for(const l of p.layers)l.params=['FCN','CONV_2D'].includes(l.kind)?'SD':'-';
 p.settings={...(p.settings||{}),apply_pingpong_sequential:true};
 const r=C.generate(p),files=r.files;
 // An imported or old project has the option OFF by default.
 const off=structuredClone(p);delete off.settings.apply_pingpong_sequential;
 const base=C.generate(off);
 assert.equal(base.summary.sequentialPingPong,false);
 assert(base.files['src/main.cpp'].includes('static NoodleTensor input, pool1, pool2, reshape, hidden1, hidden2, logits, softmax;'));
 assert(base.files['src/main.cpp'].includes('static Conv conv1_cfg, conv2_cfg;'));
 assert(base.files['src/main.cpp'].includes('static FCNFile fc1_cfg, fc2_cfg, fc3_cfg;'));
 assert(!base.files['src/main.cpp'].includes('static NoodleTensor tensor_a;'));
 assert(base.files['src/main.cpp'].includes('noodle_i8_conv2d_fused(&input,conv1_cfg,mean_pool,&pool1)'));
 assert(base.files['src/main.cpp'].includes('noodle_i8_fcn(&reshape,fc1_cfg,&hidden1)'));
 assert(base.files['src/main.cpp'].includes('noodle_i8_softmax_progmem(&logits,r15_softmax_lut'));
 assert.equal(base.files['sdcard/t00.bin'].length,r.files['sdcard/t00.bin'].length);
 assert.equal(S.normalizeProject(off).settings.apply_pingpong_sequential,false);
 assert.equal(S.normalizeProject(p).settings.apply_pingpong_sequential,true);
 assert(r.summary.experimental && r.summary.sdLayers===5 && r.summary.sequentialPingPong);
 assert.equal(r.summary.layers.length,9);
 assert(files['src/main.cpp'].includes('noodle_i8_conv2d_fused(&tensor_a,conv_cfg,mean_pool,&tensor_b)'));
 assert(files['src/main.cpp'].includes('noodle_i8_conv2d_fused(&tensor_b,conv_cfg,mean_pool,&tensor_a)'));
 assert(files['src/main.cpp'].includes('noodle_i8_fcn(&tensor_b,fcn_cfg,&tensor_a)'));
 assert(files['src/main.cpp'].includes('noodle_i8_softmax_progmem(&tensor_a,r15_softmax_lut'));
 assert(!files['src/main.cpp'].includes('@@'));
 assert.equal((files['src/main.cpp'].match(/static NoodleTensor /g)||[]).length,2);
 assert.equal((files['src/main.cpp'].match(/static FCNFile /g)||[]).length,1);
 assert(files['platformio.ini'].includes('NOODLE_FS_PREALLOCATED=1'));
 assert(!files['lib/Noodle/src/noodle_int8_layers.cpp'].includes('Serial.'));
 assert(!files['lib/Noodle/src/noodle_int8_layers.cpp'].includes('q_avr_diag_putchar'));
 assert(files['src/main.cpp'].includes('#define Serial r15_uart'));
 assert(files['sdcard/manifest.json'].includes('preallocated_activation_bytes'));
 const lengths={'t00.bin':784,'t13.bin':1176,'t15.bin':400,'t16.bin':400,'c01.bin':150,'d01.bin':72,'c03.bin':2400,'d03.bin':192,
   'w06.bin':48000,'p06.bin':1440,'w07.bin':10080,'p07.bin':1008,'w08.bin':840,'p08.bin':120};
 for(const [n,sz] of Object.entries(lengths))assert.equal(files['sdcard/'+n].length,sz,n);
 const raw=T.base64Decode(p.source.model_base64),model=T.inspectTFLite(raw);
 const wt=T.extractTensorData(raw,model.operators[2].inputs[1]);const packed=files['sdcard/c03.bin'];
 for(let oc=0;oc<16;oc++)for(let ic=0;ic<6;ic++)for(let ky=0;ky<5;ky++)for(let kx=0;kx<5;kx++){
   const from=(((oc*5+ky)*5+kx)*6)+ic,to=(((oc*6+ic)*5+ky)*5)+kx;
   assert.equal(packed[to],wt[from],'Conv2 [O,K,K,I] -> [O,I,K,K]');
 }
 const inSpec=JSON.parse(files['test/input_spec.json']);assert.equal(inSpec.elements,784);
 assert.equal(files['model/source.tflite'].length,raw.length);
 assert.deepEqual(Buffer.from(files['model/source.tflite']),Buffer.from(raw));
 const noSm=JSON.parse(JSON.stringify(p));noSm.topology=noSm.topology.split('\n').slice(0,8).join('\n');
 const r2=C.generate(noSm);assert(!r2.files['src/main.cpp'].includes('noodle_i8_softmax_progmem(&tensor_a,r15_softmax_lut'));
 const wrong=JSON.parse(JSON.stringify(p));wrong.storage.find(x=>x.tensor_id===0).backend='SRAM';
 assert.throws(()=>C.generate(wrong),/must be SD/i);
 const wrong2=JSON.parse(JSON.stringify(p));wrong2.topology=wrong2.topology.replace('conv1(input','fc2(input');
 assert.throws(()=>C.generate(wrong2),/order\/op binding/);
 assert.throws(()=>C.generate({...p,target:'esp32s3'}),/Arduino Uno/);
 let fp=await S.bundledLeNetModel('fp32');fp.target='uno';
 assert.throws(()=>C.generate(fp),/INT8|quantization|per-tensor|input/i);
 fs.writeFileSync('example_project.json',JSON.stringify(p,null,2));
 console.log('PASS: actual 9-op LeNet INT8 ZIP, 10 SD parameter binaries, exact Conv2 layout, 784-byte input, optional Softmax, reject incompatible configurations.');
})().catch(e=>{console.error(e.stack);process.exit(1)});
