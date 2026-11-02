'use strict';
const fs=require('fs'),assert=require('assert'),crypto=require('crypto'),path=require('path');
const T=require('./tflite_import.js'),Studio=require('./studio.js'),G=require('./esp32s3_generic_fp32_codegen.js');
const dir=path.join(__dirname,'_validation_pingpong');fs.mkdirSync(dir,{recursive:true});
function test(name,expectedOps,expectedBytes){
 const filename=path.join(__dirname,name+'.tflite');
 const data=fs.readFileSync(filename),model=T.inspectTFLite(data),sha=crypto.createHash('sha256').update(data).digest('hex');
 const base=Studio.fromTFLiteModel(model,path.basename(filename),data,sha);
 assert.equal(Studio.firmwareCompatibility(base).backend,'s3_fp32_generic');
 assert.equal(Studio.validate(base).errors.length,0);
 const standard=G.generate(base);
 const ppProject=structuredClone(base);ppProject.settings={apply_pingpong_sequential:true};
 const opt=G.generate(ppProject);
 assert.equal(opt.summary.layers.length,expectedOps);
 assert.equal(opt.summary.sequentialPingPong,true);
 assert.equal(JSON.parse(opt.files['model/manifest.json']).memory_strategy,'pingpong');
 assert.equal(JSON.parse(standard.files['model/manifest.json']).memory_strategy,'liveness');
 assert.equal(opt.files['test/input_spec.json'],standard.files['test/input_spec.json']);
 assert.equal(opt.files['test/output_spec.json'],standard.files['test/output_spec.json']);
 assert.equal(Buffer.compare(Buffer.from(opt.files['model/params.bin']),Buffer.from(standard.files['model/params.bin'])),0);
 if(expectedBytes) assert.equal(opt.summary.arenaBytes,expectedBytes);
 const target=path.join(dir,name);fs.mkdirSync(target,{recursive:true});
 for(const [variant,pack] of [['baseline',standard],['pingpong',opt]]){
  const subdir=path.join(target,variant);fs.mkdirSync(subdir,{recursive:true});
  for(const fname of ['src/main.cpp','model/params.bin']){
    const file=path.join(subdir,fname);fs.mkdirSync(path.dirname(file),{recursive:true});fs.writeFileSync(file,pack.files[fname]);
  }
  const offsets=Array.from(pack.files['src/main.cpp'].match(/static const uint32_t offsets\[\]=\{([^}]+)\}/)[1].matchAll(/(\d+)u/g),x=>Number(x[1]));
  const alloc=pack.summary.arenaBytes;
  for(let i=0;i<model.operators.length;i++){
   const op=model.operators[i];const ids=op.inputs.filter(id=>id>=0&&!op.parameter_tensors.includes(id));
   const src=base.storage.findIndex(s=>s.tensor_id===ids[0]),dst=base.storage.findIndex(s=>s.tensor_id===op.outputs[0]);
   const asize=model.tensors[ids[0]].shape.reduce((a,x)=>a*x,1)*4;
   const bsize=model.tensors[op.outputs[0]].shape.reduce((a,x)=>a*x,1)*4;
   assert(offsets[src]+asize<=alloc);assert(offsets[dst]+bsize<=alloc);
   assert(offsets[src]+asize<=offsets[dst]||offsets[dst]+bsize<=offsets[src]);
  }
 }
 console.log(`PASS ${name}: ${expectedOps} operators; baseline=${standard.summary.arenaBytes} B, ping-pong=${opt.summary.arenaBytes} B; A/B offsets disjoint`);
}
test('fcn_mnist16_fp32',3,1280);
test('lenet5_fp32',9,23520);
