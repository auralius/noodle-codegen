'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),crypto=require('node:crypto');
const Planner=require('./branch_buffer_planner.js');
const C=require('./esp32s3_codegen.js');
const T=require('./tflite_import.js');
const Studio=require('./studio.js');
const mk=(id,bytes)=>({id,shape:[1,bytes],dtype:'INT8'});
const model={inputs:[0],outputs:[4],tensors:Array.from({length:5},(_,i)=>mk(i,[10,20,30,40,50][i]))};
const s=model.tensors.map(t=>({name:'t'+t.id,tensor_id:t.id,backend:'PSRAM'}));
const steps=[{inputs:[s[0]],dst:s[1]},{inputs:[s[0]],dst:s[2]},{inputs:[s[1],s[2]],dst:s[3]},{inputs:[s[3]],dst:s[4]}];
const plan=Planner.plan(model,steps,s);
assert(plan.physicalCount<5);
assert.deepEqual(plan.lifetimes.map(x=>[x.born,x.lastUsed]),[[-1,1],[0,2],[1,2],[2,3],[3,4]]);
// Explicit operator-day birth/death diagnostics; algorithm must remain read-only.
assert.equal(plan.maxLiveTensors,3);
assert.deepEqual(plan.maxLiveDays,[2,3]);
assert.equal(plan.peakLiveBytes,90);
assert.deepEqual(plan.peakLiveByteDays,[3,4]);
assert.equal(plan.minimumPhysicalBuffers,3);
assert.deepEqual(plan.maximumLivePerBackend,{PSRAM:3});
assert.equal(plan.timeline.length,steps.length+2);
assert.deepEqual(plan.timeline.map(x=>x.day),[0,1,2,3,4,5]);
assert.deepEqual(plan.timeline.map(x=>x.liveTensorCount),[1,2,3,3,2,1]);
assert.deepEqual(plan.timeline.map(x=>x.liveBytes),[10,30,60,90,90,50]);
assert.deepEqual(plan.lifetimes.map(x=>[x.bornDay,x.dieDay]),[[0,2],[1,3],[2,3],[3,4],[4,5]]);
for(const t of plan.timeline){
 const active=plan.lifetimes.filter(x=>x.bornDay<=t.day&&x.dieDay>=t.day);
 assert.equal(active.length,t.liveTensorCount);
 assert.equal(active.reduce((n,x)=>n+x.bytes,0),t.liveBytes);
 const ids=new Set(active.map(x=>x.bufferId));
 assert.equal(ids.size,active.length,'No shared physical buffer can hold two live tensors');
}
// Different storage backends must never share a physical buffer.
const mixed=s.map(x=>({...x,backend:x.tensor_id===2?'SRAM':'PSRAM'}));
const mixedPlan=Planner.plan(model,steps,mixed);
assert.equal(mixedPlan.minimumPhysicalBuffers,Object.values(mixedPlan.maximumLivePerBackend).reduce((a,b)=>a+b,0));
assert(mixedPlan.physicalCount>=mixedPlan.minimumPhysicalBuffers);
for(const b of mixedPlan.buffers)for(const id of b.tensorIds)assert.equal(mixed[id].backend,b.backend);

assert(plan.tensorBufferIds[0]!==plan.tensorBufferIds[1]);
assert(plan.tensorBufferIds[1]!==plan.tensorBufferIds[2]);
assert(plan.tensorBufferIds[2]!==plan.tensorBufferIds[3]);
assert(plan.tensorBufferIds[3]!==plan.tensorBufferIds[4]);
assert.deepEqual(Planner.plan(model,steps,s),plan,'Plan must be deterministic');
assert.throws(()=>Planner.plan(model,steps,[...s,s[0]]),/Duplicate/);
assert.throws(()=>Planner.plan(model,steps,s.slice(0,4)),/missing/i);
const bad=steps.slice();bad[1]={inputs:[s[4]],dst:s[2]};
assert.throws(()=>Planner.plan(model,bad,s),/before production/);
// Keep branch optimization isolated: no changes to the FP32 or sequential generators.
console.log('PASS: synthetic branch and concat lifetime analysis, validation and deterministic planning');
const fixture=process.argv[2];
if(fixture){
 const bytes=fs.readFileSync(fixture),model=T.inspectTFLite(bytes);
 const proj=Studio.fromTFLiteModel(model,'squeezenet1_1.tflite',bytes,crypto.createHash('sha256').update(bytes).digest('hex'));
 const original=structuredClone(proj);original.settings={...(original.settings||{}),reuse_branch_buffers:false};
 const old=C.generate(original),optimized=C.generate(proj);
 const summary=optimized.summary;
 assert.equal(summary.operators,43);assert.equal(summary.activationBytes,4975136);
 assert.equal(summary.bufferReuse,true);assert(summary.physicalBuffers<=5);
 assert(summary.allocatedBytes<summary.activationBytes*.5);
 assert.equal(old.summary.allocatedBytes,old.summary.activationBytes);
 assert.equal(old.summary.bufferReuse,false);
 assert(Buffer.from(old.files['model/params.bin']).equals(Buffer.from(optimized.files['model/params.bin'])));
 assert.equal(old.files['test/input_spec.json'],optimized.files['test/input_spec.json']);
 assert.equal(old.files['test/output_spec.json'],optimized.files['test/output_spec.json']);
 const report=JSON.parse(optimized.files['model/buffer_plan.json']);
 assert.equal(report.buffers.length,summary.physicalBuffers);
 assert(report.peakLiveBytes<=report.allocatedBytes);
 assert.equal(report.maxLiveTensors,3);
 assert.equal(report.minimumPhysicalBuffers,3);
 assert.equal(report.physicalCount,3);
 assert.equal(report.peakLiveBytes,982144);
 assert.equal(report.timeline.length,45);
 assert.equal(report.timeline[0].day,0);
 assert.equal(report.timeline.at(-1).day,44);
 assert.equal(summary.maxLiveTensors,3);
 assert.equal(summary.minimumPhysicalBuffers,3);
 assert.match(optimized.files['model/lifetime_by_day.csv'],/^day,operator_index,live_tensor_count,live_bytes,live_tensor_ids\n/);
 assert.match(optimized.files['model/tensor_lifetimes.csv'],/^tensor_id,name,backend,born_day,die_day,bytes,physical_buffer_id\n/);

 const map=new Map(report.lifetimes.map(x=>[x.tensorId,x]));
 const idsByTensor=new Map(proj.storage.map((s,i)=>[s.tensor_id,report.tensorBufferIds[i]]));
 for(const op of model.operators){
  const active=op.inputs.filter(i=>i>=0&&!op.parameter_tensors.includes(i));
  const output=op.outputs[0];
  for(const input of active)assert.notEqual(idsByTensor.get(input),idsByTensor.get(output),`Unsafe input/output alias at op ${op.id}`);
  for(let i=0;i<active.length;i++)for(let j=i+1;j<active.length;j++)assert.notEqual(idsByTensor.get(active[i]),idsByTensor.get(active[j]));
 }
 for(const a of report.lifetimes)for(const b of report.lifetimes){if(a.tensorId>=b.tensorId||a.bufferId!==b.bufferId)continue;
  assert(a.lastUsed<b.born||b.lastUsed<a.born,`${a.name}/${b.name} share overlapping buffer`);
 }
 // Fixed-layout code never changes; only the allocation/aliasing may differ.
 const oldCpp=old.files['src/main.cpp'],newCpp=optimized.files['src/main.cpp'];
 const grab=(cpp)=>cpp.match(/static const NoodleI8GraphOp ops\[\]=\{([\s\S]+?)\n\};/)[0];
 assert.equal(grab(oldCpp),grab(newCpp));
 assert.match(newCpp,/static uint8_t \*physicalBuffers\[/);
 assert.match(oldCpp,/static NoodleTensor physicalTensors\[44\]/); // Compatibility fallback has one bound tensor per logical activation.
 assert.equal(JSON.parse(optimized.files['model/manifest.json']).physical_activation_bytes,summary.allocatedBytes);
 console.log(`PASS: SqueezeNet-1.1 INT8 ${summary.activationBytes} B / 44 logical → ${summary.allocatedBytes} B / ${summary.physicalBuffers} physical; operator descriptors and XIP parameters unchanged`);
 const fs2=require('node:fs'),path=require('node:path');
 const outDir=process.env.BRANCH_TEST_EXPORT_DIR;
 if(outDir)for(const [variant,result] of [['baseline',old],['optimized',optimized]]){
  for(const name of ['src/main.cpp','model/params.bin','model/buffer_plan.json']){
   if(result.files[name]===undefined)continue;
   const file=path.join(outDir,variant,name);fs2.mkdirSync(path.dirname(file),{recursive:true});fs2.writeFileSync(file,result.files[name]);
  }
 }
}
