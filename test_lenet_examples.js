'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs'), crypto=require('node:crypto');
const T=require('./tflite_import.js');
const bundles=require('./lenet5_examples.js');
const S=require('./studio.js');
const C=require('./firmware_codegen.js');
(async()=>{
  for(const kind of bundles.names){
    const obj=bundles.get(kind);
    const raw=fs.readFileSync(__dirname+'/'+obj.filename);
    assert.deepEqual(Buffer.from(T.base64Decode(obj.base64)),raw,'Exact bundled file');
    assert.equal(crypto.createHash('sha256').update(raw).digest('hex'),obj.sha256);
    const p=await S.bundledLeNetModel(kind);
    const model=T.inspectTFLite(raw);
    assert.equal(model.operators.length,9);
    assert.deepEqual(model.operators.map(x=>x.op),['CONV_2D','AVERAGE_POOL_2D','CONV_2D','AVERAGE_POOL_2D','RESHAPE','FULLY_CONNECTED','FULLY_CONNECTED','FULLY_CONNECTED','SOFTMAX']);
    assert.equal(p.layers.length,9);
    assert.equal(p.layers.filter(x=>x.params==='-').length,4); // two pool, reshape, softmax
    assert.deepEqual(p.storage.find(x=>x.tensor_id===model.inputs[0])?.shape,'1,28,28,1');
    assert.equal(p.target,'esp32s3');
    assert.deepEqual(S.validate(p).errors,[]);
    assert.equal(S.validate(p).steps.length,9);
    assert.deepEqual(S.normalizeProject(JSON.parse(JSON.stringify(p))),p);
    assert(p.source.model_base64&&p.source.model_sha256===obj.sha256);
    const isInt8=model.input_dtypes[0]==='INT8';
    assert.equal(isInt8,kind==='int8');
    assert.throws(()=>C.generate(p),/Uno|unsupported|ESP32|not yet/i);
    console.log('PASS: bundled LeNet-5',kind,'9 ops, real weights, JSON round-trip, no false firmware claim; total storage',p.storage.length);
  }
  const source=fs.readFileSync(__dirname+'/index.html','utf8');
  assert(!source.includes('value="cute"'));
  assert(source.includes('value="lenet_int8"')&&source.includes('value="lenet_fp32"'));
  assert(source.includes('src="lenet5_examples.js"'));
  const legacy={...S.sampleUno(),format:'noodle-studio-r15/v0.5'};
  assert.equal(S.normalizeProject(legacy).format,'noodle-studio-r15/v0.5.1');
  console.log('PASS: old CUTE-YOLO selector removed, LeNet-5 variants selectable, v0.5 JSON compatible');
})().catch(e=>{console.error(e);process.exit(1)});
