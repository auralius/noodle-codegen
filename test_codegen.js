require('./bare_uart_bundle.js');
const assert = require('assert');
const fs=require('fs');
const t=require('./tflite_import.js');
globalThis.NoodleTFLite=t;
const s=require('./studio.js'), g=require('./firmware_codegen.js');
async function run(){
  const data=new Uint8Array(fs.readFileSync(__dirname+'/fcn_mnist16_int8.tflite'));
  const model=t.inspectTFLite(data),hash=await t.sha256(data);
  const project=s.fromTFLiteModel(model,'fcn_mnist16_int8.tflite',data,hash);
  assert.equal(project.target,'uno');assert.equal(s.validate(project).errors.length,0);
  // Explicitly enable the optional sequential ping-pong mode for this test.
  project.settings={apply_pingpong_sequential:true};
  const output=g.generate(project);
  assert.equal(output.summary.weights,2);assert.equal(output.summary.softmax,1);
  const main=output.files['src/main.cpp'];
  assert(main.includes('execute_fc1(tensor_a, tensor_b)'));
  assert(main.includes('execute_fc2(tensor_b, tensor_a)'));
  assert(main.includes('execute_softmax1(tensor_a, tensor_b)'));
  assert(main.includes('NOODLE_PGM_ADDRESS(r15_fc1_w)'));
  assert(output.files['include/generated_params.h'].includes('PROGMEM'));
  const bytes=g.zip(output.files);assert.equal(new DataView(bytes.buffer).getUint32(0,true),0x04034b50);
  const fromSaved=s.normalizeProject(JSON.parse(fs.readFileSync(__dirname+'/example_mnist16_saved_project.json')));
  assert.equal(g.generate(fromSaved).summary.layers.length,3);
  const notSupported=JSON.parse(JSON.stringify(project));notSupported.storage[0].backend='SD';assert.throws(()=>g.generate(notSupported),/only SRAM/);
  const wrongOrder=JSON.parse(JSON.stringify(project));wrongOrder.topology='hidden64 = fc2(input256);';assert.throws(()=>g.generate(wrongOrder),/shape\/quantization/);
  const looped=JSON.parse(JSON.stringify(project));looped.topology='hidden64 = fc1(input256);\nlogits10 = fc2(hidden64);\nhidden64 = fc1(input256);\nlogits10 = fc2(hidden64);\nprobabilities10 = softmax1(logits10);';
  const repeated=g.generate(looped);assert.equal((repeated.files['include/generated_params.h'].match(/const int8_t r15_fc1_w\[\]/g)||[]).length,1);
  console.log('PASS: TFLite -> user topology -> PROGMEM FCN params -> real Uno C++ ZIP, JSON restore, repeat layer, unsupported config rejection');
}
run().catch(e=>{console.error(e);process.exit(1)});
