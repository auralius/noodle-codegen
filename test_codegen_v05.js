require('./bare_uart_bundle.js');
'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const T = require('./tflite_import.js');
const S = require('./studio.js');
const G = require('./firmware_codegen.js');
globalThis.NoodleTFLite = T;
(async () => {
  const bytes = new Uint8Array(fs.readFileSync(path.join(__dirname, 'fcn_mnist16_int8.tflite')));
  const project = S.fromTFLiteModel(T.inspectTFLite(bytes),'MNIST16.tflite',bytes,await T.sha256(bytes));
  for (const layout of ['PROGMEM', 'SD', 'MIXED']) {
    const p=structuredClone(project);
    p.layers.filter(l=>l.kind==='FCN').forEach((l,i)=>l.params=layout==='MIXED'?(i===0?'SD':'PROGMEM'):layout);
    p.settings={...(p.settings||{}),apply_pingpong_sequential:true};
    const x=G.generate(p);
    const main=x.files['src/main.cpp'];
    const ini=x.files['platformio.ini'];
    const header=x.files['include/generated_params.h'];
    assert.equal(x.summary.activationBytes,320);
    assert.deepEqual(x.summary.preallocated,['tensor_a','tensor_b']);
    assert(main.indexOf('if (!reserve_activations())') < main.indexOf('noodle_fs_init(10)') || layout==='PROGMEM');
    assert(main.includes('noodle_i8_tensor_require_vector(&output, 10'));
    assert(main.includes('static bool verify_sd_parameters()')===(layout!=='PROGMEM'));
    assert(!main.includes('uint16_t lut_data[256]'));
    assert(main.includes('pgm_read_word(&r15_softmax1_exp_q15[d])'));
    assert(header.includes('const uint16_t r15_softmax1_exp_q15[] PROGMEM'));
    assert(ini.includes('NOODLE_BUFFER_ARENA_INITIAL_BYTES=0'));
    assert(main.includes('#define Serial r15_uart'));
    assert.equal(JSON.parse(x.files['test/output_spec.json']).output_kind,'quantized_softmax');
    assert(!Object.keys(x.files).some(k=>k.includes('mnist.csv')));
    assert(x.files['README.md'].includes('320 bytes'));
    if(layout!=='PROGMEM') {
      assert(main.includes('ERR_SD_OPEN w01.bin'));
      assert(main.includes('ERR_SD_SIZE p01.bin'));
    }
    console.log('PASS: v0.5',layout,'preallocation / SD verify / flash Softmax');
  }
  const noSoftmax=structuredClone(project);
  noSoftmax.topology=noSoftmax.topology.split('\n').filter(t=>!t.includes('softmax1(')).join('\n');
  noSoftmax.layers.filter(l=>l.kind==='FCN').forEach(l=>l.params='SD');
  noSoftmax.settings={apply_pingpong_sequential:true};
  const x=G.generate(noSoftmax);
  assert.equal(x.summary.activationBytes,320);
  assert.equal(x.summary.softmax,0);
  assert(!x.files['src/main.cpp'].includes('execute_softmax1('));
  assert(!x.files['include/generated_params.h'].includes('r15_softmax1_exp_q15'));
  assert.equal(JSON.parse(x.files['test/output_spec.json']).output_kind,'quantized_logits');
  assert(x.files['src/main.cpp'].includes('noodle_i8_find_max(&tensor_a'));
  console.log('PASS: v0.5 user-omitted Softmax -> 320 B logits+argmax');
  // Default must be unoptimized, with original buffers retained.
  const def=G.generate(project);
  assert.equal(def.summary.sequentialPingPong,false);
  assert(def.files['src/main.cpp'].includes('static NoodleTensor '+project.storage[0].name+';'));
  assert(!def.files['src/main.cpp'].includes('static NoodleTensor tensor_a, tensor_b;'));
  // Force a skip/fan-out: reuse the original input for a second FCN call.
  // When checkbox is ON this must fall back, never silently apply ping-pong.
  const graph=structuredClone(project);
  graph.settings={apply_pingpong_sequential:true};
  const first=graph.topology.split('\n')[0];
  const m=first.match(/^([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\((\w+)\);/);
  assert(m,first);
  const alias='branch_copy';
  graph.storage.push({...graph.storage.find(x=>x.name===m[1]),name:alias});
  graph.topology=first+'\n'+`${alias} = ${m[2]}(${m[3]});`+'\n'+graph.topology.split('\n').slice(1).join('\n').replace(new RegExp('\\b'+m[1]+'\\b','g'),alias);
  const forked=G.generate(graph);
  assert.equal(forked.summary.sequentialPingPong,false);
  assert.equal(forked.summary.pingPongRequested,true);
  assert(forked.files['src/main.cpp'].includes('static NoodleTensor '+alias+';'));
  const legacy=structuredClone(project);legacy.format='noodle-studio-r15/v0.4';
  assert.equal(S.normalizeProject(legacy).format,'noodle-studio-r15/v0.5.1');
  console.log('PASS: legacy R15 v0.4 JSON import');
})().catch(e=>{console.error(e);process.exit(1)});
