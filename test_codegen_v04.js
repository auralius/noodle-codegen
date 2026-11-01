require('./bare_uart_bundle.js');
'use strict';
const assert=require('assert'),fs=require('fs'),path=require('path');
const T=require('./tflite_import.js');globalThis.NoodleTFLite=T;
const S=require('./studio.js'),G=require('./firmware_codegen.js');
(async()=>{
  const bytes=new Uint8Array(fs.readFileSync(path.join(__dirname,'fcn_mnist16_int8.tflite')));
  const p=S.fromTFLiteModel(T.inspectTFLite(bytes),'fcn_mnist16_int8.tflite',bytes,await T.sha256(bytes));
  p.settings={sd_cs_pin:10};
  for(const mode of ['PROGMEM','SD','MIXED']){
    const q=JSON.parse(JSON.stringify(p));q.layers.filter(x=>x.kind==='FCN').forEach((l,i)=>l.params=mode==='MIXED'?(i===0?'SD':'PROGMEM'):mode);
    const r=G.generate(q);const spec=JSON.parse(r.files['test/input_spec.json']);
    assert.equal(spec.elements,256);assert.deepEqual(spec.shape,[1,256]);
    assert.equal(spec.protocol,'noodle-serial-img-v1');
    assert(r.files['test/send_tensor.py'].includes('def send_one('));
    assert.equal(r.files['test/dummy_input.bin'].length,256);
    assert(r.files['src/main.cpp'].includes('receive_input()'));
    assert.equal(r.summary.sdLayers,mode==='PROGMEM'?0:mode==='MIXED'?1:2);
    if(mode!=='PROGMEM'){
      assert.equal(r.files['sdcard/w01.bin'].length,16384);
      assert.equal(r.files['sdcard/p01.bin'].length,768);
      // Weight bytes must be bit-exact TFLite INT8 row-major [OUT, IN].
      const originalModel=T.inspectTFLite(bytes), firstFc=originalModel.operators.find(op=>op.op==='FULLY_CONNECTED');
      const expected=T.extractTensorData(bytes,firstFc.inputs[1]);
      assert.deepEqual(Array.from(r.files['sdcard/w01.bin']),Array.from(expected));
      // Packed metadata holds little-endian INT32 bias at the start of each 12-byte record.
      const biasBytes=T.extractTensorData(bytes,firstFc.inputs[2]);
      const packed=new DataView(r.files['sdcard/p01.bin'].buffer);
      const biasView=new DataView(biasBytes.buffer,biasBytes.byteOffset,biasBytes.byteLength);
      for(let o=0;o<64;o++)assert.equal(packed.getInt32(o*12,true),biasView.getInt32(o*4,true));
      assert.equal(new DataView(r.files['sdcard/p01.bin'].buffer).byteLength,768);
      assert(r.files['src/main.cpp'].includes('p.metadata_fn="p01.bin"'));
      assert(r.files['src/main.cpp'].includes('noodle_fs_init(10)'));
      assert(r.files['README.md'].includes('| MOSI | D11 |'));
      assert(r.files['README.md'].includes('| MISO | D12 |'));
      assert(r.files['README.md'].includes('| SCK | D13 |'));
      assert(r.files['sdcard/README.md'].includes('CS=D10, MOSI=D11, MISO=D12, SCK=D13')); 
      assert(r.files['platformio.ini'].includes('NOODLE_USE_SDFAT'));
      assert(r.files['platformio.ini'].includes('greiman/SdFat'));
    }else{
      assert(!Object.keys(r.files).some(x=>x.startsWith('sdcard/')));
      assert(r.files['platformio.ini'].includes('NOODLE_USE_NONE'));
    }
    if(mode==='SD'){
      assert.equal(r.files['sdcard/w02.bin'].length,640);
      assert.equal(r.files['sdcard/p02.bin'].length,120);
      assert(!r.files['include/generated_params.h'].includes('r15_fc1_w'));
    }
    if(mode==='MIXED'){
      assert(!r.files['sdcard/w02.bin']);
      assert(r.files['include/generated_params.h'].includes('r15_fc2_w'));
      assert(r.files['src/main.cpp'].includes('FCNProgmem p;'));
    }
    assert.equal(new DataView(G.zip(r.files).buffer).getUint32(0,true),0x04034b50);
    console.log('PASS:',mode,'parameters + generic serial input specification');
  }
  const bad=JSON.parse(JSON.stringify(p));bad.layers[0].params='SRAM';assert.throws(()=>G.generate(bad),/PROGMEM or SD/);
  const wrong=JSON.parse(JSON.stringify(p));wrong.settings={sd_cs_pin:99};assert.throws(()=>G.generate(wrong),/standard SD wiring requires CS=D10/);
})().catch(e=>{console.error(e);process.exit(1)});
