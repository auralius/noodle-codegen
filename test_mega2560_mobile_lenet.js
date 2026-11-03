'use strict';
const assert=require('assert'),fs=require('fs'),path=require('path'),crypto=require('crypto'),os=require('os');
const root=__dirname;
global.NoodleTFLite=require('./tflite_import.js');
global.NOODLE_LIB_FILES=require('./noodle_bundle.js');
const studio=require('./studio.js');const samples=require('./mobile_lenet_examples.js');
const mega=require('./mega2560_mobile_lenet_codegen.js');
const OUT=fs.mkdtempSync(path.join(os.tmpdir(),'noodle-mega-codegen-'));
for(const precision of ['int8','fp32']){
 const bytes=Buffer.from(samples.get(precision).base64,'base64'),m=NoodleTFLite.inspectTFLite(bytes);
 const hash=crypto.createHash('sha256').update(bytes).digest('hex');
 const p=studio.fromTFLiteModel(m,`mobile-lenet-16-${precision}.tflite`,bytes,hash);
 assert.equal(p.target,'mega2560');assert.equal(studio.firmwareCompatibility(p).backend,'mega_mobile_lenet');
 const issues=studio.validate(p);assert.deepEqual(issues.errors,[],issues.errors.join('; '));
 const pkg=mega.generate(p); assert(Object.keys(pkg.files).length>25);
 const gen=Object.keys(pkg.files);console.log(precision,gen.length,'files; header size',pkg.files[`include/${precision}/model_progmem.h`].length);
 for(const [k,v] of Object.entries(pkg.files)){
  if(!k.startsWith('lib/')&&k!=='model/project.json'){
   let dest=path.join(OUT,precision,k);fs.mkdirSync(path.dirname(dest),{recursive:true});fs.writeFileSync(dest,v);
  }
 }
 // The flash and model must be included, no machine-specific local symlinks.
 assert(!pkg.files['platformio.ini'].includes('/home/'));
 assert(pkg.files[precision==='int8'?'src/main-int8.cpp':'src/main-fp32.cpp'].includes('PROGMEM'));
 // Reject edited topology / impossible memory choices.
 const damaged=JSON.parse(JSON.stringify(p));damaged.layers[0].kind='FCN';assert.throws(()=>mega.generate(damaged),/edited operator type/);
 const edited=JSON.parse(JSON.stringify(p));edited.storage[0].backend='SRAM';assert.throws(()=>mega.generate(edited),/expected SD activation/);
 const topo=JSON.parse(JSON.stringify(p));topo.topology=topo.topology.replace('(', '(input99, ');assert.throws(()=>mega.generate(topo));
 // Reject superficially similar but unsupported graph without silently changing runtime.
 const broken=JSON.parse(JSON.stringify(m));broken.operators[1].config.stride_h=2;assert.throws(()=>mega.assertGraph(broken),/unsupported stride/);
}
fs.rmSync(OUT,{recursive:true,force:true});console.log('Mega2560 FP32 and INT8 model export passed');
