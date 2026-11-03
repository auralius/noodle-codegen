const assert=require('assert'),fs=require('fs');
global.NoodleTFLite=require('./tflite_import.js');
global.NOODLE_LIB_FILES=require('./noodle_bundle.js');
const studio=require('./studio.js');const examples=require('./mobile_lenet_examples.js');
for(const precision of ['int8','fp32']){
 const x=examples.get(precision);
 const b=NoodleTFLite.base64Decode(x.base64);
 const p=studio.fromTFLiteModel(NoodleTFLite.inspectTFLite(b),x.filename,b,x.sha256);
 const files=require('./mega2560_mobile_lenet_codegen.js').generate(p).files;
 const ini=files['platformio.ini'];
 assert(ini.includes('board = megaatmega2560'));
 const source=precision==='int8'?'src/main-int8.cpp':'src/main-fp32.cpp';
 assert(ini.includes('+'+'<'+source.slice(4)+'>')||ini.includes(source.slice(4)));
 assert(files[source]);assert(!files['src/main.cpp']);
 assert(ini.includes('greiman/SdFat'));
 assert(files['lib/Noodle/src/noodle_int8_layers.cpp']);
 assert.strictEqual(studio.firmwareCompatibility(p).backend,'mega_mobile_lenet');
}
console.log('Mega2560 GUI reference selection and firmware packaging checks passed');
