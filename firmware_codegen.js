/* R15 v0.5: offline AVR/Uno INT8 FCN + Softmax + SD parameter and generic serial code generator.
 * Intentional limitations: no SD activation storage, no generic graph planner,
 * no automatic layer rewiring or implicit weight conversion. Unsupported configurations fail.
 */
'use strict';
(function(root){
 const ID=/^[A-Za-z_][A-Za-z0-9_]*$/;
 const RESERVED=new Set(['class','auto','int','const','static','float','return','while','for','if','else','void','new','delete','switch','case','template','namespace','struct','volatile','this','true','false','setup','loop','Serial','run_user_topology']);
 const enc=new TextEncoder();
 function check(ok,message){if(!ok)throw Error(message);}
 function qvalue(t){check(t&&t.dtype==='INT8'&&t.quantization&&t.quantization.scales.length===1&&t.quantization.zero_points.length===1,'Expected per-tensor INT8 activation quantization');return {scale:t.quantization.scales[0],zp:t.quantization.zero_points[0]};}
 function identicalQuant(a,b){return a && b && Math.abs(a.scale-b.scale)<=1e-7*Math.max(1,Math.abs(b.scale)) && a.zp===b.zp;}
 function qmult(x){check(Number.isFinite(x)&&x>=0,'Invalid requantization multiplier');if(!x)return [0,0];let exp=0,q=x;while(q<0.5){q*=2;exp--;check(exp>=-32,'Multiplier underflow');}while(q>=1){q/=2;exp++;check(exp<=30,'Multiplier overflow');}let n=Math.floor(q*2147483648+0.5);if(n===2147483648){n/=2;exp++;}return [n,exp];}
 function i32s(bytes){check(bytes.length%4===0,'Invalid INT32 tensor buffer size');const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);return Array.from({length:bytes.length/4},(_,i)=>dv.getInt32(i*4,true));}
 function literalF(n){check(Number.isFinite(n),'Non-finite quantization');return Number(n).toPrecision(10)+'f';}
 function arrdecl(name,typ,values,cols){let out=`const ${typ} ${name}[] PROGMEM = {\n`;for(let i=0;i<values.length;i+=cols)out+='  '+values.slice(i,i+cols).join(', ')+(i+cols<values.length?',':'')+'\n';return out+'};\n';}
 function i32le(values){const out=new Uint8Array(values.length*4),dv=new DataView(out.buffer);values.forEach((v,i)=>dv.setInt32(i*4,v,true));return out;}
 function packedMeta(bias,multipliers,shifts){check(bias.length===multipliers.length&&bias.length===shifts.length,'Metadata length mismatch');const out=new Uint8Array(bias.length*12),dv=new DataView(out.buffer);for(let i=0;i<bias.length;i++){dv.setInt32(i*12,bias[i],true);dv.setInt32(i*12+4,multipliers[i],true);dv.setInt32(i*12+8,shifts[i],true);}return out;}
 function parseTopology(s){let out=[];for(const [index,raw] of s.split('\n').entries()) {const text=raw.split('//')[0].trim();if(!text)continue;const m=text.match(/^([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;?$/);if(m){out.push({type:'layer',dst:m[1],layer:m[2],src:m[3],line:index+1});continue;}const d=text.match(/^destroy\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;?$/);if(d){out.push({type:'destroy',buffer:d[1],line:index+1});continue;}throw Error('Unsupported topology at line '+(index+1)+': '+text);}return out;}
 function generate(project){
  const tfl=root.NoodleTFLite || (typeof require!=='undefined'?require('./tflite_import.js'):null);
  check(tfl,'TFLite parser missing');check(project&&project.target==='uno','R15 v0.5 generates only Arduino Uno / Nano');
  check(project.source?.type==='tflite'&&project.source.model_base64,'Import an INT8 TFLite model first (or open a saved project containing its bytes)');
  const modelBytes=tfl.base64Decode(project.source.model_base64),model=tfl.inspectTFLite(modelBytes);
  check(model.inputs.length===1 && model.outputs.length===1,'One model input and output supported');
  check(model.tensors[model.inputs[0]].dtype==='INT8','Model input must be INT8');
  const stores=new Map(), layers=new Map();
  for(const b of project.storage){check(ID.test(b.name)&&!RESERVED.has(b.name),'Invalid storage name '+b.name);check(!stores.has(b.name),'Duplicate buffer '+b.name);check(b.backend==='SRAM',`R15 v0.5 supports only SRAM activations on Uno (${b.name} selected ${b.backend})`);stores.set(b.name,b);}
  for(const l of project.layers){check(ID.test(l.name)&&!RESERVED.has(l.name),'Invalid layer name '+l.name);check(!layers.has(l.name),'Duplicate layer '+l.name);check(Number.isInteger(l.op_index)&&l.op_index>=0&&l.op_index<model.operators.length,'Layer '+l.name+' is not bound to a TFLite operator');layers.set(l.name,l);}
  const steps=parseTopology(project.topology);check(steps.some(s=>s.type==='layer'),'No layer calls in topology');
  const bound=new Map(),deleted=new Set(),callLines=[],paramHeaders=[],wrappers=[],stepDesc=[],sdManifest=[];
  // Reserve every activation used by explicit topology BEFORE SdFat initialization.
  // Growth-only maximum capacity per name; never alter the user-defined call order.
  const prealloc=new Map();
  function reserve(b,n,q){
    const old=prealloc.get(b);
    if(!old||n>old.n)prealloc.set(b,{n,scale:q.scale,zp:q.zp});
  }
  const sdFiles={};
  let start=null;
  for(const b of project.storage){if(b.tensor_id===model.inputs[0]){check(!start,'Duplicate input tensor storage');start=b.name;}}
  check(start,'Model input storage was removed; re-import TFLite');
  const inTensor=model.tensors[model.inputs[0]];check(qvalue(inTensor).scale===1&&qvalue(inTensor).zp===-128,'The built-in Uno image sender requires input scale 1.0 and zero-point -128. Add a different input adapter before generating this model.');const nInput=inTensor.shape.reduce((a,b)=>a*b,1);check(nInput>0&&nInput<=65535,'Input tensor size unsupported');
  bound.set(start,{n:nInput,q:qvalue(inTensor),tensor:model.inputs[0]});
  reserve(start,nInput,qvalue(inTensor));
  let terminal=null;let fcCount=0;let softmaxCount=0;const emitted=new Set();
  for(const s of steps){
   check(stores.has(s.dst??s.buffer),'Line '+s.line+': output/storage not declared');
   if(s.type==='destroy'){check(bound.has(s.buffer),'Line '+s.line+': cannot destroy uninitialized buffer');deleted.add(s.buffer);callLines.push(`  noodle_tensor_free(&${s.buffer});`);continue;}
   check(stores.has(s.src),'Line '+s.line+': input buffer '+s.src+' not declared');check(!deleted.has(s.src)&&!deleted.has(s.dst),'Line '+s.line+': use after destroy');
   check(bound.has(s.src),'Line '+s.line+': reading uninitialized buffer '+s.src);
   check(s.src!==s.dst,'Line '+s.line+': in-place FCN/Softmax outputs are not supported by this generator');
   const layer=layers.get(s.layer);check(layer,'Line '+s.line+': layer '+s.layer+' missing');const op=model.operators[layer.op_index];
   check(op,'Bad TFLite op index');const inpT=model.tensors[op.inputs[0]],outT=model.tensors[op.outputs[0]];
   const src=bound.get(s.src), expectedN=inpT.shape.reduce((a,b)=>a*b,1),outN=outT.shape.reduce((a,b)=>a*b,1);
   check(src.n===expectedN&&identicalQuant(src.q,qvalue(inpT)),'Line '+s.line+': '+s.layer+' input shape/quantization does not match '+s.src+' (topology cannot silently change scales)');
   check(outN<=65535&&outN>0,'Unsupported layer output size');
   if(op.op==='FULLY_CONNECTED'){
    check(layer.kind==='FCN'&&['PROGMEM','SD'].includes(layer.params),'Layer '+s.layer+': only FCN with PROGMEM or SD parameters is supported on Uno');
    check(op.inputs.length>=3&&op.config.weights_format===0,'Unsupported FULLY_CONNECTED configuration');
    const wt=model.tensors[op.inputs[1]],bt=model.tensors[op.inputs[2]];
    check(wt.dtype==='INT8'&&bt.dtype==='INT32'&&wt.shape.length===2&&wt.shape[0]===outN&&wt.shape[1]===expectedN,'Unsupported FCN weights/bias shape');
    check(bt.shape.length===1&&bt.shape[0]===outN,'FCN bias shape mismatch');
    const wBytes=tfl.extractTensorData(modelBytes,wt.id),bBytes=tfl.extractTensorData(modelBytes,bt.id);
    check(wBytes.length===expectedN*outN&&bBytes.length===4*outN,'FCN parameter length mismatch');
    const wq=wt.quantization;check(wq&&wq.scales.length===outN&&wq.zero_points.every(z=>z===0),'Only symmetric per-channel INT8 FCN weights supported');
    check(bt.quantization?.scales.length===outN,'Missing per-channel bias scales');
    const multipliers=[], shifts=[];
    for(let j=0;j<outN;j++){
      const product=qvalue(inpT).scale*wq.scales[j];
      check(Math.abs(product-bt.quantization.scales[j])<=1e-5*Math.max(product,1e-9),'Bias scale not compatible with input and weights');
      const [m,sh]=qmult(product/qvalue(outT).scale);multipliers.push(m);shifts.push(sh);
    }
    const wv=Array.from(wBytes,v=>v>127?v-256:v);
    const prefix='r15_'+s.layer;
    const bias=i32s(bBytes);
    const fileId=String(layer.op_index+1).padStart(2,'0');
    if(!emitted.has(s.layer)){
      if(layer.params==='PROGMEM') paramHeaders.push(`// ${s.layer}: FCN ${expectedN} -> ${outN}, TFLite op ${layer.op_index}\n`+arrdecl(prefix+'_w','int8_t',wv,16)+arrdecl(prefix+'_b','int32_t',bias,8)+arrdecl(prefix+'_m','int32_t',multipliers,8)+arrdecl(prefix+'_s','int32_t',shifts,8));
      else {
        const weightName='w'+fileId+'.bin',metaName='p'+fileId+'.bin';
        sdFiles['sdcard/'+weightName]=new Uint8Array(wBytes);
        sdFiles['sdcard/'+metaName]=packedMeta(bias,multipliers,shifts);
        sdManifest.push({layer:s.layer,weight_file:weightName,metadata_file:metaName,weight_bytes:wBytes.length,metadata_bytes:12*outN,metadata_record:'bias:int32_le,multiplier:int32_le,shift:int32_le'});
      }
    }
    const act=op.config.activation||'NONE';check(act==='NONE'||act==='RELU','Fused FCN activation '+act+' not supported');
    const oq=qvalue(outT),iq=qvalue(inpT);const amin=act==='RELU'?Math.max(-128,oq.zp):-128;
    if(!emitted.has(s.layer)){
      let init;
      if(layer.params==='PROGMEM')init=`  FCNProgmem p;\n  p.weight_far=NOODLE_PGM_ADDRESS(${prefix}_w); p.bias_far=NOODLE_PGM_ADDRESS(${prefix}_b);\n  p.multiplier_far=NOODLE_PGM_ADDRESS(${prefix}_m); p.shift_far=NOODLE_PGM_ADDRESS(${prefix}_s);`;
      else init=`  FCNFile p;\n  p.weight_fn="w${fileId}.bin"; p.metadata_fn="p${fileId}.bin";`;
      wrappers.push(`static bool execute_${s.layer}(NoodleTensor &input, NoodleTensor &output) {\n${init}\n  p.input_scale=${literalF(iq.scale)}; p.output_scale=${literalF(oq.scale)};\n  p.input_zero_point=${iq.zp}; p.output_zero_point=${oq.zp};\n  p.activation_min=${amin}; p.activation_max=127;\n  p.act=${act==='RELU'?'ACT_RELU':'ACT_NONE'}; p.O=${outN};\n  return noodle_fcn(&input, &output, p)==${outN};\n}\n`);
    }
    fcCount++;
   } else if(op.op==='SOFTMAX'){
    check(layer.kind==='Softmax'||layer.kind==='SOFTMAX','Layer '+s.layer+' was changed to a non-Softmax operator');
    check(op.config.beta===undefined||Math.abs(op.config.beta-1)<1e-9,'Softmax beta must equal 1');
    check(outN===expectedN&&outN<=1024,'Unsupported Softmax size');
    check(Math.abs(qvalue(outT).scale-1/256)<1e-8&&qvalue(outT).zp===-128,'Only standard INT8 softmax quantization supported');
    if(!emitted.has(s.layer)){
      // Same Q15 lookup equation and integer rounding as Noodle's INT8 Softmax,
      // but LUT is in AVR program memory instead of a 512-byte stack array.
      const smName='r15_'+s.layer+'_exp_q15';
      const inputScale=qvalue(inpT).scale;
      const expQ15=Array.from({length:256},(_,d)=>Math.max(0,Math.min(32767,Math.round(Math.exp(-d*inputScale)*32767))));
      paramHeaders.push(`// ${s.layer}: AVR PROGMEM softmax Q15 lookup (512 flash bytes, 0 SRAM bytes).\n`+arrdecl(smName,'uint16_t',expQ15,12));
      wrappers.push(`static bool execute_${s.layer}(NoodleTensor &input, NoodleTensor &output) {\n  if (!input.buffer.data || input.rank != NOODLE_TENSOR_1D || input.W != 1 || input.C != ${outN}) return false;\n  int8_t *dst = noodle_i8_tensor_require_vector(&output, ${outN}, 0.00390625f, -128);\n  if (!dst) return false;\n  const int8_t *src = input.buffer.data;\n  int8_t maxv=src[0];\n  for(uint16_t i=1;i<${outN};++i)if(src[i]>maxv)maxv=src[i];\n  uint32_t sum=0;\n  for(uint16_t i=0;i<${outN};++i){\n    const uint8_t d=static_cast<uint8_t>(static_cast<int16_t>(maxv)-src[i]);\n    sum+=pgm_read_word(&${smName}[d]);\n  }\n  if(!sum)return false;\n  for(uint16_t i=0;i<${outN};++i){\n    const uint8_t d=static_cast<uint8_t>(static_cast<int16_t>(maxv)-src[i]);\n    const uint32_t num=static_cast<uint32_t>(pgm_read_word(&${smName}[d]))*256u+sum/2u;\n    uint32_t q=num/sum;\n    if(q>255u)q=255u;\n    dst[i]=static_cast<int8_t>(static_cast<int16_t>(q)-128);\n  }\n  return true;\n}\n`);
    }
    softmaxCount++;
   }else throw Error('Layer '+s.layer+': '+op.op+' not yet implemented by Uno firmware generator');
   emitted.add(s.layer);
   callLines.push(`  if (!execute_${s.layer}(${s.src}, ${s.dst})) return false;`);
   bound.set(s.dst,{n:outN,q:qvalue(outT),tensor:op.outputs[0]});
   reserve(s.dst,outN,qvalue(outT));
   terminal=s.dst;stepDesc.push(s.layer+': '+op.op+' '+expectedN+' -> '+outN);
  }
  check(fcCount>0,'No FCN layers');
  check(bound.get(terminal)?.tensor===model.outputs[0]||(!softmaxCount&&bound.get(terminal)?.tensor===model.operators.at(-2)?.outputs?.[0]),'Final topology does not produce the expected model output');
  check(!deleted.has(terminal),'Final output was destroyed');
  const sdCount=sdManifest.length;const sdCs=10;
  check(project.settings?.sd_cs_pin===undefined || project.settings.sd_cs_pin===10,
    'Arduino Uno/Nano standard SD wiring requires CS=D10 (MOSI=D11 MISO=D12 SCK=D13). Remove the old nonstandard SD pin override from your project JSON.');
  // Lightweight optimization for a strictly linear FCN/Softmax chain only:
  // one input -> one output per stage, no destroy, no branches or fan-out.
  // The tensor names remain in the user's project; only physical descriptors
  // and fixed scratch arrays are mapped to A/B in the generated application.
  const layerSteps=steps.filter(x=>x.type==='layer');
  const sequentialPingPong=project.settings?.apply_pingpong_sequential===true &&
      layerSteps.length>=2 && layerSteps.length===steps.length &&
      layerSteps.every((x,i)=>i===0?x.src===start:x.src===layerSteps[i-1].dst);
  let declared,setup,preallocate,arenaBytes,executionCalls,inputPhysical=start,terminalPhysical=terminal;
  if(sequentialPingPong){
    const aSize=Math.max(nInput,...layerSteps.filter((x,i)=>i%2===1).map(x=>prealloc.get(x.dst).n));
    const bSize=Math.max(...layerSteps.filter((x,i)=>i%2===0).map(x=>prealloc.get(x.dst).n));
    arenaBytes=aSize+bSize;
    declared=`static NoodleTensor tensor_a, tensor_b;\nstatic int8_t activation_a[${aSize}];\nstatic int8_t activation_b[${bSize}];`;
    setup=`  noodle_tensor_init(&tensor_a);\n  noodle_tensor_init(&tensor_b);`;
    preallocate=`// Linear network: external SRAM buffers ping-pong, no heap arena growth.\nstatic bool reserve_activations() {\n  noodle_buffer_bind_external(&tensor_a.buffer,activation_a,sizeof(activation_a));\n  noodle_buffer_bind_external(&tensor_b.buffer,activation_b,sizeof(activation_b));\n  return true;\n}\n`;
    executionCalls=layerSteps.map((x,i)=>`  if (!execute_${x.layer}(${i%2===0?'tensor_a':'tensor_b'}, ${i%2===0?'tensor_b':'tensor_a'})) return false;`);
    inputPhysical='tensor_a';
    terminalPhysical=layerSteps.length%2?'tensor_b':'tensor_a';
  }else{
    declared=project.storage.map(b=>'static NoodleTensor '+b.name+';').join('\n');
    setup=project.storage.map(b=>'  noodle_tensor_init(&'+b.name+');').join('\n');
    arenaBytes=Array.from(prealloc.values()).reduce((sum,v)=>sum+v.n,0);
    const allocCode=Array.from(prealloc,([name,x])=>`  if (!noodle_i8_tensor_require_vector(&${name}, ${x.n}, ${literalF(x.scale)}, ${x.zp})) { Serial.println(F("ERR_ALLOC ${name}")); return false; }`).join('\n');
    preallocate=`// Nonlinear topology: preserve the explicit user buffers.\nstatic bool reserve_activations() {\n${allocCode}\n  return true;\n}\n`;
    executionCalls=callLines;
  }
  check(arenaBytes<=1700,`Uno activation storage (${arenaBytes} bytes) exceeds conservative 1700-byte activation-only ceiling; 2 KiB SRAM must also hold SdFat, globals, and stack. Reduce user buffers or target a larger MCU.`);
  // Check each generated SD file independently, to fail during setup rather than mid-FCN.
  const sdCheck=sdManifest.length?`static bool verify_sd_parameters() {\n${sdManifest.map(x=>[ [x.weight_file,x.weight_bytes], [x.metadata_file,x.metadata_bytes] ].map(([path,bytes])=>`  { NDL_File f = noodle_fs_open_read("${path}");\n    if (!f) { Serial.println(F("ERR_SD_OPEN ${path}")); return false; }\n    const uint32_t size = f.size(); f.close();\n    if (size != ${bytes}ul) { Serial.println(F("ERR_SD_SIZE ${path}")); return false; } }`).join('\n')).join('\n')}\n  return true;\n}\n`:'';
  const progHeader=`// Generated by Noodle Studio R15 v0.5 from ${project.source.name}\n// INT8 FCN parameters: PROGMEM only when selected, row-major OUT x IN.\n#pragma once\n#include <stdint.h>\n#include <avr/pgmspace.h>\n\n`+paramHeaders.join('\n');
  const main=`// Noodle Studio R15 v0.5 - Arduino Uno/Nano INT8\n// User-controlled execution order and SRAM buffers. PROGMEM/SD params; no graph runtime.\n#include <Arduino.h>\n#include <noodle.h>\n#include <noodle_int8.h>\n#include "generated_params.h"\n#if defined(__AVR_ATmega328P__)\n#include "bare_uart.h"\n#define Serial r15_uart\n#endif\n\n${declared}\n\n${preallocate}\n${sdCheck}\n${wrappers.join('\n')}\nbool run_user_topology() {\n${executionCalls.join('\n')}\n  return true;\n}\n\n// Binary serial protocol: host sends IMG\\n, waits RDYIMG, then sends blocks of 32.\n// MCU sends K after each block. Converts 0..255 pixels to TFLite INT8 using q=pixel-128.\nstatic bool receive_input() {\n  int8_t *dst=noodle_i8_tensor_require_vector(&${inputPhysical}, ${nInput}, ${literalF(qvalue(inTensor).scale)}, ${qvalue(inTensor).zp});\n  if (!dst) return false;\n  for(uint16_t i=0;i<${nInput};++i){\n    const unsigned long began=millis();\n    while(Serial.available()==0){if(millis()-began>3000UL)return false;}\n    int pixel=Serial.read();\n    dst[i]=static_cast<int8_t>(pixel-128);\n    if((i%32)==31)Serial.write('K');\n  }\n  return true;\n}\n\nvoid setup() {\n  Serial.begin(115200);\n${setup}\n  if (!reserve_activations()) { while(true) {} }\n${sdCount?`  if (!noodle_fs_init(${sdCs})) { Serial.println(F("ERR_SD_INIT")); while(true) {} }\n  Serial.println(F("SD_READY"));\n  if (!verify_sd_parameters()) { while(true) {} }\n`:""}  Serial.println(F("BOOT Noodle Studio R15 UNO INT8"));\n  Serial.println(F("READY"));\n}\n\nvoid loop() {\n  static uint8_t cmd=0;\n  while(Serial.available()) {\n    const char ch=static_cast<char>(Serial.read());\n    const char expected[]={'I','M','G','\\n'};\n    if(ch==expected[cmd])++cmd;else cmd=(ch=='I'?1:0);\n    if(cmd!=4)continue;\n    cmd=0;\n    Serial.println(F("RDYIMG"));\n    if(!receive_input()){ Serial.println(F("ERR_INPUT")); Serial.println(F("READY"));return;}\n    unsigned long t0=millis();\n    if(!run_user_topology()){Serial.println(F("ERR_INFERENCE"));Serial.println(F("READY"));return;}\n    unsigned long dt=millis()-t0;\n    uint16_t idx=0;int8_t maxval=0;\n    noodle_i8_find_max(&${terminalPhysical},&idx,&maxval);\n    Serial.print(F("PRED ")); Serial.print(idx); Serial.print(F(" MS "));Serial.println(dt);\n    Serial.print(F("OUTPUT_INT8"));\n    for(uint16_t i=0;i<${bound.get(terminal).n};++i){Serial.print(' ');Serial.print(static_cast<int>(${terminalPhysical}.buffer.data[i]));}\n    Serial.println();\n    Serial.println(F("READY"));\n    return;\n  }\n}\n`;
  const ini=`[env:uno-int8]\nplatform = atmelavr\nboard = uno\nframework = arduino\nmonitor_speed = 115200\n${sdCount?"lib_deps =\n  greiman/SdFat @ ^2.2.3\n":""}build_flags =\n  -D NOODLE_USE_INT8\n  -D ${sdCount?"NOODLE_USE_SDFAT":"NOODLE_USE_NONE"}\n  -D NOODLE_INT8_FCN_BLOCK=32\n${sequentialPingPong?"  -D NOODLE_BUFFER_ARENA_INITIAL_BYTES=0\n":""}`;
  const inputSpec={protocol:'noodle-serial-img-v1',shape:inTensor.shape,elements:nInput,transport_dtype:'uint8',baud:115200,chunk_bytes:32,input_quantization:{model_dtype:'int8',scale:qvalue(inTensor).scale,zero_point:qvalue(inTensor).zp,device_transform:'q = pixel_u8 - 128'},notes:'Transport is generic uint8; raw pixel interpretation and image/sensor preprocessing belong to the application.'};
  const last=bound.get(terminal),outputSpec={shape:model.tensors[last.tensor].shape,dtype:'int8',scale:last.q.scale,zero_point:last.q.zp,serial_lines:['PRED <class> MS <milliseconds>','OUTPUT_INT8 <values>','READY'],output_kind:softmaxCount?'quantized_softmax':'quantized_logits',argmax_equivalent_to_softmax:!softmaxCount};
  const sender=root.NOODLE_SERIAL_SENDER || (typeof require!=='undefined'?require('./serial_sender_bundle.js'):null);check(sender,'Generic Python sender missing');
  const readme=`# Noodle Studio R15 v0.5 - Arduino Uno INT8 FCN\n\nGenerated from **${project.source.name}**, model SHA-256 **${project.source.model_sha256||'not supplied'}**.\n\nTarget: Arduino Uno/Nano ATmega328P. SRAM activation buffers. Per-layer FCN parameters in PROGMEM or SD, selected by user.\n\n## User topology\n\n\`\`\`cpp\n${project.topology}\n\`\`\`\n\n## Build + test\n\n\`\`\`bash\npio run -e uno-int8\npio run -e uno-int8 -t upload\npython3 -m pip install pyserial\npython3 test/send_tensor.py --port /dev/ttyUSB0 --manifest test/input_spec.json --pattern ramp --repeat 3\n\`\`\`\n\nRun a dummy payload first; arbitrary random buffers test communication and execution, NOT classification accuracy. For real data use --input-file <exactly ${nInput} bytes>. Input transformation q=pixel-128 is valid only for input scale 1, zero point -128; any model-specific preprocessing remains the user's responsibility.\n\n**Serial protocol:** send IMG\\n; wait for RDYIMG; send ${nInput} raw uint8 bytes in 32-byte chunks; wait for one K byte after each full chunk. MCU replies PRED, OUTPUT_INT8, READY. The last partial chunk has no K in this protocol version. No frame-level CRC yet.\n\n${sdCount?`**SD PARAMETERS — STANDARD ARDUINO UNO/NANO SPI WIRING (FIXED):**

| SD module pin | Arduino Uno / Nano ATmega328P |
|---|---|
| CS (chip select) | D10 |
| MOSI | D11 |
| MISO | D12 |
| SCK | D13 |
| GND | GND |
| VCC | Follow SD adapter specification |

The generated firmware uses noodle_fs_init(10) for CS=D10, and the Arduino hardware SPI peripheral uses D11/D12/D13. **Do not connect a bare 3.3V microSD card directly to 5V Uno/Nano I/O.** Use an SD module with appropriate regulation/level shifting, or proper external 3.3V interface.

Copy every .bin in the generated sdcard/ directory to the **SD card root** with unchanged names.\n\n${sdManifest.map(x=>`- ${x.layer}: ${x.weight_file} (${x.weight_bytes} bytes) + ${x.metadata_file} (${x.metadata_bytes} bytes)`).join('\n')}\n\nThe metadata file contains a 12-byte LE record per output neuron: signed INT32 bias, multiplier, and shift. The Noodle library reads these as packed metadata; do not copy files from a different export.\n`: 'All selected FCN parameters are compiled into PROGMEM; no SD card or parameter .bin files are required.\n'}\n## Model scope and limits\n\n- Supported: Uno/Nano, INT8 FCN with per-channel symmetric weights + optional INT8 Softmax, one input/output, SRAM activations, PROGMEM/SD parameters.\n- Optional (disabled by default) sequential FCN ping-pong uses two tensor descriptors and fixed scratch arrays; skip connections, branches, and destroy calls keep explicit storage allocations.\n- INT8 Softmax, **when present in the user topology**, uses a 512-byte flash-resident Q15 LUT rather than a 512-byte stack allocation. Generated AVR-specific code preserves the Noodle INT8 Softmax LUT equation and rounding.
- Remove the Softmax call from the user topology (without changing the source TFLite) to classify by logits alone; Codegen will then report logits as OUTPUT_INT8. This is a user-controlled change, not a silent optimization.
- Generated setup() reserves activation buffer capacities (**${arenaBytes} bytes** for this topology) before SdFat initialization. Unused declared storage is not reserved. The AVR application uses polling UART0 instead of Arduino Serial software buffers.
- SD file presence and sizes are checked at boot. Model/SD weights must come from the same export.
- SRAM estimation counts activation bytes only; static library globals, stack peaks, and SdFat RAM remain platform- and compiler-dependent.\n- The MCU input and sender are intentionally generic raw-byte transport, not a camera driver.\n- AVR build/flash and board-level numeric tests are required. Host tests are NOT hardware validation.\n\nLayers: ${stepDesc.join('; ')}.\n`;
  const baseFiles={'platformio.ini':ini,'src/main.cpp':main,'src/bare_uart.h':root.R15_BARE_UART_SOURCE || (typeof require!=='undefined'?require('./bare_uart_bundle.js'):null),'include/generated_params.h':progHeader,'README.md':readme,'test/send_tensor.py':sender,'test/input_spec.json':JSON.stringify(inputSpec,null,2)+'\n','test/output_spec.json':JSON.stringify(outputSpec,null,2)+'\n','test/dummy_input.bin':new Uint8Array(nInput).map((_,i)=>i&255),'model/source.tflite':modelBytes,'model/project.json':JSON.stringify(project,null,2)+'\n'};
  if(sdCount){baseFiles['sdcard/README.md']='Copy wXX.bin and pXX.bin to SD root.\nUno/Nano SD SPI: CS=D10, MOSI=D11, MISO=D12, SCK=D13, GND=GND; VCC as specified by SD module.\nUse level shifting/regulation for 5V Uno/Nano.\nMetadata pXX.bin is packed int32 LE bias/multiplier/shift per neuron.\n';baseFiles['sdcard/manifest.json']=JSON.stringify(sdManifest,null,2)+'\n';} 
  const lib=root.NOODLE_LIB_FILES || (typeof require!=='undefined'?require('./noodle_bundle.js'):null);check(lib,'Bundled Noodle library missing');
  const files={...baseFiles,...sdFiles,...lib};return {files,summary:{sequentialPingPong,pingPongRequested:project.settings?.apply_pingpong_sequential===true,layers:stepDesc,source:project.source.name,weights:fcCount,sdLayers:sdCount,softmax:softmaxCount,buffers:project.storage.map(x=>x.name),inputCount:nInput,final:terminal,activationBytes:arenaBytes,preallocated:sequentialPingPong?['tensor_a','tensor_b']:Array.from(prealloc.keys())}};
 }
 function crc32(data){let crc=0xffffffff;for(const v of data){crc^=v;for(let i=0;i<8;i++)crc=(crc>>>1)^(crc&1?0xedb88320:0);}return (crc^0xffffffff)>>>0;}
 function zip(files){const le16=(a,v,o)=>new DataView(a.buffer).setUint16(o,v,true),le32=(a,v,o)=>new DataView(a.buffer).setUint32(o,v>>>0,true);
  const blocks=[],central=[];let offset=0;const sorted=Object.entries(files).sort(([a],[b])=>a.localeCompare(b));check(sorted.length<=65535,'Too many ZIP entries');
  for(const [name,value] of sorted){check(/^[a-zA-Z0-9_./-]+$/.test(name)&&!name.startsWith('/')&&!name.includes('..'),'Unsafe filename '+name);
    const path=enc.encode(name),payload=typeof value==='string'?enc.encode(value):value,crc=crc32(payload);
    const local=new Uint8Array(30+path.length);le32(local,0x04034b50,0);le16(local,20,4);le16(local,0,6);le16(local,0,8);le32(local,crc,14);le32(local,payload.length,18);le32(local,payload.length,22);le16(local,path.length,26);local.set(path,30);
    blocks.push(local,payload);const c=new Uint8Array(46+path.length);le32(c,0x02014b50,0);le16(c,20,4);le16(c,20,6);le32(c,crc,16);le32(c,payload.length,20);le32(c,payload.length,24);le16(c,path.length,28);le32(c,offset,42);c.set(path,46);central.push(c);offset+=local.length+payload.length;
  }
  const centralSize=central.reduce((s,x)=>s+x.length,0);const tail=new Uint8Array(22);le32(tail,0x06054b50,0);le16(tail,sorted.length,8);le16(tail,sorted.length,10);le32(tail,centralSize,12);le32(tail,offset,16);
  const all=[...blocks,...central,tail],size=all.reduce((s,x)=>s+x.length,0),out=new Uint8Array(size);let at=0;for(const b of all){out.set(b,at);at+=b.length;}return out;
 }
 const API={generate,zip};if(typeof module!=='undefined'&&module.exports)module.exports=API;root.NoodleCodegen=API;
})(typeof globalThis!=='undefined'?globalThis:this);
