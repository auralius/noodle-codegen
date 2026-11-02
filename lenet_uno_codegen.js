/* R15 v0.6.2. Generates static Noodle API calls, with AVR low-memory kernels inside Noodle. */
'use strict';
(function(root){
 const ck=(v,m)=>{if(!v)throw Error('LeNet5 Uno v0.6: '+m);};
 const T=root.NoodleTFLite||(typeof require!=='undefined'?require('./tflite_import.js'):null);
 const S=root.NOODLE_SERIAL_SENDER||(typeof require!=='undefined'?require('./serial_sender_bundle.js'):null);
 const N=root.NOODLE_LIB_FILES||(typeof require!=='undefined'?require('./noodle_bundle.js'):null);
 const b64=(s)=>T.base64Decode(s);
 const sq=(t)=>{ck(t?.dtype==='INT8'&&t.quantization?.scales?.length===1&&t.quantization.zero_points.length===1,'activation must have per-tensor INT8 quantization');return {s:t.quantization.scales[0],z:t.quantization.zero_points[0]};};
 function mult(x){ck(Number.isFinite(x)&&x>0,'invalid real multiplier '+x);let sh=0,q=x;while(q<0.5){q*=2;sh--;ck(sh>=-32,'multiplier underflow');}while(q>=1){q/=2;sh++;ck(sh<=30,'multiplier overflow');}let m=Math.floor(q*2147483648+0.5);if(m>=2147483648){m=Math.floor(m/2);sh++;}return [m,sh];}
 function meta(bytes,wt,bias,inT,outT){
   const o=wt.shape[0],rawB=T.extractTensorData(bytes,bias.id),data=new DataView(rawB.buffer,rawB.byteOffset,rawB.byteLength);
   ck(wt.quantization?.scales.length===o&&wt.quantization.zero_points.every(z=>z===0),'per-channel symmetric weights required');
   ck(bias.quantization?.scales.length===o,'per-channel INT32 bias required');
   ck(rawB.byteLength===4*o,'bias length');const a=new Uint8Array(o*12),dv=new DataView(a.buffer);
   for(let i=0;i<o;i++){
     const bscale=wt.quantization.scales[i]*sq(inT).s;
     ck(Math.abs(bias.quantization.scales[i]-bscale)<=1e-5*Math.max(bscale,1e-9),'bias scale mismatch at '+i);
     const [mul,shift]=mult(bscale/sq(outT).s);
     dv.setInt32(i*12,data.getInt32(i*4,true),true);dv.setInt32(i*12+4,mul,true);dv.setInt32(i*12+8,shift,true);
   }
   return a;
 }
 function config(project,model){
   ck(project?.target==='uno','select Arduino Uno/Nano target');
   ck(project?.source?.type==='tflite'&&project.source.model_base64,'model data missing');
   ck(model.inputs.length===1&&model.outputs.length===1&&model.operators.length===9,'expected nine-op reference LeNet5 graph');
   ck(model.operators.map(o=>o.op).join('|')==='CONV_2D|AVERAGE_POOL_2D|CONV_2D|AVERAGE_POOL_2D|RESHAPE|FULLY_CONNECTED|FULLY_CONNECTED|FULLY_CONNECTED|SOFTMAX','model graph is not the supported LeNet5');
   ck(model.tensors[model.inputs[0]].shape.join(',')==='1,28,28,1','input must be NHWC 28x28x1');
   ck(sq(model.tensors[model.inputs[0]]).s===1&&sq(model.tensors[model.inputs[0]]).z===-128,'sender supports uint8 pixel -> signed INT8 only when scale=1 and zero point=-128');
   const shapes=['1,28,28,6','1,14,14,6','1,10,10,16','1,5,5,16','1,400','1,120','1,84','1,10','1,10'];
   for(let i=0;i<9;i++)ck(model.tensors[model.operators[i].outputs[0]].shape.join(',')===shapes[i],`unexpected shape at op ${i}`);
   const paths=new Map(), stores=new Map(),layers=new Map();
   for(const x of project.storage){ck(!stores.has(x.name),'duplicate storage');stores.set(x.name,x);}
   for(const x of project.layers){ck(!layers.has(x.name),'duplicate layer');layers.set(x.name,x);}
   // Topology identities come from user's project. Do not reorder or silently choose storage.
   const calls=project.topology.split('\n').map(line=>line.replace(/\/\/.*/,'').trim()).filter(Boolean);
   ck(calls.length===9||calls.length===8,'expected nine layer calls (or eight with Softmax explicitly omitted)');
   const steps=[];let prev=model.inputs[0];let terminal=null;
   for(let i=0;i<calls.length;i++){
     const match=calls[i].match(/^([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\(\s*([A-Za-z_]\w*)\s*\)\s*;?$/);
     ck(match,'unsupported topology expression: '+calls[i]);
     const [,dst,layerName,src]=match;
     const l=layers.get(layerName);ck(l&&l.op_index===i,'topology order/op binding changed at '+layerName);
     ck(stores.has(dst)&&stores.has(src),'unknown storage at step '+i);
     ck(model.operators[i].inputs[0]===prev,'nonlinear input tensor at step '+i);
     ck(i===0?stores.get(src).tensor_id===model.inputs[0]:steps.at(-1).dst===src,'noncanonical dataflow at '+layerName);
     ck(stores.get(dst).tensor_id===model.operators[i].outputs[0],'destination tensor mismatch at '+layerName);
     if([0,2,5,6,7].includes(i))ck(l.params==='SD','all Conv/FCN layer parameters must be SD at this stage: '+layerName);
     if([1,3,4,8].includes(i))ck(l.params==='-','parameter-free op should have -: '+layerName);
     steps.push({dst,src,layerName,index:i});prev=model.operators[i].outputs[0];terminal=dst;
   }
   ck(calls.length===9||steps[7].index===7,'missing final FCN');
   ck(project.settings?.sd_cs_pin===undefined||Number(project.settings.sd_cs_pin)===10,'Uno SD CS must be D10');
   const roles={input:steps[0].src,conv1:steps[0].dst,pool1:steps[1].dst,conv2:steps[2].dst,pool2:steps[3].dst,reshape:steps[4].dst,hidden1:steps[5].dst,hidden2:steps[6].dst,logits:steps[7].dst,softmax:steps[8]?.dst};
   const types={input:'SD',conv1:'SD',pool1:'SD',conv2:'SD',pool2:'SD',reshape:'SD',hidden1:'SRAM',hidden2:'SRAM',logits:'SRAM',softmax:'SRAM'};
   for(const [role,name] of Object.entries(roles)){
     if(!name)continue;
     const b=stores.get(name);ck(b.backend===types[role],`${name} (${role}) must be ${types[role]} for experimental Uno codegen`);
     if(b.backend==='SD'){
       ck(/^\/?[A-Za-z0-9_]{1,8}\.bin$/i.test(b.path||''),`${name}: SD filename must fit FAT 8.3 (e.g. /${role.slice(0,5)}.bin)`);
       const p=b.path.replace(/^\//,'');ck(!paths.has(p),`SD activation paths must be unique (${p})`);paths.set(p,name);
     }
   }
   const C1=model.operators[0].config,C2=model.operators[2].config;
   ck(C1.padding===0&&C1.stride_h===1&&C1.stride_w===1&&C1.dilation_h===1&&C1.dilation_w===1,'conv1 must be SAME, 1x1 stride/dilation');
   ck(C2.padding===1&&C2.stride_h===1&&C2.stride_w===1&&C2.dilation_h===1&&C2.dilation_w===1,'conv2 must be VALID, 1x1 stride/dilation');
   for(const opi of [0,2])ck(model.operators[opi].config.activation==='RELU','expected fused RELU in conv '+opi);
   for(const [i,j] of [[0,1],[2,3]]){
     const a=sq(model.tensors[model.operators[i].outputs[0]]),b=sq(model.tensors[model.operators[j].outputs[0]]);
     ck(a.s===b.s&&a.z===b.z,'avgpool needs identical quantization to conv');
   }
   const qsm=sq(model.tensors[model.outputs[0]]);ck(qsm.s===1/256&&qsm.z===-128,'unexpected softmax quantization');
   return {stores,roles,steps,softmax:steps.length===9,terminal};
 }
 // Explicit, unoptimized sequential LeNet layout. Keeps every logical activation
 // descriptor and layer configuration. The SD streaming kernels remain unchanged.
 // This is the opt-out baseline; low-memory Uno users can enable ping-pong.
 function unoptimizedMain(template){
   let src=template;
   const replaceBetween=(begin,end,replacement)=>{
     const a=src.indexOf(begin),b=src.indexOf(end,a+begin.length);
     ck(a>=0&&b>a,'missing LeNet codegen template anchor: '+begin);
     src=src.slice(0,a)+replacement+src.slice(b);
   };
   replaceBetween('// Two tensor descriptors follow','// Fixed 150-byte work array',
     '// Individual tensor descriptors (ping-pong disabled).\n'+
     'static NoodleTensor input, pool1, pool2, reshape, hidden1, hidden2, logits, softmax;\n');
   replaceBetween('// One reusable Conv configuration','static void bind_pool_parameters()',
     '// Distinct layer configurations (ping-pong disabled).\n'+
     'static Conv conv1_cfg, conv2_cfg;\n'+
     'static FCNFile fc1_cfg, fc2_cfg, fc3_cfg;\n'+
     'static NoodleI8Pool mean_pool;\n');
   replaceBetween('static bool init_storage(){','// The input SD file MUST be open',`static bool init_storage(){
  noodle_tensor_init(&input);
  noodle_tensor_init(&pool1);
  noodle_tensor_init(&pool2);
  noodle_tensor_init(&reshape);
  noodle_tensor_init(&hidden1);
  noodle_tensor_init(&hidden2);
  noodle_tensor_init(&logits);
  noodle_tensor_init(&softmax);
  if(!noodle_i8_tensor_bind_file(&input,"@@PATH_INPUT@@")) return false;
  if(!noodle_i8_tensor_bind_file(&pool1,"@@PATH_POOL1@@")) return false;
  if(!noodle_i8_tensor_bind_file(&pool2,"@@PATH_POOL2@@")) return false;
  if(!noodle_i8_tensor_bind_file(&reshape,"@@PATH_RESHAPE@@")) return false;
  return true;
}
`);
   src=src.replace('  // tensor_a carried logits from the previous inference, if any.\n','');
   src=src.replace('noodle_i8_tensor_bind_file(&tensor_a,"@@PATH_INPUT@@") ||','noodle_i8_tensor_bind_file(&input,"@@PATH_INPUT@@") ||');
   src=src.replace('noodle_i8_tensor_file_2d(&tensor_a,1,28,','noodle_i8_tensor_file_2d(&input,1,28,');
   replaceBetween('static bool run_user_topology(){','void setup(){',`static bool run_user_topology(){
  Serial.println(F("STAGE CONV1_POOL1"));
  bind_conv1();
  R15_STAGE_TIC();
  if(noodle_i8_conv2d_fused(&input,conv1_cfg,mean_pool,&pool1)!=14) return false;
  
  R15_STAGE_TOC("CONV1_POOL1");Serial.println(F("STAGE CONV2_POOL2"));
  bind_conv2();
  R15_STAGE_TIC();
  if(noodle_i8_conv2d_fused(&pool1,conv2_cfg,mean_pool,&pool2)!=5) return false;
  
  R15_STAGE_TOC("CONV2_POOL2");Serial.println(F("STAGE RESHAPE"));
  R15_STAGE_TIC();
  if(noodle_i8_flat_file(&pool2,&reshape)!=400) return false;
  R15_STAGE_TOC("RESHAPE");
  r15_bind_scratch(hidden1,r15_filter,sizeof(r15_filter),120,@@H1_SCALE@@f,@@H1_ZP@@);
  Serial.println(F("STAGE FC1"));
  bind_fc1();
  R15_STAGE_TIC();
  if(noodle_i8_fcn(&reshape,fc1_cfg,&hidden1)!=120) return false;
  R15_STAGE_TOC("FC1");
  r15_bind_scratch(hidden2,r15_shared_scratch.hidden84,sizeof(r15_shared_scratch.hidden84),84,@@H2_SCALE@@f,@@H2_ZP@@);
  Serial.println(F("STAGE FC2"));
  bind_fc2();
  R15_STAGE_TIC();
  if(noodle_i8_fcn(&hidden1,fc2_cfg,&hidden2)!=84) return false;
  R15_STAGE_TOC("FC2");
  r15_bind_scratch(logits,r15_filter,sizeof(r15_filter),10,@@LOGITS_SCALE@@f,@@LOGITS_ZP@@);
  Serial.println(F("STAGE FC3"));
  bind_fc3();
  R15_STAGE_TIC();
  if(noodle_i8_fcn(&hidden2,fc3_cfg,&logits)!=10) return false;
  R15_STAGE_TOC("FC3");
  r15_bind_scratch(softmax,r15_io,sizeof(r15_io),10,@@SOFTMAX_SCALE@@f,@@SOFTMAX_ZP@@);
  Serial.println(F("STAGE SOFTMAX"));
  @@SOFTMAX_EXEC@@
}
`);
   return src;
 }
 function generate(project){
   ck(T&&N&&S,'offline TFLite, Noodle or sender bundle missing');
   const source=b64(project.source.model_base64),model=T.inspectTFLite(source),v=config(project,model);
   const files={},manifest=[];
   function emit(i,kind){
     const op=model.operators[i],inT=model.tensors[op.inputs[0]],outT=model.tensors[op.outputs[0]],wt=model.tensors[op.inputs[1]],bt=model.tensors[op.inputs[2]];
     ck(wt.dtype==='INT8'&&bt.dtype==='INT32','expected INT8 weights and INT32 bias');
     let raw=T.extractTensorData(source,wt.id),expected=wt.shape.reduce((a,b)=>a*b,1);ck(raw.length===expected,'weight byte count incorrect');
     if(kind==='conv'){
       ck(wt.shape.length===4&&wt.shape[1]===5&&wt.shape[2]===5,'conv weights must be [O,5,5,I]');
       const [O,K1,K2,I]=wt.shape;const chw=new Uint8Array(raw.length);
       for(let o=0;o<O;o++)for(let c=0;c<I;c++)for(let y=0;y<5;y++)for(let x=0;x<5;x++)
         chw[((o*I+c)*5+y)*5+x]=raw[((o*5+y)*5+x)*I+c];
       raw=chw;
     } else ck(wt.shape.length===2,'FCN must be [OUT,IN]');
     const id=String(i+1).padStart(2,'0'),wn=(kind==='conv'?'c':'w')+id+'.bin',pn=(kind==='conv'?'d':'p')+id+'.bin';
     const p=meta(source,wt,bt,inT,outT);
     files['sdcard/'+wn]=raw;files['sdcard/'+pn]=p;
     manifest.push({op_index:i,kind,weights:wn,metadata:pn,weight_bytes:raw.length,metadata_bytes:p.length,layout:kind==='conv'?'[O][I][KY][KX]':'[OUT][IN]',record:'bias/multiplier/shift int32 LE'});
     return {wn,pn,inT,outT,wt};
   }
   const c1=emit(0,'conv'),c2=emit(2,'conv'),f1=emit(5,'fcn'),f2=emit(6,'fcn'),f3=emit(7,'fcn');
   const qr=(t)=>sq(t);const name=(key)=>v.roles[key];const path=(key)=>v.stores.get(name(key)).path.replace(/^\//,'');
   const smIn=qr(model.tensors[model.operators[8].inputs[0]]),smLut=Array.from({length:256},(_,d)=>Math.round(Math.exp(-d*smIn.s)*32767));
   const lutText=`const uint16_t r15_softmax_lut[256] PROGMEM={\n`+Array.from({length:16},(_,i)=>'  '+smLut.slice(i*16,i*16+16).join(', ')).join(',\n')+'\n};\n';
   let header='#pragma once\n#include <stdint.h>\n#include <avr/pgmspace.h>\n'+(v.softmax?lutText:'');
   files['include/generated_params.h']=header;
   const paths=Object.fromEntries(Object.keys(v.roles).filter(k=>k!=='softmax'||v.softmax).map(k=>[k,v.stores.get(name(k)).backend==='SD'?path(k):'']));
   // Source template token substitutions keep canonical order visible (fusion is explicitly documented).
   const sequentialPingPong=project.settings?.apply_pingpong_sequential===true;
   let main=sequentialPingPong?MAIN:unoptimizedMain(MAIN);
   for(const k of ['input','pool1','pool2','reshape']) main=main.replaceAll('@@PATH_'+k.toUpperCase()+'@@',paths[k]);
   const qin=qr(model.tensors[model.inputs[0]]);
   // Distinct named configurations when disabled; one reusable Conv and FCN
   // configuration only when explicitly opted in.
   const configureConv=(v,x,o,pad)=>{
     const a=qr(x.inT),b=qr(x.outT);
     return [
       `${v}.K=5;${v}.P=${pad};${v}.S=1;${v}.O=${o};`,
       `${v}.weight_fn="${x.wn}";${v}.metadata_fn="${x.pn}";${v}.act=ACT_RELU;`,
       `${v}.input_scale=${a.s.toPrecision(10)}f;${v}.input_zero_point=${a.z};`,
       `${v}.output_scale=${b.s.toPrecision(10)}f;${v}.output_zero_point=${b.z};`,
       `${v}.activation_min=${Math.max(-128,b.z)};${v}.activation_max=127;`].join('\n  ');
   };
   const configureFcn=(v,x,kind)=>{
     const a=qr(x.inT),b=qr(x.outT),relu=model.operators[kind].config.activation==='RELU';
     return [`${v}.weight_fn="${x.wn}";${v}.metadata_fn="${x.pn}";${v}.O=${x.wt.shape[0]};`,
       `${v}.input_scale=${a.s.toPrecision(10)}f;${v}.input_zero_point=${a.z};`,
       `${v}.output_scale=${b.s.toPrecision(10)}f;${v}.output_zero_point=${b.z};`,
       `${v}.activation_min=${relu?Math.max(-128,b.z):-128};${v}.activation_max=127;`].join('\n  ');
   };
   for(const [label,body] of Object.entries({
     CONV1:configureConv(sequentialPingPong?'conv_cfg':'conv1_cfg',c1,6,2),CONV2:configureConv(sequentialPingPong?'conv_cfg':'conv2_cfg',c2,16,0),
     FC1:configureFcn(sequentialPingPong?'fcn_cfg':'fc1_cfg',f1,5),FC2:configureFcn(sequentialPingPong?'fcn_cfg':'fc2_cfg',f2,6),FC3:configureFcn(sequentialPingPong?'fcn_cfg':'fc3_cfg',f3,7)
   }))main=main.replace('@@BIND_'+label+'@@',body);
   const lastQ={H1:qr(model.tensors[model.operators[5].outputs[0]]),H2:qr(model.tensors[model.operators[6].outputs[0]]),LOGITS:qr(model.tensors[model.operators[7].outputs[0]]),SOFTMAX:qr(model.tensors[model.operators[8].outputs[0]])};
   for(const [role,q] of Object.entries(lastQ)) main=main.replaceAll('@@'+role+'_SCALE@@',q.s.toPrecision(10)).replaceAll('@@'+role+'_ZP@@',String(q.z));
   main=main.replaceAll('@@INPUT_SCALE@@',qin.s.toPrecision(10)).replaceAll('@@INPUT_ZP@@',String(qin.z));
   main=main.replace('@@SOFTMAX_EXEC@@',v.softmax?`return noodle_i8_softmax_progmem(&${sequentialPingPong?'tensor_a':'logits'},r15_softmax_lut,${lastQ.LOGITS.s.toPrecision(10)}f,&${sequentialPingPong?'tensor_b':'softmax'})==10;`:'return true;');
   if(!v.softmax) main=main.replace('  Serial.println(F("STAGE SOFTMAX"));\n','');
   main=main.replaceAll('@@TERMINAL@@',v.softmax?(sequentialPingPong?'tensor_b':'softmax'):(sequentialPingPong?'tensor_a':'logits'));
   files['src/main.cpp']=main;
   files['src/bare_uart.h']=BARE_UART;
   files['platformio.ini']='[env:uno-int8]\nplatform = atmelavr\nboard = uno\nframework = arduino\nmonitor_speed = 115200\nlib_deps =\n  greiman/SdFat @ ^2.2.3\nbuild_flags =\n  -D NOODLE_USE_INT8\n  -D NOODLE_USE_SDFAT\n  -D NOODLE_INT8_AVR_LOW_MEMORY=1\n  -D NOODLE_INT8_AVR_SD_ONLY_CONV=1\n  -D NOODLE_INT8_AVR_CONV_SHARED_WINDOW=1\n  -D NOODLE_INT8_AVR_SD_ONLY_PACKED_FCN=1\n  -D NOODLE_BUFFER_ARENA_INITIAL_BYTES=0\n  -D NOODLE_FS_PREALLOCATED=1\n  -D NOODLE_INT8_FCN_BLOCK=16\n  -D NOODLE_INT8_SD_PLANE_FAST=0\n  -D NOODLE_INT8_SD_KROW_CACHE=0\n  -D R15_SKIP_BOOT_SD_VERIFY\n';
   const inputSpec={protocol:'noodle-serial-img-v1',shape:[1,28,28,1],elements:784,transport_dtype:'uint8',baud:115200,chunk_bytes:32,input_quantization:{model_dtype:'int8',scale:qin.s,zero_point:qin.z,device_transform:'q = pixel_u8 - 128'},notes:'General raw uint8 input; no resize (original 28x28).'};
   const last=v.softmax?model.outputs[0]:model.operators[7].outputs[0];const oq=qr(model.tensors[last]);
   files['test/input_spec.json']=JSON.stringify(inputSpec,null,2)+'\n';
   files['test/output_spec.json']=JSON.stringify({shape:model.tensors[last].shape,dtype:'int8',scale:oq.s,zero_point:oq.z,output_kind:v.softmax?'quantized_softmax':'quantized_logits'},null,2)+'\n';
   files['test/send_tensor.py']=S;
   // Precreated activation files are included in the ZIP, not fabricated at runtime.
   // Conv1/AvgPool1 and Conv2/AvgPool2 are fused; their transient tmp1/tmp3
   // tensors do not need backing files.
   const activationFiles={};
   for(const [role,opIndex] of [['input',-1],['pool1',1],['pool2',3],['reshape',4]]){
     const tid=opIndex<0?model.inputs[0]:model.operators[opIndex].outputs[0];
     const size=model.tensors[tid].shape.reduce((a,b)=>a*b,1);
     ck(size>0&&size<=65535,'invalid activation file '+role);
     const fn=paths[role];
     ck(!files['sdcard/'+fn],'duplicate SD filename '+fn);
     activationFiles[fn]=size;
     files['sdcard/'+fn]=new Uint8Array(size); // dummy file; overwritten by streaming
   }
   files['sdcard/manifest.json']=JSON.stringify({model:project.source.name,sha256:project.source.model_sha256,activation_files:paths,preallocated_activation_bytes:activationFiles,parameters:manifest},null,2)+'\n';
   files['README.md']=README.replaceAll('@@MODEL@@',project.source.name).replaceAll('@@TOPOLOGY@@',project.topology).replaceAll('@@FILES@@',manifest.map(x=>`${x.weights} ${x.weight_bytes} B | ${x.metadata} ${x.metadata_bytes} B`).join('\n')).replaceAll('@@MODE@@',v.softmax?'full INT8 Softmax':'logits only').replaceAll('@@ACTIVATIONS@@',Object.entries(activationFiles).map(([p,n])=>`${p} ${n} B`).join(', '));
   files['README.md'] += '\n## Codegen buffer policy\n\n'+(sequentialPingPong?
     'Sequential ping-pong explicitly enabled: two descriptors, shared Conv and FCN configs.\n':
     'Ping-pong disabled (default): individual tensor descriptors and layer configurations. This may exceed the Uno SRAM budget; the opt-in mode was used for the working 2 KB demonstration.\n');
   files['README.md'] += '\n## AVR Conv2D shared-window acceleration\n\nThis LeNet-5 export enables `NOODLE_INT8_AVR_CONV_SHARED_WINDOW=1`. The AVR fused 2x2 pooling kernel reuses overlapping input window values with four accumulators in lifetime-shared storage. All other geometries use the existing low-memory path. Disable the flag to return to the original convolution algorithm. The exported firmware binds the scratch explicitly whether ping-pong is selected or not. This path passed 10 consecutive on-device training-sample inferences but still needs cold-start and test-set validation.\n';
   files['README.md'] += '\n## Optional application-only stage timings\n\nAdd `-D NOODLE_APP_LAYER_TIMING=1` to PlatformIO `build_flags` to emit `TIME CONV1_POOL1 <ms>`, `TIME CONV2_POOL2 <ms>`, `TIME FC1 <ms>`, etc., while leaving Noodle silent. Timings measure the operator call (and not UART stage prints).\n';
   files['model/source.tflite']=source;files['model/project.json']=JSON.stringify(project,null,2)+'\n';
   for(const [fn,body] of Object.entries(N))files[fn]=body;
   ck(!main.includes('@@'),'unreplaced tokens: '+[...main.matchAll(/@@[A-Z_0-9]+@@/g)].map(m=>m[0]).join(', '));
   return {files,summary:{layers:v.steps.map(x=>x.layerName),sdLayers:5,activationBytes:266,preallocatedActivationBytes:activationFiles,experimental:true,scratchAliasing:true,sequentialPingPong}};
 }
 const api={generate,config};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.NoodleLeNetUno=api;
 // TEMPLATE BELOW (injected by build script)
 const MAIN="// R15 v0.6.2 Noodle-runtime-first LeNet5 on Uno - EXPERIMENTAL\n// Static topology. All arithmetic uses noodle_i8_* operators (sequential SD-backed FCN weights).\n// SD inputs and outputs are preallocated at Codegen export time.\n// Generated from the original TFLite INT8 weights and quantization metadata.\n#include <Arduino.h>\n#include <noodle.h>\n#include <noodle_int8.h>\n#include \"generated_params.h\"\n#include <limits.h>\n#if defined(__AVR_ATmega328P__)\n#include \"bare_uart.h\"\n#define Serial r15_uart\n#endif\n\n// Two tensor descriptors follow the complete topology. After SD flattening,\n// they are rebound as externally owned SRAM views for FC1/FC2/FC3/Softmax.\n// Each operator still receives two distinct input/output descriptors.\nstatic NoodleTensor tensor_a;\nstatic NoodleTensor tensor_b;\n// Fixed 150-byte work array handles the largest 5x5x6 convolution filter.\nstatic int8_t r15_filter[150];\nstatic uint8_t r15_io[32];\n// Lifetime-specific SRAM reuse: Conv filter bytes are dead before FC1 starts;\n// FC1 output is dead before FC3 overwrites it; I/O scratch is dead at Softmax.\n// Descriptors are rebound after their old values die. The two activation\n// buffers alternate during the dense stages; serial/SD I/O uses r15_io.\n// Lifetime reuse: Conv2D scratch and FC2 activations never coexist.\n// The union supplies 4-byte alignment for the four int32 accumulators.\nunion R15SharedScratch { int32_t conv_acc[4]; int8_t hidden84[84]; };\nstatic R15SharedScratch r15_shared_scratch;\nstatic void r15_bind_scratch(NoodleTensor &t,int8_t *buffer,uint16_t cap,uint16_t n,float scale,int32_t zp){\n  // These are non-owning views into persistent scratch. Do not free or grow them.\n  noodle_buffer_bind_external(&t.buffer,buffer,cap);\n  t.C=n;t.W=1;t.rank=NOODLE_TENSOR_1D;t.scale=scale;t.zero_point=zp;\n  t.file_path=nullptr;\n}\nstatic void r15_bind_scratch(NoodleTensor &t,uint8_t *buffer,uint16_t cap,uint16_t n,float scale,int32_t zp){\n  r15_bind_scratch(t,(int8_t*)buffer,cap,n,scale,zp);\n}\n\n// Parameter bindings only. All operator arithmetic runs in Noodle.\n// One reusable Conv configuration and one reusable FCNFile configuration.\n// Layers execute sequentially; nothing keeps a reference after the call returns.\nstatic Conv conv_cfg;\nstatic FCNFile fcn_cfg;\nstatic NoodleI8Pool mean_pool;\nstatic void bind_pool_parameters(){\n  mean_pool.M=2;mean_pool.T=2;mean_pool.mode=NOODLE_I8_POOL_MEAN;\n}\nstatic void bind_conv1(){\n  @@BIND_CONV1@@\n}\nstatic void bind_conv2(){\n  @@BIND_CONV2@@\n}\nstatic void bind_fc1(){\n  @@BIND_FC1@@\n}\nstatic void bind_fc2(){\n  @@BIND_FC2@@\n}\nstatic void bind_fc3(){\n  @@BIND_FC3@@\n}\nstatic bool init_storage(){\n  noodle_tensor_init(&tensor_a);\n  noodle_tensor_init(&tensor_b);\n  // SRAM scratch is bound only after the SD stages have finished using these\n  // descriptors. No arena allocation is required for the activation tensors.\n  if(!noodle_i8_tensor_bind_file(&tensor_a,\"@@PATH_INPUT@@\")){Serial.println(F(\"ERR_INIT_BIND_A\"));return false;}\n  if(!noodle_i8_tensor_bind_file(&tensor_b,\"@@PATH_POOL1@@\")){Serial.println(F(\"ERR_INIT_BIND_B\"));return false;}\n  return true;\n}\n// The input SD file MUST be open before the RDYIMG response. Otherwise the\n// host can fill the Uno's 32-byte RX FIFO during a blocking SdFat open.\n// Python sends the next block only after the single-byte K acknowledgment.\nstatic bool receive_input(NDL_File &f){\n  for(uint16_t start=0;start<784;start+=32){\n    const uint8_t len=(784-start<32)?(uint8_t)(784-start):32;\n    for(uint8_t k=0;k<len;++k){\n      const unsigned long began=millis();\n      while(!Serial.available()){\n        if(millis()-began>3000ul){\n          Serial.print(F(\"ERR_INPUT_TIMEOUT_BLOCK \"));\n          Serial.print(start/32);\n          Serial.print(F(\" OFFSET \"));\n          Serial.print(k);\n          return false;\n        }\n      }\n      const int b=Serial.read();\n      r15_io[k]=(uint8_t)(int8_t)(b-128);\n    }\n    const size_t n=f.write(r15_io,len);\n    if(n!=len){\n      Serial.print(F(\"ERR_INPUT_WRITE_BLOCK \"));\n      Serial.print(start/32);\n      Serial.print(F(\" WRITTEN \"));\n      Serial.print((unsigned)n);\n      return false;\n    }\n    if(len==32)Serial.write('K');\n  }\n  // Caller closes the file before starting the convolution stages.\n  // tensor_a carried logits from the previous inference, if any.\n  if(!noodle_i8_tensor_bind_file(&tensor_a,\"@@PATH_INPUT@@\") ||\n     !noodle_i8_tensor_file_2d(&tensor_a,1,28,@@INPUT_SCALE@@f,@@INPUT_ZP@@)){\n    Serial.println(F(\"ERR_INPUT_TENSOR_BIND\"));\n    return false;\n  }\n  return true;\n}\n\n// Application-only stage profiling (enable with -D NOODLE_APP_LAYER_TIMING=1).\n#if defined(NOODLE_APP_LAYER_TIMING) && NOODLE_APP_LAYER_TIMING\nstatic unsigned long r15_stage_started;\n#define R15_STAGE_TIC() do {r15_stage_started=millis();} while(0)\n#define R15_STAGE_TOC(label) do { \\\n  Serial.print(F(\"TIME \" label \" \")); \\\n  Serial.println(millis()-r15_stage_started); \\\n} while(0)\n#else\n#define R15_STAGE_TIC() do {} while(0)\n#define R15_STAGE_TOC(label) do {} while(0)\n#endif\n\nstatic bool run_user_topology(){\n  // SD: in.bin (A) -> p1.bin (B) -> p2.bin (A) -> flat.bin (B).\n  // File names differ for every operation, but only two descriptors exist.\n  if(!noodle_i8_tensor_bind_file(&tensor_b,\"@@PATH_POOL1@@\")) return false;\n  Serial.println(F(\"STAGE CONV1_POOL1\"));\n  bind_conv1();\n  R15_STAGE_TIC();\n  if (noodle_i8_conv2d_fused(&tensor_a,conv_cfg,mean_pool,&tensor_b)!=14)return false;\n  R15_STAGE_TOC(\"CONV1_POOL1\");\n  if(!noodle_i8_tensor_bind_file(&tensor_a,\"@@PATH_POOL2@@\")) return false;\n  Serial.println(F(\"STAGE CONV2_POOL2\"));\n  bind_conv2();\n  R15_STAGE_TIC();\n  if (noodle_i8_conv2d_fused(&tensor_b,conv_cfg,mean_pool,&tensor_a)!=5)return false;\n  R15_STAGE_TOC(\"CONV2_POOL2\");\n  if(!noodle_i8_tensor_bind_file(&tensor_b,\"@@PATH_RESHAPE@@\")) return false;\n  Serial.println(F(\"STAGE RESHAPE\"));\n  R15_STAGE_TIC();\n  if (noodle_i8_flat_file(&tensor_a,&tensor_b)!=400)return false;\n\n  R15_STAGE_TOC(\"RESHAPE\");\n  // The last SD input remains live in tensor_b during FC1. Tensor_a is dead,\n  // so rebind it to the 150-byte Conv filter, now free to serve as SRAM output.\n  r15_bind_scratch(tensor_a,r15_filter,sizeof(r15_filter),120,@@H1_SCALE@@f,@@H1_ZP@@);\n  Serial.println(F(\"STAGE FC1\"));\n  bind_fc1();\n  R15_STAGE_TIC();\n  if (noodle_i8_fcn(&tensor_b,fcn_cfg,&tensor_a)!=120)return false;\n\n  R15_STAGE_TOC(\"FC1\");\n  // The flattened SD input is dead after FC1. Rebind tensor_b to hidden84.\n  r15_bind_scratch(tensor_b,r15_shared_scratch.hidden84,sizeof(r15_shared_scratch.hidden84),84,@@H2_SCALE@@f,@@H2_ZP@@);\n  Serial.println(F(\"STAGE FC2\"));\n  bind_fc2();\n  R15_STAGE_TIC();\n  if (noodle_i8_fcn(&tensor_a,fcn_cfg,&tensor_b)!=84)return false;\n  \n  R15_STAGE_TOC(\"FC2\");Serial.println(F(\"STAGE FC3\"));\n  bind_fc3();\n  R15_STAGE_TIC();\n  if (noodle_i8_fcn(&tensor_b,fcn_cfg,&tensor_a)!=10)return false;\n  \n  R15_STAGE_TOC(\"FC3\");Serial.println(F(\"STAGE SOFTMAX\"));\n  @@SOFTMAX_EXEC@@\n}\nvoid setup(){\n  Serial.begin(115200);\n  bind_pool_parameters();\n  noodle_i8_avr_bind_scratch(r15_filter,sizeof(r15_filter),r15_io,sizeof(r15_io));\n  noodle_i8_avr_bind_conv_accumulators(r15_shared_scratch.conv_acc);\n  if(!init_storage()){Serial.println(F(\"ERR_INIT_MEMORY\"));while(true){}}\n  if(!noodle_fs_init(10)){Serial.println(F(\"ERR_SD_INIT\"));while(true){}}\n\n  Serial.println(F(\"READY\"));\n}\nvoid loop(){\n  static uint8_t cmd=0;\n  while(Serial.available()){\n    const char ch=(char)Serial.read();\n    const char expected[]={'I','M','G','\\n'};\n    if(ch==expected[cmd])++cmd;else cmd=(ch=='I'?1:0);\n    if(cmd<4)continue;\n    cmd=0;\n    // Pre-open input file BEFORE telling the host to transmit any bytes.\n    // This prevents input transfer while SdFat opens the SD file.\n\n    NDL_File input_file;\n\n\n    // The 784-byte file must already exist on the SD card.\n    // Open for writing, but do not create/truncate it (lower SdFat stack cost).\n    const bool opened = noodle_fs_open_write_only_existing_into(input_file, \"@@PATH_INPUT@@\");\n\n\n    if (!opened) {\n      Serial.println(F(\"ERR_INPUT_OPEN\"));\n      Serial.println(F(\"READY\"));\n      return;\n    }\n\n    // Keep the validated initial 32-byte write and rewind before UART transfer.\n    // No new buffer is allocated; r15_io is the shared I/O scratch.\n    for (uint8_t i=0; i<sizeof(r15_io); ++i) r15_io[i]=0x80;\n    if (input_file.write(r15_io,sizeof(r15_io)) != sizeof(r15_io) ||\n        !input_file.seekSet(0)) {\n      input_file.close();\n      Serial.println(F(\"ERR_INPUT_PREPARE\"));\n      Serial.println(F(\"READY\"));\n      return;\n    }\n\n    // Keep input_file open while receive_input() writes its 32-byte blocks.\n    Serial.println(F(\"RDYIMG\"));\n    const bool input_ok=receive_input(input_file);\n    input_file.close();\n    if(!input_ok){Serial.println(F(\"ERR_INPUT\"));Serial.println(F(\"READY\"));return;}\n    const unsigned long start=millis();\n    if(!run_user_topology()){Serial.println(F(\"ERR_INFERENCE\"));Serial.println(F(\"READY\"));return;}\n    const unsigned long dt=millis()-start;\n    uint16_t index=0;int8_t value=0;\n    noodle_i8_find_max(&@@TERMINAL@@,&index,&value);\n    Serial.print(F(\"PRED \"));Serial.print(index);Serial.print(F(\" MS \"));Serial.println(dt);\n    Serial.print(F(\"OUTPUT_INT8\"));\n    for(uint8_t i=0;i<10;++i){Serial.print(' ');Serial.print((int)@@TERMINAL@@.buffer.data[i]);}\n    Serial.println();Serial.println(F(\"READY\"));\n    return;\n  }\n}\n";
 const BARE_UART="#pragma once\n\n// R15/Uno UART0 driver: polling, no RX/TX ring buffers, no dynamic allocation.\n// AVR ATmega328P UART0 at F_CPU=16000000, 115200 baud (U2X0).\n// Deliberately tiny surface matching the Serial calls in src/main.cpp.\n// The Python host must send IMG\\n only after READY and wait for each K ACK.\n#if !defined(__AVR_ATmega328P__)\n#error \"This UART0 experiment targets the ATmega328P (Uno/Nano) only\"\n#endif\n\n#include <Arduino.h>\n#include <avr/io.h>\n#include <avr/pgmspace.h>\n#include <stdint.h>\n\nclass R15BareUart {\npublic:\n  void begin(uint32_t baud) {\n    // Doubled UART speed yields UBRR=16 at 16 MHz / 115200 baud.\n    // General computation also supports other baud values.\n    const uint16_t divisor =\n        (uint16_t)((F_CPU + (4UL * baud)) / (8UL * baud) - 1UL);\n    UCSR0B = 0;                               // RX/TX disabled while configuring\n    UCSR0A = _BV(U2X0);                       // double-speed baud generator\n    UBRR0H = (uint8_t)(divisor >> 8);\n    UBRR0L = (uint8_t)divisor;\n    UCSR0C = _BV(UCSZ01) | _BV(UCSZ00);       // 8 data bits, no parity, 1 stop\n    UCSR0B = _BV(RXEN0) | _BV(TXEN0);         // RX/TX enabled, no UART IRQ\n  }\n\n  int available() const { return (UCSR0A & _BV(RXC0)) ? 1 : 0; }\n\n  int read() {\n    if (!(UCSR0A & _BV(RXC0))) return -1;\n    return (int)UDR0;                        // consume single UART HW byte\n  }\n\n  size_t write(uint8_t value) {\n    while (!(UCSR0A & _BV(UDRE0))) {}\n    UCSR0A = _BV(U2X0) | _BV(TXC0);           // clear TX-complete status\n    UDR0 = value;\n    return 1;\n  }\n\n  void flush() {\n    while (!(UCSR0A & _BV(TXC0))) {}         // wait until last byte fully sent\n  }\n\n  void print(const __FlashStringHelper* str) {\n    const char* p = reinterpret_cast<const char*>(str);\n    char ch;\n    while ((ch = (char)pgm_read_byte(p++)) != '\\0') write((uint8_t)ch);\n  }\n\n  void print(const char* str) {\n    if (!str) return;\n    while (*str) write((uint8_t)*str++);\n  }\n\n  void print(char value) { write((uint8_t)value); }\n  void print(signed char value, int base = DEC) { print((long)value, base); }\n  void print(unsigned char value, int base = DEC) { print((unsigned long)value, base); }\n  void print(int value, int base = DEC) { print((long)value, base); }\n  void print(unsigned int value, int base = DEC) { print((unsigned long)value, base); }\n\n  void print(long value, int base = DEC) {\n    if (base == DEC && value < 0) {\n      write('-');\n      // Handles LONG_MIN without signed overflow.\n      print((unsigned long)(-(value + 1L)) + 1UL, base);\n    } else {\n      print((unsigned long)value, base);\n    }\n  }\n\n  void print(unsigned long value, int base = DEC) {\n    if (base < 2 || base > 16) base = DEC;\n    char digits[11]; // 10 decimal digits for 32-bit value + spare\n    uint8_t count = 0;\n    do {\n      const uint8_t digit = (uint8_t)(value % (unsigned)base);\n      digits[count++] = (digit < 10) ? ('0' + digit) : ('A' + digit - 10);\n      value /= (unsigned)base;\n    } while (value && count < sizeof(digits));\n    while (count) write((uint8_t)digits[--count]);\n  }\n\n  void println() { write('\\r'); write('\\n'); }\n\n  template <typename T>\n  void println(T value) { print(value); println(); }\n\n  template <typename T>\n  void println(T value, int base) { print(value, base); println(); }\n};\n\n// One UART instance, local to the generated application translation unit.\nstatic R15BareUart r15_uart;\n";
 const README="# R15 v0.6.6 — Sequential INT8 LeNet-5 for Uno/Nano\n\nSource model: `@@MODEL@@`\n\n## Storage and execution\n\nThe AVR packed FCN kernel uses two file handles, sequential weight reads, and no per-block weight seeks. The AVR fused Conv2D + 2x2 pooling kernel shares overlapping input windows to minimize SD seeks and reads; a general kernel fallback remains available. The application calls the existing Noodle INT8 operators. Noodle itself produces **no serial debug output**. The AVR application uses bare-metal UART polling with the existing `IMG\\n` / `RDYIMG` / `K` protocol and prints stages, final `PRED`, `OUTPUT_INT8`, and `READY`.\n\nWith the optional checkbox enabled, two `NoodleTensor` descriptors ping-pong across sequential layers and one `Conv` / one `FCNFile` configuration are reused. With the checkbox unchecked, the exporter retains individual descriptors/configurations. The pool configuration and fixed scratch storage are retained. The 84-byte FC2 activation storage is also reused as four convolution accumulators while FC2 is inactive. Conv+AveragePool stages are fused; unmaterialized Conv outputs are logical graph nodes.\n\n`NOODLE_FS_PREALLOCATED=1` tells Noodle to **skip output create/truncate/size validation during operator execution**. The exported `sdcard/` contains all required `.bin` files, including four correctly sized zero-filled activation files. Copy **every file inside `sdcard/` to the SD-card root**. Preallocation never loads entire activations to SRAM; operators continue streaming.\n\nActivations: `@@ACTIVATIONS@@`\n\nParameters: `@@FILES@@`\n\n## Building and running\n\nArduino Uno/Nano ATmega328P. The generated `platformio.ini` targets Uno via normal bootloader; keep your custom USBasp/no-bootloader settings if using the hardware-validated Nano setup. Connect SdFat using CS=D10, MOSI=D11, MISO=D12, SCK=D13 and an appropriate level-shifting SD module.\n\n```\npio run -e uno-int8\npio run -e uno-int8 -t upload\npython3 test/send_tensor.py --port /dev/ttyACM0 --manifest test/input_spec.json --pattern zeros --repeat 1 --inference-timeout 86400\n```\n\nExported topology (operator order unchanged):\n\n```cpp\n@@TOPOLOGY@@\n```\n\nModel output: @@MODE@@. Numerical validation against the original TFLite graph remains recommended. File lengths are not checked by the low-memory operator under the preallocated policy; make sure the SD card contains the matching export.\n\nThe shared-window convolution acceleration is enabled for this supported LeNet-5; only stride-one + 2x2 pooling takes the optimized path. The sequential ping-pong setting remains independently opt-in. The sequential optimization is deliberately limited to this supported LeNet-5 architecture. No generic graph memory planner or other target behavior is changed.\n";
})(typeof globalThis!=='undefined'?globalThis:this);
