/* Noodle Studio R15: operator-driven ESP32-S3 FLOAT32 graph export.
 * Separate backend from the hardware-validated UINT8/INT8 code generator.
 * Model-driven arbitrary supported topology, liveness-based PSRAM arena, read-only flash mmap.
 */
'use strict';
(function(root){
const T=root.NoodleTFLite||(typeof require!=='undefined'?require('./tflite_import.js'):null);
const Sequential=root.NoodleSequentialNetwork||(typeof require!=='undefined'?require('./sequential_network_detector.js'):null);
const NO=0xffffffff;
function ok(v,what){if(!v)throw Error('ESP32-S3 FP32: '+what);}
function byteCount(t){return t.shape.reduce((a,b)=>a*b,1)*4;}
function align(n,a=16){return Math.ceil(n/a)*a;}
function crc32(b){let c=0xffffffff;for(const v of b){c^=v;for(let k=0;k<8;k++)c=(c>>>1)^((c&1)?0xedb88320:0);}return(c^0xffffffff)>>>0;}
function validateGraph(project,model){
 ok(project.target==='esp32s3','ESP32-S3 target required');
 ok(model.inputs.length===1&&model.outputs.length===1,'Single input/output required');
 ok(model.tensors[model.inputs[0]].dtype==='FLOAT32'&&model.tensors[model.outputs[0]].dtype==='FLOAT32','FLOAT32 input/output required');
 const types=new Set(['SUB','MUL','CONV_2D','MAX_POOL_2D','AVERAGE_POOL_2D','CONCATENATION','MEAN','RESHAPE','FULLY_CONNECTED','SOFTMAX']);
 ok(model.operators.every(x=>types.has(x.op)),'Unsupported FLOAT32 operator in this graph: '+model.operators.filter(x=>!types.has(x.op)).map(x=>x.op).join(', '));
 ok(project.layers.length===model.operators.length,'TFLite layer count changed');
 const storage=new Map(project.storage.map((s,i)=>[s.name,{...s,index:i}]));
 ok(project.storage.every(s=>s.backend==='PSRAM'&&Number.isInteger(s.tensor_id)),'All FP32 activation tensors must be PSRAM-backed');
 const layers=new Map(project.layers.map(l=>[l.name,l]));
 const steps=project.topology.split('\n').map(x=>x.split('//')[0].trim()).filter(Boolean);
 ok(steps.length===model.operators.length,'One execution statement per TFLite operator required');
 const graph=[];
 for(let i=0;i<steps.length;++i){
  const match=steps[i].match(/^([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\)\s*;?$/);
  ok(match,'Unsupported topology statement at '+i);
  const dst=storage.get(match[1]),layer=layers.get(match[2]),op=model.operators[i];
  const inputs=match[3].split(',').map(v=>storage.get(v.trim()));
  const active=op.inputs.filter(id=>id>=0&&!op.parameter_tensors.includes(id));
  ok(layer?.op_index===i&&(layer.kind===op.op || (op.op==='FULLY_CONNECTED' && layer.kind==='FCN') || (op.op==='SOFTMAX'&&layer.kind==='Softmax')),'Layer order/type mismatch at '+i);
  ok(dst&&dst.tensor_id===op.outputs[0]&&inputs.length===active.length&&inputs.every((v,j)=>v&&v.tensor_id===active[j]),'Topology differs from TFLite at '+i);
  ok(layer.params===(op.parameter_tensors.length&&!['MEAN','RESHAPE'].includes(op.op)?'XIP':'-'),'Invalid parameter backend at '+i);
  graph.push({layer,op,dst,inputs});
 }
 for(const s of project.storage){let t=model.tensors[s.tensor_id];ok(t&&t.dtype==='FLOAT32'&&s.shape===t.shape.join(','),'Invalid FP32 activation '+s.name);}
 return graph;
}
// Two-buffer A/B for strict single-input chains only. Output never aliases its input.
function pingPongPlan(project,graph,model){
 ok(graph.length>0,'Ping-pong requires a nonempty graph');
 const result=Sequential.analyze(model,{storage:project.storage});
 ok(result.sequential,'Ping-pong requires a strict single-input sequential chain: '+result.reason+
    (result.operatorIndex>=0?' at op '+result.operatorIndex:''));
 const ids=result.tensorIds;
 // The validated Codegen topology must match the model chain as well.
 for(let i=0;i<graph.length;i++)
  ok(graph[i].inputs.length===1 && graph[i].inputs[0].tensor_id===ids[i] &&
     graph[i].dst.tensor_id===ids[i+1],
     'Ping-pong topology differs from the model chain at op '+i);
 const bytesA=Math.max(...ids.filter((_,i)=>i%2===0).map(id=>align(byteCount(model.tensors[id]))));
 const bytesB=Math.max(...ids.filter((_,i)=>i%2===1).map(id=>align(byteCount(model.tensors[id]))));
 const offsets=new Map(ids.map((id,i)=>[id,i%2===0?0:bytesA]));
 const arenaBytes=bytesA+bytesB;
 ok(arenaBytes<=8*1024*1024,'Ping-pong activation buffers exceed 8 MiB: '+arenaBytes);
 return {offset:offsets,arenaBytes,peakLive:arenaBytes,method:'pingpong',bufferABytes:bytesA,bufferBBytes:bytesB};
}
// Default FP32 arena plan: calculate live intervals and reuse only dead regions.
function memoryPlan(project,graph,model){
 const last=new Map(),inputId=model.inputs[0];
 for(const [i,g] of graph.entries())for(const b of g.inputs)last.set(b.tensor_id,i);
 last.set(model.outputs[0],graph.length);
 const regions=new Map(),free=[];let highwater=0,peakLive=0,live=0;
 function compact(){free.sort((a,b)=>a.off-b.off);for(let i=0;i<free.length-1;){if(free[i].off+free[i].size===free[i+1].off){free[i].size+=free[i+1].size;free.splice(i+1,1);}else i++;}}
 function put(id){ok(!regions.has(id),'Tensor allocated more than once');let size=align(byteCount(model.tensors[id])),slot=free.findIndex(x=>x.size>=size);let offset;
  if(slot>=0){offset=free[slot].off;if(free[slot].size===size)free.splice(slot,1);else{free[slot].off+=size;free[slot].size-=size;}}
  else{offset=highwater;highwater+=size;}regions.set(id,{off:offset,size});live+=size;peakLive=Math.max(peakLive,live);
 }
 const assigned=new Map();put(inputId);assigned.set(inputId,regions.get(inputId).off);
 for(const [i,g] of graph.entries()){
  const id=g.dst.tensor_id;put(id);assigned.set(id,regions.get(id).off);
  for(const s of g.inputs){const sid=s.tensor_id;if(last.get(sid)===i){const r=regions.get(sid);free.push({...r});live-=r.size;regions.delete(sid);compact();}}
 }
 ok(highwater <= 8*1024*1024,'FP32 PSRAM tensor arena exceeds 8 MiB: '+highwater);
 return {offset:assigned,arenaBytes:highwater,peakLive,method:'liveness'};
}
function generate(project){
 ok(project.source?.type==='tflite' && project.source.model_base64,'Embedded TFLite source required');
 const source=T.base64Decode(project.source.model_base64),model=T.inspectTFLite(source);
 const graph=validateGraph(project,model);
 const plan=project.settings?.apply_pingpong_sequential===true ? pingPongPlan(project,graph,model) : memoryPlan(project,graph,model);
 const maxImage=0x800000,buffer=new Uint8Array(maxImage),view=new DataView(buffer.buffer),layouts=[];let cursor=16;
 const cache=new Map();
 function add(id){if(cache.has(id))return cache.get(id);
  const raw=T.extractTensorData(source,id);ok(raw.length&&raw.length%4===0,'Empty/misaligned constant '+id);
  cursor=align(cursor,4);ok(cursor+raw.length<=maxImage,'FP32 parameters exceed 8MB model partition');
  const off=cursor;buffer.set(raw,off);cursor+=raw.length;cache.set(id,off);layouts.push({tensor:id,offset:off,bytes:raw.length});return off;
 }
 const opTypes={CONV_2D:1,MAX_POOL_2D:2,CONCATENATION:3,SUB:4,MUL:5,MEAN:6,RESHAPE:7,AVERAGE_POOL_2D:8,FULLY_CONNECTED:9,SOFTMAX:10};
 const ops=[];
 for(const {op,inputs,dst,layer} of graph){
  const a=model.tensors[inputs[0].tensor_id],d=model.tensors[dst.tensor_id],cfg=op.config;
  const x={kind:opTypes[op.op],relu:0,src0:inputs[0].index,src1:inputs[1]?.index??0,dst:dst.index,kh:0,kw:0,sh:1,sw:1,weights:NO,bias:NO,name:layer.name};
  if(op.op==='CONV_2D'){
   ok(op.parameter_tensors.length===2,'Conv must have weights and bias');
   const w=model.tensors[op.parameter_tensors[0]],b=model.tensors[op.parameter_tensors[1]];
   ok(w.dtype==='FLOAT32'&&b.dtype==='FLOAT32'&&w.shape.length===4&&w.shape[0]===d.shape[3]&&w.shape[3]===a.shape[3]&&b.shape[0]===w.shape[0],'Unsupported conv weights');
   ok((cfg.activation==='NONE'||cfg.activation==='RELU')&&cfg.dilation_h===1&&cfg.dilation_w===1,'Unsupported conv fused activation/dilation');
   x.kh=w.shape[1];x.kw=w.shape[2];x.sh=cfg.stride_h;x.sw=cfg.stride_w;x.relu=cfg.activation==='RELU'?1:0;x.weights=add(w.id);x.bias=add(b.id);
  }else if(op.op==='MAX_POOL_2D'||op.op==='AVERAGE_POOL_2D'){
   ok(cfg.activation==='NONE'||(op.op==='AVERAGE_POOL_2D'&&cfg.activation==='RELU'),'Unsupported pool activation');x.kh=cfg.filter_h;x.kw=cfg.filter_w;x.sh=cfg.stride_h;x.sw=cfg.stride_w;x.relu=cfg.activation==='RELU'?1:0;
  }else if(op.op==='SUB'||op.op==='MUL'){
   ok(op.parameter_tensors.length===1,'RGB arithmetic needs constant');
   const t=model.tensors[op.parameter_tensors[0]];ok(t.dtype==='FLOAT32'&&t.shape.join(',')==='3','RGB arithmetic constant expected');x.weights=add(t.id);
  }else if(op.op==='CONCATENATION'){
   ok(inputs.length===2&&cfg.axis===-1&&cfg.activation==='NONE','Only two-branch NHWC concat supported');
  }else if(op.op==='MEAN'){
   ok(cfg.keep_dims===true&&op.parameter_tensors.length===1,'Spatial mean required');
   const data=T.extractTensorData(source,op.parameter_tensors[0]);const axes=new DataView(data.buffer,data.byteOffset,data.byteLength);
   ok(data.length===8&&axes.getInt32(0,true)===1&&axes.getInt32(4,true)===2,'Only H,W mean axes supported');
  }else if(op.op==='FULLY_CONNECTED'){
   ok(op.parameter_tensors.length===2,'FullyConnected needs FLOAT32 weights and bias');
   const w=model.tensors[op.parameter_tensors[0]],b=model.tensors[op.parameter_tensors[1]];
   ok(w.dtype==='FLOAT32'&&b.dtype==='FLOAT32'&&w.shape.length===2&&w.shape[0]===d.shape.at(-1)&&w.shape[1]===a.shape.reduce((v,k)=>v*k,1)&&b.shape.length===1&&b.shape[0]===w.shape[0],'Unsupported fully connected tensor dimensions');
   ok((cfg.activation==='NONE'||cfg.activation==='RELU')&&cfg.weights_format===0&&!cfg.keep_num_dims,'Unsupported FCN options');
   x.relu=cfg.activation==='RELU'?1:0;x.weights=add(w.id);x.bias=add(b.id);
  }else if(op.op==='SOFTMAX'){
   ok(cfg.beta===1&&a.shape.reduce((v,k)=>v*k,1)===d.shape.reduce((v,k)=>v*k,1),'Softmax requires beta=1 and matching shape');
  }else if(op.op==='RESHAPE')ok(a.shape.reduce((x,y)=>x*y,1)===d.shape.reduce((x,y)=>x*y,1),'Invalid reshape');
  ops.push(x);
 }
 // Model header is separate from FP32 weights, firmware validates magic, size and CRC.
 buffer.set([78,68,76,70,88,73,80,0],0);view.setUint32(8,cursor,true);view.setUint32(12,crc32(buffer.subarray(16,cursor)),true);
 const params=buffer.slice(0,cursor);
 const tensorDecl=project.storage.map(s=>{
  let shape=model.tensors[s.tensor_id].shape;let dims=shape.length===4?shape.slice(1):[1,1,shape.at(-1)];
  ok(dims.length===3&&dims.every(v=>v>0&&v<=65535),'Unsupported tensor rank '+s.name);
  return `  {nullptr, ${byteCount(model.tensors[s.tensor_id])/4}u, ${dims.join(', ')}}, // ${s.name}`;
 }).join('\n');
 const opDecl=ops.map(o=>`  {${o.kind}, ${o.relu}, ${o.src0}, ${o.src1}, ${o.dst}, ${o.kh}, ${o.kw}, ${o.sh}, ${o.sw}, ${o.weights}u, ${o.bias}u}, // ${o.name}`).join('\n');
 const ptrOffsets=project.storage.map(s=>plan.offset.get(s.tensor_id));
 ok(ptrOffsets.every(x=>Number.isInteger(x)),'Unassigned tensor arena offsets');
 const inputIdx=project.storage.findIndex(s=>s.tensor_id===model.inputs[0]),outputIdx=project.storage.findIndex(s=>s.tensor_id===model.outputs[0]);
 const serialName='NOODLE_GENERIC_FP32_N16R8';
 const inputShape=model.tensors[model.inputs[0]].shape,outputShape=model.tensors[model.outputs[0]].shape;
 const inputElements=byteCount(model.tensors[model.inputs[0]])/4,outputElements=byteCount(model.tensors[model.outputs[0]])/4;
 ok(inputShape.length===4&&inputShape[0]===1 || inputShape.length===2&&inputShape[0]===1,'FP32 serial input supports batch-one NHWC/flat tensors');
 ok(outputElements<=4096,'Serial FLOAT32 output limited to 4096 elements');
 const main=`// Noodle Studio ESP32-S3 generic FP32, serial test.\n#include <Arduino.h>\n#include <esp_heap_caps.h>\n#include <esp_partition.h>\n#include <esp_spi_flash.h>\n#include <string.h>\n#include <math.h>\n#include <noodle_s3_nhwc_fp32.h>\n
static constexpr uint32_t EXPECTED_BYTES=${cursor}u;
static constexpr uint32_t EXPECTED_CRC=0x${crc32(params.subarray(16)).toString(16)}u;
static constexpr uint32_t INPUT_IDX=${inputIdx}u,OUTPUT_IDX=${outputIdx}u;
static constexpr uint32_t SERIAL_BYTES=${inputElements*4}u,CHUNK=32u;
static const char *opNames[]={${ops.map(x=>JSON.stringify(x.name)).join(',')}};
static NoodleS3F32Tensor tensors[]={
${tensorDecl}
};
static const uint32_t offsets[]={${ptrOffsets.map(x=>x+'u').join(',')}};
static const NoodleS3F32Op operators[]={
${opDecl}
};
static uint8_t*arena=nullptr;
static const uint8_t*flash=nullptr;
static spi_flash_mmap_handle_t mappedHandle=0;
static bool initialized=false;
static uint32_t crc32(const uint8_t*data,size_t len){uint32_t c=0xffffffffu;for(size_t i=0;i<len;++i){c^=data[i];for(int bit=0;bit<8;++bit)c=(c>>1)^((c&1u)?0xedb88320u:0u);}return ~c;}
static bool setupMemory(){
 if(!psramFound()){Serial.println("ERR_PSRAM_MISSING");return false;}
 Serial.printf("PSRAM total=%u free=%u\\n",(unsigned)ESP.getPsramSize(),(unsigned)ESP.getFreePsram());
 const auto*part=esp_partition_find_first(ESP_PARTITION_TYPE_DATA,(esp_partition_subtype_t)0x40,"model");
 if(!part||part->size<EXPECTED_BYTES){Serial.println("ERR_MODEL_PARTITION");return false;}
 const void*mapped=nullptr;
 if(esp_partition_mmap(part,0,EXPECTED_BYTES,SPI_FLASH_MMAP_DATA,&mapped,&mappedHandle)!=ESP_OK){Serial.println("ERR_FLASH_MMAP");return false;}
 flash=(const uint8_t*)mapped;
 if(memcmp(flash,"NDLFXIP",7)!=0||flash[7]!=0||(uint32_t(flash[8])|(uint32_t(flash[9])<<8)|(uint32_t(flash[10])<<16)|(uint32_t(flash[11])<<24))!=EXPECTED_BYTES){Serial.println("ERR_MODEL_HEADER");return false;}
 if(crc32(flash+16,EXPECTED_BYTES-16)!=EXPECTED_CRC){Serial.println("ERR_MODEL_CRC");return false;}
 arena=(uint8_t*)heap_caps_malloc(${plan.arenaBytes}u,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
 if(!arena){Serial.println("ERR_PSRAM_ARENA_ALLOC");return false;}
 for(size_t k=0;k<sizeof(tensors)/sizeof(*tensors);++k)tensors[k].data=(float*)(arena+offsets[k]);
 Serial.printf("TENSORS=%u OPS=%u PARAMS=%u ARENA=%u FREE_PSRAM=%u\\n",(unsigned)(sizeof(tensors)/sizeof(*tensors)),(unsigned)(sizeof(operators)/sizeof(*operators)),(unsigned)EXPECTED_BYTES,(unsigned)${plan.arenaBytes}u,(unsigned)ESP.getFreePsram());
 return true;
}
static bool receiveInput(){
 float*input=tensors[INPUT_IDX].data;
 uint8_t*raw=(uint8_t*)input;
 Serial.println("RDYIMG");
 uint32_t off=0;
 while(off<SERIAL_BYTES){
  uint32_t n=SERIAL_BYTES-off<CHUNK?SERIAL_BYTES-off:CHUNK,k=0;
  uint32_t last=millis();
  while(k<n){int b=Serial.read();if(b>=0){raw[off+k++]=(uint8_t)b;last=millis();}else if(millis()-last>15000){Serial.println("ERR_INPUT_TIMEOUT");return false;}else delay(1);}
  off+=n;if(n==CHUNK){Serial.write('K');Serial.flush();}
 }
 // Raw little-endian FLOAT32 tensor bytes: no model-specific RGB normalization.
 return true;
}
static void runModel(){
 uint32_t t0=millis();
 for(size_t i=0;i<sizeof(operators)/sizeof(*operators);++i){
  uint32_t step=millis();
  if(!noodle_s3_f32_execute(operators[i],tensors,sizeof(tensors)/sizeof(*tensors),flash,EXPECTED_BYTES)){
   Serial.printf("ERR_INFERENCE_OP %u %s\\n",(unsigned)i,opNames[i]);Serial.println("READY");return;
  }
  Serial.printf("OP %u %-15s %lu ms\\n",(unsigned)i,opNames[i],(unsigned long)(millis()-step));
 }
 auto&out=tensors[OUTPUT_IDX];uint32_t best=0;
 for(uint32_t k=1;k<out.elements;++k)if(out.data[k]>out.data[best])best=k;
 Serial.printf("PRED %u MS %lu\\n",(unsigned)best,(unsigned long)(millis()-t0));
 Serial.print("OUTPUT_FLOAT32");
 for(uint32_t k=0;k<out.elements;++k){Serial.write(' ');Serial.print(out.data[k],7);}Serial.println();Serial.println("READY");
}
void setup(){
 Serial.begin(115200);delay(1200);Serial.println("${serialName}");
 initialized=setupMemory();Serial.println(initialized?"READY":"BOOT_FAILED");
}
void loop(){
 if(!initialized){delay(1000);return;}
 static char cmd[12];static uint8_t used=0;
 while(Serial.available()){
  int c=Serial.read();if(c==13)continue;
  if(c=='\\n'){cmd[used]=0;used=0;
   if(strcmp(cmd,"HELLO")==0)Serial.println("READY");
   else if(strcmp(cmd,"IMG")==0){if(receiveInput())runModel();else Serial.println("READY");}
   else{Serial.println("ERR_CMD");Serial.println("READY");}
  }else if(used<sizeof(cmd)-1)cmd[used++]=(char)c;
  else{used=0;Serial.println("ERR_CMD_LENGTH");Serial.println("READY");}
 }
 delay(1);
}
`;
 const ini=`[env:esp32-s3-n16r8-fp32]\nplatform = espressif32@6.12.0\nboard = esp32-s3-devkitc-1\nframework = arduino\nboard_build.flash_size = 16MB\nboard_upload.flash_size = 16MB\nboard_build.arduino.memory_type = qio_opi\nboard_build.flash_mode = qio\nboard_build.psram_type = opi\nboard_build.partitions = partitions_16mb_squeezenet_fp32.csv\nboard_upload.maximum_size = 7340032\nextra_scripts = pre:pio_flash_model.py\nbuild_flags =\n  -D BOARD_HAS_PSRAM\n  -D NOODLE_USE_NONE\n  -D ARDUINO_USB_MODE=1\n  -D ARDUINO_USB_CDC_ON_BOOT=1\nmonitor_speed = 115200\nupload_protocol = esptool\n`;
 const optIni=ini.replace('framework = arduino\n','framework = arduino\nbuild_type = release\n')+'\n; Optional -O2 speed trial; the original release build remains the baseline.\n' +'[env:esp32-s3-n16r8-fp32-o2]\nextends = env:esp32-s3-n16r8-fp32\nbuild_type = release\nbuild_unflags = -Os\nbuild_flags =\n' +'  ${env:esp32-s3-n16r8-fp32.build_flags}\n  -O2\n';
 const optimizeReadme="# Compiler optimization comparison — ESP32-S3 N16R8\n\nThese Noodle Studio ESP32-S3 firmware exports include two PlatformIO environments, both built in release mode:\n\n- **Baseline:** `esp32-s3-n16r8` (INT8) or `esp32-s3-n16r8-fp32` (FP32). PlatformIO's default compiler optimization is preserved (normally `-Os` for Arduino ESP32).\n- **Speed experiment:** the matching environment with `-o2` suffix. Explicitly removes `-Os` and applies `-O2` for code compiled from source (including Noodle kernels).\n\n### Benchmark commands\n\nINT8:\n\n```sh\npio run -e esp32-s3-n16r8-o2 -t upload\npython3 test/send_tensor.py --port /dev/ttyACM0 --pattern zeros\n```\n\nFP32:\n\n```sh\npio run -e esp32-s3-n16r8-fp32-o2 -t upload\npython3 test/send_tensor.py --port /dev/ttyACM0 --pattern zeros\n```\n\nUse the **USB OTG** port for native USB CDC serial on the Freenove-style ESP32-S3 board. Close the monitor before the sender. Uploading still writes the XIP parameter image at the normal dedicated partition offset.\n\n### Scientific comparison\n\nRecord median and variability over repeated identical image inferences, code/flash size, free PSRAM, and output checksums. Changing optimization level may change floating-point rounding; compare numeric output to TFLite rather than only class predictions.\n\n`-Os` is optimized for **code size**. `-O2` targets **execution speed** but can produce **larger code and more instruction-cache pressure**. Neither is guaranteed to be faster on the ESP32-S3. Precompiled Arduino framework binaries may not inherit all of the application optimization settings. Do not use `-Ofast` until numerical equivalence has been established.\n\nThe original baseline environments remain available and unchanged except for explicit `build_type = release`.\n";
 const partitions=`# Name, Type, SubType, Offset, Size, Flags\nnvs,data,nvs,0x9000,0x5000,\notadata,data,ota,0xe000,0x2000,\napp0,app,ota_0,0x10000,0x700000,\nmodel,data,0x40,0x800000,0x800000,\n`;
 const uploader=`Import("env")\nfrom os.path import join\nenv.Append(FLASH_EXTRA_IMAGES=[("0x800000", join(env.subst("$PROJECT_DIR"), "model", "params.bin"))])\n`;
 const sender= root.NOODLE_GENERIC_FP32_SENDER ||(typeof require!=='undefined'?require('./generic_fp32_sender_bundle.js'):null);
 ok(sender,'FP32 sender bundle missing');
 const spec={protocol:'noodle-serial-img-v1',shape:inputShape,dtype:'FLOAT32',transport_dtype:'float32-le',bytes:inputElements*4,layout:inputShape.length===4?'NHWC':'FLAT',input_scaling:'none — raw FP32 transport',baud:115200,chunk_bytes:32,output_elements:outputElements};
 const outputSpec={shape:outputShape,dtype:'FLOAT32'};
 const lib=root.NOODLE_LIB_FILES||(typeof require!=='undefined'?require('./noodle_bundle.js'):null);
 const h=root.NOODLE_S3_GENERIC_FP32_H||(typeof require!=='undefined'?require('fs').readFileSync(__dirname+'/noodle_s3_generic_fp32.h','utf8'):null);
 const cpp=root.NOODLE_S3_GENERIC_FP32_CPP||(typeof require!=='undefined'?require('fs').readFileSync(__dirname+'/noodle_s3_generic_fp32.cpp','utf8'):null);
 ok(lib&&h&&cpp,'Noodle FP32 library sources missing');
 const guide=`# ESP32-S3 N16R8 — generic FLOAT32 model (experimental)\n\nThis export is independent of the tested INT8 backend. Imported FP32 TFLite has ${graph.length} operations. Activations use ${plan.method==='pingpong' ? `two fixed A/B ping-pong regions (${plan.bufferABytes} + ${plan.bufferBBytes} = ${plan.arenaBytes} bytes)` : `a statically lifetime-reused PSRAM arena (${plan.arenaBytes} bytes, peak live ${plan.peakLive} bytes)`}. Model FP32 weights/biases use a 8 MiB memory-mapped flash partition at 0x800000 (params.bin ${cursor} bytes). FFat is not used.\n\nConnect the USB OTG port for the serial sender.\n\n1. Build/flash firmware and parameters together: pio run -e esp32-s3-n16r8-fp32 -t upload.\n2. Install: pip install pyserial Pillow.\n3. Run: python3 test/send_tensor.py --port /dev/ttyACM0 --pattern zeros (or use --input-file for exact raw FP32 tensors).\n4. Compare with original FP32 TFLite using test/reference.py.\n\nThe serial payload contains ${inputElements} raw little-endian FLOAT32 values. No implicit normalization is done. Outputs contain ${outputElements} FLOAT32 values. Use --image only if its normalization matches training.\n\n**Validation status:** This generator's graph and buffer planner are tested locally. Physical ESP32-S3 execution, numerical equivalence to TFLite, and FP32 latency for NEW graphs are NOT yet verified. The 125.1 s previously published result used different buffer scheduling and operators, so exact timing equality is not guaranteed.\n`;
 const reference=`#!/usr/bin/env python3
import argparse,numpy as np
from pathlib import Path
try:
 from tflite_runtime.interpreter import Interpreter
except ImportError:
 import tensorflow as tf
 Interpreter=tf.lite.Interpreter
p=argparse.ArgumentParser()
p.add_argument('--model',default='model/source.tflite')
p.add_argument('--input-file',help='Raw little-endian FP32 input (must match the model tensor)')
a=p.parse_args()
i=Interpreter(model_path=a.model);i.allocate_tensors()
di=i.get_input_details()[0];do=i.get_output_details()[0]
if a.input_file:
 x=np.frombuffer(Path(a.input_file).read_bytes(),dtype='<f4').reshape(di['shape']).copy()
else: x=np.zeros(di['shape'],dtype=np.float32)
i.set_tensor(di['index'],x);i.invoke()
y=i.get_tensor(do['index']).ravel()
print('PRED',int(np.argmax(y)))
print('OUTPUT_FLOAT32',' '.join(f'{v:.7f}' for v in y))
`;

 const files={...lib,'src/main.cpp':main,'platformio.ini':optIni,'README_COMPILER_OPTIMIZATION.md':optimizeReadme,'partitions_16mb_squeezenet_fp32.csv':partitions,
  'pio_flash_model.py':uploader,'lib/Noodle/src/noodle_s3_nhwc_fp32.h':h,'lib/Noodle/src/noodle_s3_nhwc_fp32.cpp':cpp,
  'model/params.bin':params,'model/source.tflite':source,'model/project.json':JSON.stringify(project,null,2)+'\n',
  'model/manifest.json':JSON.stringify({precision:'fp32',model_sha256:project.source.model_sha256,params_bytes:cursor,crc32:crc32(params.subarray(16)),arena_bytes:plan.arenaBytes,peak_live_bytes:plan.peakLive,memory_strategy:plan.method,partition_offset:'0x800000',constant_offsets:layouts},null,2)+'\n',
  'test/send_tensor.py':sender,'test/input_spec.json':JSON.stringify(spec,null,2)+'\n','test/output_spec.json':JSON.stringify(outputSpec,null,2)+'\n','test/reference.py':reference,'README_ESP32S3_GENERIC_FP32.md':guide};
 delete files['lib/Noodle/src/noodle_s3_nhwc_i8.cpp'];
 delete files['lib/Noodle/src/noodle_s3_nhwc_i8.h'];
 delete files['lib/Noodle/src/noodle_i8_nhwc.h'];
 delete files['lib/Noodle/src/noodle_i8_nhwc.cpp'];
 return {files,summary:{layers:ops,sdLayers:0,sequentialPingPong:plan.method==='pingpong',arenaBytes:plan.arenaBytes,paramsBytes:cursor,peakLive:plan.peakLive}};
}
root.NoodleESP32S3GenericFP32={generate,memoryPlan,pingPongPlan};
if(typeof module!=='undefined'&&module.exports)module.exports={generate,memoryPlan,pingPongPlan};
})(typeof globalThis!=='undefined'?globalThis:this);
