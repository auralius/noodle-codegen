/* Noodle Studio R15: ESP32-S3 N16R8 XIP+PSRAM W8A8 graph exporter.
 * Keeps the Uno/Nano Codegen unmodified. Supported first target: UINT8 input,
 * INT8 NHWC CONV_2D / MAX_POOL_2D / CONCATENATION / QUANTIZE / SUB / MUL /
 * spatial MEAN / RESHAPE, UINT8 output. Fail closed for anything else.
 */
'use strict';
(function(root){
const importTFLite=root.NoodleTFLite||(typeof require!=='undefined'?require('./tflite_import.js'):null);
const BranchPlanner=root.NoodleBranchBufferPlanner||(typeof require!=='undefined'?require('./branch_buffer_planner.js'):null);
const Sequential=root.NoodleSequentialNetwork||(typeof require!=='undefined'?require('./sequential_network_detector.js'):null);
const BAD=0xffffffff;
function check(ok,message){if(!ok)throw Error('ESP32-S3: '+message);}
function fmt(v){check(Number.isFinite(v),'Invalid floating point parameter');return Number(v).toExponential(9)+'f';}
function bytes4(b){return new DataView(b.buffer,b.byteOffset,b.byteLength);}
function qmult(x){check(x>0 && Number.isFinite(x),'Invalid quantization multiplier '+x);
 let e=0,q=x;while(q<0.5){q*=2;--e;check(e>=-31,'Quantization multiplier underflow');}
 while(q>=1){q/=2;++e;check(e<=30,'Quantization multiplier overflow');}
 let n=Math.floor(q*2147483648+0.5);if(n===2147483648){n/=2;++e;}return [n,e];
}
function align4(n){return(n+3)&~3;}
function crc32(b){let x=0xffffffff;for(const v of b){x^=v;for(let i=0;i<8;i++)x=(x>>>1) ^ ((x&1)?0xedb88320:0);}return (x^0xffffffff)>>>0;}
function numshape(s){return typeof s==='string'?s.split(',').map(Number):s;}
function nbytes(s){return s.reduce((a,b)=>a*b,1);}
function tquant(t){check(t&&t.quantization&&t.quantization.scales.length===1&&t.quantization.zero_points.length===1,'Tensor '+(t?.id??'?')+' needs per-tensor quantization');return {scale:t.quantization.scales[0],zp:t.quantization.zero_points[0]};}
function typed(v){return (v===null||v===undefined)?BAD:v;}
function generate(project){
 check(project?.target==='esp32s3','Only ESP32-S3 projects accepted');
 check(project?.source?.type==='tflite'&&project.source.model_base64,'Embedded TFLite model required');
 check(project.settings?.apply_pingpong_sequential!==true,'Disable sequential ping-pong on this branching graph');
 const bin=importTFLite.base64Decode(project.source.model_base64);
 const model=importTFLite.inspectTFLite(bin);
 const supported=new Set(['CONV_2D','MAX_POOL_2D','CONCATENATION','QUANTIZE','SUB','MUL','MEAN','RESHAPE']);
 check(model.inputs.length===1 && model.outputs.length===1,'Single input/output required');
 check(model.tensors[model.inputs[0]].dtype==='UINT8','UINT8 RGB input required');
 check(model.tensors[model.outputs[0]].dtype==='UINT8','UINT8 output required');
 check(model.tensors[model.inputs[0]].shape.join(',')==='1,224,224,3','This first release requires 224x224 RGB');
 check(model.operators.every(o=>supported.has(o.op)),'Unsupported operator in TFLite graph');
 check(project.layers.length===model.operators.length,'Layer count diverges from TFLite');
 const byStorage=new Map(project.storage.map((b,i)=>[b.name,{...b,index:i}]));
 check(project.storage.every(s=>s.backend==='PSRAM' && Number.isInteger(s.tensor_id)), 'Every activation must be PSRAM with a TFLite tensor ID');
 const byLayer=new Map(project.layers.map(x=>[x.name,x]));
 const steps=project.topology.split('\n').map(x=>x.split('//')[0].trim()).filter(Boolean);
 check(steps.length===model.operators.length,'One topology call per original operator required (no destroys)');
 const graph=[];
 for(let i=0;i<steps.length;i++){
  const m=steps[i].match(/^([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_][\w]*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\)\s*;?$/);
  check(m,'Unsupported topology statement '+steps[i]);
  const layer=byLayer.get(m[2]),op=model.operators[i];
  check(layer?.op_index===i && layer.kind===op.op,'Topology must retain original TFLite execution order at '+i);
  const dst=byStorage.get(m[1]),inputs=m[3].split(',').map(v=>byStorage.get(v.trim()));
  check(dst && inputs.every(Boolean),'Missing storage in '+steps[i]);
  const actualInputs=op.inputs.filter(id=>id>=0&&!op.parameter_tensors.includes(id));
  check(actualInputs.length===inputs.length && inputs.every((b,j)=>b.tensor_id===actualInputs[j]) && dst.tensor_id===op.outputs[0],
   'Tensor graph changed at '+layer.name);
  check(layer.params===(op.parameter_tensors.length && !['MEAN','RESHAPE'].includes(op.op)?'XIP':'-'),
   'Parameter backend mismatch at '+layer.name);
  graph.push({op,layer,dst,inputs});
 }
 // Revalidate each activation against the TFLite source; no implicit planner.
 const tensors=[];
 for(const b of project.storage){
  const t=model.tensors[b.tensor_id];check(t,'Unknown tensor '+b.tensor_id);
  check(t.dtype==='INT8'||t.dtype==='UINT8','Only quantized activations supported: '+b.name);
  check(numshape(b.shape).join(',')===t.shape.join(','),'Shape changed for '+b.name);
  const shape=t.shape, q=tquant(t), dims=shape.length===4?shape.slice(1):shape.length===2?[1,1,shape[1]]:null;
  check(dims && dims.every(v=>v>0 && v<=65535),'Unsupported activation rank for '+b.name);
  tensors.push({dims,bytes:nbytes(shape),q,type:t.dtype,name:b.name});
 }
 // Keep the existing sequential paths untouched. Optimize only branched INT8 graphs.
 // Explicit false provides a byte-for-byte legacy allocation strategy fallback.
 const strictlySequential=Sequential?.analyze(model,{storage:project.storage}).sequential===true;
 const reuseBranches=!strictlySequential && project.settings?.reuse_branch_buffers!==false;
 check(!reuseBranches || BranchPlanner?.plan,'Branch planner module missing');
 const memoryPlan=reuseBranches?BranchPlanner.plan(model,graph,project.storage):null;
 check(!memoryPlan || memoryPlan.buffers.every(b=>b.backend==='PSRAM'),'Unsupported physical buffer backend');
 const params=new Uint8Array(2000000),layout=[],ids=new Map(),used=new Set();let cursor=16;
 function addParam(name,data){
  check(data && data.length>0,'Missing constant '+name);
  cursor=align4(cursor);check(cursor+data.length<=params.length,'Model params exceed 2 MB flash partition');
  const off=cursor;params.set(data,off);cursor+=data.length;
  layout.push({name,offset:off,bytes:data.length});return off;
 }
 function getTensorData(tid){return importTFLite.extractTensorData(bin,tid);}
 function p(tid){if(ids.has(tid))return ids.get(tid);
  const v=addParam('tflite_tensor_'+tid,getTensorData(tid));ids.set(tid,v);return v;
 }
 const ops=[];
 for(const entry of graph){
  const {op,inputs,dst,layer}=entry,source=model.tensors[inputs[0].tensor_id],output=model.tensors[dst.tensor_id];
  const iq=tquant(source),oq=tquant(output),cfg=op.config;
  const desc={id:op.id,name:layer.name,kind:op.op,src0:inputs[0].index,src1:inputs[1]?.index??0,dst:dst.index,
   kh:0,kw:0,sh:1,sw:1,weights:BAD,bias:BAD,mult:BAD,shift:BAD,scale:1,zp:0,relu:0};
  if(op.op==='CONV_2D'){
    check(op.parameter_tensors.length===2,'Conv requires INT8 weights + INT32 bias');
    const w=model.tensors[op.parameter_tensors[0]],b=model.tensors[op.parameter_tensors[1]];
    check(w.dtype==='INT8' && b.dtype==='INT32'&&w.shape.length===4&&w.shape[0]===output.shape.at(-1)&&w.shape[3]===source.shape.at(-1),'Unsupported convolution weights');
    check(!w.quantization.zero_points.some(z=>z!==0),'Conv weight zero points must be 0');
    check(w.quantization.scales.length===w.shape[0] && b.shape[0]===w.shape[0],'Per-output channel weight scales required');
    check(cfg.activation==='RELU'||cfg.activation==='NONE','Unsupported conv activation');
    check(cfg.stride_h>0&&cfg.stride_w>0&&cfg.dilation_h===1&&cfg.dilation_w===1,'Unsupported convolution stride/dilation');
    desc.kh=w.shape[1];desc.kw=w.shape[2];desc.sh=cfg.stride_h;desc.sw=cfg.stride_w;
    const h=source.shape[1],ow=output.shape[2],oh=output.shape[1],iw=source.shape[2];
    const dim=(n,k,s,pad)=>pad===0?Math.ceil(n/s):Math.floor((n-k+s)/s);
    check(oh===dim(h,desc.kh,desc.sh,cfg.padding) && ow===dim(iw,desc.kw,desc.sw,cfg.padding),'Convolution spatial shape mismatch');
    desc.weights=p(w.id);desc.bias=p(b.id);desc.relu=cfg.activation==='RELU'?1:0;
    const mult=new Uint8Array(w.shape[0]*4),shift=new Uint8Array(w.shape[0]*4),mv=bytes4(mult),sv=bytes4(shift);
    w.quantization.scales.forEach((scale,i)=>{const [m,s]=qmult(iq.scale*scale/oq.scale);mv.setInt32(i*4,m,true);sv.setInt32(i*4,s,true);});
    desc.mult=addParam('op'+op.id+'_multipliers',mult);desc.shift=addParam('op'+op.id+'_shifts',shift);
  } else if(op.op==='MAX_POOL_2D'){
    check(cfg.filter_h>0&&cfg.filter_w>0&&cfg.stride_h>0&&cfg.stride_w>0&&cfg.activation==='NONE','Unsupported pool options');
    desc.kh=cfg.filter_h;desc.kw=cfg.filter_w;desc.sh=cfg.stride_h;desc.sw=cfg.stride_w;
    check(oq.scale===iq.scale && oq.zp===iq.zp,'MaxPool quant mismatch');
  } else if(op.op==='CONCATENATION'){
    check(inputs.length===2 && cfg.axis===-1 && cfg.activation==='NONE','Only two-input NHWC channel concat supported');
    check(inputs.every(i=>{const q=tquant(model.tensors[i.tensor_id]);return q.scale===oq.scale&&q.zp===oq.zp;}),'Concat inputs must share quantization');
  } else if(op.op==='SUB'||op.op==='MUL'){
    check(op.parameter_tensors.length===1,'Binary operation needs a constant');
    const c=model.tensors[op.parameter_tensors[0]];
    check(c.dtype==='INT8'&&c.shape.join(',')==='3','Only RGB channel constants supported');
    const cq=tquant(c);desc.weights=p(c.id);desc.scale=cq.scale;desc.zp=cq.zp;
    check(cfg.activation==='NONE','Unsupported binary fused activation');
  } else if(op.op==='MEAN'){
    check(op.parameter_tensors.length===1 && cfg.keep_dims===true,'Spatial MEAN with keep_dims required');
    const d=getTensorData(op.parameter_tensors[0]);check(d.length===8,'Expected 2 int32 spatial axes');
    const v=bytes4(d);check(v.getInt32(0,true)===1&&v.getInt32(4,true)===2,'Only axes 1,2 spatial mean supported');
  } else if(op.op==='RESHAPE'){
    check(op.parameter_tensors.length===1,'RESHAPE requires a shape constant');
    check(nbytes(source.shape)===nbytes(output.shape),'Reshape byte length mismatch');
  }
  ops.push(desc);
 }
 const payload=params.subarray(16,cursor);const hdr=bytes4(params);
 params.set([78,68,76,56,88,73,80,0],0); // NDL8XIP\0
 hdr.setUint32(8,cursor,true);hdr.setUint32(12,crc32(payload),true);
 const modelBin=params.slice(0,cursor);
 const kind={CONV_2D:1,MAX_POOL_2D:2,CONCATENATION:3,QUANTIZE:4,SUB:5,MUL:6,MEAN:7,RESHAPE:8};
 const specs=tensors.map(t=>`  {nullptr, ${t.bytes}u, ${t.dims.map(x=>x+'u').join(', ')}, ${t.q.zp}, ${fmt(t.q.scale)}, ${t.type==='UINT8'?'true':'false'}}, // ${t.name}`).join('\n');
 const descriptors=ops.map(o=>`  {${kind[o.kind]}, ${o.relu}, ${o.src0}, ${o.src1}, ${o.dst}, ${o.kh}, ${o.kw}, ${o.sh}, ${o.sw}, ${o.weights}u, ${o.bias}u, ${o.mult}u, ${o.shift}u, ${fmt(o.scale)}, ${o.zp}}, // ${o.name}`).join('\n');
 const names=ops.map(o=>JSON.stringify(o.name)).join(', ');
 // All logical activations reference a physical NoodleTensor/NoodleBuffer.
 // The branch planner determines lifetimes; buffer bindings carry storage.
 const physicalSizes=memoryPlan ? memoryPlan.buffers.map(b=>b.bytes) : tensors.map(t=>t.bytes);
 const physicalIds=memoryPlan ? memoryPlan.tensorBufferIds : tensors.map((_,i)=>i);
 const physicalCount=physicalSizes.length;
 const poolDefinitions=`static const uint32_t bufferSizes[]={${physicalSizes.map(n=>n+'u').join(',')}};\nstatic const uint16_t tensorBufferIds[]={${physicalIds.join(',')}};\nstatic NoodleTensor physicalTensors[${physicalCount}]={};\nstatic uint8_t *physicalBuffers[${physicalCount}]={};\n`;
 const initAlloc=`  for(size_t i=0;i<${physicalCount}u;i++){
    physicalBuffers[i]=(uint8_t*)heap_caps_malloc(bufferSizes[i],MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
    if(!physicalBuffers[i]){Serial.println("ERR_PSRAM_ALLOC");return false;}
    if(!noodle_buffer_bind_external(&physicalTensors[i].buffer,
        reinterpret_cast<NoodleData*>(physicalBuffers[i]),bufferSizes[i])){
      Serial.println("ERR_BUFFER_BIND");return false;
    }
  }
  for(size_t i=0;i<sizeof(tensors)/sizeof(*tensors);i++)tensors[i].storage=&physicalTensors[tensorBufferIds[i]];
`;
 const allocationReport=memoryPlan ? `  Serial.printf("BUFFER_REUSE logical=%u physical=%u PSRAM_BYTES=%u\\n",(unsigned)(sizeof(tensors)/sizeof(*tensors)),(unsigned)${memoryPlan.physicalCount},(unsigned)${memoryPlan.allocatedBytes});\n` : '';
 const main=`// Generated by Noodle Studio: ESP32-S3 N16R8, serial-input INT8 graph test.\n#include <Arduino.h>\n#include <esp_heap_caps.h>\n#include <esp_partition.h>\n#include <esp_spi_flash.h>\n#include <string.h>\n#include <stdint.h>\n#include <noodle_i8_nhwc.h>\n\nstatic constexpr uint32_t EXPECTED_BYTES=${cursor}u;\nstatic constexpr uint32_t EXPECTED_CRC=0x${crc32(payload).toString(16)}u;\nstatic constexpr uint32_t PARAM_NONE=0xffffffffu;\nstatic const char *names[]={${names}};\n${poolDefinitions}static NoodleI8NHWCTensor tensors[]={\n${specs}\n};\nstatic const NoodleI8NHWCOp ops[]={\n${descriptors}\n};\nstatic spi_flash_mmap_handle_t mappedHandle=0;\nstatic const uint8_t *flashData=nullptr;\nstatic bool initialized=false;
static constexpr size_t INPUT_INDEX=${project.storage.findIndex(x=>x.tensor_id===model.inputs[0])};
static constexpr size_t OUTPUT_INDEX=${project.storage.findIndex(x=>x.tensor_id===model.outputs[0])};
static constexpr size_t CHUNK_BYTES=32;
static uint32_t crc32(const uint8_t *p,size_t n){uint32_t c=0xffffffffu;for(size_t k=0;k<n;k++){c^=p[k];for(int i=0;i<8;i++)c=(c>>1)^((c&1u)?0xedb88320u:0u);}return ~c;}
static bool setupMemory(){
  if(!psramFound()){Serial.println("ERR_PSRAM_MISSING");return false;}
  Serial.printf("PSRAM total=%u free=%u\\n",(unsigned)ESP.getPsramSize(),(unsigned)ESP.getFreePsram());
  const auto *part=esp_partition_find_first(ESP_PARTITION_TYPE_DATA,(esp_partition_subtype_t)0x40,"model");
  if(!part||part->size<EXPECTED_BYTES){Serial.println("ERR_MODEL_PARTITION");return false;}
  const void *mapped=nullptr;
  if(esp_partition_mmap(part,0,EXPECTED_BYTES,SPI_FLASH_MMAP_DATA,&mapped,&mappedHandle)!=ESP_OK){Serial.println("ERR_FLASH_MMAP");return false;}
  flashData=(const uint8_t*)mapped;
  if(memcmp(flashData,"NDL8XIP",7)!=0 || flashData[7]!=0 || (uint32_t(flashData[8])|(uint32_t(flashData[9])<<8)|(uint32_t(flashData[10])<<16)|(uint32_t(flashData[11])<<24))!=EXPECTED_BYTES){Serial.println("ERR_MODEL_HEADER");return false;}
  if(crc32(flashData+16,EXPECTED_BYTES-16)!=EXPECTED_CRC){Serial.println("ERR_MODEL_CRC");return false;}
${initAlloc}${allocationReport}
  Serial.printf("TENSORS=%u OPS=%u PARAMS=%u FREE_PSRAM=%u\\n",(unsigned)(sizeof(tensors)/sizeof(*tensors)),(unsigned)(sizeof(ops)/sizeof(*ops)),(unsigned)EXPECTED_BYTES,(unsigned)ESP.getFreePsram());
  return true;
}
// Noodle Serial IMG v1: exact UINT8 bytes in model NHWC order, 32-byte K ACK.
// Full 32-byte chunks get ACK; an incomplete final chunk does not.
static bool receiveInput(){
  auto &input=tensors[INPUT_INDEX];
  uint8_t *inputData=noodle_i8_nhwc_data(input);
  if(!inputData)return false;
  Serial.println("RDYIMG");
  uint32_t offset=0;
  while(offset<input.bytes){
    const uint32_t count=(input.bytes-offset<CHUNK_BYTES)?input.bytes-offset:CHUNK_BYTES;
    uint32_t k=0;
    uint32_t last=millis();
    while(k<count){
      int c=Serial.read();
      if(c>=0){inputData[offset+k]=(uint8_t)c;++k;last=millis();}
      else if(millis()-last>15000){Serial.println("ERR_INPUT_TIMEOUT");return false;}
      else delay(1);
    }
    offset+=count;
    if(count==CHUNK_BYTES){Serial.write('K');Serial.flush();}
  }
  return true;
}
static void runModel(){
  const uint32_t start=millis();
  for(size_t k=0;k<sizeof(ops)/sizeof(*ops);k++){
    const uint32_t tick=millis();
    if(!noodle_i8_nhwc_execute(ops[k],tensors,sizeof(tensors)/sizeof(*tensors),flashData,EXPECTED_BYTES)){
      Serial.printf("ERR_INFERENCE_OP %u %s\\n",(unsigned)k,names[k]);Serial.println("READY");return;
    }
    Serial.printf("OP %u %-15s %lu ms\\n",(unsigned)k,names[k],(unsigned long)(millis()-tick));
  }
  auto &out=tensors[OUTPUT_INDEX];
  const uint8_t *outputData=noodle_i8_nhwc_data(out);
  if(!outputData){Serial.println("ERR_OUTPUT_BUFFER");return;}
  uint32_t best=0;for(uint32_t k=1;k<out.bytes;k++)if(outputData[k]>outputData[best])best=k;
  Serial.printf("PRED %u MS %lu\\n",(unsigned)best,(unsigned long)(millis()-start));
  Serial.print("OUTPUT_UINT8");for(uint32_t k=0;k<out.bytes;k++){Serial.print(' ');Serial.print((unsigned)outputData[k]);}Serial.println();
  Serial.println("READY");
}
void setup(){
  Serial.begin(115200);
  delay(1200);
  Serial.println("NOODLE_SQUEEZENET_INT8_N16R8");
  initialized=setupMemory();
  if(initialized)Serial.println("READY");else Serial.println("BOOT_FAILED");
}
void loop(){
  if(!initialized){delay(1000);return;}
  static char cmd[12];static uint8_t used=0;
  while(Serial.available()){
    const int c=Serial.read();
    if(c==13)continue;
    if(c=='\\n'){
      cmd[used]=0;used=0;
      if(strcmp(cmd,"HELLO")==0){Serial.println("READY");}
      else if(strcmp(cmd,"IMG")==0){if(receiveInput())runModel();else Serial.println("READY");}
      else {Serial.println("ERR_CMD");Serial.println("READY");}
    }else if(used<sizeof(cmd)-1){cmd[used++]=(char)c;}
    else{used=0;Serial.println("ERR_CMD_LENGTH");Serial.println("READY");}
  }
  delay(1);
}\n`;
 const ini=`[env:esp32-s3-n16r8]\nplatform = espressif32@6.12.0\nboard = esp32-s3-devkitc-1\nframework = arduino\nboard_build.flash_size = 16MB\nboard_upload.flash_size = 16MB\nboard_build.arduino.memory_type = qio_opi\nboard_build.flash_mode = qio\nboard_build.psram_type = opi\nboard_build.partitions = partitions_16mb_squeezenet.csv\nboard_upload.maximum_size = 7340032\nextra_scripts = pre:pio_flash_model.py\nbuild_flags =\n  -D BOARD_HAS_PSRAM\n  -D NOODLE_USE_INT8\n  -D NOODLE_USE_NONE\n  -D NOODLE_MAX_K=3\n  -D ARDUINO_USB_MODE=1\n  -D ARDUINO_USB_CDC_ON_BOOT=1\nmonitor_speed = 115200\nupload_protocol = esptool\n`;
 const optIni=ini.replace('framework = arduino\n','framework = arduino\nbuild_type = release\n')+'\n; Optional -O2 speed trial; the original release build remains the baseline.\n' +'[env:esp32-s3-n16r8-o2]\nextends = env:esp32-s3-n16r8\nbuild_type = release\nbuild_unflags = -Os\nbuild_flags =\n' +'  ${env:esp32-s3-n16r8.build_flags}\n  -O2\n';
  const pointwiseIni=optIni+"\n; Original scalar convolution regression fallback (1x1 Pointwise4 disabled).\n[env:esp32-s3-n16r8-scalar]\nextends = env:esp32-s3-n16r8\nbuild_flags =\n  ${env:esp32-s3-n16r8.build_flags}\n  -D NOODLE_S3_I8_POINTWISE_PIXELS=0\n";
 const optimizeReadme="# Compiler optimization comparison — ESP32-S3 N16R8\n\nThese Noodle Studio ESP32-S3 firmware exports include two PlatformIO environments, both built in release mode:\n\n- **Baseline:** `esp32-s3-n16r8` (INT8) or `esp32-s3-n16r8-fp32` (FP32). PlatformIO's default compiler optimization is preserved (normally `-Os` for Arduino ESP32).\n- **Speed experiment:** the matching environment with `-o2` suffix. Explicitly removes `-Os` and applies `-O2` for code compiled from source (including Noodle kernels).\n\n### Benchmark commands\n\nINT8:\n\n```sh\npio run -e esp32-s3-n16r8-o2 -t upload\npython3 test/send_tensor.py --port /dev/ttyACM0 --image test/dog.jpg\n```\n\nFP32:\n\n```sh\npio run -e esp32-s3-n16r8-fp32-o2 -t upload\npython3 test/send_tensor.py --port /dev/ttyACM0 --image test/dog.jpg\n```\n\nUse the **USB OTG** port for native USB CDC serial on the Freenove-style ESP32-S3 board. Close the monitor before the sender. Uploading still writes the XIP parameter image at the normal dedicated partition offset.\n\n### Scientific comparison\n\nRecord median and variability over repeated identical image inferences, code/flash size, free PSRAM, and output checksums. Changing optimization level may change floating-point rounding; compare numeric output to TFLite rather than only class predictions.\n\n`-Os` is optimized for **code size**. `-O2` targets **execution speed** but can produce **larger code and more instruction-cache pressure**. Neither is guaranteed to be faster on the ESP32-S3. Precompiled Arduino framework binaries may not inherit all of the application optimization settings. Do not use `-Ofast` until numerical equivalence has been established.\n\nThe original baseline environments remain available and unchanged except for explicit `build_type = release`.\n";
 const csv=`# Name, Type, SubType, Offset, Size, Flags\nnvs,data,nvs,0x9000,0x5000,\notadata,data,ota,0xe000,0x2000,\napp0,app,ota_0,0x10000,0x700000,\nmodel,data,0x40,0x800000,0x200000,\nffat,data,fat,0xA00000,0x600000,\n`;
 const uploadScript=`# Added by Noodle Studio. PlatformIO includes the XIP image in its normal upload.
Import("env")
from os.path import join
env.Append(FLASH_EXTRA_IMAGES=[("0x800000", join(env.subst("$PROJECT_DIR"), "model", "params.bin"))])
`;
 const reader=`#!/usr/bin/env python3\n# Optional reference: pip install tensorflow (or tflite-runtime).\nimport argparse, numpy as np\np=argparse.ArgumentParser();p.add_argument('--model',default='model/source.tflite');a=p.parse_args()\ntry:\n import tflite_runtime.interpreter as rt;Interpreter=rt.Interpreter\nexcept ImportError:\n import tensorflow as tf;Interpreter=tf.lite.Interpreter\nx=np.fromfunction(lambda y,x,c: np.where(c==0,x*255//223,np.where(c==1,y*255//223,np.bitwise_xor(x.astype(np.int32),y.astype(np.int32))%256)),(224,224,3),dtype=int).astype(np.uint8)[None]\ninterpreter=Interpreter(model_path=a.model);interpreter.allocate_tensors();i=interpreter.get_input_details()[0];o=interpreter.get_output_details()[0];interpreter.set_tensor(i['index'],x);interpreter.invoke();out=interpreter.get_tensor(o['index']).ravel();print('PRED',int(out.argmax()),'RAW_UINT8',int(out.max()));print('OUTPUT_UINT8',' '.join(map(str,out)))\n`;
 const guide=`# Noodle SqueezeNet 1.1 — ESP32-S3 N16R8, INT8 serial-input test\n\nThis is the first **serial-input graph-execution** stage. It does **not** integrate the camera/TFT yet. Uses the original Qualcomm W8A8 TFLite, and generates all 43 operators. It is not hardware-validated.\n\n## Memory\n- PSRAM: ${memoryPlan ? memoryPlan.allocatedBytes+" bytes in "+memoryPlan.physicalCount+" reusable physical buffers ("+tensors.length+" logical tensors, "+memoryPlan.logicalBytes+" logical bytes)" : tensors.reduce((a,t)=>a+t.bytes,0)+" bytes, "+tensors.length+" separate activations"}; sequential ping-pong OFF.\n- Raw XIP partition at 0x800000: model/params.bin (${cursor} bytes); quantized weights/biases/multipliers/shifts. No FFat weight reads.\n- Internal SRAM: stacks and runtime scratch. The FFat partition is reserved but unused for model parameters.\n\n## Upload and serial test\n1. Connect the ESP32-S3 native USB port (USB-Serial/JTAG) and select its port (e.g. \`/dev/ttyACM0\`).\n2. Compile and upload both firmware and XIP parameter partition in one PlatformIO command: \`pio run -e esp32-s3-n16r8 -t upload --upload-port /dev/ttyACM0\`.\n3. Install sender dependencies: \`python3 -m pip install pyserial Pillow\`.\n4. Send the exact deterministic gradient used by \`test/reference.py\`: \`python3 test/send_tensor.py --port /dev/ttyACM0 --pattern gradient\`.\n5. Send a real photograph: \`python3 test/send_tensor.py --port /dev/ttyACM0 --image photo.jpg\`.\n6. Send a raw UINT8 NHWC tensor: \`python3 test/send_tensor.py --port /dev/ttyACM0 --input-file image.bin\`.\n
Upload is still one command; the input test uses a second command. No firmware code modification or manual model flash is needed. Do not run a serial monitor at the same time as the sender.\n\nDo not flash any other ESP32 board/partitioning at this offset. Reflashing the firmware can erase parts of the model, depending on flashing options. The bootloader checks image size and CRC to catch invalid contents.\n\n## Reference\nThe generic sender uses model-derived input shape and quantization; it sends raw RGB bytes (no host-side /255 normalization). To compare with the same input, run \`python3 test/send_tensor.py --dry-run --pattern gradient --save-payload gradient.bin\`.\n\nRun \`python3 test/reference.py --model model/source.tflite\` using TensorFlow or tflite-runtime. The default sender gradient and reference use the same deterministic RGB gradient. Compare all 1000 returned UINT8 logits, not just argmax.\n\n## Limitations\n- Portable Noodle INT8 NHWC kernels retain the Pointwise4 specialization (default 4) and scalar fallback; target-specific memory mapping remains in the generated application.\n- ${memoryPlan ? "Branched activation buffers use a compile-time lifetime plan and are represented by shared physical `NoodleTensor`/externally-bound `NoodleBuffer` objects; manual resizing is unnecessary. Use `settings.reuse_branch_buffers=false` in project JSON for legacy one-buffer-per-tensor regression." : "No branch-buffer reuse in this legacy configuration."} Camera/TFT not included in this serial-input export.\n- Some standalone SUB/MUL/MEAN and QUANTIZE operations use float-assisted rescaling; exact TFLite byte-for-byte agreement must be measured before claiming correctness.\n- Board build and actual inference require PlatformIO and physical hardware. No claim of successful hardware execution until tested.\n`;
 const sender=root.NOODLE_S3_SERIAL_SENDER || (typeof require!=='undefined'?require('./s3_serial_sender_bundle.js'):null);
 check(sender,'ESP32-S3 generic serial sender missing');
 const inputTensor=model.tensors[model.inputs[0]], outputTensor=model.tensors[model.outputs[0]];
 const inputSpec={protocol:'noodle-serial-img-v1',shape:inputTensor.shape,dtype:inputTensor.dtype,transport_dtype:'uint8',bytes:nbytes(inputTensor.shape),layout:'NHWC',scale:tquant(inputTensor).scale,zero_point:tquant(inputTensor).zp,baud:115200,chunk_bytes:32,output_elements:nbytes(outputTensor.shape)};
 const outputSpec={shape:outputTensor.shape,dtype:outputTensor.dtype,scale:tquant(outputTensor).scale,zero_point:tquant(outputTensor).zp};
 const lib=root.NOODLE_LIB_FILES || (typeof require!=='undefined'?require('./noodle_bundle.js'):null);
 check(lib,'Noodle library bundle unavailable');
 const files={...lib,'src/main.cpp':main,'platformio.ini':pointwiseIni,'README_COMPILER_OPTIMIZATION.md':optimizeReadme,'partitions_16mb_squeezenet.csv':csv,
  'model/params.bin':modelBin,'model/source.tflite':bin,'model/project.json':JSON.stringify(project,null,2)+'\n',
  'model/manifest.json':JSON.stringify({sha256:project.source.model_sha256,params_bytes:cursor,crc32:crc32(payload),partition_offset:'0x800000',weights_layout:layout,operator_api:'noodle_i8_nhwc_execute',tensor_storage:'NoodleTensor/NoodleBuffer external binding',...(memoryPlan?{memory_strategy:'branch-liveness',logical_activation_bytes:memoryPlan.logicalBytes,physical_activation_bytes:memoryPlan.allocatedBytes,physical_buffer_count:memoryPlan.physicalCount,maximum_live_tensors:memoryPlan.maxLiveTensors,minimum_physical_buffers:memoryPlan.minimumPhysicalBuffers,peak_live_bytes:memoryPlan.peakLiveBytes}:{})},null,2)+'\n',
  ...(memoryPlan?{
    'model/buffer_plan.json':JSON.stringify(memoryPlan,null,2)+'\n',
    'model/lifetime_by_day.csv':['day,operator_index,live_tensor_count,live_bytes,live_tensor_ids',
      ...memoryPlan.timeline.map(d=>[d.day,d.operatorIndex??'',d.liveTensorCount,d.liveBytes,
        '"'+d.liveTensorIds.join(';')+'"'].join(','))].join('\n')+'\n',
    'model/tensor_lifetimes.csv':['tensor_id,name,backend,born_day,die_day,bytes,physical_buffer_id',
      ...memoryPlan.lifetimes.map(t=>[t.tensorId, '"'+String(t.name).replace(/"/g,'""')+'"',
        '"'+String(t.backend).replace(/"/g,'""')+'"',t.bornDay,t.dieDay,t.bytes,t.bufferId].join(','))].join('\n')+'\n'
  }:{}),
  'test/reference.py':reader,'test/send_tensor.py':sender,'test/input_spec.json':JSON.stringify(inputSpec,null,2)+'\n','test/output_spec.json':JSON.stringify(outputSpec,null,2)+'\n','pio_flash_model.py':uploadScript,'README_ESP32S3_SQUEEZENET.md':guide,
  }; // Legacy NHWC operator files are never included in the canonical firmware ZIP.
 delete files['lib/Noodle/src/noodle_s3_nhwc_i8.h'];
 delete files['lib/Noodle/src/noodle_s3_nhwc_i8.cpp'];
 const generated={files,summary:{layers:ops.map(x=>x.name),sdLayers:0,sequentialPingPong:false,activationBytes:tensors.reduce((a,t)=>a+t.bytes,0),allocatedBytes:memoryPlan?.allocatedBytes??tensors.reduce((a,t)=>a+t.bytes,0),physicalBuffers:memoryPlan?.physicalCount??tensors.length,peakLiveBytes:memoryPlan?.peakLiveBytes??null,maxLiveTensors:memoryPlan?.maxLiveTensors??null,minimumPhysicalBuffers:memoryPlan?.minimumPhysicalBuffers??null,maxLiveDays:memoryPlan?.maxLiveDays??null,bufferReuse:!!memoryPlan,xipBytes:cursor,operators:ops.length}};
 { // Canonical SqueezeNet INT8 export: native Noodle CHW; ignore retired experimental_chw setting
   const adapter=root.NoodleS3CHW||(typeof require!=='undefined'?require('./esp32s3_chw_transform.js'):null);
   check(adapter?.transform,'CHW converter missing');
   return adapter.transform(generated,project);
 }
 return generated;
}
const API={generate}; if(typeof module!=='undefined' && module.exports)module.exports=API;
root.NoodleESP32S3=API;
})(typeof globalThis!=='undefined'?globalThis:this);
