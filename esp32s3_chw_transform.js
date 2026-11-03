/* Canonical CHW export transformation for SqueezeNet W8A8.
 * The former NHWC generator is only an internal metadata/serialization intermediate.
 * TFLite OHWI conv weights are reordered to native Noodle OIHW.
 * Input protocol remains standard NHWC bytes; the generated receiver
 * transposes once, directly into the internal CHW activation tensor.
 */
'use strict';
(function(root){
const T=root.NoodleTFLite || (typeof require!=='undefined'?require('./tflite_import.js'):null);
function check(ok,why){if(!ok)throw Error('CHW export: '+why);}
function crc32(data) {let c=0xffffffff;for(const b of data){c^=b;for(let i=0;i<8;i++)c=(c>>>1)^((c&1)?0xedb88320:0);}return (~c)>>>0;}
function transform(generated,project){
  check(T && generated?.files && project?.source?.model_base64,'Missing TFLite project or generated baseline');
  const model=T.inspectTFLite(T.base64Decode(project.source.model_base64));
  const files={...generated.files};
  const original=new Uint8Array(files['model/params.bin']);
  const parameters=original.slice();
  const manifest=JSON.parse(files['model/manifest.json']);
  let changed=0;
  for(const op of model.operators){
    if(op.op!=='CONV_2D')continue;
    const tid=op.parameter_tensors[0],t=model.tensors[tid];
    const [O,KH,KW,I]=t.shape;
    check(t.shape.length===4 && O>0 && I>0 && KH>0 && KW>0,'Unsupported Conv2D tensor');
    const name='tflite_tensor_'+tid;
    const item=manifest.weights_layout.find(x=>x.name===name);
    check(item && item.bytes===O*KH*KW*I,'Missing/invalid XIP weight region: '+name);
    const offset=item.offset,src=original.subarray(offset,offset+item.bytes);
    const dst=parameters.subarray(offset,offset+item.bytes);
    // TFLite [out][kernel_y][kernel_x][input] -> Noodle [out][input][kernel_y][kernel_x]
    for(let o=0;o<O;o++)for(let i=0;i<I;i++)for(let y=0;y<KH;y++)for(let x=0;x<KW;x++)
      dst[((o*I+i)*KH+y)*KW+x]=src[((o*KH+y)*KW+x)*I+i];
    item.tensor_format='OIHW';changed++;
  }
  check(changed>0,'No convolution weights found');
  const crc=crc32(parameters.subarray(16));
  new DataView(parameters.buffer,parameters.byteOffset,parameters.byteLength).setUint32(12,crc,true);
  files['model/params.bin']=parameters;
  manifest.crc32=crc;manifest.operator_api='noodle_i8_graph_execute';
  manifest.activation_layout='CHW';manifest.model_source_layout='NHWC';
  manifest.weights_format='OIHW (Noodle native CHW)';
  files['model/manifest.json']=JSON.stringify(manifest,null,2)+'\n';
  let main=files['src/main.cpp'];
  check(main.includes('noodle_i8_nhwc_execute')&&main.includes('inputData[offset+k]=(uint8_t)c;'), 'Unexpected v0.6.19 main template');
  main=main.replaceAll('noodle_i8_nhwc','noodle_i8_graph').replaceAll('NoodleI8NHWC','NoodleI8Graph');
  main=main.replace('#include <noodle_i8_graph.h>','#include <noodle_int8.h>');
  main=main.replace('inputData[offset+k]=(uint8_t)c;',
    'const uint32_t position=offset+k; const uint32_t pixel=position/input.c; const uint32_t channel=position%input.c; inputData[channel*uint32_t(input.h)*input.w+pixel]=(uint8_t)c;');
  main=main.replace(/EXPECTED_CRC=0x[0-9a-f]+u/,`EXPECTED_CRC=0x${crc.toString(16)}u`);
  main=main.replace('// Noodle Serial IMG v1: exact UINT8 bytes in model NHWC order, 32-byte K ACK.',
    '// Noodle Serial IMG v1: NHWC UINT8 wire input; one-pass CHW scatter into NoodleTensor.');
  main=main.replace('NOODLE_SQUEEZENET_INT8_N16R8','NOODLE_SQUEEZENET_INT8_CHW');
  check(main.includes('noodle_i8_graph_execute')&&!main.includes('noodle_i8_nhwc_execute'),'CHW main conversion failed');
  files['src/main.cpp']=main;
  // Noodle graph entrypoint and Pointwise4 are integrated into noodle_int8_{api,layers}.
  // The unified Noodle INT8 API is the only exported execution path.
  for (const name of ['noodle_int8.h','noodle_int8_api.cpp','noodle_int8_layers.cpp']) {
    const data=(typeof require!=='undefined'?require('fs').readFileSync(__dirname+'/lib/Noodle/src/'+name,'utf8'):root.NOODLE_LIB_FILES['lib/Noodle/src/'+name]);
    check(data,'Missing unified Noodle INT8 source '+name);
    files['lib/Noodle/src/'+name]=data;
  }
  delete files['lib/Noodle/src/noodle_i8_nhwc.h'];
  delete files['lib/Noodle/src/noodle_i8_nhwc.cpp'];
  const inputSpec=JSON.parse(files['test/input_spec.json']);
  check(inputSpec.layout==='NHWC','Input protocol changed unexpectedly');
  inputSpec.internal_activation_layout='CHW';
  files['test/input_spec.json']=JSON.stringify(inputSpec,null,2)+'\n';
  // Production preset: one normal firmware environment, no temporary
  // Pointwise8/16, scalar, direct/tiled benchmarking environments.
  // Other Noodle storage and MCU backends retain their original build policy.
  const ini=files['platformio.ini'];
  const optionalBuilds=ini.indexOf('\n; Optional -O2 speed trial');
  const baseIni=optionalBuilds>=0?ini.slice(0,optionalBuilds):ini;
  const pointwiseToken='  -D NOODLE_USE_INT8';
  check(baseIni.includes(pointwiseToken),'Unexpected S3 build flags');
  files['platformio.ini']=baseIni.replaceAll('NOODLE_S3_I8_POINTWISE_PIXELS','NOODLE_INT8_POINTWISE_PIXELS')
    .replace(pointwiseToken,
      '  -D NOODLE_INT8_POINTWISE_PIXELS=4\n'
      +'  -D NOODLE_INT8_CONV_IMPL=1\n'
      +'  -D NOODLE_INT8_CONV3X3_ACC32=1\n'
      +pointwiseToken).trimEnd()+'\n';
  check((files['platformio.ini'].match(/NOODLE_INT8_POINTWISE_PIXELS=/g)||[]).length===1,
    'Pointwise width must be defined exactly once');
  files['README_ESP32S3_SQUEEZENET.md']=files['README_ESP32S3_SQUEEZENET.md']
    .replace('Portable Noodle INT8 NHWC kernels retain the Pointwise4 specialization', 'Unified Noodle INT8 CHW kernels retain the Pointwise4 specialization')
    + '\n## Native Noodle CHW execution\n'
    +'Unified CHW is the standard SqueezeNet INT8 execution path. '
    +'Codegen reorders CONV_2D weights from OHWI to OIHW and updates the XIP CRC. '
    +'Incoming RGB bytes use NHWC wire order and are scattered into CHW activations without an extra staging tensor. '
    +'The three reusable physical NoodleBuffers are managed by the birth/death branch planner.\n';
  files['README_CHW_UNIFIED.md']='# Unified Noodle INT8 CHW\n\n'
    +'CHW is automatic; no checkbox or project setting is required. '
    +'Old `settings.experimental_chw` values are ignored.\n\n'
    +'- CONV_2D weights are converted OHWI to OIHW. Flash firmware and matching params.bin together.\n'
    +'- Activation tensors use CHW; transport remains raw RGB UINT8 in NHWC wire order.\n'
    +'- Existing Noodle INT8 convolution, pooling and concatenation, with Pointwise4 in noodle_int8_layers.cpp.\n'
    +'- No dedicated NHWC INT8 operator files are emitted.\n';
  files['README_CONV_PRODUCTION.md'] =
    '# Noodle INT8 CHW production convolution preset (R15 v0.6.27)\n\n'
    +'- Build with `pio run -e esp32-s3-n16r8 -t upload`.\n'
    +'- Native CHW Conv2D remains one Noodle operator.\n'
    +'- 3x3 uses tiled convolution with checked INT32 accumulation and an automatic INT64 fallback when the conservative bound fails.\n'
    +'- 1x1 uses the hardware-tested Pointwise4 implementation.\n'
    +'- Generic convolution and alternate parameter storage backends remain supported.\n'
    +'- Incoming UART image bytes are NHWC; internal activations use CHW.\n'
    +'- Parameters and the three-buffer birth/death planner are unchanged.\n'
    +'\nExpected `test/bus.jpeg` output: class 779, CRC32 `a046a893`.\n'
    +'Measured hardware reference: 33.789 seconds (ESP32-S3 N16R8, single input).\n'
    +'Recompile and verify after changing the library or model.\n';
  delete files['README_CONV3X3_ACC32_EXPERIMENT.md'];
  delete files['README_CHW_EXPERIMENT.md'];
  const summary={...generated.summary,unifiedCHW:true,convertedConvolutions:changed,xipBytes:parameters.length};
  return {...generated,files,summary};
}
const API={transform}; if(typeof module!=='undefined'&&module.exports)module.exports=API;
root.NoodleS3CHW=API;
})(typeof globalThis!=='undefined'?globalThis:this);
