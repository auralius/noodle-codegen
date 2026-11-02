/* Noodle Studio R15 GUI prototype, dependency-free and entirely local. */
'use strict';

const NS_FORMAT = 'noodle-studio-r15/v0.5.1';
const IDENT = /^[A-Za-z_][A-Za-z_0-9]*$/;
const BOARD = {
  uno: {
    label: 'Arduino Uno / Nano (ATmega328P)',
    storage: ['SRAM', 'SD'],
    params: ['PROGMEM', 'SRAM', 'SD'],
    tags: ['2 KiB SRAM', '32 KiB Flash', 'Near PROGMEM', 'SD optional']
  },
  esp32s3: {
    label: 'ESP32-S3 N16R8',
    storage: ['SRAM', 'PSRAM', 'FFat', 'SD'],
    params: ['XIP', 'SRAM', 'PSRAM', 'FFat', 'SD'],
    tags: ['8 MiB PSRAM', '16 MiB Flash', 'FFat partition', 'XIP']
  }
};
const OPERATIONS = ['FCN', 'Conv2D', 'DepthwiseConv2D', 'PointwiseConv2D', 'MaxPool', 'AvgPool', 'Concat', 'Add', 'ReLU', 'HybridBlock', 'GAP', 'Reshape', 'Softmax', 'SOFTMAX', 'FULLY_CONNECTED', 'CONV_2D', 'MAX_POOL_2D', 'AVERAGE_POOL_2D', 'CONCATENATION', 'MEAN', 'RESHAPE', 'QUANTIZE', 'SUB', 'MUL'];
const NONPARAM = new Set(['MaxPool','AvgPool','Concat','Add','ReLU','GAP','Reshape','MAX_POOL_2D','AVERAGE_POOL_2D','CONCATENATION','MEAN','RESHAPE','Softmax','SOFTMAX','QUANTIZE']);
const RESERVED = new Set(['destroy', 'close']);
const TFLITE = (typeof NoodleTFLite !== 'undefined') ? NoodleTFLite : null;
const SEQUENTIAL = (typeof NoodleSequentialNetwork !== 'undefined') ? NoodleSequentialNetwork : null;
let state;
const $ = id => document.getElementById(id);
const copy = x => JSON.parse(JSON.stringify(x));
const textEscape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fileBackend = name => ['SD','FFat'].includes(name);
const prettyShape = x => typeof x === 'string' ? x : (x || []).join(',');

function sampleUno() {
  return {
    format:NS_FORMAT,
    target:'uno',
    source:{name:'Small INT8 FCN example', type:'manual'},
    storage:[
      {name:'buffB',backend:'SRAM',path:'',shape:'1,16'},
      {name:'buffA',backend:'SRAM',path:'',shape:''},
      {name:'out1',backend:'SD',path:'/out1.bin',shape:''}
    ],
    layers:[
      {name:'layer1',kind:'FCN',params:'PROGMEM',output_shape:'1,8',notes:'INT8 demo, actual weights not imported'},
      {name:'layer2',kind:'FCN',params:'SD',output_shape:'1,4',notes:'INT8 demo, actual weights not imported'}
    ],
    topology:'buffA = layer1(buffB);\nout1 = layer2(buffA);\n\n// destroy(buffA);  // optional explicit lifetime'
  };
}
function sampleFire() {
  return {
    format:NS_FORMAT,target:'esp32s3',source:{name:'SqueezeNet Fire module example',type:'manual'},
    storage:['x','s','e1','e3','y'].map(name=>({name,backend:'PSRAM',path:'',shape:name==='x'?'1,55,55,128':''})),
    layers:[
      {name:'squeeze',kind:'Conv2D',params:'XIP',output_shape:'1,55,55,16'},
      {name:'expand1',kind:'Conv2D',params:'XIP',output_shape:'1,55,55,64'},
      {name:'expand3',kind:'Conv2D',params:'XIP',output_shape:'1,55,55,64'},
      {name:'concat',kind:'Concat',params:'-',output_shape:'1,55,55,128'}
    ],
    topology:'s = squeeze(x);\ne1 = expand1(s);\ne3 = expand3(s);\ny = concat(e1, e3);\n\ndestroy(s);\ndestroy(e1);\ndestroy(e3);'
  };
}

function fromR14Graph(graph) {
  if (!graph || !Array.isArray(graph.steps) || !graph.steps.length) throw new Error('Not an R14 generated_graph.json (missing steps).');
  const tensorToBuffer = new Map();
  const foundStorage = new Map();
  const layers = [];
  const topology = [];
  for (const step of graph.steps) {
    const n = String(step.name);
    if (!IDENT.test(n)) throw new Error('Invalid R14 layer name: ' + n);
    const inputs = (step.input_tensor_ids||[]).map(id => {
      if (tensorToBuffer.has(id)) return tensorToBuffer.get(id);
      // Input to graph, normally TFLite tensor 58 on R14.
      const unbound = (id === 58 ? 'X' : 'T'+id);
      if (!foundStorage.has(unbound)) foundStorage.set(unbound,{name:unbound,backend:'PSRAM',path:'',shape:id === 58?'1,224,224,3':''});
      tensorToBuffer.set(id,unbound);
      return unbound;
    });
    const output = String(step.buffer || 'T'+step.output_tensor_id);
    if (!IDENT.test(output)) throw new Error('Invalid R14 buffer name: '+output);
    if (!foundStorage.has(output)) foundStorage.set(output,{name:output,backend:'PSRAM',path:'',shape:''});
    const kind = String(step.kind||'Unknown');
    layers.push({name:n,kind,params:NONPARAM.has(kind)?'-':'XIP',output_shape:prettyShape(step.output_shape),notes:'Imported from R14 op '+step.op_index});
    topology.push(`${output} = ${n}(${inputs.join(', ')});`);
    tensorToBuffer.set(step.output_tensor_id,output);
  }
  return {format:NS_FORMAT,target:'esp32s3',source:{name:'R14 generated_graph.json',type:'R14 imported',model_sha256:graph.model_sha256||''},
    storage:[...foundStorage.values()],layers,topology:topology.join('\n')};
}

function normalizeProject(p) {
  if (!p || !Array.isArray(p.storage) || !Array.isArray(p.layers) || typeof p.topology !== 'string' || !(p.target in BOARD))
    throw new Error('Unsupported JSON. Need a Noodle Studio project with target, storage, layers and topology.');
  if(p.format && !['noodle-studio-r15/v0.1','noodle-studio-r15/v0.2','noodle-studio-r15/v0.3','noodle-studio-r15/v0.4','noodle-studio-r15/v0.5',NS_FORMAT].includes(p.format))
    throw new Error('Unsupported project format '+p.format);
  return {format:NS_FORMAT,target:p.target,...(p.settings?{settings:{...p.settings,sd_cs_pin:Number(p.settings.sd_cs_pin??10),apply_pingpong_sequential:p.settings.apply_pingpong_sequential===true}}:{}),source:copy(p.source||{name:'Imported project',type:'manual'}),
    storage:p.storage.map(x=>({name:String(x.name||''),backend:String(x.backend||''),path:String(x.path||''),shape:prettyShape(x.shape||''),
      ...(x.tensor_id===undefined?{}:{tensor_id:x.tensor_id}),...(x.dtype?{dtype:x.dtype}:{})})),
    layers:p.layers.map(x=>({name:String(x.name||''),kind:String(x.kind||''),params:String(x.params??''),output_shape:prettyShape(x.output_shape||''),notes:String(x.notes||''),
      ...(x.input_shape?{input_shape:prettyShape(x.input_shape)}:{}),...(x.op_index===undefined?{}:{op_index:x.op_index}),
      ...(x.parameter_tensors?{parameter_tensors:copy(x.parameter_tensors)}:{}),...(x.op_options?{op_options:copy(x.op_options)}:{})})),
    topology:p.topology};
}
function suggestLabel(op,counts){
  const n=(counts[op]=(counts[op]||0)+1);
  const stems={FULLY_CONNECTED:'fc',SOFTMAX:'softmax',CONV_2D:'conv',DEPTHWISE_CONV_2D:'dwconv',MAX_POOL_2D:'pool',AVERAGE_POOL_2D:'avgpool',RESHAPE:'reshape',ADD:'add',CONCATENATION:'concat',MEAN:'mean'};
  return (stems[op]||'op'+op.toLowerCase().replace(/[^a-z0-9]+/g,'_'))+n;
}
function fromTFLiteModel(model, filename, bytes, hash) {
  if(!model.inputs.length||!model.outputs.length||!model.operators.length) throw new Error('TFLite has no usable input, output, or operators.');
  const dataInputs=model.inputs.map(i=>model.tensors[i]);
  if(dataInputs.some(t=>!['INT8','UINT8','FLOAT32'].includes(t.dtype)))throw new Error('R15 imports INT8, UINT8, or FLOAT32 model inputs for inspection. This model has input types: '+dataInputs.map(t=>t.dtype).join(', '));
  const opTypes=model.operators.map(o=>o.op);
  const isFCOps=opTypes.every(n=>['FULLY_CONNECTED','SOFTMAX','RESHAPE','RELU'].includes(n));
  const target=isFCOps && dataInputs.every(t=>t.dtype==='INT8')?'uno':'esp32s3';
  const backend=target==='uno'?'SRAM':'PSRAM',paramBackend=target==='uno'?'PROGMEM':'XIP';
  const nameMap=new Map(),storage=[],layers=[],calls=[],counts={};
  function addTensor(id,hint){
    if(nameMap.has(id))return nameMap.get(id);
    const t=model.tensors[id];if(!t)throw new Error('Invalid activation tensor '+id);
    let name=hint.replace(/[^a-zA-Z0-9_]/g,'_');if(!IDENT.test(name))name='tensor_'+id;
    let base=name,k=2;while(storage.some(x=>x.name===name)){name=base+'_'+k;k++;}
    nameMap.set(id,name);storage.push({name,backend,path:'',shape:t.shape.join(','),tensor_id:id,dtype:t.dtype});return name;
  }
  model.inputs.forEach((id,i)=>{const shape=model.tensors[id].shape;addTensor(id,i===0?'input'+(shape.at(-1)||''):'input'+(i+1));});
  for(const op of model.operators){
    const name=suggestLabel(op.op,counts);
    const active=op.inputs.filter(i=>i>=0&&!op.parameter_tensors.includes(i));
    if(op.op==='RESHAPE'&&active.length>1){ /* Constant reshape target is metadata, not an activation input */ }
    if(!op.outputs.length||op.outputs.length!==1)throw new Error('Operator '+op.id+' has unsupported multiple outputs');
    const tail=model.tensors[op.outputs[0]].shape.at(-1)||op.id;
    const nextOp=model.operators[op.id+1];
    const dstHint=op.op==='SOFTMAX'?'probabilities'+tail:op.op==='FULLY_CONNECTED'?(nextOp?.op==='SOFTMAX'?'logits'+tail:'hidden'+tail):model.outputs.includes(op.outputs[0])?'output'+tail:'tmp'+(op.id+1);
    const dst=addTensor(op.outputs[0],dstHint);
    const src=active.map(i=>addTensor(i,'input_'+i));
    const p=op.parameter_tensors.map(i=>{const t=model.tensors[i];return {id:i,shape:t.shape,dtype:t.dtype,bytes:t.parameter_bytes,quantization:copy(t.quantization)};});
    layers.push({name,kind:op.op==='FULLY_CONNECTED'?'FCN':op.op==='SOFTMAX'?'Softmax':op.op,params:NONPARAM.has(op.op)?'-':op.parameter_tensors.length?paramBackend:'-',
      ...(src.length===1?{input_shape:model.tensors[active[0]].shape.join(',')}:{}),output_shape:model.tensors[op.outputs[0]].shape.join(','),
      op_index:op.id,parameter_tensors:p,op_options:copy(op.config),notes:'Imported TFLite '+op.op+' v'+op.version+(op.config.activation?', '+op.config.activation:'')});
    calls.push(`${dst} = ${name}(${src.join(', ')});`);
  }
  const otherDtypes=model.tensors.filter(x=>x.parameter_bytes>0&& !['INT8','INT32'].includes(x.dtype));
  // FP32 constants in non-FCN operations may exist; importer preserves them without pretending they are INT8 weights.
  return {format:NS_FORMAT,target,source:{name:filename,type:'tflite',file:filename,model_sha256:hash,
      model_size_bytes:bytes.length,model_base64:TFLITE.base64Encode(bytes),
      schema_version:model.schema_version,inputs:model.inputs,outputs:model.outputs,
      tensors:copy(model.tensors),operators:copy(model.operators),warnings:[
        ...(otherDtypes.length?['Contains '+otherDtypes.length+' non-INT8/INT32 constant tensors; preserved in the embedded TFLite model.']:[])
      ]},
    storage,layers,topology:calls.join('\n')};
}
// Firmware export is deliberately narrower than TFLite import/inspection.
// Do not infer a supported exporter from the selected board or input dtype alone.
const LENET5_INT8_SHA='0c89db896852c0d69eb7be4e15fdd3981015247f0ed57ca666d8feb6e21c5073';
const S3_INT8_OPS=new Set(['CONV_2D','MAX_POOL_2D','CONCATENATION','QUANTIZE','SUB','MUL','MEAN','RESHAPE']);
const S3_FP32_OPS=new Set(['SUB','MUL','CONV_2D','MAX_POOL_2D','AVERAGE_POOL_2D','CONCATENATION','MEAN','RESHAPE','FULLY_CONNECTED','SOFTMAX']);
const UNO_FCN_OPS=new Set(['FULLY_CONNECTED','SOFTMAX']);
function firmwareCompatibility(project){
  const src=project?.source||{};
  const no=(reason,detail='')=>({supported:false,backend:null,reason,detail});
  if(src.type!=='tflite'||!src.model_base64) return no('Firmware export requires a TFLite model with embedded parameters. The example/topology preview is not executable firmware.');
  const ops=src.operators||[],tensors=src.tensors||[],inputs=src.inputs||[],outputs=src.outputs||[];
  if(inputs.length!==1||outputs.length!==1) return no('Current exporters require exactly one model input and one output.');
  const input=tensors[inputs[0]],output=tensors[outputs[0]];
  if(!input||!output||!ops.length) return no('Missing TFLite tensor or operator metadata. Re-import the original model.');
  const shape=(input.shape||[]).join(',');
  const signature=`Input: ${input.dtype} [${shape}], output: ${output.dtype}; ${ops.length} operators.`;
  const unsupported=(allow)=>{
    const counts={};for(const o of ops){if(!allow.has(o.op))counts[o.op]=(counts[o.op]||0)+1;}
    return Object.entries(counts).map(([kind,n])=>kind+' × '+n).join(', ');
  };
  if(project.target==='uno'){
    if(src.model_sha256===LENET5_INT8_SHA && input.dtype==='INT8')
      return {supported:true,backend:'uno_lenet5',reason:'Supported: LeNet-5 INT8 Uno/SD exporter.',detail:signature};
    if(input.dtype==='INT8' && ops.every(o=>UNO_FCN_OPS.has(o.op)) && ops.some(o=>o.op==='FULLY_CONNECTED'))
      return {supported:true,backend:'uno_fcn',reason:'FCN-only Uno exporter available; quantization, storage, and topology are checked during generation.',detail:signature};
    return no('Uno firmware currently supports the validated LeNet-5 INT8 model or FCN-only INT8 graphs.',signature+(unsupported(UNO_FCN_OPS)?' Unsupported on the generic Uno FCN path: '+unsupported(UNO_FCN_OPS)+'.':''));
  }
  if(project.target==='esp32s3'){
    const standardShape=shape==='1,224,224,3';
    if(standardShape&&input.dtype==='UINT8'&&output.dtype==='UINT8'&&ops.length===43&&ops.every(o=>S3_INT8_OPS.has(o.op)))
      return {supported:true,backend:'s3_int8',reason:'Compatible with the ESP32-S3 SqueezeNet-class INT8 exporter; graph/storage checks apply.',detail:signature};
    if(standardShape&&input.dtype==='FLOAT32'&&output.dtype==='FLOAT32'&&ops.length===41&&ops.every(o=>S3_FP32_OPS.has(o.op)))
      return {supported:true,backend:'s3_fp32',reason:'Compatible with the ESP32-S3 SqueezeNet-class FP32 exporter; graph/storage checks apply.',detail:signature};
    if(input.dtype==='FLOAT32'&&output.dtype==='FLOAT32'&&ops.every(o=>S3_FP32_OPS.has(o.op)))
      return {supported:true,backend:'s3_fp32_generic',reason:'Experimental operator-driven ESP32-S3 FP32 exporter available; options/shape/storage validated during generation.',detail:signature};
    const extra=unsupported(input.dtype==='FLOAT32'?S3_FP32_OPS:S3_INT8_OPS);
    const details=[signature,extra?'Missing ESP32-S3 export bindings for operators: '+extra+'.':'',
      'ESP32-S3 generic FP32 currently supports Conv2D, AveragePool2D, MaxPool2D, FCN, Softmax, Reshape, and selected SqueezeNet operators; additional INT8 families are not implemented.'].filter(Boolean).join(' ');
    return no('TFLite imported for inspection, but no ESP32-S3 firmware exporter supports this graph yet.',details);
  }
  return no('No firmware exporter is registered for the selected board.');
}

function shapeTokens(s) {
  const str = String(s||'').trim(); if (!str) return null;
  const items = str.split(',').map(x=>x.trim());
  if (!items.length || items.some(x=> !/^[1-9][0-9]*$/.test(x))) return false;
  return items.map(Number);
}
function shapeMatch(a,b) {return a && b && a.length === b.length && a.every((x,i)=>x===b[i]);}
function parseTopology(source) {
  const steps = [], errors = [];
  for (const [i, raw] of source.split('\n').entries()) {
    const line = raw.split('//')[0].trim();
    if (!line) continue;
    let m = line.match(/^([A-Za-z_]\w*)\s*=\s*([A-Za-z_]\w*)\s*\(\s*([A-Za-z_][\w]*(?:\s*,\s*[A-Za-z_]\w*)*)?\s*\)\s*;?$/);
    if (m) {steps.push({line:i+1,type:'layer',output:m[1],layer:m[2],inputs:m[3]?m[3].split(',').map(x=>x.trim()):[]});continue;}
    m=line.match(/^(destroy|close)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*;?$/);
    if(m){steps.push({line:i+1,type:m[1],buffer:m[2]});continue;}
    errors.push(`Line ${i+1}: unsupported expression. Use A = layer(B); or destroy(A);`);
  }
  return {steps,errors};
}
function validate(project) {
  const errors=[],warnings=[];
  if(!(project.target in BOARD)) return {errors:['Unknown board'],warnings,steps:[]};
  const caps=BOARD[project.target];
  if(project.source?.type==='tflite'&&!project.source.model_base64) errors.push('TFLite model binary missing. Re-import original .tflite to retain weights and biases.');
  if(project.source?.type==='tflite'&&project.source.warnings)warnings.push(...project.source.warnings);
  const currentExport=firmwareCompatibility(project);
  if(currentExport.backend==='s3_int8') warnings.push('ESP32-S3 UINT8/W8A8 SqueezeNet-family exporter: PSRAM activations and XIP parameters; numerical equivalence should be validated.');
  if(currentExport.backend==='s3_fp32_generic') warnings.push('Generic ESP32-S3 FP32 exporter is experimental; only supported operator options and PSRAM/XIP paths are generated.');
  if(currentExport.backend==='s3_fp32') warnings.push('ESP32-S3 FP32 SqueezeNet-family exporter: experimental; PSRAM activations and XIP parameters.');
  const bufferNames=new Set(),layerNames=new Set(),byBuffer = new Map(),byLayer=new Map();
  for (const b of project.storage) {
    if(!IDENT.test(b.name)) errors.push(`Storage name “${b.name}” must be a valid C++ identifier.`);
    if(bufferNames.has(b.name)) errors.push(`Duplicate storage name: ${b.name}`);
    bufferNames.add(b.name);byBuffer.set(b.name,b);
    if(!caps.storage.includes(b.backend)) errors.push(`Storage ${b.name}: ${b.backend} is unsupported on ${caps.label}.`);
    if(fileBackend(b.backend)&&!b.path) errors.push(`Storage ${b.name}: file path is required.`);
    if(!fileBackend(b.backend)&&b.path) warnings.push(`Storage ${b.name}: file path is ignored for ${b.backend}.`);
    if(shapeTokens(b.shape)===false) errors.push(`Storage ${b.name}: invalid shape “${b.shape}”. Use dimensions such as 1,16.`);
  }
  for(const l of project.layers){
    if(!IDENT.test(l.name)|| RESERVED.has(l.name)) errors.push(`Layer name “${l.name}” is invalid or reserved.`);
    if(layerNames.has(l.name)) errors.push(`Duplicate layer name: ${l.name}`);
    layerNames.add(l.name);byLayer.set(l.name,l);
    if(bufferNames.has(l.name)) warnings.push(`Layer ${l.name} has the same name as a buffer; consider renaming.`);
    if(!NONPARAM.has(l.kind) && !caps.params.includes(l.params)) errors.push(`Layer ${l.name}: parameter backend ${l.params} is unsupported on ${caps.label}.`);
    if(shapeTokens(l.output_shape)===false) errors.push(`Layer ${l.name}: invalid output shape “${l.output_shape}”.`);
    if(l.kind==='HybridBlock') warnings.push(`${l.name}: custom HybridBlock requires a user-provided kernel binding.`);
    if(project.source?.type==='tflite' && project.target==='uno' && !['FCN','Softmax'].includes(l.kind) && project.source?.model_sha256!=='0c89db896852c0d69eb7be4e15fdd3981015247f0ed57ca666d8feb6e21c5073')warnings.push(`${l.name}: ${l.kind} imported for topology preview; only fixed-shape LeNet5 INT8 CNN has an experimental Uno export.`);
  }
  const parsed = parseTopology(project.topology);errors.push(...parsed.errors);
  const knownShapes = new Map(project.storage.map(b=>[b.name,shapeTokens(b.shape)||null]));
  const dead=new Set();
  for(const s of parsed.steps){
    if(s.type!=='layer'){
      if(!byBuffer.has(s.buffer)) errors.push(`Line ${s.line}: unknown storage ${s.buffer}.`);
      if(dead.has(s.buffer)) errors.push(`Line ${s.line}: ${s.buffer} was already destroyed/closed.`);
      const b=byBuffer.get(s.buffer);
      if(b && s.type==='close' && !fileBackend(b.backend)) errors.push(`Line ${s.line}: close() requires SD or FFat storage.`);
      dead.add(s.buffer);continue;
    }
    if(!byBuffer.has(s.output)) errors.push(`Line ${s.line}: output ${s.output} has no storage declaration.`);
    if(!byLayer.has(s.layer)) errors.push(`Line ${s.line}: unknown layer ${s.layer}.`);
    if(dead.has(s.output)) errors.push(`Line ${s.line}: output ${s.output} was destroyed/closed.`);
    if(!s.inputs.length) errors.push(`Line ${s.line}: ${s.layer} requires an input.`);
    for(const name of s.inputs){
      if(!byBuffer.has(name)) errors.push(`Line ${s.line}: input ${name} has no storage declaration.`);
      if(dead.has(name)) errors.push(`Line ${s.line}: input ${name} was destroyed/closed.`);
    }
    if(s.inputs.includes(s.output)) warnings.push(`Line ${s.line}: ${s.output} is used as input and output; verify in-place kernel support.`);
    const layer=byLayer.get(s.layer);
    if(layer){
      const cat=['Concat','CONCATENATION','Add'];
      if(cat.includes(layer.kind)&&s.inputs.length<2) errors.push(`Line ${s.line}: ${layer.kind} expects two or more input buffers.`);
      if(!cat.includes(layer.kind)&&s.inputs.length!==1) warnings.push(`Line ${s.line}: ${layer.kind} commonly takes one input; verify this operator.`);
      if(s.inputs.length>1 && layer.kind==='Add'){
        const shapes=s.inputs.map(x=>knownShapes.get(x));
        if(shapes.every(Boolean)&&!shapes.every(sh=>shapeMatch(shapes[0],sh))) errors.push(`Line ${s.line}: Add inputs have incompatible known shapes.`);
      }
      const next=shapeTokens(layer.output_shape);
      if(next && byBuffer.has(s.output)) knownShapes.set(s.output,next);
      else if(byBuffer.has(s.output)) knownShapes.set(s.output,null);
    }
  }
  return {errors,warnings,steps:parsed.steps};
}

function cppPreview(project) {
  const result=validate(project), lines=[];
  lines.push('// Noodle Studio R15 — STATIC CALL PLAN (preview only)');
  lines.push('// Click Generate Firmware ZIP to export real Uno INT8 FCN/Softmax firmware.');
  lines.push('// Target: '+BOARD[project.target].label);
  lines.push('// Storage declarations selected by the user:');
  for(const b of project.storage) lines.push('//   '+b.name+' : '+b.backend+(fileBackend(b.backend)?' '+b.path:'')+(b.shape?' shape=['+b.shape+']':''));
  lines.push('// Layer parameter bindings:');
  for(const l of project.layers) lines.push('//   '+l.name+' : '+l.kind+' params='+l.params);
  lines.push('');lines.push('bool run_user_topology() {');
  for(const s of result.steps){
    if(s.type==='layer') lines.push('  if (!execute_'+s.layer+'('+[...s.inputs,s.output].join(', ')+')) return false;');
    else if(s.type==='destroy') lines.push('  destroy_'+s.buffer+'();');
    else lines.push('  close_'+s.buffer+'();');
  }
  lines.push('  return true;');lines.push('}');
  return lines.join('\n')+'\n';
}

function selectedOptions(values,current) {
  let opts = values.map(v=>`<option value="${textEscape(v)}"${v===current?' selected':''}>${textEscape(v)}</option>`);
  if(!values.includes(current)) opts.unshift(`<option value="${textEscape(current)}" selected>⚠ ${textEscape(current)} — unsupported</option>`);
  return opts.join('');
}
function render() {
  $('pingPongOption').checked=state?.settings?.apply_pingpong_sequential===true;
  $('board').value=state.target;
  $('sdPinSettings').hidden=state.target!=='uno';
  $('capabilities').innerHTML=BOARD[state.target].tags.map(x=>`<span class="pill">${textEscape(x)}</span>`).join('');
  $('storageTable').querySelector('tbody').innerHTML=state.storage.map((b,i)=>`<tr data-row="${i}">
    <td><input data-bfield="name" data-index="${i}" aria-label="Storage name ${i+1}" value="${textEscape(b.name)}"></td>
    <td><select data-bfield="backend" data-index="${i}" aria-label="Backend ${i+1}" class="${BOARD[state.target].storage.includes(b.backend)?'':'unsupported'}">${selectedOptions(BOARD[state.target].storage,b.backend)}</select></td>
    <td><input data-bfield="path" data-index="${i}" placeholder="/out1.bin" aria-label="File path ${i+1}" value="${textEscape(b.path)}" ${fileBackend(b.backend)?'':'disabled'}></td>
    <td><input data-bfield="shape" data-index="${i}" placeholder="1,16 (optional)" aria-label="Buffer shape ${i+1}" value="${textEscape(b.shape)}"></td>
    <td><button class="iconbtn" data-remove-buffer="${i}" title="Remove storage" aria-label="Remove storage ${i+1}">×</button></td></tr>`).join('');
  $('layerTable').querySelector('tbody').innerHTML=state.layers.map((l,i)=>`<tr data-row="${i}">
    <td><input data-lfield="name" data-index="${i}" aria-label="Layer name ${i+1}" value="${textEscape(l.name)}"></td>
    <td><select data-lfield="kind" data-index="${i}" aria-label="Layer op ${i+1}">${selectedOptions(OPERATIONS,l.kind)}</select></td>
    <td><select data-lfield="params" data-index="${i}" aria-label="Parameter backend ${i+1}" class="${NONPARAM.has(l.kind)||BOARD[state.target].params.includes(l.params)?'':'unsupported'}">${selectedOptions(NONPARAM.has(l.kind)?['-']:BOARD[state.target].params,l.params)}</select></td>
    <td><input data-lfield="output_shape" data-index="${i}" placeholder="1,8 (optional)" aria-label="Output shape ${i+1}" value="${textEscape(l.output_shape)}"></td>
    <td><button class="iconbtn" data-remove-layer="${i}" title="Remove layer" aria-label="Remove layer ${i+1}">×</button></td></tr>`).join('');
  $('topology').value=state.topology;
  renderModelInfo();
  renderDerived();
}
function renderModelInfo(){
  const info=$('modelInfo');if(!info)return;
  const source=state.source||{};
  if(source.type!=='tflite'){
    info.innerHTML=`<div class="model-muted">${textEscape(source.name||'No imported model')} · ${textEscape(source.type||'manual')} source. Choose <b>Import TFLite</b> to start from a quantized model.</div>`;return;
  }
  const modelSize=source.model_size_bytes||0;
  const counts=(source.operators||[]).reduce((a,x)=>(a[x.op]=(a[x.op]||0)+1,a),{});
  const quant=(source.tensors||[]).filter(x=>x.quantization?.scales?.length).length;
  info.innerHTML=`<div class="model-stats"><span><b>${textEscape(source.name)}</b></span><span>${textEscape(modelSize.toLocaleString())} bytes</span><span>${(source.operators||[]).length} operations</span><span>${quant} quantized tensors</span><span>${source.model_base64?'Embedded weights ✓':'⚠ Model missing'}</span></div>
    <div class="model-sha">SHA256: <code>${textEscape(source.model_sha256||'unavailable (browser crypto)')}</code></div>
    <div class="model-oplist">${Object.entries(counts).map(([k,v])=>`<span class="pill">${textEscape(k)} × ${v}</span>`).join('')}</div>
    <div class="model-export-notice ${firmwareCompatibility(state).supported?'export-supported':'export-unsupported'}" role="status"><b>${firmwareCompatibility(state).supported?'Firmware export available':'Inspection only — firmware export unavailable'}</b><span>${textEscape(firmwareCompatibility(state).reason)} ${textEscape(firmwareCompatibility(state).detail||'')}</span></div>
    <details class="model-details"><summary>Inspect quantization and parameter metadata</summary><div class="model-tensor-scroll"><table><thead><tr><th>Tensor</th><th>Shape</th><th>Type</th><th>Scale(s)</th><th>Zero point(s)</th><th>Bytes</th></tr></thead><tbody>${(source.tensors||[]).map(t=>`<tr><td title="${textEscape(t.name)}">${t.id} · ${textEscape(t.name)}</td><td>${textEscape(t.shape.join('×'))}</td><td>${textEscape(t.dtype)}</td><td>${(t.quantization?.scales||[]).length}${(t.quantization?.scales||[]).length===1?' ('+textEscape(t.quantization.scales[0])+')':''}</td><td>${(t.quantization?.zero_points||[]).length}${(t.quantization?.zero_points||[]).length===1?' ('+textEscape(t.quantization.zero_points[0])+')':''}</td><td>${t.parameter_bytes}</td></tr>`).join('')}</tbody></table></div></details>`;
}
// Browser-only graph classification. This never selects a memory plan or modifies firmware.
// Analyze the imported TFLite metadata, not the editable topology; the backend checks
// the latter and storage safety when firmware is actually generated.
function sourceSequentialAnalysis(project){
  const src=project?.source;
  if(src?.type!=='tflite') return {sequential:null,reason:'Import a TFLite model to inspect its graph.'};
  if(!SEQUENTIAL?.analyze) return {sequential:null,reason:'Sequential-network detector is unavailable.'};
  if(!Array.isArray(src.inputs)||!Array.isArray(src.outputs)||!Array.isArray(src.operators))
    return {sequential:null,reason:'Imported model is missing graph metadata.'};
  try { return SEQUENTIAL.analyze(src); }
  catch(err) { return {sequential:null,reason:'Cannot inspect this graph: '+err.message}; }
}
function renderSequentialAnalysis(){
  const el=$('sequentialAnalysis');if(!el)return;
  const result=sourceSequentialAnalysis(state);
  const cap=firmwareCompatibility(state);
  const supportsPingPong=cap.backend==='s3_fp32_generic';
  const requested=state.settings?.apply_pingpong_sequential===true;
  const type=result.sequential===true?'sequential':result.sequential===false?'branched':'unavailable';
  const title=result.sequential===true ? `Sequential source graph (${result.operatorCount} operators)` :
    result.sequential===false?'Non-sequential source graph':'Graph classification unavailable';
  const detail=result.sequential===false ? `${result.reason}${Number.isInteger(result.operatorIndex)&&result.operatorIndex>=0?' (operator '+result.operatorIndex+')':''}.` :
    result.sequential===null?result.reason:'';
  let policy;
  if(supportsPingPong && result.sequential===true)
    policy=requested?'Ping-pong requested. Codegen verifies edited topology and buffers when generating firmware.':'Default: lifetime-based allocator. Check the box to request ping-pong.';
  else if(supportsPingPong && requested)
    policy='Ping-pong is requested but the graph is not eligible; Codegen will reject unsafe export.';
  else if(supportsPingPong)
    policy='The existing lifetime-based allocator remains selected.';
  else
    policy='Classification only. This target/backend keeps its existing memory strategy.';
  el.className='sequential-diagnostic '+type;
  el.innerHTML=`<strong>${textEscape(title)}</strong>${detail?' · '+textEscape(detail):''}<div>${textEscape(policy)}</div><small>Source TFLite classification only; edited topology and storage are validated separately during Codegen.</small>`;
}

function renderDerived(){
  renderSequentialAnalysis();
  const out=validate(state);
  const tag=$('validationBadge');
  tag.className='status '+(out.errors.length?'bad':out.warnings.length?'warn':'good');
  tag.textContent=out.errors.length?`${out.errors.length} errors`:out.warnings.length?`${out.warnings.length} warnings`:'Valid design';
  const panel=$('validation');
  const issues=[...out.errors.map(t=>({t,c:'error'})),...out.warnings.map(t=>({t,c:''}))];
  panel.innerHTML=issues.length?`<div class="issues">${issues.slice(0,25).map(x=>`<div class="issue ${x.c}">${textEscape(x.t)}</div>`).join('')}${issues.length>25?`<div class="issue">${issues.length-25} additional issues omitted</div>`:''}</div>`:`<div class="allgood">No unsupported board choices, unknown buffers, or invalid topology expressions detected. <br><br><b>Note:</b> This checks structure, not actual kernel execution or memory limits.</div>`;
  $('graph').innerHTML=out.steps.filter(x=>x.type==='layer').length ? out.steps.filter(x=>x.type==='layer').map((s,i)=>`<div class="graphrow"><span class="step">${i+1}</span><span class="edge">${textEscape(s.inputs.join(' + '))}</span><span class="arrow">→</span><span class="edge op">${textEscape(s.layer)}</span><span class="arrow">→</span><span class="edge">${textEscape(s.output)}</span></div>`).join(''):`<div class="empty">Add a layer call to the topology editor.</div>`;
  const capability=firmwareCompatibility(state);
  $('firmwareBtn').disabled=!capability.supported;
  $('firmwareBtn').title=capability.supported?capability.reason:capability.reason+' '+(capability.detail||'');
  if(!capability.supported) $('generateStatus').textContent='Inspection only: '+capability.reason+' '+(capability.detail||'');
  else if($('generateStatus').textContent.startsWith('Inspection only:')) $('generateStatus').textContent=capability.reason;
  $('cppPreview').textContent=cppPreview(state);
}
function download(name, content, type='text/plain') {
  const blob=new Blob([content],{type});
  const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=name;link.style.display='none';document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);
}
function exportJSON(){
  const result=validate(state);
  if(result.errors.length && !confirm(`Project contains ${result.errors.length} validation error(s). Export anyway?`))return;
  download('noodle_r15_project.json',JSON.stringify(state,null,2)+'\n','application/json');
}
function handleInput(e) {
  const target=e.target, idx=Number(target.dataset.index);
  if(target.dataset.bfield && state.storage[idx]) state.storage[idx][target.dataset.bfield]=target.value;
  else if(target.dataset.lfield && state.layers[idx]){
    const f=target.dataset.lfield;
    state.layers[idx][f]=target.value;
    if(f==='kind') state.layers[idx].params=NONPARAM.has(target.value)?'-':BOARD[state.target].params[0];
  } else return;
  // Re-render only when selectable backend/op changes; text input must keep its cursor.
  if(target.dataset.bfield==='backend'||target.dataset.lfield==='kind') render();else renderDerived();
}
async function handleImportFile(e){
  const file=e.target.files[0];if(!file)return;
  try{
    const json=JSON.parse(await file.text());
    const project=Array.isArray(json.steps)?fromR14Graph(json):normalizeProject(json);
    if(project.source?.type==='tflite') {
      if(!TFLITE)throw new Error('TFLite importer unavailable');
      if(!project.source.model_base64){ /* Legacy metadata-only JSON may open, but validation blocks claims about missing parameters. */ } else {
      const bin=TFLITE.base64Decode(project.source.model_base64);const model=TFLITE.inspectTFLite(bin);
      if(bin.length!==project.source.model_size_bytes)throw new Error('Embedded model byte length does not match project metadata');
      const hash=await TFLITE.sha256(bin);
      if(hash&&project.source.model_sha256&&hash!==project.source.model_sha256)throw new Error('Saved project contains corrupted or mismatched TFLite model bytes');
      if((project.source.operators||[]).length!==model.operators.length)throw new Error('Saved model graph does not match its metadata');
      }
    }
    state=project;$('example').value=''; render();
  }catch(err){alert('JSON import failed: '+err.message);}finally{e.target.value='';}
}
async function bundledLeNetModel(which){
  if(!TFLITE || typeof NoodleLeNetExamples==='undefined')throw new Error('Embedded LeNet-5 data is missing. Open the complete updated GUI or standalone HTML.');
  const item=NoodleLeNetExamples.get(which);
  const bytes=TFLITE.base64Decode(item.base64);
  const digest=await TFLITE.sha256(bytes);
  if(digest!==item.sha256)throw new Error('Bundled LeNet-5 TFLite checksum mismatch.');
  const model=TFLITE.inspectTFLite(bytes);
  const project=fromTFLiteModel(model,item.filename,bytes,digest);
  project.source.example='LeNet-5 '+which.toUpperCase()+' · reference model from LENET5.zip';
  return project;
}

async function bundledZooTFLite(key){
  if(!TFLITE || typeof NoodleModelZoo==='undefined') throw new Error('Bundled model zoo is missing.');
  const item=NoodleModelZoo.getTFLite(key);
  const bytes=TFLITE.base64Decode(item.base64);
  const digest=await TFLITE.sha256(bytes);
  if(digest!==item.sha256) throw new Error('Bundled TFLite checksum mismatch for '+item.filename);
  const model=TFLITE.inspectTFLite(bytes);
  const project=fromTFLiteModel(model,item.filename,bytes,digest);
  project.source.example='Tested model zoo · '+item.filename;
  return project;
}
function bundledZooProject(key){
  if(typeof NoodleModelZoo==='undefined') throw new Error('Bundled model zoo is missing.');
  const project=normalizeProject(copy(NoodleModelZoo.getProject(key)));
  project.source=project.source||{};
  project.source.example='Tested model zoo · '+(project.source.file||project.source.name||key);
  return project;
}

async function handleTFLiteFile(e){
  const file=e.target.files[0];if(!file)return;
  try {
    if(!TFLITE)throw new Error('TFLite importer script is missing.');
    const bytes=new Uint8Array(await file.arrayBuffer());
    const model=TFLITE.inspectTFLite(bytes);
    const digest=await TFLITE.sha256(bytes);
    const newState=fromTFLiteModel(model,file.name,bytes,digest);
    state=newState;$('example').value='';render();
    $('modelInfo').scrollIntoView({behavior:'smooth',block:'nearest'});
  } catch(err){alert('TFLite import failed: '+err.message);}
  finally {e.target.value='';}
}
function init(){
  state=sampleUno();render();
  $('board').addEventListener('change',e=>{state.target=e.target.value;render();});
  $('example').addEventListener('change',async e=>{
    const v=e.target.value;
    if(!v) return;
    try {
      if(v==='zoo_fcn_int8') state=await bundledZooTFLite('fcn_mnist16_int8');
      else if(v==='zoo_lenet_int8') state=bundledZooProject('lenet_int8_project');
      else if(v==='zoo_sq_int8') state=bundledZooProject('squeezenet_int8_project');
      else if(v==='zoo_sq_fp32') state=bundledZooProject('squeezenet_fp32_project');
      else if(v==='r14') {
        const r=JSON.parse($('r14Bundled').textContent);
        state=fromR14Graph(r);
      } else state=v==='fire'?sampleFire():sampleUno();
      render();
    } catch(err){ alert('Bundled example failed: '+err.message); }
  });
  $('storageTable').addEventListener('input',handleInput);
  $('storageTable').addEventListener('change',handleInput);
  $('layerTable').addEventListener('input',handleInput);
  $('layerTable').addEventListener('change',handleInput);
  $('storageTable').addEventListener('click',e=>{const el=e.target.closest('[data-remove-buffer]');if(!el)return;state.storage.splice(Number(el.dataset.removeBuffer),1);render();});
  $('layerTable').addEventListener('click',e=>{const el=e.target.closest('[data-remove-layer]');if(!el)return;state.layers.splice(Number(el.dataset.removeLayer),1);render();});
  $('addBuffer').addEventListener('click',()=>{
    let n=1;while(state.storage.some(x=>x.name==='buff'+n))n++;
    state.storage.push({name:'buff'+n,backend:BOARD[state.target].storage[0],path:'',shape:''});render();
  });
  $('addLayer').addEventListener('click',()=>{
    let n=1;while(state.layers.some(x=>x.name==='layer'+n))n++;
    state.layers.push({name:'layer'+n,kind:'FCN',params:BOARD[state.target].params[0],output_shape:''});render();
  });
  $('topology').addEventListener('input',e=>{state.topology=e.target.value;renderDerived();});
  $('pingPongOption').addEventListener('change',e=>{
    state.settings=state.settings||{};
    state.settings.apply_pingpong_sequential=e.target.checked;
    renderDerived();
  });
  $('exportBtn').addEventListener('click',exportJSON);
  $('cppBtn').addEventListener('click',()=>download('noodle_r15_static_call_plan.cpp',cppPreview(state)));
  $('firmwareBtn').addEventListener('click',async()=>{
    $('generateStatus').textContent='Generating...';
    try {
      if(!window.NoodleCodegen)throw Error('Firmware Codegen module was not loaded.');
      const issues=validate(state);if(issues.errors.length)throw Error(issues.errors.slice(0,4).join('\n'));
      if(state.source?.type==='tflite'&&state.source.model_sha256){
        const data=TFLITE.base64Decode(state.source.model_base64);
        const hash=await TFLITE.sha256(data);
        if(hash!==state.source.model_sha256)throw Error('Embedded TFLite checksum mismatch. Re-import the original model.');
      }
      const capability=firmwareCompatibility(state);
      if(!capability.supported) throw Error(capability.reason+' '+(capability.detail||''));
      const generators={s3_int8:()=>NoodleESP32S3.generate(state),s3_fp32:()=>NoodleESP32S3FP32.generate(state),s3_fp32_generic:()=>NoodleESP32S3GenericFP32.generate(state),uno_lenet5:()=>NoodleLeNetUno.generate(state),uno_fcn:()=>NoodleCodegen.generate(state)};
      const names={s3_int8:'noodle_r15_esp32s3_squeezenet_int8_xip_psram.zip',s3_fp32:'noodle_r15_esp32s3_squeezenet_fp32_xip_psram.zip',s3_fp32_generic:'noodle_r15_esp32s3_generic_fp32_firmware.zip',uno_lenet5:'noodle_r15_v06_UNO_LENET5_EXPERIMENTAL.zip',uno_fcn:'noodle_r15_v06_uno_int8_firmware.zip'};
      const packageFiles=generators[capability.backend]();
      download(names[capability.backend],NoodleCodegen.zip(packageFiles.files),'application/zip');
      const policy=packageFiles.summary.sequentialPingPong?'Ping-pong ENABLED (two PSRAM regions)':(capability.backend==='s3_fp32_generic'?'Static lifetime arena':(state.settings?.apply_pingpong_sequential?'Ping-pong not applicable; explicit buffers preserved':'Ping-pong OFF (default)'));
      $('generateStatus').textContent=capability.reason+' '+policy+' · '+packageFiles.summary.layers.length+' layer calls · '+packageFiles.summary.sdLayers+' SD parameter layers.';
    }catch(err){$('generateStatus').textContent='Generation rejected: '+err.message;alert('Firmware generation unavailable: '+err.message);}
  });
  $('importBtn').addEventListener('click',()=>$('importFile').click());
  $('tfliteBtn').addEventListener('click',()=>$('tfliteFile').click());
  $('tfliteFile').addEventListener('change',handleTFLiteFile);
  $('importFile').addEventListener('change',handleImportFile);
  // Fully local, no server: uploaded TFLite bytes remain in the browser and are embedded in the saved JSON.
}
if(typeof document!=='undefined') document.addEventListener('DOMContentLoaded',init);
if(typeof module!=='undefined' && module.exports) module.exports={sampleUno,sampleFire,bundledLeNetModel,fromR14Graph,fromTFLiteModel,normalizeProject,validate,firmwareCompatibility,parseTopology,cppPreview};
